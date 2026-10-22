import argparse
import copy
import shutil
import subprocess
import time
from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutput

from hybrid_attn import set_attn
from prefill_profile import make_inputs, profile
from smolvlm_cuda_graphs import PROMPT, GraphDecoder, eos_ids, load


class Encoder(torch.nn.Module):
    """The SigLIP transformer: 12 encoder layers + the last layer norm (the patch embedding stays in PyTorch)."""

    def __init__(self, vision):
        super().__init__()
        self.encoder = vision.encoder
        self.norm = vision.post_layernorm

    def forward(self, x):
        return self.norm(self.encoder(inputs_embeds=x)[0])


class TrtEncoder:
    def __init__(self, engine_path):
        import tensorrt as trt

        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.engine, self.ctx = engine, engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.out_shape = tuple(engine.get_tensor_shape(self.out))
        dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self.in_dtype = dtypes[engine.get_tensor_dtype(self.inp)]
        self.out_dtype = dtypes[engine.get_tensor_dtype(self.out)]

    def __call__(self, x):
        x = x.to(self.in_dtype).contiguous()
        y = torch.empty(self.out_shape, dtype=self.out_dtype, device=x.device)
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.set_tensor_address(self.out, y.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return y.half()


class TrtVision(torch.nn.Module):
    """Drop-in for model.model.vision_model: patch embedding in PyTorch, encoder in TensorRT."""

    def __init__(self, vision, trt_encoder):
        super().__init__()
        self.embeddings = vision.embeddings
        self.trt = trt_encoder

    def forward(self, pixel_values, patch_attention_mask=None, *args, **kwargs):
        hidden = self.embeddings(pixel_values, patch_attention_mask)
        return BaseModelOutput(last_hidden_state=self.trt(hidden))


def capture_encoder_input(model, inputs):
    got = {}

    def hook(module, args, kwargs):
        got["x"] = (kwargs["inputs_embeds"] if "inputs_embeds" in kwargs else args[0]).detach().clone()

    handle = model.model.vision_model.encoder.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model(**inputs, use_cache=False)
    handle.remove()
    return got["x"]


def export_onnx(model, x, path):
    set_attn(model, "eager")  # plain matmul + softmax attention exports cleanly; TensorRT fuses it again
    enc = copy.deepcopy(Encoder(model.model.vision_model)).half().eval()  # fp16 types: newer TensorRT takes the precision from the ONNX
    set_attn(model, "sdpa")
    args = (x.half(),)
    kw = dict(input_names=["embeds"], output_names=["hidden"], opset_version=17)
    with torch.no_grad():
        try:
            torch.onnx.export(enc, args, str(path), dynamo=False, **kw)
        except TypeError:  # older torch has no `dynamo` argument
            torch.onnx.export(enc, args, str(path), **kw)


def build_engine(onnx, engine):
    trtexec = shutil.which("trtexec") or ("/usr/src/tensorrt/bin/trtexec" if Path("/usr/src/tensorrt/bin/trtexec").exists() else None)
    if trtexec:
        r = subprocess.run([trtexec, f"--onnx={onnx}", "--fp16", f"--saveEngine={engine}"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("trtexec failed: " + r.stdout[-600:])
        return "trtexec"
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    nf, has_fp16_flag = trt.NetworkDefinitionCreationFlag, hasattr(trt.BuilderFlag, "FP16")
    flag = 1 << int(nf.EXPLICIT_BATCH) if hasattr(nf, "EXPLICIT_BATCH") else 0
    if not has_fp16_flag:  # TensorRT without BuilderFlag.FP16: a strongly typed network uses the types in the ONNX file
        flag |= 1 << int(nf.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx).read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if has_fp16_flag:
        config.set_flag(trt.BuilderFlag.FP16)
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError("TensorRT build failed")
    Path(engine).write_bytes(data)
    return f"python API, TensorRT {trt.__version__}, {'FP16 flag' if has_fp16_flag else 'strongly typed fp16'}"


def check_packages():
    for pkg in ("onnx", "tensorrt"):
        try:
            __import__(pkg)
        except ImportError:
            raise SystemExit(f"missing Python package: pip install {pkg}")


def main():
    check_packages()
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workdir", default="trt_siglip")
    args = ap.parse_args()

    model, processor = load()
    dec = GraphDecoder(model, args.max_new_tokens, args.max_len)
    dec.capture()
    eos = eos_ids(processor.tokenizer)
    inputs = make_inputs(processor, args.image, args.prompt, split=True)
    x = capture_encoder_input(model, inputs)

    work = Path(args.workdir)
    work.mkdir(exist_ok=True)
    onnx, engine = work / "siglip_encoder.onnx", work / "siglip_encoder.engine"
    t = time.perf_counter()
    export_onnx(model, x, onnx)
    t_export = time.perf_counter() - t
    t = time.perf_counter()
    how = build_engine(onnx, engine)
    t_build = time.perf_counter() - t
    print(
        f"encoder input {tuple(x.shape)} | onnx export {t_export:.0f} s | engine build {t_build:.0f} s ({how}) "
        f"| engine {engine.stat().st_size / 2**20:.0f} MB"
    )

    trt_encoder = TrtEncoder(engine)
    orig = model.model.vision_model
    with torch.no_grad():
        ref = orig.post_layernorm(orig.encoder(inputs_embeds=x)[0]).float()
        diff = (trt_encoder(x).float() - ref).abs()
    print(f"encoder output vs PyTorch fp16: max abs diff {diff.max():.4f} | mean abs diff {diff.mean():.5f} | mean abs value {ref.abs().mean():.3f}")

    results = {}
    results["pytorch"] = (*profile(model, dec, inputs, args.runs), dec.generate(inputs, eos))
    model.model.vision_model = TrtVision(orig, trt_encoder)
    results["tensorrt"] = (*profile(model, dec, inputs, args.runs), dec.generate(inputs, eos))
    model.model.vision_model = orig

    base_total, base_parts, base_ids = results["pytorch"]
    print(f"{'option':<9} {'prefill ms':>10} {'SigLIP ms':>10} {'SigLIP speed-up':>15} {'prefill saved ms':>16}  same answer")
    for name, (total, parts, ids) in results.items():
        print(
            f"{name:<9} {total:10.1f} {parts['siglip']:10.1f} {base_parts['siglip'] / parts['siglip']:15.2f} "
            f"{base_total - total:16.1f}  {ids == base_ids}"
        )


if __name__ == "__main__":
    main()
