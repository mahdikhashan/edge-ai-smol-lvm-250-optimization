import argparse
import resource
import time
from pathlib import Path

import torch
from PIL import Image

from smolvlm_cuda_graphs import PREPROCESS, PROMPT, GraphDecoder, chat_text, eos_ids, load, make_inputs


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1e3


def mean_ms(fn, runs):
    def run():
        fn()
        torch.cuda.synchronize()

    run()  # warm-up
    t = time.perf_counter()
    for _ in range(runs):
        run()
    return (time.perf_counter() - t) * 1e3 / runs


def bench_preprocess(processor, path, prompt, runs, image_device="cpu", size=None):
    text = chat_text(processor, prompt)
    image = Image.open(path).convert("RGB")
    kw = {} if image_device == "cpu" else {"device": image_device}
    if size is not None:
        kw["size"] = {"longest_edge": size}
    steps = {
        "JPEG decode": lambda: Image.open(path).convert("RGB"),
        "tile + normalize": lambda: processor.image_processor(images=[image], return_tensors="pt", **kw),
        "tokenize": lambda: processor.tokenizer(text, return_tensors="pt"),
        "make_inputs total (incl. copy to GPU)": lambda: make_inputs(processor, path, prompt, image_device=image_device, size=size),
    }
    print(f"preprocessing (JPEG decode and tokenize on CPU, tile + normalize on {image_device.upper()}):")
    for name, fn in steps.items():
        ms = mean_ms(fn, runs)
        print(f"  {name:<38} {ms:7.1f} ms")
    return ms  # the last step is the make_inputs total


def log_memory(tag):
    mb = 2**20
    ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB
    print(
        f"memory {tag}: RAM peak {ram:.0f} MB | VRAM allocated {torch.cuda.memory_allocated() / mb:.0f} MB, "
        f"peak {torch.cuda.max_memory_allocated() / mb:.0f} MB, reserved {torch.cuda.memory_reserved() / mb:.0f} MB"
    )


def bench(dec, inputs, use_graph, runs):
    prefill, decode = [], []
    for i in range(runs + 1):  # first run is warm-up
        _, p = timed(lambda: dec.prefill(inputs))
        ids, d = timed(lambda: dec.decode(use_graph))
        if i:
            prefill.append(p)
            decode.append(d / dec.max_new)
    return ids.tolist(), sum(prefill) / runs, sum(decode) / runs


def bench_stop(dec, inputs, eos, runs):
    prefill, decode = [], []
    for i in range(runs + 1):  # first run is warm-up
        _, p = timed(lambda: dec.prefill(inputs))
        _, d = timed(lambda: dec.decode(True, eos))
        if i:
            prefill.append(p)
            decode.append(d)
    return dec.steps, sum(prefill) / runs, sum(decode) / runs


def load_trt(model, inputs, workdir="trt_siglip"):
    """Build (or load) the TensorRT engine for the SigLIP encoder. The engine has a fixed tile count."""
    from trt_siglip import TrtEncoder, TrtVision, build_engine, capture_encoder_input, check_packages, export_onnx

    check_packages()
    x = capture_encoder_input(model, inputs)
    work = Path(workdir)
    work.mkdir(exist_ok=True)
    onnx, engine = work / f"siglip_{x.shape[0]}tiles.onnx", work / f"siglip_{x.shape[0]}tiles.engine"
    if engine.exists():
        print(f"TensorRT engine loaded: {engine}")
    else:
        export_onnx(model, x, onnx)
        print(f"TensorRT engine built: {engine} ({build_engine(onnx, engine)})")
    return TrtVision(model.model.vision_model, TrtEncoder(engine))


def bench_bare(model, inputs, n, runs):
    def gen(k):
        return model.generate(**inputs, max_new_tokens=k, min_new_tokens=k, do_sample=False)

    prefill, total = [], []
    for i in range(runs + 1):  # first run is warm-up
        _, p = timed(lambda: gen(1))
        out, t = timed(lambda: gen(n + 1))
        if i:
            prefill.append(p)
            total.append(t)
    pre = sum(prefill) / runs
    ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    return ids[1:], pre, (sum(total) / runs - pre) / n


def until_eos(ids, eos):
    for i, t in enumerate(ids):
        if t in eos:
            return ids[:i]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--preprocess", choices=PREPROCESS, default="default")
    ap.add_argument("--size", type=int, default=None, help="resize the longest image side to this many pixels (fewer tiles)")
    ap.add_argument("--trt", action="store_true", help="also run with a TensorRT engine for the SigLIP encoder")
    args = ap.parse_args()

    fast, image_device = PREPROCESS[args.preprocess]
    model, processor = load(fast=fast)
    print(f"preprocess: {args.preprocess} ({type(processor.image_processor).__name__}, image on {image_device})")
    log_memory("after load")
    inputs = make_inputs(processor, args.image, args.prompt, image_device=image_device, size=args.size)
    print(
        f"prompt tokens: {inputs['input_ids'].shape[1]} ({inputs['pixel_values'].shape[1]} tiles) | max_len: {args.max_len} "
        f"| size: {args.size or 'default'}"
    )
    input_ms = bench_preprocess(processor, args.image, args.prompt, args.runs, image_device, args.size)
    dec = GraphDecoder(model, args.max_new_tokens, args.max_len)
    dec.capture()

    eos = eos_ids(processor.tokenizer)
    outputs, timing = {}, {}
    outputs["bare"], pre, tpot = bench_bare(model, inputs, args.max_new_tokens, args.runs)
    print(f"{'bare':>5}: prefill {pre:6.1f} ms | decode {tpot:5.2f} ms/token | {1000 / tpot:6.1f} tok/s")
    for mode in ("eager", "graph"):
        ids, pre, tpot = bench(dec, inputs, mode == "graph", args.runs)
        outputs[mode] = ids
        timing[mode] = (pre, tpot)
        print(f"{mode:>5}: prefill {pre:6.1f} ms | decode {tpot:5.2f} ms/token | {1000 / tpot:6.1f} tok/s")

    print("identical outputs (eager vs graph):", outputs["eager"] == outputs["graph"])
    # bare blocks EOS (min_new_tokens), so compare only up to graph's first EOS
    g = until_eos(outputs["graph"], eos)
    print("bare matches graph up to EOS:", outputs["bare"][:len(g)] == g)

    steps, pre, dec_ms = bench_stop(dec, inputs, eos, args.runs)
    print(f"graph + stop at EOS: prefill {pre:6.1f} ms | decode {dec_ms:6.1f} ms for {steps} steps")
    pre_g, tpot_g = timing["graph"]
    print(
        f"request (input + prefill + decode), graph: {input_ms + pre_g + tpot_g * args.max_new_tokens:.0f} ms "
        f"with {args.max_new_tokens} steps | {input_ms + pre + dec_ms:.0f} ms with stop at EOS ({steps} steps)"
    )

    if args.trt:
        ids = dec.generate(inputs, eos)
        orig = model.model.vision_model
        model.model.vision_model = load_trt(model, inputs)
        steps_t, pre_t, dec_t = bench_stop(dec, inputs, eos, args.runs)
        same = dec.generate(inputs, eos) == ids
        model.model.vision_model = orig
        print(f"graph + stop at EOS + TensorRT SigLIP: prefill {pre_t:6.1f} ms | decode {dec_t:6.1f} ms for {steps_t} steps")
        print(
            f"request (input + prefill + decode), stop at EOS + TensorRT: {input_ms + pre_t + dec_t:.0f} ms "
            f"({input_ms + pre + dec_ms:.0f} ms without TensorRT) | same answer: {same}"
        )
    log_memory("after benchmark")


if __name__ == "__main__":
    main()
