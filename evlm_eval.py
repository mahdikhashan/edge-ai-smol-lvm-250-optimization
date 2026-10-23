"""EVLM-style evaluation (Farahani et al., EDGE 2026) with the fastest inference setup measured in this repo.

Two benchmarks on the data from get_data.py:
- clips:   4 s power line clips (120 frames, 720x480). Histogram frame filtering, then the VLM on each key frame.
           Reports key frames, filter time, VLM time per call, tokens per answer, decode ms per token, and the
           segment latency (filter + VLM, like the paper's Table V; PNG decode is reported apart).
- insplad: InsPLAD-fault crops. The two prompts of the paper's Table III (component and rust), scored on the labels.

Every speed-up is a flag, so one script covers the ablation. The defaults are the best setup:
fast GPU preprocessing, --size 1024, CUDA-graph decoding, stop at EOS, TensorRT SigLIP, histogram filter.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.modeling_outputs import BaseModelOutput

from bench import timed, until_eos
from smolvlm_cuda_graphs import PREPROCESS, GraphDecoder, chat_text, encode, eos_ids, load

CLIP_PROMPT = "Is there a transmission tower in the green area?"  # mission intent of the paper's Section VI-E
COMPONENT_PROMPT = "What is the power-line component in the image?"  # Table III
RUST_PROMPT = "Is rust visible?"  # Table III


class LazyTrt(torch.nn.Module):
    """SigLIP encoder in TensorRT, one engine per tile count (built on first use, cached in trt_siglip/)."""

    def __init__(self, model, workdir="trt_siglip"):
        super().__init__()
        from trt_siglip import TrtEncoder, build_engine, check_packages, export_onnx

        check_packages()
        self.vision = model.model.vision_model
        self.model_ref = [model]  # a list keeps the model out of this module's tree
        self.work, self.engines = Path(workdir), {}
        self._build = (TrtEncoder, build_engine, export_onnx)

    def forward(self, pixel_values, patch_attention_mask=None, *args, **kwargs):
        if patch_attention_mask is not None and not bool(patch_attention_mask.all()):
            # padded patches need the attention mask; the engine has none, so use PyTorch for this image
            return self.vision(pixel_values, patch_attention_mask, *args, **kwargs)
        hidden = self.vision.embeddings(pixel_values, patch_attention_mask)
        n = hidden.shape[0]
        if n not in self.engines:
            self.engines[n] = self._engine(hidden)
        return BaseModelOutput(last_hidden_state=self.engines[n](hidden))

    def _engine(self, hidden):
        TrtEncoder, build_engine, export_onnx = self._build
        self.work.mkdir(exist_ok=True)
        n = hidden.shape[0]
        onnx, engine = self.work / f"siglip_{n}tiles.onnx", self.work / f"siglip_{n}tiles.engine"
        if not engine.exists():
            with torch.inference_mode(False):
                x = hidden.clone()  # a normal tensor; inference tensors cannot be traced
                export_onnx(self.model_ref[0], x, onnx, vision=self.vision)
            print(f"TensorRT engine built: {engine} ({build_engine(onnx, engine)})")
        return TrtEncoder(engine)


class Runner:
    """Prompt + image -> answer token ids, with the inference setup chosen by the flags."""

    def __init__(self, args):
        fast, self.image_device = PREPROCESS[args.preprocess]
        self.model, self.processor = load(fast=fast)
        self.eos = eos_ids(self.processor.tokenizer)
        self.image_kwargs = {"size": {"longest_edge": args.size}} if args.size else None
        self.decode, self.stop, self.max_new = args.decode, args.stop, args.max_new_tokens
        self.dec = None
        if args.trt:
            self.model.model.vision_model = LazyTrt(self.model)

    def make(self, image, prompt):
        text = chat_text(self.processor, prompt)
        inputs = encode(self.processor, text, image, self.image_device, self.image_kwargs).to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        return inputs

    def decoder(self, n):
        if self.dec is None or n + self.max_new >= self.dec.max_len:
            max_len = -(-(n + self.max_new + 1) // 64) * 64  # smallest cache that fits, in steps of 64
            self.dec = GraphDecoder(self.model, self.max_new, max_len)
            if self.decode == "graph":
                self.dec.capture()
        return self.dec

    def answer(self, inputs):
        """Returns (answer ids, prefill ms, decode ms, VLM ms). Decode = the steps after the first token."""
        n = inputs["input_ids"].shape[1]
        if self.decode == "bare":
            def generate(k, **extra):
                with torch.inference_mode():
                    return self.model.generate(**inputs, max_new_tokens=k, do_sample=False, **extra)

            # generate() cannot time its phases, so a separate one-token call measures the prefill
            _, prefill_ms = timed(lambda: generate(1))
            out, vlm_ms = timed(lambda: generate(self.max_new, **({} if self.stop else {"min_new_tokens": self.max_new})))
            ids, decode_ms = out[0, n:].tolist(), max(vlm_ms - prefill_ms, 0.0)
        else:
            dec = self.decoder(n)
            first, prefill_ms = timed(lambda: dec.prefill(inputs))
            out, decode_ms = timed(lambda: dec.decode(self.decode == "graph", self.eos if self.stop else None))
            ids, vlm_ms = torch.cat([first.view(1), out]).tolist(), prefill_ms + decode_ms
        return until_eos(ids, self.eos), prefill_ms, decode_ms, vlm_ms

    def ask(self, image, prompt):
        """Answer text, token count and times (ms) for one image and prompt."""
        inputs, input_ms = timed(lambda: self.make(image, prompt))
        ids, prefill_ms, decode_ms, vlm_ms = self.answer(inputs)
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return {"answer": text, "tokens": len(ids), "input_ms": input_ms, "prefill_ms": prefill_ms,
                "decode_ms": decode_ms, "vlm_ms": vlm_ms}


def per_token(calls):
    """Decode ms per generated token: the decode runs one step for each answer token (the last step makes EOS)."""
    return sum(c["decode_ms"] for c in calls) / max(sum(c["tokens"] for c in calls), 1)


def call_stats(calls):
    return {"vlm_ms_per_call": round(float(np.mean([c["input_ms"] + c["vlm_ms"] for c in calls])), 1),
            "prefill_ms": round(float(np.mean([c["prefill_ms"] for c in calls])), 1),
            "tokens_per_answer": round(float(np.mean([c["tokens"] for c in calls])), 2),
            "decode_ms_per_token": round(per_token(calls), 2)}


def histogram(arr):
    """512-bin RGB histogram (8 levels per channel), L2-normalized (paper Section V-E)."""
    q = arr >> 5
    h = np.bincount((q[..., 0].astype(np.int32) << 6 | q[..., 1] << 3 | q[..., 2]).ravel(), minlength=512)
    h = h.astype(np.float32)
    return h / (np.linalg.norm(h) + 1e-12)


def run_clips(runner, args, log):
    root = Path(args.data) / "powerline_clips"
    clips = sorted(p for p in root.glob("*/clip_*") if not args.envs or p.parent.name in args.envs.split(","))
    if not clips:
        raise SystemExit(f"no clips in {root}: run get_data.py --what clips")
    runner.ask(Image.open(next(clips[0].glob("*.png"))).convert("RGB"), args.prompt)  # warm-up: graph capture, engine

    print(
        f"{'clip':<20} {'frames':>6} {'key':>4} {'png ms/f':>8} {'filter ms/f':>11} {'VLM ms/call':>11} "
        f"{'prefill ms':>10} {'tokens':>6} {'ms/token':>8} {'segment s':>9}  answer"
    )
    rows, all_calls = [], []
    for clip in clips:
        frames = sorted(clip.glob("*.png"))
        keys, prev, t_decode, t_filter = [], None, 0.0, 0.0
        for i, path in enumerate(frames):
            t = time.perf_counter()
            image = Image.open(path).convert("RGB")
            arr = np.asarray(image)
            t_decode += time.perf_counter() - t
            t = time.perf_counter()
            keep = True
            if args.filter == "hist":
                h = histogram(arr)
                keep = prev is None or np.corrcoef(h, prev)[0, 1] < args.tau
                if keep:
                    prev = h
            t_filter += time.perf_counter() - t
            if keep:
                keys.append((i, image))
        name = f"{clip.parent.name}/{clip.name}"
        calls = []
        for i, image in keys:
            c = runner.ask(image, args.prompt)
            calls.append(c)
            log({"task": "clips", "clip": name, "frame": i, "prompt": args.prompt, **c})
        all_calls += calls
        # PNG decode is left out: it comes from how the dataset is stored (the paper decodes video)
        segment = t_filter + sum(c["input_ms"] + c["vlm_ms"] for c in calls) / 1e3
        rows.append((len(keys), segment, t_filter, t_decode))
        st = call_stats(calls)
        print(
            f"{name:<20} {len(frames):6d} {len(keys):4d} {1e3 * t_decode / len(frames):8.2f} "
            f"{1e3 * t_filter / len(frames):11.2f} {st['vlm_ms_per_call']:11.1f} {st['prefill_ms']:10.1f} "
            f"{st['tokens_per_answer']:6.1f} {st['decode_ms_per_token']:8.2f} {segment:9.2f}  {calls[-1]['answer'][:40]}"
        )
    return {"clips": len(rows), "key_frames": round(float(np.mean([r[0] for r in rows])), 2), **call_stats(all_calls),
            "filter_s_per_segment": round(float(np.mean([r[2] for r in rows])), 3),
            "png_decode_s_per_segment": round(float(np.mean([r[3] for r in rows])), 3),
            "segment_s": round(float(np.mean([r[1] for r in rows])), 3)}


def insplad_samples(root, limit):
    """Held-out images: defect_supervised/<asset>/val/<label>/*.jpg (InsPLAD-fault has no test split)."""
    groups = {}
    for p in sorted(root.glob("defect_supervised/*/val/*/*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            groups.setdefault((p.parent.parent.parent.name, p.parent.name), []).append(p)
    return [(asset, label, p) for (asset, label), ps in sorted(groups.items()) for p in ps[:limit]]


def says_yes(answer):
    a = answer.lower()
    if a.startswith("no") or "no rust" in a or "not visible" in a or "no visible" in a:
        return False
    return a.startswith("yes") or "rust" in a


def run_insplad(runner, args, log):
    samples = insplad_samples(Path(args.data) / "insplad", args.limit)
    if not samples:
        raise SystemExit("no InsPLAD images: run get_data.py --what insplad")
    runner.ask(Image.open(samples[0][2]).convert("RGB"), COMPONENT_PROMPT)  # warm-up

    comp, rust, calls = {}, {}, []
    for asset, label, path in samples:
        image = Image.open(path).convert("RGB")
        c, r = runner.ask(image, COMPONENT_PROMPT), runner.ask(image, RUST_PROMPT)
        c_text, r_text = c["answer"], r["answer"]
        words = [w for w in asset.lower().replace("_", "-").split("-") if len(w) > 3 and w != "upper"]
        comp.setdefault(asset, []).append(any(w in c_text.lower() for w in words))
        if label.lower() in ("good", "rust"):
            rust.setdefault(asset, []).append(says_yes(r_text) == (label.lower() == "rust"))
        calls += [c, r]
        log({"task": "insplad", "asset": asset, "label": label, "image": str(path), "component": c, "rust": r})

    print(f"{'asset':<34} {'n':>4} {'component acc':>13} {'rust n':>6} {'rust acc':>8}")
    for asset in comp:
        r = rust.get(asset, [])
        print(f"{asset:<34} {len(comp[asset]):4d} {np.mean(comp[asset]):13.2f} {len(r):6d} {np.mean(r) if r else float('nan'):8.2f}")
    all_c, all_r = sum(comp.values(), []), sum(rust.values(), [])
    return {"samples": len(samples), "component_acc": round(float(np.mean(all_c)), 3),
            "rust_acc": round(float(np.mean(all_r)), 3) if all_r else None, **call_stats(calls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("clips", "insplad"), default="clips")
    ap.add_argument("--data", default="data")
    ap.add_argument("--preprocess", choices=PREPROCESS, default="fast-gpu")
    ap.add_argument("--size", type=int, default=1024, help="longest image side before tiling; 0 = processor default")
    ap.add_argument("--decode", choices=("graph", "eager", "bare"), default="graph")
    ap.add_argument("--stop", action=argparse.BooleanOptionalAction, default=True, help="stop at EOS")
    ap.add_argument("--trt", action=argparse.BooleanOptionalAction, default=True, help="TensorRT SigLIP encoder")
    ap.add_argument("--filter", choices=("hist", "none"), default="hist", help="clips: key-frame filter")
    ap.add_argument("--tau", type=float, default=0.95, help="clips: histogram correlation threshold")
    ap.add_argument("--prompt", default=CLIP_PROMPT, help="clips: operator intent")
    ap.add_argument("--envs", default="", help="clips: comma-separated environments (default: all)")
    ap.add_argument("--limit", type=int, default=10, help="insplad: images per asset and label")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    config = {k: getattr(args, k) for k in ("preprocess", "size", "decode", "stop", "trt", "filter", "tau")}
    tag = f"{args.task}_{args.preprocess}_s{args.size}_{args.decode}_{'stop' if args.stop else 'nostop'}_" \
          f"{'trt' if args.trt else 'notrt'}" + (f"_{args.filter}" if args.task == "clips" else "")
    Path(args.out).mkdir(exist_ok=True)
    with open(Path(args.out) / f"{tag}.jsonl", "w") as f:
        def log(record):
            f.write(json.dumps(record) + "\n")

        runner = Runner(args)
        summary = (run_clips if args.task == "clips" else run_insplad)(runner, args, log)
        log({"summary": summary, "config": config})
    print("SUMMARY", json.dumps({"task": args.task, **config, **summary}))


if __name__ == "__main__":
    main()
