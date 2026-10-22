import argparse

import torch
from PIL import Image
from transformers import AutoProcessor

from bench import mean_ms
from smolvlm_cuda_graphs import MODEL_ID, PROMPT, GraphDecoder, chat_text, encode, eos_ids, load

OPTIONS = {"slow": (False, "cpu"), "fast-cpu": (True, "cpu"), "fast-gpu": (True, "cuda")}


def synced(fn):
    def run():
        out = fn()
        torch.cuda.synchronize()
        return out

    return run


def make(processor, path, text, device):
    image = Image.open(path).convert("RGB")
    inputs = encode(processor, text, image, device).to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    return inputs


def run(name, args, dec, eos):
    fast, device = OPTIONS[name]
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=fast)
    text = chat_text(processor, args.prompt)
    image = Image.open(args.image).convert("RGB")
    kw = {} if device == "cpu" else {"device": device}

    torch.cuda.reset_peak_memory_stats()
    step_ms = mean_ms(synced(lambda: processor.image_processor(images=[image], return_tensors="pt", **kw)), args.runs)
    total_ms = mean_ms(synced(lambda: make(processor, args.image, text, device)), args.runs)
    peak = torch.cuda.max_memory_allocated() / 2**20

    inputs = make(processor, args.image, text, device)
    ids = dec.generate(inputs, eos)
    return type(processor.image_processor).__name__, step_ms, total_ms, peak, inputs["pixel_values"], ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--options", default="slow,fast-cpu,fast-gpu")
    args = ap.parse_args()

    model, processor = load()
    dec = GraphDecoder(model, args.max_new_tokens, args.max_len)
    dec.capture()
    eos = eos_ids(processor.tokenizer)

    print(
        f"{'option':<9} {'image processor':<24} {'tile+norm ms':>12} {'total ms':>9} {'peak VRAM MB':>12} "
        f"{'pixel diff max/mean':>20}  same answer"
    )
    ref = None
    for name in args.options.split(","):
        try:
            cls, step_ms, total_ms, peak, pv, ids = run(name, args, dec, eos)
        except Exception as e:
            print(f"{name:<9} failed: {type(e).__name__}: {e}")
            continue
        ref = ref or (pv, ids)
        if pv.shape == ref[0].shape:
            d = (pv.float() - ref[0].float()).abs()
            diff = f"{d.max():.3f}/{d.mean():.4f}"
        else:
            diff = "shape differs"
        print(f"{name:<9} {cls:<24} {step_ms:12.1f} {total_ms:9.1f} {peak:12.0f} {diff:>20}  {ids == ref[1]}")


if __name__ == "__main__":
    main()
