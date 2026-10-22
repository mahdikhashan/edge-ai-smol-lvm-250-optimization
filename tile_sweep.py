import argparse
import gc

import torch
from PIL import Image

from bench import bench_stop, mean_ms
from smolvlm_cuda_graphs import PREPROCESS, PROMPT, GraphDecoder, chat_text, eos_ids, load


def config_kwargs(name):
    """default | nosplit | sizeN (resize the longest side to N pixels before the 512-pixel tiles are cut)."""
    if name == "default":
        return {}
    if name == "nosplit":
        return {"do_image_splitting": False}
    if name.startswith("size"):
        return {"size": {"longest_edge": int(name[4:])}}
    raise ValueError(f"unknown config {name}")


def make(processor, path, text, kw, image_device):
    image = Image.open(path).convert("RGB")
    kw = dict(kw, **({} if image_device == "cpu" else {"device": image_device}))
    try:
        inputs = processor(text=text, images=[image], return_tensors="pt", images_kwargs=kw)
    except (TypeError, ValueError):
        inputs = processor(text=text, images=[image], return_tensors="pt", **kw)
    inputs = inputs.to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    return inputs


def run(name, path, args, model, processor, eos, image_device):
    text = chat_text(processor, args.prompt)
    kw = config_kwargs(name)
    inputs = make(processor, path, text, kw, image_device)
    tiles, tokens = inputs["pixel_values"].shape[1], inputs["input_ids"].shape[1]
    input_ms = mean_ms(lambda: make(processor, path, text, kw, image_device), args.runs)
    max_len = -(-(tokens + args.max_new_tokens + 1) // 64) * 64  # smallest cache that fits, in steps of 64
    dec = GraphDecoder(model, args.max_new_tokens, max_len)
    dec.capture()
    steps, pre, dec_ms = bench_stop(dec, inputs, eos, args.runs)
    ids = dec.generate(inputs, eos)
    del dec
    gc.collect()
    torch.cuda.empty_cache()
    return tiles, tokens, max_len, input_ms, pre, dec_ms, steps, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--configs", default="default,size1536,size1024,size512,nosplit")
    ap.add_argument("--preprocess", choices=PREPROCESS, default="fast-gpu")
    args = ap.parse_args()

    fast, image_device = PREPROCESS[args.preprocess]
    model, processor = load(fast=fast)
    eos = eos_ids(processor.tokenizer)

    for path in args.image:
        print(f"\nimage: {path} | preprocess: {args.preprocess}")
        print(
            f"{'config':<9} {'tiles':>5} {'tokens':>6} {'max_len':>7} {'input ms':>8} {'prefill ms':>10} "
            f"{'decode ms':>9} {'steps':>5} {'request ms':>10} {'speed-up':>8}  {'same':<5} answer"
        )
        ref = None
        for name in args.configs.split(","):
            try:
                tiles, tokens, max_len, input_ms, pre, dec_ms, steps, ids = run(name, path, args, model, processor, eos, image_device)
            except Exception as e:
                print(f"{name:<9} failed: {type(e).__name__}: {e}")
                continue
            total = input_ms + pre + dec_ms
            ref = ref or (total, ids)
            answer = processor.tokenizer.decode(ids, skip_special_tokens=True).strip().replace("\n", " ")
            print(
                f"{name:<9} {tiles:5d} {tokens:6d} {max_len:7d} {input_ms:8.1f} {pre:10.1f} {dec_ms:9.1f} {steps:5d} "
                f"{total:10.1f} {ref[0] / total:8.2f}  {str(ids == ref[1]):<5} {answer[:60]}"
            )


if __name__ == "__main__":
    main()
