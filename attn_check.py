import argparse
import gc

import torch

from bench import bench, until_eos
from smolvlm_cuda_graphs import PROMPT, GraphDecoder, eos_ids, load, make_inputs


def run(attn, args):
    model, processor = load(attn=None if attn == "default" else attn)
    cfg = model.config
    used = "/".join(getattr(c, "_attn_implementation", "?") for c in (cfg.text_config, cfg.vision_config))
    inputs = make_inputs(processor, args.image, args.prompt)
    dec = GraphDecoder(model, args.max_new_tokens, args.max_len)
    dec.capture()
    torch.cuda.reset_peak_memory_stats()
    ids, pre, tpot = bench(dec, inputs, True, args.runs)
    peak = torch.cuda.max_memory_allocated() / 2**20
    eos = eos_ids(processor.tokenizer)
    del model, dec
    gc.collect()
    torch.cuda.empty_cache()
    return used, until_eos(ids, eos), pre, tpot, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--impls", default="default,eager,sdpa")
    args = ap.parse_args()

    print(f"{'option':<8} {'text/vision':<13} {'prefill ms':>10} {'decode ms/tok':>13} {'peak VRAM MB':>12}  same answer")
    ref = None
    for attn in args.impls.split(","):
        try:
            used, ids, pre, tpot, peak = run(attn, args)
        except Exception as e:
            print(f"{attn:<8} failed: {type(e).__name__}: {e}")
            continue
        ref = ids if ref is None else ref
        print(f"{attn:<8} {used:<13} {pre:10.1f} {tpot:13.2f} {peak:12.0f}  {ids == ref}")


if __name__ == "__main__":
    main()
