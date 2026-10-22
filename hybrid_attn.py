import argparse
import gc

import torch

from bench import bench, until_eos
from smolvlm_cuda_graphs import PROMPT, GraphDecoder, eos_ids, load, make_inputs


def set_attn(model, impl):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
    else:
        for c in (model.config.text_config, model.config.vision_config):
            c._attn_implementation = impl


class HybridDecoder(GraphDecoder):
    """Prefill runs with sdpa. The decode graph is captured with eager attention and replayed as is."""

    def capture(self):
        set_attn(self.model, "eager")
        self.decode_attn = self.model.config.text_config._attn_implementation
        self.own_mask = True
        try:
            super().capture()
        finally:
            set_attn(self.model, "sdpa")


def run(mode, args):
    model, processor = load(attn="eager" if mode == "eager" else "sdpa")
    inputs = make_inputs(processor, args.image, args.prompt)
    cls = HybridDecoder if mode == "hybrid" else GraphDecoder
    dec = cls(model, args.max_new_tokens, args.max_len)
    dec.capture()
    prefill_attn = model.config.text_config._attn_implementation
    used = f"{prefill_attn}/{getattr(dec, 'decode_attn', prefill_attn)}"
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
    ap.add_argument("--modes", default="sdpa,eager,hybrid")
    args = ap.parse_args()

    print(
        f"{'option':<8} {'prefill/decode':<14} {'prefill ms':>10} {'decode ms/tok':>13} "
        f"{'total ms':>9} {'peak VRAM MB':>12}  same answer"
    )
    ref = None
    for mode in args.modes.split(","):
        try:
            used, ids, pre, tpot, peak = run(mode, args)
        except Exception as e:
            print(f"{mode:<8} failed: {type(e).__name__}: {e}")
            continue
        ref = ids if ref is None else ref
        total = pre + tpot * args.max_new_tokens
        print(f"{mode:<8} {used:<14} {pre:10.1f} {tpot:13.2f} {total:9.1f} {peak:12.0f}  {ids == ref}")


if __name__ == "__main__":
    main()
