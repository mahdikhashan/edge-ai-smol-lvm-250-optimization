import argparse
import gc

import torch

from bench import timed, until_eos
from smolvlm_cuda_graphs import PROMPT, GraphDecoder, eos_ids, load, make_inputs


class EosDecoder(GraphDecoder):
    """Graph decoder that can stop early: it checks for EOS after every `every` steps (one GPU->CPU sync per check)."""

    def __init__(self, model, max_new_tokens, max_len, eos, every=None):
        super().__init__(model, max_new_tokens, max_len)
        self.eos = torch.tensor(sorted(e for e in eos if e is not None), device=model.device)
        self.every = every
        self.steps = 0

    @torch.inference_mode()
    def decode(self, use_graph=True):
        self.steps = self.max_new
        for i in range(self.max_new):
            self.graph.replay()
            if self.every and (i + 1) % self.every == 0 and torch.isin(self.out[: i + 1], self.eos).any().item():
                self.steps = i + 1
                break
        return self.out.clone()


def parse(name, args):
    """fixed = all steps | stopK = check EOS every K steps | capN = only N steps, no check."""
    if name == "fixed":
        return args.max_new_tokens, None
    if name.startswith("stop"):
        return args.max_new_tokens, int(name[4:])
    if name.startswith("cap"):
        return int(name[3:]), None
    raise ValueError(f"unknown strategy {name}")


def run(name, args, model, inputs, eos):
    n, every = parse(name, args)
    dec = EosDecoder(model, n, args.max_len, eos, every)
    dec.capture()
    prefill, decode = [], []
    for i in range(args.runs + 1):  # first run is warm-up
        first, p = timed(lambda: dec.prefill(inputs))
        out, d = timed(lambda: dec.decode())
        if i:
            prefill.append(p)
            decode.append(d)
    ids = until_eos(torch.cat([first.view(1), out]).tolist(), eos)
    steps = dec.steps
    del dec
    gc.collect()
    torch.cuda.empty_cache()
    return steps, sum(prefill) / args.runs, sum(decode) / args.runs, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--strategies", default="fixed,stop1,stop4,stop8,stop16,cap16,cap32")
    args = ap.parse_args()

    model, processor = load()
    inputs = make_inputs(processor, args.image, args.prompt)
    eos = eos_ids(processor.tokenizer)

    print(f"{'strategy':<9} {'steps':>5} {'decode ms':>10} {'total ms':>9} {'saved ms':>9}  complete answer")
    ref = base = None
    for name in args.strategies.split(","):
        try:
            steps, pre, dec_ms, ids = run(name, args, model, inputs, eos)
        except Exception as e:
            print(f"{name:<9} failed: {type(e).__name__}: {e}")
            continue
        total = pre + dec_ms
        if ref is None:
            ref, base = ids, total
            print(f"answer tokens: {len(ids)}")
        print(f"{name:<9} {steps:5d} {dec_ms:10.1f} {total:9.1f} {base - total:9.1f}  {ids == ref}")


if __name__ == "__main__":
    main()
