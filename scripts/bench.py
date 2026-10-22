import argparse
import time

import torch

from smolvlm_cuda_graphs import GraphDecoder, load, make_inputs


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1e3


def bench(dec, inputs, use_graph, runs):
    prefill, decode = [], []
    for i in range(runs + 1):  # first run is warm-up
        _, p = timed(lambda: dec.prefill(inputs))
        ids, d = timed(lambda: dec.decode(use_graph))
        if i:
            prefill.append(p)
            decode.append(d / dec.max_new)
    return ids.tolist(), sum(prefill) / runs, sum(decode) / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe any visible defects.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    model, processor = load()
    inputs = make_inputs(processor, args.image, args.prompt)
    dec = GraphDecoder(model, args.max_new_tokens)
    dec.capture()

    outputs = {}
    for mode in ("eager", "graph"):
        ids, pre, tpot = bench(dec, inputs, mode == "graph", args.runs)
        outputs[mode] = ids
        print(f"{mode:>5}: prefill {pre:6.1f} ms | decode {tpot:5.2f} ms/token | {1000 / tpot:6.1f} tok/s")

    print("identical outputs:", outputs["eager"] == outputs["graph"])


if __name__ == "__main__":
    main()
