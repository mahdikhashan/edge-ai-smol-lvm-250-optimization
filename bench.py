import argparse
import time

import torch

from smolvlm_cuda_graphs import PROMPT, GraphDecoder, load, make_inputs


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    model, processor = load()
    inputs = make_inputs(processor, args.image, args.prompt)
    dec = GraphDecoder(model, args.max_new_tokens)
    dec.capture()

    outputs = {}
    outputs["bare"], pre, tpot = bench_bare(model, inputs, args.max_new_tokens, args.runs)
    print(f"{'bare':>5}: prefill {pre:6.1f} ms | decode {tpot:5.2f} ms/token | {1000 / tpot:6.1f} tok/s")
    for mode in ("eager", "graph"):
        ids, pre, tpot = bench(dec, inputs, mode == "graph", args.runs)
        outputs[mode] = ids
        print(f"{mode:>5}: prefill {pre:6.1f} ms | decode {tpot:5.2f} ms/token | {1000 / tpot:6.1f} tok/s")

    print("identical outputs (eager vs graph):", outputs["eager"] == outputs["graph"])
    print("identical outputs (bare vs graph):", outputs["bare"] == outputs["graph"])


if __name__ == "__main__":
    main()
