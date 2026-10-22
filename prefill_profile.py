import argparse

import torch
from PIL import Image

from bench import timed
from smolvlm_cuda_graphs import PROMPT, GraphDecoder, chat_text, load


class Timers:
    """CUDA-event timers around the forward pass of the given modules."""

    def __init__(self, modules):
        self.events = {name: [] for name in modules}
        self.handles = []
        for name, m in modules.items():
            self.handles.append(m.register_forward_pre_hook(self._pre(name)))
            self.handles.append(m.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(module, args):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.events[name].append([e, None])

        return hook

    def _post(self, name):
        def hook(module, args, output):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.events[name][-1][1] = e

        return hook

    def reset(self):
        for ev in self.events.values():
            ev.clear()

    def ms(self):  # call after torch.cuda.synchronize()
        return {name: sum(s.elapsed_time(e) for s, e in ev) for name, ev in self.events.items()}

    def remove(self):
        for h in self.handles:
            h.remove()


def make_inputs(processor, path, prompt, split):
    text = chat_text(processor, prompt)
    image = Image.open(path).convert("RGB")
    kw = {} if split else {"do_image_splitting": False}
    try:
        inputs = processor(text=text, images=[image], return_tensors="pt", images_kwargs=kw)
    except (TypeError, ValueError):
        inputs = processor(text=text, images=[image], return_tensors="pt", **kw)
    inputs = inputs.to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    return inputs


def profile(model, dec, inputs, runs):
    timers = Timers({
        "siglip": model.model.vision_model,
        "connector": model.model.connector,
        "decoder": model.model.text_model,
        "lm_head": model.lm_head,
    })
    totals, parts = [], []
    for i in range(runs + 1):  # first run is warm-up
        timers.reset()
        _, total = timed(lambda: dec.prefill(inputs))
        if i:
            totals.append(total)
            parts.append(timers.ms())
    timers.remove()
    mean = {k: sum(p[k] for p in parts) / runs for k in parts[0]}
    return sum(totals) / runs, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--variants", default="default,nosplit")
    args = ap.parse_args()

    model, processor = load()
    dec = GraphDecoder(model, args.max_new_tokens, args.max_len)

    print(
        f"{'variant':<8} {'tiles':>5} {'tokens':>6} {'prefill ms':>10} {'SigLIP ms':>10} {'%':>4} "
        f"{'connector':>9} {'decoder ms':>10} {'%':>4} {'lm_head':>8} {'other':>6}"
    )
    first = None
    for name in args.variants.split(","):
        try:
            inputs = make_inputs(processor, args.image, args.prompt, split=name == "default")
            total, p = profile(model, dec, inputs, args.runs)
        except Exception as e:
            print(f"{name:<8} failed: {type(e).__name__}: {e}")
            continue
        tiles, tokens = inputs["pixel_values"].shape[1], inputs["input_ids"].shape[1]
        other = total - sum(p.values())
        print(
            f"{name:<8} {tiles:5d} {tokens:6d} {total:10.1f} {p['siglip']:10.1f} {100 * p['siglip'] / total:4.0f} "
            f"{p['connector']:9.1f} {p['decoder']:10.1f} {100 * p['decoder'] / total:4.0f} {p['lm_head']:8.1f} {other:6.1f}"
        )
        first = first or (total, p["siglip"])

    if first:
        total, siglip = first
        print("prefill time saved if SigLIP were faster (first variant):")
        for s in (1.5, 2, 4):
            print(f"  SigLIP {s}x faster: {siglip * (1 - 1 / s):6.1f} ms saved, prefill {total - siglip * (1 - 1 / s):6.1f} ms")


if __name__ == "__main__":
    main()
