import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, StaticCache

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROMPT = "Describe any visible defects."
# name -> (use_fast for the image processor, device that resizes and normalizes the image)
PREPROCESS = {"default": (None, "cpu"), "slow": (False, "cpu"), "fast-cpu": (True, "cpu"), "fast-gpu": (True, "cuda")}


def load(dtype=torch.float16, device="cuda", attn=None, fast=None):
    kwargs = {} if attn is None else {"attn_implementation": attn}
    processor = AutoProcessor.from_pretrained(MODEL_ID, **({} if fast is None else {"use_fast": fast}))
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, **kwargs).to(device).eval()
    return model, processor


def eos_ids(tok):
    return {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_utterance>")}


def chat_text(processor, prompt):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def encode(processor, text, image, image_device="cpu"):
    kw = {} if image_device == "cpu" else {"device": image_device}
    try:
        return processor(text=text, images=[image], return_tensors="pt", images_kwargs=kw)
    except (TypeError, ValueError):
        return processor(text=text, images=[image], return_tensors="pt", **kw)


def make_inputs(processor, image_path, prompt, dtype=torch.float16, device="cuda", image_device="cpu"):
    text = chat_text(processor, prompt)
    image = Image.open(image_path).convert("RGB")
    inputs = encode(processor, text, image, image_device).to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    return inputs


class GraphDecoder:
    def __init__(self, model, max_new_tokens=64, max_len=4096, own_mask=False):
        self.model = model
        self.max_len = max_len
        self.max_new = max_new_tokens
        self.cache = StaticCache(config=model.config.text_config, max_cache_len=max_len)

        dev = model.device
        self.ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.idx = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(max_new_tokens, dtype=torch.long, device=dev)
        self.graph = None
        # eager attention builds its own mask with a CPU->GPU copy, which CUDA graph capture forbids
        self.own_mask = own_mask or model.config.text_config._attn_implementation == "eager"

    def mask(self):
        allowed = torch.arange(self.max_len, device=self.pos.device) <= self.pos[:, None]
        dtype = self.model.dtype
        return ((~allowed).to(dtype) * torch.finfo(dtype).min)[:, None, None, :]

    def step(self):
        extra = {"attention_mask": self.mask()} if self.own_mask else {}
        h = self.model.model.text_model(
            input_ids=self.ids, cache_position=self.pos, past_key_values=self.cache, use_cache=True, **extra
        ).last_hidden_state
        nxt = self.model.lm_head(h[:, -1]).argmax(-1, keepdim=True)
        self.ids.copy_(nxt)
        self.out.index_copy_(0, self.idx, nxt.view(1))
        self.pos.add_(1)
        self.idx.add_(1)

    @torch.inference_mode()
    def capture(self):
        # warm up on a side stream, writing to the last (unused) cache slot
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.pos.fill_(self.max_len - 1)
                self.idx.zero_()
                self.step()
        torch.cuda.current_stream().wait_stream(s)

        self.pos.fill_(self.max_len - 1)
        self.idx.zero_()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()

    @torch.inference_mode()
    def prefill(self, inputs):
        self.cache.reset()
        n = inputs["input_ids"].shape[1]
        assert n + self.max_new < self.max_len, f"prompt {n} + {self.max_new} new tokens do not fit max_len {self.max_len}"
        logits = self.model(
            **inputs, past_key_values=self.cache, use_cache=True,
            cache_position=torch.arange(n, device=self.model.device),
        ).logits
        first = logits[:, -1].argmax(-1, keepdim=True)
        self.ids.copy_(first)
        self.pos.fill_(n)
        self.idx.zero_()
        return first

    @torch.inference_mode()
    def decode(self, use_graph=True):
        for _ in range(self.max_new):
            self.graph.replay() if use_graph else self.step()
        return self.out.clone()

    def generate(self, inputs, eos_ids, use_graph=True):
        first = self.prefill(inputs)
        ids = torch.cat([first.view(1), self.decode(use_graph)]).tolist()
        for i, t in enumerate(ids):
            if t in eos_ids:
                return ids[:i]
        return ids


if __name__ == "__main__":
    import sys

    model, processor = load()
    inputs = make_inputs(processor, sys.argv[1], PROMPT)
    dec = GraphDecoder(model)
    dec.capture()
    tok = processor.tokenizer
    print(tok.decode(dec.generate(inputs, eos_ids(tok)), skip_special_tokens=True))
