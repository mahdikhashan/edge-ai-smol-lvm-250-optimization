import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, StaticCache

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def load(dtype=torch.float16, device="cuda"):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
    return model, processor


def make_inputs(processor, image_path, prompt, dtype=torch.float16, device="cuda"):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    return inputs


class GraphDecoder:
    def __init__(self, model, max_new_tokens=64, max_len=4096):
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

    def step(self):
        h = self.model.model.text_model(
            input_ids=self.ids, cache_position=self.pos, past_key_values=self.cache, use_cache=True
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

    def generate(self, inputs, use_graph=True):
        first = self.prefill(inputs)
        return torch.cat([first.view(1), self.decode(use_graph)]).tolist()


if __name__ == "__main__":
    import sys

    model, processor = load()
    inputs = make_inputs(processor, sys.argv[1], "Describe any visible defects.")
    dec = GraphDecoder(model)
    dec.capture()
    print(processor.tokenizer.decode(dec.generate(inputs), skip_special_tokens=True))
