## Optimization ranking

The rank uses three things: the time an optimization can save in one request, how sure we are, and how much work it needs.

Test setup: one Colab GPU, one image, 875 prompt tokens (about 13 tiles), graph decoding with max_len 1024.

Current time for one request (fast GPU preprocessing, stop at EOS, 9-token answer): input 16 ms, prefill 272 ms, decode 81 ms. Total: 369 ms. Prefill is 74% of the request, and SigLIP alone is about 50% of it. The first measurement was 1135 ms.

"Est." marks a number we did not measure. Check all numbers on the Jetson Orin.

### Open optimizations

| Rank | Optimization | Stage | Evidence | Expected gain | Effort |
|---|---|---|---|---|---|
| 1 | Fewer image tiles (lower the split resolution, or do not split) | Input, prefill, cache | Prefill takes 272 ms (74% of the request). The prompt has 875 tokens. Prefill time grows with the tile count. | Large. It scales with the tile count. It can lower accuracy. | Low |
| 2 | Cache vision embeddings for near-identical frames | Prefill | SigLIP takes 183 ms (50% of the request). A cache hit skips it. | Up to 183 ms for repeated scenes | Medium |
| 3 | Run the VLM only on frames that YOLO flags (distill or fine-tune for the task) | Whole request | Each skipped frame saves all 369 ms. | Depends on the share of skipped frames | High |
| 4 | TensorRT engine for SigLIP | Prefill | SigLIP takes 183 ms (68% of the prefill, 50% of the request). | 2 times faster: 91 ms. 4 times faster: 137 ms. Maximum: 183 ms. | High |
| 5 | Token pruning or merging in SigLIP (for example ToMe) | Prefill | Same as the TensorRT row. SigLIP time grows with the token count. | Unknown. It can lower accuracy. | Medium |
| 6 | CUDA graph for the decoder part of the prefill (needs a fixed tile count and a padded prompt) | Prefill | The decoder prefill has a fixed time of about 35 ms. One eager decode step takes about 32 ms. | Up to about 32 ms (est.) | Medium |
| 7 | Weight quantization (INT8, INT4) | Decode | Decode takes 81 ms (22% of the request). Weights use 497 MB. | At most part of 81 ms. It can lower accuracy. | Medium |
| 8 | Batch the tiles in one call | Prefill | The model probably does this already. Not checked. | Probably none | Low |
| 9 | Compiled runtimes (TensorRT-LLM, llama.cpp, MLC-LLM, ONNX Runtime with TensorRT) | Decode, prefill | CUDA graphs already remove most of the launch cost. | Small (est.) | High |
| 10 | Maximum clocks (`nvpmodel -m 0`, `jetson_clocks`) | All, Jetson only | Not measured. Colab has no Jetson clocks. | Steadier timing | Low |
| 11 | Pipeline frames: the CPU prepares frame N+1 while the GPU runs frame N | Throughput | CPU work takes 16 ms. GPU work takes about 353 ms. | About 4% more frames per second. Latency does not change. | Medium |
| 12 | Zero-copy unified memory | Copy to GPU | The whole input preparation takes 16 ms. The copy is only a part of it. | A few milliseconds (est.) | Medium |
| 13 | Hardware JPEG decode (nvJPEG, Jetson decoder) | Input | JPEG decode takes 0.7 ms for the test image. | Small for this image. More for a large frame. | Medium |
| 14 | FP8 or INT8 KV cache | Decode | The cache takes about 23 MB for each 1000 slots. | Small | Medium |
| 15 | Reuse the prefix cache between frames | Prefill | The image tokens come early in the prompt, so the shared prefix is probably short. | Small (est.) | Medium |
| 16 | Speculative or prompt-lookup decoding | Decode | Decode takes 81 ms for a 9-token answer. | Small (est.) | Medium |

### Done optimizations

| Optimization | Stage | Result |
|---|---|---|
| CUDA graphs (manual capture; `torch.compile` with `reduce-overhead` is not tried) | Decode | 31.8 ms/token (eager) to 16.7 ms/token at max_len 4096. 1.9 times faster. |
| Static KV cache with a small `max_len` | Decode | 16.7 ms/token at max_len 4096 to 8.2 ms/token at max_len 1024. 2.0 times faster. |
| FP16 weights | All | Weights use 497 MB of VRAM. |
| Stop at EOS: check for the end token after every step (in `GraphDecoder.decode` and `bench.py`). A lower `max_new_tokens` is not needed. | Decode | Decode for a 9-token answer: 519 ms (64 steps) to 79 ms (9 steps). The check costs less than 1 ms. |
| Resize and normalize on the GPU: fast image processor with `device="cuda"` (in `gpu_preprocess.py`, not in the main script) | Preprocessing | Full input preparation: 299 ms (`slow`) to 11.6 ms. Pixel difference is small (mean 0.005). Same answer on one image. In `bench.py`, the request (graph decoding, 64 tokens) falls from 1107 ms to 823 ms. |
| Hybrid attention: `sdpa` for prefill, `eager` for decode (in `hybrid_attn.py`, not in the main script) | Prefill, decode | Total for 64 tokens: 796 ms (`sdpa`) to 700 ms. Prefill 286 ms, decode 6.47 ms/token, peak VRAM 889 MB. For a 9-token answer the gain is only about 4 ms. |
| SDPA attention (already the default for the text and vision models) | Prefill | `sdpa` prefill takes 280 ms. `eager` prefill takes 393 ms. Peak VRAM: 889 MB and 2012 MB. |
