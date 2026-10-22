# SmolVLM-256M: CUDA Graph Decoding

This project makes SmolVLM-256M generate text faster on a GPU. It records one decode step as a CUDA graph and replays the graph for each token.

## Files

| File | Use |
|---|---|
| `smolvlm_cuda_graphs.py` | Graph decoder. Run it on one image to get an answer. |
| `bench.py` | Benchmark: bare model, eager decoding, graph decoding. Also times CPU preprocessing and logs memory. |
| `attn_check.py` | Experiment: compares attention implementations (default, `eager`, `sdpa`, `sdpa+mask`). Run with `make attn`. |
| `smolvlm_256m_attention_check.ipynb` | Notebook for the attention experiment. |
| `smolvlm_256m_cuda_graph_decoding.ipynb` | Notebook that runs both scripts and shows the results. |
| `Makefile` | `make run`, `make bench`, or `make` for both. Set the image with `IMAGE=path`. |
| `CHANGELOG.md` | List of changes. |

## Optimization ranking

The rank uses three things: the time an optimization can save in one request, how sure we are, and how much work it needs.

Test setup: one Colab GPU, one image, 875 prompt tokens (about 13 tiles), 64 new tokens, graph decoding with max_len 1024.

Time for one request: preprocessing 325 ms, prefill 286 ms, decode 524 ms (8.2 ms per token). Total: 1135 ms.

"Est." marks a number we did not measure. Check all numbers on the Jetson Orin.

### Open optimizations

| Rank | Optimization | Stage | Evidence | Expected gain | Effort |
|---|---|---|---|---|---|
| 1 | Resize and normalize on the GPU (fast image processor, DALI, or VPI) | Preprocessing (CPU) | Tile and normalize take 308 ms of 1135 ms. | Up to 308 ms (est.) | Low |
| 2 | Fewer image tiles (lower the split resolution, or do not split) | Preprocessing, prefill, cache | The prompt has 875 tokens. Tile time and prefill time grow with the tile count. | Large. It scales with the tile count. It can lower accuracy. | Low |
| 3 | Stop at EOS and lower `max_new_tokens` | Decode | Each token costs 8.2 ms. The script always runs all 64 steps. | About 8.2 ms for each token removed | Low |
| 4 | Run the VLM only on frames that YOLO flags (distill or fine-tune for the task) | Whole request | Each skipped frame saves all 1135 ms. | Depends on the share of skipped frames | High |
| 5 | Cache vision embeddings for near-identical frames | Prefill | SigLIP is probably the largest part of the 286 ms prefill (est.). | Most of the prefill for repeated scenes | Medium |
| 6 | Pipeline frames: the CPU prepares frame N+1 while the GPU runs frame N | Throughput | CPU work takes 325 ms. GPU work takes about 810 ms. | About 1.4 times more frames per second. Latency does not change. | Medium |
| 7 | Use `eager` attention for decode only and `sdpa` for prefill | Decode | `eager` decode takes 6.45 ms/token. `sdpa` decode takes 8.15 ms/token. The cause is not known. | Up to 1.7 ms for each token (est.) | Medium |
| 8 | Weight quantization (INT8, INT4) | Decode | Weights use 497 MB. Decode has a fixed cost of about 5.3 ms per token. | Unknown. Profile first. It can lower accuracy. | Medium |
| 9 | TensorRT engine for SigLIP | Prefill | Prefill takes 286 ms (25% of the request). | Unknown | High |
| 10 | Token pruning or merging in SigLIP (for example ToMe) | Prefill | Same as rank 9. | Unknown. It can lower accuracy. | Medium |
| 11 | Compiled runtimes (TensorRT-LLM, llama.cpp, MLC-LLM, ONNX Runtime with TensorRT) | Decode, prefill | CUDA graphs already remove most of the launch cost. | Small (est.) | High |
| 12 | Zero-copy unified memory | Copy to GPU | The copy takes about 15 ms. | Up to 15 ms | Medium |
| 13 | Hardware JPEG decode (nvJPEG, Jetson decoder) | Preprocessing | JPEG decode takes 0.7 ms for the test image. | Small for this image. More for a large frame. | Medium |
| 14 | Maximum clocks (`nvpmodel -m 0`, `jetson_clocks`) | All, Jetson only | Not measured. Colab has no Jetson clocks. | Steadier timing | Low |
| 15 | Batch the tiles in one call | Prefill | The model probably does this already. Not checked. | Probably none | Low |
| 16 | FP8 or INT8 KV cache | Decode | The cache takes about 23 MB for each 1000 slots. | Small | Medium |
| 17 | Reuse the prefix cache between frames | Prefill | The image tokens come early in the prompt, so the shared prefix is probably short. | Small (est.) | Medium |
| 18 | Speculative or prompt-lookup decoding | Decode | Decode takes 8.2 ms per token. The answers are short. | Small (est.) | Medium |

### Done optimizations

| Optimization | Stage | Result |
|---|---|---|
| CUDA graphs (manual capture; `torch.compile` with `reduce-overhead` is not tried) | Decode | 31.8 ms/token (eager) to 16.7 ms/token at max_len 4096. 1.9 times faster. |
| Static KV cache with a small `max_len` | Decode | 16.7 ms/token at max_len 4096 to 8.2 ms/token at max_len 1024. 2.0 times faster. |
| FP16 weights | All | Weights use 497 MB of VRAM. |
| SDPA attention (already the default for the text and vision models) | Prefill | `sdpa` prefill takes 280 ms. `eager` prefill takes 393 ms. Peak VRAM: 889 MB and 2012 MB. |
