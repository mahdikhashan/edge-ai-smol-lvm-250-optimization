# SmolVLM-256M: CUDA Graph Decoding

This project makes SmolVLM-256M generate text faster on a GPU. It records one decode step as a CUDA graph and replays the graph for each token.

## Files

| File | Use |
|---|---|
| `smolvlm_cuda_graphs.py` | Graph decoder. Run it on one image to get an answer. |
| `bench.py` | Benchmark: bare model, eager decoding, graph decoding. Also times preprocessing and logs memory. `--preprocess` sets the image processor: `default`, `slow`, `fast-cpu` or `fast-gpu`. |
| `attn_check.py` | Experiment: compares attention implementations (default, `eager`, `sdpa`, `sdpa+mask`). Run with `make attn`. |
| `smolvlm_256m_attention_check.ipynb` | Notebook for the attention experiment. |
| `hybrid_attn.py` | Experiment: `sdpa` for prefill and `eager` for decode (`hybrid`), compared with `sdpa` and `eager`. Run with `make hybrid`. |
| `smolvlm_256m_hybrid_attention.ipynb` | Notebook for the hybrid attention experiment. |
| `gpu_preprocess.py` | Experiment: image preprocessing with the slow processor, the fast processor on the CPU, and the fast processor on the GPU. Run with `make preprocess`. |
| `smolvlm_256m_gpu_preprocessing.ipynb` | Notebook for the preprocessing experiment. |
| `smolvlm_256m_benchmark.ipynb` | Notebook that runs `bench.py` for each `--preprocess` option (`slow`, `fast-cpu`, `fast-gpu`). |
| `eos_stop.py` | Experiment: stop the decode loop at EOS (`stopK`) or cap the steps (`capN`), compared with all steps (`fixed`). Run with `make eos`. |
| `smolvlm_256m_eos_stop.ipynb` | Notebook for the EOS experiment. |
| `prefill_profile.py` | Experiment: splits the prefill time into SigLIP, connector, decoder and LM head, for the default tiling and with no image splitting. Run with `make prefill`. |
| `smolvlm_256m_prefill_profile.ipynb` | Notebook for the prefill profile. |
| `trt_siglip.py` | Experiment: exports the SigLIP encoder to ONNX, builds an FP16 TensorRT engine, and swaps it into the model. Run with `make trt`. |
| `smolvlm_256m_trt_siglip.ipynb` | Notebook for the TensorRT experiment. |
| `tile_sweep.py` | Experiment: runs a range of tile settings (`default`, `sizeN`, `nosplit`) and reports tiles, tokens, cache length, request time and answer. Run with `make tiles`. |
| `smolvlm_256m_tile_sweep.ipynb` | Notebook for the tile experiment. |
| `get_data.py` | Downloads the EVLM paper datasets (only the needed files): one 4 s clip per environment from the UZH power line dataset, and the InsPLAD fault crops. Run with `make data`. |
| `evlm_eval.py` | EVLM benchmark with the best setup. Task `clips`: frame filter + VLM, segment latency. Task `insplad`: component and rust accuracy. Each speed-up is a flag for ablation. Run with `make evlm`. |
| `smolvlm_256m_evlm_eval.ipynb` | Notebook: dataset setup, both tasks, and an ablation table. |
| `smolvlm_256m_cuda_graph_decoding.ipynb` | Notebook that runs both scripts and shows the results. |
| `Makefile` | `make run`, `make bench`, or `make` for both. Set the image with `IMAGE=path`. |
| `CHANGELOG.md` | List of changes. |

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

## Prefill profile

Test setup: one Colab GPU, one image, `max_len` 1024. The script `prefill_profile.py` puts CUDA event timers around four parts of the prefill. The `nosplit` variant turns the image splitting off.

| Variant | Tiles | Tokens | Prefill (ms) | SigLIP (ms) | SigLIP share | Connector (ms) | Decoder (ms) | Decoder share | LM head (ms) | Other (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 13 | 875 | 268.9 | 182.7 | 68% | 1.1 | 80.6 | 30% | 1.6 | 3.0 |
| nosplit | 1 | 80 | 53.2 | 11.9 | 22% | 0.1 | 39.0 | 73% | 0.4 | 1.8 |

Prefill time saved if SigLIP were faster (default variant):

| SigLIP speed-up | Time saved (ms) | Prefill (ms) |
|---|---|---|
| 1.5 times | 60.9 | 208.0 |
| 2 times | 91.3 | 177.6 |
| 4 times | 137.0 | 131.9 |

- SigLIP uses 68% of the prefill (183 ms). This is about 50% of the whole 369 ms request. It is the largest single part.
- The compute estimate was too high. It said SigLIP needs about 10 times the compute of the decoder. The measured time ratio is 2.3 times.
- A tile costs about 14 ms in SigLIP and about 3 ms in the decoder (64 tokens). This is about 17 ms of prefill for each tile (estimate from the two variants).
- The decoder prefill has a fixed time of about 35 ms and about 0.05 ms for each token (estimate from two points). The fixed time is close to the time of one eager decode step (32 ms). The cause is probably launch overhead. We did not verify this.
- The LM head takes only 1.6 ms. Computing the logits for all tokens costs little, so keeping only the last logits is not worth it.
- A TensorRT engine for SigLIP can save at most 183 ms. A SigLIP that is 2 times faster saves 91 ms (25% of the request). We do not know what TensorRT gives on this model.
- Fewer tiles gives the largest gain. With no splitting, the prefill takes 53 ms and not 269 ms. This can lower accuracy.
- The numbers come from one image and one GPU.

## TensorRT engine for SigLIP

Test setup: one Colab GPU, one image (13 tiles), TensorRT 11.4 with a strongly typed fp16 network. The script `trt_siglip.py` exports the SigLIP encoder to ONNX and builds the engine. The ONNX export took 2 s, the engine build took 26 s, and the engine is 163 MB.

| Option | Prefill (ms) | SigLIP (ms) | SigLIP speed-up | Prefill saved (ms) | Same answer |
|---|---|---|---|---|---|
| pytorch | 285.2 | 196.9 | 1.00 | 0.0 | True |
| tensorrt | 253.0 | 160.6 | 1.23 | 32.2 | True |

Encoder output against PyTorch fp16: maximum absolute difference 1.5469, mean absolute difference 0.00550, mean absolute value 0.315.

- The engine saves 32 ms of prefill, about 8-9% of the request. A SigLIP that is 2 times faster would save 91 ms.
- The SigLIP time includes the patch embedding, which stays in PyTorch.
- TensorRT warned that it ran on the default CUDA stream. A separate stream could be faster. We did not test this.
- The engine has a fixed shape (13 tiles). It is tied to the GPU and to the TensorRT version, so it needs a new build on the Orin.
- The answer matched for one image only. The difference was measured against PyTorch fp16 and not against fp32.

## EVLM benchmark

The EVLM paper (Farahani et al., EDGE 2026) uses two public datasets (its Section V-B). It has no appendix.

| Dataset | Source | Used for | What `get_data.py` downloads |
|---|---|---|---|
| UZH power line dataset (Xing et al., IROS 2023, ref. [35]) | [download.ifi.uzh.ch](https://download.ifi.uzh.ch/rpg/powerline_tracking_dataset/power_line_dataset.zip), 25 GB zip | Speed tests on 4 s clips | One clip of 120 frames (720x480, 30 fps) for each of 8 environments, 501 MB |
| InsPLAD (Vieira e Silva et al., 2023, ref. [43]) | [Mendeley Data](https://data.mendeley.com/datasets/5n3fjgvfyz/1), 6.4 GB zip, CC BY-NC 3.0 | Accuracy test | `supervised_fault_classification.zip`: 5 assets, labels good, rust, missing-cap and bird-nest. 873 MB download, 860 MB extracted |

- Each UZH environment folder is one continuous flight. The frames are split at random into train, val and test, so `get_data.py` sorts them by frame number and cuts a continuous clip.
- The paper does not name the UZH folders. Our mapping: forest = `powerline_1_forest`, city1 = `powerlines_16_city` (start of the flight; the middle is grass), mountain = `powerline_7_mountain`, city2 = `powerline_5_japan`, sand = `powerline_3_sand`, warehouse = `powerline_11_warehouse`, wasteland = `powerline_4_wasteland`, industrial = `powerline_2_industrial`.
- InsPLAD-fault has no test split. `evlm_eval.py` uses the `val` split.
- A first check on the clips (CPU only, on a Mac): the histogram filter (512 bins, threshold 0.95) keeps 1 to 2 key frames per 4 s clip and takes about 2.3 ms per frame. PNG decode takes about 7 ms per frame.
- We use the base model. The paper fine-tunes it with LoRA, and those weights are not public. LoRA does not change the speed after the weights are merged, so the latency can be compared.
- To compare with the paper, run on the Jetson Orin and compare the time for each VLM call (paper: about 6.448 s / 3.45 calls = 1.87 s, our estimate from Table V), the tokens per answer and the decode ms per token. Do not compare the time for each segment directly: the number of key frames changes it. The notebook's last cell does this for a paper-like baseline and the best setup.
- The segment time in `evlm_eval.py` is the filter time plus the VLM calls. PNG decode (about 7-10 ms per frame on a Mac CPU) is reported apart, because the paper decodes video.
