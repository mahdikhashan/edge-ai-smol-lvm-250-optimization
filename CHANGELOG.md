# Changelog

All notable changes to this project are recorded here, newest first.

## 2026-10-22

### Fixed
- `trt_siglip.py`: engine build failed with "BuilderFlag has no attribute FP16" on new TensorRT versions. The encoder is now exported in fp16. The builder sets the FP16 flag if it exists, and builds a strongly typed network if it does not. The engine wrapper reads the engine tensor types.
- `trt_siglip.py`: the ONNX export failed with "Module onnx is not installed". The notebook now installs `onnx` with `tensorrt`, and the script stops at the start with a clear message if either package is missing.
- `GraphDecoder` with `eager` attention: graph capture failed with "Cannot copy between CPU and CUDA tensors". The decode step now passes an explicit 4D mask built on the GPU when the text model uses `eager` attention. The `sdpa` path is unchanged.

### Added
- `bench.py --size N` (resize the longest image side to N pixels, so the image has fewer tiles) and `bench.py --trt` (build or load a TensorRT engine for the SigLIP encoder and rerun the stop-at-EOS benchmark with it, with an answer check). The engine file name includes the tile count.
- `smolvlm_256m_benchmark.ipynb`: new last cell runs the benchmark with `--size 1024`, `--max-len 448` and `--trt`.
- `tile_sweep.py` and `smolvlm_256m_tile_sweep.ipynb`: experiment that runs a range of tile settings (`default`, `sizeN`, `nosplit`) on one or more images and reports tiles, prompt tokens, cache length, input/prefill/decode time with stop at EOS, request time, speed-up and the answer. `make tiles` runs it.
- `trt_siglip.py` and `smolvlm_256m_trt_siglip.ipynb`: experiment that exports the SigLIP encoder to ONNX, builds an FP16 TensorRT engine (`trtexec`, or the Python API if `trtexec` is not found), swaps it into the model, and compares prefill time, SigLIP time, encoder output difference and answer against PyTorch. `make trt` runs it. `.gitignore` ignores the ONNX and engine files.
- `prefill_profile.py` and `smolvlm_256m_prefill_profile.ipynb`: experiment that measures the share of the prefill time used by SigLIP, the connector, the decoder and the LM head (CUDA events), for the default tiling and with no image splitting. It also prints the time saved if SigLIP were faster. `make prefill` runs it.
- `smolvlm_256m_benchmark.ipynb`: new last cell runs the benchmark with `fast-gpu` and prints the input time, the graph timing, and the request time with 64 steps and with stop at EOS.
- `GraphDecoder.decode(eos=...)` stops after the step that makes an EOS token (check after every step). `generate` uses it. `bench.py` prints a `graph + stop at EOS` line and the request time with 64 steps and with the stop.
- `eos_stop.py` and `smolvlm_256m_eos_stop.ipynb`: experiment that compares running all decode steps (`fixed`), checking for EOS every K steps (`stopK`) and capping the steps (`capN`): steps run, decode time, total time, time saved, answer complete. `make eos` runs it.
- `smolvlm_256m_benchmark.ipynb`: runs the full benchmark (bare, eager, graph, preprocessing timers, memory log, `--max-len 1024`) once for each `--preprocess` option.
- `bench.py --preprocess {default,slow,fast-cpu,fast-gpu}`: picks the image processor and the device that resizes and normalizes the image. `default` keeps the old behavior. `make bench PREPROCESS=fast-gpu` uses it.
- `encode` helper and `PREPROCESS` table in `smolvlm_cuda_graphs.py`. `load()` takes `fast`, and `make_inputs()` takes `image_device`. `gpu_preprocess.py` imports `encode` from there.
- `gpu_preprocess.py` and `smolvlm_256m_gpu_preprocessing.ipynb`: experiment that compares the slow image processor, the fast processor on the CPU and the fast processor on the GPU (tile + normalize time, total input time, peak VRAM, pixel difference, answer match). `make preprocess` runs it.
- `hybrid_attn.py` and `smolvlm_256m_hybrid_attention.ipynb`: experiment that captures the decode graph with `eager` attention and runs the prefill with `sdpa`, compared with plain `sdpa` and plain `eager` (prefill, decode, total time, peak VRAM, answer match). `make hybrid` runs it.
- `attn_check.py`: option name `sdpa+mask` runs `sdpa` attention with the decoder's own GPU-built mask (`GraphDecoder(own_mask=True)`), to find out whether the faster `eager` decode comes from the mask or from the attention. New last cell in `smolvlm_256m_attention_check.ipynb` runs it.
- `attn_check.py` and `smolvlm_256m_attention_check.ipynb`: experiment that reports the attention implementation in use and compares `default`, `eager` and `sdpa` (prefill, graph decode, peak VRAM, answer match). `make attn` runs it.
- `README.md`: project overview, file list, and a ranked table of optimizations (open and done) based on the measured results.
- `bench.py`: times the CPU preprocessing steps (JPEG decode, tile + normalize, tokenize, and the `make_inputs` total) before the model benchmark.
- `bench.py`: logs RAM (peak process memory) and VRAM (allocated, peak, reserved) after model load and after the benchmark.
- `chat_text` helper in `smolvlm_cuda_graphs.py`, shared by `make_inputs` and the benchmark.
- `run.ipynb`: final cell with the benchmark results and a short analysis (table of decode time by mode and `max_len`).
- `bench.py --max-len`: sets the static KV-cache length, to test how cache size affects graph decode time. Prints the prompt token count.
- `run.ipynb`: last cell loops the benchmark over `--max-len` 512, 1024 and 2048 (4096 is the default run in the cell above).
- `GraphDecoder.prefill` asserts that the prompt and new tokens fit in `max_len`.
- `bench.py`: bare baseline (plain `model.generate`, no static cache or graph) on the same image and prompt, with an output comparison against graph decoding.
- `smolvlm_cuda_graphs.py`: CUDA-graph decoding for SmolVLM-256M (eager prefill, graph-replayed decode with a static KV cache).
- `bench.py`: benchmark comparing eager and graph decoding, with an output-equality check.
- `Makefile`: `make run`, `make bench`, and `make` (both), with `IMAGE` and `PYTHON` overrides.
- `run.ipynb`: notebook that runs the script and the benchmark.
- `frame.jpg`: sample test image.
- `.gitignore` entries for `__pycache__/` and `GH_TOKEN`.
- `CHANGELOG.md`.

### Changed
- `make_inputs` and `encode` in `smolvlm_cuda_graphs.py` take `size` / `image_kwargs` for the tile setting.
- README: added the TensorRT SigLIP results (SigLIP 1.23 times faster, prefill 32 ms less).
- README: added the prefill profile results. SigLIP uses 68% of the prefill (about 50% of the request). The LM head costs 1.6 ms.
- README ranking: re-ranked with the measured SigLIP share. Added "CUDA graph for the decoder part of the prefill" as a new open row.
- Notebook `smolvlm_256m_benchmark.ipynb`: added the stop-at-EOS result cell. The request takes 369 ms with the stop against 806 ms with 64 steps.
- README ranking: re-ranked the open optimizations for the new request time (369 ms, 74% prefill). Prefill items moved up. Frame pipelining and quantization moved down. The hybrid attention row notes the small gain for short answers.
- Notebook `smolvlm_256m_eos_stop.ipynb`: added the result cell. `stop1` cuts decode from 519 ms to 79 ms for a 9-token answer.
- README ranking: the stop-at-EOS row moved to the done table. The ranks below it moved up by one.
- `bench_preprocess` returns the `make_inputs` total in ms.
- Notebook `smolvlm_256m_benchmark.ipynb`: added the result cell. With graph decoding and `fast-gpu`, the request takes 823 ms against 3152 ms for the bare model with the slow processor (3.8 times faster).
- README: the GPU preprocessing row in the done table now shows the end-to-end gain.
- `bench.mean_ms` synchronizes the GPU after each call, so GPU preprocessing steps are timed correctly.
- Notebook `smolvlm_256m_gpu_preprocessing.ipynb`: added the result cell. The fast processor on the GPU prepares the input in 11.6 ms against 299 ms for the slow processor.
- README ranking: the GPU preprocessing row moved to the done table. The ranks below it moved up by one.
- Notebook `smolvlm_256m_hybrid_attention.ipynb`: added the result cell. The hybrid takes 700 ms for 64 tokens against 796 ms for `sdpa` and 808 ms for `eager`.
- README ranking: the hybrid attention row moved to the done table. The ranks below it moved up by one.
- Notebook `smolvlm_256m_attention_check.ipynb`: added the mask-test result cell. The mask explains only 0.18 ms of the 1.7 ms/token decode gap, so the attention itself is the cause.
- README ranking: "eager for decode only, sdpa for prefill" moved to rank 4, with the measured gain (about 107 ms for 64 tokens).
- Notebook `smolvlm_256m_attention_check.ipynb`: added a result cell. The default attention is already `sdpa`. `sdpa` is faster in prefill and uses less VRAM, and `eager` is faster in decode.
- README ranking: rank 7 is now "eager for decode only, sdpa for prefill". The SDPA check moved to the done table.
- `load()` takes an optional `attn` argument that sets the attention implementation.
- Notebook result cell: added request-time and memory tables and updated numbers with preprocessing and memory measurements.
- Renamed `run.ipynb` to `smolvlm_256m_cuda_graph_decoding.ipynb`.
- `bench.py`: the bare-vs-graph output check now compares only up to the first EOS, because the bare run blocks EOS to force a fixed token count.
- `eos_ids` helper in `smolvlm_cuda_graphs.py`, shared by the script and `bench.py`.
- `run.ipynb`: updated kernel metadata (VS Code kernel name, removed `language_info`).
- `run.ipynb`: added a title cell and a short method description at the top.
- `run.ipynb`: first two cells now display `frame.jpg` and print the prompt used by the script.
- Prompt is a single `PROMPT` constant in `smolvlm_cuda_graphs.py`, shared by the script and `bench.py`.
- Moved scripts from `scripts/` to the repo root; updated the Makefile and notebook paths.
- `generate` now stops at the first EOS or `<end_of_utterance>` token, removing junk output after the answer.
- Use the `dtype` argument instead of the deprecated `torch_dtype`.
- Updated the run notebook.

### Removed
- Untracked the committed `__pycache__` directory.
