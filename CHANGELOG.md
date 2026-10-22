# Changelog

All notable changes to this project are recorded here, newest first.

## 2026-10-22

### Added
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
