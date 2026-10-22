# Changelog

All notable changes to this project are recorded here, newest first.

## 2026-10-22

### Added
- `smolvlm_cuda_graphs.py`: CUDA-graph decoding for SmolVLM-256M (eager prefill, graph-replayed decode with a static KV cache).
- `bench.py`: benchmark comparing eager and graph decoding, with an output-equality check.
- `Makefile`: `make run`, `make bench`, and `make` (both), with `IMAGE` and `PYTHON` overrides.
- `run.ipynb`: notebook that runs the script and the benchmark.
- `frame.jpg`: sample test image.
- `.gitignore` entries for `__pycache__/` and `GH_TOKEN`.
- `CHANGELOG.md`.

### Changed
- `run.ipynb`: added a title cell and a short method description at the top.
- `run.ipynb`: first two cells now display `frame.jpg` and print the prompt used by the script.
- Prompt is a single `PROMPT` constant in `smolvlm_cuda_graphs.py`, shared by the script and `bench.py`.
- Moved scripts from `scripts/` to the repo root; updated the Makefile and notebook paths.
- `generate` now stops at the first EOS or `<end_of_utterance>` token, removing junk output after the answer.
- Use the `dtype` argument instead of the deprecated `torch_dtype`.
- Updated the run notebook.

### Removed
- Untracked the committed `__pycache__` directory.
