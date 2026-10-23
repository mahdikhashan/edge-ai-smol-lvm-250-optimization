## Files and notebooks

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
