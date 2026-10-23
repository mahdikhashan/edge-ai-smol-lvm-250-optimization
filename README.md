## SmolVLM-256M Edge Inference Optimization

Faster inference for SmolVLM-256M on edge GPUs, without custom CUDA or Triton kernels. The repository also runs the benchmark of the EVLM power line inspection paper (Farahani et al., EDGE 2026) with the fastest setup. Tests ran on a Colab GPU. The Jetson Orin run is still open.

### Setup and run

Make sure a CUDA GPU and Python 3 are available, then clone the repository:

```bash
git clone https://github.com/mahdikhashan/edge-ai-smol-lvm-250-optimization
cd edge-ai-smol-lvm-250-optimization
```

it needs PyTorch, transformers 4.56 or newer, Pillow and NumPy. The TensorRT experiments also need `tensorrt` and `onnx`. On the Jetson, TensorRT comes with JetPack.

```bash
pip install "transformers>=4.56" pillow numpy onnx tensorrt
```

- run the CUDA-graph decoder on one image:

```bash
make run IMAGE=frame.jpg
```

#### Experiments

Each target runs one script. Each script has a notebook (`smolvlm_256m_*.ipynb`) that runs it on Colab.

- `make bench`: bare model, eager and CUDA-graph decoding, preprocessing timers, memory and stop at EOS. Options: `PREPROCESS=fast-gpu`, and `--size` and `--trt` in `bench.py`.
- `make attn`: attention implementations (`sdpa`, `eager`).
- `make hybrid`: `sdpa` for prefill and `eager` for decode.
- `make preprocess`: image preprocessing on the CPU and on the GPU.
- `make eos`: stop the decode at the end token.
- `make prefill`: prefill time of SigLIP, connector, decoder and LM head.
- `make trt`: TensorRT engine for the SigLIP encoder.
- `make tiles`: number of image tiles against speed and answer.
- `make data`, then `make evlm`: EVLM benchmark on the paper datasets (about 1.4 GB of data).

---

#### Results

On one Colab GPU:

- one frame with a 9-token answer: 159 ms with the best setup (5 tiles, fast GPU preprocessing, CUDA-graph decoding, stop at EOS, TensorRT SigLIP). The bare model takes about 954 ms (estimate).
- EVLM benchmark: each VLM call is 6.1 times faster than a paper-like baseline (221 ms and 1355 ms).

#### Debug

- `make -n <target>` shows the command of a target, and `python <script>.py --help` shows the options of a script.
- TensorRT engines are kept in `trt_siglip/`. Delete the folder after a change of GPU or TensorRT version.

#### Documentation

- [Files and notebooks](docs/FILES.md)
- [Optimization ranking](docs/RANKING.md)
- [Prefill profile](docs/PREFILL_PROFILE.md)
- [TensorRT engine for SigLIP](docs/TENSORRT.md)
- [EVLM benchmark and ablation](docs/EVLM.md)
- [Changelog](CHANGELOG.md)
