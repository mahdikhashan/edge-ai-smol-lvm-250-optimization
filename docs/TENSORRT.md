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
