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
