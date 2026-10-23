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

## EVLM ablation results

Test setup: one Colab GPU (not the Jetson Orin), two clips (forest and sand), about 2 VLM calls for each row. The mission prompt of the paper's Section VI-E. Each row turns off one part of the best setup. The segment time is the filter time plus the VLM calls, without PNG decode.

| Turned off | Key frames | VLM ms per call | Tokens | Decode ms per token | Segment (s) | Per call against best |
|---|---|---|---|---|---|---|
| nothing (best) | 1.00 | 221.4 | 19.5 | 6.90 | 0.54 | 1.00x |
| TensorRT | 1.00 | 239.8 | 19.5 | 6.86 | 0.57 | 1.08x |
| fast GPU preprocessing | 1.00 | 351.7 | 19.5 | 8.02 | 0.68 | 1.59x |
| fewer tiles (default tiles) | 1.00 | 389.5 | 19.5 | 8.26 | 0.71 | 1.76x |
| stop at EOS | 1.00 | 506.6 | 19.5 | 21.26 | 0.84 | 2.29x |
| CUDA graphs (eager decode) | 1.00 | 794.6 | 19.5 | 34.21 | 1.11 | 3.59x |
| graphs and stop (bare `generate`) | 1.00 | 829.5 | 19.5 | 36.04 | 1.16 | 3.75x |
| frame filter | 120.00 | 187.8 | 14.1 | 6.98 | 22.54 | - |
| all parts (paper-like baseline) | 1.00 | 1354.7 | 19.5 | 33.87 | 1.69 | 6.12x |

- The best setup is 6.1 times faster for each VLM call than the paper-like baseline (221 ms and 1355 ms). Decode is 4.9 times faster for each token (6.9 ms and 33.9 ms).
- CUDA graphs give the largest decode gain. Without them, a token takes 34 to 36 ms.
- TensorRT saves 18 ms for each call. It does not change the decode time for each token, because it only makes the prefill faster.
- The default tiles cost 168 ms for each call. The decode is also slower (8.3 ms for each token), because the longer prompt needs a larger cache.
- The 21.3 ms per token without stop at EOS is not a slower step. The decode runs all 64 steps, but only the 19.5 answer tokens are counted. One step still takes about 6.9 ms.
- The frame filter gives the largest gain for each segment: 22.5 s without it and 0.54 s with it, about 42 times less.
- In the best setup, the filter is now the largest part of a segment: about 0.32 s (120 frames at about 3 ms on the Colab CPU), against 0.22 s for the VLM.
- The slow preprocessing also makes the decode slower (8.0 ms for each token). The slow processor probably makes more tiles at `--size 1024`. We did not verify this, because the results do not record the tile count.
- Each row has about 2 calls, so differences under about 20 to 30 ms are noise.
- The answers have 19.5 tokens on average. The earlier single-image tests had 9 tokens.
- These times are from Colab. The comparison with the paper needs the "Comparison with the paper" cell on the Jetson Orin.
