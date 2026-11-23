# DE1-SoC board measurements · November 22, 2026

## Matched TinyViT-21M inference

The operator ran the two-thread MNN runner with the qualified 90 MHz NI82 image. The benchmark replayed the same three fixtures across four modes for two passes, reversing mode order on the second pass. Each session ran three warm-ups. Each mean below covers six timed graph calls and excludes preprocessing and session startup.

| Mode | Mean inference (ms) | Saving from preceding row (ms) |
|---|---:|---:|
| HPS FP32 | 3549.0 | — |
| HPS mixed precision | 3478.0 | 71.0 |
| Hybrid, separate FC1 and FC2 | 3472.6 | 5.4 |
| Hybrid, resident FC1–GELU–FC2 | 3437.2 | 35.4 |

At fixed precision the resident FPGA path saved 40.8 ms per image (1.17%). Including the precision change, the reduction from FP32 was 111.8 ms (3.15%).

| Paired latency saving (ms) | Pass 1 | Pass 2 |
|---|---:|---:|
| Mixed CPU to separate offload | 13.3 | −2.4 |
| Separate offload to resident pair | 34.0 | 36.6 |
| Mixed CPU to resident pair | 47.3 | 34.2 |

Fusion saved time in both passes. Separate offload alone varied with the pass. Reproduce from the deployed demo directory with `sudo python3 benchmark21.py fixtures comparison-before.csv --threads 2` using a fresh output filename.

## Live USB camera

HPS Linux captured MJPEG frames and preprocessed selected frames. Each row is one reported prediction. Request time begins at frame selection. Inference covers the MNN graph call. The two runs saw different frames.

| Mode | Frame | Prediction | Inference (ms) | Preprocessing (ms) | Request (ms) |
|---|---:|---|---:|---:|---:|
| CPU FP32 | 1 | sunglasses | 4528.2 | 112.5 | 4791.5 |
| CPU FP32 | 144 | window shade | 4506.8 | 112.4 | 4660.4 |
| CPU FP32 | 284 | sunglasses | 4428.9 | 111.5 | 4578.2 |
| Hybrid | 2 | fountain pen | 3873.9 | 166.2 | 4411.6 |
| Hybrid | 134 | ballpoint | 3853.5 | 121.4 | 4018.6 |
| Hybrid | 254 | ballpoint | 3871.0 | 120.9 | 4031.3 |

CPU frames reported zero FPGA jobs. Every hybrid frame reported 207 jobs, two offloaded layers and one resident pair. Mean live inference was 4488.0 ms on FP32 CPU and 3866.1 ms in hybrid mode. Mean request time was 4676.7 ms and 4153.8 ms respectively. Completed predictions per second, sensor-to-display latency and peak memory were not reported in these terminal excerpts.

## HPS-only accuracy

Two-thread FP32 MNN classified the same 200 saved Imagenette validation images (20 per class) with both models. Both received identical bicubic-resized, center-cropped, normalized inputs and scored all 1,000 ImageNet classes.

| Model | Top-1 correct | Top-5 correct |
|---|---:|---:|
| TinyViT-5M | 158/200 (79.0%) | 195/200 (97.5%) |
| TinyViT-21M | 167/200 (83.5%) | 199/200 (99.5%) |

The board used Pillow 3.1.2 and NumPy 1.11.0. The separate preprocessing fixture check matched landscape.png exactly and reported maximum absolute normalized errors of 0.0175070763 for portrait.png and 0.0175070167 for jpeg_landscape.jpg. Both classifiers used the same board transform.

Source: operator-reported board terminal output and summaries from comparison-before-run2.csv. The original per-image board CSV files were not supplied with these results. See the [accuracy manifest](../../demo/accuracy200/manifest.json) and [board run instructions](../../README.md).
