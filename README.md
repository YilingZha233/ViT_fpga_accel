# TinyViT-21M and TinyViT-5M · DE1-SoC camera demo

See the [physical board measurements](evidence/21m/board-measurements.md) for matched inference timings, live camera results and HPS accuracy.

The default candidate runs **TinyViT-21M, 224 × 224, 1,000 classes, using MNN**. Both supplied models come from the same 21,198,568-parameter PyTorch checkpoint as the team's earlier 21M evaluation. The default is a persistent, two-thread FP32 CPU session. The hybrid candidate moves one MLP block to the team's NI82 design. The earlier 30-layer board result belongs to a smaller model. See [analysis and measurements](ANALYSIS.md), [verification status](VALIDATION.md) and [sources](SOURCES.md).

## Run on HPS Linux

Connect the UVC camera to the board's **HPS USB host**. The deployment unit is the entire `demo/` directory, including models, executables, labels and fixtures. From a development machine, run `scp -r demo ubuntu@BOARD_IP:~/tinyvit21-demo` after replacing BOARD_IP with the board address. Then connect with `ssh ubuntu@BOARD_IP`. USB serial is also suitable for a text console.

The following commands run **inside the board shell**. The camera application requires Python 3.5+, NumPy, Pillow and FFmpeg with V4L2 support. The supplied static ARMv7 executables include the MNN runtime. The board should have enough free RAM for the model/session and the camera processes.

```sh
cd ~/tinyvit21-demo
ls -lh models/*.mnn
python3 -c 'import numpy,PIL; print(numpy.__version__,PIL.__version__)'
python3 hps_local_demo.py --verify-fixtures fixtures
python3 hps_local_demo.py --image fixtures/jpeg_landscape.jpg --snapshot-out cpu-snapshot
ls -l /dev/video0
ffmpeg -hide_banner -f v4l2 -list_formats all -i /dev/video0
python3 hps_local_demo.py --mode cpu-fp32 --threads 2 --device /dev/video0 --timings cpu-live.csv
```

The FFmpeg listing exits after reporting supported formats. The default camera mode is MJPEG, 640×480 at 30 capture frames/s; change `--size` and `--fps` to a supported combination if needed. Snapshot mode saves the input tensor, 1,000 logits, crop and top-five labels, then exits. Use a fresh snapshot directory and CSV name for each invocation. For a live session press Ctrl+C, or pass `--max-frames 101` for a bounded run.

| Mode | MNN graph | Where it executes | Expected jobs / linears / resident pairs per frame |
|---|---|---|---|
| `cpu-fp32` | 21M FP32 | HPS | 0 / 0 / 0 |
| `cpu-mixed` | Same 21M weights, stage 1 block 1 FC1 and FC2 in W8A8 | HPS | 0 / 0 / 0 |
| `hybrid-unfused` | Same mixed model | NI82 executes the two linears separately | 207 / 2 / 0 |
| `hybrid` | Same mixed model | NI82 runs both linears and keeps their activated intermediate on FPGA | 207 / 2 / 1 |

The launcher verifies the selected file against `models/manifest.json`. It checks per-frame completion counts and returns an error on a missing hook, unexpected count or incomplete resident chain. Use `cpu-mixed` as the denominator for the **incremental FPGA effect**.

### Bring up the FPGA candidate

The [bundled FC1 capacity experiment image](hardware/bitstreams/fc1-capacity/matmul-soc-90mhz.sof) matches the November 12 build. The 21M hybrid runner requires the separate NI82 image identified below.

Use the team's **90 MHz NI82** bitstream and matching bridge/boot configuration. The bundled image is [matmul-soc-native-v4-90mhz.sof](hardware/bitstreams/ni82/matmul-soc-native-v4-90mhz.sof), SHA256 `ff58e98c77d5471a02939c27d8cf1a044f25414a1c4190ee0fedef9d6019c5ad`. A normal reboot can restore the SD-card image. The driver uses a 16 MiB DMA/cache window at 0x30000000. The team's boot temporarily limited Linux to 768 MiB and enabled the bridges. Follow the [temporary boot procedure](mnn-fpga/docs/matmul-soc-integration.md) with the clock, image and reservation in the [NI82 qualification record](mnn-fpga/docs/tinyvit-native-resident-offload.md). Run one hardware client at a time.

```sh
cd ~/tinyvit21-demo
cat /proc/cmdline
sudo cat /proc/iomem
sudo ./bin/ni82_smoke21
sudo python3 hps_local_demo.py --mode hybrid-unfused --image fixtures/jpeg_landscape.jpg --snapshot-out fpga-separate
sudo python3 hps_local_demo.py --mode hybrid --image fixtures/jpeg_landscape.jpg --snapshot-out fpga-resident
sudo python3 hps_local_demo.py --mode hybrid --device /dev/video0 --timings hybrid-live.csv --max-frames 101
python3 summarize_hps_local.py hybrid-live.csv
```

The smoke executable includes an integer reference check for the 21M workload, 784 × 192 → 768 → 192. The model and CPU demo assets are included. The [physical board measurements](evidence/21m/board-measurements.md) include live camera predictions and the four-mode benchmark.

### TinyViT-5M camera demo

Deploy the updated `demo/` directory together, including the new runner and mixed model. Both candidates use the same UVC capture, camera crop, persistent MNN session, ImageNet labels and optional browser preview. From the deployed directory, check a saved frame first. For hybrid execution, bring up the 90 MHz NI82 image and run the smoke check above.

```sh
python3 hps_local_demo.py --candidate 5m --mode cpu-mixed --image fixtures/jpeg_landscape.jpg --snapshot-out five-cpu
sudo python3 hps_local_demo.py --candidate 5m --mode hybrid --image fixtures/jpeg_landscape.jpg --snapshot-out five-fpga
sudo python3 hps_local_demo.py --candidate 5m --mode hybrid --device /dev/video0 --max-frames 20 --timings five-live.csv --preview
```

To verify FPGA inference on an actual camera frame, capture one hybrid prediction and replay the saved source JPEG in CPU mixed mode. Use fresh output directories for both runs.

```sh
sudo python3 hps_local_demo.py --candidate 5m --mode hybrid --device /dev/video0 --snapshot-out five-camera-fpga
python3 hps_local_demo.py --candidate 5m --mode cpu-mixed --image five-camera-fpga/source.jpg --snapshot-out five-camera-cpu
cmp five-camera-fpga/input.bin five-camera-cpu/input.bin
```

Each `result.json` contains the top-five predictions and hardware counters. The saved `logits.bin` files contain 1,000 little-endian FP32 scores for comparison.

The terminal and CSV identify the candidate. Each accepted 5M hybrid frame requires **437 FPGA jobs, 30 linear operations and 4 resident MLP pairs**. Use `--mode hybrid-unfused` for the 434-job, 30-linear, zero-pair path. Use `--mode cpu-mixed` to compare identical weights and preprocessing on the HPS, or `--mode cpu-fp32` for the original FP32 baseline. Both CPU modes require zero hardware counters.

The bundled 5M mixed export starts with this checkout’s official distilled FP32 model. It applies [the team’s 37-layer W8A8 policy](mnn-fpga/mnn/app/tinyvit_w8a8_accuracy_allowlist.txt) using MNN 3.6.2 and 128 Imagenette training crops. To reproduce it, run `scripts/prepare_21m_data.py` on Imagenette2-160, compile `mnn-fpga/mnn/app/create_mixed_quantized_model.cpp` with the same include paths as `scripts/rebuild_models.sh`, then run `mnn-fpga/mnn/app/quantize_tinyvit_w8a8.py` with `--model demo/models/tinyvit-5m-fp32.mnn --calibration-dir DATA/train --mixed-builder PATH_TO_BUILDER --output NEW_MODEL.mnn`.

Host MNN scored the bundled mixed model at 158/200 top-1 and 195/200 top-5 on the existing Imagenette validation sample. Its top prediction matched FP32 on 198/200 inputs. The ARM runner’s offline plan found 30 linears and four pairs. For the exact mixed graph from the teammate’s board measurements, add `--model /home/ubuntu/tinyvit/tiny-vit-mixed-w8a8-accuracy.mnn` in mixed or hybrid mode. The launcher checks its published hash. [Upstream NI82 qualification](mnn-fpga/docs/tinyvit-native-resident-offload.md)

On November 23, 2026, the DE1-SoC passed the NI82 smoke test and classified five live UVC frames with the bundled mixed model. Each frame reported 437 FPGA jobs, 30 linears and four resident pairs. Median inference time was 1.117 s for this five-frame run. A separate live snapshot and its CPU replay produced byte-identical input tensors, both predicted `window shade`, and had a logit cosine similarity of 0.997589. The smoke test stalled after a Linux software reboot and passed after a physical HPS reset followed by the documented U-Boot and JTAG sequence.

### TinyViT-5M co-design camera mode

See the [TinyViT-5M board report](evidence/5m/board-measurements.md) for the three-mode latency comparison, recorded demos, and commands.

`hybrid-codesign` uses the upstream nine calibrated FP32 projections, optimized HPS attention and GELU, DMC1 chained DMA, and overlapped output staging. It checks for **400 FPGA jobs, 39 linear operations and four resident pairs** per frame. The upstream 1.595× figure was measured on warmed MNN forwards against its original FP32 checkpoint. [Benchmark and timing scope](https://github.com/PlutoPuppy/mnn-fpga/blob/a75a4b2342079a2195761a5ffc8b22a63494e9f3/docs/tinyvit-codesign-optimization.md)

Build the ARM stream executable with `sh scripts/build_runner.sh` in Linux or WSL with CMake and the arm-linux-gnueabihf toolchain. The script writes `demo/bin/tinyvit_stream_codesign`. The qualified deployment archive, `tinyvit-codesign-90mhz.tar.gz`, supplies the matching DMC1 SOF, mixed model and projection ranges. Its SHA-256 is `591b3d33bcf4cf85695ddad18dcf143389408858a59ebcfa3bc0db8746182668`. Use the temporary U-Boot/JTAG boot procedure to load the DMC1 SOF, then run from the deployed demo directory on the board.

```sh
sudo python3 hps_local_demo.py --candidate 5m --mode hybrid-codesign \
  --model models/tiny-vit-mixed-w8a8-accuracy.mnn \
  --projection-ranges models/projection-nine.tsv \
  --device /dev/video0 --max-frames 5 --timings codesign-live.csv
```

Use `--mode cpu-mixed --model models/tiny-vit-mixed-w8a8-accuracy.mnn` for the same-model CPU comparison. For the upstream FP32 baseline, use `--mode cpu-fp32 --model models/tiny-vit.mnn` with the original checkpoint. Omit `--projection-ranges` in either CPU mode. The camera CSV records both forward time (`infer_ms`) and JPEG receipt to logits (`received_to_result_ms`).

A repeated three-mode live camera comparison on the DE1-SoC measured the following mean latencies (20 samples per mode after dropping the first frame of each run):

| TinyViT-5M mode | JPEG receipt to logits | MNN inference | FPGA jobs / offloaded operations / resident pairs per frame |
|---|---:|---:|---:|
| Original FP32 CPU | 1,435.7 ms | 1,264.9 ms | 0 / 0 / 0 |
| Mixed CPU | 1,367.2 ms | 1,190.7 ms | 0 / 0 / 0 |
| DMC1 FPGA co-design | **1,003.8 ms** | **832.0 ms** | 400 / 39 / 4 |

Against original FP32 CPU, DMC1 reduced mean end-to-end camera latency by **30.1% (1.43 times faster)** and mean MNN inference latency by **34.2% (1.52 times faster)**. The upstream saved-input forward benchmark separately measured 1,207.1 ms for FP32 and 757.0 ms for DMC1 (1.595 times); that is model-only timing, not live camera latency. The live comparison used the same ARM runner and two HPS threads at 640x480 MJPEG/30 fps, with sequential captures from the same live feed. See the [full measurement report](evidence/5m/board-measurements.md) for methodology, earlier runs, raw CSV locations, recorded demos and run commands. The earlier 10-frame comparison measured 845.7 ms DMC1 inference and 1,022.5 ms camera latency; its raw CSVs are on the board in `~/tinyvit5m-camera/dmc1-latency-PxsvmJ/`.

### What the audience sees

By default the board prints the frame number, predicted ImageNet class, inference time, preprocessing time and hardware counters. Add `--preview` to the live command to publish a read-only view at `http://BOARD_IP:8080`. The crop and prediction in the result have the same frame ID; the independent live preview may be newer. Capture, JPEG decode, resize, crop, normalization, inference dispatch and label selection all run on the board. Closing the browser leaves classification running. Use a persistent board shell such as tmux if the SSH client will disconnect.

The v2 recordings are available here: [CPU FP32 v2](evidence/5m/videos/cpu-fp32-v2.mp4) and [DMC1 FPGA offload v2](evidence/5m/videos/fpga-dmc1-1.5x-v2.mp4).

### Run a mode with the browser preview

Run the selected command in PuTTY from `~/tinyvit5m-camera`. Prediction output remains in PuTTY; open `http://BOARD_IP:8080` in a browser, replacing `BOARD_IP` with the board's Ethernet or Wi-Fi address. For the direct PC Ethernet connection used here, the board address is `192.168.137.2`.

#### Original FP32 CPU

```sh
python3 hps_local_demo.py --candidate 5m --mode cpu-fp32 \
  --model models/tiny-vit.mnn \
  --runner "$PWD/bin/tinyvit_stream_codesign" \
  --device /dev/video0 --preview
```

#### Mixed CPU

```sh
python3 hps_local_demo.py --candidate 5m --mode cpu-mixed \
  --model models/tiny-vit-mixed-w8a8-accuracy.mnn \
  --runner "$PWD/bin/tinyvit_stream_codesign" \
  --device /dev/video0 --preview
```

#### DMC1 FPGA offload

Load the 90 MHz DMC1 image first. The FPGA mode requires `sudo` for hardware access.

```sh
sudo python3 hps_local_demo.py --candidate 5m --mode hybrid-codesign \
  --model models/tiny-vit-mixed-w8a8-accuracy.mnn \
  --projection-ranges models/projection-nine.tsv \
  --runner "$PWD/bin/tinyvit_stream_codesign" \
  --device /dev/video0 --preview
```

If the board is directly connected to the Windows PC and its address is missing after a reboot, restore the temporary address with `sudo ip addr replace 192.168.137.2/24 dev eth0` before opening the preview. On a router network, use the address assigned by that network instead.

The default transform is the 21M baseline transform: RGB, bicubic resize of the shorter edge to 256, 224×224 center crop, divide by 255, ImageNet mean/std, FP32 NCHW. The alternate `--transform mnn-224` reproduces the direct bilinear resize used by the **older** MNN experiment. Use the same transform for both sides of a performance or accuracy comparison.

### Measure the 21M baseline and hybrid gain

Put the same saved natural images in a directory on the board. These commands replay them in a persistent session with three warmups per mode. Two passes reverse mode order. They write one record per measured image, including tensor hash, model hash, thread count, runtime counters and runner peak RSS. Execute comparable runs with the same clock, cooling, image set, process privileges and build. Keep camera and benchmark processes separate.

```sh
python3 benchmark21.py saved-images baseline-t1.csv --modes cpu-fp32 cpu-mixed --threads 1
sudo python3 benchmark21.py saved-images comparison-t2.csv --threads 2
```

Compute paired per-image differences and report median/p95 and sample count. Report FP32 → mixed CPU, mixed CPU → unfused, and unfused → resident separately. For live throughput use completed predictions divided by elapsed capture time, with the first request reported separately. The live summary computes an interval rate between completed events. The latest-frame policy deliberately skips stale frames.

- `infer_ms` measures the complete synchronous graph call, including any in-graph DMA/control. Model loading, session setup, frame decode, preprocessing, input copy and output publication have separate boundaries.
- `received_to_result_ms` starts when this program receives a complete JPEG and ends at returned logits. Exposure, camera/driver buffering and remote rendering are outside that interval. `request_ms` starts when the frame is selected.
- `runner_peak_kib` and `app_peak_kib` are separate Linux process peaks. The FFmpeg process, reserved DDR and FPGA BRAM have different accounting. Use `--profile` in separate runs for the driver phase JSON, including transfer, completion polling and local feedback.
- Test terminal-only first, then browser preview if desired. The OS of an SSH client changes neither the executable nor its execution site. Moving preprocessing onto HPS removes the per-frame network round trip but consumes HPS CPU cycles. Compare completion rate and frame age on the board before claiming a live speedup.

## Build and reproduce

- **ARM application.** With Git, make, CMake and Zig 0.13.0 installed, set `ZIG_EXE=/absolute/path/to/zig` in the environment and run `sh scripts/build_runner.sh`. It pins Alibaba MNN, applies both team patches, enables Release, the MNN thread pool and Cortex-A9/NEON, and builds the stream runner and hardware smoke tool. A model reload is unnecessary between frames.
- **Models.** Use the [21M 224 checkpoint](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth) and [Imagenette2-160](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz). The source checkpoint hash is in the manifest. In a build environment with PyTorch 2.8.0, torchvision 0.23.0, timm 1.0.19, ONNX 1.16.2, MNN 3.6.2, Pillow, NumPy and a C++17 compiler, run `sh scripts/rebuild_models.sh checkpoint.pth imagenette2-160.tgz /path/to/pinned/MNN new-export`. This builds and evaluates into a new directory. A new export may have a different model UUID or calibration bytes; verify its logits and hash before deployment.
- **Hardware.** Run `mkdir -p build`, then `python3 mnn-fpga/platform/prepare_matmul_soc_native.py --friend-root hardware/soc-ni82-base --output build/ni82`. Use the upstream Quartus workflow in `mnn-fpga/platform/build_matmul_soc.py` and the NI82 qualification record. Keep the current matmul-soc main branch and standalone nonlinearity branches as references. Regenerating project sources and obtaining a timing-qualified programmed board are separate steps.

## HPS accuracy comparison on 200 images

The physical DE1-SoC HPS run produced the following results on 200 labeled images.

| FP32 MNN model | Top-1 correct | Top-5 correct |
|---|---:|---:|
| TinyViT-5M | 158/200 (79.0%) | 195/200 (97.5%) |
| TinyViT-21M | 167/200 (83.5%) | 199/200 (99.5%) |

TinyViT-21M gained 4.5 percentage points in top-1 accuracy and 2.0 points in top-5 accuracy. These counts come from the board operator’s terminal output and match the earlier host CPU totals. The [evaluation manifest](demo/accuracy200/manifest.json) identifies the images and models.

Run `python3 compare_accuracy.py ~/accuracy-200.csv` from the deployed demo folder. Use a new output filename. The two-thread CPU runner scores the original ImageNet 1,000 classes. It loads one FP32 MNN model at a time and checks for zero FPGA jobs. Both models receive identical normalized tensors. The script prints top-1 and top-5 and writes paired rankings.

The bundled original JPEGs are a deterministic sample (seed 496200) of 20 validation images per class from [Imagenette2-160](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz). Report results as balanced Imagenette subset accuracy. The manifest identifies each archive member, ImageNet label and SHA-256. The 5M FP32 export uses the [official distilled 224 checkpoint](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth), static ONNX opset 17 and MNN 3.6.2. The existing 21M FP32 model is used unchanged.

HPS Linux captures USB camera frames through V4L2, decodes MJPEG to RGB, and prints predictions in the board terminal viewed through PuTTY on Windows. Windows preprocessing appears in the earlier demo recording. For the accuracy test, the board reads saved images and applies the same preprocessing as the live demo: bicubic resize of the shorter side to 256, preserving aspect ratio, followed by a 224 × 224 center crop. It scales RGB to [0, 1], subtracts channel means (0.485, 0.456, 0.406), divides by standard deviations (0.229, 0.224, 0.225), and supplies channel-first FP32 tensors.
