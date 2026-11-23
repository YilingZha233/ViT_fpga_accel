# TinyViT-21M · DE1-SoC camera demo

The active demo runs **TinyViT-21M, 224 × 224, 1,000 classes, using MNN**. Both supplied models come from the same 21,198,568-parameter PyTorch checkpoint as the team's earlier 21M evaluation. The default is a persistent, two-thread FP32 CPU session. The hybrid candidate moves one MLP block to the team's NI82 design. The earlier 30-layer board result belongs to a smaller model. See [analysis and measurements](ANALYSIS.md), [verification status](VALIDATION.md) and [sources](SOURCES.md).

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

The smoke executable includes an integer reference check for the 21M workload, 784 × 192 → 768 → 192. The model and CPU demo assets are included. The physical HPS camera and physical 21M FPGA run remain to be measured.

### What the audience sees

By default the board prints the frame number, predicted ImageNet class, inference time, preprocessing time and hardware counters. Add `--preview` to the live command to publish a read-only view at `http://BOARD_IP:8080`. The crop and prediction in the result have the same frame ID; the independent live preview may be newer. Capture, JPEG decode, resize, crop, normalization, inference dispatch and label selection all run on the board. Closing the browser leaves classification running. Use a persistent board shell such as tmux if the SSH client will disconnect.

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

Run `python3 compare_accuracy.py ~/accuracy-200.csv` from the deployed demo folder. Use a new output filename. The two-thread CPU runner scores the original ImageNet 1,000 classes. It loads one FP32 MNN model at a time and checks for zero FPGA jobs. Both models receive RGB bicubic short-side resize to 256, a 224-pixel center crop and identical normalized tensors. The script prints top-1 and top-5 and writes paired rankings.

The bundled original JPEGs are a deterministic sample (seed 496200) of 20 validation images per class from [Imagenette2-160](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz). Report results as balanced Imagenette subset accuracy. The manifest identifies each archive member, ImageNet label and SHA-256. The 5M FP32 export uses the [official distilled 224 checkpoint](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth), static ONNX opset 17 and MNN 3.6.2. The existing 21M FP32 model is used unchanged.
