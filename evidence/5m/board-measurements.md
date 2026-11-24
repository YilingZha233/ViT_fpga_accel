# TinyViT‑5M live camera measurements on the DE1‑SoC

The 90 MHz DMC1 FPGA configuration reduced mean live camera latency from **1,435.7 ms to 1,003.8 ms** versus FP32 CPU in the repeated comparison. That is a **1.43× speedup** and a **30.1% latency reduction**. MNN inference alone improved from 1,264.9 ms to 832.0 ms, a **1.52× speedup**.

## Repeated three-mode comparison

| TinyViT‑5M mode | Mean JPEG receipt to logits | Mean MNN inference | FPGA jobs / operations / resident pairs per frame |
|---|---:|---:|---:|
| Original FP32 CPU | 1,435.7 ms | 1,264.9 ms | 0 / 0 / 0 |
| Mixed CPU | 1,367.2 ms | 1,190.7 ms | 0 / 0 / 0 |
| DMC1 FPGA offload | **1,003.8 ms** | **832.0 ms** | 400 / 39 / 4 |

Each mode used the same ARM runner, two HPS threads, and a 640×480 MJPEG feed requested at 30 fps. The run order was FP32, mixed, DMC1, DMC1, mixed, FP32. Each run classified 11 frames. Dropping the first frame of each run left 20 samples per mode. The modes saw different frames from the same live feed. The mixed CPU and DMC1 paths used the same mixed checkpoint.

Camera latency starts when the HPS receives a complete JPEG and ends when the application receives the logits. It includes JPEG decoding, preprocessing and runner IPC. Sensor exposure, camera buffering and display fall outside this timing window. MNN inference is the model forward pass. The six raw CSVs and run logs are on the board in `~/tinyvit5m-camera/dmc1-repeat-er4XDV/`.

## Earlier board measurements

| Board run | Mode | Mean JPEG receipt to logits | Mean MNN inference |
|---|---|---:|---:|
| Initial DMC1 comparison, 10 frames per mode | Original FP32 CPU | 1,442.9 ms | 1,279.8 ms |
| Initial DMC1 comparison, 10 frames per mode | Mixed CPU | 1,344.1 ms | 1,179.5 ms |
| Initial DMC1 comparison, 10 frames per mode | DMC1 FPGA offload | 1,022.5 ms | 845.7 ms |
| Earlier NI82 comparison, 20 frames per mode | Bundled mixed CPU | 1,313.0 ms | 1,140.3 ms |
| Earlier NI82 comparison, 20 frames per mode | NI82 FPGA offload | 1,253.5 ms | 1,079.4 ms |

The initial DMC1 run measured **1.51× MNN inference speedup** and **1.41× camera speedup** versus original FP32. Its CSVs are in `~/tinyvit5m-camera/dmc1-latency-PxsvmJ/`. The earlier NI82 run used the bundled mixed checkpoint and reported 437 FPGA jobs, 30 offloaded operations and four resident pairs. It reduced camera latency by 4.5% versus its mixed CPU baseline. Its CSVs are in `~/tinyvit5m-camera/latency-U6LsaN/`.

In the separate upstream benchmark, warmed MNN forwards on the same 20 saved inputs averaged 1,207.1 ms on original FP32 CPU and 757.0 ms on DMC1, a **1.595×** speedup. [Upstream co-design benchmark](https://github.com/PlutoPuppy/mnn-fpga/blob/a75a4b2342079a2195761a5ffc8b22a63494e9f3/docs/tinyvit-codesign-optimization.md)

## Recorded demos

The clips illustrate live predictions and hardware counters. The latency means above come from the board CSVs.

- [FP32 CPU demo](videos/cpu-fp32.mp4)
- [Mixed CPU baseline demo](videos/cpu-mixed.mp4)
- [DMC1 FPGA offload demo with the initial 1.51× MNN result](videos/fpga-dmc1-1.51x.mp4)
- [CPU FP32 v2 recording](videos/cpu-fp32-v2.mp4)
- [DMC1 FPGA offload v2 recording](videos/fpga-dmc1-1.5x-v2.mp4)

## Run the live demos

Open PuTTY on Windows, select **Serial**, set **COM3** and **115200 baud**, and log in to the DE1‑SoC as `ubuntu` if prompted. Connect the USB camera and run one demo at a time from the board shell. Use **Ctrl+C** to stop a continuous demo, or append `--max-frames 5` for five classifications.

```sh
cd ~/tinyvit5m-camera
```

### Original FP32 CPU

```sh
python3 hps_local_demo.py --candidate 5m --mode cpu-fp32 \
  --model models/tiny-vit.mnn \
  --runner "$PWD/bin/tinyvit_stream_codesign" --device /dev/video0
```

### Mixed CPU

```sh
python3 hps_local_demo.py --candidate 5m --mode cpu-mixed \
  --model models/tiny-vit-mixed-w8a8-accuracy.mnn \
  --runner "$PWD/bin/tinyvit_stream_codesign" --device /dev/video0
```

Both CPU modes print `jobs=0 layers=0 pairs=0`.

### DMC1 FPGA offload

The board must have the 90 MHz DMC1 image (`matmul-soc-native-dmc1-90mhz.sof`) loaded. It is in the qualified deployment bundle. If the board rebooted normally, reload DMC1 during the temporary U-Boot/JTAG sequence before running the FPGA command. [Deployment setup](../../README.md#tinyvit-5m-co-design-camera-mode) · [Temporary U-Boot/JTAG procedure](../../mnn-fpga/docs/matmul-soc-integration.md#tested-boot-sequence-on-this-board)

```sh
sudo python3 hps_local_demo.py --candidate 5m --mode hybrid-codesign \
  --model models/tiny-vit-mixed-w8a8-accuracy.mnn \
  --projection-ranges models/projection-nine.tsv \
  --runner "$PWD/bin/tinyvit_stream_codesign" --device /dev/video0
```

The FPGA run prints `jobs=400 layers=39 pairs=4`. To collect comparable logs, append `--size 640x480 --fps 30 --threads 2 --max-frames 11 --timings NEW_FILE.csv` to each command, using a fresh filename for every run.
