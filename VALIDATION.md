# Verification record · 21 November 2026

## Physical HPS accuracy comparison · 22 November 2026

The operator ran `compare_accuracy.py` on the physical DE1-SoC using two HPS threads, FP32 MNN and identical bicubic-preprocessed inputs for both models. The reported terminal counts match the host verification on all four aggregate metrics. See the [board results and evaluation procedure](README.md#hps-accuracy-comparison-on-200-images).

## New TinyViT-21M artifacts

| Check | Environment and input | Observed result |
|---|---|---|
| Source identity | Official 21M distillation checkpoint, pinned Cream code, strict state-dict load. Static ONNX opset 17, MNN 3.6.2 converter. | 21,198,568 trainable parameters. FP32 and calibrated mixed model hashes and sizes recorded in [model manifest](demo/models/manifest.json). |
| Mixed graph construction | 128 Imagenette train images, frozen bicubic short-side/center-crop transform; one stage 1 block 1 MLP selected. Team mixed-model builder validates graph structure and inserts explicit quant/dequant boundaries. | Two W8A8 linears (192→768, 768→192), four explicit conversion nodes. The rest of the model stays FP32. |
| Classifier and export checks | Apple Silicon host CPU, PyTorch 2.8.0, MNN 3.6.2, 40 Imagenette validation images, four per class, 1,000 logits per image. | FP32 MNN and mixed CPU each 37/40 top-1, 40/40 top-5. FP32/PyTorch and mixed/FP32 top-1 agreement both 40/40. [Per-image evidence](evidence/21m/host-validation.json) |
| ARM application build | Pinned MNN with both team patches, Zig 0.13.0, Cortex-A9 ARMv7 hard-float musl, Release, MNN thread pool enabled, static Linux executable. | The shipped executable ran both actual 21M models in an emulated ARM Linux guest. The old prebuilt ARM executable was excluded. |
| Persistent service | QEMU virt, two emulated Cortex-A15 cores running Cortex-A9-targeted code. Real normalized tensors from tench and parachute validation images, order A, B, A, 1,000 real model logits on each request. | CPU FP32 and CPU mixed both returned the corresponding PyTorch class each time. The final logits for A matched its first response byte-for-byte within each mode. |
| Full hybrid software path | Same emulated ARM guest, same trained mixed model and callbacks. A separate test-only binary substitutes the team's NI82 register/RAM software model for the physical device. Both individual and resident dispatch were exercised on A, B, A. | Both modes matched the PyTorch class on all three requests, with 207 jobs and two replaced linears per image. Resident mode completed one pair, skipped four intermediate ops and ended with pending=-1. Repeated A was byte-identical within each mode. Measured model-device traffic was 1,806,336 local feedback bytes and 1,204,224 logical hidden round-trip bytes avoided per resident image. These are software-model counters. [Emulation record](evidence/21m/arm-verification.json) |
| Camera preprocessing and frontend | Pillow 9.4.0 / NumPy 1.24.2, three image/tensor fixtures. Separate launcher smoke cases exercised snapshot, CSV, timing parsing and erroneous job count with a protocol test double. | Exact tensor matches on all three fixtures. Four launcher modes completed with expected counters. A zero-job hybrid result caused failure. No webcam was attached in these checks. |
| Driver regressions | Team's host tiling, requantization, NI82 driver and layout tests, including 44 native INT8 matrix/pair cases and 72 scalar NC4 DMA layout cases. | All four host test programs passed. These compare against software references. |
| Quartus project generation | Team NI82 generator applied to the pinned SoC commit. Generated top, weight control, input buffer, native output stage, Qsys and QSF sources compared to the qualification manifest. | All six matched the recorded SHA256 with the corresponding LF/CRLF convention. Quartus fitting and FPGA programming were not run in this integration. |

The production binary uses the physical device path for hybrid mode. The test double and the NI82 software-model binary are excluded from the deployment kit. The raw emulator log includes timings for debugging; do not report its timings or memory numbers as DE1-SoC measurements.

## Physical acceptance still required

1. On the actual HPS image, verify the supplied preprocessing fixtures and run the 21M FP32 and mixed CPU snapshots. Compare the saved logits for the same input and record dual-core settings, MNN build, hashes and clock.
2. Load the qualified NI82 image using the team's bridge and memory-reservation procedure. Run the new 21M shape smoke check, then exercise unfused and resident inference. Verify per-frame 207/2/0 and 207/2/1 hardware counters and compare full logits and labeled predictions. The stock SD image can replace the temporary FPGA image after reboot.
3. Run saved-frame paired comparisons and then live HPS-only and hybrid camera tests. Report model and application latency distributions, completed frames/s and separate memory measurements. A physical end-to-end result for this 21M demo has not yet been recorded.
