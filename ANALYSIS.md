# TinyViT capstone integration review · 21 November 2026

The [physical board measurements](evidence/21m/board-measurements.md) supersede the original pre-deployment status below. **The demo is TinyViT-21M on MNN.** The team has physical DE1-SoC results for the 21M CPU model on an earlier runtime and separate physical MNN/NI82 results for a smaller graph. This integration ports the original 21M checkpoint to MNN and connects a selected 21M MLP to the existing driver.

## Which TinyViT?

| Property | Active 21M demo | Previous MNN hardware experiment |
|---|---|---|
| Graph identity | Official 224×224 distillation checkpoint. SHA256 matches the November 21M baseline source. 21,198,568 trainable parameters before export. | Original MNN SHA256 starts f7aee628. Stage widths match the official 5M configuration. Exact weight-training provenance is not established by shape alone. |
| Stage embedding widths | 96, 192, 384, 576 | 64, 128, 160, 320 |
| First transformer MLP | 784 × 192 → 768 → 192 | 784 × 128 → 512 → 128 |
| Full classifier | Batch 1, RGB 224×224, 1,000 ImageNet logits | Same input/output counts; different internal work |
| Default image geometry | Bicubic short edge 256, center crop 224 | Prior MNN holdout used bilinear resize directly to 224 |
| Integrated hardware target | Stage 1 block 1 FC1 and FC2, with optional resident activation | 30 static pointwise linears and 4 resident pairs |

The 21M graph starts with a convolutional patch embedding and a high-resolution MBConv stage. Downsampling produces 784, 196 and 49 tokens in its three transformer stages. Each window-attention block includes normalization, static QKV projection, dynamic attention scores QKᵀ, softmax, multiplication by V, output projection, local convolution and an MLP with FC1, GELU and FC2, with residual connections. The classification head pools features and emits 1,000 scores. MNN can represent a token-wise linear projection as a 1×1 convolution. Offloading QKV projection and offloading QKᵀ are different graph changes. [Official architecture](https://github.com/microsoft/Cream/blob/4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9/TinyViT/models/tiny_vit.py) · [MNN operator profile](mnn-fpga/docs/tinyvit-cpu-operator-profile.md)

## What each contribution supplies

| Contributor / workstream | Implemented and evidenced | Role in this demo |
|---|---|---|
| Tiffany / Yiling | HPS CPU/kernel baselines, labeled 21M evaluation, UVC capture and an earlier live classifier, HPS-local decode/preprocessing code, labels and fixed tensor fixtures. The published video used Windows preprocessing. | HPS-local capture, preprocessing, latest-frame selection and display now feed persistent MNN sessions. |
| Jerry / Chenjia | INT8 weight-stationary systolic array, FPGA operand/output banks, tiling, CSR/control, board-level SoC integration and throughput tests. Current SoC main also includes partial-sum and bias/GELU work. | NI82 is generated from the specifically qualified older SoC revision with Stephanie's overlay. Later main-branch features do not automatically appear in this bitstream. |
| Stephanie / Yixin | Full-graph MNN runtime, named-operation dispatch, real DMA, operator profiling, quantization repair, native INT8 hooks, full-reduction arithmetic, persistent packed weights and resident MLP scheduling; physically tested on the prior graph. | Reuse the runtime and NI82 contract. This integration supplies new 21M model exports, two-layer policy, counters and a live multi-image stream. |
| Winnie / Weini and nonlinear workstream | NEON comparison, bias/GELU interface, standalone fixed-point softmax and LayerNorm, reference vectors, simulation and standalone timing. | Softmax and LayerNorm remain on HPS. The 21M resident MLP uses a table generated from the selected model's activation and scales. |

The wiki summary and updates are dated snapshots. Its resources tab points to the TinyViT architecture paper, quantization and accelerator literature, project files and prior profiling. Study results from other accelerators do not establish a speedup for this board. [Summary](https://vit-fpga-wiki.pages.dev/#/summary) · [Updates](https://vit-fpga-wiki.pages.dev/#/updates) · [Resources](https://vit-fpga-wiki.pages.dev/#/resources) · [Pinned source inventory](SOURCES.md)

## How the hardware work evolved

| Experiment | Scope and result | Interpretation |
|---|---|---|
| Standalone square INT8 matmul | At N=64/128/256/512, board FPGA 6.043/11.392/18.302/18.884 GOP/s vs two-thread tiled NEON 4.989/6.083/6.654/6.950. Speedup 1.21/1.87/2.75/2.72×. | Operation count is 2N³. Tile occupancy and DMA matter. In a separate 75 MHz tile sweep, 128³ with bias disabled achieved 28.78 compute GOP/s and 9.71 GOP/s including I/O. |
| 80 MHz selected dynamic QK, Nov 3 | One stage-3 attention MatMul, 320 padded PIO jobs. Single zero example improved 5.74%; nonuniform example worsened 2.71%. | Initial physical integration. Two samples are insufficient for a model speedup. |
| 90 MHz first-stage QK, repeated | 20 random images. MNN CPU 1178.207 ms, DMA offload 1224.513 ms. Selected QK 4.822 vs 59.108 ms. Start-to-done including polling was 0.657 ms. | Conversion, layout and transfers overwhelmed compute. Dynamic operands offer little weight reuse. |
| One static FC1 with legacy hardware | Capacity-aware tiling reduced mean inference by 19.605 ms vs fixed K=64, but added 31.305 ms vs CPU, with uncertainty across 20 images. Corrected W8A8 model with only FC1 offloaded still took 1123.207 vs 1092.412 ms on its CPU baseline. | Native INT8 removes driver-side float conversions. One layer still incurs overhead; per-band saturation and rounding also distorted results. |
| NI82 multiple static linears | In the prior model, FPGA took 9 FC1, 4 FC2, 8 QKV and 9 output projections. Seven quantized depthwise convolutions, dynamic QK/PV and other FP32 work remained on HPS. | NI82 accumulates across the full K in INT32 and requantizes once. Per-channel coefficients and bias, prepacked static weights, capacity-aware tiles and DMA replace per-band reconstruction. |
| NI82 resident MLP | Four FC1–activation–FC2 chains retained hidden activations in FPGA RAM in the prior model. The selected 22-layer alternative measured 1014.593 ms in its own random-image series; all 30 performed better on the matched natural-image comparison. | Fusion saves intermediate transfer and CPU layout/activation work. CPU and FPGA graph execution were synchronous. |

Evidence: [tile sweep](evidence/wiki/matmul-tile-sweep-75mhz-2026-11-16.csv) · [QK timing](mnn-fpga/docs/tinyvit-qk-dma-timing-diagnosis.md) · [FC1 tiling](mnn-fpga/docs/tinyvit-fc1-capacity-benchmark.md) · [mixed FC1](mnn-fpga/docs/tinyvit-mixed-w8a8-fc1-benchmark.md) · [NI82 native results](mnn-fpga/docs/tinyvit-native-resident-offload.md)

## Prior physical MNN measurements

Same 20 random RGB inputs, two Cortex-A9 threads, 90 MHz NI82, one inference per fresh process, detailed timing disabled. The graph timer excludes model load, camera and preprocessing. The last column is calculated as 1,000 divided by mean milliseconds for sequential batch-one inference.

| Previous graph and mode | Mean inference, ms | Mean process wall, ms | Jobs/image | Inference-equivalent images/s |
|---|---:|---:|---:|---:|
| FP32 CPU | 1155.982 | 1624.037 | 0 | 0.865 |
| Corrected mixed CPU | 1080.605 | 1508.985 | 0 | 0.925 |
| 30 separate FPGA linears | 1048.245 | 1560.853 | 434 | 0.954 |
| 30 linears with 4 resident pairs | 1006.955 | 1522.157 | 437 | 0.993 |

The earlier graph saved 75.377 ms with CPU mixed precision alone, then 32.360 ms with separate FPGA linears, then about 41.29 ms with fusion. Resident execution saved 73.650 ms vs mixed CPU (6.82%, 1.073×) and 149.027 ms vs FP32 (12.89%, 1.148×). The confidence interval for the paired saving vs mixed CPU was 69.093–78.179 ms. A fresh resident process on the same random experiment took 13.172 ms longer than mixed CPU. Keeping sessions and weight caches alive matters for an interactive demo. The profiled 30 CPU linears took 234.286 ms vs 35.556 ms for FPGA launch/completion polling and 220.130 ms for the complete resident adapter. Pure hardware cycles were not independently counted. [Team's full experiment](mnn-fpga/docs/tinyvit-native-resident-offload.md)

## Quantization, accuracy and memory

| Dataset and implementation | Measured comparison | What the result supports |
|---|---|---|
| Historical 21M CPU, 10-image software arithmetic pilot | FP32 and Q1/Q2/Q3 each got 10/10 top-1. Q1 uses per-tensor W/A; Q2 adds per-channel weight scales; Q3 uses full-sum fixed-point floor requantization. Q4 rounds every 16-term band and merges with saturation: 0/10. | Rounding position matters. None of these simulations measures board INT8 speed. |
| Historical 21M CPU, earlier runtime, 1,000 labeled ImageNet-mini images | FP32 top-1/top-5 84.9/97.0%; native INT8 83.6/96.6%. Across batch processes, peak RSS reached 149.48 vs 72.96 MiB. In separate timed pilots FP32 1T/2T took 7640.891/4480.743 ms; INT8 1T/2T took 5072.312/3198.891 ms. | Historical CPU context for 21M. The audited dataset contains 1,000 unique images, 676 labels; wiki wording about 2,000 should not double the sample count. |
| Prior smaller MNN graph, 200-image Imagenette holdout, host CPU | FP32 149/200 (74.5%), blanket W8A8 21/200 (10.5%), selected mixed W8A8 148/200 (74.0%). Mixed agreed with FP32 top-1 on 193/200. Weight-only INT8 with FP32 activations agreed on 195/200. | Blanket activation quantization damaged sensitive layers. The corrected policy retained 37 W8A8 and 31 FP32 convolutions plus floating-point nonlinearities. |
| Prior smaller graph, 20 labeled natural images, physical DE1-SoC | FP32, mixed CPU, all-resident and selected-resident each got 16/20 correct with identical top-1 decisions. All-resident took 1011.565 ms vs mixed CPU 1090.452 ms. | Integrated hardware check on a small, reused Imagenette set. |
| **Active 21M MNN graph, 40 disjoint Imagenette validation images, host CPU** | FP32 and two-layer W8A8 both got **37/40 top-1, 40/40 top-5**. They agreed on all 40 top-1 predictions; mean logit cosine 0.999955. FP32 MNN matched PyTorch top-1 on all 40; largest absolute logit difference 0.02780. | Reproducible 21M conversion/quantization check with 1,000 output classes. The 40 validation examples span 10 classes, four each. Physical hybrid accuracy and full ImageNet accuracy remain unmeasured. |

For the new 21M export, 128 balanced Imagenette training images supplied KL activation calibration. Weights use per-output-channel symmetric INT8 and maximum absolute range. Only stage 1 block 1 FC1/FC2 use W8A8; FP32 boundaries surround them in the mixed CPU graph. Quantizing this one pair reduces the MNN file from **96,343,288 to 95,459,192 bytes** (884,096 bytes, 0.92%). Linux peak RSS depends on the runtime/session, activations, cached weights and camera process. The board CSV records RSS for the new graph. [21M host validation](evidence/21m/host-validation.json) · [21M calibration/validation manifest](evidence/21m/imagenette-manifest.json) · [historical 21M audit](evidence/historical/hps_accuracy1000_2026-11-08.md) · [previous quantization repair](mnn-fpga/docs/tinyvit-quantization-accuracy-repair.md)

For the smaller MNN graph, the reported model shrank from 26,602,144 to 18,748,848 bytes (29.52%). Four resident pairs reduced host upload+output traffic from 11,131,136 to 10,515,200 bytes/image, saving 615,936 bytes (5.53%); local feedback increased to 1,200,128 bytes. The qualified NI82 system uses 22,686/32,070 ALMs, 367/397 M10Ks (92%) and 35/87 DSPs. The A/B/output banks are 16/64/64 KiB. HPS DDR contains the packed weights. These figures distinguish file size, working memory, traffic per image and physical FPGA capacity. [Hardware, memory and traffic measurements](mnn-fpga/docs/tinyvit-native-resident-offload.md)

## What the 21M partition changes

For 784 tokens the selected pair computes **192 → 768 → 192**. At a 64 KiB B bank, FC1 uses 80-row blocks and up to 128 output channels. FC2 uses 16-row blocks and up to 80 output channels. Across the image, FC1 requires 60 jobs and FC2 requires 147. Both schedules submit 207 jobs. In resident mode a 256-entry model-derived table combines GELU and conversion to the next layer's input scale. The hidden activation stays in FPGA RAM until consumed by FC2. The unfused mode returns FC1 to HPS first. These modes enable a direct measurement of the effect of fusion at fixed model precision.

The NI82 driver contract limits one complete reduction to K ≤ 1024. Later 21M FC2 layers have K=1536 or 2304. The older model's 30-layer policy and four-pair count cannot be applied unchanged. Softmax, LayerNorm, QKV, QKᵀ, P×V, embedding, residual and classifier execute on HPS in the new candidate. The hardware softmax research block passed 578 rows / 25,999 RTL outputs at 90 MHz standalone (175 ALMs, 3 M10Ks, 0 DSPs); standalone LayerNorm passed 744 rows / 200,296 outputs (510 ALMs, 4 M10Ks, 8 DSPs). The reported 2.532% attention-context relative RMSE and 3.485% LayerNorm relative RMSE are local numerical metrics. [Weekly summary](evidence/wiki/live-summary.json) · [Softmax sources](hardware/standalone/softmax) · [LayerNorm sources](hardware/standalone/layernorm)

## What can go on the demo poster

The [physical board measurements](evidence/21m/board-measurements.md) document the deployed 21M results. Completed-frame throughput and camera-to-display latency still need separate measurements.

The full 21M software path was also exercised in emulated ARM Linux. Both unfused and resident modes ran through the real MNN callbacks with a separate NI82 software-device build. Each submitted 207 jobs per image. The resident case skipped four intermediate operations and retained one MLP intermediate; the two distinct inputs matched PyTorch's top-1. [21M validation record](VALIDATION.md)
