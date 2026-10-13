# LARE: Latency-Aware Ranking for Early-Exit Architecture Search on RISC-V Accelerator

LARE is an open-source framework for latency-aware neural architecture search (NAS) and deployment
of quantized early-exit (EE) CNNs on
[ZTA-Q](https://github.com/kevinli-hw/ZTA-Q/tree/feature/ZTA-Q), a RISC-V-based SIMD accelerator
built on [ztachip](https://github.com/ztachip/ztachip). LARE transforms TFLite models into
sequential execution stages with runtime exit decisions. It jointly optimizes exit placement,
confidence thresholds and exit-head architectures, and ranks the candidates by latency profiled on
the FPGA.

LARE was evaluated with five CNNs on a Digilent Arty A7-100T FPGA at 83.3 MHz. With an accuracy
target of 1 percentage point (pp) below the backbone on the calibration set, it reaches up to
**5.94×** end-to-end speedup over the static CNN baseline on CIFAR-10 and up to **3.03×** on
CIFAR-100. The largest test-accuracy loss is 2.19 pp.

## Highlights

- **Latency-aware NAS.** The search jointly optimizes exit placement, confidence thresholds and
  classifier-head architectures, and ranks the candidates by measured execution time instead of MAC
  counts. A profiling-based model estimates the latency of candidate exit heads, so the search needs
  no exhaustive board measurements.
- **End-to-end deployment flow.** Quantized EE TFLite models are transformed into sequential
  execution stages on ZTA-Q. The SIMD NPU runs the operators and the RISC-V host takes the exit
  decisions at runtime.
- **Lightweight host-side processing.** Integer-only preprocessing and softmax-free exit evaluation
  cut the host-side overhead, with negligible accuracy loss.

## Latency-aware neural architecture search

### MAC count vs. measured latency

Early-exit branches were built for five CIFAR-10 backbones (SqueezeNet, MobileNetV1/V2 and
ResNet-18/50), with MAC-balanced exits at about 33% and 67% of the accumulated MACs. Each exit head
consists of global average pooling and two fully connected layers, trained with the backbone frozen.
The training set is split into 5,000 class-balanced calibration images (500 per class) and 45,000
training images.

After quantization, the unconditional exit rates $`\alpha_1`$ and $`\alpha_2`$ are taken from the
calibration set. A configuration $`C`$ has backbone stages $`S_i`$ and exit heads $`E_i`$, and its
expected MAC cost is Eq. (1):

```math
M(C) = M(S_1) + M(E_1) + (1-\alpha_1)\big(M(S_2) + M(E_2)\big) + (1-\alpha_1-\alpha_2)\,M(S_3)
```

On ZTA-Q the MAC count does not track latency:

- MAC-free operators, such as max pooling in SqueezeNet, add noticeable latency.
- The later pointwise convolutions in MobileNetV2 are costly, although the accumulated MAC curve
  grows smoothly.
- The latency per MAC varies across convolution types (3×3, pointwise and depthwise) and output
  feature-map resolutions. It reaches 45× that of the most efficient operator group in the same
  backbone.

### Profiling-based latency model

LARE replaces $`M(S_i)`$ and $`M(E_i)`$ with the latencies $`T(S_i)`$ and $`T(E_i)`$, which gives
Eq. (2). $`T(S_i)`$ is the sum of the measured operator times within the segment.

```math
T(C) = T(S_1) + T(E_1) + (1-\alpha_1)\big(T(S_2) + T(E_2)\big) + (1-\alpha_1-\alpha_2)\,T(S_3)
```

Grouping and tiling stay the same for a given operator type and output resolution. The latency of
an exit head of type $`h`$ attached at a place of resolution $`R`$ therefore grows linearly with its
MACs, as in Eq. (3):

```math
T(E) = T(op) + T(dec) = \big[A_{h,R}\,M(\mathrm{head}) + K_{h,R}\big] + T(dec)
```

- $`A_{h,R}`$ is the effective execution time per MAC.
- $`K_{h,R}`$ is the time that does not scale with the input width, such as the head's fixed-width
  GAP and FC layers.
- $`T(dec)`$ is the measured host-side exit-decision latency.

For each of the four head types, $`A_{h,R}`$ and $`K_{h,R}`$ are fitted by least squares on heads
measured on the board:

- resolutions $`R \in \{32, 16, 8, 4, 2\}`$;
- two to three input widths between 64 and 1024 per resolution;
- representative heads placed on the backbone models, averaged over five runs.

The model evaluates candidate exit placements and head architectures without exhaustive FPGA
measurements. The [FPGA evaluation](#fpga-evaluation) compares its predictions with the board.

### Joint search with latency-aware ranking

- **Backbones:** SqueezeNet, MobileNetV1, MobileNetV2, ResNet-18 and ResNet-50, trained on CIFAR-10
  and CIFAR-100.
- **Calibration set:** 500 samples per class for CIFAR-10 and 50 per class for CIFAR-100.
- **Candidate exit positions:** block boundaries. They keep the search space small and still give
  meaningful intermediate features.
- **Exit heads**, each trained with its backbone frozen:
  - **Lightweight**: global average pooling (GAP) and two fully connected (FC) layers.
  - **Heavy_CV**: a standard 3×3 convolution before the lightweight head.
  - **Heavy_PW**: a 1×1 pointwise convolution in place of the standard one.
  - **Heavy_DW**: a 3×3 depthwise convolution followed by a 1×1 pointwise convolution.

The search takes an accuracy target $`A_{tgt}`$ and uses $`T(C)`$ as the ranking function. The exit
score $`S_{h,p}(x)`$ is the maximum softmax probability of head $`h`$ at place $`p`$. All heads share
one threshold grid. The grid is refined above 0.99 because the confidences of easy samples
concentrate there:

```math
\mathcal{T} = \{0.01, 0.02, \dots, 0.99\} \cup \{0.991, 0.992, \dots, 0.999\} \cup \{+\infty\}
```

```text
C* ← none;  T* ← +∞
for each p1 < p2 in P,  h1, h2 in H,  τ1, τ2 in the threshold grid:
    E1 ← {x ∈ D : S(h1,p1)(x) ≥ τ1}            # samples leaving at exit 1
    E2 ← {x ∈ D \ E1 : S(h2,p2)(x) ≥ τ2}       # samples leaving at exit 2
    E0 ← D \ (E1 ∪ E2)                         # samples reaching the main exit
    α1 ← |E1| / |D|;  α2 ← |E2| / |D|
    A  ← fraction of D classified correctly by the exit each sample leaves from
    C  ← (p1, h1, τ1, p2, h2, τ2)
    if A ≥ A_tgt and T(C) < T*:
        C* ← C;  T* ← T(C)
return C*
```

The configurations that meet the accuracy target are ranked by their estimated latency, and the
top-ranked one is selected.

## End-to-end early-exit deployment flow

### Operator list reordering

The operator list exported by TFLite places the intermediate exit heads and the final classifier
after the backbone operators. That order preserves the data dependencies, but it prevents early
termination during sequential execution on ZTA-Q. LARE reorders the list to interleave the exit
heads with their backbone stages. The RISC-V host CPU then dispatches the operators to the SIMD
accelerator in order, evaluates the exit conditions at the designated points, and stops the
inference once the confidence threshold is met.

### Sequential staged execution

LARE reuses the existing NPU datapath of ZTA-Q. The accelerator has a data plane for DDR
communication and processing elements (PEs) connected through a stream processor for activation
processing. The host offloads the neural-network operators to the accelerator, and the VexRiscv CPU
handles the runtime exit decisions. Cycle counters measure the accelerator execution and the
host-side decision latency, and these profiles feed the latency model of the search.

The VexRiscv CPU implements only the RV32IM ISA, without a floating-point unit. Two host-side
operations are simplified to avoid floating-point work:

- **Preprocessing:** input normalization is approximated with per-channel integer offsets.
- **Exit decision:** the softmax confidence selected by the search is replaced by a score margin
  between the two largest integer logits, with its thresholds set by calibration.

Host-side latency on the board. Preprocessing is per image; the decision is per evaluated exit.

| Host-side operation       |   Float | Simplified | Speedup |
| ------------------------- | ------: | ---------: | ------: |
| Preprocessing             | 37.2 ms |    0.49 ms |     76× |
| Exit decision (CIFAR-10)  | 0.49 ms |   0.010 ms |     49× |
| Exit decision (CIFAR-100) | 3.95 ms |   0.024 ms |    163× |

## Results

### Latency-scored vs. MAC-scored search

Both searches were run under the same accuracy targets, from the backbone (main-exit) accuracy down
to 6 pp below it. For each target, each search selects its best configuration, and the latency of
that configuration on the test set is evaluated with the profiling model $`T(C)`$. For most targets,
across the five CNNs and both datasets, the latency-scored search finds the faster configuration.

The table below compares the two searches at the same test accuracy. Each entry is the relative
reduction (%) in $`T(C)`$ achieved by the latency-scored search at the target main − k pp:

| Dataset   | Model       | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
| --------- | ----------- | ----: | ----: | ----: | ----: | ----: |
| CIFAR-10  | SqueezeNet  |  +1.6 |  +1.9 |  +2.6 |  +2.9 |  +5.1 |
| CIFAR-10  | MobileNetV1 | +13.8 |  +9.1 |   0.0 |  +2.2 |  +3.1 |
| CIFAR-10  | MobileNetV2 |  +5.7 |  +0.9 |  +9.0 | +10.0 | +10.2 |
| CIFAR-10  | ResNet-18   |  −1.6 |   0.0 |  +1.1 |  +3.9 |  +7.3 |
| CIFAR-10  | ResNet-50   |   0.0 |  −1.0 |  +4.9 |  +0.7 |  +2.3 |
| CIFAR-100 | SqueezeNet  |   0.0 |  +0.1 |  +0.1 |   0.0 |  +1.6 |
| CIFAR-100 | MobileNetV1 |  +3.2 |  +2.5 |  +6.1 |  +7.6 |  +8.7 |
| CIFAR-100 | MobileNetV2 |  −2.2 |  −3.3 |  −2.4 |  +5.6 |  +4.9 |
| CIFAR-100 | ResNet-18   |  +4.8 |  +2.7 |  +3.9 |  +0.7 |  +0.6 |
| CIFAR-100 | ResNet-50   |  +0.5 |  +1.7 |   0.0 |  +5.1 |  +6.4 |

The latency-scored search keeps its advantage in most cases, with up to 13.8% lower latency. The few
negative entries come from the limited calibration set, which does not fully capture the test
distribution.

### FPGA evaluation

Setup:

- Board: Digilent Arty A7-100T at 83.3 MHz.
- Data: the full test sets.
- Accuracy target: 1 pp below the backbone accuracy, on the calibration set.

Columns:

- **Baseline**: the static CNN.
- **LARE Inference**: the configuration found by the latency-aware search, with float preprocessing
  and softmax exit decisions.
- **Predicted**: the latency predicted with $`T(C)`$, and its deviation from the measured LARE
  Inference latency.
- **LARE End-to-End**: the same configuration with integer-only preprocessing and score-margin exit
  decisions.

Each cell gives top-1 accuracy (%), mean latency (ms) and speedup over the baseline. All latencies
include the host preprocessing.

**CIFAR-10**

| Model       | Baseline       | LARE Inference        | Predicted     | LARE End-to-End       |
| ----------- | -------------- | --------------------- | ------------- | --------------------- |
| SqueezeNet  | 91.87 / 76.1   | 90.66 / 63.9 / 1.19×  | 63.1 (−1.3%)  | 90.52 / 26.2 / 2.90×  |
| MobileNetV1 | 92.67 / 98.6   | 91.55 / 54.9 / 1.80×  | 54.4 (−0.8%)  | 91.08 / 17.1 / 5.77×  |
| MobileNetV2 | 92.36 / 92.0   | 90.90 / 52.8 / 1.74×  | 52.3 (−1.0%)  | 90.72 / 15.5 / 5.94×  |
| ResNet-18   | 96.06 / 411.4  | 95.19 / 194.3 / 2.12× | 192.5 (−0.9%) | 95.19 / 160.8 / 2.56× |
| ResNet-50   | 96.31 / 1013.0 | 94.80 / 393.9 / 2.57× | 393.3 (−0.2%) | 94.17 / 351.2 / 2.88× |

**CIFAR-100**

| Model       | Baseline       | LARE Inference        | Predicted     | LARE End-to-End       |
| ----------- | -------------- | --------------------- | ------------- | --------------------- |
| SqueezeNet  | 67.38 / 77.5   | 66.63 / 75.8 / 1.02×  | 75.2 (−0.8%)  | 65.93 / 32.1 / 2.41×  |
| MobileNetV1 | 70.88 / 99.0   | 69.74 / 78.6 / 1.26×  | 77.9 (−0.9%)  | 69.68 / 35.6 / 2.78×  |
| MobileNetV2 | 68.89 / 92.9   | 67.08 / 74.1 / 1.25×  | 74.3 (+0.3%)  | 67.07 / 30.7 / 3.03×  |
| ResNet-18   | 78.22 / 412.0  | 76.27 / 304.9 / 1.35× | 303.1 (−0.6%) | 76.03 / 264.7 / 1.56× |
| ResNet-50   | 79.66 / 1013.5 | 78.18 / 776.3 / 1.31× | 776.8 (+0.1%) | 78.18 / 735.4 / 1.38× |

- The searched configurations reach up to 2.57× (CIFAR-10) and 1.35× (CIFAR-100) speedups over the
  baselines. The host-side simplifications raise them to 5.94× and 3.03×.
- The host-side simplifications matter most for SqueezeNet, MobileNetV1 and MobileNetV2, whose
  inference time is comparable to the float preprocessing.
- The smaller gains on CIFAR-100 suggest that the harder classification task leaves fewer early-exit
  opportunities.
- The latency model's average absolute prediction error over the ten configurations is about 0.7%,
  and the maximum is 1.3%.
- The accuracy target is set on the limited calibration set, so the test-accuracy loss can exceed
  1 pp. The largest loss is 2.19 pp (ResNet-18 on CIFAR-100, end-to-end).

## Platform setup

LARE runs on the ZTA-Q platform. The steps below build it for the Arty A7-100T.

### 1. FPGA bitstream

See [Documentation/Vivado.md](Documentation/Vivado.md). To meet timing at 83.3 MHz, set `CLK5` to
83.33 MHz and `CLK6` to 166.66 MHz in `HW/examples/GHRD/create_project.tcl` (the default is 125 MHz):

```
CONFIG.CLKOUT5_REQUESTED_OUT_FREQ {83.33} \
CONFIG.CLKOUT6_USED {true} \
CONFIG.CLKOUT6_REQUESTED_OUT_FREQ {166.66} \
```

### 2. Firmware

```bash
export PATH=/opt/riscv/bin:$PATH
cd SW/compiler
make clean all                       # build the DSL compiler
cd ../fs
python3 bin2c.py                     # embed models/inputs into the firmware filesystem
cd ..
make clean all -f makefile.kernels   # build the computation kernels
make clean all                       # produces SW/build/ztachip.elf
```

### 3. JTAG (OpenOCD)

```bash
cd <openocd_riscv install folder>
sudo src/openocd -f usb_connect.cfg -c 'set MURAX_CPU0_YAML cpu0.yaml' -f soc_init.cfg
```

The [ZTA-Q README](https://github.com/kevinli-hw/ZTA-Q/blob/feature/ZTA-Q/README.md) covers model
switching and the GDB-driven batch test scripts.

## Related

- **ZTA-Q:** an open-source RISC-V platform for accurate quantized CNN inference
  ([arXiv:2610.01867](https://arxiv.org/abs/2610.01867),
  branch [`feature/ZTA-Q`](https://github.com/kevinli-hw/ZTA-Q/tree/feature/ZTA-Q)).
- **ztachip** documentation, for the accelerator internals and the DSL:
  1. [Technical overview](Documentation/Overview.md)
  2. [Hardware Architecture](Documentation/HardwareDesign.md)
  3. [Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/ztachip_programmer_guide.pdf)
  4. [VisionAI Stack Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/visionai_programmer_guide.pdf)

This fork keeps ztachip's original architecture, programming model and build system. The original
project is **https://github.com/ztachip/ztachip**, licensed under Apache-2.0.
