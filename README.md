# ETAS: An Open-Source NAS Framework for Early-Exit CNNs on Edge RISC-V Accelerator

ETAS is an open-source framework for hardware-aware neural architecture search (NAS) and deployment
of quantized early-exit (EE) CNNs on
[ZTA-Q](https://github.com/kevinli-hw/ZTA-Q/tree/feature/ZTA-Q), a RISC-V-based SIMD accelerator
built on [ztachip](https://github.com/ztachip/ztachip). ETAS turns TFLite models into sequential
execution stages with runtime exit decisions. It jointly optimizes exit placement, confidence
thresholds and exit-head architectures, using latency profiled on the FPGA.

ETAS was evaluated with five CNNs on a Digilent Arty A7-100T FPGA at 83.3 MHz. It reaches up to
**5.94×** inference speedup over the static baseline on CIFAR-10 and up to **3.03×** on CIFAR-100,
with minor accuracy degradation.

## Highlights

- **TFLite-to-FPGA deployment flow for quantized EE models.** The operator list is reordered into
  sequential stages. The SIMD NPU runs the operators and the RISC-V host takes the exit decisions at
  runtime.
- **Low host-side overhead.** Integer-only input preprocessing and a softmax-free exit decision cut
  the host cost, with negligible accuracy loss.
- **Hardware-aware NAS.** A latency model built from per-operator FPGA measurements lets the search
  optimize exit placement, confidence thresholds and head architectures together, without exhaustive
  board runs.

## Early-exit deployment flow

### Operator list reordering

The operator list exported by TFLite places the exit heads and the final classifier after all
backbone operators. That order preserves the data dependencies, but it prevents early termination
when the operators run one after another. ETAS reorders the list so that each exit head directly
follows its backbone stage. The host can then decide to exit before the next backbone operator is
dispatched.

### Sequential staged execution

The VexRiscv host CPU (RV32IM, no floating-point unit) offloads every neural-network operator to the
SIMD accelerator and handles only the runtime exit decisions. The accelerator has a data plane for
DDR communication and processing elements (PEs) connected through a stream processor for activation
processing. Cycle counters profile the operator execution and the host-side decisions, and these
profiles feed the latency model of the search.

Two host-side operations are simplified:

- **Preprocessing:** float input normalization with mean and standard deviation is replaced by
  per-channel integer offsets.
- **Exit decision:** the softmax confidence is replaced by an integer score margin between the two
  largest logits.

Host-side latency on the board. Preprocessing is per image; the decision is per evaluated exit.

| Host-side operation       |   Float | Simplified | Speedup |
| ------------------------- | ------: | ---------: | ------: |
| Preprocessing             | 37.2 ms |    0.49 ms |     76× |
| Exit decision (CIFAR-10)  | 0.49 ms |   0.010 ms |     49× |
| Exit decision (CIFAR-100) | 3.95 ms |   0.024 ms |    163× |

## Hardware-aware neural architecture search

### MAC count vs. measured latency

A MAC-based search estimates the cost of a two-exit model from its expected MACs. Here $`M(S_i)`$ is
the backbone cost of stage $`i`$, $`M(E_i)`$ the cost of exit head $`i`$, and $`\alpha_1`$, $`\alpha_2`$ the
unconditional exit rates of the two exits:

```math
M(S) = M(S_1) + M(E_1) + (1-\alpha_1)\big(M(S_2) + M(E_2)\big) + (1-\alpha_1-\alpha_2)\,M(S_3)
```

On ZTA-Q the MAC count does not track latency:

- Operators with little or no MAC contribution, such as max pooling in SqueezeNet, still add
  noticeable latency.
- The latency per MAC varies with the operator type (3×3, pointwise or depthwise convolution) and
  the output resolution. It is up to 45× higher than for the most efficient operator in the same
  backbone.

### Latency model

ETAS replaces $`M(S_i)`$ and $`M(E_i)`$ with measured latencies $`T(S_i)`$ and $`T(E_i)`$. The latency of
each backbone segment is the sum of its measured operator times. For an exit head of type $`\tau`$
at a tap of resolution $`R`$, the latency grows linearly with the head's MACs:

```math
T(E) = T(op) + T(dec) = \big[A_{\tau,R}\,M(\mathrm{head}) + K_{\tau,R}\big] + T(dec)
```

- $`A_{\tau,R}`$ is the effective execution time per MMAC.
- $`K_{\tau,R}`$ is the time that does not scale with the input width, such as the fixed-width GAP and
  FC layers.
- $`T(dec)`$ is the measured host-side decision latency.

$`A_{\tau,R}`$ and $`K_{\tau,R}`$ are fitted by least squares on heads measured on the board:

- resolutions $`R \in \{32, 16, 8, 4, 2\}`$;
- two to three input widths between 64 and 1024 per resolution;
- the average of five runs for each head.

For the configurations selected by the search, the model's prediction is within 1.3% of the
measured latency.

### Joint search

- **Backbones:** SqueezeNet, MobileNetV1, MobileNetV2, ResNet-18 and ResNet-50, trained on CIFAR-10
  and CIFAR-100.
- **Candidate exits:** the block boundaries of each backbone.
- **Exit heads**, each trained with the backbone frozen:
  - **Lightweight**: global average pooling (GAP) and two fully connected (FC) layers.
  - **Heavy_CV**: a 3×3 convolution before the lightweight head.
  - **Heavy_PW**: a 1×1 pointwise convolution in place of the 3×3 one.
  - **Heavy_DW**: a 3×3 depthwise convolution followed by a 1×1 pointwise convolution.
- **Data:** the training set is split into a class-balanced calibration set of 5,000 images, used
  for quantization calibration and threshold selection, and 45,000 training images.
- **Search:**
  1. Confidence thresholds are discretized into 1000 values, from 0.000 to 0.999.
  2. For every combination of exit placements, head architectures and thresholds, the search takes
     the exit rates and the accuracy on the calibration set, and the expected latency from the
     latency model.
  3. It returns the configuration with the lowest expected latency that meets the accuracy target.

## Results

### Latency-scored vs. MAC-scored search

Both searches were run under the same accuracy targets, from the backbone accuracy down to 6
percentage points (pp) below it. For most targets, across the five CNNs and both datasets, the
latency-scored search finds faster configurations. At matched test accuracy it gives up to 13.8%
more speedup than the MAC-scored search.

### FPGA evaluation

Setup:

- Board: Digilent Arty A7-100T at 83.3 MHz.
- Data: the full test sets.
- Accuracy target: 1 pp below the backbone accuracy, on the calibration set.

Each cell gives top-1 accuracy (%), mean latency (ms) and speedup over the baseline. The latency
includes the host preprocessing. The columns are:

- **ETAS NAS**: the configuration found by the latency-scored search, with float preprocessing and
  softmax exit decisions.
- **ETAS E2E**: the same configuration with integer preprocessing and score-margin exit decisions.

**CIFAR-10**

| Model       | Baseline       | ETAS NAS              | ETAS E2E              |
| ----------- | -------------- | --------------------- | --------------------- |
| SqueezeNet  | 91.87 / 76.1   | 90.66 / 63.9 / 1.19×  | 90.52 / 26.2 / 2.90×  |
| MobileNetV1 | 92.67 / 98.6   | 91.55 / 54.9 / 1.80×  | 91.08 / 17.1 / 5.77×  |
| MobileNetV2 | 92.36 / 92.0   | 90.90 / 52.8 / 1.74×  | 90.72 / 15.5 / 5.94×  |
| ResNet-18   | 96.06 / 411.4  | 95.19 / 194.3 / 2.12× | 95.19 / 160.8 / 2.56× |
| ResNet-50   | 96.31 / 1013.0 | 94.80 / 393.9 / 2.57× | 94.17 / 351.2 / 2.88× |

**CIFAR-100**

| Model       | Baseline       | ETAS NAS              | ETAS E2E              |
| ----------- | -------------- | --------------------- | --------------------- |
| SqueezeNet  | 67.38 / 77.5   | 66.63 / 75.8 / 1.02×  | 65.93 / 32.1 / 2.41×  |
| MobileNetV1 | 70.88 / 99.0   | 69.74 / 78.6 / 1.26×  | 69.68 / 35.6 / 2.78×  |
| MobileNetV2 | 68.89 / 92.9   | 67.08 / 74.1 / 1.25×  | 67.07 / 30.7 / 3.03×  |
| ResNet-18   | 78.22 / 412.0  | 76.27 / 304.9 / 1.35× | 76.03 / 264.7 / 1.56× |
| ResNet-50   | 79.66 / 1013.5 | 78.18 / 776.3 / 1.31× | 78.18 / 735.4 / 1.38× |

- The host-side simplifications matter most for SqueezeNet, MobileNetV1 and MobileNetV2, whose
  inference time is comparable to the float preprocessing.
- The smaller gains on CIFAR-100 suggest that the harder classification task leaves fewer early-exit
  opportunities.

## Platform setup

ETAS runs on the ZTA-Q platform. The steps below build it for the Arty A7-100T.

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
