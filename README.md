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
