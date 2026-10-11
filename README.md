# ztachip — Early-Exit Inference (`feature/ETA`)

> **Work in progress.** This branch starts from
> [`feature/ZTA-Q`](https://github.com/kevinli-hw/ZTA-Q/tree/feature/ZTA-Q) and for now reuses its
> resources as they are: per-channel INT8 quantization, the firmware and the FPGA batch test harness.
> The early-exit code, models and test scripts will be added once they are cleaned up, and this
> README will be updated with them.

ETA adds early-exit inference to ztachip. Small classifier heads are attached to intermediate layers
of a CNN backbone. At each exit the firmware checks the head's confidence and stops the inference
when it is high enough, so easy inputs skip the rest of the network. The aim is a lower average
latency on the FPGA at a small accuracy cost.

## Planned contents

- **Multi-exit runtime**: exit heads at selected backbone layers, with a gate per exit that waits for
  the head's result, reads its logits and decides whether to stop.
- **Exit configuration**: where to place the exits and how to set their thresholds.
- **Models**: SqueezeNet, MobileNetV1, MobileNetV2, ResNet-18 and ResNet-50 on CIFAR-10 and
  CIFAR-100, as full-integer INT8 TFLite with per-channel weight scales.
- **Board evaluation**: accuracy and latency measured on the FPGA over the full test sets.

## Status

| Part | Status |
| ---- | ------ |
| Per-channel quantization, firmware, batch test harness | inherited from `feature/ZTA-Q` |
| Early-exit runtime | to be added |
| Early-exit models and build scripts | to be added |
| Board test scripts and results | to be added |

## Build and FPGA setup

Until the early-exit parts land, the ZTA-Q instructions apply unchanged. See the
[`feature/ZTA-Q` README](https://github.com/kevinli-hw/ZTA-Q/blob/feature/ZTA-Q/README.md) for the
models, preprocessing, bitstream and firmware build, JTAG setup and batch tests.

## Upstream

This fork keeps ztachip's original architecture, programming model, and build system. For the
accelerator internals and the DSL:

1. [Technical overview](Documentation/Overview.md)
2. [Hardware Architecture](Documentation/HardwareDesign.md)
3. [Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/ztachip_programmer_guide.pdf)
4. [VisionAI Stack Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/visionai_programmer_guide.pdf)

Original project: **https://github.com/ztachip/ztachip** — licensed under Apache-2.0.
