# LiteHand-YOLO

### Lightweight Attention-Enhanced YOLOv8n for Hand Detection

LiteHand-YOLO is a lightweight, single-class hand detection model developed by modifying the Ultralytics YOLOv8 framework. The architecture combines depthwise convolutions, Efficient Channel Attention (ECA), Coordinate Attention (CoordAtt), C2f feature extraction, SPPF contextual aggregation, and bidirectional multi-scale feature fusion.

The project is intended for **real-time hand detection and tracking applications**, with particular emphasis on reducing model complexity while maintaining competitive detection performance on resource-constrained hardware.

---

## Overview

LiteHand-YOLO modifies the YOLOv8-style detection pipeline by introducing lightweight convolution and attention components while retaining multi-scale detection.

The main design components are:

* **Depthwise convolution (`DWConv`)** for lightweight spatial processing
* **Efficient Channel Attention (`ECA`)** for channel-wise feature recalibration
* **Coordinate Attention (`CoordAtt`)** for lightweight spatially aware attention
* **C2f blocks** for feature extraction
* **SPPF** for contextual feature aggregation
* **Bidirectional feature fusion** for combining multi-scale representations
* **P3/P4/P5 detection** for multi-scale hand detection

The modified Ultralytics source code is included in the repository so that the architecture can be built and evaluated without relying on an external modified implementation.

---

## Architecture

The overall architecture is:

```text
                         LiteHand-YOLO
                              │
                              ▼
                         Input Image
                              │
                              ▼
                    ┌───────────────────┐
                    │ Lightweight       │
                    │ Backbone          │
                    │                   │
                    │ Conv              │
                    │ DWConv            │
                    │ ECA               │
                    │ C2f               │
                    │ CoordAtt          │
                    │ C2f               │
                    │ DWConv            │
                    │ C2f               │
                    │ DWConv + SPPF      │
                    └─────────┬─────────┘
                              │
                         P3 / P4 / P5
                              │
                              ▼
                    ┌───────────────────┐
                    │ Feature Fusion    │
                    │                   │
                    │ Top-down pathway  │
                    │ Bottom-up pathway │
                    │ Upsample / Concat │
                    │ Conv / C2f        │
                    └─────────┬─────────┘
                              │
                              ▼
                       P3 / P4 / P5
                              │
                              ▼
                         Detect Head
                              │
                              ▼
                       Hand Predictions
```

### Backbone

The backbone progressively reduces spatial resolution while increasing the number of feature channels.

```text
Input
  │
  ▼
Conv 16 / stride 2
  │
  ▼
DWConv 32 / stride 2
  │
  ▼
ECA
  │
  ▼
C2f 32
  │
  ▼
DWConv 64 / stride 2
  │
  ▼
CoordAtt
  │
  ▼
C2f 64                    → P3 / 8
  │
  ▼
DWConv 128 / stride 2
  │
  ▼
C2f 128                   → P4 / 16
  │
  ▼
DWConv 256 / stride 2
  │
  ▼
SPPF 256                  → P5 / 32
```

### Neck and Feature Fusion

The neck uses both top-down and bottom-up pathways.

#### Top-down pathway

```text
P5
 │
 ▼
Upsample ×2
 │
 ▼
Concat with P4
 │
 ▼
C2f 128
 │
 ▼
Upsample ×2
 │
 ▼
Concat with P3
 │
 ▼
C2f 64
```

#### Bottom-up pathway

```text
P3
 │
 ▼
Conv / Downsample
 │
 ▼
Concat with P4 pathway
 │
 ▼
C2f 128
 │
 ▼
Conv / Downsample
 │
 ▼
Concat with P5
 │
 ▼
C2f 256
```

### Detection Head

The final detection layer uses three feature scales:

```yaml
- [[16, 19, 22], 1, Detect, [nc]]
```

Conceptually:

```text
P3 ─────┐
        │
P4 ─────┼────► Detect ────► Hand Predictions
        │
P5 ─────┘
```

The detector is configured for a single class:

```yaml
task: detect
nc: 1
```

---

## Model Configuration

The LiteHand-YOLO architecture is defined by:

```text
yolov8-enhanced-structure/
└── ultralytics/
    └── cfg/
        └── models/
            └── v8/
                └── litehand-yolov8n.yaml
```

The configuration contains:

| Stage            | Components             |
| ---------------- | ---------------------- |
| Early backbone   | Conv, DWConv, ECA, C2f |
| P3 stage         | DWConv, CoordAtt, C2f  |
| P4 stage         | DWConv, C2f            |
| P5 stage         | DWConv, SPPF           |
| Neck             | Upsample, Concat, C2f  |
| Bottom-up fusion | Conv, Concat, C2f      |
| Detection        | P3/P4/P5 → Detect      |

The current architecture has been locally verified at approximately:

```text
121 layers
2.06M parameters
6.4 GFLOPs
```

These values should be rechecked after any architectural modification.

---

## Reported Results

The following results correspond to the project's reported experimental evaluation.

### HKD Evaluation

| Model             | Parameters |  GFLOPs |    mAP@50 | mAP@50-95 |
| ----------------- | ---------: | ------: | --------: | --------: |
| YOLOv8n           |      3.16M |     8.9 |     99.18 |     90.12 |
| **LiteHand-YOLO** |  **2.06M** | **6.4** | **99.26** | **91.42** |

Reported LiteHand-YOLO precision and recall:

| Metric    | Result |
| --------- | -----: |
| Precision | 97.82% |
| Recall    | 98.26% |

### CPU Inference

Reported real-time CPU measurements:

| Model             |      FPS |       Latency |
| ----------------- | -------: | ------------: |
| YOLOv8n           |     8.30 |     120.51 ms |
| **LiteHand-YOLO** | **8.83** | **113.22 ms** |

Reported benchmark conditions:

```text
CPU: Intel Core i7-6700
RAM: 8 GB
Input resolution: 640 × 480
Frames: 300
Execution device: CPU
```

These values are project-reported results. For publication or formal comparison, the exact dataset split, checkpoint, software environment, inference settings, preprocessing pipeline, and hardware should be recorded and the measurements independently reproduced.

---

## Repository Structure

```text
LiteHand-YOLO/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── .gitignore
│
├── benchmark/
│   └── realtime_compare.py
│
├── demo/
│   └── realtime.py
│
├── weights/
│   ├── README.md
│   └── litehand_yolov8n.pt
│
└── yolov8-enhanced-structure/
    │
    ├── pyproject.toml
    ├── tests/
    │   └── test_litehand.py
    │
    └── ultralytics/
        │
        ├── cfg/
        │   └── models/
        │       └── v8/
        │           └── litehand-yolov8n.yaml
        │
        ├── nn/
        │   ├── modules/
        │   │   ├── attention.py
        │   │   └── block.py
        │   │
        │   └── tasks.py
        │
        └── utils/
            ├── eca.py
            └── coord_attention.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ZahraAlipour703/LiteHand-YOLO.git
cd LiteHand-YOLO
```

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install -U pip
```

Install the modified local Ultralytics package:

```bash
python -m pip install -e ./yolov8-enhanced-structure
```

---

## Verify the Architecture

The repository includes a test that:

1. Loads the local Ultralytics implementation.
2. Locates the LiteHand-YOLO YAML configuration.
3. Builds the model.
4. Prints the model information.
5. Runs a dummy forward pass.

Run:

```bash
python ./yolov8-enhanced-structure/tests/test_litehand.py
```

A successful run should report that the repository-local `ultralytics` package was loaded and finish with:

```text
Forward pass successful
```

---

## Inference

The repository includes a real-time inference script in:

```text
demo/realtime.py
```

Run webcam inference:

```powershell
python .\demo\realtime.py `
    --model .\weights\litehand_yolov8n.pt `
    --source 0 `
    --device auto
```

Use CPU explicitly:

```powershell
python .\demo\realtime.py `
    --model .\weights\litehand_yolov8n.pt `
    --source 0 `
    --device cpu
```

Use a video:

```powershell
python .\demo\realtime.py `
    --model .\weights\litehand_yolov8n.pt `
    --source path\to\video.mp4 `
    --device cpu
```

Use an image:

```powershell
python .\demo\realtime.py `
    --model .\weights\litehand_yolov8n.pt `
    --source path\to\image.jpg `
    --device cpu
```

The script automatically prefers the repository's local modified Ultralytics source.

---

## Benchmarking

The repository provides:

```text
benchmark/realtime_compare.py
```

for comparing LiteHand-YOLO with a baseline YOLOv8n checkpoint.

The benchmark reports runtime-oriented measurements such as:

* inference latency
* FPS
* number of detections
* mean detection confidence

The runtime confidence statistics are **not** equivalent to dataset-level precision, recall, or mAP.

### Example

```powershell
python .\benchmark\realtime_compare.py `
    --baseline path\to\yolov8n_hkd.pt `
    --litehand .\weights\litehand_yolov8n.pt `
    --source 0 `
    --device cpu `
    --frames 300
```

The baseline checkpoint is not included in the repository and must be supplied separately.

For reproducible benchmarking, record:

```text
Model checkpoint
Dataset / video source
Frame count
Input resolution
Image size
Device
CPU / GPU model
Software environment
PyTorch version
Ultralytics version / repository commit
```

---

## Custom Modules

LiteHand-YOLO introduces two custom attention modules.

### Efficient Channel Attention

Implemented in:

```text
yolov8-enhanced-structure/
└── ultralytics/
    └── nn/
        └── modules/
            └── attention.py
```

The module performs lightweight channel attention while preserving the input channel dimension.

### Coordinate Attention

Coordinate Attention is implemented in the same module and incorporates directional spatial information while preserving the channel dimension.

The modules are registered with the model parser so that they can be referenced directly from the YAML architecture definition.

---

## Reproducibility

The repository includes the modified model source, configuration, inference utilities, and a model construction test.

For experimental reproduction, the following should be kept fixed:

```text
Architecture YAML
Model checkpoint
Dataset split
Image size
Preprocessing
Confidence / IoU thresholds
Hardware
PyTorch version
CUDA version, when applicable
Python version
Repository commit
```

Performance metrics may vary with implementation details and runtime environment.

---

## Weights

The repository contains the LiteHand-YOLO checkpoint under:

```text
weights/
└── litehand_yolov8n.pt
```

The checkpoint should be used with the corresponding LiteHand-YOLO YAML architecture.

The baseline YOLOv8n checkpoint used for comparative benchmarking is not included.

---

## Paper

The research paper associated with this implementation will be linked here:

```text
Paper: [to be added]
```

When the paper is publicly available, this section should be updated with the official publication or preprint link.

---

## Citation

If you use LiteHand-YOLO in research or development, please cite the project using the metadata in:

```text
CITATION.cff
```

A BibTeX entry can also be added here once the associated paper or preprint is publicly available.

```bibtex
@software{alipour_litehand_yolo,
  author  = {Zahra Alipour},
  title   = {LiteHand-YOLO: Lightweight Attention-Enhanced YOLOv8n for Hand Detection},
  year    = {2026},
  url     = {https://github.com/ZahraAlipour703/LiteHand-YOLO}
}
```

---

## License

LiteHand-YOLO is released under the:

**GNU Affero General Public License v3.0 (AGPL-3.0)**

See:

```text
LICENSE
```

This project modifies components of the Ultralytics YOLO codebase. Users should review the applicable Ultralytics licensing and attribution requirements when redistributing or deploying the software.

---

## Acknowledgements

This project builds on the Ultralytics YOLO framework and uses PyTorch and OpenCV for model development and inference.

The custom architecture and experimental evaluation in this repository were developed as part of the LiteHand-YOLO project.

---

## Author

**Zahra Alipour**

GitHub:

https://github.com/ZahraAlipour703

Repository:

https://github.com/ZahraAlipour703/LiteHand-YOLO

---

## Project Status

LiteHand-YOLO is an experimental research implementation.

The repository currently provides:

* custom lightweight YOLO architecture
* ECA and Coordinate Attention modules
* modified Ultralytics model parser integration
* model configuration
* pretrained LiteHand-YOLO checkpoint
* real-time inference script
* runtime benchmarking script
* reproducibility test

Future research work may include more extensive ablation studies, additional datasets, broader hardware evaluation, and comparison with other lightweight detection architectures.
