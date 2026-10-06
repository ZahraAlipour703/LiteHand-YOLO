# LiteHand-YOLO

## Efficient Attention-Enhanced YOLOv8n for Robust Hand Tracking

LiteHand-YOLO is a lightweight, single-class hand detection model built by modifying the Ultralytics YOLO framework. The architecture is designed to reduce model complexity while preserving multi-scale detection capability for real-time hand tracking applications, particularly on resource-constrained hardware.

The model combines:

* Depthwise convolution (`DWConv`) for lightweight spatial processing
* Efficient Channel Attention (`ECA`)
* Coordinate Attention (`CoordAtt`)
* C2f feature extraction
* SPPF contextual feature aggregation
* Bidirectional multi-scale feature fusion
* P3/P4/P5 detection

The project provides the modified Ultralytics source code, model configuration, real-time demo, and benchmarking utilities required to reproduce and use the model.

---

## 1. Overview

LiteHand-YOLO is based on the YOLOv8-style detection framework and uses a compact custom architecture for one-class hand detection.

The overall pipeline is:

```text
Input Image
     │
     ▼
┌─────────────────────┐
│ Lightweight         │
│ Backbone            │
│                     │
│ DWConv + ECA        │
│ DWConv + CoordAtt   │
│ C2f + SPPF          │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Multi-scale Feature │
│ Fusion              │
│                     │
│ Top-down pathway    │
│ Bottom-up pathway   │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Detection Head      │
│                     │
│ P3 / P4 / P5        │
└─────────────────────┘
     │
     ▼
 Hand Predictions
```

The model configuration is:

```text
yolov8-enhanced-structure/ultralytics/cfg/models/v8/litehand-yolov8n.yaml
```

The configuration defines a single detection class:

```yaml
task: detect
nc: 1
```

---

## 2. Architecture

### 2.1 Backbone

The backbone progressively reduces spatial resolution while increasing channel capacity.

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

### 2.2 Neck

The neck performs bidirectional feature fusion.

First, high-level semantic information is propagated from P5 toward P3:

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

A bottom-up pathway then propagates information from P3 toward P5:

```text
P3
 │
 ▼
Downsample
 │
 ▼
Concat with P4 pathway
 │
 ▼
C2f 128
 │
 ▼
Downsample
 │
 ▼
Concat with P5
 │
 ▼
C2f 256
```

### 2.3 Detection Head

The final detection layer combines three feature scales:

```yaml
- [[16, 19, 22], 1, Detect, [nc]]
```

Conceptually:

```text
P3 ─────┐
        │
P4 ─────┼──► Detect ──► Hand predictions
        │
P5 ─────┘
```

This provides multi-scale detection across small, medium, and larger spatial representations.

---

## 3. Current Model Configuration

The architecture is defined entirely by:

```text
yolov8-enhanced-structure/ultralytics/cfg/models/v8/litehand-yolov8n.yaml
```

The current configuration contains:

| Stage            | Main components        |
| ---------------- | ---------------------- |
| Early backbone   | Conv, DWConv, ECA, C2f |
| P3 stage         | DWConv, CoordAtt, C2f  |
| P4 stage         | DWConv, C2f            |
| P5 stage         | DWConv, SPPF           |
| Neck             | Upsample, Concat, C2f  |
| Bottom-up fusion | Conv, Concat, C2f      |
| Detection        | P3/P4/P5 → Detect      |

The current architecture has been locally verified with approximately:

```text
121 layers
2.06M parameters
6.4 GFLOPs
```

These values should be rechecked whenever the architecture YAML or custom modules are changed.

---

## 4. Repository Structure

```text
LiteHand-YOLO/
│
├── README.md
├── .gitignore
│
├── benchmark/
│   └── realtime_compare.py
│
├── demo/
│   └── realtime.py
│
├── weights/
│   └── README.md
│
└── yolov8-enhanced-structure/
    │
    ├── pyproject.toml
    │
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

# 5. Installation

## 5.1 Clone the repository

```bash
git clone https://github.com/ZahraAlipour703/LiteHand-YOLO.git
cd LiteHand-YOLO
```

## 5.2 Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 5.3 Install the modified Ultralytics package

LiteHand-YOLO uses the modified Ultralytics source included in this repository.

Install it in editable mode:

```bash
pip install --upgrade pip
pip install -e ./yolov8-enhanced-structure
```

Do not rely on an unrelated system-wide `ultralytics` installation when working with the LiteHand-YOLO architecture.

---

# 6. Verify the Installation

Check which Ultralytics package Python is loading:

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

The printed path should point to the repository's:

```text
yolov8-enhanced-structure/ultralytics/
```

rather than an unrelated global or `site-packages` installation.

You can also check the installed version:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

---

# 7. Build and Test the LiteHand-YOLO Model

A model construction and forward-pass test is provided at:

```text
yolov8-enhanced-structure/tests/test_litehand.py
```

Run:

```bash
python yolov8-enhanced-structure/tests/test_litehand.py
```

The test:

1. Locates the LiteHand-YOLO YAML relative to the repository.
2. Loads the custom Ultralytics implementation.
3. Builds the model.
4. Prints model information.
5. Creates a dummy input tensor.
6. Executes a forward pass.

A successful run should end with:

```text
Running forward pass...
Forward pass successful
```

---

# 8. Programmatic Model Loading

The model can also be constructed directly:

```python
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

model = YOLO(
    ROOT
    / "yolov8-enhanced-structure"
    / "ultralytics"
    / "cfg"
    / "models"
    / "v8"
    / "litehand-yolov8n.yaml"
)

model.info()
```

The model configuration should be used with the modified Ultralytics source included in this repository.

---

# 9. Pretrained Weights

Model checkpoints are intentionally not stored in Git.

The expected LiteHand-YOLO checkpoint location is:

```text
weights/litehand_yolov8n.pt
```

After obtaining the checkpoint, place it at:

```text
LiteHand-YOLO/
└── weights/
    └── litehand_yolov8n.pt
```

The same directory can be used for additional benchmark checkpoints such as:

```text
weights/
├── litehand_yolov8n.pt
└── yolov8n_hkd.pt
```

Large checkpoints should remain outside Git and be distributed through an appropriate release or external storage location.

---

# 10. Real-Time Demo

The real-time inference script is:

```text
demo/realtime.py
```

It supports webcam, image, and video sources.

## Webcam

After placing the weights in `weights/`:

```bash
python demo/realtime.py
```

By default, this uses:

```text
weights/litehand_yolov8n.pt
```

and webcam source:

```text
0
```

## Specify a model

```bash
python demo/realtime.py \
    --model weights/litehand_yolov8n.pt
```

## Specify a video

```bash
python demo/realtime.py \
    --model weights/litehand_yolov8n.pt \
    --source path/to/video.mp4
```

## Use CPU

```bash
python demo/realtime.py \
    --model weights/litehand_yolov8n.pt \
    --device cpu
```

## Use CUDA

```bash
python demo/realtime.py \
    --model weights/litehand_yolov8n.pt \
    --device cuda:0
```

## Change image size

```bash
python demo/realtime.py \
    --model weights/litehand_yolov8n.pt \
    --imgsz 640
```

## Change confidence threshold

```bash
python demo/realtime.py \
    --model weights/litehand_yolov8n.pt \
    --conf 0.25
```

## Save output

```bash
python demo/realtime.py \
    --model weights/litehand_yolov8n.pt \
    --save
```

Generated runtime outputs are written under `runs/`, which is excluded from version control.

---

# 11. Python Inference Example

```python
from ultralytics import YOLO

model = YOLO("weights/litehand_yolov8n.pt")

results = model.predict(
    source="path/to/image.jpg",
    imgsz=640,
    conf=0.25
)

for result in results:
    print(result.boxes)
```

For webcam or video inference:

```python
from ultralytics import YOLO

model = YOLO("weights/litehand_yolov8n.pt")

model.predict(
    source=0,
    imgsz=640,
    conf=0.25,
    show=True
)
```

---

# 12. Training

Training uses the Ultralytics API provided by the local modified package.

Example:

```python
from ultralytics import YOLO

model = YOLO(
    "yolov8-enhanced-structure/ultralytics/cfg/models/v8/litehand-yolov8n.yaml"
)

model.train(
    data="path/to/data.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    device=0
)
```

The dataset is not included in this repository.

Before reporting experimental results, document the complete training configuration, including:

```text
Dataset
Train/validation/test split
Input resolution
Epochs
Batch size
Optimizer
Learning rate
Scheduler
Augmentations
Random seed
Hardware
Python version
PyTorch version
Ultralytics version
CUDA version
```

Do not commit dataset files or generated training runs to the repository.

---

# 13. Validation

A trained checkpoint can be evaluated with:

```python
from ultralytics import YOLO

model = YOLO("weights/litehand_yolov8n.pt")

metrics = model.val(
    data="path/to/data.yaml",
    imgsz=640
)

print(metrics)
```

For research reporting, distinguish clearly between:

### Detection metrics

```text
Precision
Recall
mAP@0.5
mAP@0.5:0.95
```

### Efficiency metrics

```text
Parameters
GFLOPs
Latency
FPS
Memory usage
```

Runtime confidence statistics such as mean prediction confidence are not substitutes for precision, recall, or mAP.

---

# 14. Real-Time Benchmark

The repository includes:

```text
benchmark/realtime_compare.py
```

This script compares a YOLOv8n checkpoint and a LiteHand-YOLO checkpoint under the same inference loop.

Expected checkpoints:

```text
weights/yolov8n_hkd.pt
weights/litehand_yolov8n.pt
```

Example:

```bash
python benchmark/realtime_compare.py \
    --source path/to/video.mp4 \
    --device cpu \
    --frames 300
```

The script records:

```text
Latency
FPS
Detection count
Mean confidence
```

and writes the runtime data to:

```text
runs/benchmark/realtime_comparison.csv
```

Example screenshots are stored under:

```text
runs/benchmark/screenshots/
```

These outputs are ignored by Git.

---

# 15. Benchmarking Protocol

Runtime comparisons should be performed under controlled conditions.

At minimum, record:

```text
Hardware
CPU / GPU model
Batch size
Image size
Precision
Framework version
Number of frames
Warm-up procedure
Timing method
Input resolution
```

FPS and latency should not be compared across experiments when the underlying hardware or inference settings differ.

For fair architectural comparison, use the same:

* input data
* image size
* batch size
* device
* precision
* number of frames
* timing protocol

---

# 16. Custom Attention Modules

LiteHand-YOLO adds two attention modules:

```text
ECA
CoordAtt
```

Their primary implementation is located at:

```text
yolov8-enhanced-structure/ultralytics/nn/modules/attention.py
```

The modules are exported through the Ultralytics module namespace and registered with the model parser in:

```text
yolov8-enhanced-structure/ultralytics/nn/tasks.py
```

The corresponding utility files:

```text
yolov8-enhanced-structure/ultralytics/utils/eca.py
yolov8-enhanced-structure/ultralytics/utils/coord_attention.py
```

remain as compatibility wrappers around the primary implementations.

This organization keeps the YAML-referenced attention modules available through the same model-building mechanism used by the modified Ultralytics implementation.

---

# 17. Architecture Configuration

The definitive architecture specification is the YAML file:

```text
yolov8-enhanced-structure/ultralytics/cfg/models/v8/litehand-yolov8n.yaml
```

The YAML should be treated as the source of truth for:

```text
Layer order
Module type
Input connections
Channel dimensions
Feature fusion
Detection outputs
```

Any architecture diagram or manuscript figure should match this file exactly.

---

# 18. Reproducibility

For reproducible research, each reported experiment should preserve the exact configuration used to obtain the result.

Recommended experiment record:

```yaml
experiment: litehand_yolov8n

model: litehand-yolov8n.yaml

imgsz: 640
epochs: 100
batch: 16

seed: 42

device: <record hardware>

python_version: <record>
pytorch_version: <record>
ultralytics_version: <record>
cuda_version: <record>
```

The final project should maintain a clear relationship between:

```text
Model configuration
        ↓
Training configuration
        ↓
Checkpoint
        ↓
Validation results
        ↓
Benchmark results
```

---

# 19. Dataset

Datasets are not included in this repository.

Prepare the dataset independently and provide its path through the Ultralytics data configuration:

```text
path/to/data.yaml
```

A public release should document:

```text
Dataset name
Dataset source
License
Number of images
Class definitions
Train/validation/test split
Preprocessing
Annotation format
```

The repository should not contain private or redistributed dataset files unless their licenses explicitly permit redistribution.

---

# 20. Project Scope

LiteHand-YOLO is intended primarily for:

```text
Hand detection
Real-time computer vision
Robotic vision
Human-machine interaction
Resource-constrained inference
Research on lightweight object detection
```

The current repository implements a single-class hand detector.

The architecture should not be described as a general-purpose multi-class detector unless the configuration and training setup have been extended accordingly.

---

# 21. Current Limitations

The repository currently provides:

```text
Custom LiteHand-YOLO architecture
Modified Ultralytics implementation
Model construction test
Real-time inference demo
Runtime comparison script
```

For a fully reproducible research release, the following should also be finalized:

```text
Released pretrained checkpoint
Public weight-download location
Reproducible training configuration
Ablation results
Controlled benchmark results
Dataset preparation documentation
Citation metadata
Automated tests
```

---

# 22. Research Reporting

When reporting LiteHand-YOLO results, separate the following concepts:

### Accuracy

```text
Precision
Recall
mAP@0.5
mAP@0.5:0.95
```

### Model complexity

```text
Parameters
GFLOPs
```

### Runtime performance

```text
Latency
FPS
```

### Runtime descriptive statistics

```text
Detection count
Mean confidence
```

These quantities measure different properties and should not be used interchangeably.

---

# 23. Third-Party Software

LiteHand-YOLO includes a modified Ultralytics source tree.

The bundled source is distributed under the GNU Affero General Public License version 3 (AGPL-3.0), as indicated by the included project metadata and license file.

The full license text is available at:

```text
yolov8-enhanced-structure/LICENSE
```

Users and redistributors should review the license requirements and preserve applicable attribution and license notices.

---

# 24. Citation

For academic use, cite the LiteHand-YOLO research work associated with this repository once the final publication metadata is available.

Please also preserve the required attribution and citation information for the underlying Ultralytics framework and any other third-party components used in experiments.

A `CITATION.cff` file should be added to the repository for machine-readable GitHub citation support.

---

# 25. Development

The modified Ultralytics source is located under:

```text
yolov8-enhanced-structure/
```

After modifying model modules or the architecture configuration, rerun:

```bash
python yolov8-enhanced-structure/tests/test_litehand.py
```

At minimum, verify:

```text
Package import
YAML parsing
Model construction
Parameter count
Forward pass
Inference
```

Changes to:

```text
ultralytics/nn/modules/attention.py
ultralytics/nn/tasks.py
ultralytics/cfg/models/v8/litehand-yolov8n.yaml
```

should always be tested together.

---

# 26. License and Disclaimer

The software is provided for research and development purposes.

No warranty is provided for performance, accuracy, robustness, or suitability for a particular application.

Users are responsible for verifying the appropriate licensing, dataset permissions, and deployment requirements for their intended use.

---

## Project

**LiteHand-YOLO**
Efficient Attention-Enhanced YOLOv8n for Robust Hand Tracking

Repository:

```text
https://github.com/ZahraAlipour703/LiteHand-YOLO
```
