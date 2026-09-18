# Ultra-Nano YOLO Detector

A lightweight single-class object detection architecture based on the Ultralytics YOLO framework, designed for efficient real-time detection under constrained computational resources.

The model combines **depthwise convolutions, lightweight attention, C2f feature extraction, SPPF contextual aggregation, and bidirectional multi-scale feature fusion** while retaining the standard YOLO-style P3/P4/P5 detection hierarchy.

---

## 1. Overview

This project investigates a compact YOLO-style detector designed to reduce computational cost while maintaining useful detection performance.

The architecture follows the conventional three-stage detection pipeline:

```text
Input Image
     │
     ▼
┌──────────────┐
│   Backbone   │  Feature extraction
└──────────────┘
     │
     ▼
┌──────────────┐
│     Neck     │  Multi-scale feature fusion
└──────────────┘
     │
     ▼
┌──────────────┐
│     Head     │  Object detection
└──────────────┘
     │
     ▼
 P3 / P4 / P5
 Detection
```

The backbone progressively reduces spatial resolution while increasing channel capacity. The neck performs both **top-down** and **bottom-up** feature fusion, and the final detection head receives three feature scales.

Ultralytics model YAML files represent architectures as ordered layers using the form:

```text
[from, repeats, module, args]
```

and support multi-input operations such as concatenation and multi-scale detection.

---

# 2. Motivation

The primary motivation is to investigate whether a compact detector can provide an appropriate balance between:

* detection accuracy,
* computational complexity,
* memory consumption,
* inference speed, and
* suitability for resource-constrained hardware.

The design therefore avoids simply increasing network depth or width.

Instead, computationally lighter components are introduced at selected stages:

* **Depthwise convolution (`DWConv`)** for efficient spatial processing.
* **Efficient Channel Attention (`ECA`)** for lightweight channel-wise feature recalibration.
* **Coordinate Attention (`CoordAtt`)** for incorporating positional information into channel attention.
* **C2f blocks** for efficient feature transformation and gradient flow.
* **SPPF** for multi-scale contextual aggregation.
* **Bidirectional feature fusion** for combining semantic and spatial information.

The purpose of these modifications should be evaluated experimentally rather than assumed. In particular, each architectural modification should ultimately be supported by an ablation study.

---

# 3. Architecture

The model is defined in:

```text
configs/ultra_nano_detect_fixed_fusion.yaml
```

The architecture contains three major components.

## 3.1 Backbone

The backbone extracts hierarchical features at progressively lower spatial resolutions.

```text
Input
  │
  ├── Conv 16, stride 2
  │
  ├── DWConv 32, stride 2
  ├── ECA 32
  ├── C2f 32
  │
  ├── DWConv 64, stride 2
  ├── CoordAtt 64
  ├── C2f 64          → P3 / 8
  │
  ├── DWConv 128, stride 2
  ├── C2f 128         → P4 / 16
  │
  ├── DWConv 256, stride 2
  └── SPPF 256        → P5 / 32
```

The `/8`, `/16`, and `/32` notation denotes the effective spatial stride relative to the input image.

These three scales are subsequently used by the detection head.

---

# 4. Backbone Layer Table

| Index | Module   | Output Channels | Stride | Feature Level |
| ----: | -------- | --------------: | -----: | ------------- |
|     0 | Conv     |              16 |      2 | P1/2          |
|     1 | DWConv   |              32 |      2 | P2/4          |
|     2 | ECA      |              32 |      1 | P2/4          |
|     3 | C2f      |              32 |      1 | P2/4          |
|     4 | DWConv   |              64 |      2 | P3/8          |
|     5 | CoordAtt |              64 |      1 | P3/8          |
|     6 | C2f      |              64 |      1 | P3/8          |
|     7 | DWConv   |             128 |      2 | P4/16         |
|     8 | C2f      |             128 |      1 | P4/16         |
|     9 | DWConv   |             256 |      2 | P5/32         |
|    10 | SPPF     |             256 |      1 | P5/32         |

---

# 5. Neck

The neck performs bidirectional feature fusion.

It first propagates high-level semantic information from P5 toward P3:

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

This is the **top-down pathway**.

The network then performs a bottom-up pathway:

```text
P3
 │
 ▼
Downsample
 │
 ▼
Concat with P4 feature
 │
 ▼
C2f 128
 │
 ▼
Downsample
 │
 ▼
Concat with P5 feature
 │
 ▼
C2f 256
```

The resulting feature maps provide the three detection scales.

---

# 6. Neck Layer Table

| Index | Module   | Output Channels | Function             |
| ----: | -------- | --------------: | -------------------- |
|    11 | Upsample |             256 | P5 → P4              |
|    12 | Concat   |               — | P5/P4 fusion         |
|    13 | C2f      |             128 | P4 refinement        |
|    14 | Upsample |             128 | P4 → P3              |
|    15 | Concat   |               — | P4/P3 fusion         |
|    16 | C2f      |              64 | P3 refinement        |
|    17 | Conv     |              64 | P3 → P4 downsampling |
|    18 | Concat   |               — | Bottom-up P4 fusion  |
|    19 | C2f      |             128 | P4 refinement        |
|    20 | Conv     |             128 | P4 → P5 downsampling |
|    21 | Concat   |               — | Bottom-up P5 fusion  |
|    22 | C2f      |             256 | P5 refinement        |

The use of P3/P4/P5 feature scales follows the standard multi-scale detection design used by YOLO-family architectures. Ultralytics' model configurations likewise construct detection outputs from multiple spatial scales and pass them to `Detect`.

---

# 7. Detection Head

The final detection layer is:

```yaml
- [[16, 19, 22], 1, Detect, [nc]]
```

where:

```text
16 → P3 feature
19 → P4 feature
22 → P5 feature
```

and:

```yaml
nc: 1
```

specifies a single detection class.

Therefore:

```text
P3 ─────┐
        │
P4 ─────┼──► Detect ──► Predictions
        │
P5 ─────┘
```

The use of a multi-input `Detect` layer is consistent with the Ultralytics YAML architecture format.

---

# 8. Complete Architecture

```text
                         BACKBONE
                            │
Input
 │
 ▼
Conv 16
 │
 ▼
DWConv 32
 │
 ├── ECA 32
 │
 └── C2f 32
       │
       ▼
   DWConv 64
       │
   CoordAtt 64
       │
   C2f 64 ────────────────────────────────┐
       │                                  │
       ▼                                  │
   DWConv 128                             │
       │                                  │
   C2f 128 ───────────────────────┐       │
       │                           │       │
       ▼                           │       │
   DWConv 256                      │       │
       │                           │       │
     SPPF                          │       │
       │                           │       │
       ▼                           │       │
   Upsample ×2                     │       │
       │                           │       │
       ├──────────────► Concat ◄───┘       │
       │                    │              │
       │                  C2f 128          │
       │                    │              │
       │              Upsample ×2          │
       │                    │              │
       │                    ├────► Concat ◄┘
       │                    │
       │                  C2f 64
       │                    │
       │                  P3
       │                    │
       │                DownConv
       │                    │
       │                 Concat ◄── C2f 128
       │                    │
       │                  C2f 128
       │                    │
       │                 DownConv
       │                    │
       │                 Concat ◄── SPPF
       │                    │
       │                  C2f 256
       │
       └─────────────────────────────────────┐
                                              │
                 P3 ─────────────────────────┤
                 P4 ─────────────────────────┼──► Detect
                 P5 ─────────────────────────┤
                                              │
                                              ▼
                                         Predictions
```

---

# 9. Custom Modules

The architecture references:

```text
DWConv
ECA
CoordAtt
```

These should be implemented in the repository if they are not provided by the exact Ultralytics version being used.

A recommended structure is:

```text
src/
└── models/
    ├── modules/
    │   ├── __init__.py
    │   ├── dwconv.py
    │   ├── eca.py
    │   └── coordatt.py
    │
    └── tasks.py
```

The custom modules must be imported into the appropriate Ultralytics module namespace and registered with the model parser when required. Ultralytics explicitly documents this requirement for custom YAML modules.

---

# 10. Recommended Repository Structure

The project should be organized as follows:

```text
ultra-nano-yolo/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── ultra_nano_detect_fixed_fusion.yaml
│   └── data.yaml
│
├── models/
│   ├── __init__.py
│   │
│   └── modules/
│       ├── __init__.py
│       ├── dwconv.py
│       ├── eca.py
│       └── coordatt.py
│
├── scripts/
│   ├── train.py
│   ├── validate.py
│   ├── predict.py
│   ├── benchmark.py
│   └── visualize_model.py
│
├── tools/
│   └── architecture/
│       ├── yolo_ultra_nano_generated.tex
│       ├── input.jpg
│       └── README.md
│
├── experiments/
│   ├── baseline/
│   ├── eca/
│   ├── coordatt/
│   ├── fusion/
│   └── full_model/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── tables/
│
├── weights/
│   └── .gitkeep
│
└── datasets/
    └── .gitkeep
```

Do **not** commit large datasets or model weights unless the repository policy explicitly requires them.

---

# 11. Configuration

The main architecture configuration is:

```text
configs/ultra_nano_detect_fixed_fusion.yaml
```

Current configuration:

```yaml
task: detect
nc: 1
scale: "s"
```

The `nc` parameter defines the number of classes. In Ultralytics configurations, the model configuration can also use compound scaling definitions to control depth, width, and maximum channels.

### Important

If this project is intended to reproduce a specific experiment, record the exact:

* Ultralytics version
* PyTorch version
* CUDA version
* Python version
* GPU
* image size
* batch size
* optimizer
* learning rate
* epochs
* augmentation configuration
* random seed

in a reproducibility section or experiment configuration.

---

# 12. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the project uses a locally modified Ultralytics source tree, install that version rather than silently using an unrelated pip installation.

Verify:

```bash
python -c "import torch; import ultralytics; print(torch.__version__); print(ultralytics.__version__)"
```

---

# 13. Model Construction Test

Before training, verify that the YAML can actually be parsed.

Example:

```python
from ultralytics import YOLO

model = YOLO(
    "configs/ultra_nano_detect_fixed_fusion.yaml",
    task="detect"
)

model.info()
```

`model.info()` is useful for checking the parsed architecture, parameter count, and computational information. Ultralytics documents this as a standard way to inspect a model.

A model should **not** be considered ready for experimentation merely because the YAML file is syntactically valid.

The following must also be verified:

1. Every referenced module resolves correctly.
2. Every tensor connection has compatible spatial dimensions.
3. Every concatenation has compatible height and width.
4. The detection head receives the intended three feature maps.
5. The forward pass succeeds.
6. Parameter and FLOP counts are recorded.

---

# 14. Training

Example:

```python
from ultralytics import YOLO

model = YOLO(
    "configs/ultra_nano_detect_fixed_fusion.yaml",
    task="detect"
)

model.train(
    data="configs/data.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    device=0
)
```

The exact training parameters used for reported experiments must be documented.

Do not compare two architectures using different training conditions unless the experimental design explicitly requires it.

---

# 15. Validation

Example:

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")

metrics = model.val(
    data="configs/data.yaml",
    imgsz=640
)

print(metrics)
```

At minimum, report:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95
* parameter count
* computational complexity
* inference latency/FPS

For a real-time detector, latency should be measured under clearly specified hardware and inference conditions.

---

# 16. Prediction

Example:

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")

results = model.predict(
    source="path/to/image.jpg",
    imgsz=640,
    conf=0.25
)
```

---

# 17. Architecture Visualization

The repository contains a LaTeX/TikZ visualization of the architecture.

The figure is a **documentation artifact** and should reproduce the computational structure defined by the YAML.

Recommended location:

```text
tools/
└── architecture/
    ├── yolo_ultra_nano_generated.tex
    ├── input.jpg
    └── README.md
```

If using PlotNeuralNet, its standard repository separates Python architecture-generation code from LaTeX layer resources. The `layers/` directory contains reusable layer definitions, while generated `.tex` files can be compiled with `pdflatex`.

For the PlotNeuralNet version, the normal dependency is:

```text
PlotNeuralNet/
├── layers/
│   ├── init.tex
│   ├── Box.sty
│   └── RightBandedBox.sty
├── pycore/
├── pyexamples/
└── examples/
```

The upstream `layers/init.tex` loads the required TikZ libraries and layer definitions.

### Windows recommendation

Avoid absolute Windows paths such as:

```text
D:\zra\PROJECTS\...
```

inside `\subimport`.

Instead, keep the required `layers/` directory relative to the `.tex` file:

```text
architecture/
├── yolo_ultra_nano_generated.tex
└── layers/
    ├── init.tex
    ├── Box.sty
    └── RightBandedBox.sty
```

Then use:

```latex
\subimport{layers/}{init.tex}
```

or:

```latex
\input{layers/init.tex}
```

depending on the structure of the figure.

---

# 18. Input Image for the LaTeX Figure

If the architecture figure contains:

```latex
\includegraphics{input.jpg}
```

then `input.jpg` must exist in the LaTeX compilation directory or in the specified relative path.

Recommended:

```text
tools/
└── architecture/
    ├── yolo_ultra_nano_generated.tex
    └── input.jpg
```

Then:

```latex
\includegraphics[width=...]{input.jpg}
```

will work without an absolute Windows path.

If no input image is required, remove the `\includegraphics` node rather than leaving a missing file reference.

---

# 19. Experiments

The experimental design should separate architectural changes.

Recommended experiment groups:

```text
experiments/
├── baseline/
├── eca/
├── coordatt/
├── lightweight_conv/
├── fusion/
└── full_model/
```

A useful ablation structure is:

| Model      | DWConv | ECA | CoordAtt | Bidirectional Fusion |
| ---------- | -----: | --: | -------: | -------------------: |
| Baseline   |      – |   – |        – |                    – |
| + DWConv   |      ✓ |   – |        – |                    – |
| + ECA      |      ✓ |   ✓ |        – |                    – |
| + CoordAtt |      ✓ |   ✓ |        ✓ |                    – |
| Full model |      ✓ |   ✓ |        ✓ |                    ✓ |

The exact baseline must be defined explicitly. Do not call a model a "baseline" unless its architecture and initialization are documented.

---

# 20. Reproducibility

Each experiment should record:

```text
Model configuration
Dataset version
Train/validation/test split
Input resolution
Epochs
Batch size
Optimizer
Learning rate
Scheduler
Augmentations
Random seed
GPU
CUDA version
PyTorch version
Ultralytics version
Training time
Inference hardware
```

A recommended experiment record is:

```yaml
experiment: full_model
model: ultra_nano_detect_fixed_fusion.yaml

seed: 42
imgsz: 640
epochs: 100
batch: 16

optimizer: <record actual optimizer>
lr0: <record actual value>

device: <record GPU>

torch_version: <record>
ultralytics_version: <record>
cuda_version: <record>
```

---

# 21. Evaluation Protocol

The final comparison should include both **accuracy and efficiency**.

Recommended table:

| Model    | Params | GFLOPs | Precision | Recall | mAP50 | mAP50-95 | FPS | Latency |
| -------- | -----: | -----: | --------: | -----: | ----: | -------: | --: | ------: |
| Baseline |      — |      — |         — |      — |     — |        — |   — |       — |
| Proposed |      — |      — |         — |      — |     — |        — |   — |       — |

The reported FPS/latency must specify:

```text
GPU/CPU:
Batch size:
Image size:
Precision:
Framework:
Warm-up:
Number of repetitions:
```

This prevents misleading speed comparisons.

---

# 22. Why These Components?

## DWConv

Depthwise convolution separates spatial filtering from channel mixing and can substantially reduce convolutional computation compared with standard convolution.

Its use in this project is intended to reduce the computational burden of the backbone.

## ECA

ECA provides channel attention without requiring a large fully connected projection.

It is positioned early in the network to recalibrate feature channels with relatively low overhead.

## Coordinate Attention

Coordinate Attention incorporates positional information into channel attention.

It is placed at the P3 stage, where spatial information remains relatively detailed and may be useful for localization.

## C2f

C2f blocks provide feature transformation and information flow while keeping the architecture relatively compact.

## SPPF

SPPF aggregates contextual information at the deepest feature level while maintaining an efficient implementation.

## Bidirectional Fusion

The top-down pathway transfers semantic information toward higher-resolution features.

The bottom-up pathway subsequently propagates refined localization information back toward deeper feature levels.

The purpose of this design is to improve cross-scale information exchange without introducing a substantially larger backbone.

---

# 23. Design Hypothesis

The central hypothesis of the project is:

> A compact detector can improve the accuracy–efficiency trade-off by combining lightweight convolution, selective attention, and bidirectional multi-scale feature fusion rather than increasing network depth and width.

This is a hypothesis to be experimentally tested, not a claimed result.

The strongest evidence should therefore come from:

1. controlled baseline comparison,
2. component ablation,
3. computational analysis,
4. latency measurement, and
5. qualitative error analysis.

---

# 24. Important Scientific Consideration

The architectural diagram must correspond exactly to the executable YAML.

For example, if the YAML contains:

```yaml
- [[16, 19, 22], 1, Detect, [nc]]
```

the diagram must show three inputs to the detection head:

```text
P3 → Detect
P4 → Detect
P5 → Detect
```

It should not show three independent detection heads unless the implementation actually contains three detection modules.

Likewise, an architectural figure must not introduce an operation that does not exist in the YAML.

The YAML should therefore be treated as the primary architectural specification.

---

# 25. Current Architecture Summary

```text
Input
  │
  ▼
Lightweight Backbone
  │
  ├── DWConv
  ├── ECA
  ├── CoordAtt
  ├── C2f
  └── SPPF
  │
  ▼
Top-Down Fusion
  │
  ├── Upsample
  ├── Concat
  └── C2f
  │
  ▼
P3
  │
  ▼
Bottom-Up Fusion
  │
  ├── Downsample
  ├── Concat
  └── C2f
  │
  ▼
P4 / P5
  │
  ▼
Multi-Scale Detect
  │
  ├── P3/8
  ├── P4/16
  └── P5/32
```

---

# 26. Repository Status

Before claiming the architecture is experimentally validated, the following checklist should be completed:

* [ ] YAML parses successfully.
* [ ] Custom modules are implemented.
* [ ] Custom modules are correctly registered/imported.
* [ ] Forward pass succeeds.
* [ ] Tensor shapes have been verified.
* [ ] `model.info()` has been recorded.
* [ ] Baseline model is defined.
* [ ] Dataset split is fixed.
* [ ] Training configuration is fixed.
* [ ] Ablation experiments are completed.
* [ ] Accuracy metrics are recorded.
* [ ] Parameter count is recorded.
* [ ] GFLOPs are recorded.
* [ ] Inference latency is measured.
* [ ] FPS is measured under controlled conditions.
* [ ] Qualitative predictions are inspected.
* [ ] Failure cases are documented.
* [ ] Architecture figure matches the final YAML.
* [ ] README reflects the final implementation.

---

# 27. Citation and Attribution

This project uses the Ultralytics YOLO framework and, where applicable, PlotNeuralNet for architecture visualization.

Ultralytics' documentation should be consulted for the exact version of the model parser, available modules, YAML syntax, training API, and licensing requirements.

For architecture visualization, PlotNeuralNet provides predefined LaTeX/TikZ layer components and examples for generating neural-network diagrams.

If this repository is intended for publication, include the appropriate citations and license notices for all third-party components actually used.

---

# 28. License

Add the project's actual license here.

Do not claim that the project is MIT/Apache/GPL/etc. until the repository has explicitly selected and included that license.

Third-party framework licenses must also be respected.

---

## Project Principle

The goal of this repository is not simply to create a smaller YOLO model.

The objective is to **experimentally characterize whether lightweight convolution, attention mechanisms, and bidirectional multi-scale fusion can provide a useful accuracy–efficiency trade-off for the target detection problem.**

All architectural claims should therefore be supported by reproducible experiments and ablation studies.
