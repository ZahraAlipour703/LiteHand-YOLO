@'
# LiteHand-YOLO

### Lightweight Attention-Enhanced YOLOv8n for Hand Detection

LiteHand-YOLO is a lightweight single-class hand detection model developed by modifying the Ultralytics YOLOv8 framework. The architecture combines depthwise convolutions, Efficient Channel Attention (ECA), Coordinate Attention (CoordAtt), C2f blocks, SPPF, and bidirectional multi-scale feature fusion.

## Highlights

- **2.06M parameters**
- **6.4 GFLOPs**
- P3/P4/P5 multi-scale detection
- ECA and Coordinate Attention
- Lightweight depthwise convolutions
- Real-time-oriented inference and benchmarking tools
- Modified Ultralytics source included for reproducibility

## Reported Results

### HKD evaluation

| Model | Params | GFLOPs | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 3.16M | 8.9 | 99.18 | 90.12 |
| LiteHand-YOLO | 2.06M | 6.4 | **99.26** | **91.42** |

LiteHand-YOLO reported **97.82% precision** and **98.26% recall** on the evaluated setting.

### CPU inference

| Model | FPS | Latency |
|---|---:|---:|
| YOLOv8n | 8.30 | 120.51 ms |
| LiteHand-YOLO | **8.83** | **113.22 ms** |

Reported CPU benchmark conditions: Intel Core i7-6700, 8 GB RAM, 640×480 input, 300 frames.

These numbers are project-reported experimental results and should be rechecked against the exact dataset split, checkpoints, software environment, and hardware before being used as final paper results.

## Architecture

```text
Input
  │
  ▼
Conv 16
  │
  ▼
DWConv 32 → ECA → C2f
  │
  ▼
DWConv 64 → CoordAtt → C2f   → P3
  │
  ▼
DWConv 128 → C2f              → P4
  │
  ▼
DWConv 256 → SPPF             → P5
  │
  ▼
Top-down + Bottom-up Feature Fusion
  │
  ▼
P3 / P4 / P5
  │
  ▼
Detect