from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

# Always prefer the Ultralytics implementation bundled with this repository.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO


MODEL_CFG = (
    ROOT
    / "ultralytics"
    / "cfg"
    / "models"
    / "v8"
    / "litehand-yolov8n.yaml"
)


print("Ultralytics loaded from:")
print(ultralytics.__file__)

print("\nModel configuration:")
print(MODEL_CFG)

if not MODEL_CFG.exists():
    raise FileNotFoundError(
        f"LiteHand-YOLO model YAML not found: {MODEL_CFG}"
    )

model = YOLO(str(MODEL_CFG))

print("\nModel information:")
model.info()

x = torch.zeros(
    1,
    3,
    640,
    640
)

print("\nRunning forward pass...")

with torch.no_grad():
    output = model.model(x)

print("Forward pass successful")
print(f"Output type: {type(output)}")