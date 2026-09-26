from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[1]


MODEL_CFG = (
    ROOT
    /
    "ultralytics"
    /
    "cfg"
    /
    "models"
    /
    "v8"
    /
    "litehand-yolov8n.yaml"
)


print(MODEL_CFG)

model = YOLO(str(MODEL_CFG))

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
print(type(output))