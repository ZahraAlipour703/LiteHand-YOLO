from pathlib import Path
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import ECA, CoordAtt


MODEL_CFG = (
    Path(__file__).parents[1]
    /
    "yolov8-enhanced-structure"
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


def main():

    print("=" * 60)
    print("LiteHand-YOLO verification")
    print("=" * 60)


    print("\n[1] Checking modules")

    print(ECA)
    print(CoordAtt)


    print("\n[2] Loading YAML")

    model = YOLO(str(MODEL_CFG))

    print("Model loaded successfully")


    print("\n[3] Model summary")

    model.info()


    print("\n[4] Forward test")

    x = torch.zeros(
        1,
        3,
        640,
        640
    )

    with torch.no_grad():

        output = model.model(x)


    print("Forward pass OK")
    print(type(output))


    print("\nLiteHand-YOLO READY")


if __name__ == "__main__":
    main()