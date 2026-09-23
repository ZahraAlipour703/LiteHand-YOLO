import torch

from ultralytics import YOLO
from ultralytics.nn.modules import ECA, CoordAtt


def main():
    print("Testing LiteHand-YOLO")
    print("=" * 60)

    # 1. Check custom modules
    print("\n[1] Custom modules")
    print("ECA:", ECA)
    print("CoordAtt:", CoordAtt)

    # 2. Build model from YAML
    print("\n[2] Building model")

    model = YOLO("Litehand-yolov8n.yaml")

    print("Model created successfully.")

    # 3. Print model information
    print("\n[3] Model information")
    model.info()

    # 4. Test a forward pass
    print("\n[4] Forward pass")

    x = torch.zeros(
        1,
        3,
        640,
        640,
    )

    with torch.no_grad():
        output = model.model(x)

    print("Forward pass successful.")
    print("Output type:", type(output))

    print("\nLiteHand-YOLO installation test PASSED.")


if __name__ == "__main__":
    main()