from pathlib import Path
import argparse
import sys

import cv2
import torch
from ultralytics import YOLO


# ------------------------------------------------------------------
# Repository setup
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# Use the repository's modified Ultralytics implementation.
LOCAL_ULTRALYTICS = REPO_ROOT / "yolov8-enhanced-structure"

if LOCAL_ULTRALYTICS.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def parse_source(source):
    """Convert numeric webcam source such as '0' into int."""
    try:
        return int(source)
    except ValueError:
        return source


def get_device(requested):
    """Resolve requested device."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    return requested


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Real-time inference with LiteHand-YOLO."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="weights/litehand_yolov8n.pt",
        help="Path to LiteHand-YOLO weights."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index, image path, or video path."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold."
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/demo",
        help="Output directory."
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------------

    model_path = REPO_ROOT / args.model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found:\n{model_path}\n\n"
            "Download the LiteHand-YOLO weights and place them in "
            "`weights/`, or pass --model PATH."
        )

    source = parse_source(args.source)
    device = get_device(args.device)

    print("=" * 60)
    print("LiteHand-YOLO Real-Time Demo")
    print("=" * 60)
    print(f"Model : {model_path}")
    print(f"Source: {source}")
    print(f"Device: {device}")
    print("=" * 60)

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    model = YOLO(str(model_path))
    model.to(device)

    # --------------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------------

    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        show=True,
        save=args.save,
        project=str(REPO_ROOT / args.output),
        verbose=False
    )

    return results


if __name__ == "__main__":
    main()