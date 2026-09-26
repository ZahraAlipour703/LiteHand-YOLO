from pathlib import Path
import argparse
import csv
import time

import cv2
import numpy as np
import torch
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = REPO_ROOT / "yolov8-enhanced-structure"

if LOCAL_ULTRALYTICS.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO


def synchronize(device):
    """Synchronize CUDA before timing GPU inference."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(model, frame, device, imgsz=640):
    """Run one inference and return latency + result."""
    synchronize(device)

    start = time.perf_counter()

    result = model.predict(
        frame,
        imgsz=imgsz,
        device=device,
        verbose=False
    )[0]

    synchronize(device)

    latency_ms = (time.perf_counter() - start) * 1000

    return latency_ms, result


def summarize_result(result):
    """
    Extract runtime-safe descriptive metrics.

    IMPORTANT:
    These are NOT precision or accuracy. They are runtime
    confidence statistics only.
    """
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "detections": 0,
            "mean_confidence": 0.0
        }

    conf = boxes.conf.detach().cpu().numpy()

    return {
        "detections": len(conf),
        "mean_confidence": float(np.mean(conf))
    }


def main():

    parser = argparse.ArgumentParser(
        description="Compare YOLOv8n and LiteHand-YOLO in real time."
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default="weights/yolov8n_hkd.pt"
    )

    parser.add_argument(
        "--litehand",
        type=str,
        default="weights/litehand_yolov8n.pt"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video path or webcam index."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=300
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/benchmark"
    )

    parser.add_argument(
        "--screenshot-interval",
        type=int,
        default=50
    )

    parser.add_argument(
        "--no-display",
        action="store_true"
    )

    args = parser.parse_args()

    baseline_path = REPO_ROOT / args.baseline
    litehand_path = REPO_ROOT / args.litehand

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline weights not found: {baseline_path}"
        )

    if not litehand_path.exists():
        raise FileNotFoundError(
            f"LiteHand weights not found: {litehand_path}"
        )

    device = args.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    source = int(args.source) if args.source.isdigit() else args.source

    output_dir = REPO_ROOT / args.output
    screenshot_dir = output_dir / "screenshots"

    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "realtime_comparison.csv"

    print("=" * 70)
    print("LiteHand-YOLO Real-Time Benchmark")
    print("=" * 70)
    print(f"Baseline : {baseline_path}")
    print(f"LiteHand : {litehand_path}")
    print(f"Source   : {source}")
    print(f"Device   : {device}")
    print("=" * 70)

    baseline = YOLO(str(baseline_path))
    litehand = YOLO(str(litehand_path))

    baseline.to(device)
    litehand.to(device)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    baseline_times = []
    litehand_times = []

    header = [
        "frame",
        "baseline_latency_ms",
        "baseline_fps",
        "baseline_detections",
        "baseline_mean_confidence",
        "litehand_latency_ms",
        "litehand_fps",
        "litehand_detections",
        "litehand_mean_confidence"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)
        writer.writerow(header)

        frame_id = 0

        while frame_id < args.frames:

            ret, frame = cap.read()

            if not ret:
                break

            frame_id += 1

            frame = cv2.resize(frame, (640, 480))

            # ------------------------------------------------------
            # Baseline
            # ------------------------------------------------------

            baseline_latency, baseline_result = benchmark_model(
                baseline,
                frame,
                device,
                args.imgsz
            )

            # ------------------------------------------------------
            # LiteHand
            # ------------------------------------------------------

            litehand_latency, litehand_result = benchmark_model(
                litehand,
                frame,
                device,
                args.imgsz
            )

            baseline_times.append(baseline_latency)
            litehand_times.append(litehand_latency)

            baseline_fps = (
                1000.0 / np.mean(baseline_times[-30:])
            )

            litehand_fps = (
                1000.0 / np.mean(litehand_times[-30:])
            )

            baseline_stats = summarize_result(baseline_result)
            litehand_stats = summarize_result(litehand_result)

            # ------------------------------------------------------
            # Visualization
            # ------------------------------------------------------

            baseline_vis = baseline_result.plot()
            litehand_vis = litehand_result.plot()

            cv2.putText(
                baseline_vis,
                f"YOLOv8n | {baseline_fps:.2f} FPS | "
                f"{baseline_latency:.1f} ms",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                litehand_vis,
                f"LiteHand | {litehand_fps:.2f} FPS | "
                f"{litehand_latency:.1f} ms",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 200, 0),
                2
            )

            combined = np.hstack([
                baseline_vis,
                litehand_vis
            ])

            # ------------------------------------------------------
            # Screenshots
            # ------------------------------------------------------

            if frame_id % args.screenshot_interval == 0:

                cv2.imwrite(
                    str(
                        screenshot_dir /
                        f"frame_{frame_id:04d}_comparison.jpg"
                    ),
                    combined
                )

            # ------------------------------------------------------
            # Display
            # ------------------------------------------------------

            if not args.no_display:

                cv2.imshow(
                    "YOLOv8n vs LiteHand-YOLO",
                    combined
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ------------------------------------------------------
            # CSV
            # ------------------------------------------------------

            writer.writerow([
                frame_id,
                round(baseline_latency, 3),
                round(baseline_fps, 3),
                baseline_stats["detections"],
                round(baseline_stats["mean_confidence"], 4),

                round(litehand_latency, 3),
                round(litehand_fps, 3),
                litehand_stats["detections"],
                round(litehand_stats["mean_confidence"], 4)
            ])

            if frame_id % 20 == 0:

                print(
                    f"Frame {frame_id:03d} | "
                    f"YOLOv8n {baseline_fps:.2f} FPS | "
                    f"LiteHand {litehand_fps:.2f} FPS"
                )

    cap.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    baseline_mean = np.mean(baseline_times)
    litehand_mean = np.mean(litehand_times)

    baseline_fps = 1000.0 / baseline_mean
    litehand_fps = 1000.0 / litehand_mean

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"YOLOv8n       : "
        f"{baseline_mean:.2f} ms/frame | "
        f"{baseline_fps:.2f} FPS"
    )

    print(
        f"LiteHand-YOLO : "
        f"{litehand_mean:.2f} ms/frame | "
        f"{litehand_fps:.2f} FPS"
    )

    fps_change = (
        (litehand_fps - baseline_fps)
        / baseline_fps
        * 100
    )

    latency_change = (
        (litehand_mean - baseline_mean)
        / baseline_mean
        * 100
    )

    print(f"\nFPS change     : {fps_change:+.2f}%")
    print(f"Latency change : {latency_change:+.2f}%")
    print(f"\nCSV            : {csv_path}")
    print(f"Screenshots    : {screenshot_dir}")


if __name__ == "__main__":
    main()