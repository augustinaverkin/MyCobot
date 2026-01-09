"""
Simple camera preview with background subtraction:
 - Loads background if saved; otherwise captures once and saves to disk.
 - Subtracts background to detect new objects in the ROI.
 - Marks contours and labels dominant color (red, green, blue, yellow).
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 550, 180, 1330, 980
BACKGROUND_PATH = Path("background_roi.npy")


def load_or_capture_background(cap: cv2.VideoCapture) -> np.ndarray:
    """Load background from disk; if missing, capture one frame (cropped) and save."""
    if BACKGROUND_PATH.exists():
        bg = np.load(BACKGROUND_PATH)
        print(f"Loaded background from {BACKGROUND_PATH} with shape {bg.shape}")
        return apply_gray_world(bg)

    print("Capturing new background frame...")
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to capture background frame.")
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    roi = apply_gray_world(roi)
    np.save(BACKGROUND_PATH, roi)
    print(f"Saved background to {BACKGROUND_PATH}")
    return roi


def build_color_masks(bgr_img: np.ndarray) -> dict:
    """Build HSV masks for red, yellow, green, blue."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    masks = {}
    # Hue ranges in degrees: red wraps, so two ranges
    red1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    masks["red"] = cv2.bitwise_or(red1, red2)
    masks["yellow"] = cv2.inRange(hsv, (16, 80, 50), (40, 255, 255))
    masks["green"] = cv2.inRange(hsv, (36, 60, 40), (85, 255, 255))
    masks["blue"] = cv2.inRange(hsv, (86, 60, 40), (140, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for k, m in masks.items():
        masks[k] = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        masks[k] = cv2.morphologyEx(masks[k], cv2.MORPH_CLOSE, kernel, iterations=1)
    return masks


def classify_color_hist(bgr_img: np.ndarray, mask: np.ndarray) -> str:
    """Classify dominant color using hue histogram on masked region."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][mask > 0]
    s = hsv[..., 1][mask > 0]
    v = hsv[..., 2][mask > 0]

    if h.size == 0:
        return "unknown"

    valid = (s > 50) & (v > 50)
    if valid.sum() < 20:
        return "unknown"

    h_valid = h[valid]
    hist, _ = np.histogram(h_valid, bins=180, range=(0, 180))
    peak = int(np.argmax(hist))

    if peak >= 170 or peak <= 15:
        return "red"
    if 16 <= peak <= 35:
        return "yellow"
    if 36 <= peak <= 85:
        return "green"
    if 86 <= peak <= 140:
        return "blue"
    return "unknown"


def detect_changes(roi_bgr: np.ndarray, bg_roi: np.ndarray) -> Tuple[np.ndarray, list]:
    """Subtract background, filter by color masks, find contours, and annotate."""
    diff = cv2.absdiff(roi_bgr, bg_roi)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    color_masks = build_color_masks(roi_bgr)

    annotated = roi_bgr.copy()
    detections = []

    for label, c_mask in color_masks.items():
        combined = cv2.bitwise_and(fg, c_mask)
        # Smooth the combined mask to reduce fragmented contours
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10000:  # quick reject before size check
                continue
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(annotated, [box], -1, (0, 255, 255), 2)

            cx_rect, cy_rect = map(int, rect[0])
            cv2.circle(annotated, (cx_rect, cy_rect), 6, (0, 255, 0), -1)
            cv2.drawMarker(annotated, (cx_rect, cy_rect), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
            cv2.putText(annotated, label, (cx_rect + 8, cy_rect - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            w, h = rect[1]
            if min(w, h) >= 180:  # expect ~200x200 with some tolerance
                detections.append({"center": (cx_rect, cy_rect), "label": label, "area": area, "size": (w, h)})

    return annotated, detections


def apply_gray_world(bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world white balance to reduce color cast from lighting."""
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray_mean = means.mean()
    scale = gray_mean / means
    balanced = img * scale
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced


def configure_white_balance(cap: cv2.VideoCapture) -> None:
    """Disable auto white balance and set a fixed temperature (if supported)."""
    # Some drivers ignore these; we set and then read back to report.
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 10000)
        awb = cap.get(cv2.CAP_PROP_AUTO_WB)
        wb_temp = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        print(f"White balance: auto={awb} temp={wb_temp}")
    except Exception as exc:
        print(f"White balance control not supported: {exc}")


def open_camera(index: int, width: int, height: int) -> None:
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"Could not open camera index {index}", file=sys.stderr)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    configure_white_balance(cap)

    ok, frame = cap.read()
    actual_w = actual_h = 0
    if ok and frame is not None:
        actual_h, actual_w = frame.shape[:2]
    else:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_w == 0 or actual_h == 0:
        print("Failed to get a valid frame from the camera.", file=sys.stderr)
        cap.release()
        return

    window_name = "MyCobot Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"Camera opened on index {index} at {actual_w}x{actual_h}")
    print(f"Cropping ROI: x={ROI_X1}..{ROI_X2}, y={ROI_Y1}..{ROI_Y2}")
    print(f"Background file: {BACKGROUND_PATH} (delete it to recapture)")
    print("Press 'q' or ESC to quit.")

    bg_roi = load_or_capture_background(cap)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame", file=sys.stderr)
                break
            roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]  # y1:y2, x1:x2
            if roi.size == 0 or bg_roi.shape != roi.shape:
                print("ROI shape mismatch or empty. Check resolution or crop bounds.", file=sys.stderr)
                break

            roi_balanced = apply_gray_world(roi)

            annotated, detections = detect_changes(roi_balanced, bg_roi)
            for det in detections:
                cx, cy = det["center"]
                gx, gy = cx + ROI_X1, cy + ROI_Y1
                cv2.putText(annotated, f"ROI: ({cx},{cy}) Frame: ({gx},{gy})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview webcam feed with background subtraction.")
    parser.add_argument("--camera", type=int, default=2, help="Camera index (default: 2)")
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Requested frame width (default: 1920)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Requested frame height (default: 1080)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    open_camera(args.camera, args.width, args.height)


if __name__ == "__main__":
    main()
