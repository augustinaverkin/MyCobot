import math
from dataclasses import dataclass
from time import sleep
from typing import Sequence, Tuple, List, Dict

import cv2
import numpy as np
from pymycobot import MyCobot280

# Reuse camera utilities
from camera_preview import (
    ROI_X1,
    ROI_Y1,
    ROI_X2,
    ROI_Y2,
    configure_white_balance,
    load_or_capture_background,
    apply_gray_world,
    detect_changes,
)

PORT = "/dev/ttyACM0"  # Ubuntu device
BAUD = 115200          # Confirmed good for your robot

# Motion parameters
APPROACH_OFFSET = 30.0      # mm above pick/place before descending
LIFT_Z = 220.0              # mm clearance after picking before moving laterally
MOVE_SPEED = 80         # 1..100, keep conservative
READY_ANGLES = [0, 15, -15, 30, 0, 0]
READY_COORDS = [73.2, -48.5, 284.9, 175.53, -3.18, -95.78]  # provided ready pose
PICK_HEIGHT = 105.0
PICK_ORIENTATION = (175, 0, -95)  # rx, ry, rz for pickup
DEFAULT_ORIENTATION = PICK_ORIENTATION
LINEAR_MODE = 1
Z_MIN = -60.0
Z_MAX = 230.0  # keep well below the head to avoid self-collision
STEP_MM = 15  # step size for incremental moves (unused in continuous mode)
TOOL_Z_OFFSET = 0  # gripper tip is 85mm below flange

# Affine transform from camera pixels -> robot (x, y) mm.
# Computed from correspondences:
# (120,652)->(211,-54), (671,658)->(211,60), (579,274)->(142,47)
PIX_TO_ROBOT_A = np.array(
    [
        [-0.00196178779, 0.180157512],
        [0.207068122, -0.0157559043],
    ],
    dtype=float,
)
PIX_TO_ROBOT_B = np.array([93.77271693, -68.57532507], dtype=float)

# Drop zones by color; default uses DROP_BASE with optional spacing.
DROP_BASE = (150.0, -150.0, 120.0)
COLOR_DROPS = {
    "red": (150.0, -150.0, 120.0),
    "green": (160.0, 160.0, 120.0),
    "blue": (250.0, -180.0, 120.0),
    "yellow": (30, 169, 120.0),
}
DROP_SPACING = 0  # mm shift between dropped cubes


@dataclass
class Pose:
    x: float
    y: float
    z: float


def clamp_speed(speed: int) -> int:
    return max(1, min(100, speed))


def get_coords_safe(mc: MyCobot280, attempts: int = 3, delay: float = 0.1) -> list | None:
    """Read coords with a few retries; return None on failure."""
    for _ in range(attempts):
        try:
            coords = mc.get_coords()
            if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                return coords
        except Exception:
            pass
        sleep(delay)
    return None


def pixels_to_robot(x_pix: float, y_pix: float) -> Tuple[float, float]:
    """Convert ROI pixel coords (as provided in calibration) to robot XY using affine calibration."""
    vec = np.array([x_pix, y_pix], dtype=float)
    x_robot, y_robot = PIX_TO_ROBOT_A @ vec + PIX_TO_ROBOT_B
    return float(x_robot), float(y_robot)


def compute_drop_pose(idx: int, color: str) -> Pose:
    """Choose drop pose based on color, with optional spacing per color."""
    base = COLOR_DROPS.get(color.lower(), DROP_BASE)
    dx = (idx % 4) * DROP_SPACING
    dy = (idx // 4) * DROP_SPACING
    return Pose(base[0] + dx, base[1] + dy, base[2])


def fetch_cubes_from_camera(camera_index: int = 2, width: int = 1920, height: int = 1080) -> List[Dict]:
    """Capture a frame, subtract background, and return detected cubes with global pixel coords and color."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    configure_white_balance(cap)

    bg_roi = load_or_capture_background(cap)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Failed to read frame from camera")

    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    if roi.size == 0 or roi.shape != bg_roi.shape:
        cap.release()
        raise RuntimeError("ROI shape mismatch; check resolution or crop bounds.")

    roi_balanced = apply_gray_world(roi)
    annotated, detections = detect_changes(roi_balanced, bg_roi)

    results = []
    for det in detections:
        cx, cy = det["center"]
        results.append(
            {
                "pixel_center": (cx, cy),
                "label": det.get("label", "unknown"),
                "area": det.get("area"),
                "size": det.get("size"),
            }
        )

    cap.release()
    cv2.destroyAllWindows()
    return results


def set_vacuum(mc: MyCobot280, enabled: bool) -> None:
    """Control the vacuum on pins 2/5."""
    mc.set_basic_output(2, 1 if enabled else 0)
    mc.set_basic_output(5, 0 if enabled else 1)
    print(f"Vacuum {'ON' if enabled else 'OFF'}")


def resolve_orientation(mc: MyCobot280, user_orientation: Sequence[float] | None = None) -> Tuple[float, float, float]:
    """Always return the target pickup orientation unless explicitly overridden."""
    if user_orientation is not None:
        return (float(user_orientation[0]), float(user_orientation[1]), float(user_orientation[2]))
    return DEFAULT_ORIENTATION


def sync_move_coords(
    mc: MyCobot280,
    coords: Sequence[float],
    speed: int = MOVE_SPEED,
    desc: str = "",
    mode: int = LINEAR_MODE,
) -> None:
    if desc:
        print(desc, [round(c, 2) for c in coords], f"@ speed {speed}")
    mc.sync_send_coords(coords, clamp_speed(speed), mode=mode)
    sleep(0.1)
    try:
        actual = get_coords_safe(mc)
        if isinstance(actual, (list, tuple)) and len(actual) >= 3:
            print("  Executed ->", [round(c, 2) for c in actual])
        else:
            print("  Executed ->", actual)
    except Exception as exc:
        print("  Failed to read coords after move:", exc)


def build_coords(pose: Pose, orientation: Sequence[float], z_override: float | None = None) -> list[float]:
    z_value = pose.z if z_override is None else z_override
    return [pose.x, pose.y, z_value, *orientation]


def force_pick_orientation(mc: MyCobot280, speed: int = MOVE_SPEED) -> None:
    """Set current pose to use the pickup orientation (keep XYZ)."""
    coords = mc.get_coords()
    if not coords or len(coords) < 6:
        print("Cannot read current coords to set orientation.")
        return
    target = [
        coords[0],
        coords[1],
        np.clip(coords[2], Z_MIN, Z_MAX),
        *PICK_ORIENTATION,
    ]
    sync_move_coords(mc, target, speed, "Align orientation to pickup", mode=LINEAR_MODE)


def move_axis_to(
    mc: MyCobot280,
    orientation: Sequence[float],
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    speed: int = MOVE_SPEED,
    desc: str = "",
) -> None:
    """Move only the specified axis (or axes) while keeping orientation fixed."""
    cur = get_coords_safe(mc)
    if not cur:
        raise RuntimeError("Unable to read current coordinates.")
    tx = cur[0] if x is None else x
    ty = cur[1] if y is None else y
    tz_raw = cur[2] if z is None else z + TOOL_Z_OFFSET
    tz = np.clip(tz_raw, Z_MIN, Z_MAX)
    waypoint = [
        tx,
        ty,
        tz,
        orientation[0],
        orientation[1],
        orientation[2],
    ]
    sync_move_coords(mc, waypoint, speed, desc, mode=LINEAR_MODE)


def move_z_force(mc: MyCobot280, z_target: float, orientation: Sequence[float], speed: int = MOVE_SPEED, desc: str = "") -> None:
    """Move only Z to the target, keeping current X/Y and the given orientation."""
    cur = get_coords_safe(mc)
    if not cur:
        print("  Warning: unable to read coords for Z move; skipping.")
        return
    tz = float(np.clip(z_target + TOOL_Z_OFFSET, Z_MIN, Z_MAX))
    waypoint = [
        cur[0],
        cur[1],
        tz,
        orientation[0],
        orientation[1],
        orientation[2],
    ]
    sync_move_coords(mc, waypoint, speed, desc, mode=LINEAR_MODE)


def configure_motion(mc: MyCobot280) -> None:
    """Ensure we use consistent settings and clear stale queue."""
    try:
        mc.clear_queue()
    except Exception:
        pass
    mc.set_fresh_mode(0)       # execute commands in order (avoid skipping early moves)
    mc.set_reference_frame(0)  # base frame
    mc.set_end_type(1)         # tool coordinates
    mc.set_movement_type(1)    # linear moves


def pick_and_place(
    mc: MyCobot280,
    pick_pose: Pose,
    drop_pose: Pose,
    *,
    orientation: Sequence[float] | None = None,
    approach_offset: float = APPROACH_OFFSET,
    lift_z: float = LIFT_Z,
    speed: int = MOVE_SPEED,
    settle_s: float = 0.05,
    dry_run: bool = False,
) -> None:
    """Full skill: approach, pick with vacuum, lift, move, place, and retreat."""
    if orientation is None:
        orientation = PICK_ORIENTATION
    orientation = resolve_orientation(mc, orientation)

    safe_z = min(
        Z_MAX,
        max(lift_z, pick_pose.z + abs(approach_offset), drop_pose.z + abs(approach_offset)),
    )

    pick_above = build_coords(pick_pose, orientation, z_override=safe_z)
    pick_point = build_coords(pick_pose, orientation)
    drop_above = build_coords(drop_pose, orientation, z_override=safe_z)
    drop_point = build_coords(drop_pose, orientation)

    plan = [
        ("Move above pick", pick_above),
        ("Descend to pick", pick_point),
        ("Lift after pick", pick_above),
        ("Move above drop", drop_above),
        ("Descend to drop", drop_point),
        ("Retreat from drop", drop_above),
    ]

    print("Planned sequence:")
    for label, coords in plan:
        print(f"  - {label}: {[round(c, 2) for c in coords]}")

    if dry_run:
        print("Dry run enabled: no motion executed.")
        return

    # Execute sequence with axis-separated moves
    # Move in X then Y to above pick (force Z to safe_z during translation)
    move_axis_to(mc, orientation, x=pick_pose.x, z=safe_z, speed=speed, desc="Move X to pick X")
    move_axis_to(mc, orientation, y=pick_pose.y, z=safe_z, speed=speed, desc="Move Y to pick Y")

    # Descend to pick height
    move_z_force(mc, pick_pose.z, orientation, speed=speed, desc="Descend to pick")
    set_vacuum(mc, True)
    sleep(settle_s)

    # Lift back up
    move_z_force(mc, safe_z, orientation, speed=speed, desc="Lift after pick")

    # Move to above drop: set Y first (safer with long gripper), then X, at safe_z
    move_axis_to(mc, orientation, y=drop_pose.y, z=safe_z, speed=speed, desc="Move Y to drop Y")
    move_axis_to(mc, orientation, x=drop_pose.x, z=safe_z, speed=speed, desc="Move X to drop X")

    # Descend to drop
    move_z_force(mc, drop_pose.z, orientation, speed=speed, desc="Descend to drop")

    set_vacuum(mc, False)
    sleep(settle_s)

    # Retreat up
    move_z_force(mc, safe_z, orientation, speed=speed, desc="Retreat from drop")


def main() -> None:
    print("Connecting to myCobot 280 M5 2023...")
    mc = MyCobot280(PORT, BAUD)

    try:
        mc.power_on()
        sleep(1)
        set_vacuum(mc, False)  # ensure vacuum starts off
        configure_motion(mc)

        print("Moving to ready pose...")
        sync_move_coords(mc, READY_COORDS, MOVE_SPEED, "Move to ready coords", mode=LINEAR_MODE)
        sleep(1)
        force_pick_orientation(mc, speed=MOVE_SPEED)

        print("Capturing cubes from camera...")
        cubes = fetch_cubes_from_camera()
        if not cubes:
            print("No cubes detected.")
            return

        for idx, cube in enumerate(cubes):
            px, py = cube["pixel_center"]  # ROI coordinates
            color = cube.get("label", "unknown")
            x_robot, y_robot = pixels_to_robot(px, py)
            pick = Pose(x_robot, y_robot, PICK_HEIGHT)
            drop = compute_drop_pose(idx, color)

            print(f"[{idx}] Cube color={color} ROI=({px},{py}) -> robot=({x_robot:.1f},{y_robot:.1f})")
            pick_and_place(
                mc,
                pick_pose=pick,
                drop_pose=drop,
                orientation=PICK_ORIENTATION,
                speed=MOVE_SPEED,
                dry_run=False,
            )

        print("All cubes moved to drop zone.")
    finally:
        try:
            set_vacuum(mc, False)
        except Exception as exc:
            print("Cleanup issue:", exc)


if __name__ == "__main__":
    main()
