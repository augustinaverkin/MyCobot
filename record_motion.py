"""Simple record/playback helper for myCobot 280 via pymycobot."""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Sequence, Optional
import threading
import select
import tty
import termios
from queue import SimpleQueue

from pymycobot import MyCobot280

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
LINEAR_MODE = 1
SAFE_Z_MIN = -60.0
SAFE_Z_MAX = 350.0
REALTIME_DT = 0.05  # default streaming interval (s) for real-time mode (20 Hz)
VAC_PIN_ON = 2
VAC_PIN_OFF = 5


def clamp_speed(speed: int) -> int:
    return max(1, min(100, int(speed)))


def set_vacuum(mc: MyCobot280, enabled: bool) -> None:
    """Control the vacuum on pins 2/5."""
    mc.set_basic_output(VAC_PIN_ON, 1 if enabled else 0)
    mc.set_basic_output(VAC_PIN_OFF, 0 if enabled else 1)
    print(f"Vacuum {'ON' if enabled else 'OFF'}")


def sanitize_coords(coords: Sequence[float], z_min: float, z_max: float) -> list[float] | None:
    """Validate coords and clamp Z into bounds; return None if unusable."""
    if not isinstance(coords, (list, tuple)):
        return None
    if len(coords) < 6:
        return None
    try:
        vals = [float(c) for c in coords[:6]]
    except (TypeError, ValueError):
        return None
    vals[2] = max(z_min, min(z_max, vals[2]))
    return vals


def resample_timeline(
    timeline: list[tuple[float, Sequence[float], Optional[bool]]], dt: float
) -> list[tuple[float, list[float], Optional[bool]]]:
    """Linearly interpolate poses to a fixed timestep; always include original keyframes and vacuum events."""
    if not timeline or dt <= 0:
        return []
    resampled: list[tuple[float, list[float], Optional[bool]]] = []
    end_time = timeline[-1][0]

    times = set()
    if dt > 0:
        k = 0
        while True:
            t = round(k * dt, 6)
            if t > end_time + 1e-6:
                break
            times.add(t)
            k += 1
    orig_vac: dict[float, Optional[bool]] = {}
    for t, _, vac in timeline:
        times.add(float(t))
        orig_vac[float(t)] = vac

    sorted_times = sorted(times)
    idx = 0
    n = len(timeline)

    for t in sorted_times:
        while idx + 1 < n and timeline[idx + 1][0] < t - 1e-9:
            idx += 1
        if idx + 1 >= n:
            resampled.append((t, [float(c) for c in timeline[-1][1][:6]], orig_vac.get(t)))
            continue
        t0, c0, _ = timeline[idx]
        t1, c1, _ = timeline[idx + 1]
        span = max(1e-6, t1 - t0)
        r = max(0.0, min(1.0, (t - t0) / span))
        interp = [float(c0[i]) + (float(c1[i]) - float(c0[i])) * r for i in range(6)]
        resampled.append((t, interp, orig_vac.get(t)))

    return resampled


def configure_motion(mc: MyCobot280) -> None:
    """Set consistent motion/frame settings and clear any queued commands."""
    try:
        mc.clear_queue()
    except Exception:
        pass
    mc.set_fresh_mode(0)
    mc.set_reference_frame(0)
    mc.set_end_type(1)
    mc.set_movement_type(1)


def record_motion(
    mc: MyCobot280,
    duration_s: float,
    interval_s: float,
    output: Path,
    *,
    drag_teach: bool = False,
    stop_event: threading.Event | None = None,
) -> None:
    """Sample tool coords over time and save to disk."""
    mc.power_on()
    configure_motion(mc)
    if drag_teach:
        try:
            mc.release_all_servos()
            print("Servos released for drag-teach. Move the arm by hand now.")
        except Exception as exc:
            print(f"Could not release servos for drag-teach: {exc}")

    start = time.time()
    end_time = start + duration_s if duration_s > 0 else float("inf")
    raw_samples: List[dict] = []
    warn_count = 0

    try:
        while time.time() < end_time:
            if stop_event and stop_event.is_set():
                print("Recording stopped by user.")
                break
            now = time.time()
            coords = None
            try:
                coords = mc.get_coords()
            except Exception:
                coords = None

            if isinstance(coords, (list, tuple)) and len(coords) >= 6:
                samples.append(
                    {"t": round(now - start, 3), "coords": [float(c) for c in coords[:6]]}
                )
            else:
                if warn_count < 5:
                    print("Warning: failed to read coords; skipping sample.")
                    warn_count += 1
            next_tick = now + interval_s
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("Recording stopped early by user.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Recorded {len(samples)} samples to {output.resolve()}")

    if drag_teach:
        print("Re-engaging servos...")
        mc.power_on()
        configure_motion(mc)


def record_main_steps(
    mc: MyCobot280,
    duration_s: float,
    poll_interval_s: float,
    output: Path,
    *,
    drag_teach: bool = False,
    stop_event: threading.Event | None = None,
    capture_event: threading.Event | None = None,
    step_spacing_s: float = 0.05,
    action_queue: SimpleQueue | None = None,
) -> None:
    """Record key poses when user presses 's' (or via capture_event). Timestamps are rebuilt at fixed spacing."""
    mc.power_on()
    configure_motion(mc)
    if drag_teach:
        try:
            mc.release_all_servos()
            print("Servos released for drag-teach. Move the arm by hand now.")
        except Exception as exc:
            print(f"Could not release servos for drag-teach: {exc}")

    print("Main steps mode:")
    print("  - Press 's' in the terminal to capture a key pose (first capture becomes start).")
    print("  - Press 'v' to capture + Vacuum ON, 'b' to capture + Vacuum OFF.")
    print("  - Press 'q' to stop early.")
    print("  - Or click the GUI 'Mark step' / 'Stop' if using the GUI.")
    print("  - Duration<=0 keeps listening until Stop/`q`.")

    start = time.time()
    end_time = start + duration_s if duration_s > 0 else float("inf")
    raw_samples: List[dict] = []
    warn_count = 0

    def capture_pose(tag: str = "key", vacuum: Optional[bool] = None) -> None:
        nonlocal warn_count
        coords = None
        try:
            coords = mc.get_coords()
        except Exception:
            coords = None
        if isinstance(coords, (list, tuple)) and len(coords) >= 6:
            raw_samples.append(
                {
                    "t": round(time.time() - start, 3),
                    "coords": [float(c) for c in coords[:6]],
                    "tag": tag,
                    "vacuum": vacuum,
                }
            )
            print(f"Captured {tag} step: {raw_samples[-1]['coords']}")
        else:
            if warn_count < 5:
                print("Warning: failed to read coords; key step not captured.")
                warn_count += 1

    def process_action(action: str) -> None:
        if action == "capture":
            capture_pose()
        elif action == "vac_on":
            try:
                set_vacuum(mc, True)
            except Exception as exc:
                print(f"Vacuum ON failed: {exc}")
            capture_pose(tag="vac_on", vacuum=True)
        elif action == "vac_off":
            try:
                set_vacuum(mc, False)
            except Exception as exc:
                print(f"Vacuum OFF failed: {exc}")
            capture_pose(tag="vac_off", vacuum=False)

    fd = None
    old_settings = None
    tty_ok = False
    try:
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            tty_ok = True
    except Exception:
        tty_ok = False
        print("Note: terminal raw input not available; use GUI 'Mark step' or Stop buttons.")

    try:
        while time.time() < end_time:
            if stop_event and stop_event.is_set():
                print("Recording stopped by Stop button.")
                break

            while action_queue and not action_queue.empty():
                try:
                    act = action_queue.get_nowait()
                except Exception:
                    act = None
                if act:
                    process_action(str(act))

            if capture_event and capture_event.is_set():
                capture_event.clear()
                process_action("capture")

            if tty_ok:
                r, _, _ = select.select([sys.stdin], [], [], poll_interval_s)
                if r:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "s":
                        process_action("capture")
                    elif ch.lower() == "v":
                        process_action("vac_on")
                    elif ch.lower() == "b":
                        process_action("vac_off")
                    elif ch.lower() == "q":
                        print("Recording stopped by 'q'.")
                        break
            else:
                time.sleep(poll_interval_s)
            # Otherwise loop continues until duration
    except KeyboardInterrupt:
        print("Recording stopped early by user.")
    finally:
        if tty_ok and fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if not raw_samples:
        print("No samples captured.")
    else:
        spacing = step_spacing_s if step_spacing_s > 0 else 0.05
        samples: List[dict] = []
        for idx, item in enumerate(raw_samples):
            samples.append(
                {
                    "t": round(idx * spacing, 3),
                    "coords": item["coords"],
                    "tag": item.get("tag", "key"),
                    "vacuum": item.get("vacuum"),
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        print(f"Recorded {len(samples)} key steps to {output.resolve()} with spacing {spacing}s")

    if drag_teach:
        print("Re-engaging servos...")
        mc.power_on()
        configure_motion(mc)


def normalize_samples(samples: List[dict]) -> List[tuple[float, Sequence[float], Optional[bool]]]:
    """Normalize timestamps so playback starts at t=0."""
    if not samples:
        return []
    t0 = float(samples[0].get("t", 0.0))
    normalized = []
    for item in samples:
        coords = item.get("coords", [])
        if not isinstance(coords, (list, tuple)) or len(coords) < 6:
            continue
        t_rel = float(item.get("t", 0.0)) - t0
        normalized.append(
            (max(0.0, t_rel), [float(c) for c in coords[:6]], item.get("vacuum"))
        )
    return normalized


def playback_motion(
    mc: MyCobot280,
    samples: List[dict],
    speed: int,
    loops: int,
    loop_delay: float,
    z_min: float = SAFE_Z_MIN,
    z_max: float = SAFE_Z_MAX,
    real_time: bool = False,
    real_time_dt: float = REALTIME_DT,
    stop_event: threading.Event | None = None,
) -> None:
    """Replay recorded coords with the given speed setting."""
    timeline = normalize_samples(samples)
    if not timeline:
        print("No valid samples to play.")
        return

    mc.power_on()
    configure_motion(mc)

    if real_time:
        skipped_bad = 0
        clamped = 0
        stream = resample_timeline(timeline, real_time_dt if real_time_dt > 0 else REALTIME_DT)
        if not stream:
            print("Playback: nothing to stream after resampling.")
            return

        for loop_idx in range(loops):
            if stop_event and stop_event.is_set():
                print("Playback stopped by user.")
                return

            first_coords = sanitize_coords(stream[0][1], z_min, z_max)
            if first_coords is None:
                print("Playback: first sample invalid; aborting.")
                return
            if first_coords[2] != stream[0][1][2]:
                clamped += 1

            if stream[0][2] is not None:
                try:
                    set_vacuum(mc, bool(stream[0][2]))
                except Exception as exc:
                    print(f"Vacuum set failed at start: {exc}")

            mc.sync_send_coords(first_coords, clamp_speed(speed), mode=LINEAR_MODE)

            loop_start = time.time()

            for t_rel, coords, vac in stream[1:]:
                if stop_event and stop_event.is_set():
                    print("Playback stopped by user.")
                    return
                target_time = loop_start + t_rel
                wait = target_time - time.time()
                if wait > 0:
                    time.sleep(wait)

                clean = sanitize_coords(coords, z_min, z_max)
                if clean is None:
                    skipped_bad += 1
                    continue
                if clean[2] != coords[2]:
                    clamped += 1
                if vac is not None:
                    mc.sync_send_coords(clean, clamp_speed(speed), mode=LINEAR_MODE)
                    try:
                        set_vacuum(mc, bool(vac))
                    except Exception as exc:
                        print(f"Vacuum set failed: {exc}")
                    continue
                mc.send_coords(clean, clamp_speed(speed), mode=LINEAR_MODE)

            if loop_idx < loops - 1 and loop_delay > 0:
                time.sleep(loop_delay)

        if skipped_bad:
            print(f"Playback: skipped {skipped_bad} invalid samples.")
        if clamped:
            print(f"Playback: clamped Z on {clamped} samples to [{z_min}, {z_max}].")
        return

    skipped = 0
    clamped = 0

    for loop_idx in range(loops):
        loop_start = time.time()
        for t_rel, coords, vac in timeline:
            if stop_event and stop_event.is_set():
                print("Playback stopped by user.")
                return
            target_time = loop_start + t_rel
            wait = target_time - time.time()
            if wait > 0:
                if stop_event and stop_event.is_set():
                    print("Playback stopped by user.")
                    return
                time.sleep(wait)
            clean = sanitize_coords(coords, z_min, z_max)
            if clean is None:
                skipped += 1
                continue
            if clean[2] != coords[2]:
                clamped += 1
            if vac is not None:
                mc.sync_send_coords(clean, clamp_speed(speed), mode=LINEAR_MODE)
                try:
                    set_vacuum(mc, bool(vac))
                except Exception as exc:
                    print(f"Vacuum set failed: {exc}")
                continue
            mc.sync_send_coords(clean, clamp_speed(speed), mode=LINEAR_MODE)
        if loop_idx < loops - 1 and loop_delay > 0:
            time.sleep(loop_delay)
    if skipped:
        print(f"Playback: skipped {skipped} samples with invalid coords.")
    if clamped:
        print(f"Playback: clamped Z on {clamped} samples to [{z_min}, {z_max}].")


def load_motion(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Motion file must contain a JSON list of samples.")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record and play myCobot motions.")
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Baud rate (default: {DEFAULT_BAUD})")

    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record tool coords over time.")
    rec.add_argument("--duration", type=float, default=20.0, help="Seconds to record (default: 20)")
    rec.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds (default: 0.1)")
    rec.add_argument("--output", type=Path, default=Path("motion.json"), help="Output JSON path (default: motion.json)")
    rec.add_argument(
        "--drag",
        action="store_true",
        help="Release servos for drag-teach (move arm by hand while sampling).",
    )

    rec_main = sub.add_parser("record-main", help="Record key steps only when you press 's' (keyframes).")
    rec_main.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Seconds to listen for key steps (0 or negative = no timeout; default: 60)",
    )
    rec_main.add_argument("--interval", type=float, default=0.05, help="Poll interval seconds (default: 0.05)")
    rec_main.add_argument("--output", type=Path, default=Path("motion.json"), help="Output JSON path (default: motion.json)")
    rec_main.add_argument(
        "--step-spacing",
        type=float,
        default=0.05,
        help="Spacing (s) between saved keyframes for playback timeline (default: 0.05).",
    )
    rec_main.add_argument(
        "--drag",
        action="store_true",
        help="Release servos for drag-teach (move arm by hand while sampling).",
    )

    play = sub.add_parser("play", help="Play a recorded motion file.")
    play.add_argument("--input", type=Path, default=Path("motion.json"), help="Input JSON path (default: motion.json)")
    play.add_argument("--speed", type=int, default=30, help="Speed 1-100 for playback (default: 30)")
    play.add_argument("--loops", type=int, default=1, help="Number of times to repeat playback (default: 1)")
    play.add_argument(
        "--loop-delay",
        type=float,
        default=0.5,
        help="Pause between loops in seconds (default: 0.5)",
    )
    play.add_argument("--z-min", type=float, default=SAFE_Z_MIN, help=f"Clamp Z minimum (default: {SAFE_Z_MIN})")
    play.add_argument("--z-max", type=float, default=SAFE_Z_MAX, help=f"Clamp Z maximum (default: {SAFE_Z_MAX})")
    play.add_argument(
        "--real-time",
        action="store_true",
        help="Stream poses continuously (first pose sync, then interpolated async stream).",
    )
    play.add_argument(
        "--real-time-dt",
        type=float,
        default=REALTIME_DT,
        help=f"Streaming interval (seconds) for real-time mode (default: {REALTIME_DT}, e.g., 0.05=20Hz).",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    mc = MyCobot280(args.port, args.baud)

    if args.command == "record":
        record_motion(
            mc,
            duration_s=args.duration,
            interval_s=args.interval,
            output=args.output,
            drag_teach=args.drag,
        )
    elif args.command == "record-main":
        record_main_steps(
            mc,
            duration_s=args.duration,
            poll_interval_s=args.interval,
            output=args.output,
            drag_teach=args.drag,
            stop_event=None,
            capture_event=None,
            step_spacing_s=args.step_spacing,
        )
    elif args.command == "play":
        samples = load_motion(args.input)
        playback_motion(
            mc,
            samples,
            speed=args.speed,
            loops=args.loops,
            loop_delay=args.loop_delay,
            z_min=args.z_min,
            z_max=args.z_max,
            real_time=args.real_time,
            real_time_dt=args.real_time_dt,
        )


if __name__ == "__main__":
    main()
