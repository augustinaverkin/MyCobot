"""Tiny Tk GUI to record and play back myCobot motions."""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from queue import SimpleQueue

from record_motion import (
    DEFAULT_BAUD,
    DEFAULT_PORT,
    MyCobot280,
    SAFE_Z_MIN,
    SAFE_Z_MAX,
    playback_motion,
    record_motion,
    record_main_steps,
    load_motion,
    set_vacuum,
)


class MotionGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("myCobot Record/Play")
        self.mc: MyCobot280 | None = None
        self.mc_port = DEFAULT_PORT
        self.mc_baud = DEFAULT_BAUD
        self._build_ui()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.port_var = tk.StringVar(value=DEFAULT_PORT)
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        self.duration_var = tk.StringVar(value="20.0")
        self.interval_var = tk.StringVar(value="0.1")
        self.output_var = tk.StringVar(value="motion.json")
        self.input_var = tk.StringVar(value="motion.json")
        self.speed_var = tk.StringVar(value="30")
        self.loops_var = tk.StringVar(value="1")
        self.delay_var = tk.StringVar(value="0.5")
        self.drag_var = tk.BooleanVar(value=False)
        self.stop_event = None
        self.capture_event = None
        self.zmin_var = tk.StringVar(value=str(SAFE_Z_MIN))
        self.zmax_var = tk.StringVar(value=str(SAFE_Z_MAX))
        self.realtime_var = tk.BooleanVar(value=False)
        self.realtime_dt_var = tk.StringVar(value="0.05")
        self.step_spacing_var = tk.StringVar(value="0.05")

        row = 0
        ttk.Label(frm, text="Port").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.port_var, width=18).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(frm, text="Baud").grid(row=row, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.baud_var, width=10).grid(row=row, column=3, sticky="w", padx=4, pady=2)

        row += 1
        ttk.Label(frm, text="Record duration (s)").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.duration_var, width=10).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(frm, text="Interval (s)").grid(row=row, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.interval_var, width=10).grid(row=row, column=3, sticky="w", padx=4, pady=2)

        row += 1
        ttk.Label(frm, text="Record file").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.output_var, width=30).grid(row=row, column=1, columnspan=2, sticky="we", padx=4, pady=2)
        ttk.Button(frm, text="Browse", command=self._choose_output).grid(row=row, column=3, sticky="w", padx=4, pady=2)

        row += 1
        ttk.Label(frm, text="Play file").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.input_var, width=30).grid(row=row, column=1, columnspan=2, sticky="we", padx=4, pady=2)
        ttk.Button(frm, text="Browse", command=self._choose_input).grid(row=row, column=3, sticky="w", padx=4, pady=2)

        row += 1
        ttk.Label(frm, text="Play speed (1-100)").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.speed_var, width=10).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(frm, text="Loops").grid(row=row, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.loops_var, width=10).grid(row=row, column=3, sticky="w", padx=4, pady=2)

        row += 1
        ttk.Label(frm, text="Loop delay (s)").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.delay_var, width=10).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(frm, text="Drag teach (release servos during record)", variable=self.drag_var).grid(
            row=row, column=2, columnspan=2, sticky="w", padx=4, pady=2
        )

        row += 1
        ttk.Label(frm, text="Z min").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.zmin_var, width=10).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(frm, text="Z max").grid(row=row, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.zmax_var, width=10).grid(row=row, column=3, sticky="w", padx=4, pady=2)

        row += 1
        ttk.Checkbutton(frm, text="Play continuous (first pose sync, interpolated stream)", variable=self.realtime_var).grid(
            row=row, column=0, columnspan=4, sticky="w", padx=4, pady=2
        )

        row += 1
        ttk.Label(frm, text="Stream interval (s) in real-time").grid(row=row, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.realtime_dt_var, width=10).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(frm, text="Keyframe spacing (s) on save").grid(row=row, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(frm, textvariable=self.step_spacing_var, width=10).grid(row=row, column=3, sticky="w", padx=4, pady=2)

        row += 1
        self.record_btn = ttk.Button(frm, text="Record", command=self._on_record)
        self.record_btn.grid(row=row, column=0, columnspan=2, sticky="we", padx=4, pady=6)
        self.play_btn = ttk.Button(frm, text="Play", command=self._on_play)
        self.play_btn.grid(row=row, column=2, columnspan=2, sticky="we", padx=4, pady=6)
        row += 1
        self.record_main_btn = ttk.Button(frm, text="Record main steps", command=self._on_record_main)
        self.record_main_btn.grid(row=row, column=0, columnspan=2, sticky="we", padx=4, pady=2)
        self.mark_step_btn = ttk.Button(frm, text="Mark step (keyframe)", command=self._on_mark_step, state="disabled")
        self.mark_step_btn.grid(row=row, column=2, columnspan=2, sticky="we", padx=4, pady=2)
        row += 1
        self.vac_on_btn = ttk.Button(frm, text="Vacuum ON (record)", command=lambda: self._on_vacuum(True), state="disabled")
        self.vac_on_btn.grid(row=row, column=0, columnspan=2, sticky="we", padx=4, pady=2)
        self.vac_off_btn = ttk.Button(frm, text="Vacuum OFF (record)", command=lambda: self._on_vacuum(False), state="disabled")
        self.vac_off_btn.grid(row=row, column=2, columnspan=2, sticky="we", padx=4, pady=2)
        row += 1
        self.stop_btn = ttk.Button(frm, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.grid(row=row, column=0, columnspan=4, sticky="we", padx=4, pady=2)

        row += 1
        self.status = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status, foreground="blue").grid(row=row, column=0, columnspan=4, sticky="w", padx=4, pady=4)

        row += 1
        self.log = tk.Text(frm, height=10, width=60, state="disabled")
        self.log.grid(row=row, column=0, columnspan=4, sticky="nsew", padx=4, pady=4)
        frm.rowconfigure(row, weight=1)
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(2, weight=1)

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.output_var.set(path)

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.input_var.set(path)

    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.root.update_idletasks()

    def _set_busy(self, busy: bool, label: str = "") -> None:
        self.record_btn.configure(state="disabled" if busy else "normal")
        self.play_btn.configure(state="disabled" if busy else "normal")
        self.record_main_btn.configure(state="disabled" if busy else "normal")
        active = bool(busy and self.capture_event)
        self.mark_step_btn.configure(state="normal" if active else "disabled")
        self.vac_on_btn.configure(state="normal" if active else "disabled")
        self.vac_off_btn.configure(state="normal" if active else "disabled")
        self.stop_btn.configure(state="normal" if busy else "disabled")
        if label:
            self.status.set(label)
        elif not busy:
            self.status.set("Idle")

    def _ensure_mc(self) -> bool:
        port = self.port_var.get() or DEFAULT_PORT
        try:
            baud = int(self.baud_var.get())
        except ValueError:
            messagebox.showerror("Invalid baud", "Baud rate must be an integer.")
            return False

        if self.mc and port == self.mc_port and baud == self.mc_baud:
            return True

        try:
            self.mc = MyCobot280(port, baud)
            self.mc_port, self.mc_baud = port, baud
            self._log(f"Connected to {port} @ {baud}")
            return True
        except Exception as exc:
            messagebox.showerror("Connection failed", str(exc))
            return False

    def _on_record(self) -> None:
        if not self._ensure_mc():
            return
        try:
            duration = float(self.duration_var.get())
            interval = float(self.interval_var.get())
        except ValueError:
            messagebox.showerror("Invalid timing", "Duration and interval must be numbers.")
            return

        output = Path(self.output_var.get())
        drag = bool(self.drag_var.get())
        self.stop_event = threading.Event()
        self.capture_event = threading.Event()
        self.action_queue = SimpleQueue()

        def worker() -> None:
            self._set_busy(True, "Recording...")
            try:
                record_motion(
                    self.mc,
                    duration_s=duration,
                    interval_s=interval,
                    output=output,
                    drag_teach=drag,
                    stop_event=self.stop_event,
                )
                self._log(f"Saved recording to {output}")
            except Exception as exc:
                self._log(f"Record error: {exc}")
            finally:
                self.action_queue = None
                self.capture_event = None
                self.stop_event = None
                self._set_busy(False, "Idle")

        threading.Thread(target=worker, daemon=True).start()

    def _on_play(self) -> None:
        if not self._ensure_mc():
            return
        try:
            speed = int(self.speed_var.get())
            loops = int(self.loops_var.get())
            loop_delay = float(self.delay_var.get())
            z_min = float(self.zmin_var.get())
            z_max = float(self.zmax_var.get())
            realtime_dt = float(self.realtime_dt_var.get())
        except ValueError:
            messagebox.showerror("Invalid playback values", "Speed, loops, loop delay, Z limits, and stream interval must be numbers.")
            return
        if z_min >= z_max:
            messagebox.showerror("Invalid Z limits", "Z min must be less than Z max.")
            return
        if realtime_dt <= 0:
            messagebox.showerror("Invalid interval", "Stream interval must be > 0.")
            return
        realtime = bool(self.realtime_var.get())

        input_path = Path(self.input_var.get())
        if not input_path.exists():
            messagebox.showerror("Missing file", f"Cannot find {input_path}")
            return
        self.stop_event = threading.Event()
        self.capture_event = None

        def worker() -> None:
            self._set_busy(True, "Playing...")
            try:
                samples = load_motion(input_path)
                playback_motion(
                    self.mc,
                    samples,
                    speed=speed,
                    loops=max(1, loops),
                    loop_delay=max(0.0, loop_delay),
                    z_min=z_min,
                    z_max=z_max,
                    real_time=realtime,
                    real_time_dt=realtime_dt,
                    stop_event=self.stop_event,
                )
                self._log(f"Playback complete from {input_path}")
            except Exception as exc:
                self._log(f"Playback error: {exc}")
            finally:
                self.capture_event = None
                self.stop_event = None
                self._set_busy(False, "Idle")

        threading.Thread(target=worker, daemon=True).start()

    def _on_record_main(self) -> None:
        if not self._ensure_mc():
            return
        try:
            duration = float(self.duration_var.get())
            interval = float(self.interval_var.get())
            step_spacing = float(self.step_spacing_var.get())
        except ValueError:
            messagebox.showerror("Invalid timing", "Duration, interval, and keyframe spacing must be numbers.")
            return

        output = Path(self.output_var.get())
        drag = bool(self.drag_var.get())
        self.stop_event = threading.Event()
        self.capture_event = threading.Event()
        self.action_queue = SimpleQueue()

        def worker() -> None:
            self._set_busy(True, "Recording main steps...")
            try:
                record_main_steps(
                    self.mc,
                    duration_s=duration,
                    poll_interval_s=interval,
                    output=output,
                    drag_teach=drag,
                    stop_event=self.stop_event,
                    capture_event=self.capture_event,
                    action_queue=self.action_queue,
                    step_spacing_s=step_spacing,
                )
                self._log(f"Saved main steps to {output}")
            except Exception as exc:
                self._log(f"Record-main error: {exc}")
            finally:
                self.action_queue = None
                self.capture_event = None
                self.stop_event = None
                self._set_busy(False, "Idle")

        threading.Thread(target=worker, daemon=True).start()

    def _on_mark_step(self) -> None:
        if self.action_queue:
            self.action_queue.put("capture")
            self._log("Keyframe capture requested.")

    def _on_vacuum(self, enabled: bool) -> None:
        if self.action_queue:
            self.action_queue.put("vac_on" if enabled else "vac_off")
            self._log(f"Vacuum {'ON' if enabled else 'OFF'} requested and will be recorded.")
        # Also actuate immediately for user feedback
        if self.mc:
            try:
                set_vacuum(self.mc, enabled)
            except Exception as exc:
                self._log(f"Vacuum command failed: {exc}")

    def _on_stop(self) -> None:
        if self.stop_event:
            self.stop_event.set()
            self._log("Stop requested.")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    MotionGUI().run()


if __name__ == "__main__":
    main()
