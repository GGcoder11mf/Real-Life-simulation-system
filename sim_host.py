"""
sim_host.py

Runs the simulation engine in its own OS process and streams snapshots to the
renderer.  Separating the two removes every source of frame-rate coupling:
the UI never waits for a simulation tick (no shared GIL, no shared lock), so
frame rate is independent of population size and tick cost.

    SimClient  - used by the frontend: spawns the worker, receives snapshots
                 on a small reader thread, sends commands.
    host_main  - the worker entry point (runs inside the child process).
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import threading
from collections import deque
import time
import traceback
from typing import Any, Dict, Optional, Tuple


# ----------------------------------------------------------------------
# Worker side
# ----------------------------------------------------------------------
def host_main(conn: Any, config: Dict[str, Any]) -> None:
    try:
        _host_loop(conn, config)
    except Exception:  # noqa: BLE001 - report to the UI instead of dying silently
        try:
            conn.send(("fatal", traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass


def _host_loop(conn: Any, config: Dict[str, Any]) -> None:
    from engine import SAVE_FILE, SimulationEngine
    from world import SUBSTEP_MINUTES
    from sim_protocol import FrameBuilder, build_daily_extras, build_layers, build_static

    conn.send(("status", "Generating city..."))
    t0 = time.perf_counter()
    if config.get("load_path"):
        engine = SimulationEngine.load(config["load_path"])
    else:
        engine = SimulationEngine(
            seed=config.get("seed"),
            world_seed=config.get("world_seed"),
            preset=config.get("preset", "large"),
        )
    if config.get("tick_duration") is not None:
        engine.tick_duration = float(config["tick_duration"])
    engine.tick_duration = max(1e-4, engine.tick_duration)
    engine.autosave_interval_days = int(config.get("autosave_days", engine.autosave_interval_days))
    def run_presim(days: int) -> None:
        t_start = time.perf_counter()

        def progress(done: int, total: int) -> None:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / max(1, done) * (total - done)
            conn.send(("progress", (done, total, eta)))
            conn.send(("status", f"Pre-simulating day {done}/{total}  (about {eta / 60:.1f} min left)"))

        conn.send(("progress", (0, days, 0.0)))
        engine.fast_forward(days, progress)
        conn.send(("progress", None))

    if int(config.get("presim_days", 0) or 0) > 0:
        run_presim(int(config["presim_days"]))
    build_s = time.perf_counter() - t0

    static_version = 1
    tick_id = 0
    status = f"City ready in {build_s:.1f}s."

    def publish_static() -> None:
        conn.send(("static", build_static(engine, static_version)))
        conn.send(("layers", build_layers(engine)))
        conn.send(("daily", build_daily_extras(engine)))

    builder = FrameBuilder()

    def publish_frame(step_ms: float = 0.0, segments: Any = None, full: bool = True) -> None:
        frame = builder.build(engine, tick_id, status, segments, full)
        frame["step_ms"] = step_ms
        conn.send(("frame", frame))

    publish_static()
    publish_frame(0.0, None, True)
    # Garbage collection of millions of long-lived objects costs 100+ ms and shows up as
    # a visible hitch.  Freeze the city we just built, stop automatic collections and
    # run small collections ourselves while waiting for the next tick.
    gc.collect()
    gc.freeze()
    gc.disable()
    gc_steps = 0
    last_step = time.monotonic()
    sent_structure = getattr(engine.world, "structure_version", 0)
    last_day = engine.calendar.day_index
    dirty = False
    max_steps = int(config.get("max_steps", 0))
    steps_done = 0

    while True:
        # ---- commands
        while conn.poll(0):
            cmd = conn.recv()
            name = cmd[0]
            if name == "quit":
                return
            elif name == "pause":
                engine.toggle_pause()
                dirty = True
            elif name == "speed":
                engine.change_speed(bool(cmd[1]))
                status = f"Tick interval {engine.tick_duration * 1000:.1f} ms"
                dirty = True
            elif name == "set_tick_duration":
                engine.tick_duration = max(1e-4, float(cmd[1]))
                speed_x = 120.0 / engine.tick_duration
                status = f"Sim speed x{speed_x:.1f} ({engine.tick_duration:.3f}s per 2m)"
                dirty = True
            elif name == "presim":
                try:
                    engine.paused = False
                    run_presim(int(cmd[1]))
                    static_version += 1
                    publish_static()
                    last_day = engine.calendar.day_index
                    sent_structure = getattr(engine.world, "structure_version", 0)
                    status = f"Pre-simulated {int(cmd[1])} days."
                except Exception as exc:  # noqa: BLE001
                    status = f"Pre-simulation failed: {exc}"
                    conn.send(("progress", None))
                builder = FrameBuilder()
                last_step = time.monotonic()
                dirty = True
            elif name == "select":
                if cmd[1] in engine.population.people:
                    engine.selected_person_id = cmd[1]
                dirty = True
            elif name == "inspect":
                engine.inspected_building = cmd[1]
                dirty = True
            elif name == "cycle":
                engine.cycle_selected_person()
                dirty = True
            elif name == "save":
                try:
                    engine.save(SAVE_FILE)
                    status = f"Saved simulation to {SAVE_FILE}"
                except Exception as exc:  # noqa: BLE001
                    status = f"Save failed: {exc}"
                dirty = True
            elif name == "load":
                try:
                    conn.send(("status", "Loading save..."))
                    engine = SimulationEngine.load(SAVE_FILE)
                    static_version += 1
                    publish_static()
                    last_day = engine.calendar.day_index
                    status = f"Loaded simulation from {SAVE_FILE}"
                except Exception as exc:  # noqa: BLE001
                    status = f"Load failed: {exc}"
                dirty = True
        now = time.monotonic()
        if engine.paused or now - last_step < engine.tick_duration:
            if dirty:
                tick_id += 1
                publish_frame()
                dirty = False
            if not engine.paused and gc_steps > 0:
                gc_steps -= 1
                gc.collect(1 if gc_steps % 8 == 0 else 0)
            sleep_for = 0.003 if engine.paused else min(0.003, max(0.0, engine.tick_duration - (now - last_step)))
            if sleep_for > 0:
                time.sleep(sleep_for)
            continue
        t = time.perf_counter()
        segments = engine.step_sub()
        step_ms = (time.perf_counter() - t) * 1000.0
        last_step = max(last_step + engine.tick_duration, time.monotonic() - engine.tick_duration)
        tick_id += 1
        steps_done += 1
        gc_steps = min(gc_steps + 1, 4000)
        if gc_steps >= 4000:                     # never starved of idle time: collect anyway
            gc.collect(1)
            gc_steps = 2000
        full = False
        if engine.calendar.day_index != last_day:
            last_day = engine.calendar.day_index
            conn.send(("layers", build_layers(engine)))
            conn.send(("daily", build_daily_extras(engine)))
            if getattr(engine.world, "structure_version", 0) != sent_structure:
                static_version += 1
                conn.send(("static", build_static(engine, static_version)))
                sent_structure = getattr(engine.world, "structure_version", 0)
        publish_frame(step_ms, segments, full)
        dirty = False
        if max_steps and steps_done >= max_steps:
            return


# ----------------------------------------------------------------------
# UI side
# ----------------------------------------------------------------------
class SimClient:
    """Owns the worker process and the latest snapshots received from it."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ctx = mp.get_context("spawn")
        self._conn, child_conn = ctx.Pipe(duplex=True)
        self.process = ctx.Process(target=host_main, args=(child_conn, config or {}), daemon=True, name="rlss-sim")
        self.process.start()
        child_conn.close()
        self.lock = threading.Lock()
        self.static: Optional[Dict[str, Any]] = None
        self.layers: Optional[Dict[str, Any]] = None
        self.daily: Optional[Dict[str, Any]] = None
        self.frame: Optional[Dict[str, Any]] = None
        self.prev_frame: Optional[Dict[str, Any]] = None
        self.frame_time = 0.0
        self.frame_interval = 0.1
        self.status = "Starting simulation process..."
        self.progress: Optional[Tuple[int, int, float]] = None
        self.fatal: Optional[str] = None
        self.frames_received = 0
        self._timeline: "deque" = deque(maxlen=24)
        self.static_version = 0
        self.layers_version = 0
        self.daily_version = 0
        self._alive = True
        self._reader = threading.Thread(target=self._read_loop, name="sim-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while self._alive:
            try:
                if not self._conn.poll(0.05):
                    if not self.process.is_alive() and self.fatal is None and self.frame is None:
                        self.fatal = "Simulation process exited unexpectedly."
                    continue
                kind, payload = self._conn.recv()
            except (EOFError, OSError):
                break
            with self.lock:
                if kind == "frame":
                    payload = self._merge_frame(payload)
                    now = time.monotonic()
                    if self.frame is not None:
                        self.frame_interval = max(0.01, min(2.0, now - self.frame_time))
                    self.prev_frame = self.frame
                    self._timeline.append((now, payload))
                    self.frame = payload
                    self.frame_time = now
                    self.frames_received += 1
                elif kind == "static":
                    self.static = payload
                    self.static_version += 1
                elif kind == "layers":
                    self.layers = payload
                    self.layers_version += 1
                elif kind == "daily":
                    self.daily = payload
                    self.daily_version += 1
                elif kind == "progress":
                    self.progress = payload
                elif kind == "status":
                    self.status = payload
                elif kind == "fatal":
                    self.fatal = payload

    def _merge_frame(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Delta frames carry only moved citizens; rebuild the full arrays here."""
        if not payload.get("delta"):
            self._base = payload
            return payload
        base = getattr(self, "_base", None)
        if base is None:
            return payload
        merged = dict(base)
        merged.update(payload)
        idx = payload["patch_idx"]
        for key, vals in payload["patch"].items():
            col = base[key].copy()
            if idx.shape[0] and idx[-1] < col.shape[0]:
                col[idx] = vals
            merged[key] = col
        merged.pop("delta", None)
        self._base = merged
        return merged

    def timeline(self):
        with self.lock:
            return list(self._timeline)

    def snapshot(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], float, float]:
        with self.lock:
            return self.frame, self.prev_frame, self.frame_time, self.frame_interval

    def send(self, *cmd: Any) -> None:
        try:
            self._conn.send(cmd)
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        self._alive = False
        self.send("quit")
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
