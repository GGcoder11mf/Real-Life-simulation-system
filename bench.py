"""Headless benchmark / profiling harness for the RLSS simulator.

Runs the engine for N sim-days with fixed seeds, reporting:
  - throughput (sim-days per real minute) and per-substep ms p50/p95/p99/max
  - where each substep's time goes: hourly planning slices, needs-effect slices,
    hourly dynamics, event engine, daily-cycle stages, citizen movement, routing
  - route tier counters
  - aggregate behaviour fingerprint, plus (with --hash) a bit-exact state hash so
    a refactor can prove it did not change the simulation's results.

Instrumentation wraps instance methods from the outside, so the engine itself
carries no profiling overhead.

Usage:
    python bench.py [--days 3] [--preset large|small|classic] [--size 320x160]
                    [--hash out.json] [--expect baseline.json] [--json]
    python bench.py --suite          # small / default / stress-test populations
    python bench.py --gpu cuda       # CUDA A* routing (off | cuda | hybrid | auto)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from engine import SimulationEngine


# ---------------------------------------------------------------------------
# Bit-exact state hashing (determinism / precision guard)
# ---------------------------------------------------------------------------
def _canon(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted((_canon(v) for v in o), key=repr)
    if isinstance(o, np.ndarray):
        return hashlib.sha1(np.ascontiguousarray(o).tobytes()).hexdigest() + str(o.dtype) + str(o.shape)
    if isinstance(o, (float, np.floating)):
        return repr(float(o))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, bytes):
        return hashlib.sha1(o).hexdigest()
    return o


def state_hash(engine: SimulationEngine) -> Dict[str, str]:
    """Per-section hashes of the full saved state plus live (unsaved) movement state."""
    data = engine.to_dict()
    data.pop("tick_duration", None)
    parts = {k: hashlib.sha1(json.dumps(_canon(v), default=repr).encode()).hexdigest()[:16] for k, v in data.items()}
    live = []
    for pid in sorted(engine.population.people):
        p = engine.population.people[pid]
        route = p._route
        live.append((pid, tuple(p.location), tuple(p.destination) if p.destination else None, p._route_i,
                     repr(p._acc), p._delay, p.mode, p.floor,
                     None if route is None else hashlib.sha1(np.ascontiguousarray(route).tobytes()).hexdigest()))
    parts["live_people"] = hashlib.sha1(repr(live).encode()).hexdigest()[:16]
    parts["movers"] = hashlib.sha1(repr(sorted(engine.world.movers)).encode()).hexdigest()[:16]
    occ = [(x, y, list(t.occupants)) for y, row in enumerate(engine.world.tiles) for x, t in enumerate(row) if t.occupants]
    parts["occupants"] = hashlib.sha1(repr(occ).encode()).hexdigest()[:16]
    return parts


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
class Probe:
    """Accumulates wall time per label; wraps bound methods on instances."""

    def __init__(self) -> None:
        self.totals: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)

    def add(self, label: str, dt: float) -> None:
        self.totals[label] += dt
        self.calls[label] += 1

    def wrap(self, obj: Any, name: str, label_fn) -> None:
        orig = getattr(obj, name)
        add = self.add
        clock = time.perf_counter

        def wrapper(*args, **kwargs):
            t = clock()
            try:
                return orig(*args, **kwargs)
            finally:
                add(label_fn(*args, **kwargs), clock() - t)

        setattr(obj, name, wrapper)


def instrument(engine: SimulationEngine, probe: Probe) -> None:
    plan_n = engine.PLAN_SLICES
    eff_n = engine.EFFECT_SLICES

    def slot_label(slot: int) -> str:
        if slot < plan_n:
            return "hour.plan"
        if slot < plan_n + eff_n:
            return "hour.effects"
        if slot == plan_n + eff_n:
            return "hour.dynamics+weather"
        return "hour.events"

    def daily_label(slot: int, day_index: int) -> str:
        jobs = engine._daily_jobs
        if slot >= len(jobs):
            return "daily.idle"
        kind, arg = jobs[slot]
        if kind == "people":
            return "daily.people"
        if kind == "tail":
            return f"daily.tail{arg}"
        name = getattr(arg, "__qualname__", None) or getattr(arg, "__name__", "?")
        return f"daily.{name.replace('<locals>.', '')}[{slot}]"

    # _daily_slot runs inside _run_slot; record its time and subtract it from the slot bucket.
    daily_time = [0.0]
    orig_daily = engine._daily_slot

    def daily_wrapper(slot, day_index):
        label = "daily.world+slot0" if slot == 0 else None
        t = time.perf_counter()
        orig_daily(slot, day_index)
        dt = time.perf_counter() - t
        daily_time[0] += dt
        probe.add(label or daily_label(slot, day_index), dt)

    engine._daily_slot = daily_wrapper

    orig_slot = engine._run_slot

    def slot_wrapper(slot):
        before = daily_time[0]
        t = time.perf_counter()
        orig_slot(slot)
        dt = time.perf_counter() - t - (daily_time[0] - before)
        probe.add(slot_label(slot), dt)

    engine._run_slot = slot_wrapper
    probe.wrap(engine.population, "move_substep", lambda *a, **k: "move")
    rs = getattr(engine.world, "route_service", None)
    if rs is not None:
        probe.wrap(rs, "batch_routes_arrays", lambda *a, **k: "routing (inside plan)")


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------
def _pct(data: List[float], q: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(len(s) - 1, int(q * len(s)))]


def run_bench(days: int, seed: int, world_seed: int, preset: str = "large", size: Optional[tuple] = None,
              do_hash: bool = False, save_timing: bool = True) -> dict:
    t0 = time.perf_counter()
    engine = SimulationEngine(seed=seed, world_seed=world_seed, preset=preset, world_size=size)
    engine.autosave_interval_days = 10 ** 9  # no disk writes during bench
    init_s = time.perf_counter() - t0

    probe = Probe()
    instrument(engine, probe)
    sub_ms: List[float] = []
    clock = time.perf_counter
    t_sim = clock()
    for _ in range(days * 24 * 30):
        t = clock()
        engine.step_sub()
        sub_ms.append((clock() - t) * 1000.0)
    sim_s = clock() - t_sim
    hashed = state_hash(engine) if do_hash else None

    save_s = None
    if save_timing:
        t = clock()
        blob = engine._serialize_save_payload(engine.to_dict())
        save_s = round(clock() - t, 3)
        del blob

    people = list(engine.population.people.values())
    n = max(1, len(people))
    labor_force, employed, unemployed = engine.population.labor_stats()
    fingerprint = {
        "population": len(people),
        "mean_happiness": round(sum(p.happiness for p in people) / n, 6),
        "mean_stress": round(sum(p.stress for p in people) / n, 6),
        "mean_energy": round(sum(p.energy for p in people) / n, 6),
        "employed": employed,
        "unemployed_adults": unemployed,
        "vacancies": len(engine.world.open_job_postings()),
        "births": engine.population.births_total,
        "deaths": engine.population.deaths_total,
        "companies": len(engine.world.companies),
    }
    routing_stats = {}
    stats_obj = getattr(getattr(engine.world, "route_service", None), "stats", None)
    if stats_obj is not None:
        routing_stats = {k: v for k, v in vars(stats_obj).items() if not k.startswith("_")}

    breakdown = {k: round(v / days, 3) for k, v in sorted(probe.totals.items(), key=lambda kv: -kv[1])}
    accounted = sum(v for k, v in probe.totals.items() if not k.startswith("routing"))
    breakdown["other (calendar, bookkeeping)"] = round((sim_s - accounted) / days, 3)

    return {
        "days": days,
        "seed": seed,
        "world_seed": world_seed,
        "world": f"{engine.world.width}x{engine.world.height}",
        "init_s": round(init_s, 2),
        "sim_s": round(sim_s, 2),
        "s_per_sim_day": round(sim_s / days, 3),
        "sim_days_per_minute": round(60.0 * days / sim_s, 2),
        "substep_ms": {
            "mean": round(statistics.fmean(sub_ms), 2),
            "p50": round(statistics.median(sub_ms), 2),
            "p95": round(_pct(sub_ms, 0.95), 2),
            "p99": round(_pct(sub_ms, 0.99), 2),
            "max": round(max(sub_ms), 2),
        },
        "seconds_per_sim_day_by_phase": breakdown,
        "save_serialize_s": save_s,
        "routing": routing_stats,
        "fingerprint": fingerprint,
        "state_hash": hashed,
    }


def _print(result: dict) -> None:
    print(f"RLSS bench: {result['days']} days  world={result['world']}  pop={result['fingerprint']['population']}"
          f"  seed={result['seed']}/{result['world_seed']}")
    print(f"  init {result['init_s']}s   sim {result['sim_s']}s   -> {result['s_per_sim_day']} s/sim-day"
          f"   ({result['sim_days_per_minute']} sim-days per real minute)")
    s = result["substep_ms"]
    print(f"  substep ms  mean={s['mean']}  p50={s['p50']}  p95={s['p95']}  p99={s['p99']}  max={s['max']}")
    print("  seconds per sim-day by phase:")
    total = result["s_per_sim_day"] or 1.0
    for k, v in result["seconds_per_sim_day_by_phase"].items():
        print(f"    {k:<44} {v:8.3f}s  {100.0 * v / total:5.1f}%")
    if result["save_serialize_s"] is not None:
        print(f"  save serialize: {result['save_serialize_s']}s")
    r = result["routing"]
    if r:
        print(f"  routing: backend={r.get('backend')} numba={r.get('numba_hits')} cache={r.get('cache_hits')}"
              f" gpu={r.get('gpu_hits')} fail={r.get('failures')}"
              f"  cuda[{r.get('cuda_mode')}]={r.get('cuda_hits')} dev={r.get('cuda_device')}")
        if r.get("cuda_note"):
            print(f"  cuda note: {r['cuda_note']}")
    print(f"  fingerprint: {json.dumps(result['fingerprint'], default=float)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--world-seed", type=int, default=99)
    parser.add_argument("--preset", default="large", choices=["large", "small", "classic"])
    parser.add_argument("--size", help="explicit world size WxH, e.g. 320x160 (~25k citizens)")
    parser.add_argument("--hash", metavar="OUT", help="write the bit-exact state hash to this JSON file")
    parser.add_argument("--expect", metavar="BASELINE", help="compare the state hash with a saved one; exit 1 on drift")
    parser.add_argument("--suite", action="store_true", help="run small / default / stress populations")
    parser.add_argument("--json", action="store_true", help="emit raw JSON only")
    parser.add_argument("--gpu", choices=["off", "cuda", "hybrid", "auto"],
                        help="CUDA A* routing mode (default: $RLSS_GPU or auto)")
    args = parser.parse_args()
    if args.gpu:
        os.environ["RLSS_GPU"] = args.gpu

    if args.suite:
        configs = [("small", "classic", None), ("default", "large", None), ("stress", "large", (320, 160))]
        results = []
        for name, preset, size in configs:
            res = run_bench(args.days, args.seed, args.world_seed, preset, size, save_timing=False)
            results.append(res)
            print(f"{name:>8}: pop={res['fingerprint']['population']:>6}  {res['s_per_sim_day']:>7}s/sim-day  "
                  f"{res['sim_days_per_minute']:>6} sim-days/min  substep p99={res['substep_ms']['p99']}ms", flush=True)
        if args.json:
            print(json.dumps(results, indent=2, default=float))
        return

    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    result = run_bench(args.days, args.seed, args.world_seed, args.preset, size,
                       do_hash=bool(args.hash or args.expect))
    if args.json:
        print(json.dumps(result, indent=2, default=float))
    else:
        _print(result)
    h = result["state_hash"]
    if args.hash:
        with open(args.hash, "w", encoding="utf-8") as fh:
            json.dump(h, fh, indent=1)
    if args.expect:
        with open(args.expect, encoding="utf-8") as fh:
            base = json.load(fh)
        drift = sorted(k for k in set(base) | set(h) if base.get(k) != h.get(k))
        if drift:
            print(f"STATE DRIFT in: {', '.join(drift)}")
            sys.exit(1)
        print("state hash: IDENTICAL to baseline (bit-exact)")


if __name__ == "__main__":
    main()
