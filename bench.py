"""Headless benchmark harness for the RLSS simulator.

Runs the engine for N sim-days with fixed seeds, reporting:
  - ms/tick p50 / p95 / max (hourly ticks; daily-cycle ticks reported separately)
  - route tier counters (if the routing layer exposes them)
  - aggregate behavior fingerprint (mean happiness/stress, unemployment, treasury)
    so refactors can detect unintended behavior drift.

Usage:
    python bench.py [--days 14] [--seed 1234] [--world-seed 99] [--json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from engine import SimulationEngine


def run_bench(days: int, seed: int, world_seed: int, preset: str = "large") -> dict:
    t0 = time.perf_counter()
    engine = SimulationEngine(seed=seed, world_seed=world_seed, preset=preset)
    engine.autosave_interval_days = 10 ** 9  # no disk writes during bench
    init_s = time.perf_counter() - t0

    hourly_ms: list[float] = []
    daily_ms: list[float] = []
    total_ticks = days * 24
    for _ in range(total_ticks):
        is_daily = engine.calendar.hour == 0
        t = time.perf_counter()
        engine.step()
        dt = (time.perf_counter() - t) * 1000.0
        (daily_ms if is_daily else hourly_ms).append(dt)

    metrics = engine.macro_metrics() if hasattr(engine, "macro_metrics") else {}
    people = list(engine.population.people.values())
    n = max(1, len(people))
    labor_force, employed, unemployed = engine.population.labor_stats()
    fingerprint = {
        "population": len(people),
        "mean_happiness": round(sum(p.happiness for p in people) / n, 4),
        "mean_stress": round(sum(p.stress for p in people) / n, 4),
        "mean_energy": round(sum(p.energy for p in people) / n, 4),
        "labor_force": labor_force,
        "employed": employed,
        "unemployed_adults": unemployed,
        "unemployment_rate": round(unemployed / labor_force, 4) if labor_force else 0.0,
        "vacancies": len(engine.world.open_job_postings()),
        "births": engine.population.births_total,
        "deaths": engine.population.deaths_total,
        "companies": len(engine.world.companies),
        "activity": engine.population.activity_heatmap(engine.calendar.hour),
        "macro": {k: round(float(v), 4) for k, v in metrics.items()} if isinstance(metrics, dict) else {},
    }

    routing_stats = {}
    stats_obj = getattr(getattr(engine.world, "route_service", None), "stats", None)
    if stats_obj is not None:
        routing_stats = {k: v for k, v in vars(stats_obj).items() if not k.startswith("_")}

    def pct(data: list[float], q: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return s[min(len(s) - 1, int(q * len(s)))]

    return {
        "days": days,
        "seed": seed,
        "world_seed": world_seed,
        "init_s": round(init_s, 2),
        "hourly_ms": {
            "p50": round(statistics.median(hourly_ms), 2),
            "p95": round(pct(hourly_ms, 0.95), 2),
            "max": round(max(hourly_ms), 2),
            "mean": round(statistics.fmean(hourly_ms), 2),
        },
        "daily_ms": {
            "p50": round(statistics.median(daily_ms), 2) if daily_ms else 0.0,
            "max": round(max(daily_ms), 2) if daily_ms else 0.0,
        },
        "total_sim_s": round(sum(hourly_ms + daily_ms) / 1000.0, 2),
        "routing": routing_stats,
        "fingerprint": fingerprint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--world-seed", type=int, default=99)
    parser.add_argument("--preset", default="large", choices=["large", "small", "classic"])
    parser.add_argument("--json", action="store_true", help="emit raw JSON only")
    args = parser.parse_args()

    result = run_bench(args.days, args.seed, args.world_seed, args.preset)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"RLSS bench: {result['days']} days  seed={result['seed']} world={result['world_seed']}")
    print(f"  init: {result['init_s']}s   total sim: {result['total_sim_s']}s")
    h = result["hourly_ms"]
    print(f"  hourly tick ms  p50={h['p50']}  p95={h['p95']}  max={h['max']}  mean={h['mean']}")
    d = result["daily_ms"]
    print(f"  daily tick ms   p50={d['p50']}  max={d['max']}")
    if result["routing"]:
        print(f"  routing: {result['routing']}")
    print(f"  fingerprint: {json.dumps(result['fingerprint'])}")


if __name__ == "__main__":
    main()
