"""
sim_protocol.py

Builders for the compact, picklable snapshots that the simulation process
streams to the renderer.  Static world data is sent once (and again when it
changes); a dynamic "frame" is sent after every simulation tick.

Everything per-citizen is packed into numpy arrays so a frame for ~10k people
is well under 100 KB.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from world import BUILDING_TYPES, TERRAIN_ORDER, ZONE_TYPES

TERRAIN_KEYS = list(TERRAIN_ORDER)          # index -> key (includes road/rail)
BUILDING_CODES = {name: i + 1 for i, name in enumerate(BUILDING_TYPES)}
ZONE_CODES = {name: i for i, name in enumerate(ZONE_TYPES)}

ACTIVITIES = [
    "sleep", "rest", "work", "commute_work", "commute_home", "commute_school", "study",
    "study_self", "socialize", "exercise", "errand", "clinic", "job_search", "break", "other",
]
ACTIVITY_CODES = {name: i for i, name in enumerate(ACTIVITIES)}


def build_static(engine: Any, version: int) -> Dict[str, Any]:
    world = engine.world
    H, W = world.height, world.width
    terrain = np.zeros((H, W), dtype=np.uint8)
    bld_type = np.zeros((H, W), dtype=np.uint8)
    bld_id = np.zeros((H, W), dtype=np.int32)
    bld_floors = np.zeros((H, W), dtype=np.uint8)
    bld_tier = np.zeros((H, W), dtype=np.uint8)
    elev = np.zeros((H, W), dtype=np.uint8)
    zone = np.zeros((H, W), dtype=np.uint8)
    flags = np.zeros((H, W), dtype=np.uint8)
    terrain_index = {k: i for i, k in enumerate(TERRAIN_KEYS)}
    for y, row in enumerate(world.tiles):
        for x, tile in enumerate(row):
            terrain[y, x] = terrain_index.get(tile.color_key(), 4)
            elev[y, x] = int(max(0.0, min(1.0, tile.elevation)) * 255)
            zone[y, x] = ZONE_CODES.get(tile.zone, 0)
            f = 0
            if tile.landmark:
                f |= 1
            if tile.transit_stop:
                f |= 2
            flags[y, x] = f
    bl = list(world.buildings.values())
    n = len(bl)
    b_x = np.zeros(n, dtype=np.int16)
    b_y = np.zeros(n, dtype=np.int16)
    b_w = np.ones(n, dtype=np.uint8)
    b_h = np.ones(n, dtype=np.uint8)
    b_face = np.zeros(n, dtype=np.uint8)
    b_code = np.zeros(n, dtype=np.uint8)
    b_floors = np.ones(n, dtype=np.uint8)
    b_tier = np.zeros(n, dtype=np.uint8)
    b_id = np.zeros(n, dtype=np.int32)
    for i, b in enumerate(bl):
        ax = b.anchor_x if b.anchor_x >= 0 else b.tile[0]
        ay = b.anchor_y if b.anchor_y >= 0 else b.tile[1]
        tier = int(max(0.0, min(1.0, (np.log10(max(b.value, 1.0)) - 4.5) / 3.5)) * 255)
        code = BUILDING_CODES.get(b.building_type, 0)
        b_x[i], b_y[i], b_w[i], b_h[i] = ax, ay, b.width_tiles, b.height_tiles
        b_face[i], b_code[i], b_floors[i], b_tier[i], b_id[i] = b.facing, code, min(255, b.floors), tier, b.building_id
        bld_type[ay:ay + b.height_tiles, ax:ax + b.width_tiles] = code
        bld_id[ay:ay + b.height_tiles, ax:ax + b.width_tiles] = b.building_id
        bld_floors[ay:ay + b.height_tiles, ax:ax + b.width_tiles] = min(255, b.floors)
        bld_tier[ay:ay + b.height_tiles, ax:ax + b.width_tiles] = tier
    return {
        "version": version,
        "width": W,
        "height": H,
        "terrain": terrain,
        "building_type": bld_type,
        "building_id": bld_id,
        "floors": bld_floors,
        "tier": bld_tier,
        "structure_version": getattr(world, "structure_version", 0),
        "signals": np.array(
            [(*box, jid) for jid, box in enumerate(world.signal_boxes, start=1)], dtype=np.int16
        ).reshape(-1, 5),
        "buildings": {
            "x": b_x, "y": b_y, "w": b_w, "h": b_h, "facing": b_face, "code": b_code,
            "floors": b_floors, "tier": b_tier, "id": b_id,
        },
        "elevation": elev,
        "zone": zone,
        "flags": flags,
        "district_map": np.asarray(getattr(world, "_district_map", np.zeros((H, W), dtype=np.int32)), dtype=np.int16),
        "district_names": {did: d.name for did, d in world.districts.items()},
        "landmarks": [(name, tuple(pos)) for name, pos in world.landmarks],
        "transit_lines": [
            {"name": line.name, "type": line.line_type, "stops": [tuple(stop) for stop in line.stops]}
            for line in world.transit_lines.values()
        ],
        "building_names": {},
    }


def build_layers(engine: Any) -> Dict[str, Any]:
    """Slowly changing per-tile value layers used for heat-map overlays."""
    world = engine.world
    H, W = world.height, world.width
    dmap = np.asarray(world._district_map)
    max_id = max(world.districts) if world.districts else 0
    wealth = np.zeros(max_id + 1, dtype=np.float32)
    crime = np.zeros(max_id + 1, dtype=np.float32)
    for did, d in world.districts.items():
        wealth[did] = d.wealth
        crime[did] = d.crime_heat
    idx = np.clip(dmap, 0, max_id)
    value = np.asarray(getattr(world, "property_value_grid", np.zeros((H, W), dtype=np.float32)), dtype=np.float32)
    return {
        "property_value": value.astype(np.float16),
        "wealth": wealth[idx].astype(np.float16),
        "crime": crime[idx].astype(np.float16),
    }


def _person_lines(engine: Any, selected: Any) -> List[str]:
    population = engine.population
    household = population.household_for_person(selected.person_id)
    home = household.home_building_id if household else "-"
    infection = ["healthy", "exposed", "INFECTIOUS", "immune"][selected.infection_state]
    weekday = (engine.calendar.day_index + 3) % 7
    world = engine.world
    home_b = world.buildings.get(selected.home_building_id) if selected.home_building_id is not None else None
    if home_b is not None:
        listing = world.housing_market.get(household.listing_id) if household else None
        tenure = "owner" if (listing is not None and listing.tenure == "own") else "renter"
        landlord = ""
        if tenure == "renter" and listing is not None:
            lp = population.people.get(listing.owner_id) if listing.owner_id is not None else None
            landlord = f" of {lp.full_name}" if lp is not None else " (corporate landlord)"
        home_line = f"Home: {home_b.name[:26]} fl {selected.home_floor} unit {selected.home_unit} - {tenure}{landlord}"
    else:
        home_line = "Home: none"
    work_line = ""
    if selected.job_company_id is not None and selected.job_company_id in world.companies:
        co = world.companies[selected.job_company_id]
        work_line = f"Works at {co.name[:24]} floor {selected.work_floor}"
    own_line = f"Owns {len(selected.owned_buildings)} building(s)" if selected.owned_buildings else ""
    lines = [
        selected.full_name,
        f"{selected.gender}, age {selected.age}, stage {selected.life_stage}",
        f"Job: {selected.job_title or ('Retired' if selected.pension > 0 else 'Unemployed')}",
        f"Salary ${selected.salary:.0f}  |  Education {selected.education_level}",
        home_line,
        work_line or f"Vehicle {selected.vehicle}",
        own_line or f"Vehicle {selected.vehicle}  |  floor {selected.floor}",
        f"Location {selected.location[0]}, {selected.location[1]}",
        f"Activity: {selected.current_activity(engine.calendar.hour)}",
        f"Destination: {selected.destination[0]}, {selected.destination[1]}",
        f"Energy {selected.energy:.2f}  Health {selected.health:.2f}  ({infection})",
        f"Happiness {selected.happiness:.2f}  Stress {selected.stress:.2f}",
        f"Motivation {selected.motivation:.2f}  Karma {selected.hidden_karma:.2f}",
        f"Reputation {selected.reputation_public:.2f}  Commute {selected.commute_minutes:.0f} min",
        f"Cash ${selected.cash:.0f}  Bank ${selected.bank_balance:.0f}",
        f"Debt ${selected.debt:.0f}  Assets ${selected.assets:.0f}",
        "Top skills:",
    ]
    for key, value in sorted(selected.skills.items(), key=lambda item: item[1], reverse=True)[:5]:
        lines.append(f"  {key[:14]:14} {value:.2f}")
    if selected.memories:
        lines.append("Recent memory:")
        lines.append(f"  {selected.memories[-1].headline}")
    return lines


class FrameBuilder:
    """Builds per-sub-step snapshots.

    A *full* build (top of each hour, or after any state change) packs every
    citizen into numpy arrays and gathers the heavy UI text.  Sub-step builds
    reuse those arrays and only patch the citizens who moved, so streaming a
    frame every two simulated minutes stays cheap."""

    def __init__(self) -> None:
        self.arr: Optional[Dict[str, np.ndarray]] = None
        self.index: Dict[int, int] = {}
        self.extras: Dict[str, Any] = {}
        self.prev_movers: set = set()

    # ------------------------------------------------------------------
    def _full_arrays(self, engine: Any) -> None:
        cal = engine.calendar
        prev_hour = cal.hour if cal.minute else (cal.hour - 1) % 24
        people = list(engine.population.people.values())
        n = len(people)
        movers = engine.world.movers

        def act_code(p: Any) -> int:
            block = p.schedule.get(prev_hour)
            return ACTIVITY_CODES.get(block.activity, len(ACTIVITIES) - 1) if block is not None else len(ACTIVITIES) - 1

        arr = {
            "ids": np.fromiter((p.person_id for p in people), dtype=np.int32, count=n),
            "x": np.fromiter((p.location[0] for p in people), dtype=np.int16, count=n),
            "y": np.fromiter((p.location[1] for p in people), dtype=np.int16, count=n),
            "happy": np.fromiter((int(p.happiness * 255) for p in people), dtype=np.uint8, count=n),
            "stress": np.fromiter((int(p.stress * 255) for p in people), dtype=np.uint8, count=n),
            "health": np.fromiter((int(p.health * 255) for p in people), dtype=np.uint8, count=n),
            "infection": np.fromiter((p.infection_state for p in people), dtype=np.uint8, count=n),
            "age": np.fromiter((min(255, p.age) for p in people), dtype=np.uint8, count=n),
            "activity": np.fromiter((act_code(p) for p in people), dtype=np.uint8, count=n),
            "gender": np.fromiter((0 if p.gender == "man" else 1 for p in people), dtype=np.uint8, count=n),
            "indoors": np.fromiter((1 if p.floor >= 0 else 0 for p in people), dtype=np.uint8, count=n),
            "mode": np.fromiter((p.mode for p in people), dtype=np.uint8, count=n),
        }
        self.arr = arr
        self.index = {int(pid): i for i, pid in enumerate(arr["ids"].tolist())}
        self.prev_movers = set(movers)

    def _extras(self, engine: Any) -> None:
        world = engine.world
        population = engine.population
        selected = None
        try:
            selected = engine.selected_person()
        except Exception:
            selected = None
        metrics = engine.macro_metrics()
        dyn = engine.dynamics
        rel = []
        if selected is not None:
            rel = population.relationship_lines(selected.person_id, limit=14) or ["No meaningful ties recorded."]
        self.extras = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "relationship_lines": rel,
            "news_lines": [item.headline for item in reversed(engine.news[-40:])][:30],
            "timeline_lines": [e.title for e in reversed(engine.timeline[-40:])][:30],
            "company_lines": world.company_summary_lines(limit=16),
            "district_lines": world.district_summary_lines(limit=12),
            "market_lines": engine.systems.stock_market.market_lines(limit=8),
            "dynamics_lines": dyn.summary_lines(),
            "routing": world.route_service.stats.summary(),
            "taxi_revenue": float(world.taxi_revenue_today),
        }

    # ------------------------------------------------------------------
    def build(self, engine: Any, tick_id: int, status: str = "", segments: Optional[Dict[int, Any]] = None, full: bool = True) -> Dict[str, Any]:
        cal = engine.calendar
        world = engine.world
        people = engine.population.people
        if full or self.arr is None or self.arr["ids"].shape[0] != len(people):
            self._full_arrays(engine)
            self._extras(engine)
        arr = self.arr
        index = self.index
        segments = segments or {}
        movers = world.movers
        # Patch citizens who moved this sub-step.
        mv_ids: List[int] = []
        mv_mode: List[int] = []
        offsets: List[int] = [0]
        chunks: List[Any] = []
        time_chunks: List[Any] = []
        times_by_pid = engine.population.last_times
        total = 0
        for pid, seg in segments.items():
            i = index.get(pid)
            p = people.get(pid)
            if i is None or p is None:
                continue
            last = seg[-1]
            arr["x"][i] = last[0]
            arr["y"][i] = last[1]
            arr["indoors"][i] = 0
            arr["mode"][i] = p.mode
            mv_ids.append(pid)
            mv_mode.append(p.mode)
            chunks.append(seg)
            vt = times_by_pid.get(pid)
            time_chunks.append(vt if vt is not None else np.linspace(0.0, 120.0, len(seg), dtype=np.float32))
            total += len(seg)
            offsets.append(total)
        # Citizens who finished travelling this sub-step may have gone indoors.
        for pid in self.prev_movers - movers:
            i = index.get(pid)
            p = people.get(pid)
            if i is not None and p is not None:
                arr["x"][i], arr["y"][i] = p.location
                arr["indoors"][i] = 1 if p.floor >= 0 else 0
        self.prev_movers = set(movers)
        pts = np.concatenate(chunks).astype(np.int16) if chunks else np.zeros((0, 2), dtype=np.int16)
        pt_times = np.concatenate(time_chunks).astype(np.float32) if time_chunks else np.zeros(0, dtype=np.float32)
        selected = None
        selected_lines: List[str] = []
        selected_route: List[Any] = []
        try:
            selected = engine.selected_person()
            selected_lines = _person_lines(engine, selected)
            selected_route = selected.path[:240]
        except Exception:
            selected = None
        frame: Dict[str, Any] = {
            "tick": tick_id,
            "sim_tick": engine.tick_counter,
            "sub": engine.sub_counter,
            "calendar_label": cal.label(),
            "hour": cal.hour,
            "minute": cal.minute,
            "day_index": cal.day_index,
            "month": cal.month,
            "season": cal.season(),
            "weekday": (cal.day_index + 3) % 7,
            "weather": engine.weather.summary(),
            "weather_condition": engine.weather.condition,
            "temperature": engine.weather.temperature_c,
            "paused": engine.paused,
            "tick_duration": engine.tick_duration,
            "population": int(arr["ids"].shape[0]),
            "traffic_factor": world.traffic_factor,
            "mv_ids": np.asarray(mv_ids, dtype=np.int32),
            "mv_mode": np.asarray(mv_mode, dtype=np.uint8),
            "mv_off": np.asarray(offsets, dtype=np.int32),
            "mv_pts": pts,
            "mv_t": pt_times,
            "sim_seconds": float(world.sim_seconds),
            "selected_id": selected.person_id if selected is not None else None,
            "selected_location": tuple(selected.location) if selected is not None else None,
            "selected_lines": selected_lines,
            "selected_route": [tuple(q) for q in selected_route],
            "status": status,
            "structure_version": getattr(world, "structure_version", 0),
            "building": None,
        }
        frame.update({k: (v.copy() if k in ("x", "y", "indoors", "mode") else v) for k, v in arr.items()})
        frame.update(self.extras)
        inspected = getattr(engine, "inspected_building", None)
        if inspected is not None:
            from buildings import describe_building

            frame["building"] = describe_building(engine, inspected)
        return frame


def build_frame(engine: Any, tick_id: int, status: str = "") -> Dict[str, Any]:
    """One-off full snapshot (tests and tools)."""
    return FrameBuilder().build(engine, tick_id, status, None, True)


def build_daily_extras(engine: Any) -> Dict[str, Any]:
    """Heavier analytics payload sent once per simulated day."""
    an = engine.analytics
    return {
        "history": {k: list(v) for k, v in an.history.items()},
        "history_days": list(an.history_days),
        "age_pyramid": an.age_pyramid,
        "wealth_hist": an.wealth_hist,
        "sector_jobs": an.sector_jobs,
        "day_index": engine.calendar.day_index,
    }
