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

from world import SUBSTEPS_PER_HOUR, BUILDING_TYPES, TERRAIN_ORDER, ZONE_TYPES

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
        work_line or "No workplace",
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
    life = getattr(engine, "life", None)
    if life is not None:
        lines.extend(life.body.describe(selected.person_id))
        rec = life.school.get(selected.person_id)
        if rec is not None and rec.grade > 0:
            lines.append(f"School grade {rec.grade}  GPA {rec.gpa * 4:.1f}  absences {rec.absences}" + ("  (bullied)" if rec.bullied > 0.5 else "") + ("  (bully)" if rec.bully > 0.5 else ""))
        partner = life.dating.get(selected.person_id)
        if partner is not None and partner in engine.population.people:
            lines.append(f"Dating {engine.population.people[partner].full_name}")
        elif selected.spouse_id is not None and selected.spouse_id in engine.population.people:
            lines.append(f"Married to {engine.population.people[selected.spouse_id].full_name}; {len(selected.children_ids)} child(ren)")
        if selected.care_tile is not None:
            lines.append("In hospital care")
        if selected.incarcerated:
            lines.append("In prison")
        elif selected.arrest_record:
            lines.append(f"Record: {selected.arrest_record[-1]}")
    if selected.memories:
        lines.append("Recent memory:")
        lines.append(f"  {selected.memories[-1].headline}")
    return lines


_LIN_CACHE: Dict[int, np.ndarray] = {}


def _lin_times(n: int) -> np.ndarray:
    arr = _LIN_CACHE.get(n)
    if arr is None:
        arr = np.linspace(0.0, 120.0, n, dtype=np.float32)
        _LIN_CACHE[n] = arr
    return arr


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
        self._gen_seen = -1
        self.plist: List[Any] = []
        self.slice_i = 0

    # ------------------------------------------------------------------
    _SLICE_KEYS = ("x", "y", "happy", "stress", "health", "infection", "age", "activity", "indoors", "mode")

    def _fill(self, arr: Dict[str, np.ndarray], people: List[Any], lo: int, prev_hour: int) -> None:
        """(Re)compute the citizen rows lo..lo+len(people) of every per-citizen array (one pass)."""
        n = len(people)
        if n == 0:
            return
        hi = lo + n
        none_code = len(ACTIVITIES) - 1
        codes = ACTIVITY_CODES
        rows = []
        append = rows.append
        for p in people:
            block = p.schedule.get(prev_hour)
            act = codes.get(block.activity, none_code) if block is not None else none_code
            loc = p.location
            age = p.age
            append((loc[0], loc[1], int(p.happiness * 255), int(p.stress * 255), int(p.health * 255),
                    p.infection_state, age if age < 255 else 255, act, 1 if p.floor >= 0 else 0, p.mode))
        m = np.array(rows, dtype=np.int32)
        arr["x"][lo:hi] = m[:, 0]
        arr["y"][lo:hi] = m[:, 1]
        arr["happy"][lo:hi] = m[:, 2]
        arr["stress"][lo:hi] = m[:, 3]
        arr["health"][lo:hi] = m[:, 4]
        arr["infection"][lo:hi] = m[:, 5]
        arr["age"][lo:hi] = m[:, 6]
        arr["activity"][lo:hi] = m[:, 7]
        arr["indoors"][lo:hi] = m[:, 8]
        arr["mode"][lo:hi] = m[:, 9]

    def _full_arrays(self, engine: Any) -> None:
        cal = engine.calendar
        prev_hour = cal.hour if cal.minute else (cal.hour - 1) % 24
        people = list(engine.population.people.values())
        n = len(people)
        arr = {k: np.zeros(n, dtype=np.int16 if k in ("x", "y") else np.uint8) for k in self._SLICE_KEYS}
        arr["ids"] = np.fromiter((p.person_id for p in people), dtype=np.int32, count=n)
        arr["gender"] = np.fromiter((0 if p.gender == "man" else 1 for p in people), dtype=np.uint8, count=n)
        self._fill(arr, people, 0, prev_hour)
        self.arr = arr
        self.plist = people
        self.slice_i = 0
        self.index = {int(pid): i for i, pid in enumerate(arr["ids"].tolist())}
        self.prev_movers = set(engine.world.movers)

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
            "society_lines": self._society_lines(engine),
            "influence_lines": engine.systems.market.ranking_lines(12) + [""] + engine.systems.market.concentration_lines(),
            "routing": world.route_service.stats.summary(),
            "taxi_revenue": float(world.taxi_revenue_today),
        }

    @staticmethod
    def _society_lines(engine: Any) -> List[str]:
        life = engine.life
        pop = engine.population
        lines = ["-- Health, family and school --"] + life.summary_lines(pop)
        law = engine.systems.law.metrics(pop, engine.systems.crime)
        lines += ["", "-- Police, courts and prison --"] + [f"{k}: {v:,.2f}" if (v % 1 and v < 10) else f"{k}: {v:,.0f}" if v >= 10 or v % 1 == 0 else f"{k}: {v:.2f}" for k, v in law.items()]
        lines += ["", "-- Recent conversations --"] + life.convo_log[-8:]
        return lines

    # ------------------------------------------------------------------
    def build(self, engine: Any, tick_id: int, status: str = "", segments: Optional[Dict[int, Any]] = None, full: bool = True) -> Dict[str, Any]:
        cal = engine.calendar
        world = engine.world
        people = engine.population.people
        pop_gen = engine.population._pop_gen
        if full or self.arr is None or self.arr["ids"].shape[0] != len(people) or pop_gen != self._gen_seen:
            full = True
            self._gen_seen = pop_gen
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
        patch: List[int] = []
        lastx: List[int] = []
        lasty: List[int] = []
        lens: List[int] = []
        get_index = index.get
        get_person = people.get
        get_times = times_by_pid.get
        for pid, seg in segments.items():
            i = get_index(pid)
            p = get_person(pid)
            if i is None or p is None:
                continue
            n_seg = len(seg)
            patch.append(i)
            lastx.append(seg[-1, 0])
            lasty.append(seg[-1, 1])
            mv_ids.append(pid)
            mv_mode.append(p.mode)
            chunks.append(seg)
            vt = get_times(pid)
            time_chunks.append(vt if vt is not None else _lin_times(n_seg))
            lens.append(n_seg)
        if patch:
            pidx = np.asarray(patch, dtype=np.int32)
            arr["x"][pidx] = lastx
            arr["y"][pidx] = lasty
            arr["indoors"][pidx] = 0
            arr["mode"][pidx] = mv_mode
            offsets = [0] + np.cumsum(lens).tolist()
        # Citizens who finished travelling this sub-step may have gone indoors.
        for pid in self.prev_movers - movers:
            i = index.get(pid)
            p = people.get(pid)
            if i is not None and p is not None:
                arr["x"][i], arr["y"][i] = p.location
                arr["indoors"][i] = 1 if p.floor >= 0 else 0
                patch.append(i)
        self.prev_movers = set(movers)
        if not full:
            n_people = len(self.plist)
            si = self.slice_i
            self.slice_i = (si + 1) % SUBSTEPS_PER_HOUR
            lo = n_people * si // SUBSTEPS_PER_HOUR
            hi = n_people * (si + 1) // SUBSTEPS_PER_HOUR
            if hi > lo:
                prev_hour = cal.hour if cal.minute else (cal.hour - 1) % 24
                self._fill(arr, self.plist[lo:hi], lo, prev_hour)
                patch.extend(range(lo, hi))
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
            "selected_stats": ({"Energy": selected.energy, "Happiness": selected.happiness, "Stress": selected.stress,
                                "Health": selected.health, "Motivation": selected.motivation}
                               if selected is not None else {}),
            "selected_route": [tuple(q) for q in selected_route],
            "status": status,
            "structure_version": getattr(world, "structure_version", 0),
            "building": None,
        }
        if full:
            frame.update({k: (v.copy() if k in ("x", "y", "indoors", "mode") else v) for k, v in arr.items()})
            frame.update(self.extras)
        else:
            # Delta frame: only the citizens whose position/mode changed.
            pi = np.asarray(sorted(set(patch)), dtype=np.int32)
            frame["delta"] = True
            frame["patch_idx"] = pi
            frame["patch"] = {k: arr[k][pi] for k in self._SLICE_KEYS}
            if cal.minute == 0:
                self._extras(engine)
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
