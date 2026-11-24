"""
world.py

The world module owns the simulation map and all place-based state.
It is intentionally large and heavily commented because the overall
project is a long-lived sandbox engine rather than a short script.

Design notes:
- The map is deterministic from a seed.
- Tiles are physical simulation locations.
- Districts influence wealth, zoning, crime, rent, and jobs.
- Buildings host homes, companies, schools, and civic services.
- Transport lines and roads matter because commute time is distance-driven.
- The economy is spatial: companies exist in buildings, not as abstract rows.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import buildings as bldg
from routing import RouteService, make_entry_cost_grid

TERRAIN_ORDER = [
    "water",
    "park",
    "rural",
    "suburban",
    "urban",
    "industrial",
    "road",
    "rail",
]

TERRAIN_GLYPHS = {
    "water": "~",
    "park": '"',
    "rural": ",",
    "suburban": ";",
    "urban": "#",
    "industrial": "%",
    "road": ".",
    "rail": "=",
}

TERRAIN_COLOR_KEYS = {
    "water": "water",
    "park": "park",
    "rural": "rural",
    "suburban": "suburban",
    "urban": "urban",
    "industrial": "industrial",
    "road": "road",
    "rail": "rail",
}

ZONE_TYPES = [
    "residential_low",
    "residential_mid",
    "residential_high",
    "commercial",
    "industrial",
    "civic",
    "mixed_use",
    "green",
]

BUILDING_TYPES = [
    "house",
    "apartment",
    "shop",
    "office",
    "factory",
    "school",
    "college",
    "police",
    "court",
    "prison",
    "hospital",
    "park_facility",
    "station",
    "warehouse",
]

COMPANY_SECTORS = [
    "retail",
    "logistics",
    "manufacturing",
    "software",
    "healthcare",
    "education",
    "food",
    "transport",
    "construction",
    "finance",
    "media",
    "public_service",
]

SAVE_FLOAT_DIGITS = 6
DEFAULT_WORLD_WIDTH = 224
DEFAULT_WORLD_HEIGHT = 112
DEFAULT_DENSITY = 1.15

# Physical scale: one tile is 20 m x 20 m, so the large map is ~4.5 km x 2.2 km.
TILE_METERS = 20.0
SUBSTEP_MINUTES = 2
SUBSTEPS_PER_HOUR = 60 // SUBSTEP_MINUTES
# Travel modes (codes are stored on citizens and sent to the renderer).
MODE_WALK, MODE_BIKE, MODE_CAR, MODE_BUS, MODE_TAXI = 0, 1, 2, 3, 4
MODE_NAMES = ["walk", "bike", "car", "bus", "taxi"]
MODE_KMH = [4.8, 14.0, 38.0, 22.0, 40.0]        # free-flow speeds between signals (stops add delay)
MODE_FATIGUE = [0.0004, 0.00015, 0.0, 0.00003, 0.0]   # energy lost per tile travelled
MODE_TRAFFIC_EXP = [0.0, 0.35, 1.0, 0.8, 1.0]


# Traffic signals: one shared timing function used by both the simulation (vehicles
# stop on red) and the renderer (the lights shown).  Time is simulated seconds.
SIGNAL_CYCLE = 120.0
SIGNAL_GREEN = 52.0
SIGNAL_YELLOW = 6.0


def signal_state(jid: int, axis: int, t: float) -> int:
    """0 green, 1 yellow, 2 red for traffic on `axis` (0 = north-south, 1 = east-west)."""
    phase = (t + jid * 37.0) % SIGNAL_CYCLE
    d = (phase - (0.0 if axis == 0 else SIGNAL_CYCLE / 2.0)) % SIGNAL_CYCLE
    if d < SIGNAL_GREEN:
        return 0
    if d < SIGNAL_GREEN + SIGNAL_YELLOW:
        return 1
    return 2


def time_to_green(jid: int, axis: int, t: float) -> float:
    """Seconds until traffic on `axis` next has a green (0 if green now)."""
    phase = (t + jid * 37.0) % SIGNAL_CYCLE
    d = (phase - (0.0 if axis == 0 else SIGNAL_CYCLE / 2.0)) % SIGNAL_CYCLE
    return 0.0 if d < SIGNAL_GREEN else SIGNAL_CYCLE - d


def mode_speeds(traffic_factor: float = 1.0):
    """Tiles per hour for each travel mode under a traffic factor."""
    out = []
    for kmh, exp in zip(MODE_KMH, MODE_TRAFFIC_EXP):
        out.append(max(1.0, kmh * 1000.0 / TILE_METERS * (traffic_factor ** exp)))
    return out
TILE_FLAG_ROAD = 1
TILE_FLAG_RAIL = 2
TILE_FLAG_TRANSIT_STOP = 4
TILE_TERRAIN_CODES = {name: idx for idx, name in enumerate(TERRAIN_ORDER)}
TILE_ZONE_CODES = {name: idx for idx, name in enumerate(ZONE_TYPES)}


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def coerce_rng_state(state: Any) -> Any:
    """msgpack turns the tuple returned by Random.getstate() into lists."""
    if state is None:
        return None
    if isinstance(state, (list, tuple)) and len(state) == 3:
        version, internal, gauss = state
        return (int(version), tuple(int(value) for value in internal), gauss)
    return state


def round_for_save(value: float, digits: int = SAVE_FLOAT_DIGITS) -> float:
    return round(value, digits)


@dataclass
class Tile:
    """Single physical location in the simulation."""

    x: int
    y: int
    terrain: str
    elevation: float
    moisture: float
    district_id: int = -1
    wealth: float = 0.0
    crime_pressure: float = 0.0
    property_value: float = 0.0
    commute_factor: float = 1.0
    movement_cost: float = 1.0
    zone: str = "green"
    road: bool = False
    rail: bool = False
    transit_stop: bool = False
    landmark: Optional[str] = None
    building_id: Optional[int] = None
    occupants: List[int] = field(default_factory=list)

    def glyph(self) -> str:
        if self.road:
            return TERRAIN_GLYPHS["road"]
        if self.rail:
            return TERRAIN_GLYPHS["rail"]
        return TERRAIN_GLYPHS.get(self.terrain, "?")

    def color_key(self) -> str:
        if self.road:
            return "road"
        if self.rail:
            return "rail"
        return TERRAIN_COLOR_KEYS.get(self.terrain, "urban")

    def is_walkable(self) -> bool:
        return self.terrain != "water"

    def supports_building(self) -> bool:
        return self.terrain in {"rural", "suburban", "urban", "industrial", "park"}

    def transit_bonus(self) -> float:
        bonus = 0.0
        if self.road:
            bonus += 0.2
        if self.rail:
            bonus += 0.35
        if self.transit_stop:
            bonus += 0.25
        return bonus


@dataclass
class District:
    """Macro-area aggregating social and economic pressures."""

    district_id: int
    name: str
    center: Tuple[int, int]
    wealth: float
    zoning_mix: Dict[str, float]
    dominant_terrain: str
    crime_heat: float = 0.0
    transit_score: float = 0.0
    desirability: float = 0.0
    population_capacity: int = 0
    building_ids: List[int] = field(default_factory=list)
    company_ids: List[int] = field(default_factory=list)
    landmark_ids: List[int] = field(default_factory=list)
    average_rent: float = 0.0
    average_home_price: float = 0.0

    def describe(self) -> str:
        zone_label = max(self.zoning_mix, key=self.zoning_mix.get)
        return (
            f"{self.name} [{zone_label}] "
            f"wealth={self.wealth:.2f} crime={self.crime_heat:.2f} "
            f"transit={self.transit_score:.2f}"
        )


@dataclass
class Building:
    """Buildings are concrete containers for households, jobs, and services."""

    building_id: int
    name: str
    building_type: str
    district_id: int
    tile: Tuple[int, int]
    capacity: int
    quality: float
    rent: float
    purchase_price: float
    service_score: float
    company_id: Optional[int] = None
    household_ids: List[int] = field(default_factory=list)
    job_slots: int = 0
    level: int = 1
    open_hours: Tuple[int, int] = (8, 18)
    inventory_pressure: float = 0.0
    maintenance_backlog: float = 0.0
    floors: int = 1
    units_per_floor: int = 1
    owner_id: Optional[int] = None
    owner_kind: str = "developer"      # person | developer | company | city
    value: float = 0.0
    maintenance_fund: float = 0.0
    anchor_x: int = -1          # top-left tile of the footprint (-1 = legacy single tile)
    anchor_y: int = -1
    width_tiles: int = 1
    height_tiles: int = 1
    facing: int = 0             # side that faces the street: 0 N, 1 E, 2 S, 3 W

    def residential_capacity(self) -> int:
        if self.building_type in {"house", "apartment"}:
            return self.capacity
        return 0

    def is_civic(self) -> bool:
        return self.building_type in {"school", "college", "police", "court", "prison", "hospital", "station"}

    def is_company_site(self) -> bool:
        return self.building_type in {"shop", "office", "factory", "warehouse", "hospital"}


@dataclass
class Company:
    """Procedurally generated employers and market actors."""

    company_id: int
    name: str
    sector: str
    district_id: int
    building_id: int
    market_value: float
    cash: float
    payroll_budget: float
    productivity: float
    demand: float
    supply: float
    prestige: float
    company_scale: str = "standard"
    open_positions: List[str] = field(default_factory=list)
    employee_ids: List[int] = field(default_factory=list)
    shift_templates: List[Tuple[int, int]] = field(default_factory=list)
    bankruptcy_risk: float = 0.0
    stock_symbol: str = ""
    layoffs_pending: int = 0
    expansion_desire: float = 0.0
    owner_id: Optional[int] = None
    min_headcount: int = 4
    target_headcount: int = 12
    max_headcount: int = 18

    def health_score(self) -> float:
        liquidity = clamp(self.cash / max(1.0, self.payroll_budget), 0.0, 1.0)
        balance = clamp((self.demand - self.supply) * 0.5 + 0.5, 0.0, 1.0)
        risk = clamp(1.0 - self.bankruptcy_risk, 0.0, 1.0)
        return (liquidity + balance + risk + self.productivity) / 4.0

    def fundamental_value(self) -> float:
        payroll_base = max(20_000.0, self.payroll_budget * 18.0)
        cash_buffer = clamp(self.cash, -self.payroll_budget * 6.0, self.payroll_budget * 24.0)
        quality = 0.55 + self.productivity * 0.35 + self.prestige * 0.15
        market_balance = 0.7 + clamp(self.demand - self.supply, -0.5, 0.5) * 0.6
        risk_drag = 1.1 - self.bankruptcy_risk * 0.55
        return max(25_000.0, (payroll_base + max(0.0, cash_buffer) * 2.2) * quality * market_balance * risk_drag)

    def has_shift_coverage(self) -> bool:
        return bool(self.shift_templates)

    def staffing_gap(self) -> int:
        return max(0, self.target_headcount - len(self.employee_ids) - len(self.open_positions))

    def below_minimum_staffing(self) -> bool:
        return len(self.employee_ids) + len(self.open_positions) < self.min_headcount


@dataclass
class TransitLine:
    """Coarse transport layer for commute reduction."""

    line_id: int
    name: str
    line_type: str
    stops: List[Tuple[int, int]]
    speed_multiplier: float
    district_ids: List[int] = field(default_factory=list)

    def nearest_stop_distance(self, x: int, y: int) -> int:
        return min(abs(x - sx) + abs(y - sy) for sx, sy in self.stops) if self.stops else 9999


@dataclass
class HousingListing:
    listing_id: int
    building_id: int
    district_id: int
    rent: float
    purchase_price: float
    quality: float
    bedrooms: int
    occupied: bool = False
    household_id: Optional[int] = None
    floor: int = 0
    unit_no: int = 1
    tenure: str = "rent"               # rent | own
    owner_id: Optional[int] = None     # landlord, or the occupant for owned units


@dataclass
class JobPosting:
    posting_id: int
    company_id: int
    district_id: int
    title: str
    salary: float
    required_skill: str
    required_level: float
    shift: Tuple[int, int]
    urgency: float
    open_days: int = 0


@dataclass
class EconomySnapshot:
    day_index: int
    unemployment_rate: float
    inflation: float
    wage_index: float
    housing_index: float
    transit_index: float
    crime_index: float
    labor_force: int = 0
    employed: int = 0
    unemployed: int = 0
    vacancies: int = 0
    rent_index: float = 0.0


class WorldMap:
    """
    The main simulation space.

    WorldMap owns the terrain grid, districts, buildings, companies,
    transport, markets, and lightweight spatial queries. It does not
    directly own AI logic for people, but it provides the physical
    structure on top of which the life simulation runs.
    """

    def __init__(self, seed: int, width: int = DEFAULT_WORLD_WIDTH, height: int = DEFAULT_WORLD_HEIGHT, density: Optional[float] = None) -> None:
        self._initialize_fields(seed, width, height, density)
        self.generate_world()
        self._build_routing_arrays()

    @classmethod
    def _empty(cls, seed: int, width: int = DEFAULT_WORLD_WIDTH, height: int = DEFAULT_WORLD_HEIGHT, density: Optional[float] = None) -> "WorldMap":
        """Create an ungenerated world shell for save loading."""
        world = cls.__new__(cls)
        world._initialize_fields(seed, width, height, density)
        return world

    def _initialize_fields(self, seed: int, width: int, height: int, density: Optional[float] = None) -> None:
        self.seed = seed
        self.width = width
        self.height = height
        # Linear tile scale relative to the original 96x48 map.  Finer grids
        # keep the same real-world speeds by scaling tiles-per-hour with it.
        self.tile_scale = max(0.5, width / 96.0)
        self.traffic_factor = 1.0
        self.mode_tph = mode_speeds(1.0)
        self.mode_step = [t / SUBSTEPS_PER_HOUR for t in self.mode_tph]   # tiles per 2-minute sub-step
        self.movers: set = set()
        self.sim_seconds = 0.0
        self.signal_boxes: List[Tuple[int, int, int, int]] = []
        self.signal_grid: List[List[int]] = []
        self.bad_weather = False
        self.taxi_revenue_today = 0.0
        self.aqi_health_drain = 0.0
        # Building density multiplier: finer tiles hold more (smaller) buildings.
        self.density = float(density) if density is not None else (DEFAULT_DENSITY if width * height > 8000 else 1.0)
        self.rng = random.Random(seed)
        self.tiles: List[List[Tile]] = []
        self.districts: Dict[int, District] = {}
        self.buildings: Dict[int, Building] = {}
        self.companies: Dict[int, Company] = {}
        self.transit_lines: Dict[int, TransitLine] = {}
        self.housing_market: Dict[int, HousingListing] = {}
        self.job_market: Dict[int, JobPosting] = {}
        self.economic_history: List[EconomySnapshot] = []
        self.landmarks: List[Tuple[str, Tuple[int, int]]] = []
        self.population_locations: Dict[int, Tuple[int, int]] = {}
        self._walkable_tiles: List[Tile] = []
        self._district_tile_cache: Dict[int, List[Tile]] = {}
        self._building_type_index: Dict[str, List[Building]] = {}
        self._park_tiles: List[Tuple[int, int]] = []
        self.route_service: Optional[RouteService] = None
        self._used_symbols: set = set()
        self.inflation_rate = 0.025
        self._building_listings: Dict[int, List[HousingListing]] = {}
        self.structure_version = 0
        self._building_counter = 1
        self._company_counter = 1
        self._listing_counter = 1
        self._posting_counter = 1
        self._line_counter = 1

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------
    def generate_world(self) -> None:
        self.tiles = self._generate_base_tiles()
        self._generate_districts()
        self._lay_roads()
        self._lay_transit()
        self._compute_signals()
        self._assign_zones()
        self._stamp_landmarks()
        self._generate_buildings()
        self._generate_companies()
        self._generate_housing_market()
        self._balance_housing()
        self._generate_job_market()
        self._assign_building_structure()
        self._recalculate_all_tile_values()
        self._rebuild_spatial_indexes()
        self.record_economy_snapshot(0)

    def _index_listings(self) -> None:
        index: Dict[int, List[HousingListing]] = {}
        for listing in self.housing_market.values():
            index.setdefault(listing.building_id, []).append(listing)
        for bid, items in index.items():
            items.sort(key=lambda l: l.listing_id)
        self._building_listings = index

    def _assign_building_structure(self) -> None:
        """Give every building floors, place units on floors, and set its value."""
        self._index_listings()
        for building in self.buildings.values():
            listings = self._building_listings.get(building.building_id, [])
            building.floors = bldg.initial_floors(building, listings, self.rng)
            if building.building_type in ("house", "apartment"):
                bldg.assign_residential_units(building, listings)
            if building.building_type in ("police", "court", "prison", "station", "school", "college", "hospital", "park_facility"):
                building.owner_kind = "city" if building.building_type != "hospital" else "city"
        bldg.refresh_structure(self, immediate=True)

    def _rebuild_spatial_indexes(self) -> None:
        self._index_listings()
        self._walkable_tiles = []
        self._district_tile_cache = {district_id: [] for district_id in self.districts}
        self._park_tiles = []
        for row in self.tiles:
            for tile in row:
                if tile.is_walkable():
                    self._walkable_tiles.append(tile)
                self._district_tile_cache.setdefault(tile.district_id, []).append(tile)
                if tile.terrain == "park":
                    self._park_tiles.append((tile.x, tile.y))
        self._building_type_index = {building_type: [] for building_type in BUILDING_TYPES}
        for building in self.buildings.values():
            self._building_type_index.setdefault(building.building_type, []).append(building)

    def _build_routing_arrays(self) -> None:
        walkable, cost, vcost = make_entry_cost_grid(self.tiles)
        if self.route_service is None:
            self.route_service = RouteService(walkable, cost, vcost)
        else:
            self.route_service.invalidate(walkable, cost, vcost)
        self.route_service.precompute_goals(self._popular_route_goals())

    def _popular_route_goals(self) -> List[Tuple[int, int]]:
        """Destinations citizens commonly route to: buildings, parks, transit."""
        goals = {(int(b.tile[0]), int(b.tile[1])) for b in self.buildings.values()}
        goals.update((int(x), int(y)) for x, y in self._park_tiles)
        for row in self.tiles:
            for tile in row:
                if tile.transit_stop:
                    goals.add((int(tile.x), int(tile.y)))
        return list(goals)

    def _generate_base_tiles(self) -> List[List[Tile]]:
        grid: List[List[Tile]] = []
        cx = self.width / 2.0
        cy = self.height / 2.0
        for y in range(self.height):
            row: List[Tile] = []
            for x in range(self.width):
                nx = x / max(1, self.width - 1)
                ny = y / max(1, self.height - 1)
                elevation = self._noise(nx * 4.0, ny * 4.0)
                moisture = self._noise(nx * 4.0 + 17.0, ny * 4.0 + 31.0)
                urban_pull = 1.0 - clamp((abs(x - cx) + abs(y - cy)) / (cx + cy), 0.0, 1.0)
                if elevation < 0.18:
                    terrain = "water"
                elif moisture > 0.78 and urban_pull < 0.65:
                    terrain = "park"
                elif urban_pull < 0.28:
                    terrain = "rural"
                elif urban_pull < 0.45:
                    terrain = "industrial" if moisture < 0.35 else "suburban"
                elif urban_pull < 0.7:
                    terrain = "suburban" if moisture > 0.2 else "industrial"
                else:
                    terrain = "urban"
                movement_cost = {
                    "water": 9.0,
                    "park": 1.25,
                    "rural": 1.4,
                    "suburban": 1.1,
                    "urban": 1.0,
                    "industrial": 1.2,
                }[terrain]
                row.append(
                    Tile(
                        x=x,
                        y=y,
                        terrain=terrain,
                        elevation=elevation,
                        moisture=moisture,
                        movement_cost=movement_cost,
                        wealth=0.4 + urban_pull * 0.5,
                    )
                )
            grid.append(row)
        return grid

    def _generate_districts(self) -> None:
        district_total = max(12, (self.width * self.height) // 900)
        centers: List[Tuple[int, int]] = []
        for _ in range(district_total):
            centers.append((self.rng.randrange(self.width), self.rng.randrange(self.height)))
        names_a = ["North", "South", "East", "West", "Central", "Lake", "River", "Hill", "Union", "Metro"]
        names_b = ["Heights", "Gardens", "Point", "Crossing", "Yard", "Park", "Commons", "Market", "District", "Ward"]
        for idx, center in enumerate(centers, start=1):
            terrain_here = self.tile_at(*center).terrain
            wealth = clamp(self.rng.random() * 0.8 + (0.15 if terrain_here == "urban" else 0.05), 0.05, 1.0)
            mix = {zone: self.rng.random() for zone in ZONE_TYPES}
            if terrain_here == "industrial":
                mix["industrial"] += 2.0
            if terrain_here == "urban":
                mix["commercial"] += 1.5
                mix["mixed_use"] += 1.0
            if terrain_here == "suburban":
                mix["residential_mid"] += 1.2
            if terrain_here == "rural":
                mix["residential_low"] += 1.2
                mix["green"] += 0.9
            total = sum(mix.values())
            zoning_mix = {k: v / total for k, v in mix.items()}
            district = District(
                district_id=idx,
                name=f"{self.rng.choice(names_a)} {self.rng.choice(names_b)}",
                center=center,
                wealth=wealth,
                zoning_mix=zoning_mix,
                dominant_terrain=terrain_here,
                crime_heat=clamp(0.7 - wealth + self.rng.random() * 0.3, 0.0, 1.0),
                transit_score=0.0,
                desirability=wealth,
            )
            self.districts[idx] = district
        # Nearest-district assignment (Manhattan Voronoi), vectorised.
        cx = np.array([c[0] for c in centers], dtype=np.int32)
        cy = np.array([c[1] for c in centers], dtype=np.int32)
        xs = np.arange(self.width, dtype=np.int32)[None, None, :]
        ys = np.arange(self.height, dtype=np.int32)[None, :, None]
        dist = np.abs(xs - cx[:, None, None]) + np.abs(ys - cy[:, None, None])
        nearest = dist.argmin(axis=0) + 1  # district ids start at 1
        self._district_map = nearest.astype(np.int32)
        for y in range(self.height):
            row = self.tiles[y]
            near_row = nearest[y]
            for x in range(self.width):
                district = self.districts[int(near_row[x])]
                tile = row[x]
                tile.district_id = district.district_id
                tile.wealth = clamp(district.wealth + self.rng.uniform(-0.18, 0.18), 0.0, 1.0)
                tile.crime_pressure = clamp(district.crime_heat + self.rng.uniform(-0.15, 0.15), 0.0, 1.0)

    # Street lattice per terrain class: (spacing_x, spacing_y).  Dense blocks downtown,
    # larger blocks in the suburbs and industrial estates, nothing in open country.
    LATTICE = {"urban": (8, 7), "suburban": (12, 10), "industrial": (16, 12)}

    def _lay_roads(self) -> None:
        """Arterials plus a street lattice whose block size depends on the area."""
        W, H = self.width, self.height
        x_roads = sorted({W // 4, W // 2, (W * 3) // 4})
        y_roads = sorted({H // 4, H // 2, (H * 3) // 4})
        for x in x_roads:
            for y in range(H):
                tile = self.tiles[y][x]
                if tile.terrain != "water":
                    tile.road = True
                    tile.movement_cost = min(tile.movement_cost, 0.75)
        for y in y_roads:
            for x in range(W):
                tile = self.tiles[y][x]
                if tile.terrain != "water":
                    tile.road = True
                    tile.movement_cost = min(tile.movement_cost, 0.75)
        for y in range(H):
            row = self.tiles[y]
            for x in range(W):
                tile = row[x]
                lattice = self.LATTICE.get(tile.terrain)
                if lattice is None or tile.road:
                    continue
                district = self.districts.get(tile.district_id)
                did = tile.district_id
                sx = lattice[0] + did % 3 - 1
                sy = lattice[1] + did % 2
                cx, cy = district.center if district is not None else (0, 0)
                if x % sx == cx % sx or y % sy == cy % sy:
                    tile.road = True
                    tile.movement_cost = min(tile.movement_cost, 0.8)

    def _compute_signals(self) -> None:
        """Find signalised junctions: clusters of road tiles with 3+ arms, at most 3x3 tiles."""
        H, W = self.height, self.width
        road = [[self.tiles[y][x].road for x in range(W)] for y in range(H)]

        def arms(x: int, y: int) -> int:
            n = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and road[ny][nx]:
                    n += 1
            return n

        inter = [[road[y][x] and arms(x, y) >= 3 for x in range(W)] for y in range(H)]
        seen = [[False] * W for _ in range(H)]
        boxes: List[Tuple[int, int, int, int]] = []
        for sy in range(H):
            for sx in range(W):
                if not inter[sy][sx] or seen[sy][sx]:
                    continue
                stack = [(sx, sy)]
                seen[sy][sx] = True
                x_lo = x_hi = sx
                y_lo = y_hi = sy
                while stack:
                    cx, cy = stack.pop()
                    x_lo, x_hi = min(x_lo, cx), max(x_hi, cx)
                    y_lo, y_hi = min(y_lo, cy), max(y_hi, cy)
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if 0 <= nx < W and 0 <= ny < H and inter[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((nx, ny))
                if x_hi - x_lo <= 2 and y_hi - y_lo <= 2:
                    boxes.append((x_lo, y_lo, x_hi, y_hi))
        grid = [[0] * W for _ in range(H)]
        for jid, (x_lo, y_lo, x_hi, y_hi) in enumerate(boxes, start=1):
            for y in range(y_lo, y_hi + 1):
                for x in range(x_lo, x_hi + 1):
                    if inter[y][x]:
                        grid[y][x] = jid
        self.signal_boxes = boxes
        self.signal_grid = grid

    def _lay_transit(self) -> None:
        main_stops_h = [(x, self.height // 2) for x in range(2, self.width - 2, max(6, self.width // 12))]
        main_stops_v = [(self.width // 2, y) for y in range(2, self.height - 2, max(4, self.height // 10))]
        for stops, label in [(main_stops_h, "Red Line"), (main_stops_v, "Blue Line")]:
            line_id = self._line_counter
            self._line_counter += 1
            line = TransitLine(
                line_id=line_id,
                name=label,
                line_type="metro",
                stops=stops,
                speed_multiplier=0.55,
            )
            district_ids = set()
            for x, y in stops:
                tile = self.tiles[y][x]
                if tile.terrain != "water":
                    tile.rail = True
                    tile.transit_stop = True
                    tile.road = True
                    district_ids.add(tile.district_id)
            line.district_ids = sorted(district_ids)
            self.transit_lines[line_id] = line
        for district in self.districts.values():
            district.transit_score = self._district_transit_score(district.district_id)

    def _assign_zones(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                district = self.districts[tile.district_id]
                if tile.terrain == "water":
                    tile.zone = "green"
                    continue
                if tile.terrain == "park":
                    tile.zone = "green"
                    continue
                tile.zone = max(district.zoning_mix, key=lambda zone: district.zoning_mix[zone] + self.rng.uniform(-0.2, 0.2))
                if tile.road and tile.zone.startswith("residential"):
                    tile.zone = "mixed_use" if self.rng.random() < 0.2 else tile.zone

    def _stamp_landmarks(self) -> None:
        landmark_names = ["Grand Station", "Civic Plaza", "Memorial Park", "Commerce Tower", "Harbor Exchange"]
        picks = self.rng.sample(
            [(x, y) for y in range(self.height) for x in range(self.width) if self.tiles[y][x].terrain != "water"],
            k=min(5, self.width * self.height // 200),
        )
        for name, (x, y) in zip(landmark_names, picks):
            tile = self.tiles[y][x]
            tile.landmark = name
            self.landmarks.append((name, (x, y)))

    # direction vectors for a lot's frontage: facing -> (road offset, depth step, width step)
    _FRONTAGE = {
        0: ((0, -1), (0, 1), (1, 0)),
        1: ((1, 0), (-1, 0), (0, 1)),
        2: ((0, 1), (0, -1), (1, 0)),
        3: ((-1, 0), (1, 0), (0, 1)),
    }

    def _generate_buildings(self) -> None:
        """Place buildings on lots that front a street; large buildings span several tiles."""
        H, W = self.height, self.width
        lots: List[Tuple[int, int]] = []
        for y in range(H):
            for x in range(W):
                tile = self.tiles[y][x]
                if tile.road or not tile.supports_building():
                    continue
                if self._road_sides(x, y):
                    lots.append((x, y))
        self.rng.shuffle(lots)
        scale = self.density / 1.15
        fill = {"urban": 0.82, "suburban": 0.62, "industrial": 0.55, "rural": 0.08, "park": 0.05}
        for x, y in lots:
            tile = self.tiles[y][x]
            if tile.building_id is not None or tile.road:
                continue
            if self.rng.random() > min(0.96, fill.get(tile.terrain, 0.3) * scale):
                continue
            btype = self._choose_building_type(tile)
            if btype is None:
                continue
            sides = self._road_sides(x, y)
            facing = sides[self.rng.randrange(len(sides))]
            width, depth = self._footprint_for(btype, tile)
            tiles = self._lot_tiles(x, y, facing, width, depth)
            if tiles is None:
                tiles = self._lot_tiles(x, y, facing, 1, 1)
                width = depth = 1
            self._spawn_building(tile, tiles, btype, facing)
        for district in self.districts.values():
            district.population_capacity = sum(self.buildings[b].residential_capacity() for b in district.building_ids)
            rents = [self.buildings[b].rent for b in district.building_ids if self.buildings[b].rent > 0]
            prices = [self.buildings[b].purchase_price for b in district.building_ids if self.buildings[b].purchase_price > 0]
            district.average_rent = sum(rents) / len(rents) if rents else 0.0
            district.average_home_price = sum(prices) / len(prices) if prices else 0.0

    def _road_sides(self, x: int, y: int) -> List[int]:
        sides = []
        for facing, ((dx, dy), _, _) in self._FRONTAGE.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.tiles[ny][nx].road:
                sides.append(facing)
        return sides

    def _lot_tiles(self, x: int, y: int, facing: int, width: int, depth: int) -> Optional[List[Tuple[int, int]]]:
        _, (ddx, ddy), (wdx, wdy) = self._FRONTAGE[facing]
        tiles = []
        for k in range(width):
            for j in range(depth):
                tx = x + wdx * k + ddx * j
                ty = y + wdy * k + ddy * j
                if not (0 <= tx < self.width and 0 <= ty < self.height):
                    return None
                t = self.tiles[ty][tx]
                if t.road or t.building_id is not None or not t.supports_building():
                    return None
                tiles.append((tx, ty))
        return tiles

    def _footprint_for(self, btype: str, tile: Tile) -> Tuple[int, int]:
        r = self.rng.random()
        wealth = self.districts[tile.district_id].wealth
        if btype == "apartment":
            return (2, 1) if r < 0.35 else (1, 1)
        if btype == "shop":
            return (2, 1) if r < 0.3 else (1, 1)
        if btype == "office":
            if r < 0.30 + wealth * 0.3:
                return (2, 2)
            return (2, 1) if r < 0.6 else (1, 1)
        if btype == "factory":
            return (3, 2) if r < 0.7 else (2, 2)
        if btype == "warehouse":
            return (3, 2) if r < 0.55 else (2, 2)
        return {
            "school": (3, 2), "college": (3, 3), "police": (2, 1), "hospital": (2, 2),
            "court": (2, 2), "prison": (3, 3), "station": (2, 1),
        }.get(btype, (1, 1))

    def _choose_building_type(self, tile: Tile) -> Optional[str]:
        district = self.districts[tile.district_id]
        zone = tile.zone
        r = self.rng.random()
        if tile.terrain == "park":
            return "park_facility"
        if zone == "green":
            if self.rng.random() < 0.5:
                tile.terrain = "park"
                tile.movement_cost = min(tile.movement_cost, 1.25)
            return None
        if zone == "commercial":
            return "office" if r < 0.25 + district.wealth * 0.5 else "shop"
        if zone == "industrial":
            return "factory" if r < 0.6 else "warehouse"
        if zone == "civic":
            if r < 0.34:
                return "school"
            if r < 0.46:
                return "college"
            if r < 0.62:
                return "police"
            if r < 0.78:
                return "hospital"
            if r < 0.97:
                return "station"
            return "court" if r < 0.985 else "prison"
        if zone == "mixed_use":
            return self.rng.choice(["apartment", "shop", "office", "apartment"])
        urban = tile.terrain == "urban"
        if zone == "residential_high":
            return "apartment" if (urban or r < 0.15) else "house"
        if zone == "residential_mid":
            return "apartment" if (urban and r < 0.55) else "house"
        return "apartment" if (urban and r < 0.12) else "house"

    def _spawn_building(self, door: Tile, tiles: List[Tuple[int, int]], building_type: str, facing: int) -> None:
        district = self.districts[door.district_id]
        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]
        area = len(tiles)
        quality = clamp(district.wealth + self.rng.uniform(-0.2, 0.2), 0.1, 1.0)
        growth = area ** 0.8
        capacity = int({
            "house": self.rng.randint(2, 6),
            "apartment": self.rng.randint(8, 32),
            "shop": self.rng.randint(4, 16),
            "office": self.rng.randint(10, 48),
            "factory": self.rng.randint(15, 64),
            "warehouse": self.rng.randint(8, 24),
            "school": self.rng.randint(40, 120),
            "college": self.rng.randint(80, 220),
            "police": self.rng.randint(20, 60),
            "hospital": self.rng.randint(30, 90),
            "station": self.rng.randint(10, 30),
            "park_facility": self.rng.randint(4, 10),
            "court": self.rng.randint(20, 50),
            "prison": self.rng.randint(40, 120),
        }.get(building_type, self.rng.randint(5, 15)) * (growth if building_type not in ("house", "park_facility") else 1.0))
        capacity = max(2, capacity)
        base_value = 400 + district.wealth * 2600 + quality * 1800
        if building_type in {"office", "hospital", "college"}:
            base_value *= 1.8
        if building_type in {"factory", "warehouse"}:
            base_value *= 1.3
        building = Building(
            building_id=self._building_counter,
            name=f"{district.name} {building_type.title()} {self._building_counter}",
            building_type=building_type,
            district_id=district.district_id,
            tile=(door.x, door.y),
            capacity=capacity,
            quality=quality,
            rent=round(base_value * 0.28, 2) if building_type in {"house", "apartment"} else 0.0,
            purchase_price=round(base_value * 85, 2) if building_type in {"house", "apartment"} else 0.0,
            service_score=quality + district.transit_score * 0.3,
            job_slots=capacity if building_type not in {"house", "apartment"} else 0,
            level=self.rng.randint(1, 6) if building_type in {"apartment", "office"} else 1,
            open_hours=(6, 22) if building_type in {"shop", "hospital", "station"} else (8, 18),
            anchor_x=min(xs),
            anchor_y=min(ys),
            width_tiles=max(xs) - min(xs) + 1,
            height_tiles=max(ys) - min(ys) + 1,
            facing=facing,
        )
        self.buildings[building.building_id] = building
        for tx, ty in tiles:
            self.tiles[ty][tx].building_id = building.building_id
        district.building_ids.append(building.building_id)
        self._building_counter += 1

    def _generate_companies(self) -> None:
        for building in self.buildings.values():
            if not building.is_company_site():
                continue
            if self.rng.random() > self._company_spawn_chance(building):
                continue
            sector = self._sector_for_building(building.building_type)
            symbol = self._make_stock_symbol(sector)
            company_scale, min_headcount, target_headcount, max_headcount = self._company_scale_profile(building)
            company = Company(
                company_id=self._company_counter,
                name=self._company_name(sector, company_scale),
                sector=sector,
                district_id=building.district_id,
                building_id=building.building_id,
                market_value=round(self.rng.uniform(250_000, 10_000_000) * (0.65 + target_headcount / 70.0), 2),
                cash=round(self.rng.uniform(30_000, 600_000) * (0.7 + target_headcount / 60.0), 2),
                payroll_budget=round(self.rng.uniform(8_000, 120_000) * (0.7 + target_headcount / 55.0), 2),
                productivity=clamp(self.rng.uniform(0.35, 0.95), 0.0, 1.0),
                demand=clamp(self.rng.uniform(0.2, 0.95), 0.0, 1.0),
                supply=clamp(self.rng.uniform(0.2, 0.95), 0.0, 1.0),
                prestige=clamp(self.rng.uniform(0.2, 0.9), 0.0, 1.0),
                company_scale=company_scale,
                stock_symbol=symbol,
                shift_templates=self._default_shifts_for_sector(sector),
                expansion_desire=self.rng.uniform(0.0, 1.0),
                bankruptcy_risk=self.rng.uniform(0.0, 0.5),
                min_headcount=min_headcount,
                target_headcount=target_headcount,
                max_headcount=max_headcount,
            )
            building.company_id = company.company_id
            self.companies[company.company_id] = company
            self.districts[company.district_id].company_ids.append(company.company_id)
            self._company_counter += 1

    def _balance_housing(self) -> None:
        """Thin out surplus residential lots so vacancy stays plausible (~25-35%)."""
        target_jobs = sum(c.target_headcount for c in self.companies.values())
        ceiling = max(300, int(self.width * self.height * 0.5 * min(1.0, self.density / 1.15)))
        target_pop = min(int(target_jobs * 1.45), ceiling)
        limit = int(target_pop * 1.35)
        capacity = self.residential_population_capacity()
        if capacity <= limit:
            return
        by_building: Dict[int, List[int]] = {}
        for lid, listing in self.housing_market.items():
            by_building.setdefault(listing.building_id, []).append(lid)
        order = list(by_building)
        self.rng.shuffle(order)
        for bid in order:
            if capacity <= limit:
                break
            building = self.buildings.get(bid)
            if building is None:
                continue
            for lid in by_building[bid]:
                listing = self.housing_market.pop(lid)
                capacity -= self.PERSONS_PER_UNIT
            for ty in range(building.anchor_y, building.anchor_y + building.height_tiles):
                for tx in range(building.anchor_x, building.anchor_x + building.width_tiles):
                    if 0 <= tx < self.width and 0 <= ty < self.height and self.tiles[ty][tx].building_id == bid:
                        self.tiles[ty][tx].building_id = None
            district = self.districts.get(building.district_id)
            if district is not None and bid in district.building_ids:
                district.building_ids.remove(bid)
            del self.buildings[bid]

    def _generate_housing_market(self) -> None:
        for building in self.buildings.values():
            if building.building_type not in {"house", "apartment"}:
                continue
            units = max(1, building.capacity // 2) if building.building_type == "apartment" else 1
            for _ in range(units):
                listing = HousingListing(
                    listing_id=self._listing_counter,
                    building_id=building.building_id,
                    district_id=building.district_id,
                    rent=round(building.rent * self.rng.uniform(0.9, 1.15), 2),
                    purchase_price=round(building.purchase_price * self.rng.uniform(0.85, 1.25), 2),
                    quality=building.quality,
                    bedrooms=(self.rng.choice([1, 1, 2, 2, 3]) if building.building_type == "apartment" else max(1, min(4, building.capacity // 3))),
                )
                self.housing_market[listing.listing_id] = listing
                self._listing_counter += 1

    def _generate_job_market(self) -> None:
        titles_by_sector = {
            "retail": ["Cashier", "Supervisor", "Buyer"],
            "logistics": ["Driver", "Dispatcher", "Coordinator"],
            "manufacturing": ["Assembler", "Operator", "Foreman"],
            "software": ["Developer", "Analyst", "Administrator"],
            "healthcare": ["Nurse", "Clerk", "Technician"],
            "education": ["Teacher", "Advisor", "Registrar"],
            "food": ["Cook", "Server", "Manager"],
            "transport": ["Conductor", "Planner", "Mechanic"],
            "construction": ["Laborer", "Estimator", "Supervisor"],
            "finance": ["Clerk", "Accountant", "Advisor"],
            "media": ["Writer", "Editor", "Producer"],
            "public_service": ["Officer", "Inspector", "Caseworker"],
        }
        skill_by_sector = {
            "retail": "service",
            "logistics": "logistics",
            "manufacturing": "engineering",
            "software": "software",
            "healthcare": "medicine",
            "education": "teaching",
            "food": "service",
            "transport": "transport",
            "construction": "engineering",
            "finance": "finance",
            "media": "communication",
            "public_service": "law",
        }
        for company in self.companies.values():
            target_posts = min(company.max_headcount, max(1, company.min_headcount // 3))
            for _ in range(target_posts):
                title = self.rng.choice(titles_by_sector[company.sector])
                self.add_job_posting(
                    company,
                    title=title,
                    required_skill=skill_by_sector[company.sector],
                    salary=round(self.rng.uniform(1800, 8500), 2),
                    required_level=round(self.rng.uniform(0.1, 0.8), 2),
                    shift=self.rng.choice(company.shift_templates or [(9, 17)]),
                    urgency=self.rng.uniform(0.2, 1.0),
                )

    def add_job_posting(
        self,
        company: Company,
        title: str,
        required_skill: str,
        salary: Optional[float] = None,
        required_level: Optional[float] = None,
        shift: Optional[Tuple[int, int]] = None,
        urgency: Optional[float] = None,
    ) -> JobPosting:
        posting = JobPosting(
            posting_id=self._posting_counter,
            company_id=company.company_id,
            district_id=company.district_id,
            title=title,
            salary=round(self.rng.uniform(1800, 8500), 2) if salary is None else salary,
            required_skill=required_skill,
            required_level=round(self.rng.uniform(0.1, 0.8), 2) if required_level is None else required_level,
            shift=tuple(shift) if shift is not None else (9, 17),
            urgency=self.rng.uniform(0.2, 1.0) if urgency is None else urgency,
        )
        self.job_market[posting.posting_id] = posting
        company.open_positions.append(title)
        self._posting_counter += 1
        return posting

    # ------------------------------------------------------------------
    # Economy and space updates
    # ------------------------------------------------------------------
    def _static_tile_arrays(self) -> None:
        """Cache per-tile static terrain arrays used by the vectorised value pass."""
        H, W = self.height, self.width
        road = np.zeros((H, W), dtype=np.float32)
        water = np.zeros((H, W), dtype=np.float32)
        industrial = np.zeros((H, W), dtype=np.float32)
        park = np.zeros((H, W), dtype=np.float32)
        dmap = np.zeros((H, W), dtype=np.int32)
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if tile.road:
                    road[y, x] = 1.0
                terrain = tile.terrain
                if terrain == "water":
                    water[y, x] = 1.0
                elif terrain == "industrial":
                    industrial[y, x] = 1.0
                elif terrain == "park":
                    park[y, x] = 1.0
                dmap[y, x] = tile.district_id
        transit = np.full((H, W), 9999.0, dtype=np.float32)
        xs = np.arange(W, dtype=np.float32)[None, :]
        ys = np.arange(H, dtype=np.float32)[:, None]
        for line in self.transit_lines.values():
            for sx, sy in line.stops:
                transit = np.minimum(transit, np.abs(xs - sx) + np.abs(ys - sy))
        reach = 18.0 * self.tile_scale
        self._st_road = road
        self._st_water = water
        self._st_industrial = industrial
        self._st_park = park
        self._st_transit = np.clip(1.0 - transit / reach, 0.0, 1.0).astype(np.float32)
        self._district_map = dmap
        self._flat_tiles = [tile for row in self.tiles for tile in row]

    def _recalculate_all_tile_values(self) -> None:
        if getattr(self, "_st_road", None) is None or self._st_road.shape != (self.height, self.width):
            self._static_tile_arrays()
        max_id = max(self.districts) if self.districts else 0
        wealth = np.zeros(max_id + 1, dtype=np.float32)
        crime = np.zeros(max_id + 1, dtype=np.float32)
        for did, district in self.districts.items():
            wealth[did] = district.wealth
            crime[did] = district.crime_heat
        dmap = np.clip(self._district_map, 0, max_id)
        w = wealth[dmap]
        c = crime[dmap]
        transit = self._st_transit
        value = np.maximum(
            50.0,
            500.0
            + w * 2500.0
            + transit * 900.0
            + self._st_road * (0.2 * 700.0)
            + self._st_park * (0.15 * 500.0)
            - c * 900.0
            - self._st_water * (0.3 * 200.0)
            - self._st_industrial * (0.18 * 250.0),
        )
        commute = np.clip(1.45 - transit - self._st_road * 0.2, 0.35, 1.8)
        self.property_value_grid = value
        self.crime_grid = c
        for tile, pv, cf in zip(self._flat_tiles, value.ravel().tolist(), commute.ravel().tolist()):
            tile.property_value = pv
            tile.commute_factor = cf
        self._refresh_ambient(value)

    def _refresh_ambient(self, value: np.ndarray) -> None:
        """Per-tile hourly mood deltas, read by Person in O(1) from flat lists."""
        if getattr(self, "_st_crime_pressure", None) is None or self._st_crime_pressure.shape != (self.height, self.width):
            cp = np.zeros((self.height, self.width), dtype=np.float32)
            for y, row in enumerate(self.tiles):
                for x, tile in enumerate(row):
                    cp[y, x] = tile.crime_pressure
            self._st_crime_pressure = cp
        max_id = max(self.districts) if self.districts else 0
        transit = np.zeros(max_id + 1, dtype=np.float32)
        for did, district in self.districts.items():
            transit[did] = district.transit_score
        t = transit[np.clip(self._district_map, 0, max_id)]
        stress = self._st_crime_pressure * 0.006 - t * 0.003
        happy = (value / 4000.0) * 0.001
        self._amb_stress = stress.ravel().tolist()
        self._amb_happy = happy.ravel().tolist()

    def ambient_at(self, x: int, y: int) -> Tuple[float, float]:
        i = (y % self.height) * self.width + (x % self.width)
        return self._amb_stress[i], self._amb_happy[i]

    def tick_daily_world(self, day_index: int, population: Optional[Any] = None) -> None:
        for district in self.districts.values():
            wealth_shift = self.rng.uniform(-0.02, 0.02)
            crime_shift = self.rng.uniform(-0.015, 0.015)
            district.wealth = clamp(district.wealth + wealth_shift, 0.03, 1.0)
            district.crime_heat = clamp(district.crime_heat + crime_shift - district.wealth * 0.01, 0.0, 1.0)
            district.desirability = clamp(
                district.wealth * 0.55 + district.transit_score * 0.25 + (1.0 - district.crime_heat) * 0.2,
                0.0,
                1.0,
            )
        for company in self.companies.values():
            market_wave = self.rng.uniform(-0.08, 0.08)
            company.demand = clamp(company.demand + market_wave, 0.0, 1.0)
            company.supply = clamp(company.supply + self.rng.uniform(-0.06, 0.06), 0.0, 1.0)
            company.productivity = clamp(company.productivity + self.rng.uniform(-0.03, 0.03), 0.0, 1.0)
            company.bankruptcy_risk = clamp(
                company.bankruptcy_risk
                + 0.04 * max(0.0, company.supply - company.demand)
                - company.productivity * 0.02
                - 0.004,
                0.0,
                1.0,
            )
            company.cash += (company.demand - company.supply) * 10_000.0 + self.rng.uniform(-2500, 2500)
            company.cash = clamp(company.cash, -company.payroll_budget * 12.0, company.payroll_budget * 40.0)
            fundamental_value = company.fundamental_value()
            mean_reversion = 0.06
            daily_noise = self.rng.uniform(-0.015, 0.015)
            valuation_shift = clamp(
                (fundamental_value - company.market_value) / max(25_000.0, company.market_value) * mean_reversion + daily_noise,
                -0.08,
                0.08,
            )
            company.market_value = max(25_000.0, company.market_value * (1.0 + valuation_shift))
            if company.bankruptcy_risk > 0.78 and company.cash < 0:
                company.layoffs_pending = max(company.layoffs_pending, self.rng.randint(1, 5))
        self._recalculate_all_tile_values()
        self.refresh_markets()

    def stabilize_economy_state(self) -> None:
        for company in self.companies.values():
            payroll_floor = max(1_000.0, company.payroll_budget)
            company.cash = clamp(company.cash, -payroll_floor * 12.0, payroll_floor * 40.0)
            company.market_value = min(max(25_000.0, company.market_value), company.fundamental_value() * 6.0)

    def normalize_company_structure(self) -> None:
        self.prune_empty_companies()
        for company in self.companies.values():
            building = self.buildings.get(company.building_id)
            if building is None:
                continue
            company.open_positions = [
                posting.title
                for posting in self.job_market.values()
                if posting.company_id == company.company_id
            ]
            if not company.company_scale or company.company_scale == "standard" and company.max_headcount == 18:
                company.company_scale, company.min_headcount, company.target_headcount, company.max_headcount = self._company_scale_profile(building)
            building.job_slots = max(building.job_slots, company.max_headcount)

    def company_has_live_openings(self, company_id: int) -> bool:
        return any(posting.company_id == company_id for posting in self.job_market.values())

    def remove_company(self, company_id: int) -> bool:
        company = self.companies.get(company_id)
        if company is None:
            return False
        stale_postings = [
            posting_id
            for posting_id, posting in self.job_market.items()
            if posting.company_id == company_id
        ]
        for posting_id in stale_postings:
            del self.job_market[posting_id]
        district = self.districts.get(company.district_id)
        if district is not None:
            district.company_ids = [cid for cid in district.company_ids if cid != company_id]
        building = self.buildings.get(company.building_id)
        if building is not None and building.company_id == company_id:
            building.company_id = None
        del self.companies[company_id]
        return True

    def prune_empty_companies(self, include_with_openings: bool = False) -> List[Company]:
        removed: List[Company] = []
        for company in list(self.companies.values()):
            has_openings = self.company_has_live_openings(company.company_id)
            if company.employee_ids or (has_openings and not include_with_openings):
                continue
            removed.append(company)
            self.remove_company(company.company_id)
        return removed

    def refresh_markets(self) -> None:
        expired = [
            pid for pid, posting in self.job_market.items()
            if posting.open_days >= 45
        ]
        for pid in expired:
            posting = self.job_market.pop(pid)
            company = self.companies.get(posting.company_id)
            if company and posting.title in company.open_positions:
                company.open_positions.remove(posting.title)
        for posting in self.job_market.values():
            posting.open_days += 1
            posting.urgency = clamp(posting.urgency + 0.02, 0.0, 1.0)
        # Housing tracks inflation (daily) plus a desirability premium; the old
        # curve was centred below 1.0 and deflated prices ~30% a year.
        drift = self.inflation_rate / 365.0
        for listing in self.housing_market.values():
            district = self.districts[listing.district_id]
            noise = self.rng.uniform(-0.0004, 0.0004)
            rent_growth = clamp(1.0 + drift + (district.desirability - 0.5) * 0.0006 + noise, 0.997, 1.003)
            price_growth = clamp(1.0 + drift * 1.2 + (district.desirability - 0.5) * 0.0010 + noise, 0.996, 1.004)
            listing.rent = round(max(250.0, listing.rent * rent_growth), 2)
            listing.purchase_price = round(max(40_000.0, listing.purchase_price * price_growth), 2)

    def record_economy_snapshot(self, day_index: int, population: Optional[Any] = None) -> None:
        open_jobs = len([posting for posting in self.job_market.values() if posting.open_days < 45])
        housing_values = [listing.purchase_price for listing in self.housing_market.values()]
        rents = [listing.rent for listing in self.housing_market.values()]
        crime = [district.crime_heat for district in self.districts.values()]
        transit = [district.transit_score for district in self.districts.values()]
        labor_force = employed = unemployed = 0
        if population is not None:
            labor_force, employed, unemployed = population.labor_stats()
        unemployment_rate = unemployed / labor_force if labor_force else 0.0
        snapshot = EconomySnapshot(
            day_index=day_index,
            unemployment_rate=round(unemployment_rate, 4),
            inflation=round(self.rng.uniform(0.01, 0.07), 3),
            wage_index=round(sum(company.payroll_budget for company in self.companies.values()) / max(1, len(self.companies)), 2),
            housing_index=round(sum(housing_values) / max(1, len(housing_values)), 2),
            transit_index=round(sum(transit) / max(1, len(transit)), 2),
            crime_index=round(sum(crime) / max(1, len(crime)), 2),
            labor_force=labor_force,
            employed=employed,
            unemployed=unemployed,
            vacancies=open_jobs,
            rent_index=round(sum(rents) / max(1, len(rents)), 2),
        )
        if self.economic_history and self.economic_history[-1].day_index == day_index:
            self.economic_history[-1] = snapshot
        else:
            self.economic_history.append(snapshot)
        if len(self.economic_history) > 90:
            self.economic_history = self.economic_history[-90:]

    PERSONS_PER_UNIT = 1.9      # observed household size; units, not bedrooms, limit who can be housed

    def residential_population_capacity(self) -> int:
        return int(len(self.housing_market) * self.PERSONS_PER_UNIT)

    def recommended_population_size(self) -> int:
        target_jobs = sum(company.target_headcount for company in self.companies.values())
        housing_capacity = self.residential_population_capacity()
        desired = int(target_jobs * 1.45)
        # Keep tick cost predictable across seeds: ~0.5 citizens per tile at most.
        ceiling = max(300, int(self.width * self.height * 0.5 * min(1.0, self.density / 1.15)))
        return max(300, min(housing_capacity, desired, ceiling))

    def calibrate_company_staffing(self, labor_force: int) -> None:
        companies = list(self.companies.values())
        current_target = sum(company.target_headcount for company in companies)
        desired_target = int(labor_force * 0.92)
        if current_target <= 0 or desired_target <= 0:
            return
        scale = min(1.0, desired_target / current_target)
        for company in companies:
            company.target_headcount = max(
                company.min_headcount,
                int(round(company.target_headcount * scale)),
            )
            company.max_headcount = max(
                company.target_headcount + 2,
                int(round(company.max_headcount * scale)),
            )

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------
    def tile_at(self, x: int, y: int) -> Tile:
        return self.tiles[y % self.height][x % self.width]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, x: int, y: int) -> Iterable[Tile]:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                yield self.tiles[ny][nx]

    def random_walkable_tile(self) -> Tile:
        if self._walkable_tiles:
            return self.rng.choice(self._walkable_tiles)
        while True:
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)
            tile = self.tiles[y][x]
            if tile.is_walkable():
                return tile

    def district_tiles(self, district_id: int) -> List[Tile]:
        cached = self._district_tile_cache.get(district_id)
        if cached is not None:
            return cached
        return [tile for row in self.tiles for tile in row if tile.district_id == district_id]

    def find_route(self, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = 2000) -> List[Tuple[int, int]]:
        return self.route_service.find_route(tuple(start), tuple(goal), max_steps=max_steps)

    def find_route_array(self, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = 2000, kind: int = 0) -> np.ndarray:
        return self.route_service.find_route_array(tuple(start), tuple(goal), max_steps=max_steps, kind=kind)

    def commute_time(self, start: Tuple[int, int], goal: Tuple[int, int], vehicle_bonus: float = 1.0) -> float:
        path = self.find_route(start, goal, max_steps=1500)
        if not path:
            return 0.0
        travel = 0.0
        for x, y in path:
            tile = self.tile_at(x, y)
            travel += tile.movement_cost * tile.commute_factor
        distance = len(path)
        rail_bonus = 0.85 if any(self.tile_at(x, y).rail for x, y in path) else 1.0
        return max(1.0, travel * 4.0 * rail_bonus / max(0.2, vehicle_bonus) + distance * 0.15)

    def estimate_commute_time(self, start: Tuple[int, int], goal: Tuple[int, int], vehicle_bonus: float = 1.0) -> float:
        """Rough commute duration in minutes, using the same tile speeds as movement."""
        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if distance <= 1:
            return 1.0
        start_tile = self.tile_at(*start)
        goal_tile = self.tile_at(*goal)
        transit_factor = 1.0 - min(0.35, (start_tile.transit_bonus() + goal_tile.transit_bonus()) * 0.35)
        # vehicle_bonus 1.0 = on foot; larger values are faster modes.
        tiles_per_hour = self.mode_tph[MODE_WALK] * max(0.2, vehicle_bonus) * transit_factor
        return max(1.0, distance / tiles_per_hour * 60.0)

    def register_population_position(self, person_id: int, x: int, y: int) -> None:
        old = self.population_locations.get(person_id)
        if old:
            old_tile = self.tile_at(*old)
            if person_id in old_tile.occupants:
                old_tile.occupants.remove(person_id)
        tile = self.tile_at(x, y)
        if person_id not in tile.occupants:
            tile.occupants.append(person_id)
        self.population_locations[person_id] = (x, y)

    def remove_population_position(self, person_id: int) -> None:
        old = self.population_locations.pop(person_id, None)
        if old:
            tile = self.tile_at(*old)
            if person_id in tile.occupants:
                tile.occupants.remove(person_id)

    # ------------------------------------------------------------------
    # Queries used by other simulation layers
    # ------------------------------------------------------------------
    def building_by_type(self, building_type: str) -> List[Building]:
        cached = self._building_type_index.get(building_type)
        if cached is not None:
            return cached
        return [building for building in self.buildings.values() if building.building_type == building_type]

    def available_housing(self) -> List[HousingListing]:
        return [listing for listing in self.housing_market.values() if not listing.occupied]

    def open_job_count(self) -> int:
        companies = self.companies
        buildings = self.buildings
        count = 0
        for posting in self.job_market.values():
            company = companies.get(posting.company_id)
            if company is not None and company.building_id in buildings:
                count += 1
        return count

    def open_job_postings(self) -> List[JobPosting]:
        valid_postings = []
        for posting in self.job_market.values():
            company = self.companies.get(posting.company_id)
            if company is None:
                continue
            if company.building_id not in self.buildings:
                continue
            valid_postings.append(posting)
        return sorted(valid_postings, key=lambda posting: (-posting.urgency, posting.salary))

    def district_summary_lines(self, limit: int = 8) -> List[str]:
        districts = sorted(self.districts.values(), key=lambda district: district.desirability, reverse=True)
        lines = []
        for district in districts[:limit]:
            lines.append(
                f"{district.name[:18]:18} W:{district.wealth:0.2f} C:{district.crime_heat:0.2f} "
                f"T:{district.transit_score:0.2f} R:{district.average_rent:0.0f}"
            )
        return lines

    def company_summary_lines(self, limit: int = 8) -> List[str]:
        companies = sorted(self.companies.values(), key=lambda company: company.market_value, reverse=True)
        lines = []
        for company in companies[:limit]:
            lines.append(
                f"{company.stock_symbol:4} {company.name[:20]:20} "
                f"{company.company_scale[:3].upper()} H:{company.health_score():0.2f} "
                f"J:{len(company.employee_ids):3d}/{company.target_headcount:3d} "
                f"R:{company.bankruptcy_risk:0.2f}"
            )
        return lines

    def landmarks_near(self, x: int, y: int, radius: int = 6) -> List[str]:
        found = []
        for name, (lx, ly) in self.landmarks:
            if abs(x - lx) + abs(y - ly) <= radius:
                found.append(name)
        return found

    def _serialize_economic_history(self) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for snapshot in self.economic_history[-90:]:
            history.append(
                {
                    "day_index": snapshot.day_index,
                    "unemployment_rate": round_for_save(snapshot.unemployment_rate),
                    "inflation": round_for_save(snapshot.inflation),
                    "wage_index": round_for_save(snapshot.wage_index, 2),
                    "housing_index": round_for_save(snapshot.housing_index, 2),
                    "transit_index": round_for_save(snapshot.transit_index),
                    "crime_index": round_for_save(snapshot.crime_index),
                    "labor_force": snapshot.labor_force,
                    "employed": snapshot.employed,
                    "unemployed": snapshot.unemployed,
                    "vacancies": snapshot.vacancies,
                    "rent_index": round_for_save(snapshot.rent_index, 2),
                }
            )
        return history

    def _serialize_tiles_compact(self) -> Dict[str, Any]:
        terrain: List[int] = []
        elevation: List[float] = []
        moisture: List[float] = []
        district_id: List[int] = []
        wealth: List[float] = []
        crime_pressure: List[float] = []
        property_value: List[float] = []
        commute_factor: List[float] = []
        movement_cost: List[float] = []
        zone: List[int] = []
        flags: List[int] = []
        landmarks: Dict[int, str] = {}
        building_ids: Dict[int, int] = {}
        for idx, tile in enumerate(tile for row in self.tiles for tile in row):
            terrain.append(TILE_TERRAIN_CODES[tile.terrain])
            elevation.append(round_for_save(tile.elevation))
            moisture.append(round_for_save(tile.moisture))
            district_id.append(tile.district_id)
            wealth.append(round_for_save(tile.wealth))
            crime_pressure.append(round_for_save(tile.crime_pressure))
            property_value.append(round_for_save(tile.property_value, 2))
            commute_factor.append(round_for_save(tile.commute_factor))
            movement_cost.append(round_for_save(tile.movement_cost))
            zone.append(TILE_ZONE_CODES.get(tile.zone, 0))
            tile_flags = 0
            if tile.road:
                tile_flags |= TILE_FLAG_ROAD
            if tile.rail:
                tile_flags |= TILE_FLAG_RAIL
            if tile.transit_stop:
                tile_flags |= TILE_FLAG_TRANSIT_STOP
            flags.append(tile_flags)
            if tile.landmark is not None:
                landmarks[idx] = tile.landmark
            if tile.building_id is not None:
                building_ids[idx] = tile.building_id
        return {
            "schema": 2,
            "terrain_order": list(TERRAIN_ORDER),
            "zone_types": list(ZONE_TYPES),
            "terrain": terrain,
            "elevation": elevation,
            "moisture": moisture,
            "district_id": district_id,
            "wealth": wealth,
            "crime_pressure": crime_pressure,
            "property_value": property_value,
            "commute_factor": commute_factor,
            "movement_cost": movement_cost,
            "zone": zone,
            "flags": flags,
            "landmarks": landmarks,
            "building_ids": building_ids,
        }

    @staticmethod
    def _deserialize_tiles_compact(width: int, height: int, data: Dict[str, Any]) -> List[List[Tile]]:
        terrain_order = data.get("terrain_order", TERRAIN_ORDER)
        zone_types = data.get("zone_types", ZONE_TYPES)
        terrain_codes = data["terrain"]
        elevation = data["elevation"]
        moisture = data["moisture"]
        district_id = data["district_id"]
        wealth = data["wealth"]
        crime_pressure = data["crime_pressure"]
        property_value = data["property_value"]
        commute_factor = data["commute_factor"]
        movement_cost = data["movement_cost"]
        zone_codes = data["zone"]
        flags = data["flags"]
        landmarks = {int(k): v for k, v in data.get("landmarks", {}).items()}
        building_ids = {int(k): v for k, v in data.get("building_ids", {}).items()}
        tiles: List[List[Tile]] = []
        for y in range(height):
            row: List[Tile] = []
            for x in range(width):
                idx = y * width + x
                tile_flags = flags[idx]
                row.append(
                    Tile(
                        x=x,
                        y=y,
                        terrain=terrain_order[terrain_codes[idx]],
                        elevation=elevation[idx],
                        moisture=moisture[idx],
                        district_id=district_id[idx],
                        wealth=wealth[idx],
                        crime_pressure=crime_pressure[idx],
                        property_value=property_value[idx],
                        commute_factor=commute_factor[idx],
                        movement_cost=movement_cost[idx],
                        zone=zone_types[zone_codes[idx]],
                        road=bool(tile_flags & TILE_FLAG_ROAD),
                        rail=bool(tile_flags & TILE_FLAG_RAIL),
                        transit_stop=bool(tile_flags & TILE_FLAG_TRANSIT_STOP),
                        landmark=landmarks.get(idx),
                        building_id=building_ids.get(idx),
                        occupants=[],
                    )
                )
            tiles.append(row)
        return tiles

    # ------------------------------------------------------------------
    # Serialization support
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 3,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "density": self.density,
            "tiles_compact": self._serialize_tiles_compact(),
            "districts": {k: v.__dict__ for k, v in self.districts.items()},
            "buildings": {k: v.__dict__ for k, v in self.buildings.items()},
            "companies": {k: v.__dict__ for k, v in self.companies.items()},
            "transit_lines": {k: v.__dict__ for k, v in self.transit_lines.items()},
            "housing_market": {k: v.__dict__ for k, v in self.housing_market.items()},
            "job_market": {k: v.__dict__ for k, v in self.job_market.items()},
            "economic_history": self._serialize_economic_history(),
            "landmarks": self.landmarks,
            "rng_state": self.rng.getstate(),
            "counters": {
                "building": self._building_counter,
                "company": self._company_counter,
                "listing": self._listing_counter,
                "posting": self._posting_counter,
                "line": self._line_counter,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldMap":
        world = cls._empty(seed=data["seed"], width=data["width"], height=data["height"], density=data.get("density"))
        if "tiles_compact" in data:
            world.tiles = cls._deserialize_tiles_compact(data["width"], data["height"], data["tiles_compact"])
        else:
            world.tiles = [[Tile(**tile_data) for tile_data in row] for row in data["tiles"]]
        world.districts = {
            int(k): District(
                **{
                    **v,
                    "center": tuple(v["center"]),
                }
            )
            for k, v in data["districts"].items()
        }
        world.buildings = {
            int(k): Building(
                **{
                    **v,
                    "tile": tuple(v["tile"]),
                    "open_hours": tuple(v.get("open_hours", (8, 18))),
                }
            )
            for k, v in data["buildings"].items()
        }
        world.companies = {
            int(k): Company(
                **{
                    **v,
                    "shift_templates": [tuple(shift) for shift in v.get("shift_templates", [])],
                }
            )
            for k, v in data["companies"].items()
        }
        world.transit_lines = {
            int(k): TransitLine(
                **{
                    **v,
                    "stops": [tuple(stop) for stop in v.get("stops", [])],
                }
            )
            for k, v in data["transit_lines"].items()
        }
        world.housing_market = {int(k): HousingListing(**v) for k, v in data["housing_market"].items()}
        world.job_market = {
            int(k): JobPosting(
                **{
                    **v,
                    "shift": tuple(v.get("shift", (9, 17))),
                }
            )
            for k, v in data["job_market"].items()
        }
        world.economic_history = [EconomySnapshot(**snap) for snap in data["economic_history"]]
        world.landmarks = [(str(name), tuple(tile)) for name, tile in data["landmarks"]]
        counters = data.get("counters", {})
        world._building_counter = counters.get("building", max(world.buildings, default=0) + 1)
        world._company_counter = counters.get("company", max(world.companies, default=0) + 1)
        world._listing_counter = counters.get("listing", max(world.housing_market, default=0) + 1)
        world._posting_counter = counters.get("posting", max(world.job_market, default=0) + 1)
        world._line_counter = counters.get("line", max(world.transit_lines, default=0) + 1)
        rng_state = coerce_rng_state(data.get("rng_state"))
        if rng_state is not None:
            world.rng.setstate(rng_state)
        world.ensure_unique_symbols()
        world._compute_signals()
        world._rebuild_spatial_indexes()
        world._recalculate_all_tile_values()
        world._build_routing_arrays()
        return world

    # ------------------------------------------------------------------
    # Internal deterministic helpers
    # ------------------------------------------------------------------
    def _noise(self, x: float, y: float) -> float:
        value = (
            math.sin(x * 1.31 + self.seed * 0.013)
            + math.cos(y * 1.73 - self.seed * 0.009)
            + math.sin((x + y) * 0.77)
            + math.cos((x - y) * 0.41)
        )
        return clamp((value + 4.0) / 8.0, 0.0, 1.0)

    def _district_transit_score(self, district_id: int) -> float:
        tiles = self.district_tiles(district_id)
        if not tiles:
            return 0.0
        stop_count = sum(1 for tile in tiles if tile.transit_stop)
        road_count = sum(1 for tile in tiles if tile.road)
        return clamp(stop_count / max(1, len(tiles) // 24) + road_count / max(1, len(tiles) // 10) * 0.1, 0.0, 1.0)

    def _nearest_transit_bonus(self, x: int, y: int) -> float:
        best = 9999
        for line in self.transit_lines.values():
            best = min(best, line.nearest_stop_distance(x, y))
        return clamp(1.0 - best / 18.0, 0.0, 1.0)

    def _company_name(self, sector: str, company_scale: str) -> str:
        if company_scale == "indie":
            founders = ["Elm", "Northlane", "Copper", "Juniper", "Bluebird", "Foxglove", "Lantern", "Brickhouse"]
            boutique_suffix = ["Studio", "Collective", "Corner", "Atelier", "Workshop", "Co-op"]
            return f"{self.rng.choice(founders)} {self.rng.choice(boutique_suffix)}"
        if company_scale == "industrial":
            industrial_prefix = ["Titan", "Forge", "Iron", "Continental", "Monolith", "Prime", "National", "Steelline"]
            industrial_suffix = ["Works", "Holdings", "Industrial", "Manufacturing", "Infrastructure", "Logistics"]
            return f"{self.rng.choice(industrial_prefix)} {sector.title()} {self.rng.choice(industrial_suffix)}"
        standard_prefix = ["Atlas", "Union", "Metro", "Summit", "Harbor", "Silver", "Civic", "Granite", "Signal", "Pioneer"]
        standard_suffix = ["Group", "Partners", "Services", "Network", "Exchange", "Enterprises"]
        return f"{self.rng.choice(standard_prefix)} {sector.title()} {self.rng.choice(standard_suffix)}"

    def _company_spawn_chance(self, building: Building) -> float:
        chances = {
            "shop": 0.14,
            "office": 0.16,
            "factory": 0.48,
            "warehouse": 0.38,
            "hospital": 0.24,
        }
        return chances.get(building.building_type, 0.2)

    def _company_scale_profile(self, building: Building) -> Tuple[str, int, int, int]:
        if building.building_type in {"factory", "warehouse"}:
            if self.rng.random() < 0.35:
                min_headcount = self.rng.randint(50, 85)
                target_headcount = self.rng.randint(max(60, min_headcount), 125)
                max_headcount = self.rng.randint(target_headcount + 12, target_headcount + 40)
                building.job_slots = max(building.job_slots, max_headcount)
                building.capacity = max(building.capacity, min_headcount // 2)
                return ("industrial", min_headcount, target_headcount, max_headcount)
            target_headcount = self.rng.randint(40, 85)
            max_headcount = self.rng.randint(target_headcount + 8, target_headcount + 30)
            building.job_slots = max(building.job_slots, max_headcount)
            building.capacity = max(building.capacity, target_headcount // 3)
            return ("standard", 30, target_headcount, max_headcount)
        if building.building_type == "shop":
            if self.rng.random() < 0.55:
                target_headcount = self.rng.randint(5, 18)
                max_headcount = self.rng.randint(target_headcount + 2, target_headcount + 8)
                building.job_slots = max(building.job_slots, max_headcount)
                return ("indie", 3, target_headcount, max_headcount)
            target_headcount = self.rng.randint(24, 55)
            max_headcount = self.rng.randint(target_headcount + 6, target_headcount + 18)
            building.job_slots = max(building.job_slots, max_headcount)
            return ("standard", 16, target_headcount, max_headcount)
        if building.building_type == "office":
            if self.rng.random() < 0.25:
                target_headcount = self.rng.randint(10, 28)
                max_headcount = self.rng.randint(target_headcount + 4, target_headcount + 12)
                building.job_slots = max(building.job_slots, max_headcount)
                return ("indie", 5, target_headcount, max_headcount)
            if self.rng.random() < 0.15:
                target_headcount = self.rng.randint(70, 120)
                max_headcount = self.rng.randint(target_headcount + 12, target_headcount + 35)
                building.job_slots = max(building.job_slots, max_headcount)
                return ("industrial", 50, target_headcount, max_headcount)
            target_headcount = self.rng.randint(35, 85)
            max_headcount = self.rng.randint(target_headcount + 8, target_headcount + 25)
            building.job_slots = max(building.job_slots, max_headcount)
            return ("standard", 22, target_headcount, max_headcount)
        if building.building_type == "hospital":
            target_headcount = self.rng.randint(45, 95)
            max_headcount = self.rng.randint(target_headcount + 8, target_headcount + 28)
            building.job_slots = max(building.job_slots, max_headcount)
            return ("standard", 30, target_headcount, max_headcount)
        target_headcount = self.rng.randint(20, 60)
        max_headcount = self.rng.randint(target_headcount + 6, target_headcount + 20)
        building.job_slots = max(building.job_slots, max_headcount)
        return ("standard", 12, target_headcount, max_headcount)

    def _make_stock_symbol(self, sector: str) -> str:
        """A ticker no other company has used (delisted symbols stay reserved)."""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for attempt in range(400):
            tail = 2 if attempt < 200 else 3      # widen the namespace if a sector's 2-letter space fills up
            symbol = (sector[:2].upper() + "".join(self.rng.choice(letters) for _ in range(tail)))[:4 if tail == 2 else 5]
            if symbol not in self._used_symbols:
                self._used_symbols.add(symbol)
                return symbol
        raise RuntimeError("could not allocate a unique stock symbol")

    def ensure_unique_symbols(self) -> int:
        """Repair duplicate tickers (e.g. from older saves); returns how many were renamed."""
        seen: set = set()
        renamed = 0
        self._used_symbols = {c.stock_symbol for c in self.companies.values()}
        for company in sorted(self.companies.values(), key=lambda c: c.company_id):
            if company.stock_symbol in seen or not company.stock_symbol:
                company.stock_symbol = self._make_stock_symbol(company.sector)
                renamed += 1
            seen.add(company.stock_symbol)
        self._used_symbols |= seen
        return renamed

    def _sector_for_building(self, building_type: str) -> str:
        mapping = {
            "shop": ["retail", "food", "finance", "media"],
            "office": ["software", "finance", "media", "construction", "public_service"],
            "factory": ["manufacturing", "construction", "logistics"],
            "warehouse": ["logistics", "transport"],
            "hospital": ["healthcare"],
        }
        return self.rng.choice(mapping.get(building_type, COMPANY_SECTORS))

    def _default_shifts_for_sector(self, sector: str) -> List[Tuple[int, int]]:
        shifts = {
            "retail": [(8, 16), (12, 20)],
            "logistics": [(6, 14), (14, 22)],
            "manufacturing": [(6, 14), (14, 22), (22, 6)],
            "software": [(9, 17), (10, 18)],
            "healthcare": [(7, 15), (15, 23), (23, 7)],
            "education": [(8, 16)],
            "food": [(7, 15), (15, 23)],
            "transport": [(5, 13), (13, 21)],
            "construction": [(7, 15)],
            "finance": [(9, 17)],
            "media": [(9, 17), (12, 20)],
            "public_service": [(8, 16), (16, 0)],
        }
        return list(shifts.get(sector, [(9, 17)]))
