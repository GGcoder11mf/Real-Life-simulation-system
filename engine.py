"""
engine.py

Main simulation engine. This file coordinates time, weather, events,
schedules, save/load, and ties together the world, entities, systems,
and UI layers. The architecture is intentionally explicit and verbose
so the simulation can be expanded rather than treated like a toy.
"""

from __future__ import annotations

import curses
import gzip
import importlib.util
import json
import math
import pickle
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import buildings as bldg
from analytics import CityAnalytics
from dynamics import CityDynamics
from entities import Person, Population
from systems import SimulationSystems
from world import SUBSTEPS_PER_HOUR, SUBSTEP_MINUTES, DEFAULT_DENSITY, DEFAULT_WORLD_HEIGHT, DEFAULT_WORLD_WIDTH, WorldMap, clamp, coerce_rng_state


SAVE_FILE = "life_sim_save.pkl"

# World presets: (width, height, density).  "large" is the default and holds
# roughly ten times the population of the original "classic" map on a much
# finer tile grid.
WORLD_PRESETS = {
    "classic": (96, 48, 1.0),
    "small": (64, 32, 1.0),
    "large": (DEFAULT_WORLD_WIDTH, DEFAULT_WORLD_HEIGHT, DEFAULT_DENSITY),
}
SAVE_CONTAINER_MAGIC = b"RLS_SAVE_V2"
MSGPACK_AVAILABLE = importlib.util.find_spec("msgpack") is not None

if MSGPACK_AVAILABLE:
    import msgpack

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

SEASON_NAMES = ["Winter", "Spring", "Summer", "Autumn"]

WEATHER_TYPES = [
    "clear",
    "cloudy",
    "rain",
    "storm",
    "fog",
    "heatwave",
    "cold_snap",
]


@dataclass
class CalendarState:
    year: int = 2026
    month: int = 1
    day: int = 1
    hour: int = 6
    minute: int = 0
    day_index: int = 0

    def advance_hour(self) -> None:
        self.hour += 1
        if self.hour >= 24:
            self.hour = 0
            self.day += 1
            self.day_index += 1
            days_in_month = self.days_in_month()
            if self.day > days_in_month:
                self.day = 1
                self.month += 1
                if self.month > 12:
                    self.month = 1
                    self.year += 1

    def season(self) -> str:
        if self.month in (12, 1, 2):
            return "Winter"
        if self.month in (3, 4, 5):
            return "Spring"
        if self.month in (6, 7, 8):
            return "Summer"
        return "Autumn"

    def days_in_month(self) -> int:
        return {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
        }[self.month]

    def label(self) -> str:
        return f"{MONTH_NAMES[self.month - 1]} {self.day}, {self.year} {self.hour:02d}:{self.minute:02d}"


@dataclass
class WeatherState:
    condition: str
    temperature_c: float
    wind_kph: float
    precipitation: float
    cloud_cover: float
    severity: float
    mean_temperature_c: float = 15.0
    humidity: float = 0.5

    def summary(self) -> str:
        return (
            f"{self.condition.title()} "
            f"{self.temperature_c:0.1f}C "
            f"wind {self.wind_kph:0.0f}kph "
            f"rain {self.precipitation:0.2f}"
        )


@dataclass
class NewsItem:
    timestamp: str
    category: str
    headline: str
    detail: str
    importance: float


@dataclass
class TimelineEvent:
    timestamp: str
    event_type: str
    title: str
    payload: Dict[str, Any] = field(default_factory=dict)


class EventEngine:
    """
    Combines cross-system state into world-changing events.

    The event engine is intentionally not a story script. It watches
    stress, weather, local crime, economics, relationships, and karma
    and injects consequences back into the simulation state.
    """

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed + 999)

    def tick(self, engine: "SimulationEngine") -> List[TimelineEvent]:
        events: List[TimelineEvent] = []
        calendar = engine.calendar
        weather = engine.weather
        snapshot = engine.world.economic_history[-1] if engine.world.economic_history else None

        if snapshot and snapshot.unemployment_rate > 0.22 and self.rng.random() < 0.08:
            for company in engine.world.companies.values():
                company.bankruptcy_risk = clamp(company.bankruptcy_risk + 0.03, 0.0, 1.0)
            events.append(
                TimelineEvent(
                    timestamp=calendar.label(),
                    event_type="economy",
                    title="Local recession pressure increased across the city.",
                )
            )
        if weather.condition in {"storm", "heatwave", "cold_snap"} and self.rng.random() < 0.12:
            affected = self.rng.choice(list(engine.world.districts.values()))
            affected.crime_heat = clamp(affected.crime_heat + 0.03, 0.0, 1.0)
            affected.desirability = clamp(affected.desirability - 0.02, 0.0, 1.0)
            events.append(
                TimelineEvent(
                    timestamp=calendar.label(),
                    event_type="weather",
                    title=f"{weather.condition.title()} disrupted life in {affected.name}.",
                    payload={"district_id": affected.district_id},
                )
            )
        if self.rng.random() < 0.08:
            high_stress_people = [person for person in engine.population.people.values() if person.stress > 0.88]
        else:
            high_stress_people = []
        if high_stress_people:
            person = self.rng.choice(high_stress_people)
            person.add_memory("personal", "Had a severe stress spiral.", -0.12, timestamp=calendar.label())
            person.motivation = clamp(person.motivation - 0.06, 0.0, 1.0)
            events.append(
                TimelineEvent(
                    timestamp=calendar.label(),
                    event_type="personal",
                    title=f"{person.full_name} hit a critical stress threshold.",
                )
            )
        if self.rng.random() < 0.05:
            goodwill_people = [
                person for person in engine.population.people.values()
                if person.hidden_karma > 0.78 and person.reputation_public > 0.68
            ]
        else:
            goodwill_people = []
        if goodwill_people:
            person = self.rng.choice(goodwill_people)
            person.add_memory("social", "Received unexpected goodwill from the community.", 0.08, timestamp=calendar.label())
            person.bank_balance += 120.0
            events.append(
                TimelineEvent(
                    timestamp=calendar.label(),
                    event_type="social",
                    title=f"Community goodwill lifted {person.full_name}'s standing.",
                )
            )
        if calendar.hour == 8 and engine.world.companies and self.rng.random() < 0.06:
            company = self.rng.choice(list(engine.world.companies.values()))
            company.expansion_desire = clamp(company.expansion_desire + 0.1, 0.0, 1.0)
            events.append(
                TimelineEvent(
                    timestamp=calendar.label(),
                    event_type="company",
                    title=f"{company.name} signaled a new expansion push.",
                    payload={"company_id": company.company_id},
                )
            )
        return events


class SimulationEngine:
    """Full-life simulator runtime."""

    def __init__(
        self,
        seed: Optional[int] = None,
        world_seed: Optional[int] = None,
        preset: str = "large",
        world_size: Optional[tuple] = None,
        density: Optional[float] = None,
    ) -> None:
        self.seed = seed if seed is not None else self.generate_seed()
        self.rng = random.Random(self.seed)
        self.world_seed = world_seed if world_seed is not None else random.SystemRandom().randrange(1, 1_000_000_007)
        width, height, preset_density = WORLD_PRESETS.get(preset, WORLD_PRESETS["large"])
        if world_size is not None:
            width, height = int(world_size[0]), int(world_size[1])
        self.world = WorldMap(self.world_seed, width, height, density if density is not None else preset_density)
        self.population = Population(self.seed, self.world, size=self.world.recommended_population_size())
        labor_force, _, _ = self.population.labor_stats()
        self.world.calibrate_company_staffing(labor_force)
        self.systems = SimulationSystems(self.seed, self.world)
        self.dynamics = CityDynamics(self.seed, self.world)
        self.analytics = CityAnalytics()
        self._bootstrap_employment()
        bldg.refresh_structure(self.world, immediate=True)
        bldg.reassign_work_floors(self.world, self.population)
        self.inspected_building: Optional[int] = None
        self.world.record_economy_snapshot(0, self.population)
        self.calendar = CalendarState()
        self.weather = self.generate_weather()
        self.event_engine = EventEngine(self.seed)
        self.news: List[NewsItem] = []
        self.timeline: List[TimelineEvent] = []
        self.running = True
        self.paused = False
        self.tick_duration = 3.0     # wall-clock seconds per 2 simulated minutes (x40 real time)
        self.tick_counter = 0
        self.sub_counter = 0
        self.autosave_interval_days = 7
        self.last_autosave_day = 0
        self.selected_person_id: Optional[int] = min(self.population.people) if self.population.people else None
        self.camera_x = self.world.width // 2
        self.camera_y = self.world.height // 2
        self.message_cache: List[str] = []
        self._macro_metrics_cache_tick = -1
        self._macro_metrics_cache: Dict[str, float] = {}
        self.crime_opened_today = 0
        self.analytics.compute(self, 0, 0)
        if self.selected_person_id is not None:
            selected = self.population.people[self.selected_person_id]
            self.camera_x, self.camera_y = selected.location
        self.log_news(
            "engine",
            f"Simulation initialized from seed {self.seed} on world seed {self.world_seed}.",
            "A new city started running autonomously.",
            0.6,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _bootstrap_employment(self) -> None:
        for _ in range(20):
            self.systems.company.spawn_hiring_waves(force=True)
            self.systems.company.run_hiring(self.population)
        self.world.prune_empty_companies()
        self.systems.company.seed_baseline_vacancies()

    @staticmethod
    def generate_seed() -> int:
        return int(time.time()) % 1_000_000_007

    WEATHER_TRANSITIONS = {
        "clear": {"clear": 0.56, "cloudy": 0.27, "rain": 0.05, "storm": 0.01, "fog": 0.06, "heatwave": 0.03, "cold_snap": 0.02},
        "cloudy": {"clear": 0.28, "cloudy": 0.36, "rain": 0.23, "storm": 0.03, "fog": 0.08, "heatwave": 0.01, "cold_snap": 0.01},
        "rain": {"clear": 0.12, "cloudy": 0.34, "rain": 0.38, "storm": 0.09, "fog": 0.06, "heatwave": 0.0, "cold_snap": 0.01},
        "storm": {"clear": 0.1, "cloudy": 0.4, "rain": 0.37, "storm": 0.1, "fog": 0.02, "heatwave": 0.0, "cold_snap": 0.01},
        "fog": {"clear": 0.32, "cloudy": 0.36, "rain": 0.1, "storm": 0.0, "fog": 0.2, "heatwave": 0.0, "cold_snap": 0.02},
        "heatwave": {"clear": 0.2, "cloudy": 0.1, "rain": 0.02, "storm": 0.06, "fog": 0.0, "heatwave": 0.62, "cold_snap": 0.0},
        "cold_snap": {"clear": 0.2, "cloudy": 0.14, "rain": 0.02, "storm": 0.02, "fog": 0.04, "heatwave": 0.0, "cold_snap": 0.58},
    }
    DIURNAL_AMPLITUDE = {
        "clear": 6.0, "cloudy": 3.0, "rain": 2.5, "storm": 2.0, "fog": 2.0, "heatwave": 7.0, "cold_snap": 4.0,
    }

    def seasonal_mean_temperature(self) -> float:
        """Smooth annual temperature curve (coldest mid-January)."""
        cal = self.calendar
        day_of_year = (cal.month - 1) * 30.4 + cal.day
        return 19.0 - 12.5 * math.cos(2.0 * math.pi * (day_of_year - 18.0) / 365.0)

    def generate_weather(self) -> WeatherState:
        """Markov weather: conditions persist for days and respect the season."""
        seasonal = self.seasonal_mean_temperature()
        previous = getattr(self, "weather", None)
        table = dict(self.WEATHER_TRANSITIONS[previous.condition if previous is not None else "clear"])
        if seasonal < 22.0:
            table["heatwave"] = 0.0
        if seasonal > 12.0:
            table["cold_snap"] = 0.0
        if seasonal > 25.0:
            table["rain"] *= 0.8
        if seasonal < 4.0:
            table["fog"] *= 1.3
        conditions = list(table.keys())
        condition = self.rng.choices(conditions, weights=[max(0.0, table[c]) for c in conditions], k=1)[0]
        temp_shift = {
            "clear": self.rng.uniform(-1.5, 2.5),
            "cloudy": self.rng.uniform(-2, 1.5),
            "rain": self.rng.uniform(-3.5, 0.5),
            "storm": self.rng.uniform(-4, 1),
            "fog": self.rng.uniform(-3, 0),
            "heatwave": self.rng.uniform(5, 9),
            "cold_snap": self.rng.uniform(-9, -4),
        }[condition]
        if previous is not None:
            # Day-to-day temperature inertia.
            temp_shift = temp_shift * 0.6 + (previous.mean_temperature_c - seasonal) * 0.4
        precipitation = {
            "clear": 0.0,
            "cloudy": 0.05,
            "rain": self.rng.uniform(0.35, 0.8),
            "storm": self.rng.uniform(0.7, 1.0),
            "fog": 0.1,
            "heatwave": 0.0,
            "cold_snap": 0.0,
        }[condition]
        cloud_cover = {
            "clear": 0.1,
            "cloudy": 0.75,
            "rain": 0.88,
            "storm": 1.0,
            "fog": 0.65,
            "heatwave": 0.15,
            "cold_snap": 0.2,
        }[condition]
        mean_temp = seasonal + temp_shift
        severity = clamp(precipitation + abs(temp_shift) / 10.0, 0.0, 1.0)
        weather = WeatherState(
            condition=condition,
            temperature_c=round(mean_temp, 1),
            wind_kph=round(self.rng.uniform(4, 38) + severity * 22, 1),
            precipitation=round(precipitation, 2),
            cloud_cover=cloud_cover,
            severity=severity,
            mean_temperature_c=mean_temp,
            humidity=clamp(0.35 + precipitation * 0.5 + cloud_cover * 0.2, 0.0, 1.0),
        )
        self._apply_diurnal_temperature(weather, self.calendar.hour)
        return weather

    def _apply_diurnal_temperature(self, weather: WeatherState, hour: int) -> None:
        amplitude = self.DIURNAL_AMPLITUDE.get(weather.condition, 3.0)
        weather.temperature_c = round(weather.mean_temperature_c + amplitude * math.sin(2.0 * math.pi * (hour - 9) / 24.0), 1)

    # ------------------------------------------------------------------
    # Logging and timeline
    # ------------------------------------------------------------------
    def log_news(self, category: str, headline: str, detail: str, importance: float) -> None:
        item = NewsItem(self.calendar.label(), category, headline, detail, importance)
        self.news.append(item)
        self.news = self.news[-120:]
        self.message_cache.append(headline)
        self.message_cache = self.message_cache[-20:]

    def log_timeline(self, event_type: str, title: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = TimelineEvent(self.calendar.label(), event_type, title, payload or {})
        self.timeline.append(event)
        self.timeline = self.timeline[-200:]

    # ------------------------------------------------------------------
    # Main ticking
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance one full simulated hour (30 two-minute sub-steps)."""
        if self.paused:
            return
        for _ in range(SUBSTEPS_PER_HOUR):
            self.step_sub()
            if self.calendar.minute == 0:
                break

    def step_sub(self) -> Dict[int, Any]:
        """Advance two simulated minutes.

        At the top of the hour the full decision logic (schedules, needs,
        economy, events) runs; every sub-step moves travelling citizens along
        their routes.  Returns the waypoints each mover traversed."""
        if self.paused:
            return {}
        cal = self.calendar
        if cal.minute == 0:
            self._run_hour_start()
        self.world.sim_seconds = cal.day_index * 86400.0 + cal.hour * 3600.0 + cal.minute * 60.0
        segments = self.population.move_substep(cal.hour)
        self.sub_counter += 1
        cal.minute += SUBSTEP_MINUTES
        if cal.minute >= 60:
            cal.minute = 0
            self._finish_hour()
        return segments

    def _run_hour_start(self) -> None:
        current_hour = self.calendar.hour
        current_day = self.calendar.day_index
        self.world.bad_weather = self.weather.condition in ("rain", "storm", "cold_snap")
        if current_hour == 0:
            self._run_daily_cycle()
        self.population.hourly_update(current_day, current_hour)
        for text in self.dynamics.tick_hour(self, current_day, current_hour, self.population.last_commuters):
            self.log_news("health", text, "Reported by the city health department.", 0.6)
            self.log_timeline("health", text)
        self._apply_weather_effects_hourly()
        self._run_event_engine()

    def _finish_hour(self) -> None:
        self.calendar.advance_hour()
        if self.calendar.hour == 0:
            self.weather = self.generate_weather()
            self.log_news("weather", f"Forecast updated: {self.weather.condition}.", self.weather.summary(), 0.3)
        else:
            self._apply_diurnal_temperature(self.weather, self.calendar.hour)
        self.tick_counter += 1

    def _run_daily_cycle(self) -> None:
        day_index = self.calendar.day_index
        self.world.tick_daily_world(day_index, self.population)
        self.population.daily_update(day_index)
        report = self.systems.tick_daily(self.population, day_index)
        self.dynamics.revenue_today += self.systems.finance.tax_collected_today
        self.dynamics.revenue_today += self.population.take_property_tax()
        spending = self.population.take_consumer_spending()
        if spending > 0:
            self.dynamics.revenue_today += self.systems.company.receive_consumer_spending(spending)
        if day_index % 30 == 15:
            self.population.run_property_market(day_index)
        if day_index % 7 == 0 and bldg.refresh_structure(self.world):
            bldg.reassign_work_floors(self.world, self.population)
        dynamics_report = self.dynamics.tick_daily(self, day_index)
        self.population.price_level = self.dynamics.price_level
        self.world.inflation_rate = self.dynamics.inflation
        self.systems.law.safety_multiplier = self.dynamics.service["safety"]
        self.systems.crime.safety_multiplier = self.dynamics.service["safety"]
        for key, items in dynamics_report.items():
            if items:
                report.setdefault(key, []).extend(items)
        self.systems.company.seed_baseline_vacancies()
        self.world.record_economy_snapshot(day_index, self.population)
        self.analytics.compute(self, day_index, len(report.get("crime", [])))
        self._macro_metrics_cache_tick = -1
        self._ingest_system_report(report)
        self._daily_story_events()
        if day_index - self.last_autosave_day >= self.autosave_interval_days:
            self.save(SAVE_FILE)
            self.last_autosave_day = day_index

    def _ingest_system_report(self, report: Dict[str, List[str]]) -> None:
        for category, items in report.items():
            for text in items[:12]:
                importance = 0.4 if category in {"hiring", "social", "crime", "courts", "health", "budget"} else 0.25
                self.log_news(category, text, f"Generated by {category} system.", importance)
                self.log_timeline(category, text)

    def _apply_weather_effects_hourly(self) -> None:
        condition = self.weather.condition
        people = self.population.people.values()
        if condition == "storm":
            for person in people:
                s = person.stress + 0.004
                person.stress = 1.0 if s > 1.0 else s
                e = person.energy - 0.003
                person.energy = 0.0 if e < 0.0 else e
        elif condition == "heatwave":
            for person in people:
                e = person.energy - 0.004
                person.energy = 0.0 if e < 0.0 else e
        elif condition == "clear":
            for person in people:
                h = person.happiness + 0.0006
                person.happiness = 1.0 if h > 1.0 else h

    def _run_event_engine(self) -> None:
        events = self.event_engine.tick(self)
        for event in events:
            self.timeline.append(event)
            self.timeline = self.timeline[-400:]
            self.log_news(event.event_type, event.title, json.dumps(event.payload) if event.payload else "Triggered by event engine.", 0.55)

    def _daily_story_events(self) -> None:
        top_company = max(self.world.companies.values(), key=lambda company: company.market_value) if self.world.companies else None
        top_district = max(self.world.districts.values(), key=lambda district: district.desirability) if self.world.districts else None
        if top_company is not None and self.calendar.day_index % 7 == 0:
            self.log_news(
                "weekly",
                f"{top_company.name} led the market this week.",
                f"{top_company.stock_symbol} is the highest-valued listed company in the city.",
                0.45,
            )
        if top_district is not None and top_district.average_rent > 1300 and self.rng.random() < 0.2:
            self.log_news(
                "housing",
                f"Rents continued climbing in {top_district.name}.",
                f"Average rent reached {top_district.average_rent:.0f}.",
                0.35,
            )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 3,
            "seed": self.seed,
            "world_seed": self.world_seed,
            "calendar": self.calendar.__dict__,
            "weather": self.weather.__dict__,
            "world": self.world.to_dict(),
            "population": self.population.to_dict(),
            "systems": self.systems.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "analytics": self.analytics.to_dict(),
            "news": [item.__dict__ for item in self.news],
            "timeline": [event.__dict__ for event in self.timeline],
            "paused": self.paused,
            "running": self.running,
            "tick_duration": self.tick_duration,
            "tick_counter": self.tick_counter,
            "sub_counter": self.sub_counter,
            "autosave_interval_days": self.autosave_interval_days,
            "last_autosave_day": self.last_autosave_day,
            "selected_person_id": self.selected_person_id,
            "camera_x": self.camera_x,
            "camera_y": self.camera_y,
            "message_cache": self.message_cache,
            "rng_state": self.rng.getstate(),
            "event_rng_state": self.event_engine.rng.getstate(),
        }

    @staticmethod
    def _serialize_save_payload(data: Dict[str, Any]) -> bytes:
        codec = "msgpack" if MSGPACK_AVAILABLE else "pickle"
        if codec == "msgpack":
            payload = msgpack.packb(data, use_bin_type=True)
        else:
            payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return SAVE_CONTAINER_MAGIC + codec.encode("ascii") + b"\n" + payload

    @staticmethod
    def _deserialize_save_payload(blob: bytes) -> Dict[str, Any]:
        if blob.startswith(SAVE_CONTAINER_MAGIC):
            codec_and_payload = blob[len(SAVE_CONTAINER_MAGIC):]
            codec_line, payload = codec_and_payload.split(b"\n", 1)
            codec = codec_line.decode("ascii")
            if codec == "msgpack":
                if not MSGPACK_AVAILABLE:
                    raise RuntimeError("This save was written with msgpack, but msgpack is not installed.")
                return msgpack.unpackb(payload, raw=False, strict_map_key=False)
            return pickle.loads(payload)
        return pickle.loads(blob)

    def save(self, path: str) -> None:
        data = self.to_dict()
        payload = self._serialize_save_payload(data)
        with gzip.open(path, "wb", compresslevel=1) as handle:
            handle.write(payload)

    @staticmethod
    def _soft_cap(value: float, threshold: float, scale: float) -> float:
        if value <= threshold:
            return value
        return threshold + math.log10(1.0 + value - threshold) * scale

    def _stabilize_loaded_state(self) -> None:
        self.world.prune_empty_companies()
        self.world.normalize_company_structure()
        self.world.stabilize_economy_state()
        for person in self.population.people.values():
            person.cash = self._soft_cap(max(0.0, person.cash), 250_000.0, 40_000.0)
            person.bank_balance = self._soft_cap(max(0.0, person.bank_balance), 750_000.0, 90_000.0)
            person.assets = self._soft_cap(max(0.0, person.assets), 1_000_000.0, 120_000.0)
            person.debt = self._soft_cap(max(0.0, person.debt), 500_000.0, 80_000.0)
        for person_id in list(self.systems.law.inmates):
            person = self.population.people.get(person_id)
            if person is not None and not person.incarcerated:
                self.systems.law._incarcerate(person)
        for stock in self.systems.stock_market.stocks.values():
            company = self.world.companies.get(stock.company_id)
            if company is None:
                continue
            stock.price = round(max(1.0, company.market_value / 120_000.0), 2)
            stock.last_change = clamp(stock.last_change, -0.25, 0.25)

    @classmethod
    def load(cls, path: str) -> "SimulationEngine":
        with open(path, "rb") as probe:
            prefix = probe.read(2)
        if prefix == b"\x1f\x8b":
            with gzip.open(path, "rb") as handle:
                data = cls._deserialize_save_payload(handle.read())
        else:
            with open(path, "rb") as handle:
                data = pickle.load(handle)

        engine = cls.__new__(cls)
        engine.seed = data["seed"]
        engine.world_seed = data.get("world_seed", data["world"].get("seed", data["seed"]))
        engine.rng = random.Random(engine.seed)
        rng_state = coerce_rng_state(data.get("rng_state"))
        if rng_state is not None:
            engine.rng.setstate(rng_state)
        engine.calendar = CalendarState(**data["calendar"])
        engine.weather = WeatherState(**data["weather"])
        engine.world = WorldMap.from_dict(data["world"])
        engine.world_seed = engine.world.seed
        engine.population = Population.from_dict(
            data["population"],
            engine.world,
            schedule_day_index=engine.calendar.day_index,
        )
        engine.systems = SimulationSystems.from_dict(engine.seed, engine.world, data["systems"])
        engine.dynamics = CityDynamics(engine.seed, engine.world)
        engine.dynamics.load_dict(data.get("dynamics", {}), engine.population)
        engine.analytics = CityAnalytics()
        engine.analytics.load_dict(data.get("analytics", {}))
        engine.crime_opened_today = 0
        engine.inspected_building = None
        engine.news = [NewsItem(**item) for item in data["news"]]
        engine.timeline = [TimelineEvent(**event) for event in data["timeline"]]
        engine.paused = bool(data.get("paused", False))
        engine.running = bool(data.get("running", True))
        engine.tick_duration = data.get("tick_duration", 3.0)
        engine.tick_counter = int(data.get("tick_counter", 0))
        engine.sub_counter = int(data.get("sub_counter", 0))
        engine.autosave_interval_days = int(data.get("autosave_interval_days", 7))
        engine.last_autosave_day = int(data.get("last_autosave_day", 0))
        engine.selected_person_id = data.get("selected_person_id")
        engine.camera_x = int(data.get("camera_x", engine.world.width // 2))
        engine.camera_y = int(data.get("camera_y", engine.world.height // 2))
        engine.message_cache = list(data.get("message_cache", []))
        engine.event_engine = EventEngine(engine.seed)
        event_rng_state = coerce_rng_state(data.get("event_rng_state"))
        if event_rng_state is not None:
            engine.event_engine.rng.setstate(event_rng_state)
        engine._macro_metrics_cache_tick = -1
        engine._macro_metrics_cache = {}
        engine.weather.mean_temperature_c = getattr(engine.weather, "mean_temperature_c", engine.weather.temperature_c)
        if engine.selected_person_id not in engine.population.people:
            engine.selected_person_id = min(engine.population.people) if engine.population.people else None
        engine.camera_x = max(0, min(engine.world.width - 1, engine.camera_x))
        engine.camera_y = max(0, min(engine.world.height - 1, engine.camera_y))
        engine._stabilize_loaded_state()
        return engine

    # ------------------------------------------------------------------
    # Input / control helpers used by the terminal UI
    # ------------------------------------------------------------------
    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.log_news("engine", "Simulation paused." if self.paused else "Simulation resumed.", "Input toggled runtime pause state.", 0.1)

    def change_speed(self, faster: bool) -> None:
        if faster:
            self.tick_duration = max(0.03, self.tick_duration * 0.7)
        else:
            self.tick_duration = min(8.0, self.tick_duration * 1.4)

    def move_camera(self, dx: int, dy: int) -> None:
        self.camera_x = max(0, min(self.world.width - 1, self.camera_x + dx))
        self.camera_y = max(0, min(self.world.height - 1, self.camera_y + dy))

    def center_on_selected_person(self) -> None:
        person = self.selected_person()
        self.camera_x, self.camera_y = person.location

    def cycle_selected_person(self) -> None:
        ids = sorted(self.population.people.keys())
        if not ids:
            return
        if self.selected_person_id not in ids:
            self.selected_person_id = ids[0]
            return
        idx = ids.index(self.selected_person_id)
        self.selected_person_id = ids[(idx + 1) % len(ids)]

    def selected_person(self) -> Person:
        if self.selected_person_id is None or self.selected_person_id not in self.population.people:
            self.selected_person_id = min(self.population.people) if self.population.people else None
        if self.selected_person_id is None:
            raise RuntimeError("Population has no citizens.")
        return self.population.people[self.selected_person_id]

    def recent_headlines(self, limit: int = 8) -> List[str]:
        return [item.headline for item in self.news[-limit:]][::-1]

    def recent_timeline_lines(self, limit: int = 8) -> List[str]:
        return [f"{event.timestamp[-5:]} {event.title[:52]}" for event in self.timeline[-limit:]][::-1]

    # ------------------------------------------------------------------
    # Derived metrics for UI and debugging
    # ------------------------------------------------------------------
    def macro_metrics(self) -> Dict[str, float]:
        if self._macro_metrics_cache_tick == self.tick_counter and self._macro_metrics_cache:
            return self._macro_metrics_cache
        snapshot = self.world.economic_history[-1] if self.world.economic_history else None
        population = list(self.population.people.values())
        labor_force, employed, unemployed = self.population.labor_stats()
        unemployment = unemployed / labor_force if labor_force else 0.0
        avg_happiness = sum(person.happiness for person in population) / max(1, len(population))
        avg_stress = sum(person.stress for person in population) / max(1, len(population))
        avg_karma = sum(person.hidden_karma for person in population) / max(1, len(population))
        avg_health = sum(person.health for person in population) / max(1, len(population))
        incarceration = len(self.systems.law.inmates) / max(1, len(population))
        extra = {k: float(v) for k, v in self.analytics.latest.items()}
        self._macro_metrics_cache = {
            **extra,
            "unemployment": unemployment,
            "labor_force": float(labor_force),
            "employed": float(employed),
            "unemployed": float(unemployed),
            "vacancies": float(len(self.world.open_job_postings())),
            "population": float(len(population)),
            "births": float(self.population.births_total),
            "deaths": float(self.population.deaths_total),
            "housing_index": snapshot.housing_index if snapshot else 0.0,
            "rent_index": snapshot.rent_index if snapshot else 0.0,
            "crime_index": snapshot.crime_index if snapshot else 0.0,
            "avg_happiness": avg_happiness,
            "avg_stress": avg_stress,
            "avg_health": avg_health,
            "avg_karma": avg_karma,
            "incarceration_rate": incarceration,
        }
        self._macro_metrics_cache_tick = self.tick_counter
        return self._macro_metrics_cache

    def weather_impact_lines(self) -> List[str]:
        return [
            f"Season: {self.calendar.season()}",
            f"Condition: {self.weather.condition}",
            f"Temp: {self.weather.temperature_c:0.1f} C",
            f"Wind: {self.weather.wind_kph:0.0f} kph",
            f"Precip: {self.weather.precipitation:0.2f}",
            f"Severity: {self.weather.severity:0.2f}",
        ]

    # ------------------------------------------------------------------
    # Console runtime
    # ------------------------------------------------------------------
    def run_terminal(self) -> None:
        from ui import TerminalUI

        ui = TerminalUI(self)
        try:
            curses.wrapper(ui.run)
        except Exception:
            print("Simulation crashed with an uncaught exception:")
            print(traceback.format_exc())
            raise


def main() -> None:
    engine = SimulationEngine()
    engine.run_terminal()


if __name__ == "__main__":
    main()
