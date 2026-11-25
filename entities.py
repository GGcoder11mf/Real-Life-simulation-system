"""
entities.py

This module models people as long-running autonomous agents.
No citizen is simulation-special; the UI can inspect any person,
but the world logic treats everyone uniformly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from movement import drive_batch, drive_vehicle, signal_array
from typing import Any, Dict, List, Optional, Tuple

from world import (
    MODE_BIKE, MODE_BUS, MODE_CAR, MODE_FATIGUE, MODE_KMH, MODE_TAXI, MODE_WALK, TILE_METERS, time_to_green,
    Building, HousingListing, JobPosting, WorldMap, clamp, coerce_rng_state, round_for_save,
)


SKILL_KEYS = [
    "software",
    "service",
    "logistics",
    "engineering",
    "medicine",
    "teaching",
    "transport",
    "finance",
    "communication",
    "law",
    "fitness",
    "craft",
]

PERSONALITY_AXES = [
    "discipline",
    "sociability",
    "risk_tolerance",
    "empathy",
    "ambition",
    "impulsiveness",
    "resilience",
]

PERSON_SAVE_FLOAT_DIGITS = 6
VEHICLE_TPH = {"none": 6, "bus_pass": 8, "bike": 9, "car": 12}
VEHICLE_SPEED_BONUS = {"none": 1.0, "bike": 2.9, "bus_pass": 3.5, "car": 5.8}
VEHICLE_DEFAULT_MODE = {"none": MODE_WALK, "bike": MODE_BIKE, "bus_pass": MODE_BUS, "car": MODE_CAR}
EDU_MULT = {"none": 1.0, "school": 1.1, "certificate": 1.22, "college": 1.32}
EDU_BONUS = {"none": 0.0, "school": 0.25, "certificate": 0.45, "college": 0.65}
_EMPTY_ROUTE = np.empty((0, 2), dtype=np.int32)
WEEKEND_SECTORS = {"retail", "food", "healthcare", "transport", "public_service", "logistics"}
WORK_TRAIT_KEYS = [
    "work_ethic",
    "reliability",
    "teamwork",
    "adaptability",
    "learning_drive",
]


FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Avery", "Casey", "Cameron", "Parker", "Riley", "Quinn",
    "Elliot", "Reese", "Devon", "Hayden", "Rowan", "Skyler", "Drew", "Sam", "Jamie", "Logan",
    "Maya", "Noah", "Priya", "Liam", "Sofia", "Mateo", "Amara", "Ethan", "Zara", "Omar", "Lena", "Kai", "Nora", "Theo",
    "Aisha", "Felix", "Ingrid", "Raj", "Camila", "Isaac", "Yuki", "Hana", "Diego", "Freya", "Ahmed", "Clara", "Marcus",
    "Esme", "Tariq", "Greta", "Jonas", "Leila", "Oscar", "Mina", "Bruno", "Alma", "Idris", "Vera", "Caleb", "Dalia",
    "Emil", "Farah", "Gideon", "Helena", "Ivan", "Junia", "Kofi", "Lucia", "Milo", "Nadia", "Orla", "Pablo", "Rina",
    "Silas", "Tessa", "Ugo", "Vikram", "Wren", "Xavier", "Yara", "Zane", "Anika", "Bashir", "Celia", "Dario", "Elif",
    "Florin", "Gia", "Hugo", "Imani", "Joaquin", "Katya", "Lior", "Mabel", "Nikos", "Odette", "Pilar", "Quentin",
    "Rosa", "Stefan", "Talia", "Umar", "Violet", "Willa", "Ximena", "Yusuf", "Zelda", "Arjun", "Beatriz", "Cyrus",
]

LAST_NAMES = [
    "Patel", "Reed", "Carter", "Shaw", "Rivera", "Bennett", "Morris", "Nguyen", "Coleman", "Santos",
    "Brooks", "Foster", "Hayes", "Ward", "Bailey", "Kim", "Murphy", "Price", "Diaz", "Powell",
    "Okafor", "Lindqvist", "Haddad", "Moreau", "Tanaka", "Novak", "Silva", "Andersson", "Rossi", "Kowalski", "Mbeki",
    "Fischer", "Castillo", "Yilmaz", "Dubois", "Nakamura", "Petrov", "Larsen", "Mendez", "Ibrahim", "Rahman", "Cohen",
    "Oliveira", "Schneider", "Volkov", "Adeyemi", "Chen", "Gupta", "Hoang", "Jensen", "Khalil", "Lopez", "Mori",
    "Nasser", "Ortiz", "Park", "Quispe", "Romero", "Sato", "Tran", "Usman", "Vargas", "Weiss", "Xu", "Yamada",
    "Zhang", "Abbott", "Barros", "Cruz", "Dalton", "Eriksen", "Fontaine", "Grimaldi", "Holm", "Ivanov", "Jovanovic",
    "Keller", "Laurent", "Moretti", "Navarro", "Olsen", "Pereira", "Quintero", "Ramos", "Stein", "Torres", "Underwood",
    "Vasquez", "Whitfield", "Yoon", "Zielinski", "Alvarez", "Becker", "Costa", "Dimitrov", "Eze", "Farouk", "Garcia",
]


@dataclass
class GeneticsProfile:
    """Inheritable tendencies used to make citizens feel varied."""

    vitality: float
    cognition: float
    stature: float
    metabolism: float
    stress_sensitivity: float
    addiction_risk: float
    longevity: float

    @classmethod
    def random(cls, rng: random.Random) -> "GeneticsProfile":
        return cls(
            vitality=rng.uniform(0.2, 1.0),
            cognition=rng.uniform(0.2, 1.0),
            stature=rng.uniform(0.2, 1.0),
            metabolism=rng.uniform(0.2, 1.0),
            stress_sensitivity=rng.uniform(0.2, 1.0),
            addiction_risk=rng.uniform(0.0, 1.0),
            longevity=rng.uniform(0.3, 1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PersonalityProfile:
    """Stable social/behavioral tendencies."""

    discipline: float
    sociability: float
    risk_tolerance: float
    empathy: float
    ambition: float
    impulsiveness: float
    resilience: float

    @classmethod
    def random(cls, rng: random.Random) -> "PersonalityProfile":
        return cls(
            discipline=rng.uniform(0.0, 1.0),
            sociability=rng.uniform(0.0, 1.0),
            risk_tolerance=rng.uniform(0.0, 1.0),
            empathy=rng.uniform(0.0, 1.0),
            ambition=rng.uniform(0.0, 1.0),
            impulsiveness=rng.uniform(0.0, 1.0),
            resilience=rng.uniform(0.0, 1.0),
        )

    def axis_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


@dataclass(slots=True)
class MemoryRecord:
    timestamp: str
    category: str
    headline: str
    impact: float
    location: Optional[Tuple[int, int]] = None
    related_people: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Relationship:
    person_id: int
    affinity: float = 0.0
    trust: float = 0.0
    romance: float = 0.0
    respect: float = 0.0
    familiarity: float = 0.0
    rivalry: float = 0.0
    gossip_knowledge: float = 0.0
    last_interaction_day: int = 0
    interaction_count: int = 0

    def closeness(self) -> float:
        return max(0.0, self.affinity * 0.35 + self.trust * 0.3 + self.familiarity * 0.2 + self.respect * 0.15)


@dataclass(slots=True)
class ScheduleBlock:
    start_hour: int
    end_hour: int
    activity: str
    target: Optional[Tuple[int, int]] = None
    building_id: Optional[int] = None
    intensity: float = 0.5


@dataclass
class Household:
    household_id: int
    member_ids: List[int]
    home_building_id: int
    listing_id: int
    savings_pool: float = 0.0
    monthly_costs: float = 0.0
    tenure: str = "rent"
    arrears_months: int = 0


class Person:
    """
    Full citizen simulation entity.

    The code is deliberately explicit instead of compact because many
    systems need hooks into a person's state: finances, education,
    crime, law, jobs, relationships, stress, and schedules.
    """

    def __init__(
        self,
        person_id: int,
        rng: random.Random,
        world: WorldMap,
        age: Optional[int] = None,
        *,
        initialize: bool = True,
    ) -> None:
        self.person_id = person_id
        self.rng = rng
        self.age = age if age is not None else rng.randint(0, 72)
        self.alive = True
        self.incarcerated = False
        self.care_tile: Optional[Tuple[int, int]] = None
        self._schedule_day_index = 0
        self.tile_scale = float(getattr(world, "tile_scale", 1.0))
        self._route = None
        self._route_i = 0
        self._failed_target = None
        self._home_cache = (None, None)
        self._work_cache = (None, None)
        self._wts = None
        self.infection_state = 0   # 0 susceptible, 1 exposed, 2 infectious, 3 recovered
        self.infection_day = 0
        self.vaccinated = False
        self.commute_minutes = 0.0
        self.pension = 0.0
        self.work_floor = 0
        self.home_floor = 0
        self.home_unit = 1
        self.floor = -1            # -1 outdoors, 0 ground floor, 1.. upper floors
        self.owned_buildings: List[int] = []
        self.mortgage = 0.0
        self.mode = MODE_WALK        # current/last travel mode
        self._acc = 0.0              # fractional tiles carried between sub-steps
        self._delay = 0              # sub-steps to wait before leaving
        if not initialize:
            return
        self.first_name = rng.choice(FIRST_NAMES)
        self.last_name = rng.choice(LAST_NAMES)
        self.full_name = f"{self.first_name} {self.last_name}"
        self.gender = rng.choice(["woman", "man"])
        self.genetics = GeneticsProfile.random(rng)
        self.personality = PersonalityProfile.random(rng)
        self.health = clamp(0.5 + self.genetics.vitality * 0.45 + rng.uniform(-0.1, 0.1), 0.0, 1.0)
        self.energy = clamp(0.6 + rng.uniform(-0.15, 0.15), 0.0, 1.0)
        self.stress = clamp(0.25 + (1.0 - self.personality.resilience) * 0.2 + rng.uniform(-0.1, 0.1), 0.0, 1.0)
        self.happiness = clamp(0.45 + rng.uniform(-0.15, 0.2), 0.0, 1.0)
        self.motivation = clamp(0.45 + self.personality.ambition * 0.3 + rng.uniform(-0.1, 0.15), 0.0, 1.0)
        self.reputation_public = clamp(0.5 + rng.uniform(-0.15, 0.15), 0.0, 1.0)
        self.hidden_karma = clamp(0.5 + rng.uniform(-0.2, 0.2), 0.0, 1.0)
        self.crime_tendency = clamp(
            self.personality.risk_tolerance * 0.35 + self.personality.impulsiveness * 0.35 + (1.0 - self.personality.empathy) * 0.3,
            0.0,
            1.0,
        )
        self.skills: Dict[str, float] = {key: clamp(rng.uniform(0.0, 0.45), 0.0, 1.0) for key in SKILL_KEYS}
        self.skill_practice: Dict[str, float] = {key: 0.0 for key in SKILL_KEYS}
        self.learning_focus_skill: Optional[str] = None
        self.education_level = rng.choice(["none", "school", "college", "certificate"])
        self.certifications: List[str] = []
        self.memories: List[MemoryRecord] = []
        self.relationships: Dict[int, Relationship] = {}
        self.family_ids: List[int] = []
        self.parent_ids: List[int] = []
        self.children_ids: List[int] = []
        self.spouse_id: Optional[int] = None
        self.relation_roles: Dict[int, str] = {}
        self.household_id: Optional[int] = None
        self.home_building_id: Optional[int] = None
        self.job_company_id: Optional[int] = None
        self.job_title: Optional[str] = None
        self.job_skill: Optional[str] = None
        self.salary: float = 0.0
        self.job_shift: Tuple[int, int] = (9, 17)
        self.student_status: Optional[str] = None
        self.current_enrollment_building: Optional[int] = None
        self.schedule: Dict[int, ScheduleBlock] = {}
        self.social_circle: List[int] = []
        self.work_traits: Dict[str, float] = {key: clamp(rng.uniform(0.2, 1.0), 0.0, 1.0) for key in WORK_TRAIT_KEYS}
        self.job_performance = clamp(0.45 + rng.uniform(-0.12, 0.12), 0.0, 1.0)
        self.job_satisfaction = clamp(0.5 + rng.uniform(-0.18, 0.18), 0.0, 1.0)
        self.days_unproductive = 0
        self.vehicle = rng.choice(["none", "bike", "car", "bus_pass"]) if self.age >= 18 else (
            rng.choice(["none", "bike"]) if self.age >= 10 else "none")
        if self.age < 18:
            self.cash = round(rng.uniform(10.0, 80.0), 2)
            self.bank_balance = 0.0
            self.debt = 0.0
            self.assets = 0.0
        else:
            # Life-cycle endowment: the young start with little, wealth builds with age.
            # Property is NOT endowed here; ownership emerges through the housing market.
            stage = clamp((self.age - 17) / 38.0, 0.08, 1.35)
            self.cash = round(rng.uniform(150.0, 4500.0) * (0.5 + stage / 2), 2)
            self.bank_balance = round(rng.uniform(0.0, 14_000.0) * stage, 2)
            self.debt = round(rng.uniform(0.0, 6_000.0) * (1.1 - stage / 2), 2)
            self.assets = round(rng.uniform(0.0, 9_000.0) * stage, 2)
        self.investment_accounts: Dict[str, float] = {}
        self.arrest_record: List[str] = []
        self.location = self._spawn_location(world)
        self.destination = self.location
        self._route = None
        self._route_i = 0
        self._failed_target = None
        self.hours_since_rest = 0
        self.last_paid_day = 0
        self.last_major_event_day = 0
        self.gossip_log: List[str] = []
        self.life_stage = self._derive_life_stage()
        self._daily_targets: Dict[str, Tuple[Optional[Tuple[int, int]], Optional[int]]] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _spawn_location(self, world: WorldMap) -> Tuple[int, int]:
        tile = world.random_walkable_tile()
        return (tile.x, tile.y)

    def _derive_life_stage(self) -> str:
        if self.age < 13:
            return "child"
        if self.age < 18:
            return "teen"
        if self.age < 22:
            return "young_adult"
        if self.age < 40:
            return "adult"
        if self.age < 60:
            return "midlife"
        return "senior"

    # ------------------------------------------------------------------
    # Narrative helpers
    # ------------------------------------------------------------------
    def add_memory(
        self,
        category: str,
        headline: str,
        impact: float,
        timestamp: str = "",
        location: Optional[Tuple[int, int]] = None,
        related_people: Optional[List[int]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.memories.append(
            MemoryRecord(
                timestamp=timestamp,
                category=category,
                headline=headline,
                impact=impact,
                location=location,
                related_people=related_people or [],
                tags=tags or [],
            )
        )
        if len(self.memories) > 60:
            self.memories = self.memories[-60:]
        self.happiness = clamp(self.happiness + impact * 0.05, 0.0, 1.0)
        self.stress = clamp(self.stress - impact * 0.03, 0.0, 1.0)

    def relationship_with(self, other_id: int) -> Relationship:
        if other_id not in self.relationships:
            self.relationships[other_id] = Relationship(person_id=other_id)
        return self.relationships[other_id]

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def _plan_hour(self, world: WorldMap, day_index: int, hour: int) -> ScheduleBlock:
        if self.incarcerated:
            return ScheduleBlock(hour, hour + 1, "rest", target=self.location, intensity=0.1)
        if self.care_tile is not None:
            return ScheduleBlock(hour, hour + 1, "rest", target=self.care_tile, intensity=0.1)
        bid = self.home_building_id
        cache = self._home_cache
        home_tile = cache[1] if (cache[0] == bid and bid is not None) else self.home_tile(world)
        # == self._should_sleep(hour), inlined ==
        on_shift = False
        if self.job_company_id is not None:
            start, end = self.job_shift
            if start != end:
                on_shift = (hour >= start or hour < end) if end < start else (start <= hour < end)
        if not on_shift:
            energy = self.energy
            if (hour >= 23 or hour <= 5 or energy < 0.14
                    or (self.hours_since_rest >= 20 and energy < 0.35)
                    or (self.stress > 0.88 and energy < 0.45)):
                return ScheduleBlock(hour, hour + 1, "sleep", target=home_tile, building_id=bid, intensity=0.2)
        if self.infection_state == 2 and self.person_id % 3 != 0:
            # Symptomatic: stay home, or seek hospital care when seriously ill.
            if self.health < 0.4 and 7 <= hour <= 21:
                cached = self._daily_targets.get("clinic")
                if cached is None:
                    cached = (self._choose_service_target(world, "hospital"), None)
                    self._daily_targets["clinic"] = cached
                return ScheduleBlock(hour, hour + 1, "clinic", target=cached[0], intensity=0.3)
            return ScheduleBlock(hour, hour + 1, "rest", target=home_tile, building_id=self.home_building_id, intensity=0.2)
        weekday = 6 if getattr(world, "holiday", False) else (day_index + 3) % 7
        work_tile = self.work_tile(world)
        school_tile = self.school_tile(world)

        obligation = self._mandatory_activity_block(world, hour, home_tile, work_tile, school_tile, weekday)
        if obligation is not None:
            return obligation

        activity = self._choose_self_directed_activity(hour, weekday >= 5)
        target = home_tile
        building_id = self.home_building_id
        if activity == "socialize":
            cached = self._daily_targets.get("socialize")
            if cached is not None:
                target = cached[0]
            else:
                target = self._choose_social_target(world)
                self._daily_targets["socialize"] = (target, None)
        elif activity == "exercise":
            cached = self._daily_targets.get("exercise")
            if cached is not None:
                target = cached[0]
            else:
                target = self._choose_park_target(world)
                self._daily_targets["exercise"] = (target, None)
        elif activity == "errand":
            cached = self._daily_targets.get("errand")
            if cached is not None:
                target = cached[0]
            else:
                target = self._choose_commercial_target(world)
                self._daily_targets["errand"] = (target, None)
        elif activity == "clinic":
            cached = self._daily_targets.get("clinic")
            if cached is not None:
                target = cached[0]
            else:
                target = self._choose_service_target(world, "hospital")
                self._daily_targets["clinic"] = (target, None)
        elif activity == "study_self" and school_tile and self.rng.random() < 0.35:
            cached = self._daily_targets.get("study_self")
            if cached is not None:
                target = cached[0]
                building_id = cached[1]
            else:
                target = school_tile
                building_id = self.current_enrollment_building
                self._daily_targets["study_self"] = (target, building_id)
        return ScheduleBlock(hour, hour + 1, activity, target=target, building_id=building_id, intensity=0.45)

    def _should_sleep(self, hour: int) -> bool:
        if self.job_company_id is not None and self._hour_in_range(hour, *self.job_shift):
            return False
        if hour >= 23 or hour <= 5:
            return True
        if self.energy < 0.14:
            return True
        if self.hours_since_rest >= 20 and self.energy < 0.35:
            return True
        if self.stress > 0.88 and self.energy < 0.45:
            return True
        return False

    def _mandatory_activity_block(
        self,
        world: WorldMap,
        hour: int,
        home_tile: Tuple[int, int],
        work_tile: Optional[Tuple[int, int]],
        school_tile: Optional[Tuple[int, int]],
        weekday: int = 0,
    ) -> Optional[ScheduleBlock]:
        # _commute_lead_hours is pure (no RNG, no state change), so it is only
        # evaluated where its value is actually used.
        if self.student_status and school_tile and self.energy > 0.08 and weekday < 5:
            if 8 <= hour < 14:
                activity = "commute_school" if self.location != school_tile else "study"
                return ScheduleBlock(hour, hour + 1, activity, target=school_tile, building_id=self.current_enrollment_building, intensity=0.8)
            if self.location != school_tile and self._is_commute_time(hour, 8, self._commute_lead_hours(school_tile)):
                return ScheduleBlock(hour, hour + 1, "commute_school", target=school_tile, building_id=self.current_enrollment_building, intensity=0.7)

        if self.job_company_id and work_tile and (weekday < 5 or self._works_weekend(world, weekday)):
            shift_start, shift_end = self.job_shift
            # == self._hour_in_range(hour, shift_start, shift_end), inlined ==
            if shift_start == shift_end:
                in_shift = False
            elif shift_end < shift_start:
                in_shift = hour >= shift_start or hour < shift_end
            else:
                in_shift = shift_start <= hour < shift_end
            if in_shift:
                if self.health < 0.16 or (self.energy < 0.08 and self.stress > 0.92):
                    return ScheduleBlock(hour, hour + 1, "rest", target=home_tile, building_id=self.home_building_id, intensity=0.5)
                if (
                    self.location == work_tile
                    and (hour - shift_start) % 24 == 4
                    and (shift_end - shift_start) % 24 >= 6
                ):
                    return ScheduleBlock(hour, hour + 1, "break", target=work_tile, intensity=0.3)
                activity = "commute_work" if self.location != work_tile else "work"
                return ScheduleBlock(hour, hour + 1, activity, target=work_tile, intensity=0.85)
            if self.location != work_tile and self.energy > 0.22 and self._is_commute_time(hour, shift_start, self._commute_lead_hours(work_tile)):
                return ScheduleBlock(hour, hour + 1, "commute_work", target=work_tile, intensity=0.6)
            if self._is_after_shift(hour, shift_start, shift_end) and self.location != home_tile:
                return ScheduleBlock(hour, hour + 1, "commute_home", target=home_tile, building_id=self.home_building_id, intensity=0.45)

        if self.job_company_id is None and self.age >= 18 and 9 <= hour < 12 and self.motivation > 0.2 and weekday < 5 and self.age < 67:
            return ScheduleBlock(hour, hour + 1, "job_search", target=home_tile, building_id=self.home_building_id, intensity=0.55)

        return None

    def _works_weekend(self, world: WorldMap, weekday: int) -> bool:
        """Weekend work only exists in 24/7 sectors, and staff rotate days off."""
        company = world.companies.get(self.job_company_id)
        if company is None or company.sector not in WEEKEND_SECTORS:
            return False
        return (self.person_id + weekday) % 2 == 0

    def _choose_self_directed_activity(self, hour: int, weekend: bool = False) -> str:
        rng = self.rng
        if self.health < 0.3 and self.cash >= 40.0 and 8 <= hour <= 20 and rng.random() < 0.6:
            return "clinic"
        if self.stress > 0.78:
            return "rest" if self.personality.sociability < 0.55 else rng.choice(["rest", "socialize"])
        if self.energy < 0.28:
            return "rest"
        if weekend:
            if self.personality.sociability > 0.5 and 11 <= hour <= 22 and rng.random() < 0.5:
                return "socialize"
            if 10 <= hour <= 19 and self.cash > 80.0 and rng.random() < 0.3:
                return "errand"
            if 8 <= hour <= 18 and rng.random() < 0.22:
                return "exercise"
        if self.personality.sociability > 0.62 and 17 <= hour <= 21 and rng.random() < 0.45:
            return "socialize"
        if self.motivation > 0.68 and rng.random() < (0.12 if weekend else 0.28):
            return "study_self"
        if self.health < 0.55 and rng.random() < 0.2:
            return "exercise"
        if self.cash > 120.0 and 10 <= hour <= 19 and rng.random() < 0.18:
            return "errand"
        if self.personality.sociability > 0.5 and rng.random() < 0.15:
            return "socialize"
        if rng.random() < 0.12:
            return "exercise"
        return "rest"

    def _hour_in_range(self, hour: int, start: int, end: int) -> bool:
        if start == end:
            return False
        if end < start:
            return hour >= start or hour < end
        return start <= hour < end

    def _is_commute_time(self, hour: int, start: int, lead_hours: int) -> bool:
        if lead_hours <= 0:
            return False
        delta = (start - hour) % 24
        return 0 < delta <= lead_hours

    def _is_after_shift(self, hour: int, start: int, end: int) -> bool:
        if start == end:
            return False
        if start < end:
            return hour >= end
        return end <= hour < start

    def _commute_lead_hours(self, target: Tuple[int, int]) -> int:
        distance = abs(self.location[0] - target[0]) + abs(self.location[1] - target[1])
        if distance <= 1:
            return 1
        return max(1, min(6, math.ceil(distance / self.tiles_per_hour())))

    def _choose_social_target(self, world: WorldMap) -> Optional[Tuple[int, int]]:
        parks = world.building_by_type("park_facility")
        stations = world.building_by_type("station")
        shops = world.building_by_type("shop")
        options = parks + stations + shops
        if not options:
            return self.home_tile(world)
        building = self.rng.choice(options)
        return building.tile

    def _choose_park_target(self, world: WorldMap) -> Optional[Tuple[int, int]]:
        if world._park_tiles:
            return self.rng.choice(world._park_tiles)
        return self.home_tile(world)

    def _choose_commercial_target(self, world: WorldMap) -> Optional[Tuple[int, int]]:
        shops = world.building_by_type("shop") + world.building_by_type("office")
        if not shops:
            return self.home_tile(world)
        return self.rng.choice(shops).tile

    def _choose_service_target(self, world: WorldMap, building_type: str) -> Optional[Tuple[int, int]]:
        buildings = world.building_by_type(building_type)
        if not buildings:
            return self.home_tile(world)
        return self.rng.choice(buildings).tile

    def current_activity(self, hour: int) -> str:
        if self.incarcerated:
            return "rest"
        block = self.current_block(hour)
        if block is not None:
            return block.activity
        if self._should_sleep(hour):
            return "sleep"
        if self.job_company_id is not None and self._hour_in_range(hour, *self.job_shift):
            return "work"
        if self.student_status and 8 <= hour < 14:
            return "study"
        if hour < 8 or hour >= 18:
            return "rest"
        return "idle"

    def current_block(self, hour: int) -> Optional[ScheduleBlock]:
        return self.schedule.get(hour)

    # ------------------------------------------------------------------
    # Map movement
    # ------------------------------------------------------------------
    def vehicle_speed_bonus(self) -> float:
        return VEHICLE_SPEED_BONUS.get(self.vehicle, 1.0)

    def acc_for_save(self) -> float:
        return round(self._acc, 4)

    def tiles_per_hour(self) -> int:
        mode = VEHICLE_DEFAULT_MODE.get(self.vehicle, MODE_WALK)
        return max(1, int(MODE_KMH[mode] * 1000.0 / TILE_METERS))

    def _choose_mode(self, world: WorldMap, dist: int, hour: int) -> int:
        """Pick how to make this trip from distance, vehicle ownership, weather, time and money."""
        if dist <= 14:
            return MODE_WALK
        vehicle = self.vehicle
        if self.age < 16:
            return MODE_WALK if dist <= 220 else MODE_BUS
        if vehicle == "car":
            return MODE_CAR if dist >= 18 else MODE_WALK
        if vehicle == "bike":
            if dist <= 420 and not world.bad_weather:
                return MODE_BIKE
            return MODE_BUS if dist >= 45 else MODE_WALK
        if vehicle == "bus_pass":
            if dist >= 45:
                return MODE_BUS
            return MODE_WALK
        # no vehicle
        if dist <= 150:
            if dist >= 30 and self.cash > 60 and (world.bad_weather or hour >= 22 or hour <= 5) and self.rng.random() < 0.35:
                return self._try_taxi(world, dist)
            return MODE_WALK
        if self.cash > 80 and self.rng.random() < 0.25:
            return self._try_taxi(world, dist)
        return MODE_BUS

    def _try_taxi(self, world: WorldMap, dist: int) -> int:
        fare = 3.0 + 0.05 * dist
        if self.cash >= fare:
            self.cash -= fare
            world.taxi_revenue_today += fare
            return MODE_TAXI
        return MODE_BUS if dist >= 45 else MODE_WALK

    # -- route storage ---------------------------------------------------
    @property
    def path(self) -> List[Tuple[int, int]]:
        route = self._route
        if route is None or self._route_i >= route.shape[0]:
            return []
        return [(int(x), int(y)) for x, y in route[self._route_i:].tolist()]

    @path.setter
    def path(self, value: Any) -> None:
        if value is None or len(value) == 0:
            self._route = None
            self._route_i = 0
            return
        if isinstance(value, np.ndarray):
            self._route = value.astype(np.int32, copy=False)
        else:
            self._route = np.asarray(value, dtype=np.int32).reshape(-1, 2)
        self._route_i = 0

    def has_path(self) -> bool:
        route = self._route
        return route is not None and self._route_i < route.shape[0]

    def set_route(self, route: Optional[np.ndarray]) -> None:
        if route is None or route.shape[0] == 0:
            self._route = None
            self._route_i = 0
        else:
            self._route = route
            self._route_i = 0

    def set_destination(self, target: Optional[Tuple[int, int]], world: WorldMap, hour: int = 12) -> None:
        if target is None:
            return
        self.destination = target
        if self.location == target:
            self._route = None
            self._route_i = 0
            return
        dist = abs(self.location[0] - target[0]) + abs(self.location[1] - target[1])
        self.mode = self._choose_mode(world, dist, hour)
        kind = 1 if self.mode in (MODE_CAR, MODE_BUS, MODE_TAXI) else 0
        self.set_route(world.find_route_array(self.location, target, max_steps=2000, kind=kind))
        self._acc = 0.0
        self._delay = (self.person_id * 7) % 6
        self._failed_target = None if self.has_path() else target

    def update_position_for_hour(self, world: WorldMap, hour: int) -> None:
        """Hour start: make sure a route exists and enrol the citizen as a mover.

        Actual walking/driving happens in Population.move_substep, 2 simulated
        minutes at a time, so citizens travel continuously along their route."""
        block = self.schedule.get(hour)
        if block and block.target:
            target = block.target
            if self.location != target and self._route is None and self._failed_target != target:
                self.set_destination(target, world, hour)
            elif self.destination != target:
                self.set_destination(target, world, hour)
            if self._route is not None and self.location != self.destination:
                world.movers.add(self.person_id)
        else:
            self.destination = self.location
            self._route = None
            self._route_i = 0

    # ------------------------------------------------------------------
    # Life simulation
    # ------------------------------------------------------------------
    def daily_update(self, world: WorldMap, day_index: int) -> None:
        self.age_if_needed(day_index)
        if world.aqi_health_drain:
            h = self.health - world.aqi_health_drain * (1.5 if self.age >= 65 or self.health < 0.45 else 1.0)
            self.health = 0.0 if h < 0.0 else h
        elif self.health < 0.995 and self.infection_state != 2:
            # Natural recovery: healthy living slowly restores health.
            self.health = self.health + 0.0006 * (0.5 + self.genetics.vitality)
        self.decay_relationships(day_index)
        self.skill_decay()
        self.resolve_basic_needs()
        self.happiness = clamp(self.happiness + (self._happiness_baseline() - self.happiness) * 0.2, 0.0, 1.0)
        self.stress = clamp(self.stress + (self._stress_baseline() - self.stress) * 0.15, 0.0, 1.0)
        self.schedule = {}
        self._daily_targets = {}
        self._schedule_day_index = day_index
        self._failed_target = None
        self._wts = None
        self.last_major_event_day = day_index if self.stress > 0.82 or self.happiness < 0.2 else self.last_major_event_day

    def hourly_position(self, world: WorldMap, day_index: int, hour: int) -> None:
        if self._schedule_day_index != day_index:
            self.schedule = {}
            self._daily_targets = {}
            self._schedule_day_index = day_index
        block = self.schedule.get(hour)
        if block is None:
            block = self._plan_hour(world, day_index, hour)
            self.schedule[hour] = block
        self.update_position_for_hour(world, hour)

    def hourly_update(self, world: WorldMap, day_index: int, hour: int) -> None:
        if self._schedule_day_index != day_index:
            self.schedule = {}
            self._daily_targets = {}
            self._schedule_day_index = day_index
        block = self.schedule.get(hour)
        if block is None:
            block = self._plan_hour(world, day_index, hour)
            self.schedule[hour] = block
        self.update_position_for_hour(world, hour)
        self.hourly_effects(world, day_index, hour)

    def hourly_effects(self, world: WorldMap, day_index: int, hour: int) -> None:
        block = self.schedule.get(hour)
        if block is None:
            block = self._plan_hour(world, day_index, hour)
            self.schedule[hour] = block
        activity = "rest" if self.incarcerated else block.activity
        self.refresh_floor(block)
        self._apply_activity_effects(activity)
        self._ambient_effects_from_location(world)
        self._maybe_generate_spontaneous_social_contact(world, day_index)

    def _happiness_baseline(self) -> float:
        baseline = 0.44 + (self.health - 0.5) * 0.18
        if self.job_company_id is not None:
            baseline += (self.job_satisfaction - 0.5) * 0.14
        elif self.age >= 18:
            baseline -= 0.07
        baseline += min(0.05, len(self.social_circle) * 0.003)
        baseline -= min(0.18, self.debt / 80_000.0)
        baseline -= max(0.0, self.stress - 0.55) * 0.18
        return clamp(baseline, 0.15, 0.78)

    def _stress_baseline(self) -> float:
        baseline = 0.2 + (1.0 - self.health) * 0.2 + (1.0 - self.energy) * 0.1
        baseline += min(0.2, self.debt / 60_000.0)
        if self.age >= 18 and self.job_company_id is None:
            baseline += 0.08
        baseline -= min(0.04, len(self.social_circle) * 0.002)
        return clamp(baseline, 0.08, 0.8)

    def refresh_floor(self, block: Optional[ScheduleBlock]) -> None:
        """Where in (or outside) a building the citizen is, from their current block."""
        if block is None:
            self.floor = -1
            return
        target = block.target
        activity = block.activity
        if target is None or self.location != target:
            self.floor = -1
        elif activity == "socialize" or activity == "exercise":
            self.floor = -1          # out in parks, plazas and streets
        elif activity == "work":
            self.floor = self.work_floor
        elif block.building_id is not None and block.building_id == self.home_building_id:
            self.floor = self.home_floor
        elif activity == "study":
            self.floor = 1 + self.person_id % 3
        else:
            self.floor = 0

    def _work_trait_score(self) -> float:
        wt = self.work_traits
        score = (
            wt["work_ethic"] * 0.28
            + wt["reliability"] * 0.24
            + wt["teamwork"] * 0.12
            + wt["adaptability"] * 0.12
        )
        self._wts = score
        return score

    def _apply_activity_effects(self, activity: str) -> None:
        energy = self.energy
        stress = self.stress
        if activity == "sleep":
            energy += 0.09
            if energy > 1.0:
                energy = 1.0
            stress -= 0.008
            self.hours_since_rest = 0
        else:
            self.hours_since_rest += 1
            energy -= 0.05
            if energy < 0.0:
                energy = 0.0

        if activity == "work":
            key = self._job_skill_key()
            skill_match = self.skills.get(key or "", 0.0)
            base = self._wts
            if base is None:
                base = self._work_trait_score()
            work_score = base + skill_match * 0.24
            perf = self.job_performance + (work_score - 0.55) * 0.025
            self.job_performance = 0.0 if perf < 0.0 else 1.0 if perf > 1.0 else perf
            sat = self.job_satisfaction + (skill_match - 0.45) * 0.015 - max(0.0, stress - 0.72) * 0.02
            self.job_satisfaction = 0.0 if sat < 0.0 else 1.0 if sat > 1.0 else sat
            if work_score < 0.42:
                self.days_unproductive += 1
            elif self.days_unproductive > 0:
                self.days_unproductive -= 1
            learning = self.work_traits["learning_drive"]
            mot = self.motivation + 0.01 + learning * 0.004
            self.motivation = 1.0 if mot > 1.0 else mot
            stress += 0.02 - self.personality.discipline * 0.008 + (0.55 - self.job_satisfaction) * 0.018
            if key:
                practice = self.skill_practice
                cur = practice.get(key, 0.0) + 0.45
                practice[key] = 3.0 if cur > 3.0 else cur
                sk = self.skills[key] + 0.00035 + learning * 0.0002
                self.skills[key] = 1.0 if sk > 1.0 else sk
        elif activity == "break":
            energy += 0.03
            if energy > 1.0:
                energy = 1.0
            stress -= 0.012
            h = self.happiness + 0.003
            self.happiness = 1.0 if h > 1.0 else h
        elif activity == "rest":
            energy += 0.006
            if energy > 1.0:
                energy = 1.0
            stress -= 0.003
            h = self.happiness + 0.002
            self.happiness = 1.0 if h > 1.0 else h
        elif activity == "study" or activity == "study_self":
            stress += 0.012
            key = self._priority_learning_skill()
            gain = (0.0008 + self.genetics.cognition * 0.0003 + self.work_traits["learning_drive"] * 0.00025) * EDU_MULT.get(self.education_level, 1.0)
            sk = self.skills[key] + gain
            self.skills[key] = 1.0 if sk > 1.0 else sk
            cur = self.skill_practice.get(key, 0.0) + 1.0
            self.skill_practice[key] = 4.0 if cur > 4.0 else cur
        elif activity == "socialize":
            h = self.happiness + 0.012 + self.personality.sociability * 0.008
            self.happiness = 1.0 if h > 1.0 else h
            stress -= 0.008
        elif activity == "exercise":
            hl = self.health + 0.004
            self.health = 1.0 if hl > 1.0 else hl
            h = self.happiness + 0.006
            self.happiness = 1.0 if h > 1.0 else h
        elif activity == "job_search":
            stress += 0.015
            m = self.motivation - 0.01
            self.motivation = 0.0 if m < 0.0 else m
        elif activity == "clinic":
            hl = self.health + 0.015
            self.health = 1.0 if hl > 1.0 else hl
            self.cash = max(0.0, self.cash - 40.0)

        self.energy = energy
        self.stress = 0.0 if stress < 0.0 else 1.0 if stress > 1.0 else stress

    def _ambient_effects_from_location(self, world: WorldMap) -> None:
        loc = self.location
        stress_delta, happy_delta = world.ambient_at(loc[0], loc[1])
        stress = self.stress + stress_delta
        self.stress = 0.0 if stress < 0.0 else 1.0 if stress > 1.0 else stress
        h = self.happiness + happy_delta
        self.happiness = 0.0 if h < 0.0 else 1.0 if h > 1.0 else h

    def _maybe_generate_spontaneous_social_contact(self, world: WorldMap, day_index: int) -> None:
        if self.age < 5:
            return
        loc = self.location
        occupants = world.tiles[loc[1] % world.height][loc[0] % world.width].occupants
        if len(occupants) < 2:
            return
        if self.rng.random() > 0.15:
            return
        self._spontaneous_contact(occupants, day_index)

    def _spontaneous_contact(self, occupants: List[int], day_index: int) -> None:
        """Second half of _maybe_generate_spontaneous_social_contact, after the 15% roll."""
        rng = self.rng
        ordered = sorted(occupants)
        other_id = ordered[int(rng.random() * len(ordered))]
        if other_id == self.person_id:
            return
        rel = self.relationship_with(other_id)
        rel.affinity = clamp(rel.affinity + self.personality.sociability * 0.04 - self.stress * 0.015, -1.0, 1.0)
        rel.familiarity = clamp(rel.familiarity + 0.05, 0.0, 1.0)
        rel.last_interaction_day = day_index
        rel.interaction_count += 1
        if other_id not in self.social_circle and rel.familiarity > 0.25 and len(self.social_circle) < 60:
            self.social_circle.append(other_id)
        if rng.random() < 0.05:
            self.add_memory("social", f"Had an unexpected conversation with citizen {other_id}.", 0.03)

    # ------------------------------------------------------------------
    # Finance and career
    # ------------------------------------------------------------------
    def assign_home(self, household_id: int, listing: HousingListing, building: Building) -> None:
        self.household_id = household_id
        self.home_building_id = building.building_id
        self.home_floor = listing.floor
        self.home_unit = listing.unit_no
        self.location = building.tile
        self.destination = self.location

    def assign_job(self, company_id: int, title: str, salary: float, shift: Tuple[int, int]) -> None:
        self.assign_job_profile(company_id, title, salary, shift, None)

    def assign_job_profile(
        self,
        company_id: int,
        title: str,
        salary: float,
        shift: Tuple[int, int],
        skill_key: Optional[str],
    ) -> None:
        self.job_company_id = company_id
        self.job_title = title
        self.job_skill = skill_key
        self.salary = salary
        self.job_shift = shift
        self.job_performance = clamp(self.job_performance + 0.06, 0.0, 1.0)
        self.job_satisfaction = clamp(self.job_satisfaction + 0.05, 0.0, 1.0)
        self.days_unproductive = 0
        self.add_memory("career", f"Took a new job as {title}.", 0.08)

    def remove_job(self) -> None:
        if self.job_title:
            self.add_memory("career", f"Lost job: {self.job_title}.", -0.12)
        self.job_company_id = None
        self.job_title = None
        self.job_skill = None
        self.salary = 0.0
        self.job_performance = clamp(self.job_performance - 0.08, 0.0, 1.0)
        self.job_satisfaction = clamp(self.job_satisfaction - 0.12, 0.0, 1.0)
        self.days_unproductive = 0

    def receive_pay(self) -> None:
        income = self.salary if self.salary > 0.0 else self.pension
        if income <= 0.0:
            return
        gross = income / 30.0
        self.cash += gross * 0.35
        self.bank_balance += gross * 0.65
        self.happiness = clamp(self.happiness + 0.015, 0.0, 1.0)

    def pay_living_costs(self, amount: float) -> float:
        """Pay a bill from cash then bank; any shortfall becomes debt. Returns the amount actually paid."""
        from_cash = min(self.cash, amount)
        self.cash -= from_cash
        remain = amount - from_cash
        from_bank = 0.0
        if remain > 0:
            from_bank = min(self.bank_balance, remain)
            self.bank_balance -= from_bank
            remain -= from_bank
        if remain > 0:
            self.debt += remain
            self.stress = clamp(self.stress + 0.08, 0.0, 1.0)
        return from_cash + from_bank

    def job_eligibility_score(self, posting: JobPosting, world: WorldMap, strongest_skill_key: Optional[str] = None) -> float:
        company = world.companies.get(posting.company_id)
        if company is None:
            return -1.0
        building = world.buildings.get(company.building_id)
        if building is None:
            return -1.0
        commute = world.estimate_commute_time(self.location, building.tile, self.vehicle_speed_bonus())
        skill = self.skills.get(posting.required_skill, 0.0)
        if strongest_skill_key is None:
            strongest_skill_key = max(self.skills, key=self.skills.get) if self.skills else posting.required_skill
        strongest_skill_value = self.skills.get(strongest_skill_key, 0.0)
        alignment_bonus = 0.12 if strongest_skill_key == posting.required_skill else max(0.0, skill - strongest_skill_value * 0.75) * 0.05
        return (
            skill * 0.52
            + self.personality.discipline * 0.1
            + self.personality.ambition * 0.08
            + self.work_traits["work_ethic"] * 0.08
            + self.work_traits["learning_drive"] * 0.05
            + self.education_bonus() * 0.08
            + alignment_bonus
            + clamp(1.0 - commute / 240.0, 0.0, 1.0) * 0.09
        )

    def education_bonus(self) -> float:
        return EDU_BONUS.get(self.education_level, 0.0)

    def seek_job(self, world: WorldMap, postings: Optional[List[JobPosting]] = None) -> Optional[int]:
        if self.job_company_id is not None:
            return None
        if postings is None:
            postings = world.open_job_postings()[:120]
        if not postings:
            return None
        strongest = max(self.skills, key=self.skills.get) if self.skills else None
        best = None
        best_score = -1e9
        for post in postings:
            score = self.job_eligibility_score(post, world, strongest)
            if score > best_score:
                best_score = score
                best = post
        if best is None or best_score < 0.0:
            return None
        if best_score < max(0.15, best.required_level * 0.55):
            return None
        return best.posting_id

    def choose_investment_symbol(self, symbols: List[str]) -> Optional[str]:
        if not symbols or self.bank_balance < 500.0:
            return None
        return self.rng.choice(symbols)

    # ------------------------------------------------------------------
    # Education
    # ------------------------------------------------------------------
    def enroll(self, status: str, building_id: int, tuition: float) -> None:
        self.student_status = status
        self.current_enrollment_building = building_id
        self.education_level = "school" if status == "school" else self.education_level
        self.pay_living_costs(tuition)
        self.add_memory("education", f"Enrolled in {status}.", 0.05)

    def complete_certification(self, name: str, skill_key: str) -> None:
        if name not in self.certifications:
            self.certifications.append(name)
        self.education_level = "certificate"
        self.skills[skill_key] = clamp(self.skills[skill_key] + 0.12, 0.0, 1.0)
        self.work_traits["learning_drive"] = clamp(self.work_traits["learning_drive"] + 0.05, 0.0, 1.0)
        self.add_memory("education", f"Completed certification: {name}.", 0.09)

    def graduate_college(self, major_skill: str) -> None:
        self.education_level = "college"
        self.skills[major_skill] = clamp(self.skills[major_skill] + 0.18, 0.0, 1.0)
        self.work_traits["adaptability"] = clamp(self.work_traits["adaptability"] + 0.05, 0.0, 1.0)
        self.work_traits["learning_drive"] = clamp(self.work_traits["learning_drive"] + 0.08, 0.0, 1.0)
        self.student_status = None
        self.current_enrollment_building = None
        self.add_memory("education", "Graduated from college.", 0.14)

    def _priority_learning_skill(self) -> str:
        job_skill = self._job_skill_key()
        if job_skill:
            return job_skill
        weakest_skill = min(self.skills.items(), key=lambda item: item[1])[0]
        if self.learning_focus_skill is None or self.learning_focus_skill not in self.skills:
            self.learning_focus_skill = weakest_skill
        focus_value = self.skills[self.learning_focus_skill]
        weakest_value = self.skills[weakest_skill]
        if focus_value > 0.82 and weakest_value < focus_value - 0.35:
            self.learning_focus_skill = weakest_skill
        return self.learning_focus_skill

    def _job_skill_key(self) -> Optional[str]:
        if self.job_skill:
            return self.job_skill
        if self.job_company_id and self.job_title:
            for key in ["software", "service", "logistics", "engineering", "finance", "law"]:
                if key in self.job_title.lower():
                    return key
        return None

    # ------------------------------------------------------------------
    # Social and family logic
    # ------------------------------------------------------------------
    def maybe_form_relationship(self, other_id: int) -> bool:
        if other_id in self.family_ids or other_id in self.parent_ids or other_id == self.spouse_id:
            return False
        rel = self.relationship_with(other_id)
        if rel.closeness() > 0.62 and self.personality.sociability > 0.35:
            rel.romance = clamp(rel.romance + 0.1, 0.0, 1.0)
            return rel.romance > 0.45
        return False

    def maybe_have_child(self, partner_id: int, next_person_id: int) -> Optional[Dict[str, Any]]:
        return None   # conception and birth are handled by LifeSystem (pregnancy, genetics, hospitals)
        if self.gender != "woman" or self.incarcerated:
            return None
        if self.age < 22 or self.age > 42:
            return None
        if self.spouse_id != partner_id:
            return None
        if self.relationship_with(partner_id).romance < 0.68:
            return None
        if self.health < 0.35 or self.cash + self.bank_balance < 500.0:
            return None
        if self.rng.random() > 0.004:
            return None
        return {
            "new_person_id": next_person_id,
            "parents": [self.person_id, partner_id],
        }

    DECAY_STRIDE = 4

    def decay_relationships(self, day_index: int) -> None:
        # Relationships fade slowly, so each person only processes them every
        # DECAY_STRIDE days (with proportionally larger steps).  Staggering by
        # person id spreads the work evenly across days.
        stride = self.DECAY_STRIDE
        if (day_index + self.person_id) % stride:
            return
        loneliness = 1.0 - self.personality.sociability
        for relationship in self.relationships.values():
            days = day_index - relationship.last_interaction_day
            if days <= 0:
                continue
            decay = min(0.03, days * 0.0008) * stride
            a = relationship.affinity - decay * loneliness
            relationship.affinity = -1.0 if a < -1.0 else 1.0 if a > 1.0 else a
            f = relationship.familiarity - decay * 0.5
            relationship.familiarity = 0.0 if f < 0.0 else f
            r = relationship.romance - decay * 0.4
            relationship.romance = 0.0 if r < 0.0 else r

    def spread_gossip(self) -> Optional[str]:
        if not self.gossip_log or self.personality.sociability < 0.2:
            return None
        return self.rng.choice(self.gossip_log)

    def set_relation_role(self, other_id: int, role: str) -> None:
        if other_id == self.person_id:
            return
        self.relation_roles[int(other_id)] = role
        if role in {"mother", "father", "parent", "child", "son", "daughter", "sibling", "wife", "husband", "spouse"}:
            if other_id not in self.family_ids:
                self.family_ids.append(other_id)

    def set_spouse(self, other: "Person") -> None:
        self.spouse_id = other.person_id
        other.spouse_id = self.person_id
        self_role = "wife" if other.gender == "woman" else "husband"
        other_role = "wife" if self.gender == "woman" else "husband"
        self.set_relation_role(other.person_id, self_role)
        other.set_relation_role(self.person_id, other_role)
        rel_self = self.relationship_with(other.person_id)
        rel_other = other.relationship_with(self.person_id)
        rel_self.romance = clamp(max(rel_self.romance, 0.72), 0.0, 1.0)
        rel_other.romance = clamp(max(rel_other.romance, 0.72), 0.0, 1.0)
        rel_self.trust = clamp(max(rel_self.trust, 0.65), 0.0, 1.0)
        rel_other.trust = clamp(max(rel_other.trust, 0.65), 0.0, 1.0)

    def add_parent(self, parent: "Person") -> None:
        if parent.person_id not in self.parent_ids:
            self.parent_ids.append(parent.person_id)
        parent_role = "mother" if parent.gender == "woman" else "father"
        child_role = "daughter" if self.gender == "woman" else "son"
        self.set_relation_role(parent.person_id, parent_role)
        parent.set_relation_role(self.person_id, child_role)
        if self.person_id not in parent.children_ids:
            parent.children_ids.append(self.person_id)
        rel_child = self.relationship_with(parent.person_id)
        rel_parent = parent.relationship_with(self.person_id)
        rel_child.trust = clamp(max(rel_child.trust, 0.6), 0.0, 1.0)
        rel_parent.trust = clamp(max(rel_parent.trust, 0.75), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal wellness model
    # ------------------------------------------------------------------
    def age_if_needed(self, day_index: int) -> None:
        if day_index > 0 and day_index % 365 == 0:
            self.age += 1
            self.life_stage = self._derive_life_stage()
            self.health = clamp(self.health - 0.01 * (1.0 - self.genetics.longevity), 0.0, 1.0)
            self.add_memory("life", f"Had a birthday and turned {self.age}.", 0.03)

    def resolve_basic_needs(self) -> None:
        if self.hours_since_rest > 18:
            self.stress = clamp(self.stress + 0.06, 0.0, 1.0)
            self.health = clamp(self.health - 0.01, 0.0, 1.0)
        if self.cash + self.bank_balance < 100.0:
            self.stress = clamp(self.stress + 0.04, 0.0, 1.0)
            self.happiness = clamp(self.happiness - 0.03, 0.0, 1.0)
        if self.debt > 20_000.0:
            self.motivation = clamp(self.motivation - 0.02, 0.0, 1.0)

    def skill_decay(self) -> None:
        skills = self.skills
        practice_map = self.skill_practice
        for skill in skills:
            value = skills[skill]
            practice = practice_map.get(skill, 0.0)
            if practice >= 0.75:
                decay = 0.00035
            elif value > 0.4:
                decay = 0.0025
            else:
                decay = 0.001
            value -= decay
            skills[skill] = 0.0 if value < 0.0 else value
            practice = practice * 0.55 - 0.05
            practice_map[skill] = 0.0 if practice < 0.0 else practice

    # ------------------------------------------------------------------
    # Location helpers
    # ------------------------------------------------------------------
    def home_tile(self, world: WorldMap) -> Optional[Tuple[int, int]]:
        bid = self.home_building_id
        cache = self._home_cache
        if cache[0] == bid and bid is not None:
            return cache[1]
        if bid is None or bid not in world.buildings:
            return self.location
        tile = world.buildings[bid].tile
        self._home_cache = (bid, tile)
        return tile

    def work_tile(self, world: WorldMap) -> Optional[Tuple[int, int]]:
        cid = self.job_company_id
        if cid is None:
            return None
        company = world.companies.get(cid)
        if company is None:
            return None
        cache = self._work_cache
        if cache[0] == company.building_id:
            return cache[1]
        tile = world.buildings[company.building_id].tile
        self._work_cache = (company.building_id, tile)
        return tile

    def school_tile(self, world: WorldMap) -> Optional[Tuple[int, int]]:
        if self.current_enrollment_building is None or self.current_enrollment_building not in world.buildings:
            return None
        return world.buildings[self.current_enrollment_building].tile

    def _serialize_memories(self) -> List[List[Any]]:
        serialized: List[List[Any]] = []
        for memory in self.memories:
            serialized.append(
                [
                    memory.timestamp,
                    memory.category,
                    memory.headline,
                    round_for_save(memory.impact, PERSON_SAVE_FLOAT_DIGITS),
                    memory.location,
                    list(memory.related_people),
                    list(memory.tags),
                ]
            )
        return serialized

    def _serialize_relationships(self) -> Dict[int, List[Any]]:
        serialized: Dict[int, List[Any]] = {}
        for pid, rel in self.relationships.items():
            serialized[int(pid)] = [
                round_for_save(rel.affinity, PERSON_SAVE_FLOAT_DIGITS),
                round_for_save(rel.trust, PERSON_SAVE_FLOAT_DIGITS),
                round_for_save(rel.romance, PERSON_SAVE_FLOAT_DIGITS),
                round_for_save(rel.respect, PERSON_SAVE_FLOAT_DIGITS),
                round_for_save(rel.familiarity, PERSON_SAVE_FLOAT_DIGITS),
                round_for_save(rel.rivalry, PERSON_SAVE_FLOAT_DIGITS),
                round_for_save(rel.gossip_knowledge, PERSON_SAVE_FLOAT_DIGITS),
                rel.last_interaction_day,
                rel.interaction_count,
            ]
        return serialized

    @staticmethod
    def _deserialize_memory(memory: Any) -> MemoryRecord:
        if isinstance(memory, dict):
            return MemoryRecord(**memory)
        timestamp, category, headline, impact, location, related_people, tags = memory
        return MemoryRecord(
            timestamp=timestamp,
            category=category,
            headline=headline,
            impact=impact,
            location=tuple(location) if location is not None else None,
            related_people=list(related_people),
            tags=list(tags),
        )

    @staticmethod
    def _deserialize_relationship(pid: int, relationship_data: Any) -> Relationship:
        if isinstance(relationship_data, dict):
            return Relationship(**relationship_data)
        return Relationship(
            person_id=pid,
            affinity=relationship_data[0],
            trust=relationship_data[1],
            romance=relationship_data[2],
            respect=relationship_data[3],
            familiarity=relationship_data[4],
            rivalry=relationship_data[5],
            gossip_knowledge=relationship_data[6],
            last_interaction_day=relationship_data[7],
            interaction_count=relationship_data[8],
        )

    # ------------------------------------------------------------------
    # Serialization support
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 3,
            "person_id": self.person_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "age": self.age,
            "gender": self.gender,
            "genetics": self.genetics.to_dict(),
            "personality": self.personality.axis_dict(),
            "health": round_for_save(self.health, PERSON_SAVE_FLOAT_DIGITS),
            "energy": round_for_save(self.energy, PERSON_SAVE_FLOAT_DIGITS),
            "stress": round_for_save(self.stress, PERSON_SAVE_FLOAT_DIGITS),
            "happiness": round_for_save(self.happiness, PERSON_SAVE_FLOAT_DIGITS),
            "motivation": round_for_save(self.motivation, PERSON_SAVE_FLOAT_DIGITS),
            "reputation_public": round_for_save(self.reputation_public, PERSON_SAVE_FLOAT_DIGITS),
            "hidden_karma": round_for_save(self.hidden_karma, PERSON_SAVE_FLOAT_DIGITS),
            "crime_tendency": round_for_save(self.crime_tendency, PERSON_SAVE_FLOAT_DIGITS),
            "skills": {key: round_for_save(value, PERSON_SAVE_FLOAT_DIGITS) for key, value in self.skills.items()},
            "skill_practice": {key: round_for_save(value, PERSON_SAVE_FLOAT_DIGITS) for key, value in self.skill_practice.items()},
            "learning_focus_skill": self.learning_focus_skill,
            "education_level": self.education_level,
            "certifications": self.certifications,
            "memories": self._serialize_memories(),
            "relationships": self._serialize_relationships(),
            "family_ids": self.family_ids,
            "parent_ids": self.parent_ids,
            "children_ids": self.children_ids,
            "spouse_id": self.spouse_id,
            "relation_roles": self.relation_roles,
            "household_id": self.household_id,
            "home_building_id": self.home_building_id,
            "job_company_id": self.job_company_id,
            "job_title": self.job_title,
            "job_skill": self.job_skill,
            "salary": round_for_save(self.salary, 2),
            "job_shift": self.job_shift,
            "student_status": self.student_status,
            "current_enrollment_building": self.current_enrollment_building,
            "social_circle": self.social_circle,
            "work_traits": {key: round_for_save(value, PERSON_SAVE_FLOAT_DIGITS) for key, value in self.work_traits.items()},
            "job_performance": round_for_save(self.job_performance, PERSON_SAVE_FLOAT_DIGITS),
            "job_satisfaction": round_for_save(self.job_satisfaction, PERSON_SAVE_FLOAT_DIGITS),
            "days_unproductive": self.days_unproductive,
            "vehicle": self.vehicle,
            "cash": round_for_save(self.cash, 2),
            "bank_balance": round_for_save(self.bank_balance, 2),
            "debt": round_for_save(self.debt, 2),
            "assets": round_for_save(self.assets, 2),
            "investment_accounts": {key: round_for_save(value, 2) for key, value in self.investment_accounts.items()},
            "arrest_record": self.arrest_record,
            "location": self.location,
            "destination": self.destination,
            "path": self.path,
            "schedule": {
                int(hour): {
                    "start_hour": block.start_hour,
                    "end_hour": block.end_hour,
                    "activity": block.activity,
                    "target": tuple(block.target) if block.target is not None else None,
                    "building_id": block.building_id,
                    "intensity": block.intensity,
                }
                for hour, block in self.schedule.items()
            },
            "schedule_day_index": self._schedule_day_index,
            "daily_targets": {
                key: [
                    tuple(target) if target is not None else None,
                    building_id,
                ]
                for key, (target, building_id) in self._daily_targets.items()
            },
            "hours_since_rest": self.hours_since_rest,
            "last_paid_day": self.last_paid_day,
            "last_major_event_day": self.last_major_event_day,
            "gossip_log": self.gossip_log,
            "life_stage": self.life_stage,
            "infection_state": self.infection_state,
            "infection_day": self.infection_day,
            "vaccinated": self.vaccinated,
            "pension": round_for_save(self.pension, 2),
            "work_floor": self.work_floor,
            "home_floor": self.home_floor,
            "home_unit": self.home_unit,
            "floor": self.floor,
            "owned_buildings": list(self.owned_buildings),
            "mode": self.mode,
            "mortgage": round_for_save(self.mortgage, 2),
            "acc": self.acc_for_save(),
            "delay": self._delay,
            "alive": self.alive,
            "incarcerated": self.incarcerated,
            "care_tile": list(self.care_tile) if self.care_tile else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], rng: random.Random, world: WorldMap, schedule_day_index: int = 0) -> "Person":
        person = cls(
            person_id=data["person_id"],
            rng=rng,
            world=world,
            age=data["age"],
            initialize=False,
        )
        person.first_name = data["first_name"]
        person.last_name = data["last_name"]
        person.full_name = data["full_name"]
        # Older saves may contain a third value; map it deterministically to man/woman.
        person.gender = data["gender"] if data["gender"] in ("man", "woman") else ("woman" if data["person_id"] % 2 == 0 else "man")
        person.genetics = GeneticsProfile(**data["genetics"])
        person.personality = PersonalityProfile(**data["personality"])
        person.health = data["health"]
        person.energy = data["energy"]
        person.stress = data["stress"]
        person.happiness = data["happiness"]
        person.motivation = data["motivation"]
        person.reputation_public = data["reputation_public"]
        person.hidden_karma = data["hidden_karma"]
        person.crime_tendency = data["crime_tendency"]
        person.skills = {key: float(value) for key, value in data["skills"].items()}
        person.skill_practice = {key: 0.0 for key in SKILL_KEYS}
        person.skill_practice.update(data.get("skill_practice", {}))
        person.learning_focus_skill = data.get("learning_focus_skill")
        person.education_level = data["education_level"]
        person.certifications = list(data["certifications"])
        person.memories = [cls._deserialize_memory(memory) for memory in data["memories"]]
        person.relationships = {
            int(pid): cls._deserialize_relationship(int(pid), rel)
            for pid, rel in data["relationships"].items()
        }
        person.family_ids = [int(pid) for pid in data["family_ids"]]
        person.parent_ids = [int(pid) for pid in data.get("parent_ids", [])]
        person.children_ids = [int(pid) for pid in data["children_ids"]]
        person.spouse_id = data.get("spouse_id")
        person.relation_roles = {int(pid): role for pid, role in data.get("relation_roles", {}).items()}
        person.household_id = data["household_id"]
        person.home_building_id = data["home_building_id"]
        person.job_company_id = data["job_company_id"]
        person.job_title = data["job_title"]
        person.job_skill = data.get("job_skill")
        person.salary = data["salary"]
        person.job_shift = tuple(data["job_shift"])
        person.student_status = data["student_status"]
        person.current_enrollment_building = data["current_enrollment_building"]
        person.social_circle = [int(pid) for pid in data["social_circle"]]
        person.work_traits = {key: 0.5 for key in WORK_TRAIT_KEYS}
        person.work_traits.update(data.get("work_traits", {}))
        person.job_performance = data.get("job_performance", 0.5)
        person.job_satisfaction = data.get("job_satisfaction", 0.5)
        person.days_unproductive = data.get("days_unproductive", 0)
        person.vehicle = data["vehicle"]
        person.cash = data["cash"]
        person.bank_balance = data["bank_balance"]
        person.debt = data["debt"]
        person.assets = data["assets"]
        person.investment_accounts = dict(data["investment_accounts"])
        person.arrest_record = list(data["arrest_record"])
        person.location = tuple(data["location"])
        person.destination = tuple(data["destination"])
        person.path = [tuple(step) for step in data.get("path", [])]
        person.infection_state = int(data.get("infection_state", 0))
        person.infection_day = int(data.get("infection_day", 0))
        person.vaccinated = bool(data.get("vaccinated", False))
        person.pension = float(data.get("pension", 0.0))
        person.work_floor = int(data.get("work_floor", 0))
        person.home_floor = int(data.get("home_floor", 0))
        person.home_unit = int(data.get("home_unit", 1))
        person.floor = int(data.get("floor", -1))
        person.owned_buildings = [int(b) for b in data.get("owned_buildings", [])]
        person.mode = int(data.get("mode", 0))
        person.mortgage = float(data.get("mortgage", 0.0))
        person._acc = float(data.get("acc", 0.0))
        person._delay = int(data.get("delay", 0))
        person.hours_since_rest = data["hours_since_rest"]
        person.last_paid_day = data["last_paid_day"]
        person.last_major_event_day = data["last_major_event_day"]
        person.gossip_log = list(data["gossip_log"])
        person.life_stage = data["life_stage"]
        person.alive = bool(data.get("alive", True))
        person.incarcerated = bool(data.get("incarcerated", False))
        ct = data.get("care_tile")
        person.care_tile = (int(ct[0]), int(ct[1])) if ct else None
        person._schedule_day_index = int(data.get("schedule_day_index", schedule_day_index))
        person.schedule = {
            int(hour): ScheduleBlock(
                start_hour=int(block["start_hour"]),
                end_hour=int(block["end_hour"]),
                activity=block["activity"],
                target=tuple(block["target"]) if block.get("target") is not None else None,
                building_id=block.get("building_id"),
                intensity=float(block.get("intensity", 0.5)),
            )
            for hour, block in data.get("schedule", {}).items()
        }
        person._daily_targets = {
            key: (
                tuple(target) if target is not None else None,
                building_id,
            )
            for key, (target, building_id) in data.get("daily_targets", {}).items()
        }
        return person


class Population:
    """Owns all people and household graphs."""

    def __init__(self, seed: int, world: WorldMap, size: int = 240) -> None:
        self.seed = seed
        self.rng = random.Random(seed + 101)
        self.world = world
        self.people: Dict[int, Person] = {}
        self.households: Dict[int, Household] = {}
        self._household_counter = 1
        self._person_counter = 1
        self.births_total = 0
        self.deaths_total = 0
        self.death_ages: List[int] = []
        self.last_commuters = 0
        self.death_hooks: List[Any] = []
        self._pop_gen = 0
        self._slice_cache: Dict[int, Any] = {}
        self.price_level = 1.0
        self.last_segments: Dict[int, np.ndarray] = {}
        self.last_times: Dict[int, np.ndarray] = {}
        self.property_tax_pending = 0.0
        self.consumer_spend_pending = 0.0
        self.consumer_spend_month = 0.0
        self.corporate_sale_proceeds = 0.0
        self.property_sales_month: Dict[str, int] = {"homes": 0, "investments": 0}
        self.landlord_income_month = 0.0
        self.corporate_rent_month = 0.0
        self.rent_collected_month = 0.0
        self.generate_population(size)

    # ------------------------------------------------------------------
    # Population generation
    # ------------------------------------------------------------------
    def generate_population(self, size: int) -> None:
        for _ in range(size):
            person = Person(self._person_counter, self.rng, self.world, age=self._sample_initial_age())
            self.people[person.person_id] = person
            self._pop_gen += 1
            self.world.register_population_position(person.person_id, *person.location)
            self._person_counter += 1
        self._assign_homes()
        self._assign_ownership()
        self._assign_default_education()
        self._assign_initial_social_graph()

    def _sample_initial_age(self) -> int:
        roll = self.rng.random()
        if roll < 0.08:
            return self.rng.randint(0, 12)
        if roll < 0.16:
            return self.rng.randint(13, 17)
        if roll < 0.42:
            return self.rng.randint(18, 29)
        if roll < 0.74:
            return self.rng.randint(30, 49)
        if roll < 0.9:
            return self.rng.randint(50, 64)
        return self.rng.randint(65, 84)

    def _assign_homes(self) -> None:
        listings = self.world.available_housing()
        self.rng.shuffle(listings)
        adults = [person for person in self.people.values() if person.age >= 18]
        minors = [person for person in self.people.values() if person.age < 18]
        self.rng.shuffle(adults)
        self.rng.shuffle(minors)
        assigned_ids: set[int] = set()

        while listings and (adults or minors):
            listing = listings.pop()
            listing.occupied = True
            building = self.world.buildings[listing.building_id]
            members = self._build_household_members(adults, minors, listing.bedrooms)
            if not members:
                continue
            household = Household(
                household_id=self._household_counter,
                member_ids=[member.person_id for member in members],
                home_building_id=building.building_id,
                listing_id=listing.listing_id,
                savings_pool=round(sum(member.bank_balance for member in members), 2),
                monthly_costs=round(listing.rent * 1.15, 2),
            )
            self.households[household.household_id] = household
            for member in members:
                member.assign_home(household.household_id, listing, building)
                assigned_ids.add(member.person_id)
                self.world.register_population_position(member.person_id, *member.location)
            listing.household_id = household.household_id
            self._link_household_family(members)
            self._household_counter += 1
        # Nobody should be left without a home: anyone who missed out on a unit joins a
        # household that still has room (shared housing), rather than wandering the streets.
        open_households = [h for h in self.households.values() if len(h.member_ids) < 6]
        self.rng.shuffle(open_households)
        cursor = 0
        for person in self.people.values():
            if person.person_id in assigned_ids:
                continue
            placed = False
            while cursor < len(open_households):
                household = open_households[cursor]
                listing = self.world.housing_market.get(household.listing_id)
                building = self.world.buildings.get(listing.building_id) if listing is not None else None
                if listing is None or building is None or len(household.member_ids) >= 6:
                    cursor += 1
                    continue
                household.member_ids.append(person.person_id)
                person.assign_home(household.household_id, listing, building)
                self.world.register_population_position(person.person_id, *person.location)
                if len(household.member_ids) >= 6:
                    cursor += 1
                placed = True
                break
            if not placed:
                fallback_tile = self.world.random_walkable_tile()
                person.location = (fallback_tile.x, fallback_tile.y)
                self.world.register_population_position(person.person_id, *person.location)

    def _build_household_members(self, adults: List[Person], minors: List[Person], bedrooms: int) -> List[Person]:
        household_size_cap = max(1, min(6, bedrooms + 2))
        members: List[Person] = []
        if adults and minors and self.rng.random() < 0.48:
            adult_count = 2 if len(adults) >= 2 and bedrooms >= 2 and self.rng.random() < 0.68 else 1
            for _ in range(min(adult_count, len(adults))):
                members.append(adults.pop())
            child_target = min(len(minors), household_size_cap - len(members), self.rng.randint(1, max(1, bedrooms + 1)))
            for _ in range(child_target):
                members.append(minors.pop())
        elif adults:
            adult_count = 2 if len(adults) >= 2 and self.rng.random() < 0.42 else 1
            adult_count = min(adult_count, len(adults), household_size_cap)
            for _ in range(adult_count):
                members.append(adults.pop())
            if minors and len(members) < household_size_cap and self.rng.random() < 0.25:
                members.append(minors.pop())
        elif minors:
            members.append(minors.pop())
        return members

    def _link_household_family(self, members: List[Person]) -> None:
        adults = [member for member in members if member.age >= 18]
        minors = [member for member in members if member.age < 18]
        if len(adults) >= 2 and not adults[0].spouse_id and not adults[1].spouse_id:
            if abs(adults[0].age - adults[1].age) <= 18:
                adults[0].set_spouse(adults[1])
        for minor in minors:
            able = [a for a in adults if 16 <= a.age - minor.age <= 50]
            if not able:
                continue                                  # no plausible parent in the house: a ward or relative
            first = able[0]
            parents = [first]
            for other in able[1:]:
                if other.gender != first.gender and first.spouse_id == other.person_id:
                    parents.append(other)                # two parents only if a married man and woman
                    break
            for parent in parents[:2]:
                minor.add_parent(parent)
        for member in members:
            for other in members:
                if member.person_id == other.person_id:
                    continue
                if other.person_id not in member.family_ids:
                    member.family_ids.append(other.person_id)
                if member.age < 18 and other.age < 18 and set(member.parent_ids) & set(other.parent_ids):
                    member.set_relation_role(other.person_id, "sibling")
                elif member.age >= 18 and other.age >= 18 and member.spouse_id != other.person_id and other.spouse_id != member.person_id:
                    if member.relation_roles.get(other.person_id) is None:
                        member.set_relation_role(other.person_id, "relative")

    # ------------------------------------------------------------------
    # Property ownership, landlords and rent
    # ------------------------------------------------------------------
    def _assign_ownership(self, owner_occupier_rate: float = 0.0) -> None:
        """Day-0 tenure.  All housing starts in corporate hands and everyone rents:
        no one is handed property, so wealth inequality from ownership has to be
        earned through the housing market (see run_property_market).  A non-zero
        `owner_occupier_rate` pre-seeds some mortgaged owner-occupiers by age."""
        world = self.world
        rng = self.rng
        people = self.people
        for bid, listings in world._building_listings.items():
            building = world.buildings.get(bid)
            if building is None or not listings:
                continue
            building.owner_id = None
            building.owner_kind = "developer"
            for listing in listings:
                listing.owner_id = None
                listing.tenure = "rent"
                household = self.households.get(listing.household_id) if listing.household_id else None
                if household is None:
                    continue
                household.tenure = "rent"
                if owner_occupier_rate <= 0.0:
                    continue
                head = max(
                    (people[m] for m in household.member_ids if m in people),
                    key=lambda p: (p.age >= 18, p.bank_balance + p.assets),
                    default=None,
                )
                if head is None or head.age < 30 or rng.random() > owner_occupier_rate:
                    continue
                equity = clamp((head.age - 28) / 45.0, 0.1, 0.65)
                self._transfer_to_buyer(listing, household, head, occupant=True, down=listing.purchase_price * equity, price=listing.purchase_price, charge_buyer=False)

    # ------------------------------------------------------------------
    # Housing market: ownership emerges from savings, income and credit
    # ------------------------------------------------------------------
    MORTGAGE_RATE_MONTHLY = 0.0061     # ~6.3% APR on a 30-year loan
    MORTGAGE_INTEREST_SHARE = 0.74

    def _refresh_building_owner(self, building: Building) -> None:
        listings = self.world._building_listings.get(building.building_id, [])
        counts: Dict[Optional[int], int] = {}
        for l in listings:
            counts[l.owner_id] = counts.get(l.owner_id, 0) + 1
        if not counts:
            return
        owner = max(counts, key=lambda k: counts[k])
        building.owner_id = owner
        building.owner_kind = "person" if owner is not None else "developer"

    def _transfer_to_buyer(self, listing: HousingListing, household: Optional[Household], buyer: Person,
                           occupant: bool, down: float, price: float, charge_buyer: bool = True) -> None:
        world = self.world
        seller = self.people.get(listing.owner_id) if listing.owner_id is not None else None
        if charge_buyer:
            buyer.pay_living_costs(down)
        buyer.mortgage += max(0.0, price - down)
        buyer.assets += price
        if seller is not None:
            proceeds = price * 0.97
            payoff = min(seller.mortgage, proceeds)
            seller.mortgage -= payoff
            seller.bank_balance += proceeds - payoff
            seller.assets = max(0.0, seller.assets - price)
            if not any(l.owner_id == seller.person_id for l in world._building_listings.get(listing.building_id, []) if l is not listing):
                if listing.building_id in seller.owned_buildings:
                    seller.owned_buildings.remove(listing.building_id)
        else:
            self.corporate_sale_proceeds += price
        listing.owner_id = buyer.person_id
        if occupant and household is not None:
            listing.tenure = "own"
            household.tenure = "own"
            household.monthly_costs = round(listing.rent * 0.35, 2)    # utilities, upkeep; mortgage is billed separately
        else:
            listing.tenure = "rent"
        if listing.building_id not in buyer.owned_buildings:
            buyer.owned_buildings.append(listing.building_id)
        building = world.buildings.get(listing.building_id)
        if building is not None:
            self._refresh_building_owner(building)

    def run_property_market(self, day_index: int) -> Dict[str, int]:
        """Monthly transactions: households buy the home they rent when they can
        carry the mortgage; savers with spare capital buy rental units and become
        landlords.  Buyers need a deposit from savings and an affordable payment."""
        world = self.world
        rng = self.rng
        people = self.people
        stats = {"homes": 0, "investments": 0}
        candidates = [
            p for p in people.values()
            if 25 <= p.age < 68 and not p.incarcerated and (p.salary > 0 or p.pension > 0)
        ]
        rng.shuffle(candidates)
        developer_units: Optional[List[HousingListing]] = None
        for p in candidates[:500]:
            if stats["homes"] + stats["investments"] >= 140:
                break
            liquid = p.cash + p.bank_balance
            income = (p.salary + p.pension) * self.price_level
            committed = p.mortgage * self.MORTGAGE_RATE_MONTHLY
            household = self.households.get(p.household_id) if p.household_id is not None else None
            listing = world.housing_market.get(household.listing_id) if household is not None else None
            # --- buy the home they rent
            if listing is not None and listing.tenure == "rent" and p.mortgage < income * 12 * 4.0 and rng.random() < 0.22:
                members = [people[m] for m in household.member_ids if m in people]
                if members and max(m.salary for m in members) <= p.salary:
                    price = listing.purchase_price
                    down = price * 0.12
                    hh_income = sum((m.salary + m.pension) for m in members if m.age >= 18)
                    payment = (price - down) * self.MORTGAGE_RATE_MONTHLY
                    if liquid >= down * 1.15 and (payment + committed) <= hh_income * 0.38 and p.mortgage + (price - down) <= income * 12 * 4.5:
                        self._transfer_to_buyer(listing, household, p, True, down, price)
                        stats["homes"] += 1
                        continue
            # --- invest in a rental unit
            if rng.random() < 0.10 and p.age >= 30:
                if developer_units is None:
                    developer_units = [l for l in world.housing_market.values() if l.owner_id is None and l.occupied]
                if not developer_units:
                    continue
                target = developer_units[rng.randrange(len(developer_units))]
                if target.owner_id is not None:
                    continue
                price = target.purchase_price
                down = price * 0.25
                payment = (price - down) * self.MORTGAGE_RATE_MONTHLY
                net_rent = target.rent * 0.78
                if liquid >= down * 1.2 and (committed + payment - net_rent) <= income * 0.40:
                    household = self.households.get(target.household_id) if target.household_id else None
                    self._transfer_to_buyer(target, household, p, False, down, price)
                    stats["investments"] += 1
        self.property_sales_month = stats
        return stats

    def service_mortgages(self) -> None:
        """Monthly mortgage payments: interest leaves the economy, principal pays the loan down."""
        for p in self.people.values():
            if p.mortgage <= 0.0:
                continue
            due = p.mortgage * self.MORTGAGE_RATE_MONTHLY
            paid = p.pay_living_costs(due)
            principal = paid * (1.0 - self.MORTGAGE_INTEREST_SHARE)
            p.mortgage = max(0.0, p.mortgage - principal)
            if p.mortgage < 1.0:
                p.mortgage = 0.0

    def collect_property_tax(self) -> float:
        """Monthly property tax on every owned building (paid by private owners)."""
        total = 0.0
        people = self.people
        for building in self.world.buildings.values():
            if building.value <= 0 or building.building_type in ("police", "court", "prison", "station", "park_facility", "school", "college"):
                continue
            tax = building.value * 0.011 / 12.0 * self.price_level
            owner = people.get(building.owner_id) if building.owner_id is not None else None
            if owner is not None:
                paid = min(owner.bank_balance, tax)
                owner.bank_balance -= paid
                tax = paid
            total += tax
        return total

    def take_consumer_spending(self) -> float:
        value = self.consumer_spend_pending
        self.consumer_spend_pending = 0.0
        return value

    def take_property_tax(self) -> float:
        value = self.property_tax_pending
        self.property_tax_pending = 0.0
        return value

    def transfer_estate(self, person: Person, dead_ids: set) -> None:
        """Pass a deceased owner's property to a spouse or child, else to a corporate landlord."""
        if not person.owned_buildings:
            return
        heir = None
        if person.spouse_id is not None and person.spouse_id in self.people and person.spouse_id not in dead_ids:
            heir = self.people[person.spouse_id]
        else:
            for cid in person.children_ids:
                child = self.people.get(cid)
                if child is not None and cid not in dead_ids and child.age >= 18:
                    heir = child
                    break
        for bid in person.owned_buildings:
            building = self.world.buildings.get(bid)
            if building is None:
                continue
            for listing in self.world._building_listings.get(bid, []):
                if listing.owner_id == person.person_id:
                    listing.owner_id = heir.person_id if heir is not None else None
                    if heir is None and listing.tenure == "own":
                        listing.tenure = "rent"
                        household = self.households.get(listing.household_id) if listing.household_id else None
                        if household is not None:
                            household.tenure = "rent"
                            household.monthly_costs = round(listing.rent * 1.15, 2)
            if building.owner_id == person.person_id:
                building.owner_id = heir.person_id if heir is not None else None
                building.owner_kind = "person" if heir is not None else "developer"
            if heir is not None and bid not in heir.owned_buildings:
                heir.owned_buildings.append(bid)
        if heir is not None:
            heir.assets += person.assets * 0.5
        person.owned_buildings = []

    def _assign_default_education(self) -> None:
        schools = self.world.building_by_type("school")
        colleges = self.world.building_by_type("college")
        for person in self.people.values():
            if 5 <= person.age < 18 and schools:
                school = self.rng.choice(schools)
                person.student_status = "school"
                person.current_enrollment_building = school.building_id
                person.education_level = "school" if person.age >= 12 else "none"
            elif 18 <= person.age <= 24 and colleges and person.education_level in {"school", "none"} and self.rng.random() < 0.22:
                college = self.rng.choice(colleges)
                person.student_status = "college"
                person.current_enrollment_building = college.building_id

    def _assign_initial_social_graph(self) -> None:
        self._district_residents: Dict[int, List[int]] = {}
        for other in self.people.values():
            if other.home_building_id is not None and other.home_building_id in self.world.buildings:
                did = self.world.buildings[other.home_building_id].district_id
                self._district_residents.setdefault(did, []).append(other.person_id)
        for person in self.people.values():
            household = self.household_for_person(person.person_id)
            if household is not None:
                for other_id in household.member_ids:
                    if other_id == person.person_id or other_id not in self.people:
                        continue
                    rel = person.relationship_with(other_id)
                    rel.affinity = self.rng.uniform(0.18, 0.72)
                    rel.trust = self.rng.uniform(0.32, 0.86)
                    rel.familiarity = self.rng.uniform(0.55, 1.0)
                    rel.last_interaction_day = 0
                    if other_id not in person.social_circle:
                        person.social_circle.append(other_id)
            district_candidates = [] if person.age < 5 else self._district_social_candidates(person, limit=10)
            for other_id in district_candidates:
                if other_id == person.person_id or self.people[other_id].age < 5:
                    continue
                rel = person.relationship_with(other_id)
                rel.affinity = self.rng.uniform(-0.2, 0.65)
                rel.trust = self.rng.uniform(0.0, 0.55)
                rel.familiarity = self.rng.uniform(0.05, 0.75)
                if rel.familiarity > 0.35:
                    if other_id not in person.social_circle:
                        person.social_circle.append(other_id)

    def _district_social_candidates(self, person: Person, limit: int = 10) -> List[int]:
        if person.home_building_id is None or person.home_building_id not in self.world.buildings:
            return []
        district_id = self.world.buildings[person.home_building_id].district_id
        residents = getattr(self, "_district_residents", {}).get(district_id, [])
        if len(residents) <= limit + 1:
            return [pid for pid in residents if pid != person.person_id]
        picked = self.rng.sample(residents, min(len(residents), limit + 1))
        return [pid for pid in picked if pid != person.person_id][:limit]

    # ------------------------------------------------------------------
    # High-level tick API
    # ------------------------------------------------------------------
    def daily_update(self, day_index: int) -> None:
        dead: List[int] = []
        for person in self.people.values():
            person.daily_update(self.world, day_index)
            if self._mortality_roll(person):
                dead.append(person.person_id)
        if dead:
            self.remove_people(dead)

    def hourly_update(self, day_index: int, hour: int) -> None:
        self.plan_hour(day_index, hour)
        for person in self.people.values():
            person.hourly_update(self.world, day_index, hour)

    def _slice(self, j: int, k: int) -> List["Person"]:
        cached = self._slice_cache.get(k)
        if cached is None or cached[0] != self._pop_gen or cached[1] != len(self.people):
            buckets: List[List["Person"]] = [[] for _ in range(k)]
            for p in self.people.values():
                buckets[p.person_id % k].append(p)
            cached = (self._pop_gen, len(self.people), buckets)
            self._slice_cache[k] = cached
        # Every add/remove of a citizen bumps _pop_gen, so a cache that survived the
        # check above holds exactly the living members of each bucket.  Callers only
        # iterate the list; it must not be mutated.
        return cached[2][j]

    def plan_slice(self, day_index: int, hour: int, j: int, k: int) -> None:
        """Plan + enrol one 1/k share of the population (id % k == j).  Spreading
        the hour-start work over several sub-steps removes the periodic stall and
        also staggers departures like real commuters.

        Equivalent to plan_hour(subset) followed by Person.hourly_position for each
        member, fused into two tight loops (same per-person order, same RNG draws)."""
        subset = self._slice(j, k)
        self.plan_hour(day_index, hour, subset, accumulate=j > 0)
        world = self.world
        people = self.people
        movers_add = world.movers.add
        for person in subset:
            if person.person_id not in people:
                continue
            block = person.schedule.get(hour) if person._schedule_day_index == day_index else None
            if block is None:
                person.hourly_position(world, day_index, hour)   # rare: slow general path
                continue
            # == Person.update_position_for_hour, inlined ==
            target = block.target
            if target:
                if person.location != target and person._route is None and person._failed_target != target:
                    person.set_destination(target, world, hour)
                elif person.destination != target:
                    person.set_destination(target, world, hour)
                if person._route is not None and person.location != person.destination:
                    movers_add(person.person_id)
            else:
                person.destination = person.location
                person._route = None
                person._route_i = 0

    def effects_slice(self, day_index: int, hour: int, j: int, k: int) -> None:
        """Hourly needs effects for one 1/k share; Person.hourly_effects inlined
        (refresh_floor, ambient effects and the social-contact pre-checks)."""
        world = self.world
        amb_stress = world._amb_stress
        amb_happy = world._amb_happy
        W = world.width
        H = world.height
        tiles = world.tiles
        for person in self._slice(j, k):
            block = person.schedule.get(hour)
            if block is None:
                block = person._plan_hour(world, day_index, hour)
                person.schedule[hour] = block
            activity = "rest" if person.incarcerated else block.activity
            loc = person.location
            # -- refresh_floor(block) --
            target = block.target
            b_act = block.activity
            if target is None or loc != target:
                person.floor = -1
            elif b_act == "socialize" or b_act == "exercise":
                person.floor = -1
            elif b_act == "work":
                person.floor = person.work_floor
            elif block.building_id is not None and block.building_id == person.home_building_id:
                person.floor = person.home_floor
            elif b_act == "study":
                person.floor = 1 + person.person_id % 3
            else:
                person.floor = 0
            person._apply_activity_effects(activity)
            # -- _ambient_effects_from_location(world) --
            x = loc[0]
            y = loc[1]
            i = (y % H) * W + (x % W)
            stress = person.stress + amb_stress[i]
            person.stress = 0.0 if stress < 0.0 else 1.0 if stress > 1.0 else stress
            h = person.happiness + amb_happy[i]
            person.happiness = 0.0 if h < 0.0 else 1.0 if h > 1.0 else h
            # -- _maybe_generate_spontaneous_social_contact pre-checks --
            if person.age < 5:
                continue
            occupants = tiles[y % H][x % W].occupants
            if len(occupants) < 2:
                continue
            if person.rng.random() > 0.15:
                continue
            person._spontaneous_contact(occupants, day_index)

    def daily_slice(self, day_index: int, j: int, k: int) -> None:
        dead: List[int] = []
        for person in self._slice(j, k):
            person.daily_update(self.world, day_index)
            if self._mortality_roll(person):
                dead.append(person.person_id)
        if dead:
            self.remove_people(dead)

    def plan_hour(self, day_index: int, hour: int, people: Optional[List["Person"]] = None, accumulate: bool = False) -> None:
        """Single scheduling pass for the hour; batches all new route requests
        so the routing service can answer them with one parallel A* call."""
        pending: List[Person] = []
        requests: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        kinds: List[int] = []
        world = self.world
        commuters = self.last_commuters if accumulate else 0
        for person in (self.people.values() if people is None else people):
            if person._schedule_day_index != day_index:
                person.schedule = {}
                person._daily_targets = {}
                person._schedule_day_index = day_index
                person._failed_target = None
            block = person.schedule.get(hour)
            if block is None:
                block = person._plan_hour(world, day_index, hour)
                person.schedule[hour] = block
            target = block.target
            if not target:
                continue
            activity = block.activity
            if activity[0] == "c" and activity.startswith("commute"):
                commuters += 1
            loc = person.location
            if loc == target:
                person.destination = target
                person._route = None
                person._route_i = 0
                continue
            route = person._route
            if person.destination != target or route is None or person._route_i >= route.shape[0]:
                person.destination = target
                if person._failed_target == target:
                    continue
                dist = abs(loc[0] - target[0]) + abs(loc[1] - target[1])
                mode = person._choose_mode(world, dist, hour)
                person.mode = mode
                pending.append(person)
                requests.append((loc, target))
                kinds.append(1 if mode in (MODE_CAR, MODE_BUS, MODE_TAXI) else 0)
        if requests:
            routes = world.route_service.batch_routes_arrays(requests, kinds=kinds)
            mode_tph = world.mode_tph
            for person, route in zip(pending, routes):
                if route.shape[0]:
                    person._route = route
                    person._route_i = 0
                    person._acc = 0.0
                    person._delay = (person.person_id * 7) % 6
                    person._failed_target = None
                    person.commute_minutes = route.shape[0] / max(1.0, mode_tph[person.mode]) * 60.0
                else:
                    person._route = None
                    person._route_i = 0
                    person._failed_target = person.destination
        self.last_commuters = commuters

    def _drive_vehicle(self, p: "Person", world: WorldMap, seconds: float = 120.0):
        """Drive one vehicle for a sub-step, stopping at red signals.

        Works at second-level resolution inside the 2-minute sub-step: the vehicle
        enters one tile every 1/speed seconds, and on reaching a signalised
        junction it waits for a green on its axis.  Returns (points, times) with a
        point per tile entry (and a repeated point to mark a wait), or None."""
        route = p._route
        n = route.shape[0]
        i = p._route_i
        loc = p.location
        if i < n and route[i, 0] == loc[0] and route[i, 1] == loc[1]:
            i += 1
        if i >= n:
            return None
        dt = 3600.0 / max(1.0, world.mode_tph[p.mode])
        pts, times, i2, nx, ny, acc, finished = drive_vehicle(
            route, i, int(loc[0]), int(loc[1]), p._acc, dt, seconds, signal_array(world), world.sim_seconds)
        p._acc = acc
        p._route_i = i2
        p.location = (nx, ny)
        return pts, times, finished

    def move_substep(self, hour: int = 12) -> Dict[int, np.ndarray]:
        """Advance every travelling citizen by one 2-minute sub-step along its route.

        Returns {person_id: waypoints traversed this sub-step} (start point included)
        so a renderer can animate continuous motion along the real path."""
        world = self.world
        movers = world.movers
        segments: Dict[int, np.ndarray] = {}
        times_out: Dict[int, np.ndarray] = {}
        self.last_times = times_out
        if not movers:
            self.last_segments = segments
            return segments
        people = self.people
        steps = world.mode_step
        # world.register_population_position, inlined below (same list ops, same order)
        locs = world.population_locations
        tiles = world.tiles
        W = world.width
        H = world.height
        done: List[int] = []
        # Drive all vehicles of this sub-step in one compiled batch.  Same selection
        # rule as the vehicle branch below, same order; each vehicle's result depends
        # only on its own state, so this equals driving them one by one in the loop.
        vehicles = []
        for pid in movers:
            p = people.get(pid)
            if p is not None and p._delay <= 0 and p._route is not None and p.mode >= MODE_CAR:
                vehicles.append(p)
        if vehicles:
            drives = drive_batch(vehicles, np.asarray(world.mode_tph, dtype=np.float64),
                                 signal_array(world), world.sim_seconds)
        vi = 0
        for pid in movers:
            p = people.get(pid)
            if p is None:
                done.append(pid)
                continue
            if p._delay > 0:
                p._delay -= 1
                continue
            route = p._route
            if route is None:
                done.append(pid)
                continue
            mode = p.mode
            if mode >= MODE_CAR:
                if vehicles[vi] is not p:     # invariant guard: both loops select identically
                    raise RuntimeError("vehicle batch out of sync with mover order")
                result = drives[vi]
                vi += 1
                if result is not None:
                    vpts, vtimes, p._route_i, x, y, p._acc, finished = result
                    p.location = (x, y)
                    result = (vpts, vtimes, finished)
                if result is None:
                    p._route = None
                    p._route_i = 0
                    done.append(pid)
                    continue
                vpts, vtimes, finished = result
                segments[pid] = vpts
                times_out[pid] = vtimes
                x, y = p.location
                old = locs.get(pid)
                if old:
                    occ = tiles[old[1] % H][old[0] % W].occupants
                    if pid in occ:
                        occ.remove(pid)
                occ = tiles[y % H][x % W].occupants
                if pid not in occ:
                    occ.append(pid)
                locs[pid] = (x, y)
                if finished:
                    p._route = None
                    p._route_i = 0
                    done.append(pid)
                continue
            speed = steps[mode]
            if mode <= MODE_BIKE and p.energy < 0.2:
                speed *= 0.6   # exhausted people slow down
            acc = p._acc + speed
            k = int(acc)
            p._acc = acc - k
            if k <= 0:
                continue
            i = p._route_i
            n = route.shape[0]
            loc = p.location
            if i < n and route[i, 0] == loc[0] and route[i, 1] == loc[1]:
                i += 1
            if i >= n:
                p._route = None
                p._route_i = 0
                done.append(pid)
                continue
            j = i + k - 1
            if j >= n:
                j = n - 1
            segments[pid] = route[(i - 1 if i > 0 else 0): j + 1]
            x = int(route[j, 0])
            y = int(route[j, 1])
            p.location = (x, y)
            old = locs.get(pid)
            if old:
                occ = tiles[old[1] % H][old[0] % W].occupants
                if pid in occ:
                    occ.remove(pid)
            occ = tiles[y % H][x % W].occupants
            if pid not in occ:
                occ.append(pid)
            locs[pid] = (x, y)
            fatigue = MODE_FATIGUE[mode]
            if fatigue:
                e = p.energy - fatigue * (j - i + 1)
                p.energy = 0.0 if e < 0.0 else e
            if j >= n - 1:
                p._route = None
                p._route_i = 0
                done.append(pid)
            else:
                p._route_i = j + 1
        for pid in done:
            movers.discard(pid)
            p = people.get(pid)
            if p is not None:
                p.refresh_floor(p.schedule.get(hour))
        self.last_segments = segments
        return segments

    def _mortality_roll(self, person: Person) -> bool:
        if person.age < 18:
            return False
        age_pressure = max(0.0, (person.age - 55) / 10.0) ** 2
        annual_chance = age_pressure * 0.02 * (1.2 - person.health)
        if person.health < 0.2:
            annual_chance += 0.02
        daily_chance = clamp(annual_chance / 365.0, 0.0, 0.02)
        return self.rng.random() < daily_chance

    def remove_person(self, person_id: int) -> bool:
        return self.remove_people([person_id]) > 0

    def remove_people(self, person_ids: List[int]) -> int:
        """Remove many people with a single sweep over survivors."""
        dead_ids: set = set()
        pending_estates: List[Person] = []
        for person_id in person_ids:
            person = self.people.pop(person_id, None)
            self._pop_gen += 1
            if person is None:
                continue
            dead_ids.add(person_id)
            pending_estates.append(person)
            for hook in self.death_hooks:
                hook(person)
            self.world.remove_population_position(person_id)
            if person.job_company_id is not None:
                company = self.world.companies.get(person.job_company_id)
                if company is not None and person_id in company.employee_ids:
                    company.employee_ids.remove(person_id)
                    company.payroll_budget = max(0.0, company.payroll_budget - person.salary / 12.0)
            if person.household_id is not None:
                household = self.households.get(person.household_id)
                if household is not None:
                    household.member_ids = [pid for pid in household.member_ids if pid != person_id]
                    listing = self.world.housing_market.get(household.listing_id)
                    if not household.member_ids:
                        if listing is not None:
                            listing.occupied = False
                            listing.household_id = None
                        self.households.pop(household.household_id, None)
            self.deaths_total += 1
            self.death_ages.append(person.age)
            if len(self.death_ages) > 400:
                self.death_ages = self.death_ages[-400:]
        if not dead_ids:
            return 0
        for deceased in pending_estates:
            self.transfer_estate(deceased, dead_ids)
        disjoint = dead_ids.isdisjoint
        for other in self.people.values():
            rels = other.relationships
            for pid in dead_ids:
                rels.pop(pid, None)
                other.relation_roles.pop(pid, None)
            if not disjoint(other.social_circle):
                other.social_circle = [pid for pid in other.social_circle if pid not in dead_ids]
            if not disjoint(other.family_ids):
                other.family_ids = [pid for pid in other.family_ids if pid not in dead_ids]
            if not disjoint(other.parent_ids):
                other.parent_ids = [pid for pid in other.parent_ids if pid not in dead_ids]
            if not disjoint(other.children_ids):
                other.children_ids = [pid for pid in other.children_ids if pid not in dead_ids]
            if other.spouse_id in dead_ids:
                other.spouse_id = None
        return len(dead_ids)

    def labor_stats(self) -> Tuple[int, int, int]:
        labor_force = [
            person
            for person in self.people.values()
            if 18 <= person.age < 65 and not person.incarcerated
        ]
        employed = sum(1 for person in labor_force if person.job_company_id is not None)
        return len(labor_force), employed, len(labor_force) - employed

    # ------------------------------------------------------------------
    # Career and household helpers
    # ------------------------------------------------------------------
    def unemployed_people(self) -> List[Person]:
        return [
            person
            for person in self.people.values()
            if person.job_company_id is None
            and 18 <= person.age < 65
            and not person.incarcerated
        ]

    def employed_people(self) -> List[Person]:
        return [
            person
            for person in self.people.values()
            if person.job_company_id is not None and not person.incarcerated
        ]

    def students(self) -> List[Person]:
        return [person for person in self.people.values() if person.student_status is not None]

    def nearest_people(self, x: int, y: int, radius: int = 4, limit: int = 12) -> List[Person]:
        found: List[Tuple[int, Person]] = []
        for person in self.people.values():
            distance = abs(person.location[0] - x) + abs(person.location[1] - y)
            if distance <= radius:
                found.append((distance, person))
        found.sort(key=lambda item: item[0])
        return [person for _, person in found[:limit]]

    def household_for_person(self, person_id: int) -> Optional[Household]:
        person = self.people[person_id]
        if person.household_id is None:
            return None
        return self.households.get(person.household_id)

    def pay_household_costs(self) -> None:
        """Monthly housing bills: rent flows to the unit's owner, owners pay upkeep, tax goes to the city."""
        world = self.world
        people = self.people
        price = self.price_level
        landlord_income = corporate = collected = 0.0
        for household in self.households.values():
            members = [people[pid] for pid in household.member_ids if pid in people]
            if not members:
                continue
            adults = [m for m in members if m.age >= 18]
            payers = adults or members
            total = household.monthly_costs * price
            share = total / len(payers)
            paid = 0.0
            for member in payers:
                paid += member.pay_living_costs(share)
            listing = world.housing_market.get(household.listing_id)
            if listing is not None and listing.tenure == "rent" and total > 0:
                rent_part = paid * min(1.0, listing.rent * price / total)
                upkeep = rent_part * 0.22
                building = world.buildings.get(listing.building_id)
                if building is not None:
                    building.maintenance_fund = building.maintenance_fund * 0.9 + upkeep
                net = rent_part - upkeep
                collected += rent_part
                owner = people.get(listing.owner_id) if listing.owner_id is not None else None
                if owner is not None:
                    owner.bank_balance += net
                    landlord_income += net
                else:
                    corporate += net
            household.arrears_months = 0 if paid >= total * 0.95 else household.arrears_months + 1
        self.landlord_income_month = landlord_income
        self.corporate_rent_month = corporate
        self.rent_collected_month = collected
        self.service_mortgages()
        # Consumption: food, goods and services.  This is the sink that stops savings
        # from growing without bound, and it becomes revenue for consumer-facing firms.
        spent = 0.0
        for p in people.values():
            if p.age >= 18:
                bill = (300.0 + 0.12 * (p.salary + p.pension)) * price
            else:
                bill = 110.0 * price
            if p.bank_balance + p.cash < 1500.0:
                bill *= 0.7   # tight budgets cut discretionary spending
            spent += p.pay_living_costs(bill)
        self.consumer_spend_pending += spent
        self.consumer_spend_month = spent
        self.property_tax_pending += self.collect_property_tax()

    # ------------------------------------------------------------------
    # Social graph and emergent family logic
    # ------------------------------------------------------------------
    def run_social_layer(self, day_index: int) -> List[Dict[str, Any]]:
        births: List[Dict[str, Any]] = []
        ids = list(self.people.keys())
        self.rng.shuffle(ids)
        for person_id in ids:
            person = self.people[person_id]
            if person.age < 18 or not person.social_circle:
                continue
            partner_id = self._choose_relationship_target(person)
            if partner_id not in self.people or partner_id == person.person_id:
                continue
            other = self.people[partner_id]
            rel_a = person.relationship_with(partner_id)
            rel_b = other.relationship_with(person.person_id)
            boost = 0.01 + min(person.personality.empathy, other.personality.empathy) * 0.03
            rel_a.affinity = clamp(rel_a.affinity + boost, -1.0, 1.0)
            rel_b.affinity = clamp(rel_b.affinity + boost, -1.0, 1.0)
            if person.maybe_form_relationship(partner_id):
                rel_b.romance = clamp(rel_b.romance + 0.08, 0.0, 1.0)
                if person.spouse_id is None and other.spouse_id is None and rel_a.romance > 0.72 and rel_b.romance > 0.68:
                    person.set_spouse(other)
            child = person.maybe_have_child(partner_id, self._person_counter)
            if child:
                births.append(child)
                self._person_counter += 1
        return births

    def _choose_relationship_target(self, person: Person) -> int:
        if person.spouse_id is not None and person.spouse_id in self.people:
            return person.spouse_id
        candidates = [
            other_id
            for other_id in person.social_circle
            if other_id in self.people
            and other_id not in person.family_ids
            and self.people[other_id].age >= 18
            and abs(self.people[other_id].age - person.age) <= 14
        ]
        if not candidates:
            return self.rng.choice(person.social_circle)
        return self.rng.choice(candidates)

    def create_child(self, new_person_id: int, parents: List[int]) -> Person:
        child = Person(new_person_id, self.rng, self.world, age=0)
        child.age = 0
        child.life_stage = "child"
        child.education_level = "none"
        child.health = clamp(0.75 + self.rng.uniform(-0.1, 0.1), 0.0, 1.0)
        child.location = self.people[parents[0]].location
        child.destination = child.location
        child.family_ids = list(parents)
        for parent_id in parents:
            if self.people[parent_id].household_id:
                child.household_id = self.people[parent_id].household_id
                child.home_building_id = self.people[parent_id].home_building_id
            child.add_parent(self.people[parent_id])
        self.people[child.person_id] = child
        self._pop_gen += 1
        self.world.register_population_position(child.person_id, *child.location)
        if child.household_id and child.household_id in self.households:
            self.households[child.household_id].member_ids.append(child.person_id)
        child.add_memory("life", "Entered the world as a newborn.", 0.1)
        self.births_total += 1
        return child

    # ------------------------------------------------------------------
    # Reporting helpers for the UI
    # ------------------------------------------------------------------
    def top_people_lines(self, limit: int = 8) -> List[str]:
        people = sorted(
            self.people.values(),
            key=lambda person: person.happiness * 0.35 + person.skills["software"] * 0.1 + person.reputation_public * 0.25 + person.bank_balance / 50_000.0,
            reverse=True,
        )
        lines = []
        for person in people[:limit]:
            title = person.job_title or "Unemployed"
            lines.append(
                f"{person.full_name[:18]:18} {title[:12]:12} "
                f"H:{person.happiness:0.2f} S:{person.stress:0.2f} ${person.bank_balance:0.0f}"
            )
        return lines

    def relationship_lines(self, person_id: int, limit: int = 8) -> List[str]:
        person = self.people[person_id]
        ordered = sorted(person.relationships.values(), key=lambda rel: rel.closeness(), reverse=True)
        lines = []
        for rel in ordered[:limit]:
            if rel.person_id not in self.people:
                continue
            other = self.people[rel.person_id]
            role = person.relation_roles.get(rel.person_id)
            role_label = f"{role[:6]:6} " if role else ""
            lines.append(
                f"{other.full_name[:18]:18} {role_label}A:{rel.affinity:0.2f} T:{rel.trust:0.2f} R:{rel.romance:0.2f}"
            )
        return lines

    def activity_heatmap(self, hour: int = 12) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for person in self.people.values():
            activity = person.current_activity(hour)
            counts[activity] = counts.get(activity, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Serialization support
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "people": {pid: person.to_dict() for pid, person in self.people.items()},
            "households": {hid: household.__dict__ for hid, household in self.households.items()},
            "counters": {
                "household": self._household_counter,
                "person": self._person_counter,
                "births": self.births_total,
                "deaths": self.deaths_total,
            },
            "rng_state": self.rng.getstate(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], world: WorldMap, schedule_day_index: int = 0) -> "Population":
        population = cls(seed=data["seed"], world=world, size=0)
        population.people = {}
        population.households = {
            int(hid): Household(**household)
            for hid, household in data["households"].items()
        }
        counters = data.get("counters", {})
        population._household_counter = counters.get("household", max(population.households, default=0) + 1)
        population._person_counter = counters.get("person", max((int(pid) for pid in data["people"]), default=0) + 1)
        population.births_total = counters.get("births", 0)
        population.deaths_total = counters.get("deaths", 0)
        world.population_locations.clear()
        for row in world.tiles:
            for tile in row:
                tile.occupants.clear()
        for pid, person_data in data["people"].items():
            person = Person.from_dict(person_data, population.rng, world, schedule_day_index=schedule_day_index)
            if not person.alive:
                continue
            population.people[int(pid)] = person
            population._pop_gen += 1
            world.register_population_position(person.person_id, *person.location)
        rng_state = coerce_rng_state(data.get("rng_state"))
        if rng_state is not None:
            population.rng.setstate(rng_state)
        world.movers = {
            pid for pid, p in population.people.items()
            if p.has_path() and p.location != p.destination
        }
        return population
