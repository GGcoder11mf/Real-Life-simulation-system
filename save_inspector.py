from __future__ import annotations

import gzip
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from world import clamp

try:
    import importlib.util

    MSGPACK_AVAILABLE = importlib.util.find_spec("msgpack") is not None
    if MSGPACK_AVAILABLE:
        import msgpack
except Exception:
    MSGPACK_AVAILABLE = False


WINDOW_SIZE = (1480, 920)
FPS = 60
SAVE_CONTAINER_MAGIC = b"RLS_SAVE_V2"
SAVE_EXTENSIONS = {".pkl", ".pickle", ".sav", ".save", ".gz"}
UI_BG = (15, 20, 28)
PANEL_BG = (24, 31, 42)
PANEL_ALT = (34, 42, 57)
PANEL_SOFT = (43, 53, 72)
CARD_BG = (30, 37, 49)
CARD_HOVER = (42, 51, 67)
CARD_SELECTED = (54, 91, 120)
TEXT = (236, 240, 245)
TEXT_MUTED = (166, 178, 193)
TEXT_FAINT = (124, 138, 156)
ACCENT = (77, 174, 255)
ACCENT_DIM = (53, 116, 170)
GOOD = (91, 214, 138)
WARN = (240, 199, 92)
BAD = (236, 103, 103)
SHADOW = (7, 10, 15)


def format_money(value: Any) -> str:
    if value is None:
        return "-"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1_000_000_000:
        return f"{sign}${amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"{sign}${amount / 1_000:.1f}K"
    return f"{sign}${amount:.0f}"


def format_ratio(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def ensure_tuple(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return (0, 0)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def preview_value(value: Any, max_len: int = 120) -> str:
    text = safe_text(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def relationship_value(rel: Any, field: str) -> float:
    if isinstance(rel, dict):
        try:
            return float(rel.get(field, 0.0))
        except (TypeError, ValueError):
            return 0.0
    if isinstance(rel, (list, tuple)):
        field_positions = {
            "affinity": 0,
            "trust": 1,
            "romance": 2,
            "respect": 3,
            "familiarity": 4,
            "rivalry": 5,
            "gossip_knowledge": 6,
            "last_interaction_day": 7,
            "interaction_count": 8,
        }
        index = field_positions.get(field)
        if index is None or index >= len(rel):
            return 0.0
        try:
            return float(rel[index])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def memory_field(memory: Any, field: str) -> Any:
    if isinstance(memory, dict):
        return memory.get(field)
    if isinstance(memory, (list, tuple)):
        field_positions = {
            "timestamp": 0,
            "category": 1,
            "headline": 2,
            "impact": 3,
            "location": 4,
            "related_people": 5,
            "tags": 6,
        }
        index = field_positions.get(field)
        if index is None or index >= len(memory):
            return None
        return memory[index]
    return None


def format_file_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def read_save_blob(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Save file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError("Empty save file (0 bytes).")
    with path.open("rb") as probe:
        prefix = probe.read(2)
    if prefix == b"\x1f\x8b":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    with path.open("rb") as handle:
        return handle.read()


def decode_save_blob(blob: bytes) -> Dict[str, Any]:
    if blob.startswith(SAVE_CONTAINER_MAGIC):
        codec_and_payload = blob[len(SAVE_CONTAINER_MAGIC):]
        codec_line, payload = codec_and_payload.split(b"\n", 1)
        codec = codec_line.decode("ascii")
        if codec == "msgpack":
            if not MSGPACK_AVAILABLE:
                raise RuntimeError("Save uses msgpack but msgpack is not installed.")
            return msgpack.unpackb(payload, raw=False, strict_map_key=False)
        return pickle.loads(payload)
    return pickle.loads(blob)


def load_save_data(path: Path) -> Dict[str, Any]:
    return decode_save_blob(read_save_blob(path))


def build_save_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    world = data.get("world", {})
    population = data.get("population", {})
    systems = data.get("systems", {})
    return {
        "schema": data.get("schema_version", world.get("schema_version", 1)),
        "seed": data.get("seed"),
        "world_seed": data.get("world_seed", world.get("seed")),
        "date": format_calendar(data.get("calendar", {})),
        "weather": format_weather(data.get("weather", {})),
        "people": len(population.get("people", {})),
        "households": len(population.get("households", {})),
        "companies": len(world.get("companies", {})),
        "districts": len(world.get("districts", {})),
        "buildings": len(world.get("buildings", {})),
        "transit": len(world.get("transit_lines", {})),
        "stocks": len(systems.get("stocks", {})),
        "news": len(data.get("news", [])),
        "timeline": len(data.get("timeline", [])),
    }


def format_calendar(calendar: Dict[str, Any]) -> str:
    if not calendar:
        return "Unknown date"
    month = int(calendar.get("month", 0))
    months = [
        "",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_name = months[month] if 0 <= month < len(months) else str(month)
    return f"{month_name} {calendar.get('day', '?')}, {calendar.get('year', '?')} {int(calendar.get('hour', 0)):02d}:00"


def format_weather(weather: Dict[str, Any]) -> str:
    if not weather:
        return "Unknown weather"
    return (
        f"{safe_text(weather.get('condition', 'clear')).title()} "
        f"{float(weather.get('temperature_c', 0.0)):.1f}C "
        f"wind {float(weather.get('wind_kph', 0.0)):.0f}kph"
    )


@dataclass
class EntityRecord:
    category: str
    record_id: str
    title: str
    subtitle: str
    search_text: str
    sort_values: Dict[str, Any]
    badges: List[str]
    details: List[Tuple[str, str]]
    sections: List[Tuple[str, List[str]]] = field(default_factory=list)


@dataclass
class SavePreview:
    path: Path
    status: str
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    size_bytes: int = 0
    modified: str = ""


class SaveIndex:
    def __init__(self, path: Path, data: Dict[str, Any]) -> None:
        self.path = path
        self.data = data
        self.summary = build_save_summary(data)
        self.records: List[EntityRecord] = []
        self.by_category: Dict[str, List[EntityRecord]] = {}
        self.sort_options: Dict[str, List[Tuple[str, str]]] = {}
        self._build()

    def _build(self) -> None:
        self.records.extend(self._build_people())
        self.records.extend(self._build_companies())
        self.records.extend(self._build_households())
        self.records.extend(self._build_districts())
        self.records.extend(self._build_buildings())
        self.records.extend(self._build_transit())
        self.records.extend(self._build_stocks())
        self.records.extend(self._build_news())
        self.records.extend(self._build_timeline())
        self.by_category = {"all": list(self.records)}
        for record in self.records:
            self.by_category.setdefault(record.category, []).append(record)
        self.sort_options = {
            "all": [("name", "Name"), ("id", "ID"), ("category", "Category")],
            "people": [("name", "Name"), ("wealth", "Wealth"), ("happiness", "Happiness"), ("stress", "Stress"), ("age", "Age")],
            "companies": [("name", "Name"), ("value", "Market Value"), ("health", "Health"), ("employees", "Employees")],
            "households": [("name", "Name"), ("members", "Members"), ("costs", "Monthly Cost"), ("savings", "Savings")],
            "districts": [("name", "Name"), ("wealth", "Wealth"), ("crime", "Crime"), ("desirability", "Desirability")],
            "buildings": [("name", "Name"), ("rent", "Rent"), ("capacity", "Capacity"), ("quality", "Quality")],
            "transit": [("name", "Name"), ("stops", "Stops"), ("speed", "Speed")],
            "stocks": [("name", "Name"), ("price", "Price"), ("change", "Change"), ("volume", "Volume")],
            "news": [("id", "Newest"), ("importance", "Importance"), ("name", "Headline")],
            "timeline": [("id", "Newest"), ("name", "Title"), ("category", "Type")],
        }

    def records_for(self, category: str) -> List[EntityRecord]:
        return self.by_category.get(category, [])

    def _build_people(self) -> List[EntityRecord]:
        population = self.data.get("population", {})
        people = population.get("people", {})
        households = population.get("households", {})
        world = self.data.get("world", {})
        companies = world.get("companies", {})
        buildings = world.get("buildings", {})
        records: List[EntityRecord] = []
        for person_id, person in people.items():
            household = households.get(str(person.get("household_id"))) or households.get(person.get("household_id"))
            company = companies.get(str(person.get("job_company_id"))) or companies.get(person.get("job_company_id"))
            home_building = buildings.get(str(person.get("home_building_id"))) or buildings.get(person.get("home_building_id"))
            wealth = float(person.get("cash", 0.0)) + float(person.get("bank_balance", 0.0)) + float(person.get("assets", 0.0)) - float(person.get("debt", 0.0))
            location = ensure_tuple(person.get("location"))
            badges = [
                person.get("job_title") or "Unemployed",
                person.get("education_level", "unknown"),
                person.get("vehicle", "none"),
            ]
            details = [
                ("ID", str(person_id)),
                ("Age", safe_text(person.get("age"))),
                ("Gender", safe_text(person.get("gender"))),
                ("Life Stage", safe_text(person.get("life_stage"))),
                ("Job", safe_text(person.get("job_title") or "Unemployed")),
                ("Employer", safe_text(company.get("name") if company else "-")),
                ("Home", safe_text(home_building.get("name") if home_building else "-")),
                ("Household", safe_text(person.get("household_id"))),
                ("Location", f"{location[0]}, {location[1]}"),
                ("Cash", format_money(person.get("cash"))),
                ("Bank", format_money(person.get("bank_balance"))),
                ("Assets", format_money(person.get("assets"))),
                ("Debt", format_money(person.get("debt"))),
                ("Net Worth", format_money(wealth)),
                ("Salary", format_money(person.get("salary"))),
                ("Health", format_ratio(person.get("health"))),
                ("Energy", format_ratio(person.get("energy"))),
                ("Stress", format_ratio(person.get("stress"))),
                ("Happiness", format_ratio(person.get("happiness"))),
                ("Motivation", format_ratio(person.get("motivation"))),
                ("Reputation", format_ratio(person.get("reputation_public"))),
                ("Karma", format_ratio(person.get("hidden_karma"))),
            ]
            skill_items = sorted((person.get("skills") or {}).items(), key=lambda item: item[1], reverse=True)[:6]
            relationship_items = sorted(
                (person.get("relationships") or {}).items(),
                key=lambda item: relationship_value(item[1], "familiarity"),
                reverse=True,
            )[:5]
            memory_items = list(person.get("memories", []))[-5:]
            sections = [
                ("Top Skills", [f"{name}: {float(value):.2f}" for name, value in skill_items] or ["No skill data"]),
                (
                    "Relationships",
                    [
                        f"Person {other_id}: affinity {relationship_value(rel, 'affinity'):.2f}, trust {relationship_value(rel, 'trust'):.2f}"
                        for other_id, rel in relationship_items
                    ]
                    or ["No relationships"],
                ),
                (
                    "Recent Memories",
                    [
                        f"{safe_text(memory_field(memory, 'timestamp'))}: {safe_text(memory_field(memory, 'headline'))}"
                        for memory in memory_items
                    ]
                    or ["No memories"],
                ),
                (
                    "Family",
                    [
                        f"Household {person.get('household_id')} with {len(household.get('member_ids', [])) if household else 0} members",
                        f"Family IDs: {', '.join(str(pid) for pid in person.get('family_ids', [])[:10]) or '-'}",
                        f"Children IDs: {', '.join(str(pid) for pid in person.get('children_ids', [])[:10]) or '-'}",
                    ],
                ),
            ]
            records.append(
                EntityRecord(
                    category="people",
                    record_id=str(person_id),
                    title=safe_text(person.get("full_name", f"Person {person_id}")),
                    subtitle=f"{person.get('job_title') or 'Unemployed'} | age {person.get('age')} | {format_money(wealth)}",
                    search_text=" ".join(
                        [
                            safe_text(person.get("full_name")),
                            safe_text(person.get("job_title")),
                            safe_text(person.get("education_level")),
                            safe_text(person.get("gender")),
                            safe_text(person.get("life_stage")),
                            safe_text(company.get("name") if company else ""),
                            safe_text(home_building.get("name") if home_building else ""),
                            " ".join(safe_text(text) for text in person.get("certifications", [])),
                        ]
                    ).lower(),
                    sort_values={
                        "name": safe_text(person.get("full_name")).lower(),
                        "id": int(person_id),
                        "category": "people",
                        "wealth": wealth,
                        "happiness": float(person.get("happiness", 0.0)),
                        "stress": float(person.get("stress", 0.0)),
                        "age": int(person.get("age", 0)),
                    },
                    badges=badges,
                    details=details,
                    sections=sections,
                )
            )
        return records

    def _build_companies(self) -> List[EntityRecord]:
        world = self.data.get("world", {})
        companies = world.get("companies", {})
        districts = world.get("districts", {})
        buildings = world.get("buildings", {})
        records: List[EntityRecord] = []
        for company_id, company in companies.items():
            district = districts.get(str(company.get("district_id"))) or districts.get(company.get("district_id"))
            building = buildings.get(str(company.get("building_id"))) or buildings.get(company.get("building_id"))
            employees = len(company.get("employee_ids", []))
            if employees <= 0:
                continue
            health = self._company_health_score(company)
            details = [
                ("ID", str(company_id)),
                ("Sector", safe_text(company.get("sector"))),
                ("Scale", safe_text(company.get("company_scale"))),
                ("District", safe_text(district.get("name") if district else "-")),
                ("Building", safe_text(building.get("name") if building else "-")),
                ("Employees", safe_text(employees)),
                ("Open Positions", safe_text(len(company.get("open_positions", [])))),
                ("Value", format_money(company.get("market_value"))),
                ("Cash", format_money(company.get("cash"))),
                ("Payroll", format_money(company.get("payroll_budget"))),
                ("Demand", format_ratio(company.get("demand"))),
                ("Supply", format_ratio(company.get("supply"))),
                ("Productivity", format_ratio(company.get("productivity"))),
                ("Prestige", format_ratio(company.get("prestige"))),
                ("Health", format_ratio(health)),
                ("Risk", format_ratio(company.get("bankruptcy_risk"))),
                ("Target Headcount", safe_text(company.get("target_headcount"))),
                ("Stock Symbol", safe_text(company.get("stock_symbol") or "-")),
            ]
            sections = [
                ("Open Positions", [safe_text(role) for role in company.get("open_positions", [])[:12]] or ["No open positions"]),
                (
                    "Shift Templates",
                    [f"{shift[0]:02d}:00-{shift[1]:02d}:00" for shift in company.get("shift_templates", [])[:8]]
                    or ["No shift templates"],
                ),
                (
                    "Employee IDs",
                    [", ".join(str(pid) for pid in company.get("employee_ids", [])[:20])] or ["No employees"],
                ),
            ]
            records.append(
                EntityRecord(
                    category="companies",
                    record_id=str(company_id),
                    title=safe_text(company.get("name", f"Company {company_id}")),
                    subtitle=f"{company.get('sector')} | {employees} employees | {format_money(company.get('market_value'))}",
                    search_text=" ".join(
                        [
                            safe_text(company.get("name")),
                            safe_text(company.get("sector")),
                            safe_text(company.get("company_scale")),
                            safe_text(district.get("name") if district else ""),
                            safe_text(building.get("name") if building else ""),
                            safe_text(company.get("stock_symbol")),
                        ]
                    ).lower(),
                    sort_values={
                        "name": safe_text(company.get("name")).lower(),
                        "id": int(company_id),
                        "category": "companies",
                        "value": float(company.get("market_value", 0.0)),
                        "health": health,
                        "employees": employees,
                    },
                    badges=[safe_text(company.get("sector")), safe_text(company.get("company_scale")), safe_text(company.get("stock_symbol") or "private")],
                    details=details,
                    sections=sections,
                )
            )
        return records

    def _company_health_score(self, company: Dict[str, Any]) -> float:
        payroll = max(1.0, float(company.get("payroll_budget", 0.0)))
        liquidity = clamp(float(company.get("cash", 0.0)) / payroll, 0.0, 1.0)
        balance = clamp((float(company.get("demand", 0.0)) - float(company.get("supply", 0.0))) * 0.5 + 0.5, 0.0, 1.0)
        risk = clamp(1.0 - float(company.get("bankruptcy_risk", 0.0)), 0.0, 1.0)
        productivity = clamp(float(company.get("productivity", 0.0)), 0.0, 1.0)
        return (liquidity + balance + risk + productivity) / 4.0

    def _build_households(self) -> List[EntityRecord]:
        population = self.data.get("population", {})
        people = population.get("people", {})
        households = population.get("households", {})
        world = self.data.get("world", {})
        buildings = world.get("buildings", {})
        records: List[EntityRecord] = []
        for household_id, household in households.items():
            member_ids = list(household.get("member_ids", []))
            members = [people.get(str(pid)) or people.get(pid) for pid in member_ids]
            member_names = [safe_text(member.get("full_name")) for member in members if member]
            building = buildings.get(str(household.get("home_building_id"))) or buildings.get(household.get("home_building_id"))
            details = [
                ("ID", str(household_id)),
                ("Members", safe_text(len(member_ids))),
                ("Home Building", safe_text(building.get("name") if building else "-")),
                ("Listing", safe_text(household.get("listing_id"))),
                ("Savings Pool", format_money(household.get("savings_pool"))),
                ("Monthly Costs", format_money(household.get("monthly_costs"))),
            ]
            sections = [
                ("Members", member_names or ["No linked members"]),
                ("Member IDs", [", ".join(str(pid) for pid in member_ids)] if member_ids else ["No member ids"]),
            ]
            records.append(
                EntityRecord(
                    category="households",
                    record_id=str(household_id),
                    title=f"Household {household_id}",
                    subtitle=f"{len(member_ids)} members | {format_money(household.get('monthly_costs'))} monthly",
                    search_text=" ".join([f"household {household_id}", safe_text(building.get("name") if building else ""), *member_names]).lower(),
                    sort_values={
                        "name": f"household {int(household_id):05d}",
                        "id": int(household_id),
                        "category": "households",
                        "members": len(member_ids),
                        "costs": float(household.get("monthly_costs", 0.0)),
                        "savings": float(household.get("savings_pool", 0.0)),
                    },
                    badges=[safe_text(len(member_ids)) + " members", format_money(household.get("monthly_costs"))],
                    details=details,
                    sections=sections,
                )
            )
        return records

    def _build_districts(self) -> List[EntityRecord]:
        world = self.data.get("world", {})
        districts = world.get("districts", {})
        records: List[EntityRecord] = []
        for district_id, district in districts.items():
            zoning_mix = district.get("zoning_mix", {})
            top_zone = max(zoning_mix, key=zoning_mix.get) if zoning_mix else "unknown"
            details = [
                ("ID", str(district_id)),
                ("Center", f"{ensure_tuple(district.get('center'))[0]}, {ensure_tuple(district.get('center'))[1]}"),
                ("Terrain", safe_text(district.get("dominant_terrain"))),
                ("Primary Zone", safe_text(top_zone)),
                ("Wealth", format_ratio(district.get("wealth"))),
                ("Crime", format_ratio(district.get("crime_heat"))),
                ("Transit", format_ratio(district.get("transit_score"))),
                ("Desirability", format_ratio(district.get("desirability"))),
                ("Capacity", safe_text(district.get("population_capacity"))),
                ("Average Rent", format_money(district.get("average_rent"))),
                ("Average Home Price", format_money(district.get("average_home_price"))),
                ("Buildings", safe_text(len(district.get("building_ids", [])))),
                ("Companies", safe_text(len(district.get("company_ids", [])))),
            ]
            sections = [
                ("Zoning Mix", [f"{name}: {float(value):.2f}" for name, value in sorted(zoning_mix.items())] or ["No zoning mix"]),
                ("Linked Buildings", [", ".join(str(item) for item in district.get("building_ids", [])[:20])] or ["No buildings"]),
                ("Linked Companies", [", ".join(str(item) for item in district.get("company_ids", [])[:20])] or ["No companies"]),
            ]
            records.append(
                EntityRecord(
                    category="districts",
                    record_id=str(district_id),
                    title=safe_text(district.get("name", f"District {district_id}")),
                    subtitle=f"{top_zone} | wealth {format_ratio(district.get('wealth'))} | crime {format_ratio(district.get('crime_heat'))}",
                    search_text=" ".join(
                        [
                            safe_text(district.get("name")),
                            safe_text(district.get("dominant_terrain")),
                            safe_text(top_zone),
                        ]
                    ).lower(),
                    sort_values={
                        "name": safe_text(district.get("name")).lower(),
                        "id": int(district_id),
                        "category": "districts",
                        "wealth": float(district.get("wealth", 0.0)),
                        "crime": float(district.get("crime_heat", 0.0)),
                        "desirability": float(district.get("desirability", 0.0)),
                    },
                    badges=[safe_text(top_zone), safe_text(district.get("dominant_terrain"))],
                    details=details,
                    sections=sections,
                )
            )
        return records

    def _build_buildings(self) -> List[EntityRecord]:
        world = self.data.get("world", {})
        buildings = world.get("buildings", {})
        districts = world.get("districts", {})
        companies = world.get("companies", {})
        records: List[EntityRecord] = []
        for building_id, building in buildings.items():
            district = districts.get(str(building.get("district_id"))) or districts.get(building.get("district_id"))
            company = companies.get(str(building.get("company_id"))) or companies.get(building.get("company_id"))
            details = [
                ("ID", str(building_id)),
                ("Type", safe_text(building.get("building_type"))),
                ("District", safe_text(district.get("name") if district else "-")),
                ("Tile", f"{ensure_tuple(building.get('tile'))[0]}, {ensure_tuple(building.get('tile'))[1]}"),
                ("Company", safe_text(company.get("name") if company else "-")),
                ("Capacity", safe_text(building.get("capacity"))),
                ("Quality", format_ratio(building.get("quality"))),
                ("Rent", format_money(building.get("rent"))),
                ("Price", format_money(building.get("purchase_price"))),
                ("Service", format_ratio(building.get("service_score"))),
                ("Job Slots", safe_text(building.get("job_slots"))),
                ("Level", safe_text(building.get("level"))),
                ("Maintenance", format_ratio(building.get("maintenance_backlog"))),
            ]
            sections = [
                (
                    "Open Hours",
                    [f"{building.get('open_hours', [8, 18])[0]:02d}:00-{building.get('open_hours', [8, 18])[1]:02d}:00"],
                ),
                ("Households", [", ".join(str(item) for item in building.get("household_ids", [])[:20])] or ["No households"]),
            ]
            records.append(
                EntityRecord(
                    category="buildings",
                    record_id=str(building_id),
                    title=safe_text(building.get("name", f"Building {building_id}")),
                    subtitle=f"{building.get('building_type')} | cap {building.get('capacity')} | rent {format_money(building.get('rent'))}",
                    search_text=" ".join(
                        [
                            safe_text(building.get("name")),
                            safe_text(building.get("building_type")),
                            safe_text(district.get("name") if district else ""),
                            safe_text(company.get("name") if company else ""),
                        ]
                    ).lower(),
                    sort_values={
                        "name": safe_text(building.get("name")).lower(),
                        "id": int(building_id),
                        "category": "buildings",
                        "rent": float(building.get("rent", 0.0)),
                        "capacity": float(building.get("capacity", 0.0)),
                        "quality": float(building.get("quality", 0.0)),
                    },
                    badges=[safe_text(building.get("building_type")), safe_text(district.get("name") if district else "")],
                    details=details,
                    sections=sections,
                )
            )
        return records

    def _build_transit(self) -> List[EntityRecord]:
        lines = self.data.get("world", {}).get("transit_lines", {})
        records: List[EntityRecord] = []
        for line_id, line in lines.items():
            details = [
                ("ID", str(line_id)),
                ("Type", safe_text(line.get("line_type"))),
                ("Stops", safe_text(len(line.get("stops", [])))),
                ("Speed", format_ratio(line.get("speed_multiplier"))),
                ("District Links", safe_text(len(line.get("district_ids", [])))),
            ]
            sections = [
                ("Stops", [f"{stop[0]}, {stop[1]}" for stop in line.get("stops", [])[:30]] or ["No stops"]),
                ("District IDs", [", ".join(str(item) for item in line.get("district_ids", [])[:20])] or ["No district ids"]),
            ]
            records.append(
                EntityRecord(
                    category="transit",
                    record_id=str(line_id),
                    title=safe_text(line.get("name", f"Transit Line {line_id}")),
                    subtitle=f"{line.get('line_type')} | {len(line.get('stops', []))} stops | speed {format_ratio(line.get('speed_multiplier'))}",
                    search_text=" ".join([safe_text(line.get("name")), safe_text(line.get("line_type"))]).lower(),
                    sort_values={
                        "name": safe_text(line.get("name")).lower(),
                        "id": int(line_id),
                        "category": "transit",
                        "stops": len(line.get("stops", [])),
                        "speed": float(line.get("speed_multiplier", 0.0)),
                    },
                    badges=[safe_text(line.get("line_type")), safe_text(len(line.get("stops", []))) + " stops"],
                    details=details,
                    sections=sections,
                )
            )
        return records

    def _build_stocks(self) -> List[EntityRecord]:
        stocks = self.data.get("systems", {}).get("stocks", {})
        world = self.data.get("world", {})
        companies = world.get("companies", {})
        records: List[EntityRecord] = []
        for symbol, stock in stocks.items():
            company = companies.get(str(stock.get("company_id"))) or companies.get(stock.get("company_id"))
            details = [
                ("Symbol", symbol),
                ("Company", safe_text(company.get("name") if company else f"Company {stock.get('company_id')}")),
                ("Price", format_money(stock.get("price"))),
                ("Change", format_ratio(stock.get("last_change"))),
                ("Shares", safe_text(stock.get("shares_outstanding"))),
                ("Volume", safe_text(stock.get("daily_volume"))),
                ("Volatility", format_ratio(stock.get("volatility"))),
            ]
            sections = [("Notes", [f"Linked company id: {safe_text(stock.get('company_id'))}"])]
            records.append(
                EntityRecord(
                    category="stocks",
                    record_id=symbol,
                    title=symbol,
                    subtitle=f"{safe_text(company.get('name') if company else 'Unlinked company')} | {format_money(stock.get('price'))}",
                    search_text=" ".join([symbol, safe_text(company.get("name") if company else "")]).lower(),
                    sort_values={
                        "name": symbol.lower(),
                        "id": symbol.lower(),
                        "category": "stocks",
                        "price": float(stock.get("price", 0.0)),
                        "change": float(stock.get("last_change", 0.0)),
                        "volume": float(stock.get("daily_volume", 0.0)),
                    },
                    badges=[format_money(stock.get("price")), f"d {format_ratio(stock.get('last_change'))}"],
                    details=details,
                    sections=sections,
                )
            )
        return records

    def _build_news(self) -> List[EntityRecord]:
        records: List[EntityRecord] = []
        news = list(self.data.get("news", []))
        for index, item in enumerate(reversed(news), start=1):
            details = [
                ("Slot", str(index)),
                ("Time", safe_text(item.get("timestamp"))),
                ("Category", safe_text(item.get("category"))),
                ("Importance", format_ratio(item.get("importance"))),
            ]
            sections = [("Detail", [safe_text(item.get("detail")) or "No detail"])]
            records.append(
                EntityRecord(
                    category="news",
                    record_id=str(index),
                    title=safe_text(item.get("headline", f"News {index}")),
                    subtitle=f"{item.get('category')} | {safe_text(item.get('timestamp'))}",
                    search_text=" ".join(
                        [safe_text(item.get("headline")), safe_text(item.get("detail")), safe_text(item.get("category"))]
                    ).lower(),
                    sort_values={
                        "name": safe_text(item.get("headline")).lower(),
                        "id": -index,
                        "category": "news",
                        "importance": float(item.get("importance", 0.0)),
                    },
                    badges=[safe_text(item.get("category")), f"imp {format_ratio(item.get('importance'))}"],
                    details=details,
                    sections=sections,
                )
            )
        return records

    def _build_timeline(self) -> List[EntityRecord]:
        records: List[EntityRecord] = []
        timeline = list(self.data.get("timeline", []))
        for index, item in enumerate(reversed(timeline), start=1):
            payload = item.get("payload", {})
            sections = [("Payload", [f"{key}: {preview_value(value)}" for key, value in payload.items()] or ["No payload"])]
            records.append(
                EntityRecord(
                    category="timeline",
                    record_id=str(index),
                    title=safe_text(item.get("title", f"Timeline Event {index}")),
                    subtitle=f"{item.get('event_type')} | {safe_text(item.get('timestamp'))}",
                    search_text=" ".join(
                        [safe_text(item.get("title")), safe_text(item.get("event_type")), safe_text(item.get("timestamp"))]
                    ).lower(),
                    sort_values={
                        "name": safe_text(item.get("title")).lower(),
                        "id": -index,
                        "category": "timeline",
                    },
                    badges=[safe_text(item.get("event_type"))],
                    details=[
                        ("Slot", str(index)),
                        ("Time", safe_text(item.get("timestamp"))),
                        ("Type", safe_text(item.get("event_type"))),
                    ],
                    sections=sections,
                )
            )
        return records


class SaveInspectorApp:
    def __init__(self, root: Path) -> None:
        pygame.init()
        pygame.key.set_repeat(350, 32)
        pygame.display.set_caption("RLS Save Inspector")
        self.screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.root = root
        self.fonts = {
            "tiny": pygame.font.SysFont("segoeui,arial", 13),
            "small": pygame.font.SysFont("segoeui,arial", 15),
            "body": pygame.font.SysFont("segoeui,arial", 17),
            "section": pygame.font.SysFont("segoeui,arial", 20, bold=True),
            "title": pygame.font.SysFont("segoeui,arial", 24, bold=True),
            "hero": pygame.font.SysFont("segoeui,arial", 30, bold=True),
        }
        self.running = True
        self.mode = "loader"
        self.loader_items: List[SavePreview] = []
        self.loader_selection = 0
        self.loader_scroll = 0
        self.loader_message = "Scanning for save files..."
        self.loader_query = ""
        self.loader_input_active = False
        self.index: Optional[SaveIndex] = None
        self.category_tabs = ["all", "people", "companies", "households", "districts", "buildings", "transit", "stocks", "news", "timeline"]
        self.selected_category = "all"
        self.search_text = ""
        self.sort_key = "name"
        self.result_selection = 0
        self.result_scroll = 0
        self.detail_scroll = 0
        self._detail_max_scroll = 0
        self._detail_viewport = pygame.Rect(0, 0, 1, 1)
        self._list_viewport = pygame.Rect(0, 0, 1, 1)
        self._loader_viewport = pygame.Rect(0, 0, 1, 1)
        self.status_message = "Type to search | Up/Down select | Tab category | F2 sort | Ctrl+L switch save"
        self._click_regions: List[Tuple[pygame.Rect, str, Any]] = []
        self._hover_regions: List[Tuple[pygame.Rect, str, Any]] = []
        self._grad_cache: Dict[Tuple[int, int, Tuple[int, int, int], Tuple[int, int, int]], pygame.Surface] = {}
        self._filter_cache_key: Optional[Tuple[int, str, str, str]] = None
        self._filter_cache: List[EntityRecord] = []
        self._last_click_time = 0
        self._last_click_item: Any = None
        self.refresh_loader()

    def refresh_loader(self) -> None:
        self.loader_items = self.scan_save_files()
        self.loader_selection = 0
        self.loader_scroll = 0
        if not self.loader_items:
            self.loader_message = "No .pkl/.save files found. Type a path below or copy one into this folder."
        else:
            ok_count = sum(1 for item in self.loader_items if item.status == "ok")
            self.loader_message = f"Found {len(self.loader_items)} file(s) ({ok_count} readable)."

    def scan_save_files(self) -> List[SavePreview]:
        candidates: List[Path] = []
        skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules"}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() in SAVE_EXTENSIONS:
                    candidates.append(path)
        candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        previews: List[SavePreview] = []
        for path in candidates[:80]:
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            try:
                summary = build_save_summary(load_save_data(path))
                previews.append(
                    SavePreview(
                        path=path,
                        status="ok",
                        summary=summary,
                        size_bytes=stat.st_size,
                        modified=modified,
                    )
                )
            except Exception as exc:
                previews.append(
                    SavePreview(
                        path=path,
                        status="error",
                        error=str(exc),
                        size_bytes=stat.st_size,
                        modified=modified,
                    )
                )
        return previews

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(1100, event.w), max(720, event.h)), pygame.RESIZABLE)
        elif event.type == pygame.KEYDOWN:
            if self.mode == "loader":
                self.handle_loader_key(event)
            else:
                self.handle_browser_key(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.handle_mouse(event.pos, event.button)
        elif event.type == pygame.MOUSEWHEEL:
            self.handle_wheel(pygame.mouse.get_pos(), event.y)

    def handle_wheel(self, pos: Tuple[int, int], delta_y: int) -> None:
        if delta_y == 0:
            return
        if self.mode == "loader":
            self.loader_scroll = max(0, self.loader_scroll - delta_y * 2)
        else:
            if self._detail_viewport.collidepoint(pos):
                self.detail_scroll = int(clamp(self.detail_scroll - delta_y * 44, 0, self._detail_max_scroll))
            else:
                self.result_scroll = max(0, self.result_scroll - delta_y * 3)

    def handle_loader_key(self, event: pygame.event.Event) -> None:
        if self.loader_input_active:
            if event.key == pygame.K_RETURN:
                raw = self.loader_query.strip().strip('"').strip("'")
                if raw:
                    self.load_path(Path(raw))
                self.loader_input_active = False
            elif event.key == pygame.K_ESCAPE:
                self.loader_input_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.loader_query = self.loader_query[:-1]
            else:
                if event.unicode and event.unicode.isprintable():
                    self.loader_query += event.unicode
            return
        if event.key in (pygame.K_DOWN, pygame.K_j):
            self.loader_selection = min(len(self.loader_items) - 1, self.loader_selection + 1) if self.loader_items else 0
        elif event.key in (pygame.K_UP, pygame.K_k):
            self.loader_selection = max(0, self.loader_selection - 1)
        elif event.key == pygame.K_RETURN and self.loader_items:
            self.load_preview(self.loader_items[self.loader_selection])
        elif event.key == pygame.K_r:
            self.refresh_loader()
        elif event.key == pygame.K_SLASH:
            self.loader_input_active = True
        elif event.key == pygame.K_ESCAPE:
            if self.index is not None:
                self.mode = "browser"
            else:
                self.running = False

    def handle_browser_key(self, event: pygame.event.Event) -> None:
        ctrl = bool(event.mod & pygame.KMOD_CTRL)
        shift = bool(event.mod & pygame.KMOD_SHIFT)
        if ctrl and event.key == pygame.K_l:
            self.mode = "loader"
            self.loader_input_active = False
            self.status_message = "Pick another save."
            return
        if event.key == pygame.K_ESCAPE:
            if self.search_text:
                self.search_text = ""
                self.result_selection = 0
                self.result_scroll = 0
                self.detail_scroll = 0
            else:
                self.mode = "loader"
                self.loader_input_active = False
            return
        if event.key == pygame.K_F2:
            options = self.current_sort_options()
            keys = [item[0] for item in options]
            if self.sort_key not in keys:
                self.sort_key = keys[0]
            else:
                self.sort_key = keys[(keys.index(self.sort_key) + 1) % len(keys)]
            self.result_selection = 0
            self.result_scroll = 0
            self.detail_scroll = 0
            return
        if event.key == pygame.K_TAB:
            step = -1 if shift else 1
            index = (self.category_tabs.index(self.selected_category) + step) % len(self.category_tabs)
            self.select_category(self.category_tabs[index])
            return
        if event.key == pygame.K_DOWN:
            count = len(self.filtered_records())
            if count:
                self.result_selection = min(count - 1, self.result_selection + 1)
                self.detail_scroll = 0
                self._ensure_result_visible()
            return
        if event.key == pygame.K_UP:
            if self.result_selection > 0:
                self.result_selection -= 1
                self.detail_scroll = 0
                self._ensure_result_visible()
            return
        if event.key == pygame.K_PAGEUP:
            self.result_selection = max(0, self.result_selection - 10)
            self.detail_scroll = 0
            self._ensure_result_visible()
            return
        if event.key == pygame.K_PAGEDOWN:
            count = len(self.filtered_records())
            if count:
                self.result_selection = min(count - 1, self.result_selection + 10)
                self.detail_scroll = 0
                self._ensure_result_visible()
            return
        if event.key == pygame.K_BACKSPACE:
            if self.search_text:
                self.search_text = self.search_text[:-1]
                self.result_selection = 0
                self.result_scroll = 0
                self.detail_scroll = 0
            return
        if not ctrl and event.unicode and event.unicode.isprintable():
            self.search_text += event.unicode
            self.result_selection = 0
            self.result_scroll = 0
            self.detail_scroll = 0

    def _ensure_result_visible(self) -> None:
        visible = max(1, self._list_viewport.h // 84)
        if self.result_selection < self.result_scroll:
            self.result_scroll = self.result_selection
        elif self.result_selection >= self.result_scroll + visible:
            self.result_scroll = self.result_selection - visible + 1

    def handle_mouse(self, pos: Tuple[int, int], button: int) -> None:
        if button in (4, 5):
            self.handle_wheel(pos, 1 if button == 4 else -1)
            return
        if button != 1:
            return
        now = pygame.time.get_ticks()
        for rect, action, payload in reversed(self._click_regions):
            if rect.collidepoint(pos):
                if action == "select_preview":
                    is_double = (self._last_click_item == payload and now - self._last_click_time < 380)
                    self.loader_selection = payload
                    self.loader_input_active = False
                    self._last_click_item = payload
                    self._last_click_time = now
                    if is_double and 0 <= payload < len(self.loader_items):
                        self.load_preview(self.loader_items[payload])
                elif action == "open_selected_preview":
                    if 0 <= self.loader_selection < len(self.loader_items):
                        self.load_preview(self.loader_items[self.loader_selection])
                elif action == "refresh_loader":
                    self.refresh_loader()
                elif action == "loader_input":
                    self.loader_input_active = True
                elif action == "select_category":
                    self.select_category(payload)
                elif action == "select_sort":
                    self.sort_key = payload
                    self.result_selection = 0
                    self.result_scroll = 0
                    self.detail_scroll = 0
                elif action == "select_result":
                    self.result_selection = payload
                    self.detail_scroll = 0
                elif action == "clear_search":
                    self.search_text = ""
                    self.result_selection = 0
                    self.result_scroll = 0
                    self.detail_scroll = 0
                elif action == "open_loader":
                    self.mode = "loader"
                    self.loader_input_active = False
                return
        if self.mode == "loader":
            self.loader_input_active = False

    def load_preview(self, preview: SavePreview) -> None:
        if preview.status != "ok":
            self.loader_message = preview.error or "This file could not be loaded."
            return
        self.load_path(preview.path)

    def load_path(self, path: Path) -> None:
        try:
            resolved = path if path.is_absolute() else (self.root / path)
            data = load_save_data(resolved)
            self.index = SaveIndex(resolved, data)
            self._filter_cache_key = None
            self._filter_cache = []
            self.mode = "browser"
            self.selected_category = "all"
            self.sort_key = "name"
            self.result_selection = 0
            self.result_scroll = 0
            self.detail_scroll = 0
            self.search_text = ""
            self.status_message = f"Loaded {resolved.name} ({len(self.index.records):,} records). Type to filter, Tab for categories, F2 to sort."
        except Exception as exc:
            self.loader_message = f"Load failed: {exc}"
            self.status_message = f"Load failed: {exc}"

    def filtered_records(self) -> List[EntityRecord]:
        if not self.index:
            return []
        key = (id(self.index), self.selected_category, self.search_text, self.sort_key)
        if self._filter_cache_key == key:
            return self._filter_cache
        records = list(self.index.records_for(self.selected_category))
        if self.search_text.strip():
            tokens = [token.lower() for token in self.search_text.split() if token.strip()]
            records = [
                record for record in records
                if all(token in record.search_text or token in record.record_id.lower() for token in tokens)
            ]
        reverse = self.sort_key not in {"name", "id", "category"}
        records.sort(key=lambda record: record.sort_values.get(self.sort_key, record.sort_values.get("name")), reverse=reverse)
        self._filter_cache_key = key
        self._filter_cache = records
        return records

    def current_sort_options(self) -> List[Tuple[str, str]]:
        if not self.index:
            return [("name", "Name")]
        return self.index.sort_options.get(self.selected_category, self.index.sort_options["all"])

    def select_category(self, category: str) -> None:
        self.selected_category = category
        options = [item[0] for item in self.current_sort_options()]
        if self.sort_key not in options:
            self.sort_key = options[0]
        self.result_selection = 0
        self.result_scroll = 0
        self.detail_scroll = 0

    def selected_record(self) -> Optional[EntityRecord]:
        records = self.filtered_records()
        if not records:
            return None
        self.result_selection = int(clamp(self.result_selection, 0, len(records) - 1))
        return records[self.result_selection]

    def draw(self) -> None:
        self.screen.fill(UI_BG)
        self._click_regions = []
        self._hover_regions = []
        if self.mode == "loader":
            self.draw_loader()
        else:
            self.draw_browser()

    def draw_loader(self) -> None:
        width, height = self.screen.get_size()
        mouse_pos = pygame.mouse.get_pos()
        self.draw_gradient((0, 0, width, height), (11, 18, 28), (22, 34, 49))
        outer = pygame.Rect(28, 24, width - 56, height - 48)
        self.draw_panel(outer, fill=PANEL_BG, radius=20)

        self.draw_text("RLS Save Inspector", self.fonts["hero"], TEXT, (outer.x + 26, outer.y + 18))
        self.draw_text(
            "Select a save file (double-click or Enter) or enter a path below.",
            self.fonts["body"],
            TEXT_MUTED,
            (outer.x + 28, outer.y + 58),
            max_w=outer.w - 220,
        )

        refresh_btn = pygame.Rect(outer.right - 156, outer.y + 22, 130, 38)
        refresh_hover = refresh_btn.collidepoint(mouse_pos)
        self.draw_panel(refresh_btn, fill=PANEL_SOFT if refresh_hover else CARD_BG, radius=10, border=(95, 115, 145) if refresh_hover else (70, 82, 102))
        self.draw_centered_text("Refresh (R)", self.fonts["small"], TEXT, refresh_btn)
        self._click_regions.append((refresh_btn, "refresh_loader", None))

        body_top = outer.y + 98
        body_h = outer.bottom - body_top - 102
        left_w = int(outer.w * 0.53)
        left = pygame.Rect(outer.x + 24, body_top, left_w, body_h)
        right = pygame.Rect(left.right + 16, body_top, outer.right - left.right - 40, body_h)
        self.draw_panel(left, fill=PANEL_ALT, radius=16)
        self.draw_panel(right, fill=PANEL_ALT, radius=16)

        self.draw_text("Save Files", self.fonts["title"], TEXT, (left.x + 18, left.y + 14))
        self.draw_text(self.loader_message, self.fonts["small"], TEXT_MUTED, (left.x + 18, left.y + 46), max_w=left.w - 36)

        list_rect = pygame.Rect(left.x + 12, left.y + 74, left.w - 24, left.h - 86)
        self._loader_viewport = list_rect
        row_h = 80
        visible_rows = max(1, list_rect.h // row_h)
        max_scroll = max(0, len(self.loader_items) - visible_rows)
        self.loader_scroll = int(clamp(self.loader_scroll, 0, max_scroll))
        if self.loader_selection < self.loader_scroll:
            self.loader_scroll = self.loader_selection
        if self.loader_selection >= self.loader_scroll + visible_rows:
            self.loader_scroll = self.loader_selection - visible_rows + 1

        has_bar = len(self.loader_items) > visible_rows
        card_w = list_rect.w - (14 if has_bar else 0)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(list_rect)
        for draw_index in range(visible_rows + 1):
            item_index = self.loader_scroll + draw_index
            if item_index >= len(self.loader_items):
                break
            preview = self.loader_items[item_index]
            card = pygame.Rect(list_rect.x, list_rect.y + draw_index * row_h, card_w, row_h - 8)
            if card.top >= list_rect.bottom:
                break
            hovered = card.collidepoint(mouse_pos)
            is_sel = item_index == self.loader_selection
            color = CARD_SELECTED if is_sel else (CARD_HOVER if hovered else CARD_BG)
            border = ACCENT if is_sel else ((95, 115, 145) if hovered else (62, 74, 94))
            self.draw_panel(card, fill=color, radius=12, border=border, shadow=False)
            self._click_regions.append((card.clip(list_rect), "select_preview", item_index))

            if preview.status == "ok" and preview.summary:
                pill_text = f"{preview.summary['people']:,} ppl"
                pill_color = (34, 90, 65)
                pill_fg = GOOD
            else:
                pill_text = "Unreadable"
                pill_color = (92, 38, 42)
                pill_fg = BAD
            pill_w = self.fonts["tiny"].size(pill_text)[0] + 18
            pill_rect = pygame.Rect(card.right - pill_w - 12, card.y + 10, pill_w, 22)
            self.draw_panel(pill_rect, fill=pill_color, radius=7, border=pill_fg, shadow=False)
            self.draw_centered_text(pill_text, self.fonts["tiny"], pill_fg, pill_rect)

            title_w = pill_rect.x - card.x - 24
            self.draw_text(preview.path.name, self.fonts["body"], TEXT, (card.x + 14, card.y + 10), max_w=title_w)

            rel = str(preview.path.relative_to(self.root)) if preview.path.is_relative_to(self.root) else str(preview.path)
            meta_right = f"{format_file_size(preview.size_bytes)}  •  {preview.modified}"
            meta_w = self.fonts["tiny"].size(meta_right)[0]
            self.draw_text(meta_right, self.fonts["tiny"], TEXT_FAINT, (card.right - meta_w - 12, card.y + 42))
            self.draw_text(rel, self.fonts["small"], TEXT_MUTED, (card.x + 14, card.y + 40), max_w=card.w - meta_w - 36)
        self.screen.set_clip(old_clip)

        if has_bar:
            track = pygame.Rect(list_rect.right - 8, list_rect.y, 6, list_rect.h)
            self.draw_scrollbar(track, len(self.loader_items), visible_rows, self.loader_scroll)

        self.draw_text("Preview", self.fonts["title"], TEXT, (right.x + 20, right.y + 14))
        selected = self.loader_items[self.loader_selection] if self.loader_items else None
        if selected is None:
            self.draw_text("No save selected.", self.fonts["body"], TEXT_MUTED, (right.x + 20, right.y + 64))
        elif selected.status != "ok" or not selected.summary:
            self.draw_text(selected.path.name, self.fonts["section"], TEXT, (right.x + 20, right.y + 60), max_w=right.w - 40)
            self.draw_text(
                f"{format_file_size(selected.size_bytes)}  •  Modified {selected.modified}",
                self.fonts["small"],
                TEXT_FAINT,
                (right.x + 20, right.y + 92),
                max_w=right.w - 40,
            )
            err_box = pygame.Rect(right.x + 20, right.y + 128, right.w - 40, 120)
            self.draw_panel(err_box, fill=(46, 26, 30), radius=12, border=BAD, shadow=False)
            y_err = err_box.y + 14
            for line in self.wrap_text(selected.error or "This file could not be decoded.", self.fonts["body"], err_box.w - 28)[:4]:
                self.draw_text(line, self.fonts["body"], BAD, (err_box.x + 14, y_err))
                y_err += 24
        else:
            summary = selected.summary
            self.draw_text(selected.path.name, self.fonts["section"], TEXT, (right.x + 20, right.y + 56), max_w=right.w - 40)
            meta_lines = [
                f"Date: {summary['date']}   |   Schema v{summary['schema']}",
                f"Weather: {summary['weather']}",
                f"Seed: {summary['seed']}   |   World: {summary['world_seed']}   |   {format_file_size(selected.size_bytes)}",
            ]
            y_meta = right.y + 88
            for line in meta_lines:
                self.draw_text(line, self.fonts["small"], TEXT_MUTED, (right.x + 20, y_meta), max_w=right.w - 40)
                y_meta += 24

            stats = [
                ("People", f"{summary['people']:,}"),
                ("Households", f"{summary['households']:,}"),
                ("Companies", f"{summary['companies']:,}"),
                ("Districts", f"{summary['districts']:,}"),
                ("Buildings", f"{summary['buildings']:,}"),
                ("Transit", f"{summary['transit']:,}"),
                ("Stocks", f"{summary['stocks']:,}"),
                ("News", f"{summary['news']:,}"),
                ("Timeline", f"{summary['timeline']:,}"),
            ]
            grid_top = y_meta + 12
            btn_h = 42
            grid_bottom = right.bottom - btn_h - 32
            cols = 3
            rows = 3
            gap_x, gap_y = 12, 12
            card_w = (right.w - 40 - gap_x * (cols - 1)) // cols
            card_h = max(54, min(74, (grid_bottom - grid_top - gap_y * (rows - 1)) // rows))
            for idx, (label, value) in enumerate(stats):
                col = idx % cols
                row = idx // cols
                rect = pygame.Rect(
                    right.x + 20 + col * (card_w + gap_x),
                    grid_top + row * (card_h + gap_y),
                    card_w,
                    card_h,
                )
                self.draw_panel(rect, fill=CARD_BG, radius=12, border=(56, 68, 88), shadow=False)
                self.draw_text(label, self.fonts["tiny"], TEXT_MUTED, (rect.x + 12, rect.y + 8))
                self.draw_text(value, self.fonts["title"], TEXT, (rect.x + 12, rect.y + 26), max_w=rect.w - 24)

            open_btn = pygame.Rect(right.x + 20, right.bottom - btn_h - 16, right.w - 40, btn_h)
            open_hover = open_btn.collidepoint(mouse_pos)
            self.draw_panel(open_btn, fill=ACCENT if open_hover else ACCENT_DIM, radius=12, border=ACCENT, shadow=False)
            self.draw_centered_text("Inspect Save (Enter)", self.fonts["body"], TEXT, open_btn)
            self._click_regions.append((open_btn, "open_selected_preview", None))

        input_label_y = outer.bottom - 88
        self.draw_text("Open by path (press / or click):", self.fonts["tiny"], TEXT_FAINT, (outer.x + 26, input_label_y))
        input_rect = pygame.Rect(outer.x + 24, outer.bottom - 68, outer.w - 48, 38)
        input_border = ACCENT if self.loader_input_active else (64, 76, 98)
        self.draw_panel(input_rect, fill=(16, 22, 32), radius=10, border=input_border, shadow=False)
        self._click_regions.append((input_rect, "loader_input", None))
        if self.loader_input_active or self.loader_query:
            cursor = "|" if (self.loader_input_active and (pygame.time.get_ticks() // 500) % 2 == 0) else ""
            path_label = self.loader_query + cursor
            path_color = TEXT
        else:
            path_label = "Press / and type a full or relative path to a save file, then press Enter"
            path_color = TEXT_FAINT
        self.draw_text(path_label, self.fonts["body"], path_color, (input_rect.x + 12, input_rect.y + 8), max_w=input_rect.w - 24)

        help_text = "Up/Down select  •  Enter / Double-click load  •  R refresh  •  / path input  •  Esc quit"
        self.draw_text(help_text, self.fonts["tiny"], TEXT_FAINT, (outer.x + 26, outer.bottom - 22), max_w=outer.w - 52)

    def draw_browser(self) -> None:
        if not self.index:
            return
        width, height = self.screen.get_size()
        mouse_pos = pygame.mouse.get_pos()
        self.draw_gradient((0, 0, width, height), (10, 16, 24), (21, 29, 41))

        top = pygame.Rect(18, 14, width - 36, 78)
        footer_h = 26
        body_top = top.bottom + 12
        body_h = height - body_top - footer_h - 14
        left_w = int(clamp(width * 0.37, 440, 560))
        left = pygame.Rect(18, body_top, left_w, body_h)
        right = pygame.Rect(left.right + 14, body_top, width - left.right - 32, body_h)

        self.draw_panel(top, fill=PANEL_BG, radius=16)
        self.draw_panel(left, fill=PANEL_BG, radius=16)
        self.draw_panel(right, fill=PANEL_BG, radius=16)

        summary = self.index.summary
        open_button = pygame.Rect(top.right - 184, top.y + 18, 164, 42)
        open_hover = open_button.collidepoint(mouse_pos)
        self.draw_panel(open_button, fill=ACCENT if open_hover else ACCENT_DIM, radius=12, border=ACCENT, shadow=False)
        self.draw_centered_text("Load Save (Ctrl+L)", self.fonts["small"], TEXT, open_button)
        self._click_regions.append((open_button, "open_loader", None))

        header_max_w = open_button.x - top.x - 36
        self.draw_text(self.index.path.name, self.fonts["title"], TEXT, (top.x + 20, top.y + 12), max_w=header_max_w)
        subtitle = (
            f"{summary['date']}  •  {summary['weather']}  •  {summary['people']:,} people  •  "
            f"{summary['companies']:,} companies  •  schema v{summary['schema']}"
        )
        self.draw_text(subtitle, self.fonts["small"], TEXT_MUTED, (top.x + 20, top.y + 46), max_w=header_max_w)

        # ---- Left panel: Search, Categories, Sort, Results ----
        search_rect = pygame.Rect(left.x + 14, left.y + 14, left.w - 28, 38)
        self.draw_panel(search_rect, fill=(16, 22, 32), radius=10, border=ACCENT_DIM if self.search_text else (64, 76, 98), shadow=False)
        clear_reserve = 0
        if self.search_text:
            clear_rect = pygame.Rect(search_rect.right - 30, search_rect.y + 6, 24, 26)
            clear_hover = clear_rect.collidepoint(mouse_pos)
            self.draw_panel(clear_rect, fill=PANEL_SOFT if clear_hover else CARD_BG, radius=6, border=(80, 94, 118), shadow=False)
            self.draw_centered_text("x", self.fonts["small"], TEXT, clear_rect)
            self._click_regions.append((clear_rect, "clear_search", None))
            clear_reserve = 34
        cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        query = (self.search_text + cursor) if self.search_text else "Search names, IDs, jobs, sectors, districts, symbols..."
        self.draw_text(
            query,
            self.fonts["body"],
            TEXT if self.search_text else TEXT_FAINT,
            (search_rect.x + 12, search_rect.y + 8),
            max_w=search_rect.w - 24 - clear_reserve,
        )

        tab_x = left.x + 14
        tab_y = search_rect.bottom + 10
        tab_h = 26
        for category in self.category_tabs:
            label = category.title()
            tab_w = max(56, self.fonts["tiny"].size(label)[0] + 20)
            if tab_x + tab_w > left.right - 14:
                tab_x = left.x + 14
                tab_y += tab_h + 6
            rect = pygame.Rect(tab_x, tab_y, tab_w, tab_h)
            active = category == self.selected_category
            hovered = rect.collidepoint(mouse_pos)
            fill = ACCENT_DIM if active else (CARD_HOVER if hovered else PANEL_SOFT)
            border = ACCENT if active else (70, 82, 102)
            self.draw_panel(rect, fill=fill, radius=8, border=border, shadow=False)
            self.draw_centered_text(label, self.fonts["tiny"], TEXT, rect)
            self._click_regions.append((rect, "select_category", category))
            tab_x += tab_w + 6

        sort_y = tab_y + tab_h + 12
        self.draw_text("Sort:", self.fonts["tiny"], TEXT_FAINT, (left.x + 14, sort_y + 5))
        sort_x = left.x + 52
        for key, label in self.current_sort_options():
            btn_w = max(56, self.fonts["tiny"].size(label)[0] + 18)
            if sort_x + btn_w > left.right - 14:
                sort_x = left.x + 52
                sort_y += 28
            rect = pygame.Rect(sort_x, sort_y, btn_w, 24)
            active = key == self.sort_key
            hovered = rect.collidepoint(mouse_pos)
            fill = ACCENT_DIM if active else (CARD_HOVER if hovered else PANEL_SOFT)
            border = ACCENT if active else (70, 82, 102)
            self.draw_panel(rect, fill=fill, radius=7, border=border, shadow=False)
            self.draw_centered_text(label, self.fonts["tiny"], TEXT, rect)
            self._click_regions.append((rect, "select_sort", key))
            sort_x += btn_w + 6

        records = self.filtered_records()
        count_y = sort_y + 32
        count_label = f"{len(records):,} match{'es' if len(records) != 1 else ''} in {self.selected_category}"
        self.draw_text(count_label, self.fonts["tiny"], TEXT_MUTED, (left.x + 14, count_y), max_w=left.w - 140)
        self.draw_text("F2 sort  •  Tab category", self.fonts["tiny"], TEXT_FAINT, (left.right - 150, count_y))

        list_rect = pygame.Rect(left.x + 12, count_y + 22, left.w - 24, left.bottom - count_y - 34)
        self._list_viewport = list_rect
        row_h = 84
        visible_rows = max(1, list_rect.h // row_h)
        max_scroll = max(0, len(records) - visible_rows)
        self.result_scroll = int(clamp(self.result_scroll, 0, max_scroll))
        if self.result_selection < self.result_scroll:
            self.result_scroll = self.result_selection
        if self.result_selection >= self.result_scroll + visible_rows:
            self.result_scroll = self.result_selection - visible_rows + 1

        has_list_bar = len(records) > visible_rows
        card_w = list_rect.w - (14 if has_list_bar else 0)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(list_rect)
        for draw_index in range(visible_rows + 1):
            item_index = self.result_scroll + draw_index
            if item_index >= len(records):
                break
            record = records[item_index]
            rect = pygame.Rect(list_rect.x, list_rect.y + draw_index * row_h, card_w, row_h - 8)
            if rect.top >= list_rect.bottom:
                break
            hovered = rect.collidepoint(mouse_pos)
            is_sel = item_index == self.result_selection
            fill = CARD_SELECTED if is_sel else (CARD_HOVER if hovered else CARD_BG)
            border = ACCENT if is_sel else ((95, 115, 145) if hovered else (58, 70, 88))
            self.draw_panel(rect, fill=fill, radius=12, border=border, shadow=False)
            self._click_regions.append((rect.clip(list_rect), "select_result", item_index))

            id_tag = f"#{record.record_id}"
            id_w = self.fonts["tiny"].size(id_tag)[0]
            self.draw_text(id_tag, self.fonts["tiny"], TEXT_FAINT, (rect.right - id_w - 12, rect.y + 10))
            self.draw_text(record.title, self.fonts["body"], TEXT, (rect.x + 12, rect.y + 8), max_w=rect.w - id_w - 32)
            self.draw_text(record.subtitle, self.fonts["tiny"], TEXT_MUTED, (rect.x + 12, rect.y + 30), max_w=rect.w - 24)

            badge_x = rect.x + 12
            badge_y = rect.y + 49
            for badge in record.badges[:3]:
                badge_text = preview_value(badge, 18)
                if not badge_text:
                    continue
                badge_w = self.fonts["tiny"].size(badge_text)[0] + 14
                if badge_x + badge_w > rect.right - 10:
                    break
                badge_rect = pygame.Rect(badge_x, badge_y, badge_w, 19)
                self.draw_panel(badge_rect, fill=(32, 62, 88), radius=6, border=(56, 102, 142), shadow=False)
                self.draw_centered_text(badge_text, self.fonts["tiny"], TEXT, badge_rect)
                badge_x += badge_w + 6
        self.screen.set_clip(old_clip)

        if has_list_bar:
            track = pygame.Rect(list_rect.right - 8, list_rect.y, 6, list_rect.h)
            self.draw_scrollbar(track, len(records), visible_rows, self.result_scroll)

        # ---- Right panel: Detail View ----
        record = self.selected_record()
        if record is None:
            self.draw_text("No matching records.", self.fonts["title"], TEXT, (right.x + 24, right.y + 24))
            self.draw_text("Try clearing or broadening your search query.", self.fonts["body"], TEXT_MUTED, (right.x + 24, right.y + 60))
        else:
            cat_badge = f"{record.category.title()}  #{record.record_id}"
            cat_w = self.fonts["small"].size(cat_badge)[0] + 20
            cat_rect = pygame.Rect(right.right - cat_w - 22, right.y + 20, cat_w, 28)
            self.draw_panel(cat_rect, fill=(32, 66, 94), radius=8, border=ACCENT_DIM, shadow=False)
            self.draw_centered_text(cat_badge, self.fonts["small"], ACCENT, cat_rect)

            self.draw_text(record.title, self.fonts["hero"], TEXT, (right.x + 22, right.y + 16), max_w=cat_rect.x - right.x - 32)
            self.draw_text(record.subtitle, self.fonts["body"], TEXT_MUTED, (right.x + 22, right.y + 56), max_w=right.w - 44)
            pygame.draw.line(self.screen, (55, 67, 86), (right.x + 20, right.y + 90), (right.right - 20, right.y + 90), 1)

            viewport = pygame.Rect(right.x + 18, right.y + 100, right.w - 36, right.h - 114)
            self._detail_viewport = viewport
            col_gap = 22
            content_w = viewport.w - 14
            col_a_w = max(260, int(content_w * 0.46))
            col_b_w = max(240, content_w - col_a_w - col_gap)
            col_a_x = viewport.x
            col_b_x = col_a_x + col_a_w + col_gap

            # Precompute vertical flow height for both columns so scrolling covers everything
            snap_h = 34 + len(record.details) * 28
            sec_h = 0
            wrapped_sections: List[Tuple[str, List[str]]] = []
            for sec_title, sec_lines in record.sections:
                sec_wrapped: List[str] = []
                for line in sec_lines:
                    sec_wrapped.extend(self.wrap_text(str(line), self.fonts["small"], col_b_w - 16))
                wrapped_sections.append((sec_title, sec_wrapped))
                sec_h += 34 + max(1, len(sec_wrapped)) * 20 + 18

            total_content_h = max(snap_h, sec_h)
            self._detail_max_scroll = max(0, total_content_h - viewport.h)
            self.detail_scroll = int(clamp(self.detail_scroll, 0, self._detail_max_scroll))

            old_clip = self.screen.get_clip()
            self.screen.set_clip(viewport)
            y_a = viewport.y - self.detail_scroll
            self.draw_text("Snapshot", self.fonts["section"], TEXT, (col_a_x, y_a))
            y_a += 34
            label_w = 136
            val_max_w = max(100, col_a_w - label_w - 16)
            for idx, (label, value) in enumerate(record.details):
                row_rect = pygame.Rect(col_a_x, y_a, col_a_w, 26)
                if row_rect.bottom >= viewport.top and row_rect.top <= viewport.bottom:
                    if idx % 2 == 0:
                        pygame.draw.rect(self.screen, (28, 36, 49), row_rect, border_radius=6)
                    self.draw_text(label, self.fonts["small"], TEXT_FAINT, (row_rect.x + 8, row_rect.y + 4), max_w=label_w - 8)
                    self.draw_text(value, self.fonts["body"], TEXT, (row_rect.x + label_w, row_rect.y + 2), max_w=val_max_w)
                y_a += 28

            y_b = viewport.y - self.detail_scroll
            for sec_title, sec_wrapped in wrapped_sections:
                if y_b + 30 >= viewport.top and y_b <= viewport.bottom:
                    self.draw_text(sec_title, self.fonts["section"], TEXT, (col_b_x, y_b))
                    pygame.draw.line(self.screen, (52, 64, 82), (col_b_x, y_b + 26), (col_b_x + col_b_w, y_b + 26), 1)
                y_b += 34
                for seg in sec_wrapped:
                    if y_b + 20 >= viewport.top and y_b <= viewport.bottom:
                        self.draw_text(seg, self.fonts["small"], TEXT_MUTED, (col_b_x + 4, y_b))
                    y_b += 20
                y_b += 18
            self.screen.set_clip(old_clip)

            if self._detail_max_scroll > 0:
                track = pygame.Rect(viewport.right - 6, viewport.y, 6, viewport.h)
                self.draw_scrollbar(track, total_content_h, viewport.h, self.detail_scroll)

        footer_y = height - footer_h - 6
        self.draw_text(self.status_message, self.fonts["tiny"], TEXT_FAINT, (22, footer_y + 4), max_w=width - 44)

    def draw_scrollbar(self, track: pygame.Rect, total: int, visible: int, offset: int) -> None:
        if total <= 0 or total <= visible or track.h <= 8:
            return
        pygame.draw.rect(self.screen, (26, 33, 46), track, border_radius=3)
        thumb_h = max(24, int(track.h * (visible / total)))
        max_off = max(1, total - visible)
        thumb_y = track.y + int((track.h - thumb_h) * (offset / max_off))
        pygame.draw.rect(self.screen, (88, 108, 138), pygame.Rect(track.x, thumb_y, track.w, thumb_h), border_radius=3)

    def draw_panel(
        self,
        rect: pygame.Rect,
        fill: Tuple[int, int, int] = PANEL_BG,
        radius: int = 16,
        border: Tuple[int, int, int] = (64, 76, 96),
        shadow: bool = True,
    ) -> None:
        if shadow:
            pygame.draw.rect(self.screen, SHADOW, rect.move(0, 3), border_radius=radius)
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        pygame.draw.rect(self.screen, border, rect, width=1, border_radius=radius)

    def fit_text(self, text: str, font: pygame.font.Font, max_w: int) -> str:
        if max_w <= 10:
            return ""
        if font.size(text)[0] <= max_w:
            return text
        ellipsis = "..."
        ell_w = font.size(ellipsis)[0]
        if ell_w >= max_w:
            return ""
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.size(text[:mid])[0] + ell_w <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + ellipsis

    def draw_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: Tuple[int, int, int],
        pos: Tuple[int, int],
        max_w: Optional[int] = None,
    ) -> None:
        rendered = self.fit_text(text, font, max_w) if max_w is not None else text
        if not rendered:
            return
        surface = font.render(rendered, True, color)
        self.screen.blit(surface, pos)

    def draw_centered_text(self, text: str, font: pygame.font.Font, color: Tuple[int, int, int], rect: pygame.Rect) -> None:
        rendered = self.fit_text(text, font, rect.w - 8)
        if not rendered:
            return
        surface = font.render(rendered, True, color)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def draw_gradient(self, rect_like: Tuple[int, int, int, int], top_color: Tuple[int, int, int], bottom_color: Tuple[int, int, int]) -> None:
        rect = pygame.Rect(rect_like)
        key = (rect.w, rect.h, top_color, bottom_color)
        target = self._grad_cache.get(key)
        if target is None:
            target = pygame.Surface((rect.w, rect.h))
            for y in range(rect.h):
                blend = y / max(1, rect.h - 1)
                color = (
                    int(top_color[0] * (1 - blend) + bottom_color[0] * blend),
                    int(top_color[1] * (1 - blend) + bottom_color[1] * blend),
                    int(top_color[2] * (1 - blend) + bottom_color[2] * blend),
                )
                pygame.draw.line(target, color, (0, y), (rect.w, y))
            self._grad_cache = {key: target}
        self.screen.blit(target, rect.topleft)

    def wrap_text(self, text: str, font: pygame.font.Font, width: int) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            if font.size(trial)[0] <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines


def main() -> None:
    app = SaveInspectorApp(Path.cwd())
    if len(sys.argv) > 1:
        app.load_path(Path(sys.argv[1]))
    app.run()


if __name__ == "__main__":
    main()
