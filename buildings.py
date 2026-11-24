"""
buildings.py

Vertical structure, ownership and occupancy of buildings.

* Every building has floors.  Houses are 1-3 storeys, apartment blocks grow
  with their unit count, and commercial buildings are sized from their
  company: headcount decides how many floors are needed and (for offices) the
  company's market value decides how prestigious/tall the tower is.
* Each floor has a function (lobby, department, wards, classrooms...) and a
  capacity.  Employees are assigned a work floor when hired; residents live on
  their unit's floor.  Citizens carry a `floor` (-1 = outside, 0 = ground) that
  the schedule updates, so a building can be inspected floor by floor.
* Residential units have an owner (a private landlord, an owner-occupier, or a
  corporate landlord) and rent flows from tenants to that owner.

Nothing here runs per citizen per tick except O(1) lookups; the heavy
`describe_building` is only evaluated for the building being inspected.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

WORKERS_PER_FLOOR = {"office": 28, "shop": 14, "factory": 45, "warehouse": 30, "hospital": 30}
MAX_FLOORS = {
    "house": 3, "apartment": 30, "shop": 3, "office": 44, "factory": 3, "warehouse": 3,
    "school": 4, "college": 8, "police": 4, "court": 5, "prison": 6, "hospital": 12,
    "park_facility": 1, "station": 2,
}
SECTOR_DEPARTMENTS = {
    "retail": ["Sales floor", "Stockroom", "Back office"],
    "logistics": ["Dispatch", "Fleet operations", "Planning"],
    "manufacturing": ["Assembly line", "Quality control", "Engineering"],
    "software": ["Engineering", "Product", "QA & Operations", "Data science"],
    "healthcare": ["Wards", "Surgery", "Diagnostics", "Outpatient"],
    "education": ["Teaching", "Administration", "Research"],
    "food": ["Kitchen", "Service floor", "Management"],
    "transport": ["Control room", "Maintenance", "Planning"],
    "construction": ["Estimating", "Site operations", "Engineering"],
    "finance": ["Trading", "Accounts", "Advisory", "Compliance"],
    "media": ["Newsroom", "Studio", "Sales & marketing"],
    "public_service": ["Casework", "Inspection", "Administration"],
}
CIVIC_LABELS = {
    "school": ["Entrance & canteen", "Classrooms", "Classrooms", "Library & gym"],
    "college": ["Atrium", "Lecture halls", "Labs", "Labs", "Library", "Faculty", "Faculty", "Administration"],
    "police": ["Front desk", "Detectives", "Records", "Command"],
    "court": ["Public hall", "Courtrooms", "Courtrooms", "Chambers", "Records"],
    "prison": ["Intake", "Cell block A", "Cell block B", "Cell block C", "Infirmary", "Admin"],
    "station": ["Concourse", "Platforms"],
    "park_facility": ["Visitor centre"],
}


def is_company_site(btype: str) -> bool:
    return btype in WORKERS_PER_FLOOR


def company_floors(btype: str, employees: int, target: int, market_value: float) -> int:
    wpf = WORKERS_PER_FLOOR[btype]
    plan = max(employees, int(target * 0.85))
    need = math.ceil(plan / wpf)
    if btype in ("office", "hospital") and plan > 40:
        need += 1  # lobby / reception floor
    if btype == "office":
        prestige = int(max(0.0, math.log2(max(market_value, 1.0) / 150_000.0)) * 1.6)
        need = max(need, 2 + prestige)
    return max(1, min(MAX_FLOORS[btype], need))


def initial_floors(building: Any, listings: List[Any], rng: Any) -> int:
    """Floors for a building at world generation time (company sites are refreshed later)."""
    btype = building.building_type
    if btype == "house":
        if building.quality < 0.45:
            return 1
        return 3 if (building.quality > 0.82 and building.capacity >= 5) else 2
    if btype == "apartment":
        units = max(1, len(listings))
        per_floor = 4 if building.quality > 0.5 else 6
        return max(2, min(MAX_FLOORS["apartment"], math.ceil(units / per_floor) + 1))
    if btype in WORKERS_PER_FLOOR:
        return company_floors(btype, 0, building.job_slots // 3 or 10, 300_000.0)
    cap = building.capacity
    per = {"school": 40, "college": 45, "police": 20, "court": 15, "prison": 25, "station": 20, "park_facility": 20}.get(btype, 30)
    return max(1, min(MAX_FLOORS.get(btype, 3), math.ceil(cap / per)))


def work_floor_range(btype: str, floors: int) -> Tuple[int, int]:
    if btype in ("office", "hospital") and floors >= 3:
        return 1, floors - 1
    return 0, floors - 1


def floor_for_rank(btype: str, floors: int, rank: int) -> int:
    if floors <= 1:
        return 0
    lo, hi = work_floor_range(btype, floors)
    if btype == "office":
        # Offices spread staff over every work floor (roomier, higher-end space).
        return lo + rank % (hi - lo + 1)
    wpf = WORKERS_PER_FLOOR.get(btype, 30)
    return min(hi, lo + rank // wpf)


def assign_residential_units(building: Any, listings: List[Any]) -> None:
    """Place a building's housing listings on its floors."""
    if building.building_type == "apartment":
        per_floor = max(1, math.ceil(len(listings) / max(1, building.floors - 1)))
        building.units_per_floor = per_floor
        for i, listing in enumerate(listings):
            listing.floor = 1 + i // per_floor
            listing.unit_no = (i % per_floor) + 1
            if listing.floor >= building.floors:
                listing.floor = building.floors - 1
    else:
        building.units_per_floor = 1
        for listing in listings:
            listing.floor = 0
            listing.unit_no = 1


def floor_plan(building: Any, company: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Describe each floor: label and nominal capacity (people)."""
    btype = building.building_type
    F = max(1, building.floors)
    plan: List[Dict[str, Any]] = []
    if btype == "house":
        for f in range(F):
            plan.append({"floor": f, "label": "Living & kitchen" if f == 0 else "Bedrooms", "capacity": 4, "kind": "home"})
    elif btype == "apartment":
        for f in range(F):
            if f == 0:
                plan.append({"floor": 0, "label": "Lobby & laundry", "capacity": 10, "kind": "common"})
            else:
                plan.append({"floor": f, "label": f"Apartments {building.units_per_floor} units", "capacity": building.units_per_floor * 4, "kind": "home"})
    elif btype in WORKERS_PER_FLOOR:
        wpf = WORKERS_PER_FLOOR[btype]
        depts = SECTOR_DEPARTMENTS.get(company.sector if company is not None else "", ["Operations", "Administration"])
        lo, hi = work_floor_range(btype, F)
        for f in range(F):
            if f < lo:
                label, cap, kind = ("Lobby & cafeteria", wpf // 2, "common")
            else:
                label = depts[(f - lo) % len(depts)]
                if btype == "shop" and f == 0:
                    label = "Sales floor"
                elif btype == "factory":
                    label = "Production hall" if f == 0 else label
                elif btype == "warehouse":
                    label = "Storage & dispatch" if f == 0 else label
                if btype == "office" and f == F - 1 and F >= 8:
                    label = "Executive suite"
                cap, kind = (wpf, "work")
            plan.append({"floor": f, "label": label, "capacity": cap, "kind": kind})
    else:
        labels = CIVIC_LABELS.get(btype, ["Main hall"])
        per = max(5, building.capacity // F)
        for f in range(F):
            plan.append({"floor": f, "label": labels[min(f, len(labels) - 1)], "capacity": per, "kind": "civic"})
    return plan


def refresh_structure(world: Any, immediate: bool = False) -> bool:
    """Resize company buildings to their company and refresh building values.

    Floors move at most one storey per call unless `immediate`.  Returns True
    if any floor count changed (so frontends can re-render the skyline)."""
    changed = False
    for company in world.companies.values():
        building = world.buildings.get(company.building_id)
        if building is None or not is_company_site(building.building_type):
            continue
        wanted = company_floors(building.building_type, len(company.employee_ids), company.target_headcount, company.market_value)
        if wanted != building.floors:
            if immediate:
                building.floors = wanted
            else:
                building.floors += 1 if wanted > building.floors else -1
            changed = True
        # Commercial property value follows the company that occupies it.
        building.value = max(50_000.0, company.market_value * 0.32 + building.floors * 60_000.0)
    index = getattr(world, "_building_listings", {})
    for bid, listings in index.items():
        building = world.buildings.get(bid)
        if building is not None and listings:
            building.value = sum(l.purchase_price for l in listings) * (0.92 if building.building_type == "apartment" else 1.0)
    if changed:
        world.structure_version = getattr(world, "structure_version", 0) + 1
    return changed


def reassign_work_floors(world: Any, population: Any) -> None:
    """Spread each company's staff across its floors (after floors change or a load)."""
    people = population.people
    for company in world.companies.values():
        building = world.buildings.get(company.building_id)
        if building is None:
            continue
        btype, floors = building.building_type, building.floors
        for rank, pid in enumerate(company.employee_ids):
            person = people.get(pid)
            if person is not None:
                person.work_floor = floor_for_rank(btype, floors, rank)


def _owner_label(engine: Any, owner_id: Optional[int], kind: str) -> str:
    if owner_id is not None:
        person = engine.population.people.get(owner_id)
        if person is not None:
            owned = len(person.owned_buildings)
            extra = f" ({owned} building{'s' if owned != 1 else ''})" if owned > 1 else ""
            return f"{person.full_name}{extra}"
        return "Estate of a former owner"
    return {"city": "City government", "company": "Company-owned"}.get(kind, "Corporate landlord")


def describe_building(engine: Any, building_id: int) -> Optional[Dict[str, Any]]:
    """Full inspector payload for one building (only evaluated on demand)."""
    world = engine.world
    population = engine.population
    building = world.buildings.get(building_id)
    if building is None:
        return None
    x, y = building.tile
    occupants = list(world.tile_at(x, y).occupants)
    people = population.people
    present_by_floor: Dict[int, int] = {}
    activity_counts: Dict[str, int] = {}
    hour = engine.calendar.hour
    prev_hour = (hour - 1) % 24
    for pid in occupants:
        p = people.get(pid)
        if p is None:
            continue
        present_by_floor[p.floor] = present_by_floor.get(p.floor, 0) + 1
        block = p.schedule.get(prev_hour)
        act = block.activity if block is not None else "other"
        activity_counts[act] = activity_counts.get(act, 0) + 1
    company = world.companies.get(building.company_id) if building.company_id is not None else None
    plan = floor_plan(building, company)
    rows = {r["floor"]: dict(r, assigned=0, present=present_by_floor.get(r["floor"], 0), detail="") for r in plan}
    lines: List[str] = []
    info: Dict[str, Any] = {}
    lines.append(f"{building.building_type.replace('_', ' ').title()} | {world.districts[building.district_id].name}")
    lines.append(f"Floors {building.floors} | Quality {building.quality:.2f} | Value ${building.value:,.0f}")
    lines.append(f"Owner: {_owner_label(engine, building.owner_id, building.owner_kind)}")

    if building.building_type in ("house", "apartment"):
        listings = world._building_listings.get(building.building_id, [])
        owned_units = 0
        rent_roll = 0.0
        vacant = 0
        residents = 0
        for l in listings:
            row = rows.get(l.floor)
            if row is None:
                continue
            row["assigned"] += 1
            hh = population.households.get(l.household_id) if l.household_id else None
            if hh is None:
                vacant += 1
                continue
            members = [m for m in hh.member_ids if m in people]
            residents += len(members)
            row["detail"] = f"{row['detail']}{len(members)}p "
            if l.tenure == "own":
                owned_units += 1
            else:
                rent_roll += l.rent
        for r in rows.values():
            if r["kind"] == "home":
                r["capacity"] = max(1, r["assigned"] * 3)
        rented = len(listings) - vacant - owned_units
        lines.append(f"Units {len(listings)} | rented {rented} | owner-occupied {owned_units} | vacant {vacant}")
        lines.append(f"Residents {residents} | monthly rent roll ${rent_roll:,.0f}")
        lines.append(f"Avg unit rent ${(sum(l.rent for l in listings) / max(1, len(listings))):,.0f}")
        info = {"residents": residents, "vacant": vacant, "rent_roll": rent_roll}
    elif company is not None:
        wpf = WORKERS_PER_FLOOR.get(building.building_type, 30)
        employees = [people[e] for e in company.employee_ids if e in people]
        working = on_break = off = 0
        for p in employees:
            f = p.work_floor
            row = rows.get(f) or rows.get(0)
            if row is not None:
                row["assigned"] += 1
            block = p.schedule.get(prev_hour)
            act = block.activity if block is not None else ""
            if act == "work":
                working += 1
            elif act == "break":
                on_break += 1
            else:
                off += 1
        for r in rows.values():
            if r["kind"] == "work" and r["assigned"] == 0:
                r["detail"] = "vacant / leasable"
        lines.append(f"{company.name} ({company.sector}, {company.company_scale})")
        lines.append(f"Market value ${company.market_value:,.0f} | cash ${company.cash:,.0f}")
        lines.append(f"Payroll ${company.payroll_budget:,.0f}/mo | health {company.health_score():.2f}")
        lines.append(f"Staff {len(employees)}/{company.target_headcount} | working {working} | break {on_break} | off-site {off}")
        info = {"staff": len(employees), "working": working, "break": on_break, "off": off}
    else:
        lines.append(f"Capacity {building.capacity} | service score {building.service_score:.2f}")
        lines.append(f"Open {building.open_hours[0]:02d}:00-{building.open_hours[1]:02d}:00")
    total_present = sum(present_by_floor.values())
    outside = present_by_floor.get(-1, 0)
    lines.append(f"People on tile now {total_present} (outside/entering {outside})")
    if activity_counts:
        top = sorted(activity_counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
        lines.append("Activity: " + ", ".join(f"{a} {n}" for a, n in top))
    return {
        "id": building.building_id,
        "name": building.name,
        "type": building.building_type,
        "floors": building.floors,
        "lines": lines,
        "rows": [rows[f] for f in sorted(rows)],
        "outside": outside,
        "info": info,
    }
