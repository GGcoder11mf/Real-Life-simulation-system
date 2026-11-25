"""
realism.py

Extra realism factors that close loops the older systems left open:

    * Public holidays      - workplaces and schools close, people stay home or go out.
    * Migration            - people arrive when jobs and housing are plentiful and
                             leave when unemployed, broke or priced out.
    * Business formation   - ambitious people with savings found small companies in
                             vacant commercial buildings (the counterpart of bankruptcy).
    * Wage indexation      - salaries catch up with inflation once a year.
    * Nutrition            - food access affects healing and health (see life.py).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from world import clamp, coerce_rng_state

HOLIDAYS = {(1, 1): "New Year's Day", (5, 1): "Labour Day", (7, 4): "Founders' Day",
            (10, 31): "Harvest Night", (12, 25): "Midwinter Day", (12, 26): "Midwinter Holiday"}


def holiday_name(calendar: Any) -> str:
    return HOLIDAYS.get((calendar.month, calendar.day), "")


class RealismSystem:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed + 9500)
        self.stats: Dict[str, float] = {"immigrants": 0, "emigrants": 0, "startups": 0, "raises": 0}
        self.last_holiday = ""

    # ------------------------------------------------------------------
    def tick(self, engine: Any, day_index: int) -> List[str]:
        out: List[str] = []
        out += self._migration(engine, day_index)
        out += self._startups(engine, day_index)
        if day_index % 365 == 364:
            out += self._indexation(engine)
        return out

    # ---------------------------------------------------------- migration
    def _migration(self, engine: Any, day_index: int) -> List[str]:
        pop, world, rng = engine.population, engine.world, self.rng
        out: List[str] = []
        people = pop.people
        n = len(people)
        labor, employed, unemployed = pop.labor_stats()
        unemp = unemployed / max(1, labor)
        vacancies = world.open_job_count()
        listings = world.available_housing()
        pull = clamp(1.0 + 4.0 * (vacancies / max(1, labor) - 0.03) - 5.0 * (unemp - 0.06), 0.0, 2.5)
        housing_room = len(listings) / max(1, len(world.housing_market))
        pull *= clamp(housing_room * 4.0, 0.0, 1.2)
        expected = n * 0.00022 * pull
        arrivals = int(expected) + (1 if rng.random() < expected - int(expected) else 0)
        for _ in range(min(arrivals, len(listings))):
            listing = listings.pop(int(rng.random() * len(listings)))
            out.append(self._arrive(engine, listing))
        # departures: long-term unemployed or broke singles with no ties
        leavers: List[int] = []
        for p in people.values():
            if p.age < 20 or p.age > 60 or p.spouse_id is not None or p.children_ids or p.incarcerated or p.care_tile:
                continue
            p_leave = 0.0
            if p.job_company_id is None:
                p_leave += 0.0012
            if p.cash + p.bank_balance < 100 and p.debt > 3000:
                p_leave += 0.0015
            if p_leave and rng.random() < p_leave:
                leavers.append(p.person_id)
        if leavers:
            pop._emigrating = True
            before_deaths, before_ages = pop.deaths_total, list(pop.death_ages)
            pop.remove_people(leavers)
            pop.deaths_total, pop.death_ages = before_deaths, before_ages     # leaving is not dying
            pop._emigrating = False
            self.stats["emigrants"] += len(leavers)
            out.append(f"{len(leavers)} residents left the city to look for work elsewhere.")
        return out[:3]

    def _arrive(self, engine: Any, listing: Any) -> str:
        from entities import Household, Person
        pop, world, rng = engine.population, engine.world, self.rng
        building = world.buildings[listing.building_id]
        age = rng.randint(22, 48)
        person = Person(pop._person_counter, pop.rng, world, age=age)
        pop._person_counter += 1
        person.cash = round(rng.uniform(600, 4000), 2)
        person.bank_balance = round(rng.uniform(0, 6000), 2)
        household = Household(
            household_id=pop._household_counter, member_ids=[person.person_id], home_building_id=building.building_id,
            listing_id=listing.listing_id, savings_pool=person.bank_balance, monthly_costs=round(listing.rent * 1.15, 2))
        pop._household_counter += 1
        pop.households[household.household_id] = household
        listing.occupied = True
        listing.household_id = household.household_id
        person.assign_home(household.household_id, listing, building)
        pop.people[person.person_id] = person
        pop._pop_gen += 1
        world.register_population_position(person.person_id, *person.location)
        engine.life.register(person)
        person.add_memory("life", "Moved to the city looking for work.", 0.2)
        self.stats["immigrants"] += 1
        return f"{person.full_name} moved to the city."

    # ----------------------------------------------------------- startups
    def _startups(self, engine: Any, day_index: int) -> List[str]:
        from world import Company
        world, pop, rng = engine.world, engine.population, self.rng
        labor, employed, unemployed = pop.labor_stats()
        unemp = unemployed / max(1, labor)
        rate = clamp(0.1 - 0.7 * (unemp - 0.05), 0.02, 0.2) * len(world.companies) / 200.0   # founders per day
        n = int(rate) + (1 if rng.random() < rate - int(rate) else 0)
        if n == 0:
            return []
        sites = [b for b in world.buildings.values() if b.building_type in ("shop", "office") and b.company_id is None]
        if not sites:
            return []
        out: List[str] = []
        pool = [p for p in pop.people.values()
                if 24 <= p.age <= 58 and not p.incarcerated and p.cash + p.bank_balance > 4000
                and p.personality.ambition > 0.55 and p.spouse_id != -1 and p.person_id not in {c.owner_id for c in world.companies.values()}]
        for _ in range(n):
            if not pool or not sites:
                break
            founder = pool.pop(int(rng.random() * len(pool)))
            site = sites.pop(int(rng.random() * len(sites)))
            sector = world._sector_for_building(site.building_type)
            capital = (founder.cash + founder.bank_balance) * 0.55
            take = min(founder.cash, capital)
            founder.cash -= take
            founder.bank_balance -= capital - take
            if founder.job_company_id is not None:
                prev = world.companies.get(founder.job_company_id)
                if prev is not None and founder.person_id in prev.employee_ids:
                    prev.employee_ids.remove(founder.person_id)
                    prev.payroll_budget = max(0.0, prev.payroll_budget - founder.salary / 12.0)
                founder.remove_job()
            target = rng.randint(4, 12)
            company = Company(
                company_id=world._company_counter, name=world._company_name(sector, "indie"), sector=sector,
                district_id=site.district_id, building_id=site.building_id, market_value=capital * 3.0, cash=capital,
                payroll_budget=target * 1500.0, productivity=clamp(0.4 + founder.personality.ambition * 0.4, 0.0, 1.0),
                demand=0.6, supply=0.5, prestige=0.3, company_scale="indie", stock_symbol=world._make_stock_symbol(sector),
                shift_templates=world._default_shifts_for_sector(sector), expansion_desire=0.7, bankruptcy_risk=0.25,
                min_headcount=2, target_headcount=target, max_headcount=target + 4, owner_id=founder.person_id)
            site.company_id = company.company_id
            site.job_slots = max(site.job_slots, company.max_headcount)
            world.companies[company.company_id] = company
            district = world.districts.get(company.district_id)
            if district is not None:
                district.company_ids.append(company.company_id)
            world._company_counter += 1
            founder.assign_job_profile(company.company_id, "Founder", 3200.0, (9, 17), engine.systems.company._skill_for_sector(sector))
            company.employee_ids.append(founder.person_id)
            founder.add_memory("career", f"Founded {company.name}.", 0.5)
            self.stats["startups"] += 1
            out.append(f"{founder.full_name} founded {company.name}.")
        return out

    # --------------------------------------------------------- indexation
    def _indexation(self, engine: Any) -> List[str]:
        infl = max(0.0, float(engine.world.inflation_rate or 0.0))
        raise_pct = clamp(infl * 0.8 + 0.005, 0.0, 0.08)
        count = 0
        for p in engine.population.people.values():
            if p.job_company_id is not None and p.salary > 0:
                bump = p.salary * raise_pct * (0.6 + 0.8 * p.job_performance)
                p.salary += bump
                c = engine.world.companies.get(p.job_company_id)
                if c is not None:
                    c.payroll_budget += bump / 12.0
                count += 1
        for posting in engine.world.job_market.values():
            posting.salary = round(posting.salary * (1.0 + raise_pct), 2)
        self.stats["raises"] += count
        return [f"Annual pay review: {count:,} workers received a {raise_pct * 100:.1f}% cost-of-living adjustment."]

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "rng": self.rng.getstate()}

    @classmethod
    def from_dict(cls, seed: int, data: Dict[str, Any]) -> "RealismSystem":
        r = cls(seed)
        r.stats.update(data.get("stats", {}))
        st = coerce_rng_state(data.get("rng"))
        if st is not None:
            r.rng.setstate(st)
        return r
