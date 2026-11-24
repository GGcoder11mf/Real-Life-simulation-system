"""
systems.py

Interacting simulation systems live here. The goal is not to provide
isolated minigames, but to let law, economy, social reputation,
education, and finance reinforce one another over long simulation runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import buildings as bldg
from entities import Person, Population
from world import Company, WorldMap, clamp, coerce_rng_state


@dataclass
class CrimeCase:
    case_id: int
    suspect_id: int
    district_id: int
    severity: float
    crime_type: str
    evidence: float
    witnesses: List[int] = field(default_factory=list)
    resolved: bool = False
    verdict: Optional[str] = None
    sentence_days: int = 0
    opened_day: int = 0
    resolved_day: int = 0


@dataclass
class CourtDocketItem:
    docket_id: int
    case_id: int
    hearing_day: int
    judge_bias: float
    public_attention: float


@dataclass
class TaxLedger:
    person_id: int
    annual_income: float = 0.0
    taxes_withheld: float = 0.0
    debt_to_state: float = 0.0
    refunds_due: float = 0.0


@dataclass
class EducationProgram:
    program_id: int
    building_id: int
    name: str
    program_type: str
    skill_key: str
    tuition: float
    duration_days: int
    seats: int
    enrolled_ids: List[int] = field(default_factory=list)


@dataclass
class Stock:
    symbol: str
    company_id: int
    price: float
    volatility: float
    dividend_yield: float
    last_change: float = 0.0


class ReputationSystem:
    """Public reputation spreads indirectly through social contact."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed + 701)
        self.gossip_feed: List[str] = []

    def tick(self, population: Population, day_index: int) -> List[str]:
        stories: List[str] = []
        for person in population.people.values():
            rumor = person.spread_gossip()
            if rumor and self.rng.random() < 0.25:
                stories.append(rumor)
            delta = 0.0
            if person.stress < 0.35 and person.hidden_karma > 0.55:
                delta += 0.005
            if person.arrest_record:
                delta -= 0.008
            if person.debt > 25_000:
                delta -= 0.003
            person.reputation_public = clamp(person.reputation_public + delta, 0.0, 1.0)
        self.gossip_feed.extend(stories)
        self.gossip_feed = self.gossip_feed[-60:]
        return stories

    def propagate_social_event(self, population: Population, source_id: int, text: str, impact: float) -> None:
        if source_id not in population.people:
            return
        source = population.people[source_id]
        source.gossip_log.append(text)
        source.gossip_log = source.gossip_log[-15:]
        for neighbor_id in source.social_circle[:12]:
            if neighbor_id not in population.people:
                continue
            rel = population.people[neighbor_id].relationship_with(source_id)
            rel.gossip_knowledge = clamp(rel.gossip_knowledge + abs(impact) * 0.25, 0.0, 1.0)
            if impact > 0:
                population.people[neighbor_id].reputation_public = clamp(population.people[neighbor_id].reputation_public + 0.002, 0.0, 1.0)


class FinanceSystem:
    """Handles taxes, debt, rent pressure, and personal finance drift."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed + 702)
        self.tax_ledgers: Dict[int, TaxLedger] = {}
        self.tax_collected_today = 0.0

    def ledger_for(self, person_id: int) -> TaxLedger:
        if person_id not in self.tax_ledgers:
            self.tax_ledgers[person_id] = TaxLedger(person_id=person_id)
        return self.tax_ledgers[person_id]

    def process_daily_finances(self, population: Population, day_index: int) -> List[str]:
        stories: List[str] = []
        if day_index % 30 == 0 and day_index > 0:
            population.pay_household_costs()
            stories.append("Monthly housing and utility costs were charged across all households.")
        collected = 0.0
        for person in population.people.values():
            if person.job_company_id is not None:
                person.receive_pay()
                ledger = self.ledger_for(person.person_id)
                ledger.annual_income += person.salary / 30.0
                withheld = person.salary / 30.0 * 0.12
                ledger.taxes_withheld += withheld
                person.bank_balance = max(0.0, person.bank_balance - withheld)
                collected += withheld
            elif person.pension > 0.0:
                person.receive_pay()
            if person.debt > 0:
                interest = person.debt * 0.0006
                person.debt += interest
                if person.debt > 12_000:
                    person.stress = clamp(person.stress + 0.004, 0.0, 1.0)
            if person.bank_balance > 1000:
                savings_gain = person.bank_balance * 0.00008
                person.bank_balance += savings_gain
        self.tax_collected_today = collected
        return stories

    def annual_tax_day(self, population: Population) -> List[str]:
        notices: List[str] = []
        for person in population.people.values():
            ledger = self.ledger_for(person.person_id)
            owed = max(0.0, ledger.annual_income * 0.16 - ledger.taxes_withheld)
            refund = max(0.0, ledger.taxes_withheld - ledger.annual_income * 0.14)
            ledger.debt_to_state = owed
            ledger.refunds_due = refund
            if owed > 0:
                person.debt += owed
                person.add_memory("finance", f"Tax filing created an additional liability of ${owed:.0f}.", -0.05)
            if refund > 0:
                person.bank_balance += refund
                person.add_memory("finance", f"Received a tax refund of ${refund:.0f}.", 0.05)
            notices.append(f"{person.full_name} tax settled: owed ${owed:.0f}, refund ${refund:.0f}.")
            self.tax_ledgers[person.person_id] = TaxLedger(person_id=person.person_id)
        return notices


class EducationSystem:
    """Education, skill growth, tuition, and credentialing."""

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.rng = random.Random(seed + 703)
        self.world = world
        self.programs: Dict[int, EducationProgram] = {}
        self._program_counter = 1
        self._enrolled_since: Dict[int, int] = {}
        self._generate_programs()

    def _generate_programs(self) -> None:
        school_buildings = self.world.building_by_type("school")
        college_buildings = self.world.building_by_type("college")
        for building in school_buildings:
            self._add_program(building.building_id, f"{building.name} General Schooling", "school", "communication", 120.0, 180, building.capacity)
        for building in college_buildings:
            for name, skill, tuition in [
                ("Software Diploma", "software", 1200.0),
                ("Business Administration", "finance", 1400.0),
                ("Technical Operations", "engineering", 1100.0),
                ("Public Safety Academy", "law", 900.0),
            ]:
                self._add_program(building.building_id, name, "college", skill, tuition, 240, building.capacity // 2)

    def _add_program(
        self,
        building_id: int,
        name: str,
        program_type: str,
        skill_key: str,
        tuition: float,
        duration_days: int,
        seats: int,
    ) -> None:
        program = EducationProgram(
            program_id=self._program_counter,
            building_id=building_id,
            name=name,
            program_type=program_type,
            skill_key=skill_key,
            tuition=tuition,
            duration_days=duration_days,
            seats=seats,
        )
        self.programs[program.program_id] = program
        self._program_counter += 1

    def enroll_candidates(self, population: Population, day_index: int) -> List[str]:
        news: List[str] = []
        candidates = [person for person in population.people.values() if person.student_status is None and 18 <= person.age <= 34]
        self.rng.shuffle(candidates)
        for person in candidates[: max(8, len(candidates) // 18)]:
            if person.job_company_id is not None and person.motivation < 0.65:
                continue
            available = [
                program for program in self.programs.values()
                if len(program.enrolled_ids) < program.seats and person.bank_balance + person.cash > program.tuition * 0.4
            ]
            if not available:
                continue
            program = self.rng.choice(available)
            program.enrolled_ids.append(person.person_id)
            self._enrolled_since[person.person_id] = day_index
            person.enroll(program.program_type, program.building_id, program.tuition * 0.2)
            news.append(f"{person.full_name} enrolled in {program.name}.")
        return news

    def tick(self, population: Population, day_index: int) -> List[str]:
        results: List[str] = []
        for program in self.programs.values():
            for person_id in list(program.enrolled_ids):
                if person_id not in population.people:
                    program.enrolled_ids.remove(person_id)
                    self._enrolled_since.pop(person_id, None)
                    continue
                person = population.people[person_id]
                person.skills[program.skill_key] = clamp(person.skills[program.skill_key] + 0.0012, 0.0, 1.0)
                person.skill_practice[program.skill_key] = min(4.0, person.skill_practice.get(program.skill_key, 0.0) + 0.8)
                enrolled_day = self._enrolled_since.get(person_id, day_index)
                elapsed = day_index - enrolled_day
                if elapsed >= program.duration_days:
                    if program.program_type == "college":
                        person.graduate_college(program.skill_key)
                    elif program.program_type == "school":
                        person.education_level = "school"
                        person.student_status = None
                        person.current_enrollment_building = None
                        person.add_memory("education", "Completed a formal schooling block.", 0.06)
                    else:
                        person.complete_certification(program.name, program.skill_key)
                    results.append(f"{person.full_name} completed {program.name}.")
                    program.enrolled_ids.remove(person_id)
                    self._enrolled_since.pop(person_id, None)
        return results


class CompanySystem:
    """Hiring, layoffs, promotions, bankruptcy pressure, and business drift."""

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.rng = random.Random(seed + 704)
        self.world = world

    def run_hiring(self, population: Population) -> List[str]:
        events: List[str] = []
        unemployed = population.unemployed_people()
        self.rng.shuffle(unemployed)
        open_postings = self.world.open_job_postings()
        window = open_postings[:120]
        dirty = False
        for person in unemployed[: max(10, len(unemployed) // 3)]:
            if dirty:
                market = self.world.job_market
                open_postings = [post for post in open_postings if post.posting_id in market]
                window = open_postings[:120]
                dirty = False
            posting_id = person.seek_job(self.world, window)
            if posting_id is None or posting_id not in self.world.job_market:
                continue
            dirty = True
            posting = self.world.job_market[posting_id]
            company = self.world.companies.get(posting.company_id)
            if company is None:
                del self.world.job_market[posting_id]
                continue
            if len(company.employee_ids) >= company.target_headcount:
                del self.world.job_market[posting_id]
                if posting.title in company.open_positions:
                    company.open_positions.remove(posting.title)
                continue
            person.assign_job_profile(company.company_id, posting.title, posting.salary, posting.shift, posting.required_skill)
            company.employee_ids.append(person.person_id)
            site = self.world.buildings.get(company.building_id)
            if site is not None:
                person.work_floor = bldg.floor_for_rank(site.building_type, site.floors, len(company.employee_ids) - 1)
            company.payroll_budget += posting.salary / 12.0
            company.productivity = clamp(company.productivity + 0.005, 0.0, 1.0)
            events.append(f"{person.full_name} was hired by {company.name} as {posting.title}.")
            del self.world.job_market[posting_id]
            if posting.title in company.open_positions:
                company.open_positions.remove(posting.title)
        return events

    CONSUMER_SECTORS = {"retail": 3.0, "food": 3.0, "healthcare": 1.5, "media": 1.0, "education": 0.8, "finance": 0.8, "transport": 0.8}

    def receive_consumer_spending(self, amount: float) -> float:
        """Distribute household spending to consumer-facing firms by size; returns the sales tax."""
        tax = amount * 0.06
        pool = amount - tax
        weights = {
            c.company_id: len(c.employee_ids) * self.CONSUMER_SECTORS.get(c.sector, 0.0)
            for c in self.world.companies.values()
        }
        total = sum(weights.values())
        if total <= 0 or pool <= 0:
            return tax
        for company in self.world.companies.values():
            w = weights[company.company_id]
            if w > 0:
                company.cash += pool * w / total
        return tax

    def vacancy_target(self, company: Company) -> int:
        """How many openings a company normally carries.

        Proportional to its staff and shaped by demand, expansion appetite and
        productivity, so large or growing firms advertise several roles while
        small or shrinking ones often advertise none.  The fractional part is a
        stable per-company offset, so targets don't flicker from day to day."""
        staff = len(company.employee_ids)
        if staff == 0:
            return 0
        rate = clamp(
            0.028
            + (company.demand - company.supply) * 0.05
            + company.expansion_desire * 0.035
            + (0.015 if company.productivity > 0.7 else 0.0)
            - company.bankruptcy_risk * 0.03,
            0.0,
            0.14,
        )
        expected = staff * rate
        frac = (company.company_id * 0.6180339887) % 1.0
        target = int(expected + frac)
        return max(0, min(target, company.max_headcount - staff))

    def seed_baseline_vacancies(self, target_vacancies: Optional[int] = None) -> List[str]:
        """Top each company up to its own vacancy target (not one posting per company)."""
        events: List[str] = []
        world = self.world
        open_counts: Dict[int, int] = {}
        for posting in world.job_market.values():
            open_counts[posting.company_id] = open_counts.get(posting.company_id, 0) + 1
        companies = [c for c in world.companies.values() if c.employee_ids]
        self.rng.shuffle(companies)
        for company in companies:
            if target_vacancies is not None and len(world.job_market) >= target_vacancies:
                break
            missing = self.vacancy_target(company) - open_counts.get(company.company_id, 0)
            for _ in range(max(0, missing)):
                posting = world.add_job_posting(
                    company,
                    title=self._title_for_sector(company.sector),
                    required_skill=self._skill_for_sector(company.sector),
                    salary=round(self.rng.uniform(2200, 9200), 2),
                    required_level=self.rng.uniform(0.1, 0.6),
                    shift=self.rng.choice(company.shift_templates or [(9, 17)]),
                    urgency=self.rng.uniform(0.2, 0.8),
                )
                events.append(f"{company.name} posted a vacancy for {posting.title}.")
        return events

    def run_attrition(self, population: Population) -> List[str]:
        events: List[str] = []
        for company in list(self.world.companies.values()):
            for person_id in list(company.employee_ids):
                person = population.people.get(person_id)
                if person is None:
                    company.employee_ids.remove(person_id)
                    continue
                if person.age >= 67:
                    person.pension = max(person.pension, person.salary * 0.45)
                    self._release_employee(company, person)
                    person.add_memory("career", "Retired and began drawing a pension.", 0.05)
                    events.append(f"{person.full_name} retired from {company.name}.")
                    continue
                quit_chance = 0.0006 + max(0.0, 0.55 - person.job_satisfaction) * 0.0022
                if self.rng.random() >= quit_chance:
                    continue
                self._release_employee(company, person)
                events.append(f"{person.full_name} left {company.name} voluntarily.")
        return events

    def run_promotions(self, population: Population) -> List[str]:
        events: List[str] = []
        self._refresh_company_productivity(population)
        for person in population.employed_people():
            if person.job_company_id not in self.world.companies:
                continue
            company = self.world.companies[person.job_company_id]
            performance = self._employee_performance_score(person, company)
            if performance > 0.78 and self.rng.random() < 0.02:
                raise_amount = person.salary * 0.08
                person.salary += raise_amount
                company.payroll_budget += raise_amount / 12.0
                person.job_performance = clamp(person.job_performance + 0.03, 0.0, 1.0)
                person.job_satisfaction = clamp(person.job_satisfaction + 0.05, 0.0, 1.0)
                person.add_memory("career", f"Received a promotion-level raise at {company.name}.", 0.1)
                events.append(f"{person.full_name} received a raise at {company.name}.")
        return events

    def run_layoffs_and_bankruptcy(self, population: Population) -> List[str]:
        events: List[str] = []
        for company in list(self.world.companies.values()):
            employee_pool = [population.people[pid] for pid in company.employee_ids if pid in population.people]
            if employee_pool:
                low_performers = sorted(employee_pool, key=lambda person: self._employee_performance_score(person, company))
                fire_count = 0
                if company.cash < 0 and company.health_score() < 0.36:
                    fire_count = min(max(1, len(low_performers) // 10), len(low_performers))
                elif company.cash < 0 and company.health_score() < 0.46 and self.rng.random() < 0.35:
                    fire_count = min(2, len(low_performers))
                for person in low_performers[:fire_count]:
                    if person.person_id in company.employee_ids:
                        self._release_employee(company, person)
                        events.append(f"{person.full_name} was fired by {company.name} for weak performance.")
            if company.layoffs_pending > 0 and company.cash < 0 and company.employee_ids:
                ranked_ids = sorted(
                    [pid for pid in company.employee_ids if pid in population.people],
                    key=lambda pid: self._employee_performance_score(population.people[pid], company),
                )
                layoffs = min(company.layoffs_pending, len(ranked_ids))
                for person_id in ranked_ids[:layoffs]:
                    person = population.people.get(person_id)
                    if person is not None:
                        self._release_employee(company, person)
                        events.append(f"{person.full_name} was laid off from {company.name}.")
                company.layoffs_pending = 0
            if company.bankruptcy_risk > 0.92 and company.cash < 0:
                for person_id in list(company.employee_ids):
                    person = population.people.get(person_id)
                    if person is not None:
                        self._release_employee(company, person)
                events.append(f"{company.name} declared bankruptcy.")
                self.world.remove_company(company.company_id)
                continue
            if not company.employee_ids and not self.world.company_has_live_openings(company.company_id):
                self.world.remove_company(company.company_id)
                events.append(f"{company.name} shut down after losing all employees.")
        return events

    def _release_employee(self, company: Company, person: Person) -> None:
        if person.person_id in company.employee_ids:
            company.employee_ids.remove(person.person_id)
        company.payroll_budget = max(0.0, company.payroll_budget - person.salary / 12.0)
        person.remove_job()

    def _employee_performance_score(self, person: Person, company: Company) -> float:
        return clamp(
            person.job_performance * 0.36
            + person.work_traits["work_ethic"] * 0.16
            + person.work_traits["reliability"] * 0.14
            + person.work_traits["teamwork"] * 0.08
            + person.happiness * 0.08
            + person.motivation * 0.08
            + company.productivity * 0.1
            - min(0.25, person.days_unproductive * 0.01),
            0.0,
            1.0,
        )

    def _refresh_company_productivity(self, population: Population) -> None:
        for company in self.world.companies.values():
            employees = [population.people[pid] for pid in company.employee_ids if pid in population.people]
            if not employees:
                company.productivity = clamp(company.productivity - 0.03, 0.0, 1.0)
                continue
            workforce_score = sum(self._employee_performance_score(person, company) for person in employees) / len(employees)
            company.productivity = clamp(company.productivity * 0.7 + workforce_score * 0.3, 0.0, 1.0)

    def spawn_hiring_waves(self, force: bool = False) -> List[str]:
        events: List[str] = []
        for company in self.world.companies.values():
            pressure = company.demand - company.supply + company.expansion_desire * 0.25
            mandatory_staffing = company.below_minimum_staffing()
            should_hire = force or mandatory_staffing or (
                self.rng.random() < 0.2
                and (pressure > 0.08 or company.expansion_desire > 0.55)
            )
            if should_hire:
                hiring_gap = company.staffing_gap()
                if hiring_gap <= 0:
                    continue
                if company.company_scale == "industrial":
                    openings = min(hiring_gap, self.rng.randint(8, 20))
                elif company.company_scale == "indie":
                    openings = min(hiring_gap, self.rng.randint(1, 2))
                else:
                    openings = min(hiring_gap, self.rng.randint(3, 8))
                skill = self._skill_for_sector(company.sector)
                title = self._title_for_sector(company.sector)
                for _ in range(openings):
                    self.world.add_job_posting(
                        company,
                        title=title,
                        required_skill=skill,
                        salary=round(self.rng.uniform(2200, 9200), 2),
                        required_level=self.rng.uniform(0.15, 0.75),
                        shift=self.rng.choice(company.shift_templates or [(9, 17)]),
                        urgency=self.rng.uniform(0.4, 1.0),
                    )
                events.append(f"{company.name} opened a hiring wave for {openings} positions.")
        return events

    def _skill_for_sector(self, sector: str) -> str:
        mapping = {
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
        return mapping.get(sector, "service")

    def _title_for_sector(self, sector: str) -> str:
        table = {
            "retail": ["Cashier", "Buyer", "Lead"],
            "logistics": ["Dispatcher", "Driver", "Planner"],
            "manufacturing": ["Operator", "Assembler", "Technician"],
            "software": ["Developer", "Analyst", "Operator"],
            "healthcare": ["Nurse", "Technician", "Clerk"],
            "education": ["Teacher", "Advisor", "Coordinator"],
            "food": ["Cook", "Server", "Manager"],
            "transport": ["Conductor", "Mechanic", "Planner"],
            "construction": ["Estimator", "Supervisor", "Laborer"],
            "finance": ["Advisor", "Analyst", "Clerk"],
            "media": ["Writer", "Editor", "Producer"],
            "public_service": ["Officer", "Inspector", "Caseworker"],
        }
        return self.rng.choice(table.get(sector, ["Worker"]))


class CrimeSystem:
    """Crime is driven by pressure, location, and personality."""

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.rng = random.Random(seed + 705)
        self.world = world
        self.cases: Dict[int, CrimeCase] = {}
        self._case_counter = 1
        self.safety_multiplier = 1.0

    def tick(self, population: Population, day_index: int) -> List[CrimeCase]:
        opened: List[CrimeCase] = []
        for person in population.people.values():
            if person.incarcerated:
                continue
            tile = self.world.tile_at(*person.location)
            pressure = person.crime_tendency * 0.45 + tile.crime_pressure * 0.35 + max(0.0, person.stress - 0.55) * 0.2
            if person.cash + person.bank_balance < 150 and pressure > 0.45:
                pressure += 0.08
            if self.rng.random() < pressure * 0.008 / self.safety_multiplier:
                crime_type = self.rng.choice(["theft", "fraud", "assault", "vandalism"])
                case = CrimeCase(
                    case_id=self._case_counter,
                    suspect_id=person.person_id,
                    district_id=tile.district_id,
                    severity=clamp(pressure + self.rng.uniform(-0.15, 0.2), 0.1, 1.0),
                    crime_type=crime_type,
                    evidence=self.rng.uniform(0.1, 1.0),
                    witnesses=[pid for pid in tile.occupants if pid != person.person_id][:5],
                    opened_day=day_index,
                )
                self.cases[case.case_id] = case
                self._case_counter += 1
                person.hidden_karma = clamp(person.hidden_karma - 0.06, 0.0, 1.0)
                person.gossip_log.append(f"Someone whispered that {person.full_name} was involved in {crime_type}.")
                opened.append(case)
        return opened


class LawSystem:
    """Police, courts, and prison interact with crime cases and reputation."""

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.rng = random.Random(seed + 706)
        self.world = world
        self.docket: Dict[int, CourtDocketItem] = {}
        self._docket_counter = 1
        self.inmates: Dict[int, int] = {}
        self.safety_multiplier = 1.0

    def investigate(self, population: Population, crime_system: CrimeSystem, day_index: int) -> List[str]:
        reports: List[str] = []
        for case in crime_system.cases.values():
            if case.resolved:
                continue
            suspect = population.people.get(case.suspect_id)
            if suspect is None or suspect.incarcerated:
                continue
            detection_score = case.evidence * 0.45 + len(case.witnesses) * 0.08 + (1.0 - suspect.reputation_public) * 0.08
            if self.rng.random() < detection_score * 0.3 * self.safety_multiplier:
                docket = CourtDocketItem(
                    docket_id=self._docket_counter,
                    case_id=case.case_id,
                    hearing_day=day_index + self.rng.randint(1, 10),
                    judge_bias=self.rng.uniform(-0.2, 0.2),
                    public_attention=clamp(case.severity + len(case.witnesses) * 0.05, 0.0, 1.0),
                )
                self.docket[docket.docket_id] = docket
                self._docket_counter += 1
                reports.append(f"Police referred case {case.case_id} ({case.crime_type}) to court.")
        return reports

    def resolve_courts(self, population: Population, crime_system: CrimeSystem, day_index: int) -> List[str]:
        results: List[str] = []
        for docket_id, docket in list(self.docket.items()):
            if day_index < docket.hearing_day:
                continue
            case = crime_system.cases.get(docket.case_id)
            if case is None or case.resolved:
                del self.docket[docket_id]
                continue
            suspect = population.people.get(case.suspect_id)
            if suspect is None:
                del self.docket[docket_id]
                continue
            conviction = case.evidence * 0.55 + case.severity * 0.3 + docket.public_attention * 0.1 - suspect.reputation_public * 0.1 + docket.judge_bias
            if conviction > 0.55:
                sentence = int(2 + case.severity * 30)
                case.verdict = "guilty"
                case.sentence_days = sentence
                self._incarcerate(suspect)
                self.inmates[suspect.person_id] = sentence
                suspect.arrest_record.append(f"{case.crime_type}:{sentence}d")
                suspect.reputation_public = clamp(suspect.reputation_public - 0.12, 0.0, 1.0)
                suspect.stress = clamp(suspect.stress + 0.12, 0.0, 1.0)
                suspect.add_memory("law", f"Convicted for {case.crime_type}.", -0.18)
                results.append(f"{suspect.full_name} was convicted for {case.crime_type} and sentenced to {sentence} days.")
            else:
                case.verdict = "not_guilty"
                suspect.add_memory("law", f"A court cleared them in a {case.crime_type} case.", 0.02)
                results.append(f"{suspect.full_name} was cleared in court for {case.crime_type}.")
            case.resolved = True
            case.resolved_day = day_index
            del self.docket[docket_id]
        self._tick_prison(population)
        self._prune_cases(crime_system, day_index)
        return results

    def _incarcerate(self, person: Person) -> None:
        person.incarcerated = True
        if person.job_company_id is not None:
            company = self.world.companies.get(person.job_company_id)
            if company is not None and person.person_id in company.employee_ids:
                company.employee_ids.remove(person.person_id)
                company.payroll_budget = max(0.0, company.payroll_budget - person.salary / 12.0)
            person.remove_job()
        person.location = self._prison_tile()
        person.destination = person.location
        person.path = []
        self.world.register_population_position(person.person_id, *person.location)

    def _prune_cases(self, crime_system: CrimeSystem, day_index: int) -> None:
        for case_id, case in list(crime_system.cases.items()):
            if case.resolved and day_index - case.resolved_day > 60:
                del crime_system.cases[case_id]
            elif day_index - case.opened_day > 180:
                del crime_system.cases[case_id]

    def _tick_prison(self, population: Population) -> None:
        for person_id in list(self.inmates.keys()):
            person = population.people.get(person_id)
            if person is None:
                del self.inmates[person_id]
                continue
            self.inmates[person_id] -= 1
            person.location = self._prison_tile()
            person.destination = person.location
            person.path = []
            self.world.register_population_position(person.person_id, *person.location)
            person.happiness = clamp(person.happiness - 0.01, 0.0, 1.0)
            if self.inmates[person_id] <= 0:
                del self.inmates[person_id]
                person.incarcerated = False

    def _prison_tile(self) -> Tuple[int, int]:
        prisons = self.world.building_by_type("prison")
        if prisons:
            return prisons[0].tile
        station = self.world.building_by_type("station")
        return station[0].tile if station else (self.world.width // 2, self.world.height // 2)

    def background_check_score(self, person: Person) -> float:
        penalty = 0.25 if person.arrest_record else 0.0
        debt_penalty = clamp(person.debt / 50_000.0, 0.0, 0.2)
        return clamp(person.reputation_public - penalty - debt_penalty, 0.0, 1.0)


class StockMarketSystem:
    """Simplified equities derived from company performance."""

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.rng = random.Random(seed + 707)
        self.world = world
        self.stocks: Dict[str, Stock] = {}
        self._bootstrap()

    def _bootstrap(self) -> None:
        for company in self.world.companies.values():
            self.stocks[company.stock_symbol] = Stock(
                symbol=company.stock_symbol,
                company_id=company.company_id,
                price=round(max(8.0, company.market_value / 120_000.0), 2),
                volatility=self.rng.uniform(0.01, 0.08),
                dividend_yield=self.rng.uniform(0.0, 0.04),
            )

    def tick(self, population: Population) -> List[str]:
        events: List[str] = []
        for symbol, stock in list(self.stocks.items()):
            if stock.company_id not in self.world.companies:
                del self.stocks[symbol]
                continue
            company = self.world.companies[stock.company_id]
            change = (company.health_score() - 0.5) * 0.08 + self.rng.uniform(-stock.volatility, stock.volatility)
            stock.last_change = change
            stock.price = round(max(1.0, stock.price * (1.0 + change)), 2)
            if abs(change) > 0.06:
                direction = "surged" if change > 0 else "fell"
                events.append(f"{symbol} {direction} to {stock.price:.2f}.")
        for company in self.world.companies.values():
            if company.stock_symbol not in self.stocks:
                self.stocks[company.stock_symbol] = Stock(
                    symbol=company.stock_symbol,
                    company_id=company.company_id,
                    price=round(max(8.0, company.market_value / 120_000.0), 2),
                    volatility=self.rng.uniform(0.01, 0.08),
                    dividend_yield=self.rng.uniform(0.0, 0.04),
                )
        self._apply_investor_behavior(population)
        return events

    def _apply_investor_behavior(self, population: Population) -> None:
        symbols = list(self.stocks.keys())
        for person in population.people.values():
            pick = person.choose_investment_symbol(symbols)
            if not pick or pick not in self.stocks:
                continue
            stock = self.stocks[pick]
            if person.bank_balance > stock.price * 2 and self.rng.random() < 0.02:
                spend = stock.price * self.rng.randint(1, 4)
                person.bank_balance -= spend
                person.investment_accounts[pick] = person.investment_accounts.get(pick, 0.0) + spend / stock.price
            if pick in person.investment_accounts and self.rng.random() < 0.015:
                shares = person.investment_accounts[pick]
                if shares > 0:
                    proceeds = min(shares, 2) * stock.price
                    person.investment_accounts[pick] = max(0.0, shares - 2)
                    person.bank_balance += proceeds

    def market_lines(self, limit: int = 8) -> List[str]:
        stocks = sorted(self.stocks.values(), key=lambda stock: abs(stock.last_change), reverse=True)
        return [
            f"{stock.symbol:4} {stock.price:7.2f} {stock.last_change:+0.03f}"
            for stock in stocks[:limit]
        ]


class SocialEventSystem:
    """Creates positive and negative social events from combined state."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed + 708)

    def tick(self, population: Population, world: WorldMap, day_index: int) -> List[str]:
        events: List[str] = []
        births = population.run_social_layer(day_index)
        for birth in births:
            child = population.create_child(birth["new_person_id"], birth["parents"])
            parent_names = ", ".join(population.people[parent_id].full_name for parent_id in birth["parents"] if parent_id in population.people)
            events.append(f"A child, {child.full_name}, was born to {parent_names}.")
        for person in population.people.values():
            if person.stress > 0.82 and person.personality.sociability > 0.45 and self.rng.random() < 0.012:
                events.append(f"{person.full_name} had a public meltdown in {world.districts[world.tile_at(*person.location).district_id].name}.")
                person.reputation_public = clamp(person.reputation_public - 0.04, 0.0, 1.0)
                person.add_memory("social", "Had a visible emotional breakdown in public.", -0.08)
            if person.happiness > 0.82 and person.social_circle and self.rng.random() < 0.01:
                partner_id = self.rng.choice(person.social_circle)
                if partner_id in population.people:
                    events.append(f"{person.full_name} hosted an upbeat social gathering with {population.people[partner_id].full_name}.")
                    person.add_memory("social", "Hosted a good social gathering.", 0.06)
        return events


class SimulationSystems:
    """Orchestrates all interacting world systems."""

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.seed = seed
        self.world = world
        self.reputation = ReputationSystem(seed)
        self.finance = FinanceSystem(seed)
        self.education = EducationSystem(seed, world)
        self.company = CompanySystem(seed, world)
        self.crime = CrimeSystem(seed, world)
        self.law = LawSystem(seed, world)
        self.stock_market = StockMarketSystem(seed, world)
        self.social = SocialEventSystem(seed)

    def tick_daily(self, population: Population, day_index: int) -> Dict[str, List[str]]:
        report: Dict[str, List[str]] = {
            "finance": self.finance.process_daily_finances(population, day_index),
            "education": self.education.tick(population, day_index),
            "enrollments": self.education.enroll_candidates(population, day_index),
            "hiring": self.company.run_hiring(population),
            "promotions": self.company.run_promotions(population),
            "hiring_waves": self.company.spawn_hiring_waves(),
            "attrition": self.company.run_attrition(population),
            "layoffs": self.company.run_layoffs_and_bankruptcy(population),
            "reputation": self.reputation.tick(population, day_index),
            "stock_market": self.stock_market.tick(population),
            "social": self.social.tick(population, self.world, day_index),
            "police": [],
            "courts": [],
            "crime": [],
            "taxes": [],
        }
        crime_cases = self.crime.tick(population, day_index)
        report["crime"] = [
            f"Crime reported: {population.people[case.suspect_id].full_name} linked to {case.crime_type}."
            for case in crime_cases
            if case.suspect_id in population.people
        ]
        report["police"] = self.law.investigate(population, self.crime, day_index)
        report["courts"] = self.law.resolve_courts(population, self.crime, day_index)
        if day_index > 0 and day_index % 365 == 0:
            report["taxes"] = self.finance.annual_tax_day(population)
        return report

    def background_check(self, person: Person) -> float:
        return self.law.background_check_score(person)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finance_ledgers": {pid: ledger.__dict__ for pid, ledger in self.finance.tax_ledgers.items()},
            "education_programs": {pid: program.__dict__ for pid, program in self.education.programs.items()},
            "education_enrolled_since": {str(k): v for k, v in self.education._enrolled_since.items()},
            "crime_cases": {cid: case.__dict__ for cid, case in self.crime.cases.items()},
            "crime_case_counter": self.crime._case_counter,
            "docket": {did: item.__dict__ for did, item in self.law.docket.items()},
            "docket_counter": self.law._docket_counter,
            "inmates": self.law.inmates,
            "stocks": {symbol: stock.__dict__ for symbol, stock in self.stock_market.stocks.items()},
            "gossip_feed": self.reputation.gossip_feed,
            "rng_states": {
                "reputation": self.reputation.rng.getstate(),
                "finance": self.finance.rng.getstate(),
                "education": self.education.rng.getstate(),
                "company": self.company.rng.getstate(),
                "crime": self.crime.rng.getstate(),
                "law": self.law.rng.getstate(),
                "stock_market": self.stock_market.rng.getstate(),
                "social": self.social.rng.getstate(),
            },
        }

    @classmethod
    def from_dict(cls, seed: int, world: WorldMap, data: Dict[str, Any]) -> "SimulationSystems":
        systems = cls(seed, world)
        systems.finance.tax_ledgers = {int(pid): TaxLedger(**ledger) for pid, ledger in data["finance_ledgers"].items()}
        systems.education.programs = {int(pid): EducationProgram(**program) for pid, program in data["education_programs"].items()}
        systems.education._enrolled_since = {int(k): v for k, v in data.get("education_enrolled_since", {}).items()}
        systems.crime.cases = {int(cid): CrimeCase(**case) for cid, case in data["crime_cases"].items()}
        systems.crime._case_counter = data["crime_case_counter"]
        systems.law.docket = {int(did): CourtDocketItem(**item) for did, item in data["docket"].items()}
        systems.law._docket_counter = data["docket_counter"]
        systems.law.inmates = {int(pid): days for pid, days in data["inmates"].items()}
        systems.stock_market.stocks = {symbol: Stock(**stock) for symbol, stock in data["stocks"].items()}
        systems.reputation.gossip_feed = list(data["gossip_feed"])
        rng_states = data.get("rng_states", {})
        for name, system in {
            "reputation": systems.reputation,
            "finance": systems.finance,
            "education": systems.education,
            "company": systems.company,
            "crime": systems.crime,
            "law": systems.law,
            "stock_market": systems.stock_market,
            "social": systems.social,
        }.items():
            state = coerce_rng_state(rng_states.get(name))
            if state is not None:
                system.rng.setstate(state)
        return systems
