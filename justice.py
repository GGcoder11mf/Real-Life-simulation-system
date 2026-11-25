"""
justice.py

A full legal pipeline replacing the old one-shot crime/court roll:

    crime committed (with a real victim and real harm)
      -> victim/witness report            (trust in police, severity)
      -> police investigation             (officers, detectives, evidence over days)
      -> arrest and custody
      -> prosecution: charge + plea offer (prosecutors, defence lawyers)
      -> trial scheduled by court capacity (judges; backlog is measured)
      -> verdict (bench), sentence: fine / probation / community service / prison
      -> prison with capacity, violence, rehabilitation, parole, recidivism

Police officers, detectives, judges, prosecutors, defenders and guards are
ordinary employees of the civic companies, so staffing is a real constraint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from world import WorldMap, clamp

CRIME_TYPES = {
    # type: (base weight, harm to victim, needs victim, wealth gain)
    "theft": (0.34, 0.0, True, 0.35),
    "robbery": (0.08, 0.45, True, 0.5),
    "assault": (0.2, 0.55, True, 0.0),
    "fraud": (0.12, 0.0, True, 0.6),
    "vandalism": (0.16, 0.0, False, 0.0),
    "arson": (0.02, 0.8, False, 0.0),
    "homicide": (0.0006, 1.0, True, 0.0),
}
ROLE_SKILL = {"Judge": "law", "Prosecutor": "law", "Public Defender": "law"}


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
    # --- legal pipeline (defaults keep old saves loadable)
    victim_id: Optional[int] = None
    status: str = "reported"      # reported|investigating|arrested|charged|trial|closed|unsolved
    reported: bool = True
    assigned_officer: Optional[int] = None
    arrest_day: int = 0
    charge: str = ""
    plea: str = ""
    judge_id: Optional[int] = None
    prosecutor_id: Optional[int] = None
    defender_id: Optional[int] = None
    private_lawyer: float = 0.0
    sentence_type: str = ""
    fine: float = 0.0
    innocent: bool = False        # ground truth for wrongful-conviction accounting
    harm: float = 0.0
    stolen: float = 0.0


@dataclass
class CourtDocketItem:
    docket_id: int
    case_id: int
    hearing_day: int
    judge_bias: float
    public_attention: float


class CrimeSystem:
    """Crime is driven by pressure, location, personality - and now has victims."""

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.rng = random.Random(seed + 705)
        self.world = world
        self.cases: Dict[int, CrimeCase] = {}
        self._case_counter = 1
        self.safety_multiplier = 1.0
        self.life: Any = None
        self.stats: Dict[str, float] = {"committed": 0, "reported": 0, "unreported": 0, "cleared": 0, "victims": 0,
                                        "stolen": 0.0, "by_type_homicide": 0}
        self.by_type: Dict[str, int] = {}
        self._pending_deaths: List[int] = []

    def tick(self, population: Any, day_index: int) -> List[CrimeCase]:
        opened: List[CrimeCase] = []
        rng = self.rng
        world = self.world
        types = list(CRIME_TYPES)
        weights = [CRIME_TYPES[t][0] for t in types]
        for person in population.people.values():
            if person.incarcerated or person.age < 14 or person.care_tile is not None:
                continue
            tile = world.tile_at(*person.location)
            pressure = person.crime_tendency * 0.45 + tile.crime_pressure * 0.35 + max(0.0, person.stress - 0.55) * 0.2
            poor = person.cash + person.bank_balance < 150
            if poor and pressure > 0.45:
                pressure += 0.08
            if len(person.arrest_record) and pressure > 0.3:
                pressure += 0.05 * min(3, len(person.arrest_record))      # recidivism
            if rng.random() >= pressure * 0.0028 / self.safety_multiplier:
                continue
            ctype = rng.choices(types, weights=[w * (2.0 if (poor and t in ("theft", "robbery", "fraud")) else 1.0)
                                                for t, w in zip(types, weights)])[0]
            _, harm, needs_victim, gain = CRIME_TYPES[ctype]
            victim = None
            if needs_victim:
                victim = self._pick_victim(population, person, tile)
                if victim is None and ctype in ("assault", "homicide", "robbery"):
                    continue
            severity = clamp(pressure + rng.uniform(-0.15, 0.2) + harm * 0.25, 0.1, 1.0)
            stolen = 0.0
            if victim is not None and gain > 0:
                stolen = min(max(0.0, victim.cash) * gain * rng.uniform(0.2, 0.9), 4000.0)
                victim.cash -= stolen
                person.cash += stolen
            elif ctype == "vandalism":
                tile.crime_pressure = clamp(tile.crime_pressure + 0.01, 0.0, 1.0)
            witnesses = [pid for pid in tile.occupants if pid != person.person_id and (victim is None or pid != victim.person_id)][:6]
            self.stats["committed"] += 1
            self.by_type[ctype] = self.by_type.get(ctype, 0) + 1
            if victim is not None:
                self.stats["victims"] += 1
                self._harm_victim(population, person, victim, ctype, severity, stolen)
                if ctype == "homicide" and victim.person_id in population.people:
                    severity = 1.0
            # Many offences are never reported: low trust, small losses, no witnesses.
            report_p = clamp(0.35 + severity * 0.5 + len(witnesses) * 0.05 - (0.15 if poor else 0.0), 0.1, 0.98)
            reported = rng.random() < report_p / max(0.7, 1.0 / self.safety_multiplier)
            if not reported:
                self.stats["unreported"] += 1
                person.hidden_karma = clamp(person.hidden_karma - 0.04, 0.0, 1.0)
                continue
            self.stats["reported"] += 1
            case = CrimeCase(
                case_id=self._case_counter,
                suspect_id=person.person_id,
                district_id=tile.district_id,
                severity=severity,
                crime_type=ctype,
                evidence=rng.uniform(0.05, 0.6) + (0.2 if witnesses else 0.0),
                witnesses=witnesses,
                opened_day=day_index,
                victim_id=victim.person_id if victim is not None else None,
                status="reported",
                harm=harm,
                stolen=stolen,
            )
            self.cases[case.case_id] = case
            self._case_counter += 1
            person.hidden_karma = clamp(person.hidden_karma - 0.06, 0.0, 1.0)
            person.gossip_log.append(f"Someone whispered that {person.full_name} was involved in {ctype}.")
            opened.append(case)
        return opened

    def _pick_victim(self, population: Any, offender: Any, tile: Any) -> Optional[Any]:
        people = population.people
        pool = [pid for pid in tile.occupants if pid != offender.person_id and pid in people]
        if not pool and offender.social_circle:
            pool = [pid for pid in offender.social_circle if pid in people][:8]
        if not pool:
            return None
        return people[pool[int(self.rng.random() * len(pool))]]

    def _harm_victim(self, population: Any, offender: Any, victim: Any, ctype: str, severity: float, stolen: float) -> None:
        life = self.life
        if ctype in ("assault", "robbery"):
            if life is not None:
                life.injure(population, victim.person_id, "assault" if ctype == "assault" else "weapon",
                            0.15 + 0.7 * severity * (0.6 if ctype == "robbery" else 1.0), f"Attacked by a stranger ({ctype})")
        elif ctype == "homicide":
            if life is not None:
                life.injure(population, victim.person_id, "weapon", 0.95, "Fatally attacked")
            victim.health = 0.0
            self._pending_deaths.append(victim.person_id)
        elif ctype == "arson":
            pass
        else:
            victim.stress = clamp(victim.stress + 0.08, 0.0, 1.0)
            victim.add_memory("crime", f"Was the victim of {ctype}" + (f" and lost ${stolen:.0f}." if stolen else "."), -0.3)
            if life is not None and ctype in ("theft", "fraud"):
                life.traumatize(population, victim.person_id, 0.03, f"Felt unsafe after {ctype}.")



@dataclass
class Prisoner:
    pid: int
    case_id: int
    days_left: int
    served: int = 0
    rehab: float = 0.0


class LawSystem:
    """Police, prosecutors, courts and prison as capacity-limited institutions."""

    ROLE_COUNTS = ("Police Officer", "Detective", "Sergeant", "Judge", "Prosecutor", "Public Defender", "Prison Guard", "Warden")

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.rng = random.Random(seed + 706)
        self.world = world
        self.docket: Dict[int, CourtDocketItem] = {}
        self._docket_counter = 1
        self.inmates: Dict[int, int] = {}
        self.safety_multiplier = 1.0
        self.life: Any = None
        self.custody: Dict[int, int] = {}                 # suspect pid -> arrest day
        self.probation: Dict[int, int] = {}               # pid -> days left
        self.stats: Dict[str, float] = {
            "arrests": 0, "convictions": 0, "acquittals": 0, "pleas": 0, "wrongful": 0, "paroled": 0,
            "reoffended": 0, "fines": 0.0, "unsolved": 0, "prison_fights": 0, "guard_injuries": 0,
            "officer_injuries": 0, "backlog": 0, "restitution": 0.0,
        }
        self._rehab: Dict[int, float] = {}

    # ---- staffing ---------------------------------------------------
    def staff(self, population: Any) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {r: [] for r in self.ROLE_COUNTS}
        people = population.people
        for company in self.world.companies.values():
            if not company.public:
                continue
            site = self.world.buildings.get(company.building_id)
            if site is None or site.building_type not in ("police", "court", "prison"):
                continue
            for pid in company.employee_ids:
                p = people.get(pid)
                if p is not None and p.job_title in out and not p.incarcerated:
                    out[p.job_title].append(pid)
        return out

    def prison_capacity(self) -> int:
        return sum(max(10, min(b.capacity, 30 * max(1, b.floors))) for b in self.world.building_by_type("prison")) or 20

    # ---- investigation -> arrest -> charge ----------------------------
    def investigate(self, population: Any, crime_system: CrimeSystem, day_index: int) -> List[str]:
        reports: List[str] = []
        people = population.people
        staff = self.staff(population)
        officers = len(staff["Police Officer"]) + len(staff["Sergeant"])
        detectives = len(staff["Detective"])
        open_cases = [c for c in crime_system.cases.values() if c.status in ("reported", "investigating")]
        # Detectives work serious cases first; patrol officers help with the rest.
        capacity = detectives * 2.5 + officers * 0.6
        open_cases.sort(key=lambda c: -(c.severity + c.harm))
        for idx, case in enumerate(open_cases):
            suspect = people.get(case.suspect_id)
            if suspect is None or suspect.age < 14:
                case.status, case.resolved = "closed", True
                continue
            if day_index - case.opened_day > 60 and case.status != "investigating":
                case.status = "unsolved"
                self.stats["unsolved"] += 1
                continue
            worked = idx < capacity
            if worked:
                case.status = "investigating"
                if case.assigned_officer is None:
                    pool = staff["Detective"] or staff["Police Officer"]
                    if pool:
                        case.assigned_officer = pool[idx % len(pool)]
                officer = people.get(case.assigned_officer) if case.assigned_officer is not None else None
                skill = officer.skills.get("law", 0.3) if officer is not None else 0.2
                case.evidence = clamp(case.evidence + self.rng.uniform(0.0, 0.1) * (0.6 + skill) + len(case.witnesses) * 0.004, 0.0, 1.0)
            else:
                case.evidence = clamp(case.evidence - 0.008, 0.0, 1.0)      # evidence decays when ignored
            if not worked or suspect.incarcerated:
                continue
            # Misidentification: the wrong person is sometimes pursued.
            if self.rng.random() < (0.35 * (case.evidence + case.severity * 0.5) * self.safety_multiplier) * 0.16:
                if suspect.person_id in self.custody:
                    continue
                self._arrest(population, suspect, case, day_index, staff)
                reports.append(f"Police arrested {suspect.full_name} for {case.crime_type} (case {case.case_id}).")
        return reports

    def _arrest(self, population: Any, suspect: Any, case: CrimeCase, day_index: int, staff: Dict[str, List[int]]) -> None:
        self.stats["arrests"] += 1
        case.status = "arrested"
        case.arrest_day = day_index
        self.custody[suspect.person_id] = day_index
        suspect.stress = clamp(suspect.stress + 0.15, 0.0, 1.0)
        suspect.add_memory("law", f"Was arrested for {case.crime_type}.", -0.4)
        if case.crime_type in ("assault", "robbery", "homicide", "arson") and self.life is not None and self.rng.random() < 0.18:
            pool = staff["Police Officer"]
            if pool:
                oid = pool[int(self.rng.random() * len(pool))]
                self.life.injure(population, oid, "duty", 0.1 + 0.25 * case.severity, f"Hurt arresting a suspect ({case.crime_type})")
                self.stats["officer_injuries"] += 1
        # Prosecutor decides: enough evidence -> charge, otherwise release.
        pros = staff["Prosecutor"]
        if case.evidence < 0.22 or not pros:
            case.status = "investigating"
            self.custody.pop(suspect.person_id, None)
            return
        case.prosecutor_id = pros[int(self.rng.random() * len(pros))]
        case.charge = {"theft": "larceny", "robbery": "armed robbery", "assault": "battery", "fraud": "fraud",
                       "vandalism": "criminal damage", "arson": "arson", "homicide": "murder"}.get(case.crime_type, case.crime_type)
        case.status = "charged"
        wealth = suspect.cash + suspect.bank_balance
        defenders = staff["Public Defender"]
        if wealth > 6000:
            case.private_lawyer = clamp(wealth / 60000.0, 0.1, 0.9)
        elif defenders:
            case.defender_id = defenders[int(self.rng.random() * len(defenders))]
        # Hearing scheduling is limited by judges: a backlog builds when courts are short-staffed.
        judges = max(1, len(staff["Judge"]))
        backlog_days = len(self.docket) / (judges * 2.0)
        hearing = day_index + 1 + int(backlog_days) + self.rng.randint(0, 3)
        did = self._docket_counter
        self._docket_counter += 1
        self.docket[did] = CourtDocketItem(did, case.case_id, hearing, self.rng.uniform(-0.1, 0.1),
                                           clamp(case.severity + len(case.witnesses) * 0.05, 0.0, 1.0))
        if staff["Judge"]:
            case.judge_id = staff["Judge"][did % len(staff["Judge"])]

    # ---- trials, sentencing, prison -----------------------------------
    def resolve_courts(self, population: Any, crime_system: CrimeSystem, day_index: int) -> List[str]:
        results: List[str] = []
        people = population.people
        staff = self.staff(population)
        judges = len(staff["Judge"])
        per_day = max(1, judges * 2)
        due = sorted((d for d in self.docket.values() if day_index >= d.hearing_day), key=lambda d: d.hearing_day)
        self.stats["backlog"] = len(due)
        for docket in due[:per_day]:
            case = crime_system.cases.get(docket.case_id)
            self.docket.pop(docket.docket_id, None)
            if case is None or case.resolved:
                continue
            suspect = people.get(case.suspect_id)
            if suspect is None:
                case.resolved, case.status = True, "closed"
                continue
            judge = people.get(case.judge_id) if case.judge_id is not None else None
            bias = docket.judge_bias + ((judge.personality.empathy - 0.5) * 0.1 if judge is not None else 0.0)
            prosecutor = people.get(case.prosecutor_id) if case.prosecutor_id is not None else None
            defender = people.get(case.defender_id) if case.defender_id is not None else None
            p_skill = prosecutor.skills.get("law", 0.3) if prosecutor is not None else 0.2
            d_skill = case.private_lawyer if case.private_lawyer else (defender.skills.get("law", 0.2) * 0.8 if defender is not None else 0.0)
            strength = case.evidence * 0.62 + case.severity * 0.18 + docket.public_attention * 0.08 + (p_skill - d_skill) * 0.18 \
                - suspect.reputation_public * 0.08 + bias
            guilty_plea = strength > 0.52 and self.rng.random() < 0.42
            if guilty_plea:
                self.stats["pleas"] += 1
                case.plea = "guilty"
            convict = guilty_plea or strength + self.rng.uniform(-0.12, 0.12) > 0.5
            suspect_cleared = False
            if convict:
                self._sentence(population, suspect, case, day_index, plea=guilty_plea)
                case.verdict = "guilty"
                self.stats["convictions"] += 1
                results.append(f"{suspect.full_name} was convicted of {case.charge or case.crime_type}"
                               f" ({case.sentence_type}{', ' + str(case.sentence_days) + ' days' if case.sentence_days else ''}).")
            else:
                case.verdict = "not_guilty"
                self.stats["acquittals"] += 1
                suspect_cleared = True
                suspect.add_memory("law", f"A court cleared them in a {case.crime_type} case.", 0.2)
                results.append(f"{suspect.full_name} was acquitted of {case.charge or case.crime_type}.")
            self.custody.pop(suspect.person_id, None)
            case.resolved = True
            case.status = "closed"
            case.resolved_day = day_index
            if suspect_cleared:
                continue
        self._tick_custody(population, day_index)
        self._tick_prison(population, day_index, staff, results)
        self._tick_probation(population)
        self._prune_cases(crime_system, day_index)
        return results

    def _sentence(self, population: Any, suspect: Any, case: CrimeCase, day_index: int, plea: bool) -> None:
        sev = case.severity
        discount = 0.6 if plea else 1.0
        prior = len(suspect.arrest_record)
        if suspect.age < 18 and case.crime_type != "homicide":
            kind = "probation" if sev > 0.4 else "community service"      # juvenile court: no prison
        elif case.crime_type in ("homicide", "arson") or (case.crime_type in ("robbery", "assault") and sev > 0.7) or prior >= 3:
            kind = "prison"
        elif sev > 0.45 and (prior >= 1 or case.crime_type in ("robbery", "assault")):
            kind = "prison" if self.rng.random() < 0.55 else "probation"
        elif case.crime_type in ("fraud", "theft") and sev > 0.3:
            kind = "fine+probation"
        else:
            kind = "fine" if self.rng.random() < 0.6 else "community service"
        case.sentence_type = kind
        days = 0
        if "prison" in kind:
            base = {"homicide": 600, "arson": 240, "robbery": 120, "assault": 70}.get(case.crime_type, 40)
            days = int(max(5, base * (0.4 + sev) * discount * (1.0 + 0.15 * prior)))
        if "fine" in kind:
            case.fine = round((300 + 4000 * sev) * discount, 2)
            paid = min(case.fine, max(0.0, suspect.cash))
            suspect.cash -= paid
            suspect.debt += case.fine - paid
            self.stats["fines"] += case.fine
        if "probation" in kind or kind == "community service":
            self.probation[suspect.person_id] = int(60 + 200 * sev)
        if case.victim_id is not None and case.stolen > 0:
            victim = population.people.get(case.victim_id)
            if victim is not None and suspect.cash > 0:
                back = min(case.stolen, suspect.cash)
                suspect.cash -= back
                victim.cash += back
                self.stats["restitution"] += back
        case.sentence_days = days
        suspect.arrest_record.append(f"{case.crime_type}:{days}d")
        suspect.reputation_public = clamp(suspect.reputation_public - 0.12, 0.0, 1.0)
        suspect.stress = clamp(suspect.stress + 0.12, 0.0, 1.0)
        suspect.add_memory("law", f"Convicted for {case.crime_type} - {kind}.", -0.35)
        if days > 0:
            if len(self.inmates) >= self.prison_capacity():
                self._parole_oldest(population)          # overcrowding: someone is released early
            self._incarcerate(suspect)
            self.inmates[suspect.person_id] = days
        if len(suspect.arrest_record) > 1:
            self.stats["reoffended"] += 1
        if case.innocent:
            self.stats["wrongful"] += 1

    def _incarcerate(self, person: Any) -> None:
        person.incarcerated = True
        person.care_tile = None
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

    def _parole_oldest(self, population: Any) -> None:
        if not self.inmates:
            return
        pid = min(self.inmates, key=lambda k: self.inmates[k])
        self._release(population, pid, parole=True)

    def _release(self, population: Any, pid: int, parole: bool = False) -> None:
        self.inmates.pop(pid, None)
        person = population.people.get(pid)
        if person is None:
            return
        person.incarcerated = False
        if parole:
            self.stats["paroled"] += 1
            self.probation[pid] = 90
        rehab = self._rehab.pop(pid, 0.0)
        person.crime_tendency = clamp(person.crime_tendency - rehab * 0.3 + 0.03, 0.0, 1.0)
        person.add_memory("law", "Was released from prison.", 0.3)
        home = person.home_tile(self.world)
        if home:
            person.location = home
            person.destination = home
            self.world.register_population_position(pid, *home)

    def _tick_custody(self, population: Any, day_index: int) -> None:
        for pid, day in list(self.custody.items()):
            if pid not in population.people or day_index - day > 14:
                self.custody.pop(pid, None)

    def _tick_prison(self, population: Any, day_index: int, staff: Dict[str, List[int]], results: List[str]) -> None:
        people = population.people
        guards = max(1, len(staff["Prison Guard"]))
        crowd = len(self.inmates) / max(1.0, self.prison_capacity())
        ratio = len(self.inmates) / guards
        for pid in list(self.inmates):
            person = people.get(pid)
            if person is None:
                del self.inmates[pid]
                continue
            self.inmates[pid] -= 1
            person.location = self._prison_tile()
            person.destination = person.location
            person.path = []
            self.world.register_population_position(pid, *person.location)
            person.happiness = clamp(person.happiness - 0.01, 0.0, 1.0)
            # Rehabilitation grows with good conduct and falls under overcrowding.
            self._rehab[pid] = clamp(self._rehab.get(pid, 0.0) + (0.004 if crowd < 1.0 else -0.002) + person.personality.discipline * 0.002, 0.0, 1.0)
            if self.life is not None and self.rng.random() < 0.0015 * (1.0 + ratio / 5.0 + max(0.0, crowd - 1.0) * 3.0):
                self.life.injure(population, pid, "assault", 0.2 + 0.4 * self.rng.random(), "Hurt in a prison fight")
                self.stats["prison_fights"] += 1
                if staff["Prison Guard"] and self.rng.random() < 0.2:
                    gid = staff["Prison Guard"][int(self.rng.random() * len(staff["Prison Guard"]))]
                    self.life.injure(population, gid, "duty", 0.15, "Hurt breaking up a fight")
                    self.stats["guard_injuries"] += 1
            if self.inmates[pid] <= 0:
                self._release(population, pid)
                results.append(f"{person.full_name} completed their sentence and was released.")

    def _tick_probation(self, population: Any) -> None:
        for pid in list(self.probation):
            p = population.people.get(pid)
            if p is None:
                del self.probation[pid]
                continue
            self.probation[pid] -= 1
            if self.probation[pid] <= 0:
                del self.probation[pid]

    def _prune_cases(self, crime_system: CrimeSystem, day_index: int) -> None:
        for case_id, case in list(crime_system.cases.items()):
            if case.resolved and day_index - case.resolved_day > 60:
                del crime_system.cases[case_id]
            elif case.status == "unsolved" and day_index - case.opened_day > 120:
                del crime_system.cases[case_id]
            elif day_index - case.opened_day > 240:
                del crime_system.cases[case_id]

    def _prison_tile(self) -> Tuple[int, int]:
        prisons = self.world.building_by_type("prison")
        if prisons:
            return prisons[0].tile
        station = self.world.building_by_type("station")
        return station[0].tile if station else (self.world.width // 2, self.world.height // 2)

    def background_check_score(self, person: Any) -> float:
        penalty = 0.25 if person.arrest_record else 0.0
        debt_penalty = clamp(person.debt / 50_000.0, 0.0, 0.2)
        return clamp(person.reputation_public - penalty - debt_penalty, 0.0, 1.0)

    def metrics(self, population: Any, crime: CrimeSystem) -> Dict[str, float]:
        staff = self.staff(population)
        total = max(1.0, crime.stats["committed"])
        solved = self.stats["arrests"]
        return {
            "Police officers": float(len(staff["Police Officer"]) + len(staff["Sergeant"])),
            "Detectives": float(len(staff["Detective"])),
            "Judges": float(len(staff["Judge"])),
            "Open cases": float(sum(1 for c in crime.cases.values() if c.status in ("reported", "investigating"))),
            "Court backlog": float(self.stats["backlog"]),
            "Pending trials": float(len(self.docket)),
            "Inmates": float(len(self.inmates)),
            "Prison capacity": float(self.prison_capacity()),
            "On probation": float(len(self.probation)),
            "Crimes committed": float(crime.stats["committed"]),
            "Reported share": float(crime.stats["reported"]) / total,
            "Clearance rate": solved / total,
            "Conviction rate": self.stats["convictions"] / max(1.0, self.stats["convictions"] + self.stats["acquittals"]),
            "Plea deals": float(self.stats["pleas"]),
            "Prison fights": float(self.stats["prison_fights"]),
        }
