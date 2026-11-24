"""
dynamics.py

City-scale dynamic systems that sit on top of the per-citizen simulation:

- Epidemics: an SEIR contagion model that spreads through co-location
  (homes, workplaces, shops, transit hubs), with hospitals, vaccination,
  waning immunity and age/health dependent mortality.
- Traffic: rush-hour congestion that slows vehicles and lengthens commutes.
- Municipal budget: taxes flow into a treasury that funds health, safety,
  education and transit; service levels feed back into the other systems.
- Air quality: industry and traffic raise pollution, parks clean it, and a
  high index slowly erodes public health.
- Prices: a smoothed inflation process driven by labour-market tightness.

Every hourly step only touches the (small) set of infectious people and the
tiles they stand on, so the cost is independent of total population.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from world import SUBSTEPS_PER_HOUR, WorldMap, clamp, mode_speeds

# Epidemiological constants (per simulated hour / day).
INCUBATION_DAYS = 2
INFECTIOUS_DAYS = 7
IMMUNITY_DAYS = 180
BASE_CONTACT_RATE = 0.0032     # per infectious co-located person per hour
SYMPTOMATIC_SHARE = 2          # people with id % 3 != 0 are symptomatic
TRAFFIC_EXPONENT = {"none": 0.0, "bike": 0.35, "bus_pass": 1.0, "car": 1.0}
BASE_TPH = {"none": 6, "bus_pass": 8, "bike": 9, "car": 12}

PATHOGENS = [
    ("Harbor Flu", 1.0, 0.7),
    ("Metro Cough", 0.8, 0.5),
    ("Kestrel Fever", 1.35, 1.0),
    ("Ward Virus", 1.15, 0.85),
]


class CityDynamics:
    """Owns epidemic, traffic, budget, pollution and price state."""

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.seed = seed
        self.world = world
        self.rng = random.Random(seed + 909)
        # --- epidemic
        self.exposed: Set[int] = set()
        self.infectious: Set[int] = set()
        self.recovered_count = 0
        self.vaccinated_count = 0
        self.infections_total = 0
        self.disease_deaths = 0
        self.new_infections_today = 0
        self.pathogen_name = "none"
        self.pathogen_strength = 1.0
        self.pathogen_lethality = 0.6
        self.outbreak_day = -1
        self.outbreaks_total = 0
        self.peak_infectious = 0
        # --- traffic
        self.traffic_factor = 1.0
        self.commuters = 0
        self.congestion_index = 0.0
        self.road_tiles = 1
        # --- budget
        self.treasury = 250_000.0
        self.revenue_today = 0.0
        self.spending_today = 0.0
        self.revenue_30d: List[float] = []
        self.spending_30d: List[float] = []
        self.budget_alloc: Dict[str, float] = {"health": 0.30, "safety": 0.22, "education": 0.28, "transit": 0.20}
        self.service: Dict[str, float] = {"health": 1.0, "safety": 1.0, "education": 1.0, "transit": 1.0}
        # --- air quality / prices
        self.aqi = 45.0
        self.price_level = 1.0
        self.inflation = 0.025
        self.rent_burden = 0.0
        self.weekday = 0
        self._count_roads()
        self.apply_traffic_factor(1.0)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _count_roads(self) -> None:
        roads = 0
        for row in self.world.tiles:
            for tile in row:
                if tile.road:
                    roads += 1
        self.road_tiles = max(1, roads)

    def apply_traffic_factor(self, factor: float) -> None:
        factor = clamp(factor, 0.3, 1.0)
        self.traffic_factor = factor
        self.world.traffic_factor = factor
        self.world.mode_tph = mode_speeds(factor)
        self.world.mode_step = [t / SUBSTEPS_PER_HOUR for t in self.world.mode_tph]

    # ------------------------------------------------------------------
    # Hourly
    # ------------------------------------------------------------------
    def tick_hour(self, engine: Any, day_index: int, hour: int, commuters: int) -> List[str]:
        """Advance traffic and contagion by one hour; returns notable events."""
        events: List[str] = []
        self.weekday = (day_index + 3) % 7
        population = engine.population
        n_people = max(1, len(population.people))
        # Traffic: demand relative to a road network that carries ~3 trips/tile/hour.
        self.commuters = commuters
        load = commuters / (self.road_tiles * 3.0)
        weather = engine.weather.condition
        weather_drag = {"rain": 0.88, "storm": 0.72, "fog": 0.85, "cold_snap": 0.9}.get(weather, 1.0)
        service = self.service["transit"]
        raw = 1.0 / (1.0 + 0.55 * max(0.0, load - 0.25) ** 1.15)
        factor = raw * weather_drag * (0.9 + 0.1 * clamp(service, 0.5, 1.4))
        self.congestion_index = clamp(load, 0.0, 3.0)
        self.apply_traffic_factor(factor)
        # Contagion
        if self.infectious:
            self._transmit(population, day_index)
        elif not self.exposed and self.rng.random() < 0.00045 * (1.6 if engine.calendar.season() == "Winter" else 1.0):
            events.extend(self._seed_outbreak(population, day_index))
        return events

    def _seed_outbreak(self, population: Any, day_index: int) -> List[str]:
        people = population.people
        if not people:
            return []
        name, strength, lethality = self.rng.choice(PATHOGENS)
        self.pathogen_name = name
        self.pathogen_strength = strength
        self.pathogen_lethality = lethality
        self.outbreak_day = day_index
        self.outbreaks_total += 1
        ids = list(people.keys())
        seeds = self.rng.sample(ids, min(len(ids), self.rng.randint(4, 12)))
        for pid in seeds:
            self._infect(people[pid], day_index, immediate=True)
        return [f"{name} outbreak detected: {len(seeds)} initial cases."]

    def _infect(self, person: Any, day_index: int, immediate: bool = False) -> None:
        person.infection_day = day_index
        self.infections_total += 1
        self.new_infections_today += 1
        if immediate:
            person.infection_state = 2
            self.infectious.add(person.person_id)
        else:
            person.infection_state = 1
            self.exposed.add(person.person_id)

    def _transmit(self, population: Any, day_index: int) -> None:
        people = population.people
        tiles = self.world.tiles
        h, w = self.world.height, self.world.width
        rng_random = self.rng.random
        # Group infectious carriers by tile.
        carriers: Dict[Tuple[int, int], int] = {}
        for pid in self.infectious:
            person = people.get(pid)
            if person is None:
                continue
            loc = person.location
            carriers[loc] = carriers.get(loc, 0) + 1
        strength = self.pathogen_strength
        for (x, y), count in carriers.items():
            occupants = tiles[y % h][x % w].occupants
            if len(occupants) < 2:
                continue
            crowd = 1.0 + min(1.5, len(occupants) / 40.0)
            p = 1.0 - (1.0 - BASE_CONTACT_RATE * strength * crowd) ** count
            for pid in occupants:
                person = people.get(pid)
                if person is None or person.infection_state != 0:
                    continue
                prob = p * (0.2 if person.vaccinated else 1.0)
                if rng_random() < prob:
                    self._infect(person, day_index)

    # ------------------------------------------------------------------
    # Daily
    # ------------------------------------------------------------------
    def tick_daily(self, engine: Any, day_index: int) -> Dict[str, List[str]]:
        report: Dict[str, List[str]] = {"health": [], "budget": [], "environment": []}
        population = engine.population
        world = self.world
        people = population.people
        self.weekday = (day_index + 3) % 7
        health_service = clamp(self.service["health"], 0.4, 1.6)
        dead: List[int] = []

        # --- progress exposed -> infectious
        for pid in list(self.exposed):
            person = people.get(pid)
            if person is None:
                self.exposed.discard(pid)
                continue
            if day_index - person.infection_day >= INCUBATION_DAYS:
                person.infection_state = 2
                self.exposed.discard(pid)
                self.infectious.add(pid)
        # --- progress infectious -> recovered / dead
        for pid in list(self.infectious):
            person = people.get(pid)
            if person is None:
                self.infectious.discard(pid)
                continue
            duration = INFECTIOUS_DAYS + (pid % 4) - int(health_service > 1.1)
            drain = 0.012 * self.pathogen_lethality
            vuln = 1.0 + (person.age >= 65) * 1.2 + (person.health < 0.4) * 0.8
            person.health = clamp(person.health - drain * vuln / health_service, 0.0, 1.0)
            person.energy = clamp(person.energy - 0.04, 0.0, 1.0)
            person.stress = clamp(person.stress + 0.01, 0.0, 1.0)
            if day_index - person.infection_day >= INCUBATION_DAYS + duration:
                risk = 0.0008 * self.pathogen_lethality * vuln * (1.4 - person.health) / health_service
                if person.age >= 75:
                    risk *= 3.0
                if self.rng.random() < risk * 6.0:
                    dead.append(pid)
                    self.infectious.discard(pid)
                    self.disease_deaths += 1
                else:
                    person.infection_state = 3
                    self.infectious.discard(pid)
                    self.recovered_count += 1
                    person.add_memory("health", f"Recovered from {self.pathogen_name}.", 0.04)
        self.peak_infectious = max(self.peak_infectious, len(self.infectious))
        if dead:
            for pid in dead:
                person = people.get(pid)
                if person is not None:
                    report["health"].append(f"{person.full_name} died of {self.pathogen_name}.")
            population.remove_people(dead)
        # --- waning immunity (stagger by person id)
        if day_index % 5 == 0:
            for person in people.values():
                if person.infection_state == 3 and (day_index - person.infection_day) > IMMUNITY_DAYS + (person.person_id % 40):
                    person.infection_state = 0
        # --- vaccination campaign
        infected_share = len(self.infectious) / max(1, len(people))
        if infected_share > 0.004 and self.treasury > 20_000:
            doses = int(min(self.treasury / 60.0, len(people) * 0.02))
            given = 0
            for person in people.values():
                if given >= doses:
                    break
                if not person.vaccinated and person.infection_state == 0 and person.age >= 5:
                    person.vaccinated = True
                    given += 1
            if given:
                self.vaccinated_count += given
                self.treasury -= given * 40.0
                self.spending_today += given * 40.0
                if given > 50:
                    report["health"].append(f"Vaccination drive: {given} doses given.")
        if not self.infectious and not self.exposed and self.pathogen_name != "none":
            report["health"].append(f"The {self.pathogen_name} outbreak has ended ({self.disease_deaths} deaths total).")
            self.pathogen_name = "none"
        self.new_infections_today = 0

        # --- air quality
        factories = len(world.building_by_type("factory")) + len(world.building_by_type("warehouse")) * 0.5
        park_tiles = len(world._park_tiles)
        target = 28.0 + factories * 0.55 + self.congestion_index * 22.0 - park_tiles * 0.015 * 0.35
        self.aqi = clamp(self.aqi * 0.85 + target * 0.15, 15.0, 260.0)
        world.aqi_health_drain = max(0.0, (self.aqi - 90.0) / 90.0) * 0.0006

        # --- budget
        revenue = self.revenue_today + sum(
            max(0.0, c.payroll_budget) * 0.0007 for c in world.companies.values()
        )
        pop = max(1, len(people))
        # Budget policy: spend roughly what comes in, with a deficit allowance.
        need = pop * 1.2
        spend_budget = max(revenue * 0.98, need * 0.5) + max(0.0, self.treasury - 300_000.0) * 0.02
        if self.treasury < 0:
            spend_budget = revenue * 0.75
        spent = 0.0
        for key, share in self.budget_alloc.items():
            amount = spend_budget * share
            demand = need * {"health": 0.9, "safety": 0.7, "education": 0.9, "transit": 0.6}[key]
            ratio = amount / max(1.0, demand)
            self.service[key] = clamp(self.service[key] * 0.9 + clamp(0.35 + 0.65 * ratio, 0.4, 1.6) * 0.1, 0.4, 1.6)
            spent += amount
        self.treasury += revenue - spent - self.spending_today
        self.revenue_30d.append(revenue)
        self.spending_30d.append(spent + self.spending_today)
        self.revenue_30d = self.revenue_30d[-30:]
        self.spending_30d = self.spending_30d[-30:]
        self.revenue_today = 0.0
        self.spending_today = 0.0
        if self.treasury < -400_000:
            self.treasury = -400_000.0
            report["budget"].append("City treasury is in deep deficit; services are being cut.")

        # --- prices
        snapshot = world.economic_history[-1] if world.economic_history else None
        unemployment = snapshot.unemployment_rate if snapshot else 0.07
        target_inflation = 0.02 + (0.07 - unemployment) * 0.35 + (self.congestion_index - 0.6) * 0.002
        self.inflation = clamp(self.inflation * 0.97 + target_inflation * 0.03 + self.rng.uniform(-0.0004, 0.0004), -0.01, 0.15)
        self.price_level *= 1.0 + self.inflation / 365.0
        return report

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, float]:
        return {
            "infectious": float(len(self.infectious)),
            "exposed": float(len(self.exposed)),
            "recovered_total": float(self.recovered_count),
            "vaccinated": float(self.vaccinated_count),
            "disease_deaths": float(self.disease_deaths),
            "peak_infectious": float(self.peak_infectious),
            "traffic_factor": self.traffic_factor,
            "congestion": self.congestion_index,
            "commuters": float(self.commuters),
            "treasury": self.treasury,
            "service_health": self.service["health"],
            "service_safety": self.service["safety"],
            "service_education": self.service["education"],
            "service_transit": self.service["transit"],
            "aqi": self.aqi,
            "price_level": self.price_level,
            "inflation": self.inflation,
        }

    def summary_lines(self) -> List[str]:
        lines = [
            f"Pathogen: {self.pathogen_name}" + (f"  (since day {self.outbreak_day})" if self.outbreak_day >= 0 and self.pathogen_name != "none" else ""),
            f"Infectious {len(self.infectious)}  Exposed {len(self.exposed)}  Recovered {self.recovered_count}",
            f"Vaccinated {self.vaccinated_count}  Disease deaths {self.disease_deaths}",
            f"Traffic speed {self.traffic_factor * 100:.0f}%  Congestion {self.congestion_index:.2f}",
            f"Air quality index {self.aqi:.0f}",
            f"Treasury ${self.treasury:,.0f}",
            "Services H{health:.2f} S{safety:.2f} E{education:.2f} T{transit:.2f}".format(**self.service),
            f"Prices x{self.price_level:.3f}  Inflation {self.inflation * 100:.1f}%",
        ]
        return lines

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "recovered_count": self.recovered_count,
            "vaccinated_count": self.vaccinated_count,
            "infections_total": self.infections_total,
            "disease_deaths": self.disease_deaths,
            "pathogen_name": self.pathogen_name,
            "pathogen_strength": self.pathogen_strength,
            "pathogen_lethality": self.pathogen_lethality,
            "outbreak_day": self.outbreak_day,
            "outbreaks_total": self.outbreaks_total,
            "peak_infectious": self.peak_infectious,
            "treasury": self.treasury,
            "budget_alloc": dict(self.budget_alloc),
            "service": dict(self.service),
            "aqi": self.aqi,
            "price_level": self.price_level,
            "inflation": self.inflation,
            "traffic_factor": self.traffic_factor,
            "commuters": self.commuters,
            "congestion_index": self.congestion_index,
            "revenue_30d": list(self.revenue_30d),
            "spending_30d": list(self.spending_30d),
        }

    def load_dict(self, data: Dict[str, Any], population: Any) -> None:
        from world import coerce_rng_state

        state = coerce_rng_state(data.get("rng_state"))
        if state is not None:
            self.rng.setstate(state)
        for key in (
            "recovered_count", "vaccinated_count", "infections_total", "disease_deaths",
            "pathogen_name", "pathogen_strength", "pathogen_lethality", "outbreak_day",
            "outbreaks_total", "peak_infectious", "treasury", "aqi", "price_level", "inflation",
        ):
            if key in data:
                setattr(self, key, data[key])
        self.budget_alloc.update(data.get("budget_alloc", {}))
        self.service.update(data.get("service", {}))
        self.revenue_30d = list(data.get("revenue_30d", []))
        self.spending_30d = list(data.get("spending_30d", []))
        self.apply_traffic_factor(float(data.get("traffic_factor", 1.0)))
        self.commuters = int(data.get("commuters", 0))
        self.congestion_index = float(data.get("congestion_index", 0.0))
        self.world.aqi_health_drain = max(0.0, (self.aqi - 90.0) / 90.0) * 0.0006
        self.exposed = {p.person_id for p in population.people.values() if p.infection_state == 1}
        self.infectious = {p.person_id for p in population.people.values() if p.infection_state == 2}
