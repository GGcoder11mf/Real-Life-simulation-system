"""
analytics.py

City-wide analytics computed once per simulated day from the live population.
The result is a flat dict of floats (for charts and tiles) plus a few
distribution arrays (age pyramid, wealth histogram, sector employment).
History is kept as bounded time series so frontends can draw trend lines.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

HISTORY_LIMIT = 540

# Metrics whose history is retained for charts.
TRACKED = [
    "landlord_income_month", "owner_occupied_share", "avg_rent", "arrears_households",
    "population", "unemployment", "employment_rate", "avg_salary", "gini", "poverty_rate",
    "avg_happiness", "avg_stress", "avg_health", "crime_per_1k", "incarceration_rate",
    "infectious", "infected_share", "avg_commute_min", "traffic_speed", "congestion", "aqi",
    "treasury", "tax_revenue", "spending", "gdp_per_capita", "rent_burden", "inflation",
    "housing_occupancy", "birth_rate", "death_rate", "pct_seniors", "vacancies",
    "service_health", "service_safety", "service_education", "service_transit",
    "median_net_worth", "price_level",
]

AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200]


def gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    v = np.sort(values.astype(np.float64))
    v = v - min(0.0, v[0])  # shift so all values are non-negative
    total = v.sum()
    if total <= 0:
        return 0.0
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum.sum() / total)) / n)


class CityAnalytics:
    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {key: [] for key in TRACKED}
        self.history_days: List[int] = []
        self.latest: Dict[str, float] = {}
        self.age_pyramid: Dict[str, List[int]] = {"men": [], "women": []}
        self.wealth_hist: List[int] = []
        self.sector_jobs: Dict[str, int] = {}
        self._last_births = 0
        self._last_deaths = 0

    # ------------------------------------------------------------------
    def compute(self, engine: Any, day_index: int, crime_opened: int = 0) -> Dict[str, float]:
        population = engine.population
        world = engine.world
        people = list(population.people.values())
        n = len(people)
        m: Dict[str, float] = {"population": float(n)}
        if n == 0:
            self.latest = m
            return m

        age = np.fromiter((p.age for p in people), dtype=np.int32, count=n)
        happy = np.fromiter((p.happiness for p in people), dtype=np.float32, count=n)
        stress = np.fromiter((p.stress for p in people), dtype=np.float32, count=n)
        health = np.fromiter((p.health for p in people), dtype=np.float32, count=n)
        energy = np.fromiter((p.energy for p in people), dtype=np.float32, count=n)
        salary = np.fromiter((p.salary for p in people), dtype=np.float64, count=n)
        pension = np.fromiter((p.pension for p in people), dtype=np.float64, count=n)
        networth = np.fromiter((p.cash + p.bank_balance + p.assets - p.debt - p.mortgage for p in people), dtype=np.float64, count=n)
        liquid = np.fromiter((p.cash + p.bank_balance - p.debt for p in people), dtype=np.float64, count=n)
        debt = np.fromiter((p.debt for p in people), dtype=np.float64, count=n)
        employed = np.fromiter((p.job_company_id is not None for p in people), dtype=bool, count=n)
        inc = np.fromiter((p.incarcerated for p in people), dtype=bool, count=n)
        commute = np.fromiter((p.commute_minutes for p in people), dtype=np.float32, count=n)
        spouse = np.fromiter((p.spouse_id is not None for p in people), dtype=bool, count=n)
        college = np.fromiter((p.education_level == "college" for p in people), dtype=bool, count=n)
        circle = np.fromiter((len(p.social_circle) for p in people), dtype=np.float32, count=n)
        vaccinated = np.fromiter((p.vaccinated for p in people), dtype=bool, count=n)
        infected = np.fromiter((p.infection_state == 2 for p in people), dtype=bool, count=n)
        gender = np.fromiter((0 if p.gender == "man" else 1 for p in people), dtype=np.int8, count=n)

        adults = (age >= 18) & (age < 65) & ~inc
        labor_force = int(adults.sum())
        employed_lf = int((adults & employed).sum())
        m["labor_force"] = float(labor_force)
        m["employed"] = float(employed_lf)
        m["unemployed"] = float(labor_force - employed_lf)
        m["unemployment"] = (labor_force - employed_lf) / labor_force if labor_force else 0.0
        m["employment_rate"] = employed_lf / labor_force if labor_force else 0.0
        m["labor_participation"] = labor_force / max(1, int(((age >= 18) & (age < 65)).sum()))
        m["vacancies"] = float(world.open_job_count())

        # Demographics
        m["pct_children"] = float((age < 18).mean())
        m["pct_seniors"] = float((age >= 65).mean())
        m["median_age"] = float(np.median(age))
        working = max(1, int(((age >= 18) & (age < 65)).sum()))
        m["dependency_ratio"] = float(((age < 18) | (age >= 65)).sum() / working)
        m["married_share"] = float(spouse[age >= 18].mean()) if (age >= 18).any() else 0.0
        m["college_share"] = float(college[age >= 22].mean()) if (age >= 22).any() else 0.0
        m["avg_social_circle"] = float(circle.mean())

        # Income and wealth
        paid = salary[salary > 0]
        m["avg_salary"] = float(paid.mean()) if paid.size else 0.0
        m["median_salary"] = float(np.median(paid)) if paid.size else 0.0
        m["gini"] = gini(networth[age >= 18]) if (age >= 18).any() else 0.0
        m["gini_liquid"] = gini(liquid[age >= 18]) if (age >= 18).any() else 0.0
        m["mortgage_debt_total"] = float(sum(p.mortgage for p in people))
        adults_nw = networth[age >= 18]
        m["median_net_worth"] = float(np.median(adults_nw)) if adults_nw.size else 0.0
        m["mean_net_worth"] = float(adults_nw.mean()) if adults_nw.size else 0.0
        m["poverty_rate"] = float((adults_nw < 1500.0).mean()) if adults_nw.size else 0.0
        if adults_nw.size >= 10:
            cut = np.quantile(adults_nw, 0.9)
            top = adults_nw[adults_nw >= cut].clip(min=0).sum()
            tot = adults_nw.clip(min=0).sum()
            m["top10_share"] = float(top / tot) if tot > 0 else 0.0
        else:
            m["top10_share"] = 0.0
        m["avg_debt"] = float(debt.mean())
        income = salary + pension
        m["gdp_per_capita"] = float(income.sum() / 30.0 / n)
        m["pension_recipients"] = float((pension > 0).sum())

        # Wellbeing
        m["avg_happiness"] = float(happy.mean())
        m["avg_stress"] = float(stress.mean())
        m["avg_health"] = float(health.mean())
        m["avg_energy"] = float(energy.mean())
        m["pct_high_stress"] = float((stress > 0.8).mean())
        m["pct_unhappy"] = float((happy < 0.3).mean())

        # Crime and law
        m["crime_per_1k"] = crime_opened * 1000.0 / n
        m["incarceration_rate"] = float(inc.mean())
        m["open_cases"] = float(sum(1 for c in engine.systems.crime.cases.values() if not c.resolved))
        m["crime_index"] = float(np.mean([d.crime_heat for d in world.districts.values()])) if world.districts else 0.0

        # Health
        m["infectious"] = float(infected.sum())
        m["infected_share"] = float(infected.mean())
        m["vaccinated_share"] = float(vaccinated.mean())

        # Mobility
        commuting = commute[employed & (commute > 0)]
        m["avg_commute_min"] = float(commuting.mean()) if commuting.size else 0.0

        # Households & housing
        listings = world.housing_market
        occupied = sum(1 for l in listings.values() if l.occupied)
        m["housing_occupancy"] = occupied / max(1, len(listings))
        burdens = []
        for household in population.households.values():
            members = [population.people.get(pid) for pid in household.member_ids]
            hh_income = sum((x.salary + x.pension) for x in members if x is not None)
            if hh_income > 0:
                burdens.append(min(2.0, household.monthly_costs / hh_income))
        m["rent_burden"] = float(np.mean(burdens)) if burdens else 0.0
        m["households"] = float(len(population.households))
        m["avg_household_size"] = n / max(1, len(population.households))

        # Vital statistics (per 1000 per year, computed from the daily delta)
        births = population.births_total - self._last_births
        deaths = population.deaths_total - self._last_deaths
        self._last_births = population.births_total
        self._last_deaths = population.deaths_total
        m["births_today"] = float(births)
        m["deaths_today"] = float(deaths)
        m["birth_rate"] = births * 365.0 * 1000.0 / n
        m["death_rate"] = deaths * 365.0 * 1000.0 / n
        m["births_total"] = float(population.births_total)
        m["deaths_total"] = float(population.deaths_total)
        if population.death_ages:
            m["avg_age_at_death"] = float(np.mean(population.death_ages))
        else:
            m["avg_age_at_death"] = 0.0

        # Economy
        companies = world.companies
        m["companies"] = float(len(companies))
        if companies:
            m["company_health"] = float(np.mean([c.health_score() for c in companies.values()]))
            m["market_cap"] = float(sum(c.market_value for c in companies.values()))
        snapshot = world.economic_history[-1] if world.economic_history else None
        m["housing_index"] = snapshot.housing_index if snapshot else 0.0
        m["rent_index"] = snapshot.rent_index if snapshot else 0.0

        # Buildings, ownership and housing tenure
        blds = world.buildings
        floors = [b.floors for b in blds.values()]
        m["avg_floors"] = float(np.mean(floors)) if floors else 0.0
        m["tallest_building"] = float(max(floors)) if floors else 0.0
        m["total_floors"] = float(sum(floors))
        m["building_value_total"] = float(sum(b.value for b in blds.values()))
        landlords = [p for p in people if p.owned_buildings]
        m["landlord_count"] = float(len(landlords))
        units_by_owner: Dict[int, int] = {}
        rented = owned = 0
        arrears = 0
        for l in listings.values():
            if not l.occupied:
                continue
            if l.tenure == "own":
                owned += 1
            else:
                rented += 1
                if l.owner_id is not None:
                    units_by_owner[l.owner_id] = units_by_owner.get(l.owner_id, 0) + 1
        occupied_units = max(1, rented + owned)
        m["owner_occupied_share"] = owned / occupied_units
        m["renter_share"] = rented / occupied_units
        m["avg_rent"] = float(np.mean([l.rent for l in listings.values()])) if listings else 0.0
        top_units = sorted(units_by_owner.values(), reverse=True)[:10]
        m["top10_landlord_unit_share"] = sum(top_units) / max(1, rented)
        m["corporate_unit_share"] = (rented - sum(units_by_owner.values())) / max(1, rented)
        m["landlord_income_month"] = float(population.landlord_income_month)
        m["corporate_rent_month"] = float(population.corporate_rent_month)
        m["arrears_households"] = float(sum(1 for h in population.households.values() if h.arrears_months > 0))
        m["floor_area_workers"] = float(sum(len(c.employee_ids) for c in companies.values()))

        # City services & environment from CityDynamics
        dyn = engine.dynamics
        snap = dyn.snapshot()
        m["traffic_speed"] = snap["traffic_factor"]
        m["congestion"] = snap["congestion"]
        m["aqi"] = snap["aqi"]
        m["treasury"] = snap["treasury"]
        m["tax_revenue"] = float(np.mean(dyn.revenue_30d)) if dyn.revenue_30d else 0.0
        m["spending"] = float(np.mean(dyn.spending_30d)) if dyn.spending_30d else 0.0
        m["service_health"] = snap["service_health"]
        m["service_safety"] = snap["service_safety"]
        m["service_education"] = snap["service_education"]
        m["service_transit"] = snap["service_transit"]
        m["inflation"] = snap["inflation"]
        m["price_level"] = snap["price_level"]
        m["disease_deaths"] = snap["disease_deaths"]
        m["recovered_total"] = snap["recovered_total"]
        m["peak_infectious"] = snap["peak_infectious"]

        # Distributions
        hist_idx = np.digitize(age, AGE_BINS) - 1
        pyramid = {"men": [], "women": []}
        for b in range(len(AGE_BINS) - 1):
            in_bin = hist_idx == b
            pyramid["men"].append(int((in_bin & (gender == 0)).sum()))
            pyramid["women"].append(int((in_bin & (gender == 1)).sum()))
        self.age_pyramid = pyramid
        edges = np.array([-1e12, 0, 500, 1500, 4000, 10000, 25000, 60000, 150000, 400000, 1e12])
        self.wealth_hist = np.histogram(adults_nw, bins=edges)[0].astype(int).tolist()
        sector_counts: Dict[str, int] = {}
        for company in companies.values():
            sector_counts[company.sector] = sector_counts.get(company.sector, 0) + len(company.employee_ids)
        self.sector_jobs = dict(sorted(sector_counts.items(), key=lambda kv: kv[1], reverse=True))

        self.latest = m
        self._record(day_index, m)
        return m

    def _record(self, day_index: int, m: Dict[str, float]) -> None:
        self.history_days.append(day_index)
        for key in TRACKED:
            self.history.setdefault(key, []).append(float(m.get(key, 0.0)))
        if len(self.history_days) > HISTORY_LIMIT:
            self.history_days = self.history_days[-HISTORY_LIMIT:]
            for key in self.history:
                self.history[key] = self.history[key][-HISTORY_LIMIT:]

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": {k: list(v) for k, v in self.history.items()},
            "history_days": list(self.history_days),
            "latest": dict(self.latest),
            "last_births": self._last_births,
            "last_deaths": self._last_deaths,
        }

    def load_dict(self, data: Dict[str, Any]) -> None:
        self.history = {k: list(v) for k, v in data.get("history", {}).items()}
        for key in TRACKED:
            self.history.setdefault(key, [])
        self.history_days = list(data.get("history_days", []))
        self.latest = dict(data.get("latest", {}))
        self._last_births = int(data.get("last_births", 0))
        self._last_deaths = int(data.get("last_deaths", 0))
