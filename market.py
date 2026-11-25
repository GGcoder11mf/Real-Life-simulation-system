"""
market.py

Real products and a measurable notion of corporate influence.

    * Every company has a product line (price, unit cost, quality, capacity).
    * Household spending is split into categories and then allocated to the
      producers in each category by a logit of quality, brand, price and
      capacity - so market share is *earned*, not assigned by headcount.
    * Firms buy inputs from suppliers (a supply chain), so manufacturers and
      logistics firms matter even though consumers never see them.
    * An influence index compares companies on one scale: market share,
      revenue, employment, supply-chain dependents, brand, capitalisation
      and wages, plus concentration (HHI) per category.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from world import WorldMap, clamp

# category -> share of household discretionary + essential spending
CATEGORY_SHARE = {
    "groceries": 0.25, "dining": 0.12, "goods": 0.12, "electronics": 0.07, "media": 0.05,
    "finance": 0.04, "transport": 0.08, "education": 0.05, "services": 0.08, "software": 0.06,
    "healthcare": 0.08,
}
# sector -> [(product name, category, base price, unit cost ratio, units per worker-day, input need)]
CATALOG: Dict[str, List[Tuple[str, str, float, float, float, float]]] = {
    "retail": [("groceries", "groceries", 38.0, 0.62, 55.0, 0.55), ("clothing", "goods", 46.0, 0.58, 30.0, 0.55),
               ("home goods", "goods", 60.0, 0.6, 22.0, 0.55), ("gadgets", "electronics", 210.0, 0.72, 8.0, 0.65)],
    "food": [("restaurant meals", "dining", 21.0, 0.55, 45.0, 0.4), ("coffee and bakery", "dining", 6.5, 0.45, 120.0, 0.4)],
    "manufacturing": [("appliances", "goods", 330.0, 0.66, 4.0, 0.5), ("electronics", "electronics", 280.0, 0.7, 5.0, 0.5),
                      ("components", "b2b", 90.0, 0.6, 15.0, 0.4)],
    "software": [("apps and subscriptions", "software", 14.0, 0.2, 250.0, 0.05), ("cloud services", "b2b", 400.0, 0.3, 4.0, 0.05)],
    "media": [("streaming and news", "media", 12.0, 0.35, 180.0, 0.1)],
    "finance": [("accounts and loans", "finance", 25.0, 0.3, 90.0, 0.05), ("insurance", "finance", 60.0, 0.5, 40.0, 0.05)],
    "transport": [("tickets", "transport", 4.0, 0.7, 400.0, 0.2)],
    "logistics": [("freight", "b2b", 180.0, 0.7, 10.0, 0.3), ("parcel delivery", "services", 9.0, 0.7, 120.0, 0.3)],
    "construction": [("building works", "b2b", 4500.0, 0.75, 0.3, 0.45), ("repairs", "services", 140.0, 0.6, 5.0, 0.45)],
    "healthcare": [("care services", "healthcare", 95.0, 0.6, 12.0, 0.2)],
    "education": [("courses", "education", 40.0, 0.5, 30.0, 0.05)],
    "public_service": [("public services", "services", 0.0, 0.0, 0.0, 0.0)],
}
# who supplies whom (buyer sector -> supplier sectors)
SUPPLIERS = {
    "retail": ("manufacturing", "logistics"), "food": ("retail", "logistics"), "manufacturing": ("manufacturing", "logistics"),
    "construction": ("manufacturing", "logistics"), "logistics": ("manufacturing",), "software": ("software",),
    "healthcare": ("manufacturing", "logistics"), "media": ("software",), "finance": ("software",),
    "transport": ("manufacturing",), "education": ("media",),
}


@dataclass
class Product:
    name: str
    category: str
    price: float
    unit_cost: float
    quality: float
    capacity: float          # units per day
    sold_30d: float = 0.0
    revenue_30d: float = 0.0
    sold_today: float = 0.0
    revenue_today: float = 0.0


@dataclass
class CompanyMarket:
    products: List[Product]
    brand: float = 0.3
    suppliers: List[int] = None  # type: ignore[assignment]
    b2b_revenue_30d: float = 0.0
    b2b_today: float = 0.0
    cogs_30d: float = 0.0
    profit_30d: float = 0.0
    influence: float = 0.0
    share: float = 0.0
    rev_hist: List[float] = None  # type: ignore[assignment]


class MarketSystem:
    def __init__(self, seed: int, world: WorldMap) -> None:
        self.rng = random.Random(seed + 9300)
        self.world = world
        self.cm: Dict[int, CompanyMarket] = {}
        self.category_revenue: Dict[str, float] = {}
        self.category_hhi: Dict[str, float] = {}
        self.ranked: List[int] = []
        self.total_consumer_spend = 0.0
        self.reserve = 0.0
        self.ticks = 0

    # ------------------------------------------------------------- setup
    def ensure_company(self, company: Any) -> Optional[CompanyMarket]:
        cm = self.cm.get(company.company_id)
        if cm is not None:
            return cm
        lines = CATALOG.get(company.sector)
        if not lines:
            return None
        rng = self.rng
        products: List[Product] = []
        quality_base = clamp(0.25 + company.productivity * 0.45 + company.prestige * 0.3 + rng.uniform(-0.1, 0.1), 0.1, 1.0)
        pick = lines if company.company_scale != "indie" else lines[:1]
        for name, cat, price, cost_ratio, upw, _need in pick:
            if price <= 0:
                continue
            price_j = price * rng.uniform(0.85, 1.2) * (1.0 + 0.25 * (quality_base - 0.5))
            products.append(Product(name, cat, round(price_j, 2), round(price_j * cost_ratio * rng.uniform(0.9, 1.05), 2),
                                    round(quality_base + rng.uniform(-0.08, 0.08), 3),
                                    max(1.0, upw * max(1, len(company.employee_ids)) * (0.6 + company.productivity))))
        cm = CompanyMarket(products=products, brand=clamp(company.prestige * 0.6 + 0.1, 0.0, 1.0), suppliers=[], rev_hist=[])
        self.cm[company.company_id] = cm
        self._link_suppliers(company, cm)
        return cm

    def _link_suppliers(self, company: Any, cm: CompanyMarket) -> None:
        wanted = SUPPLIERS.get(company.sector, ())
        pool = [c for c in self.world.companies.values() if c.sector in wanted and c.company_id != company.company_id and not c.public]
        if not pool:
            return
        pool.sort(key=lambda c: (abs(c.district_id - company.district_id), c.company_id))
        near = pool[:12]
        k = min(len(near), self.rng.randint(2, 4))
        cm.suppliers = [c.company_id for c in self.rng.sample(near, k)]

    def sync(self) -> None:
        companies = self.world.companies
        for company in companies.values():
            self.ensure_company(company)
        for cid in [cid for cid in self.cm if cid not in companies]:
            del self.cm[cid]

    # -------------------------------------------------------- consumption
    def allocate(self, pool: float) -> None:
        """Households settle bills monthly; firms are paid as the money trickles out daily."""
        self.reserve += pool

    def _allocate_now(self, pool: float) -> None:
        """Split spending across categories and into producers' cash."""
        self.sync()
        self.total_consumer_spend += pool
        by_cat: Dict[str, List[Tuple[Any, Product]]] = {}
        for cid, cm in self.cm.items():
            company = self.world.companies.get(cid)
            if company is None or not company.employee_ids:
                continue
            for prod in cm.products:
                if prod.category != "b2b":
                    by_cat.setdefault(prod.category, []).append((company, prod))
        total_share = sum(CATEGORY_SHARE[c] for c in by_cat if c in CATEGORY_SHARE) or 1.0
        for cat, entries in by_cat.items():
            share = CATEGORY_SHARE.get(cat)
            if share is None:
                continue
            cat_pool = pool * share / total_share
            ref = sum(p.price for _, p in entries) / len(entries)
            weights = []
            for company, prod in entries:
                cm = self.cm[company.company_id]
                util = 2.2 * prod.quality + 1.6 * cm.brand - 1.8 * (prod.price / max(0.01, ref)) + 0.25 * math.log1p(len(company.employee_ids))
                weights.append(math.exp(util))
            total_w = sum(weights)
            spent = 0.0
            overflow = 0.0
            allocs = []
            for (company, prod), w in zip(entries, weights):
                amt = cat_pool * w / total_w
                cap_rev = prod.capacity * prod.price * 1.4          # one day of stock + some overtime
                if amt > cap_rev:
                    overflow += amt - cap_rev
                    amt = cap_rev
                allocs.append([company, prod, amt])
            if overflow > 0:                                         # excess demand spills to firms with spare capacity
                spare = [(a, max(0.0, a[1].capacity * a[1].price * 1.4 - a[2])) for a in allocs]
                tot_spare = sum(s for _, s in spare)
                if tot_spare > 0:
                    for a, s in spare:
                        a[2] += min(s, overflow * s / tot_spare)
            for company, prod, amt in allocs:
                if amt <= 0:
                    continue
                units = amt / max(0.01, prod.price)
                prod.sold_today += units
                prod.revenue_today += amt
                company.cash += amt
                spent += amt
            self.category_revenue[cat] = self.category_revenue.get(cat, 0.0) * 0.9 + spent

    # --------------------------------------------------------- daily tick
    def tick_daily(self, population: Any, day_index: int) -> List[str]:
        self.sync()
        events: List[str] = []
        world = self.world
        rng = self.rng
        self.ticks += 1
        if self.reserve > 1.0:
            daily = self.reserve / 12.0
            self.reserve -= daily
            self._allocate_now(daily)
        # --- production capacity follows staffing and productivity
        for cid, cm in self.cm.items():
            company = world.companies.get(cid)
            if company is None:
                continue
            workers = max(1, len(company.employee_ids))
            lines = {n: (u, need, cr) for n, _c, _p, cr, u, need in CATALOG.get(company.sector, [])}
            for prod in cm.products:
                upw = lines.get(prod.name, (10.0, 0.3, 0.6))[0]
                prod.capacity = max(1.0, upw * workers * (0.5 + company.productivity))
        # --- supply chain: buyers pay suppliers a share of their revenue
        for cid, cm in self.cm.items():
            company = world.companies.get(cid)
            if company is None or not cm.suppliers:
                continue
            revenue = sum(p.revenue_today for p in cm.products)
            if revenue <= 0:
                continue
            need = sum(lines_need for lines_need in [CATALOG.get(company.sector, [("", "", 0, 0, 0, 0.3)])[0][5]])
            spend = revenue * need * 0.6
            live = [s for s in cm.suppliers if s in world.companies and world.companies[s].employee_ids]
            cm.suppliers = live or cm.suppliers
            if not live:
                self._link_suppliers(company, cm)
                continue
            per = spend / len(live)
            company.cash -= spend
            cm.cogs_30d += spend
            for sid in live:
                supplier = world.companies[sid]
                supplier.cash += per
                scm = self.cm.get(sid)
                if scm is not None:
                    scm.b2b_today += per
        # --- price adjustment from sell-through, brand from quality and share
        for cid, cm in self.cm.items():
            company = world.companies.get(cid)
            if company is None:
                continue
            day_rev = 0.0
            for prod in cm.products:
                sell = prod.sold_today / max(1.0, prod.capacity)
                if prod.category != "b2b":
                    adj = 0.015 * (sell - 0.3)
                    prod.price = round(clamp(prod.price * (1.0 + adj) * (1.0 + (world.inflation_rate or 0.0) / 365.0),
                                             prod.unit_cost * 1.05, prod.unit_cost * 4.0), 2)
                prod.unit_cost = round(prod.unit_cost * (1.0 + (world.inflation_rate or 0.0) / 365.0), 3)
                prod.quality = clamp(prod.quality + 0.0006 * (company.productivity - 0.5), 0.05, 1.0)
                prod.sold_30d = prod.sold_30d * 0.967 + prod.sold_today
                prod.revenue_30d = prod.revenue_30d * 0.967 + prod.revenue_today
                day_rev += prod.revenue_today
                prod.sold_today = 0.0
                prod.revenue_today = 0.0
            cm.b2b_revenue_30d = cm.b2b_revenue_30d * 0.967 + cm.b2b_today
            day_rev += cm.b2b_today
            cm.b2b_today = 0.0
            cm.cogs_30d *= 0.967
            cm.rev_hist = (cm.rev_hist or [])[-59:] + [day_rev]
            cm.brand = clamp(cm.brand * 0.998 + 0.002 * (sum(p.quality for p in cm.products) / max(1, len(cm.products))) + 0.0005 * cm.share, 0.0, 1.0)
            cm.profit_30d = (sum(p.revenue_30d for p in cm.products) + cm.b2b_revenue_30d) - cm.cogs_30d - company.payroll_budget
        self._compute_influence(population)
        if self.ranked and day_index % 30 == 0:
            top = world.companies.get(self.ranked[0])
            if top is not None:
                events.append(f"{top.name} is the most influential company in the city (index {self.cm[top.company_id].influence:.2f}).")
        return events

    # ----------------------------------------------------------- influence
    def _compute_influence(self, population: Any) -> None:
        world = self.world
        companies = [c for c in world.companies.values() if c.company_id in self.cm]
        if not companies:
            return
        cat_total: Dict[str, float] = {}
        cat_rev: Dict[Tuple[int, str], float] = {}
        for c in companies:
            for p in self.cm[c.company_id].products:
                if p.category == "b2b":
                    continue
                cat_total[p.category] = cat_total.get(p.category, 0.0) + p.revenue_30d
                cat_rev[(c.company_id, p.category)] = cat_rev.get((c.company_id, p.category), 0.0) + p.revenue_30d
        # concentration per category (Herfindahl-Hirschman)
        for cat, tot in cat_total.items():
            if tot <= 0:
                continue
            shares = [cat_rev[(c.company_id, cat)] / tot for c in companies if (c.company_id, cat) in cat_rev]
            self.category_hhi[cat] = sum((s * 100.0) ** 2 for s in shares)
        dependents: Dict[int, int] = {}
        for cm_id, cm in self.cm.items():
            for sid in cm.suppliers or ():
                dependents[sid] = dependents.get(sid, 0) + 1
        total_emp = max(1, sum(len(c.employee_ids) for c in companies))
        total_rev = max(1.0, sum(sum(p.revenue_30d for p in self.cm[c.company_id].products) + self.cm[c.company_id].b2b_revenue_30d for c in companies))
        total_cap = max(1.0, sum(c.market_value for c in companies))
        wages = [c.payroll_budget / max(1, len(c.employee_ids)) for c in companies if c.employee_ids]
        mean_wage = sum(wages) / max(1, len(wages)) or 1.0
        max_dep = max(dependents.values(), default=1)
        scored: List[Tuple[float, int]] = []
        for c in companies:
            cm = self.cm[c.company_id]
            rev = sum(p.revenue_30d for p in cm.products) + cm.b2b_revenue_30d
            # best category share
            best = 0.0
            for p in cm.products:
                if p.category != "b2b" and cat_total.get(p.category, 0.0) > 0:
                    best = max(best, cat_rev[(c.company_id, p.category)] / cat_total[p.category])
            cm.share = best
            wage = (c.payroll_budget / max(1, len(c.employee_ids))) / mean_wage if c.employee_ids else 0.0
            idx = (0.26 * min(1.0, best * 2.5) + 0.22 * min(1.0, rev / total_rev * 25.0) + 0.14 * min(1.0, len(c.employee_ids) / total_emp * 40.0)
                   + 0.14 * min(1.0, dependents.get(c.company_id, 0) / max(1, max_dep)) + 0.1 * cm.brand
                   + 0.08 * min(1.0, c.market_value / total_cap * 30.0) + 0.06 * min(1.0, wage / 2.0))
            cm.influence = float(idx)
            scored.append((idx, c.company_id))
        scored.sort(reverse=True)
        self.ranked = [cid for _, cid in scored]

    # ------------------------------------------------------------- reports
    def company_lines(self, company_id: int) -> List[str]:
        cm = self.cm.get(company_id)
        company = self.world.companies.get(company_id)
        if cm is None or company is None:
            return []
        rank = self.ranked.index(company_id) + 1 if company_id in self.ranked else 0
        lines = [f"Influence {cm.influence:.2f}  (rank {rank}/{len(self.ranked)})  best share {cm.share:.0%}  brand {cm.brand:.2f}"]
        for p in cm.products[:4]:
            margin = 1.0 - p.unit_cost / max(0.01, p.price)
            lines.append(f"  {p.name}: ${p.price:,.2f}  q{p.quality:.2f}  margin {margin:.0%}  sold/day {p.sold_30d * 0.033:,.0f}")
        if cm.b2b_revenue_30d > 0:
            lines.append(f"  B2B revenue 30d ${cm.b2b_revenue_30d:,.0f}")
        sup = [self.world.companies[s].name for s in (cm.suppliers or ()) if s in self.world.companies][:3]
        if sup:
            lines.append("  Suppliers: " + ", ".join(sup))
        return lines

    def ranking_lines(self, limit: int = 10) -> List[str]:
        out = []
        for i, cid in enumerate(self.ranked[:limit], 1):
            c = self.world.companies.get(cid)
            cm = self.cm.get(cid)
            if c is None or cm is None:
                continue
            out.append(f"{i:2}. {c.name[:26]:26} {cm.influence:.2f}  share {cm.share:3.0%}  staff {len(c.employee_ids)}")
        return out

    def concentration_lines(self) -> List[str]:
        rows = sorted(self.category_hhi.items(), key=lambda kv: -kv[1])[:5]
        return [f"{cat:11} HHI {hhi:6.0f} {'(concentrated)' if hhi > 2500 else '(competitive)' if hhi < 1500 else ''}" for cat, hhi in rows]

    # --------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cm": {str(cid): {
                "products": [p.__dict__ for p in cm.products], "brand": cm.brand, "suppliers": list(cm.suppliers or ()),
                "b2b30": cm.b2b_revenue_30d, "b2bt": cm.b2b_today, "cogs": cm.cogs_30d, "profit": cm.profit_30d,
                "influence": cm.influence, "share": cm.share, "hist": list(cm.rev_hist or ())[-60:]}
                for cid, cm in self.cm.items()},
            "cat_rev": dict(self.category_revenue), "hhi": dict(self.category_hhi), "ranked": list(self.ranked),
            "total": self.total_consumer_spend, "reserve": self.reserve, "rng": self.rng.getstate(),
        }

    @classmethod
    def from_dict(cls, seed: int, world: WorldMap, data: Dict[str, Any]) -> "MarketSystem":
        m = cls(seed, world)
        for cid, d in data.get("cm", {}).items():
            m.cm[int(cid)] = CompanyMarket([Product(**p) for p in d["products"]], d["brand"], list(d["suppliers"]),
                                           d["b2b30"], d["b2bt"], d["cogs"], d["profit"], d["influence"], d["share"], list(d["hist"]))
        m.category_revenue = dict(data.get("cat_rev", {}))
        m.category_hhi = dict(data.get("hhi", {}))
        m.ranked = list(data.get("ranked", []))
        m.total_consumer_spend = float(data.get("total", 0.0))
        m.reserve = float(data.get("reserve", 0.0))
        from world import coerce_rng_state
        st = coerce_rng_state(data.get("rng"))
        if st is not None:
            m.rng.setstate(st)
        return m
