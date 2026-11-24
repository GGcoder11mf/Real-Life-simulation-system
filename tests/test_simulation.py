from __future__ import annotations

import os
import tempfile
import unittest

from engine import SimulationEngine


class SimulationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SimulationEngine(seed=1234, world_seed=99)
        self.engine.autosave_interval_days = 10 ** 9

    def test_save_load_roundtrip(self) -> None:
        engine = self.engine
        for _ in range(6):
            engine.step()
        handle, path = tempfile.mkstemp(suffix=".pkl")
        os.close(handle)
        try:
            engine.save(path)
            loaded = SimulationEngine.load(path)
        finally:
            os.remove(path)

        self.assertEqual(len(engine.population.people), len(loaded.population.people))
        self.assertEqual(engine.tick_counter, loaded.tick_counter)
        self.assertEqual(engine.calendar.__dict__, loaded.calendar.__dict__)
        self.assertEqual(len(engine.world.companies), len(loaded.world.companies))
        self.assertEqual(len(engine.world.job_market), len(loaded.world.job_market))
        self.assertEqual(engine.selected_person_id, loaded.selected_person_id)

    def test_save_load_does_not_strand_people(self) -> None:
        engine = self.engine
        for _ in range(12):
            engine.step()
        handle, path = tempfile.mkstemp(suffix=".pkl")
        os.close(handle)
        try:
            engine.save(path)
            loaded = SimulationEngine.load(path)
        finally:
            os.remove(path)

        for _ in range(6):
            loaded.step()
        stranded = sum(
            1
            for person in loaded.population.people.values()
            if person.destination != person.location and not person.path
        )
        self.assertEqual(stranded, 0)
        self.assertEqual(loaded.world.route_service.stats.failures, 0)

    def test_save_load_determinism(self) -> None:
        engine = self.engine
        for _ in range(6):
            engine.step()
        handle, path = tempfile.mkstemp(suffix=".pkl")
        os.close(handle)
        try:
            engine.save(path)
            loaded = SimulationEngine.load(path)
        finally:
            os.remove(path)

        for _ in range(6):
            engine.step()
            loaded.step()

        self.assertEqual(engine.tick_counter, loaded.tick_counter)
        self.assertEqual(len(engine.population.people), len(loaded.population.people))
        self.assertEqual(
            sum(1 for person in engine.population.people.values() if person.job_company_id),
            sum(1 for person in loaded.population.people.values() if person.job_company_id),
        )
        for person_id, person in engine.population.people.items():
            other = loaded.population.people[person_id]
            self.assertAlmostEqual(person.happiness, other.happiness, places=3)
            self.assertAlmostEqual(person.stress, other.stress, places=3)

    def test_labor_metrics_are_consistent(self) -> None:
        metrics = self.engine.macro_metrics()
        labor_force, employed, unemployed = self.engine.population.labor_stats()
        self.assertEqual(int(metrics["labor_force"]), labor_force)
        self.assertEqual(int(metrics["employed"]), employed)
        self.assertEqual(int(metrics["unemployed"]), unemployed)
        self.assertEqual(labor_force, employed + unemployed)
        self.assertGreaterEqual(metrics["unemployment"], 0.0)
        self.assertLessEqual(metrics["unemployment"], 1.0)
        self.assertGreaterEqual(metrics["vacancies"], 1)

    def test_incarceration_removes_person_from_labor_force(self) -> None:
        person = next(
            person
            for person in self.engine.population.people.values()
            if person.job_company_id is not None and 18 <= person.age < 65
        )
        before, _, _ = self.engine.population.labor_stats()
        self.engine.systems.law._incarcerate(person)
        after, _, _ = self.engine.population.labor_stats()
        self.assertTrue(person.incarcerated)
        self.assertIsNone(person.job_company_id)
        self.assertEqual(after, before - 1)
        self.assertNotIn(person, self.engine.population.unemployed_people())

    def test_activity_heatmap_covers_population(self) -> None:
        self.engine.step()
        counts = self.engine.population.activity_heatmap(self.engine.calendar.hour)
        self.assertEqual(sum(counts.values()), len(self.engine.population.people))

    def test_overnight_worker_is_awake_during_shift(self) -> None:
        worker = next(
            person
            for person in self.engine.population.people.values()
            if person.job_company_id is not None and person.job_shift[0] > person.job_shift[1]
        )
        self.assertFalse(worker._should_sleep(worker.job_shift[0]))

    def test_remove_person_cleans_company_and_relationships(self) -> None:
        person = next(
            person
            for person in self.engine.population.people.values()
            if person.job_company_id is not None
        )
        company = self.engine.world.companies[person.job_company_id]
        person_id = person.person_id
        self.assertTrue(self.engine.population.remove_person(person_id))
        self.assertNotIn(person_id, self.engine.population.people)
        self.assertNotIn(person_id, company.employee_ids)
        for other in self.engine.population.people.values():
            self.assertNotIn(person_id, other.relationships)


class DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SimulationEngine(seed=21, world_seed=5, preset="small")
        cls.engine.autosave_interval_days = 10 ** 9

    def test_route_arrays_connect_start_to_goal(self) -> None:
        world = self.engine.world
        buildings = list(world.buildings.values())
        start, goal = buildings[0].tile, buildings[-1].tile
        route = world.find_route_array(start, goal)
        self.assertGreater(len(route), 0)
        self.assertEqual(tuple(route[0]), tuple(start))
        self.assertEqual(tuple(route[-1]), tuple(goal))
        steps = abs(route[1:] - route[:-1]).sum(axis=1)
        self.assertTrue((steps == 1).all())

    def test_epidemic_progresses_and_ends(self) -> None:
        engine = SimulationEngine(seed=22, world_seed=5, preset="small")
        engine.autosave_interval_days = 10 ** 9
        engine.dynamics._seed_outbreak(engine.population, 0)
        self.assertGreater(len(engine.dynamics.infectious), 0)
        for _ in range(24 * 25):
            engine.step()
        dyn = engine.dynamics
        self.assertGreater(dyn.infections_total, 12)
        self.assertGreater(dyn.recovered_count, 0)
        states = {p.infection_state for p in engine.population.people.values()}
        self.assertTrue(states <= {0, 1, 2, 3})

    def test_weekend_reduces_commuting(self) -> None:
        engine = self.engine
        worker = next(
            p for p in engine.population.people.values()
            if p.job_company_id is not None and engine.world.companies[p.job_company_id].sector in {"software", "finance", "media"}
        )
        world = engine.world
        weekday_block = worker._plan_hour(world, 0, 10)       # day 0 = Thursday
        saturday_block = worker._plan_hour(world, 2, 10)      # day 2 = Saturday
        self.assertIn(weekday_block.activity, {"work", "commute_work"})
        self.assertNotIn(saturday_block.activity, {"work", "commute_work"})

    def test_protocol_frame_and_static(self) -> None:
        from sim_protocol import build_frame, build_static

        engine = self.engine
        engine.step()
        static = build_static(engine, 1)
        self.assertEqual(static["terrain"].shape, (engine.world.height, engine.world.width))
        frame = build_frame(engine, 1)
        n = len(engine.population.people)
        self.assertEqual(len(frame["ids"]), n)
        self.assertEqual(len(frame["x"]), n)
        self.assertIn("gini", frame["metrics"])

    def test_analytics_and_budget_sane(self) -> None:
        m = self.engine.macro_metrics()
        for key in ("gini", "poverty_rate", "avg_commute_min", "aqi", "treasury", "median_age"):
            self.assertIn(key, m)
        self.assertGreaterEqual(m["gini"], 0.0)
        self.assertLessEqual(m["gini"], 1.0)

    def test_remove_people_batch(self) -> None:
        engine = SimulationEngine(seed=23, world_seed=5, preset="small")
        ids = list(engine.population.people)[:5]
        self.assertEqual(engine.population.remove_people(ids), 5)
        for other in engine.population.people.values():
            for pid in ids:
                self.assertNotIn(pid, other.relationships)
                self.assertNotIn(pid, other.social_circle)


class BuildingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SimulationEngine(seed=31, world_seed=5, preset="small")
        cls.engine.autosave_interval_days = 10 ** 9

    def test_every_building_has_floors_and_units_fit(self) -> None:
        world = self.engine.world
        for b in world.buildings.values():
            self.assertGreaterEqual(b.floors, 1)
            for l in world._building_listings.get(b.building_id, []):
                self.assertTrue(0 <= l.floor < b.floors, (b.building_type, b.floors, l.floor))

    def test_office_height_tracks_company_value(self) -> None:
        world = self.engine.world
        offices = [(world.companies[b.company_id].market_value, b.floors)
                   for b in world.buildings.values() if b.building_type == "office" and b.company_id in world.companies]
        if len(offices) >= 4:
            offices.sort()
            low = sum(f for _, f in offices[: len(offices) // 2])
            high = sum(f for _, f in offices[len(offices) // 2:])
            self.assertGreaterEqual(high, low)

    def test_work_floor_in_range_and_floor_tracking(self) -> None:
        engine = self.engine
        for _ in range(10):
            engine.step()
        world = engine.world
        for p in engine.population.people.values():
            if p.job_company_id in world.companies:
                site = world.buildings[world.companies[p.job_company_id].building_id]
                self.assertTrue(0 <= p.work_floor < site.floors)
            self.assertGreaterEqual(p.floor, -1)

    @staticmethod
    def _make_landlords(engine: SimulationEngine) -> None:
        """Give some savers a deposit so the housing market creates private landlords."""
        pop = engine.population
        for i, p in enumerate(pop.people.values()):
            if 30 <= p.age < 60 and p.salary > 0 and i % 3 == 0:
                p.bank_balance += 400_000.0
        for day in range(15, 15 + 30 * 6, 30):
            pop.run_property_market(day)

    def test_ownership_and_rent_flow(self) -> None:
        engine = SimulationEngine(seed=32, world_seed=5, preset="small")
        pop = engine.population
        self.assertFalse(any(l.owner_id is not None for l in engine.world.housing_market.values()),
                         "nobody should be endowed with property at day 0")
        self._make_landlords(engine)
        owners = {l.owner_id for l in engine.world.housing_market.values() if l.occupied and l.tenure == "rent" and l.owner_id is not None}
        self.assertTrue(owners, "expected private landlords to emerge")
        self.assertTrue(all(o in pop.people for o in owners))
        pop.pay_household_costs()
        self.assertGreater(pop.rent_collected_month, 0.0)
        self.assertGreater(pop.landlord_income_month, 0.0, "rent should reach private landlords")
        self.assertGreater(pop.consumer_spend_month, 0.0)

    def test_estate_transfers_on_death(self) -> None:
        engine = SimulationEngine(seed=33, world_seed=5, preset="small")
        pop = engine.population
        self._make_landlords(engine)
        landlord = next(p for p in pop.people.values() if p.owned_buildings)
        bids = list(landlord.owned_buildings)
        pop.remove_people([landlord.person_id])
        for bid in bids:
            for listing in engine.world._building_listings.get(bid, []):
                self.assertNotEqual(listing.owner_id, landlord.person_id)

    def test_describe_building_and_roundtrip(self) -> None:
        import buildings as B

        engine = self.engine
        co = max(engine.world.companies.values(), key=lambda c: c.market_value)
        d = B.describe_building(engine, co.building_id)
        self.assertEqual(len(d["rows"]), engine.world.buildings[co.building_id].floors)
        apt = next(b for b in engine.world.buildings.values() if b.building_type == "apartment")
        self.assertIsNotNone(B.describe_building(engine, apt.building_id))
        handle, path = tempfile.mkstemp(suffix=".pkl")
        os.close(handle)
        try:
            engine.save(path)
            loaded = SimulationEngine.load(path)
        finally:
            os.remove(path)
        self.assertEqual(
            {b.building_id: b.floors for b in engine.world.buildings.values()},
            {b.building_id: b.floors for b in loaded.world.buildings.values()},
        )
        self.assertEqual(
            sum(len(p.owned_buildings) for p in engine.population.people.values()),
            sum(len(p.owned_buildings) for p in loaded.population.people.values()),
        )


class MovementTests(unittest.TestCase):
    def test_substeps_move_continuously_along_routes(self) -> None:
        engine = SimulationEngine(seed=41, world_seed=5, preset="small")
        engine.autosave_interval_days = 10 ** 9
        for _ in range(2):          # reach 08:00 so people are commuting
            engine.step()
        last = {pid: p.location for pid, p in engine.population.people.items()}
        world = engine.world
        steps_seen = 0
        for _ in range(60):
            segments = engine.step_sub()
            for pid, seg in segments.items():
                self.assertEqual(tuple(int(v) for v in seg[0]), last[pid], "segment must start where the citizen was")
                diffs = abs(seg[1:].astype(int) - seg[:-1].astype(int)).sum(axis=1)
                mode = engine.population.people[pid].mode
                if mode >= 2:     # vehicles repeat a waypoint while waiting at a red light
                    self.assertTrue(((diffs == 1) | (diffs == 0)).all(), "waypoints must be adjacent (or repeated)")
                else:
                    self.assertTrue((diffs == 1).all(), "waypoints must be adjacent tiles")
                    self.assertLessEqual(len(seg) - 1, int(world.mode_step[mode]) + 2)
                last[pid] = tuple(int(v) for v in seg[-1])
                steps_seen += 1
        self.assertGreater(steps_seen, 0)

    def test_vehicles_prefer_roads_and_walkers_tire(self) -> None:
        engine = SimulationEngine(seed=42, world_seed=5, preset="small")
        engine.autosave_interval_days = 10 ** 9
        on_road = total = 0
        for _ in range(24 * 30 // 24 * 3):
            segments = engine.step_sub()
            for pid, seg in segments.items():
                if engine.population.people[pid].mode == 2:      # car
                    for x, y in seg:
                        total += 1
                        on_road += 1 if engine.world.tile_at(int(x), int(y)).road else 0
        if total > 40:
            self.assertGreater(on_road / total, 0.6)
        energies = [p.energy for p in engine.population.people.values()]
        self.assertTrue(all(0.0 <= e <= 1.0 for e in energies))

    def test_calendar_minutes_and_pause_resume(self) -> None:
        engine = SimulationEngine(seed=43, world_seed=5, preset="small")
        engine.step_sub()
        self.assertEqual(engine.calendar.minute, 2)
        engine.step()                 # completes the hour
        self.assertEqual(engine.calendar.minute, 0)


class LayoutTests(unittest.TestCase):
    def test_buildings_sit_on_street_lots_not_roads(self) -> None:
        from world import WorldMap

        world = WorldMap(5, 120, 60, 1.15)
        self.assertGreater(len(world.buildings), 50)
        for b in world.buildings.values():
            covered = [
                (x, y)
                for y in range(b.anchor_y, b.anchor_y + b.height_tiles)
                for x in range(b.anchor_x, b.anchor_x + b.width_tiles)
            ]
            self.assertIn(b.tile, covered)
            for x, y in covered:
                tile = world.tiles[y][x]
                self.assertFalse(tile.road, f"{b.building_type} covers road tile {x},{y}")
                self.assertEqual(tile.building_id, b.building_id)
            door = world.tiles[b.tile[1]][b.tile[0]]
            self.assertTrue(any(
                0 <= door.x + dx < world.width and 0 <= door.y + dy < world.height and world.tiles[door.y + dy][door.x + dx].road
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ), "door tile must front a street")

    def test_variety_of_building_types_and_sizes(self) -> None:
        from world import WorldMap

        world = WorldMap(6, 160, 80, 1.15)
        types = {b.building_type for b in world.buildings.values()}
        sizes = {(b.width_tiles, b.height_tiles) for b in world.buildings.values()}
        self.assertGreaterEqual(len(types), 8)
        self.assertGreaterEqual(len(sizes), 4)

    def test_housing_is_balanced_against_population(self) -> None:
        from world import WorldMap

        world = WorldMap(7, 160, 80, 1.15)
        pop = world.recommended_population_size()
        self.assertLessEqual(world.residential_population_capacity(), int(pop * 2.2))


class EconomyIntegrityTests(unittest.TestCase):
    def test_stock_symbols_are_unique_and_every_company_trades(self) -> None:
        engine = SimulationEngine(seed=51, world_seed=5, preset="large")
        world = engine.world
        symbols = [c.stock_symbol for c in world.companies.values()]
        self.assertEqual(len(symbols), len(set(symbols)), "duplicate tickers")
        engine.step()
        self.assertGreaterEqual(len(engine.systems.stock_market.stocks), len(world.companies))

    def test_duplicate_symbols_in_old_saves_are_repaired(self) -> None:
        engine = SimulationEngine(seed=52, world_seed=5, preset="small")
        world = engine.world
        companies = list(world.companies.values())
        companies[1].stock_symbol = companies[0].stock_symbol
        renamed = world.ensure_unique_symbols()
        self.assertGreaterEqual(renamed, 1)
        self.assertEqual(len({c.stock_symbol for c in world.companies.values()}), len(world.companies))

    def test_vacancies_scale_with_company_size(self) -> None:
        engine = SimulationEngine(seed=53, world_seed=5, preset="large")
        world = engine.world
        counts = {}
        for posting in world.job_market.values():
            counts[posting.company_id] = counts.get(posting.company_id, 0) + 1
        self.assertGreater(max(counts.values()), 2, "big firms should advertise several roles")
        self.assertLess(len(counts), len(world.companies), "some firms should have no openings")
        big = [c for c in world.companies.values() if len(c.employee_ids) >= 60]
        small = [c for c in world.companies.values() if 0 < len(c.employee_ids) <= 20]
        if big and small:
            avg = lambda cs: sum(counts.get(c.company_id, 0) for c in cs) / len(cs)
            self.assertGreater(avg(big), avg(small))

    def test_day_zero_inequality_is_not_baked_in(self) -> None:
        from analytics import gini

        engine = SimulationEngine(seed=54, world_seed=5, preset="large")
        adults = [p for p in engine.population.people.values() if p.age >= 18]
        self.assertTrue(all(not p.owned_buildings and p.mortgage == 0 for p in adults))
        import numpy as np

        nw = np.array([p.cash + p.bank_balance + p.assets - p.debt - p.mortgage for p in adults])
        self.assertLess(gini(nw), 0.40)

    def test_property_market_creates_owners_with_mortgages(self) -> None:
        engine = SimulationEngine(seed=55, world_seed=5, preset="small")
        pop = engine.population
        for p in pop.people.values():
            if p.salary > 0:
                p.bank_balance += 60_000.0
        stats = {"homes": 0, "investments": 0}
        for day in (15, 45, 75):
            s = pop.run_property_market(day)
            stats["homes"] += s["homes"]
            stats["investments"] += s["investments"]
        self.assertGreater(stats["homes"] + stats["investments"], 0)
        owners = [p for p in pop.people.values() if p.owned_buildings]
        self.assertTrue(owners)
        self.assertTrue(any(p.mortgage > 0 for p in owners))
        before = sum(p.mortgage for p in owners)
        pop.service_mortgages()
        self.assertLess(sum(p.mortgage for p in owners), before)

    def test_housing_prices_do_not_deflate(self) -> None:
        engine = SimulationEngine(seed=56, world_seed=5, preset="small")
        world = engine.world
        before = sum(l.purchase_price for l in world.housing_market.values())
        for _ in range(120):
            world.refresh_markets()
        after = sum(l.purchase_price for l in world.housing_market.values())
        self.assertGreater(after / before, 0.97)


class HousingCoverageTests(unittest.TestCase):
    def test_everyone_has_a_home_and_no_one_loiters_at_work_buildings(self) -> None:
        engine = SimulationEngine(seed=61, world_seed=5, preset="large")
        pop = engine.population
        homeless = [p for p in pop.people.values() if p.home_building_id is None]
        self.assertEqual(len(homeless), 0, "every citizen needs a home")
        for _ in range(8):
            engine.step()
        # Offices, factories etc. should only hold their staff (and a few visitors), not residents.
        world = engine.world
        for b in world.buildings.values():
            if b.building_type in ("office", "factory", "warehouse") and b.company_id in world.companies:
                staff = set(world.companies[b.company_id].employee_ids)
                strangers = [pid for pid in world.tile_at(*b.tile).occupants
                             if pid not in staff and pop.people[pid].floor >= 0]
                self.assertLess(len(strangers), 12, f"{b.name} is hosting {len(strangers)} non-staff indoors")

    def test_housing_units_cover_population(self) -> None:
        engine = SimulationEngine(seed=62, world_seed=5, preset="large")
        units = len(engine.world.housing_market)
        self.assertGreaterEqual(units * engine.world.PERSONS_PER_UNIT, len(engine.population.people))


class SignalTests(unittest.TestCase):
    def test_vehicles_stop_for_red_lights(self) -> None:
        from world import signal_state

        engine = SimulationEngine(seed=7, world_seed=5, preset="small")
        engine.autosave_interval_days = 10 ** 9
        world = engine.world
        self.assertGreater(len(world.signal_boxes), 0)
        engine.step()
        entries = violations = waits = 0
        sig = world.signal_grid
        for _ in range(90):
            segments = engine.step_sub()
            for pid, seg in segments.items():
                times = engine.population.last_times.get(pid)
                if times is None:
                    continue
                for k in range(1, len(seg)):
                    x, y = int(seg[k][0]), int(seg[k][1])
                    px, py = int(seg[k - 1][0]), int(seg[k - 1][1])
                    if (x, y) == (px, py):
                        waits += 1
                        continue
                    jid = sig[y][x]
                    if jid and sig[py][px] != jid:
                        entries += 1
                        axis = 0 if x == px else 1
                        if signal_state(jid, axis, world.sim_seconds + float(times[k - 1])) != 0:
                            violations += 1
        self.assertGreater(entries, 20)
        self.assertEqual(violations, 0, "a vehicle entered a junction on red")
        self.assertGreater(waits, 0, "some vehicles should have waited at red")

    def test_signal_timing_is_cyclic_and_exclusive(self) -> None:
        from world import SIGNAL_CYCLE, signal_state

        for jid in (1, 7, 30):
            for t in range(0, int(SIGNAL_CYCLE) * 2, 3):
                ns, ew = signal_state(jid, 0, t), signal_state(jid, 1, t)
                self.assertFalse(ns == 0 and ew == 0, "both axes green at once")
                self.assertEqual(signal_state(jid, 0, t), signal_state(jid, 0, t + SIGNAL_CYCLE))


if __name__ == "__main__":
    unittest.main()
