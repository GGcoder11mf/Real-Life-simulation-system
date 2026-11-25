# RLSS (Enhanced) — Autonomous City & Life Simulator

**RLSS (enhanced)** is a high-performance, realism-focused autonomous city and life simulator written in Python. It simulates ~12,000+ individual citizens on a 224×112 tile grid (1 tile = 20 m) using a decoupled multi-process architecture, procedural 2D rendering, Numba-accelerated pathfinding, and deep economic, demographic, epidemiological, and traffic systems.

---

## Table of Contents

- [Requirements & Installation](#requirements--installation)
- [Quick Start](#quick-start)
- [Architecture & Project Structure](#architecture--project-structure)
- [Key Simulation Systems](#key-simulation-systems)
  - [City Layout & Buildings](#city-layout--buildings)
  - [Movement, Traffic & Signals](#movement-traffic--signals)
  - [Economy & Property Market](#economy--property-market)
  - [Realism, Health & Environment](#realism-health--environment)
  - [Analytics & Metrics](#analytics--metrics)
- [Controls & UI](#controls--ui)
- [Benchmarks & Testing](#benchmarks--testing)

---

## Requirements & Installation

Dependencies are listed in `requirements.txt`:

- `numpy >= 1.26`
- `numba >= 0.59`
- `torch >= 2.2`
- `pygame >= 2.5`
- `msgpack >= 1.0`
- `windows-curses >= 2.3` *(Windows only, for the terminal UI)*

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Pygame Graphical Frontend (Recommended)
```bash
# Default large city (~12k citizens, 224x112 tiles)
python pygame_frontend.py

# Smaller presets for faster startup
python pygame_frontend.py --preset small
python pygame_frontend.py --preset classic

# Load an existing save file
python pygame_frontend.py --load life_sim_save.pkl

# Set wall-clock milliseconds per 2 simulated minutes
python pygame_frontend.py --tick-ms 16
```

### Save Inspector
Inspect `.pkl`, `.pickle`, `.sav`, `.save`, or `.gz` save files in a standalone GUI:
```bash
python save_inspector.py
```

---

## Architecture & Project Structure

The simulation uses a **decoupled two-process architecture** so that rendering frame rate (60 FPS) never depends on simulation tick cost or population size:

| File | Responsibility |
| :--- | :--- |
| `sim_host.py` | Runs `SimulationEngine` in a separate background process (`SimClient` / host) and streams state snapshots. |
| `sim_protocol.py` | Shared protocol constants, activity codes, building codes, and terrain keys for IPC between host and frontend. |
| `pygame_frontend.py` | Main 60 FPS Pygame renderer. Interpolates citizen/vehicle positions between snapshots, caches text surfaces, and manages UI panels and input. |
| `world_renderer.py` | Procedural world renderer (no external image textures). Pre-renders terrain, roads, lane markings, sidewalks, zebra crossings, traffic lights, metro lines, trees, shorelines, and multi-floor buildings into cached zoom-level chunks. |
| `engine.py` | `SimulationEngine` — coordinates calendar/time (2-minute sub-steps), weather, events, schedules, save/load (`RLS_SAVE_V2` gzip/msgpack/pickle), and ties all subsystems together. |
| `world.py` | `WorldMap` — vectorised world generation, road hierarchy (arterials + street lattice), lot placement, multi-tile buildings, metro/rail, and traffic signal state (`_compute_signals`, `signal_state`). |
| `entities.py` | `Person` and `Population` — citizen attributes, needs, schedules, household/housing assignment, and second-by-second vehicle driving (`_drive_vehicle`). |
| `buildings.py` | Floor-by-floor building model: capacities, residential units, commercial headcount/height scaling, property ownership, rent rolls, and estates. |
| `systems.py` | `SimulationSystems` — jobs, `CompanySystem` (vacancies, productivity, stock tickers), economy, and daily/monthly cycles. |
| `dynamics.py` | `CityDynamics` — SEIR epidemics (hospitals, vaccination, waning immunity), rush-hour congestion, municipal budget & service levels, air quality, inflation, and Markov seasonal weather. |
| `analytics.py` | `CityAnalytics` — tracks Gini coefficient, poverty rate, median net worth, top-10% wealth share, dependency ratio, age pyramid, sector employment, commute times, rent burden, housing occupancy, GDP per capita, birth/death rates, and crime per 1k. |
| `routing.py` | Numba-accelerated batched parallel A* pathfinding (`_astar_batch`, `_astar_core`) with separate cost grids for pedestrians and vehicles. |
| `ui.py` | Pure terminal UI built on `curses` with map, people, economy, news, calendar, and activity panels. |
| `save_inspector.py` | Standalone Pygame-based save file inspector. |
| `bench.py` | Headless simulation benchmarking script. |
| `tests/test_simulation.py` | Unit tests for the simulation. |

---

## Key Simulation Systems

### City Layout & Buildings
- **Road Hierarchy:** Three arterials plus a street lattice whose block size adapts to the zone (dense downtown, larger suburban and industrial blocks, none in open country).
- **Multi-Tile Footprints:** Buildings are placed on street-facing lots and never overlap roads (factories `3×2`, colleges & prisons `3×3`, hospitals & courts `2×2`, offices up to `2×2`).
- **Floor-by-Floor Simulation:** Houses are 1–3 storeys; apartment blocks grow with their unit count; commercial towers scale weekly (`+/- 1` floor) based on company market value and headcount. Citizens track their live `floor` (`-1` outside, `0` ground).
- **Housing Coverage:** Capacity is counted per unit (1.9 people), and anyone without a unit joins a household with room so no citizen is homeless at generation.

### Movement, Traffic & Signals
- **Continuous 2-Minute Sub-Steps:** Full decision logic (schedules, needs, economy) runs hourly, while citizens move every 2-minute sub-step at real physical speeds (walk `4.8 km/h`, bike `14`, bus `17`, car `28`, taxi `30`), slowed by congestion and weather.
- **Traffic Signals:** Signalised junctions run on a shared 120-second timing cycle. Vehicles advance second-by-second inside each sub-step, stopping at stop lines on red/yellow and queuing.
- **Mode Choice:** Citizens choose between walking, cycling, bus, car, or taxi based on distance, vehicle ownership, weather, time of day, and cash. Vehicles route on a separate road-preferring cost grid.

### Economy & Property Market
- **Unique Stock Tickers:** Every company has its own traded stock symbol (`WorldMap._make_stock_symbol`).
- **Dynamic Vacancies:** Sized per company (`CompanySystem.vacancy_target`) based on headcount, demand, expansion appetite, and productivity.
- **Emergent Inequality & Property Market:** Starting wealth follows a life-cycle profile, and all housing starts corporate-owned on day 0. Homeownership and private landlords emerge organically through a monthly property market (deposits, affordability checks, mortgages). Rent flows from tenant to owner (22% upkeep), owners pay property tax to the city treasury, and estates pass to spouses/children (or corporate landlords) on death.
- **Inflation & Consumption:** Consumption spending acts as a sink that prevents unbounded savings growth and becomes revenue for consumer-facing firms; housing prices track inflation.

### Realism, Health & Environment
- **SEIR Epidemics:** Disease spread with hospitals, vaccination, and waning immunity.
- **Weather & Day/Night:** Markov weather (`clear`, `cloudy`, `rain`, `storm`, `fog`, `heatwave`, `cold_snap`) with seasons and a daily temperature cycle. Nighttime renders street lamps, lit windows, and vehicle headlights.
- **Municipal Budget & Services:** Property taxes fund city services, affecting air quality, crime, and public health.
- **Schedules:** Includes staggered departures, midday work breaks, weekends, weekend-only sectors, and retirement/pensions.

### Analytics & Metrics
Tracked in `analytics.py` with history charts in the UI:
- Gini coefficient, poverty rate, median net worth, top-10% wealth share
- Dependency ratio, age pyramid, birth/death rates
- Sector employment, commute time, rent burden, housing occupancy
- GDP per capita, crime per 1k, and live traffic

---

## v7: smooth stepping and deeper lives

**Stutter fix.** The hour-start decision pass (about 170 ms) and the midnight daily cycle (about 0.5-1 s) used
to run inside a single sub-step, which showed up as one second of smooth walking followed by one second of
stutter. They are now sliced across the 30 sub-steps of the hour (`Engine._run_slot`): planning in the first 10
slots (this also staggers departures), needs/effects in the next 16, then the global systems; the daily cycle
is a list of stages spread over the 30 slots of hour 0. Frames are delta-encoded (only moved citizens plus a
rotating 1/30 refresh of every citizen), and the worker process freezes the city and runs garbage collection
in idle time (a gen-2 collection over millions of objects cost ~190 ms). Average sub-step cost with all systems
on is about 12 ms at 12.5k citizens, with a worst case around 40-80 ms (it was 170-500 ms).
Saving mid-hour flushes pending slots first, and `next_slot` is saved, so save/load stays deterministic.

**`life.py`** - struct-of-arrays anatomy for every citizen: 16 body parts (brain, eyes, ears, heart, lungs, liver,
kidneys, stomach, skin, skull, spine, ribs, arms, legs) with integrity, a permanent-scar ceiling, fractures, pain and
trauma; a 10-locus diploid genome with recessive, dominant, additive and X-linked conditions (cystic fibrosis, sickle
cell, Huntington, type 1 diabetes, hypertension, asthma, hemophilia, cancers, myopia) plus depression, anxiety and
PTSD driven by stress and trauma. Injuries come from crime, bullying, workplace accidents, road crashes, falls,
police and prison duty. Hospitals have beds, staff-driven care quality, triage by acuity and billing (insurance share
paid from the city treasury). Pregnancy lasts 270 days; births inherit one allele from each parent; conversations
(10 topics, personality compatibility) build friendships, dating, marriage, divorce and bereavement. Schools have
grades, teacher-ratio and teacher-skill driven learning, absences, bullying and dropouts.

**`justice.py`** - crimes have victims and real harm; reporting depends on severity and witnesses; police capacity
limits investigations; evidence accrues or decays; arrests, prosecution, public defenders or private lawyers, trials
scheduled by judge capacity (a measurable backlog), plea deals, fines, probation, prison with capacity, fights,
rehabilitation, parole and recidivism. Officers, detectives, judges, prosecutors and guards are ordinary employees of
public companies (`Company.public`), so staffing is a real constraint.

**`market.py`** - every company has products (price, unit cost, quality, capacity). Household spending is split into
categories and allocated by a logit of quality, brand, price and capacity; firms pay suppliers (supply chain); prices
react to sell-through. The influence index combines category share, revenue, employment, supply-chain dependents,
brand, market value and wages; per-category HHI shows concentration. See *View > Companies* and *Society & Law*.

## v8: pre-simulation, more realism, better GUI

**Pre-simulate.** `python pygame_frontend.py --presim 365` fast-forwards the city by a year before the window shows
it (the loading screen has a progress bar and ETA), or use *Sim > Pre-simulate 30 days / 1 year* at any time.
`SimulationEngine.fast_forward(days)` runs the full daily cycle (economy, hiring, health, births, crime, courts,
markets, migration, startups) plus four epidemic/traffic ticks and one event pass per day, and skips walking and
per-hour decisions. Citizens end each day asleep at home, so live play resumes cleanly at midnight. Cost is about
roughly 1.5 wall-clock seconds per simulated day at 12.5k citizens (a year takes about 9 minutes).

**Realism (`realism.py`, `life.py`).** Public holidays (workplaces and schools close); migration (people arrive when
jobs and housing are plentiful, leave when long-term unemployed or broke); business formation (ambitious people with
savings found small companies in vacant shops and offices); annual cost-of-living pay review; nutrition (food access
slows healing and wears health down).

**GUI.** Clickable tab strip above the side panel, header speed buttons with the current multiplier, event toasts
(births, weddings, arrests, new firms...), hover tooltips on citizens, mood/health bars in the citizen panel, a heat map
legend, and a pre-simulation progress overlay.

## Controls & UI

- **`F1`**: In-app help
- **`1`–`9`**: Toggle analytics & info panels
- **`H`**: Cycle heat-map overlays (including live traffic)
- **`V`**: Cycle citizen colour modes
- **`+` / `-`**: Adjust simulation speed (default `x60` real time, 1 simulated day = 24 minutes)
- **`Shift + Click`** or **`B`**: Inspect building cross-section (floors, assigned vs. present staff/residents, owner, rent roll, company value)
- **Mouse Wheel**: Zoom between `2 px` and `96 px` per tile

---

## Benchmarks & Testing

```bash
# Headless simulation benchmark (3 simulated days) with a per-phase time breakdown
python bench.py --days 3

# Small / default / stress-test (~25k citizens) populations
python bench.py --suite

# Determinism guard: record a bit-exact state hash, then prove a change kept it
python bench.py --days 2 --hash .bench/baseline.json
python bench.py --days 2 --expect .bench/baseline.json

# Headless render benchmark (300 frames)
python pygame_frontend.py --bench-frames 300

# Run unit tests
python -m unittest tests.test_simulation
```

### GPU (CUDA) routing

Route searches can run on an NVIDIA GPU through `numba.cuda` (`cuda_routing.py`).
The CUDA kernel is the same A* as the CPU kernel (same heap and tie-breaking), so
it returns bit-identical paths and never changes simulation results. Pick the mode
with the `RLSS_GPU` environment variable (inherited by the sim process) or
`bench.py --gpu`:

| Mode | Behaviour |
| :--- | :--- |
| `auto` (default) | Hybrid with a small probe share; switches to CPU-only if the GPU is slower |
| `hybrid` | Every A* batch is split between GPU and CPU, run concurrently, split adapted to measured speed |
| `cuda` | All A* searches on the GPU |
| `off` | CPU (Numba, all-but-one hardware threads) only |

On an RTX 3060 against a 12-thread Ryzen 5 the GPU is much slower for this
workload (batches of ~100 independent searches, each a serial priority-queue
walk), so `auto` settles on the CPU; the GPU path is there for machines with a
weak CPU and a strong GPU.
