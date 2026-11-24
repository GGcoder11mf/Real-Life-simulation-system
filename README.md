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
# Headless simulation benchmark (3 simulated days)
python bench.py --days 3

# Headless render benchmark (300 frames)
python pygame_frontend.py --bench-frames 300

# Run unit tests
python -m unittest tests.test_simulation
```
