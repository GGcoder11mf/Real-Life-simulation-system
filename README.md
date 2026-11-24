# RLSS (enhanced)

Autonomous city / life simulator. This copy is the performance- and realism-focused
branch of the original `RLSS` project.

## Run

    python pygame_frontend.py                 # large city (~12k citizens, 224x112 tiles)
    python pygame_frontend.py --preset small  # quick start
    python pygame_frontend.py --load life_sim_save.pkl
    python bench.py --days 3                  # headless simulation benchmark
    python pygame_frontend.py --bench-frames 300   # headless render benchmark
    python -m unittest tests.test_simulation

## What changed

**Frame rate.** The simulation now runs in its own process (`sim_host.py`); the
renderer only draws the newest snapshot, so FPS no longer depends on tick cost or
population (the old in-process thread + lock froze rendering for the length of
every tick). The map is pre-rendered per zoom level and each frame is one blit;
citizens are drawn from numpy arrays with interpolated motion; text is cached.

**Scale.** 224x112 tiles (about 5x finer grid, tile speeds scale with it) and about
12k citizens (~12x the original 958). World generation is vectorised (init 300s -> 5s),
batched numba A* routing runs in parallel, citizen hot paths were trimmed, deaths
are batched, saves use fast gzip.

**New metrics** (`analytics.py`): Gini, poverty, median net worth, top-10% share,
dependency ratio, age pyramid, sector employment, commute time, rent burden,
housing occupancy, GDP per capita, birth/death rates, crime per 1k, and more, with
history charts in the UI.

**New realism** (`dynamics.py` and others): SEIR epidemics with hospitals,
vaccination and waning immunity; rush-hour traffic that slows vehicles; municipal
budget feeding service levels; air quality; inflation; weekends and weekend-only
sectors; retirement and pensions; Markov weather with seasons and a daily
temperature cycle; day/night and weather rendering; heat-map overlays.

Controls: F1 in-app help. H heat map, V citizen colours, 1-9 panels.

## Buildings (`buildings.py`)

Every building has floors with functions and capacities. Houses are 1-3 storeys, apartment
blocks grow with their unit count (lobby + unit floors), and commercial buildings are sized
from their company: headcount sets floors needed, and office tower height follows the
company's market value (re-evaluated weekly, +/-1 floor at a time). Employees get a work
floor when hired (offices spread staff over all floors; hospitals/factories fill floor by
floor); citizens carry a live `floor` (-1 outside, 0 ground) updated by their schedule, which
now includes a midday work break and weekends.

Housing has owners: wealthy citizens become private landlords (heavy-tailed holdings), some
buildings are corporate-owned, and part of the population owns its unit. Monthly rent flows
tenant -> owner (22% to upkeep), owners pay property tax to the city treasury, and estates pass
to a spouse/child (or a corporate landlord) on death.

Inspect any building with Shift+click or `B` (selected citizen's building): floor-by-floor
cross-section with assigned vs present staff/residents, owner, rent roll, company value.

## Movement, traffic and rendering (v3)

**Continuous movement.** The world is on a physical scale (1 tile = 20 m). The simulation
advances in 2-minute sub-steps: the full decision logic (schedules, needs, economy) runs at
the top of each hour, but citizens walk/drive their route every sub-step at real speeds
(walk 4.8 km/h, bike 14, bus 17, car 28, taxi 30, slowed by congestion and weather).
Departure times are staggered, walking and cycling cost energy (and tired people slow down),
and long trips use bus/taxi. Mode choice depends on distance, vehicle ownership, weather,
time of day and cash; vehicles route on a separate road-preferring cost grid.

**Rendering.** There are no image textures any more. The world is painted procedurally in
cached chunks: asphalt with lane markings, sidewalks, zebra crossings, signalised
intersections, metro line with trains, buildings with facades/windows/roofs/shadows by type
and floors, trees, shorelines. Citizens are drawn as animated figures on the sidewalks,
cars/taxis/buses as vehicles in their lane, and only people who are outdoors are drawn.
Night adds street-lamp, lit-window and headlight glows. Zoom ranges from 2 to 96 px per tile.
Playback is buffered and interpolated along the real waypoints, so motion is smooth.

Default sim speed is x60 real time (one simulated day = 24 min); use + / - to change it.
Heat maps: H (now includes live traffic). `--tick-ms N` sets wall ms per 2 simulated minutes.

## City layout (v4)

Roads are a hierarchy: three arterials plus a street lattice whose block size depends on the
area (dense downtown, larger suburban and industrial blocks, none in open country). Buildings
are placed on lots that front a street, never on roads, and large types span several tiles
(factories 3x2, colleges and prisons 3x3, hospitals/courts 2x2, offices up to 2x2). Housing is
balanced against jobs so vacancy stays plausible. Each building is drawn from its footprint
with a per-building palette, roof style, yard, driveway and street trees. Saves made before this
version keep their old layout.

## Economy integrity (v5)

* Every company gets a unique ticker (`WorldMap._make_stock_symbol`, with `ensure_unique_symbols`
  repairing older saves), so each company has its own traded stock.
* Vacancies are sized per company (`CompanySystem.vacancy_target`): proportional to headcount and
  shaped by demand, expansion appetite and productivity. Big firms advertise several roles, some
  firms advertise none.
* Inequality is not endowed at generation. Starting wealth follows a life-cycle profile (the young
  have little), nobody owns property on day 0, and all housing starts corporate-owned. Home
  ownership and private landlords emerge through a monthly property market (deposits, affordability
  checks, mortgages), driven by savings and income. Consumption spending is the sink that stops
  savings growing without bound and becomes revenue for consumer-facing firms. Housing prices now
  track inflation (they used to deflate ~30%/yr).

## Traffic signals and housing coverage (v6)

Signalised junctions are part of the simulation (`WorldMap._compute_signals`, `signal_state`). A shared
120 s timing cycle drives both the vehicles and the lights drawn by the renderer, per junction and axis.
Vehicles are advanced second by second inside each 2-minute sub-step (`Population._drive_vehicle`):
they stop at the stop line on red or yellow and move off on green, and the renderer plays back the
timestamps so cars hold and queue. Housing capacity is counted per unit (1.9 people), and anyone left
without a unit joins a household with room, so no citizen is homeless at generation.
