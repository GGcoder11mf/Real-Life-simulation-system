# RLSS roadmap: the goal is to be best on every dimension

Standing goal from the project owner: **be the best in all terms**. Where this city simulator is
compared with Dwarf Fortress and RimWorld, close the gaps rather than pick one lane. This file is
planning only. Nothing here is scheduled or started.

## Where we stand

| Dimension | Dwarf Fortress | RimWorld | RLSS today |
|---|---|---|---|
| Per-agent depth | Enormous (body parts, memories, artifacts) | Medium | Medium: personality, genetics, memories, relationships, jobs, floors, mortgages |
| Emergent narrative | The defining feature | Strong (storyteller) | Partial: event engine and news, but events are mostly statistical, not dramatic |
| Simulation coupling | Very deep | Deep | Good and growing: economy, health, traffic, housing, law, weather interact |
| Scale (agents) | ~200 deep | dozens deep, hundreds shallow | ~12k citizens, shallower per person |
| Polish / iteration | Extreme | Extreme | Early |
| Performance engineering | Weak | Good | Strong: tiered routing, bench harness, parallel A*, process-isolated sim, fingerprints |

Strengths to protect: performance engineering, determinism (save/load tests), the benchmark harness.
Every item below must keep `bench.py` p50/p95 and the determinism tests green.

## Roadmap by dimension

### 1. Emergent narrative (largest gap)
- A **storyteller / drama director** that watches the whole city and schedules *causally linked* incidents (a layoff that
  leads to eviction, crime, a family split), with pacing so quiet stretches and crises alternate.
- **Incident chains and consequences** that persist: grudges, debts of honour, feuds, rivalries, mentors, scandals that
  change reputation and social graph structure, not just a stat tweak.
- A per-citizen and per-family **chronicle** (life-story timeline), plus a city-wide history/almanac that names notable
  people, firms, buildings and districts, and records "firsts" and records.
- Dramatic events with human scale and agency: accidents, fires, floods, strikes, elections, bankruptcies, weddings,
  funerals, protests. Players can follow a person or family across years.
- Make existing news items specific and relational (who, why, who is affected) instead of aggregate.

### 2. Per-agent depth
- Needs and mood model with explicit **thoughts/memories that feed mood** (and can be inspected), traits that interact.
- Body and health: injuries, chronic conditions, disability, ageing, pregnancy and childhood development, mental health.
- Skills with learning curves, aptitudes, specialisation, careers with promotion paths, side jobs, retirement.
- Values, beliefs and goals that drive decisions (ambition, faith, politics), not only scheduled routines.
- Possessions and homes: furnishing, vehicles, pets, heirlooms that pass down.

### 3. Simulation coupling
- Closed economic loops: wages -> consumption -> firm revenue -> hiring (started), plus supply chains between firms,
  banks and credit, rent and land value feedback, commuting costs shaping where people live.
- Public services with real capacity: hospitals with beds, schools with seats, police response, fire, utilities.
- Politics and governance: budget choices, taxes, policies that citizens react to.
- Environment: weather and seasons affecting energy, agriculture and health; pollution; disasters.

### 4. Scale
- Move hot per-citizen state to **struct-of-arrays + numba** (energy, stress, happiness, health, position) while keeping
  rich cold state on objects, so tick cost stops scaling with Python object count.
- **Level of detail**: full detail near the camera and for followed people, cheaper statistical updates elsewhere.
- Target tens of thousands of citizens at the same frame budget; sub-step cost kept low.
- Multiprocessing for independent districts if needed; keep determinism.

### 5. Polish and iteration
- Onboarding: tutorial, scenarios, a "follow a citizen / family" mode, bookmarks, search, filters.
- UI: better panels, tooltips, timelines and charts, accessibility (colour-blind palettes, font scaling), settings.
- Audio, ambient animation, better art direction; stable save format with versioned migrations.
- Modding hooks (data-driven jobs, events, buildings) and a documented API.
- Automated visual regression screenshots; user-facing error reporting.

### 6. Performance engineering (extend the lead)
- Keep the benchmark and fingerprint culture: add render benchmarks, memory budgets and per-system timing to CI.
- Profile-driven budgets per system per tick; background save with minimal stalls; incremental analytics.

## Principles
- Depth where the player looks, shallow elsewhere (LOD), never at the cost of determinism.
- Narrative must come from simulated causes. Events should be explainable from state, not random garnish.
- Each new system ships with tests, metrics in the analytics panel, and a bench check.
