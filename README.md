---
title: UAV IoT Data Collection
emoji: 🛸
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# UAV-Assisted IoT Sensor Data Collection Environment

An [OpenEnv](https://github.com/openenv-org/openenv) RL environment where an LLM-powered agent controls a UAV to navigate obstacles, visit rendezvous points, collect IoT sensor data, and return to base under energy constraints.

Built around the rotary-wing UAV data-harvesting problem studied in IEEE wireless-networks research (see [References](#references)).

## Overview

A UAV is deployed over a 2D area with **IoT sensor nodes** and **rectangular obstacles**. Sensors are clustered into **Rendezvous Points (RPs)** via greedy dominating-set. The UAV visits RPs, collects priority-weighted data, and must return to base before battery runs out.

The energy model uses the rotary-wing propulsion constants from Zeng & Zhang (2017): 17 J/m flight cost and 168.5 W hover power.

## Action Space

| Action | Effect |
|--------|--------|
| `move_N/NE/E/SE/S/SW/W/NW` | Move ~50 m in that direction (850 J) |
| `hover_collect` | Hover 10 s to collect data from RPs within 60 m (1685 J) |
| `return_base` | Fly to base via obstacle-aware shortest path |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `uav_x`, `uav_y` | float | UAV position (m) |
| `battery_pct` | float | Battery remaining (0–1) |
| `rps_visited` / `rps_total` | int | Collection progress |
| `data_collected` / `data_possible` | float | Priority-weighted progress |
| `sensors` | list[dict] | RP positions, priorities, distances |
| `obstacles` | list[dict] | Obstacle bounding boxes |
| `dist_to_base` | float | Distance to base station (m) |
| `energy_per_move` | float | Joules per 50 m move |
| `done`, `reward`, `message` | — | Step feedback |

## Tasks

| Task | Map | Obstacles | Sensors | Battery | Steps |
|------|-----|-----------|---------|---------|-------|
| `easy` | 400×300 m | 0 | 8 | 600 kJ | 80 |
| `medium` | 600×450 m | 3 | 14 | 480 kJ | 120 |
| `hard` | 800×600 m | 5 | 20 | 360 kJ | 160 |

### Scoring (0.0–1.0)

```
score = 0.35 × coverage + 0.25 × priority_ratio + 0.25 × energy_efficiency + 0.15 × safe_return
```

## Baseline Results

Tested against **Qwen2.5-Coder-32B-Instruct** via HuggingFace Inference API:

| Task | Steps | Score | Notes |
|------|-------|-------|-------|
| easy | 18 | 0.655 | All RPs visited, safe return |
| medium | — | — | (pending re-run) |
| hard | — | — | (pending re-run) |

Random agent baseline scores ~0.05 on easy (mostly crashes or times out).

## Quick Start

```bash
# Install
pip install -e ".[inference]"

# Start environment server
uvicorn uav_iot_env.server.app:app --port 8000

# Run inference (separate terminal)
export HF_TOKEN=your_token
python inference.py
```

### Docker

```bash
docker build -t uav-iot-env .
docker run -p 8000:8000 uav-iot-env
```

## Project Structure

```
├── inference.py               # LLM agent with [START]/[STEP]/[END] logs
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── README.md
└── uav_iot_env/
    ├── __init__.py
    ├── models.py              # Action, Observation, State
    ├── client.py              # EnvClient wrapper
    └── server/
        ├── app.py             # create_app() entrypoint
        ├── uav_iot_environment.py
        └── simulation/
            ├── __init__.py    # Task configs, constants
            ├── deployment.py  # Sensor/obstacle placement
            ├── energy.py      # Rotary-wing energy model
            ├── path_planning.py # Visibility-graph Dijkstra
            └── rp_selection.py  # Dominating-set clustering
```

## Reward Signals

| Event | Reward |
|-------|--------|
| Collect RP | +0.03 × priority |
| Move into obstacle | −0.05 |
| Path blocked | −0.03 |
| Hover, nothing nearby | −0.01 |
| Per-step time penalty | −0.005 |
| Battery depleted (crash) | −1.0 |
| Safe return to base | +0.2 + 0.5 × coverage |
| Stranded (can't reach base) | −0.5 |

## References

This environment is grounded in the UAV-assisted IoT data collection literature:

- **Zeng & Zhang (2017)** — "Energy-Efficient UAV Communication with Trajectory Optimization", *IEEE Trans. Wireless Commun.*, vol. 16, no. 6, pp. 3747–3760. Foundational rotary-wing energy model used in our simulation.
- **Bayerlein et al. (2021)** — "Multi-UAV Path Planning for Wireless Data Harvesting with Deep Reinforcement Learning", *IEEE Open J. Commun. Soc.*, vol. 2, pp. 1171–1187. DDQN with dual global-local map processing for multi-UAV data collection.
- **Wang et al. (2021)** — "Trajectory Design for UAV-Based IoT Data Collection: A Deep Reinforcement Learning Approach", *IEEE Internet Things J.* TD3 for 3D trajectories with imperfect CSI.
- **Chen et al. (2025)** — "LLM-Empowered Decision Transformer for UAV-Enabled Data Collection", *arXiv:2509.13934*. LLM-in-the-loop UAV control with decision transformers.

## License

MIT
