# UAV IoT Data Collection

Control a UAV to collect data from IoT sensor clusters and return to base.

## Tabs

- **Custom** — interactive visual map with drone, sensors, obstacles, and controls
- **Playground** — raw JSON interface for API testing and AI agent development

## Custom Tab

1. Select task difficulty (easy / medium / hard / expert) and click **Reset**
2. Use the **direction buttons** to fly the UAV toward sensor clusters
3. Click **Collect** when near a cluster (within 60 m) to gather data
4. Click **Return to Base** before battery runs out
5. Or click **Auto Play** to watch a greedy agent complete the mission

## Actions (Playground / API)

| Action | What it does |
|--------|-------------|
| `move_N` `move_NE` `move_E` `move_SE` `move_S` `move_SW` `move_W` `move_NW` | Fly ~50 m in that direction (costs 850 J) |
| `hover_collect` | Hover for 10 s, collects data from RPs within 60 m (costs 1685 J) |
| `return_base` | Autopilot back to base (path avoids obstacles) |

## Wind

Medium, hard, and expert tasks have stochastic wind. Each step, the wind drifts your position off course. Check `wind_x` and `wind_y` in the observation to anticipate and compensate. Easy has no wind.

## Tips

- Check `sensors` in the observation for RP locations and distances
- Prioritise high-priority RPs (higher `priority` = more score)
- Watch `battery_pct` — if it hits 0 the UAV crashes (−1.0 reward)
- Use `return_base` when you've collected enough or battery is low
- Obstacles block movement; the UAV cannot fly through them
- On windy tasks, compensate for drift near obstacles and collection points

## Scoring

```
score = 0.35 × coverage + 0.25 × priority_ratio
      + 0.25 × energy_efficiency + 0.15 × safe_return
```

Energy efficiency rewards covering more RPs per unit energy spent.

```
score = 0.35 × coverage + 0.25 × priority_ratio
      + 0.25 × energy_efficiency + 0.15 × safe_return
```
