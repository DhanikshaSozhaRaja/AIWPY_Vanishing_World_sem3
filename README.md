# Vanishing World (Pygame Edition)

A grid-based AI puzzle-survival game built with Python and Pygame, organized with `main.py` as the Pygame game engine/UI entry point and each AI solver cleanly modularized under `algorithms/`.

## Project Structure
- `main.py` — Core game rules, progressive maze generator, Pygame renderer, and event loop
- `algorithms/bfs.py` — Breadth-First Search (`bfs_search`)
- `algorithms/dfs.py` — Depth-First Search (`dfs_search`)
- `algorithms/iddfs.py` — Iterative Deepening Depth-First Search (`iddfs_search`)
- `algorithms/greedy.py` — Greedy Best-First Search (`greedy_search`)
- `algorithms/astar.py` — A\* Search (`astar_search`)
- `algorithms/csp.py` — CSP with AC-3 Arc Consistency & MRV Backtracking (`csp_search`)
- `algorithms/comparison.py` — Algorithm benchmark cache and comparison runner
- `algorithms/common.py` — Shared search helpers and active game environment bridge

## Features
- Compact wall-interrupted labyrinths (`5x5` -> `6x6` -> `7x7`)
- Vanishing-tile rule & progressively tightening Collapse Move Limit (`MAX_MOVES`)
- Mandatory All-Crystals objective, Keys (`K`) & Locked Gates (`G`) with Multi-Key escalation (Lv 6+), and One-Way Arrow tiles (Lv 3+)
- Progressive hazards: Crumbling Deadline Tiles (`T#`, Lv 4+) and Alternating Spike Traps (Lv 5+)
- Guaranteed solvability verification (`has_solution()`)
- Six modular AI solvers plus Tactical A\* Hints (`H`) and live Algorithm Comparison + Nodes Explored Bar Chart (`C`)

## Run
```bash
pip install -r requirements.txt
python main.py
```
