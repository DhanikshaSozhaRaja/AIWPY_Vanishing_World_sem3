"""Vanishing World — Standalone AI Puzzle-Survival Game in Python & Pygame."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
import sys
import time
from typing import (
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

import pygame

from algorithms import (
    CSP,
    SearchFn,
    SearchLimitReached,
    _build_csp,
    astar_heuristic,
    astar_search,
    bfs_search,
    clear_search_result_cache,
    compare_algorithms,
    csp_search,
    dfs_search,
    get_comparison_plan,
    get_search_node_budget,
    greedy_heuristic,
    greedy_search,
    iddfs_search,
    record_search_result,
    register_game_env,
)

# ---------------------------------------------------------------------------
# WINDOW, LAYOUT & COLOR CONSTANTS
# ---------------------------------------------------------------------------

WIDTH = 900
HEIGHT = 600

MAX_TILE_SIZE = 65
BOARD_Y = 130
BOARD_BOTTOM_LIMIT = 455
BOARD_SIDE_MARGIN = 55

INSTRUCTION_MAX_SCROLL = 760
MENU_BAR_HEIGHT = 27
MENU_ITEMS = ("Game", "AI Tools", "Difficulty", "Help")
AI_MOVE_DELAY_MS = 350
MAX_HINTS_PER_LEVEL = 3

BACKGROUND = (25, 30, 45)
TILE_COLOR = (65, 75, 95)
GRID_COLOR = (110, 120, 140)
PLAYER_COLOR = (50, 210, 100)
CRYSTAL_COLOR = (60, 200, 255)
EXIT_COLOR = (240, 190, 50)
WALL_COLOR = (15, 18, 28)
VANISHED_COLOR = (35, 38, 48)
WHITE = (245, 245, 245)
RED = (240, 75, 75)
PANEL_COLOR = (36, 43, 61)
PANEL_LINE = (85, 98, 122)
MUTED_TEXT = (180, 190, 207)
BUTTON_COLOR = (50, 150, 210)
BUTTON_HOVER_COLOR = (70, 175, 235)
MENU_BG = (31, 38, 55)
MENU_HOVER = (65, 82, 110)
MENU_DISABLED = (115, 122, 137)
DISABLED_FILL = (38, 43, 55)
ROW_STRIPE = (43, 51, 71)
HINT_OUTLINE = (255, 90, 210)
BEST_GREEN = (120, 235, 160)
COMP_BG = (20, 25, 40)
COMP_PROMPT = (255, 220, 100)
GOLD = (255, 200, 45)
EMPTY_STAR = (100, 105, 120)
KEY_COLOR = (255, 155, 45)
GATE_FILL_LOCKED = (82, 34, 46)
GATE_FILL_OPEN = (48, 86, 78)
GATE_BORDER_LOCKED = (235, 95, 110)
GATE_TEXT_LOCKED = (255, 175, 185)
CRUMBLE_FILL = (76, 56, 44)
CRUMBLE_URGENT = (155, 62, 48)
CRUMBLE_CRACK = (240, 175, 80)
CRUMBLE_DEAD = (30, 28, 36)
SPIKE_ARMED_BG = (86, 30, 42)
SPIKE_ARMED_FG = (255, 86, 108)
SPIKE_SAFE_BG = (42, 54, 70)
SPIKE_SAFE_FG = (118, 136, 160)

CLOSE_BUTTON_RECT = (WIDTH // 2 - 80, HEIGHT - 55, 160, 40)


# ---------------------------------------------------------------------------
# CORE GAME STATE & RULES
# ---------------------------------------------------------------------------

Position = Tuple[int, int]
Direction = str


class GameState(NamedTuple):
    player: Position
    vanished: FrozenSet[Position]
    crystals: FrozenSet[Position]
    keys: FrozenSet[Position]
    status: str  # "playing" | "won" | "lost"


ROWS = 5
COLS = 5

START_PLAYER: Position = (0, 0)
START_CRYSTALS: FrozenSet[Position] = frozenset({(0, 4), (2, 2), (4, 1)})
START_KEYS: FrozenSet[Position] = frozenset()
EXIT_TILE: Position = (4, 4)

WALLS: Set[Position] = set()
LOCKED_GATES: Set[Position] = set()
ONE_WAY_TILES: Dict[Position, Direction] = {}
CRUMBLING_TILES: Dict[Position, int] = {}
SPIKE_TILES: Dict[Position, int] = {}
SPECIAL_TILES: Set[Position] = set()
GUARANTEED_PATH: List[Position] = []
MAX_MOVES: int = 0

LEVEL_NUMBER: int = 1
CRYSTAL_COUNT: int = 3
DIFFICULTY: str = "MEDIUM"

# (min_level, max_level, board_size, wall_min, wall_max, crystal_count)
LEVEL_SETTINGS: Tuple[Tuple[int, int, int, int, int, int], ...] = (
    (1, 2, 5, 4, 6, 3),
    (3, 5, 6, 6, 9, 4),
    (6, 8, 7, 9, 13, 5),
    (9, 999999, 7, 10, 15, 6),
)

DIFFICULTY_SETTINGS: Dict[str, Dict[str, float]] = {
    "EASY": {"wall_factor": 0.80, "path_factor": 0.88},
    "MEDIUM": {"wall_factor": 1.00, "path_factor": 1.00},
    "HARD": {"wall_factor": 1.25, "path_factor": 1.12},
}

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

DIRECTION_LIST: Tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")


def get_level_settings(level: int) -> Tuple[int, int, int, int]:
    for first, last, size, wall_min, wall_max, crystals in LEVEL_SETTINGS:
        if first <= level <= last:
            return size, wall_min, wall_max, crystals
    return 7, 10, 15, 6


def set_difficulty(difficulty: str) -> bool:
    global DIFFICULTY
    upper = difficulty.upper()
    if upper in DIFFICULTY_SETTINGS:
        DIFFICULTY = upper
        return True
    return False


def configure_board(level: int) -> None:
    global ROWS, COLS, CRYSTAL_COUNT
    size, _, _, crystals = get_level_settings(level)
    ROWS = size
    COLS = size
    CRYSTAL_COUNT = crystals


def create_initial_state() -> GameState:
    return GameState(
        player=START_PLAYER,
        vanished=frozenset(),
        crystals=frozenset(START_CRYSTALS),
        keys=frozenset(START_KEYS),
        status="playing",
    )


def inside_board(position: Position) -> bool:
    row, col = position
    return 0 <= row < ROWS and 0 <= col < COLS


def get_legal_moves(state: GameState) -> List[Direction]:
    if state.status != "playing":
        return []

    forced_dir = ONE_WAY_TILES.get(state.player)
    candidate_dirs = (forced_dir,) if forced_dir else DIRECTION_LIST
    next_step = len(state.vanished) + 1

    legal: List[Direction] = []
    pr, pc = state.player
    for direction in candidate_dirs:
        dr, dc = DIRECTIONS[direction]
        dest = (pr + dr, pc + dc)
        if not inside_board(dest):
            continue
        if dest in WALLS or dest in state.vanished:
            continue
        if dest in LOCKED_GATES and len(state.keys) > 0:
            continue
        deadline = CRUMBLING_TILES.get(dest)
        if deadline is not None and next_step > deadline:
            continue
        spike_parity = SPIKE_TILES.get(dest)
        if spike_parity is not None and (next_step % 2) == spike_parity:
            continue
        legal.append(direction)

    return legal


def is_goal_state(state: GameState) -> bool:
    return (
        state.status == "won"
        and state.player == EXIT_TILE
        and len(state.crystals) == 0
    )


def move_state(state: GameState, direction: Direction) -> Optional[GameState]:
    if state.status != "playing":
        return state

    if direction not in get_legal_moves(state):
        return None

    dr, dc = DIRECTIONS[direction]
    new_player: Position = (state.player[0] + dr, state.player[1] + dc)
    new_vanished = state.vanished | {state.player}
    new_crystals = state.crystals - {new_player}
    new_keys = state.keys - {new_player}

    new_status = "playing"
    if new_player == EXIT_TILE:
        new_status = "won" if len(new_crystals) == 0 else "lost"

    if new_status == "playing" and MAX_MOVES > 0 and len(new_vanished) >= MAX_MOVES:
        new_status = "lost"

    next_state = GameState(
        player=new_player,
        vanished=new_vanished,
        crystals=new_crystals,
        keys=new_keys,
        status=new_status,
    )

    if new_status == "playing" and len(get_legal_moves(next_state)) == 0:
        return GameState(
            player=new_player,
            vanished=new_vanished,
            crystals=new_crystals,
            keys=new_keys,
            status="lost",
        )

    return next_state


# ---------------------------------------------------------------------------
# MAZE & CHALLENGING LEVEL GENERATOR
# ---------------------------------------------------------------------------

def create_fallback_path(target_length: int) -> List[Position]:
    full_path: List[Position] = []
    for row in range(ROWS):
        cols = range(COLS) if row % 2 == 0 else range(COLS - 1, -1, -1)
        for col in cols:
            full_path.append((row, col))
    return full_path[:target_length]


def manhattan_distance(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _shortest_grid_distance(
    start: Position, goal: Position, walls: Set[Position]
) -> float:
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        pos, dist = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not inside_board(nxt) or nxt in walls or nxt in seen:
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return float("inf")


def _reachable_count_from_start(walls: Set[Position]) -> int:
    if START_PLAYER in walls:
        return 0
    queue = deque([START_PLAYER])
    seen = {START_PLAYER}
    while queue:
        cur = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if inside_board(nxt) and nxt not in walls and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def _creates_two_by_two_wall_block(
    row: int, col: int, walls: Set[Position]
) -> bool:
    quads = (
        ((-1, -1), (-1, 0), (0, -1)),
        ((-1, 0), (-1, 1), (0, 1)),
        ((0, -1), (1, -1), (1, 0)),
        ((0, 1), (1, 0), (1, 1)),
    )
    for quad in quads:
        if all(
            0 <= row + dr < ROWS
            and 0 <= col + dc < COLS
            and (row + dr, col + dc) in walls
            for dr, dc in quad
        ):
            return True
    return False


def _generate_interrupting_walls(
    target_wall_count: int, min_reachable: int
) -> Set[Position]:
    walls: Set[Position] = set()
    candidates_pool = [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if (r, c) != START_PLAYER
    ]

    for _ in range(target_wall_count * 4):
        if len(walls) >= target_wall_count:
            break

        weighted: List[Tuple[Position, int]] = []
        for r, c in candidates_pool:
            pos = (r, c)
            if pos in walls or _creates_two_by_two_wall_block(r, c, walls):
                continue

            weight = 1
            if 0 < r < ROWS - 1 and 0 < c < COLS - 1:
                weight += 4

            adj_walls = sum(
                1
                for dr, dc in DIRECTIONS.values()
                if (r + dr, c + dc) in walls
            )
            if adj_walls == 1:
                weight += 3
            elif adj_walls >= 2:
                weight = max(1, weight - 2)

            if abs(r - c) <= 1 and r + c >= 2:
                weight += 2

            weighted.append((pos, weight))

        if not weighted:
            break

        total_weight = sum(w for _, w in weighted)
        pick = random.random() * total_weight
        chosen = weighted[0][0]
        for pos, w in weighted:
            pick -= w
            if pick <= 0:
                chosen = pos
                break

        walls.add(chosen)
        if _reachable_count_from_start(walls) < min_reachable:
            walls.remove(chosen)

    return walls


def _find_winding_path_with_walls(
    walls: Set[Position], target_length: int
) -> Optional[List[Position]]:
    path: List[Position] = [START_PLAYER]
    visited: Set[Position] = {START_PLAYER}
    expansions = 0
    expansion_limit = 1800

    def candidates_for(
        position: Position, prev_position: Optional[Position]
    ) -> List[Position]:
        r, c = position
        cands = []
        for dr, dc in DIRECTIONS.values():
            cand = (r + dr, c + dc)
            if inside_board(cand) and cand not in walls and cand not in visited:
                cands.append(cand)
        random.shuffle(cands)

        def score(cand: Position) -> float:
            cr, cc = cand
            onward = 0
            adj_walls = 0
            for dr, dc in DIRECTIONS.values():
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    continue
                nb = (nr, nc)
                if nb in walls:
                    adj_walls += 1
                elif nb not in visited:
                    onward += 1

            turn_bonus = 0.0
            if prev_position is not None:
                p_dr = r - prev_position[0]
                p_dc = c - prev_position[1]
                c_dr = cr - r
                c_dc = cc - c
                if (p_dr, p_dc) != (c_dr, c_dc):
                    turn_bonus = -1.2

            return onward * 1.1 - adj_walls * 0.7 + turn_bonus + random.random() * 0.9

        cands.sort(key=score)
        return cands

    stack: List[Dict[str, object]] = [
        {"pos": START_PLAYER, "cands": candidates_for(START_PLAYER, None), "idx": 0}
    ]

    while stack and expansions < expansion_limit:
        if len(path) >= target_length:
            return list(path)

        top = stack[-1]
        cands = top["cands"]  # type: ignore[assignment]
        idx = top["idx"]  # type: ignore[assignment]
        if idx >= len(cands):  # type: ignore[arg-type]
            stack.pop()
            removed = path.pop()
            visited.discard(removed)
            continue

        cand = cands[idx]  # type: ignore[index]
        top["idx"] = idx + 1  # type: ignore[operator]
        if cand in visited:
            continue

        prev_pos = path[-1] if path else None
        path.append(cand)
        visited.add(cand)
        expansions += 1
        stack.append({"pos": cand, "cands": candidates_for(cand, prev_pos), "idx": 0})

    return None


def _pick_distributed_crystals(
    path: List[Position], crystal_count: int
) -> Set[Position]:
    middle = path[1:-1]
    if len(middle) <= crystal_count:
        return set(middle[:crystal_count])

    chosen: Set[Position] = set()
    slice_size = len(middle) / crystal_count
    for i in range(crystal_count):
        start_idx = int(i * slice_size)
        end_idx = max(start_idx, int((i + 1) * slice_size) - 1)
        available = [p for p in middle[start_idx : end_idx + 1] if p not in chosen]
        if available:
            chosen.add(random.choice(available))

    if len(chosen) < crystal_count:
        remaining = [p for p in middle if p not in chosen]
        chosen.update(random.sample(remaining, crystal_count - len(chosen)))

    return chosen


def _score_puzzle_layout(
    path: List[Position],
    walls: Set[Position],
    crystals: Set[Position],
    exit_pos: Position,
) -> int:
    score = 0
    path_index = {pos: idx for idx, pos in enumerate(path)}

    bfs_to_exit = _shortest_grid_distance(START_PLAYER, exit_pos, walls)
    if bfs_to_exit == float("inf"):
        return -999999

    direct_to_exit = manhattan_distance(START_PLAYER, exit_pos)
    if direct_to_exit < max(3, int(ROWS * 0.65)):
        score -= 120
    if bfs_to_exit < ROWS + 1:
        score -= 60

    score += int(max(0.0, bfs_to_exit - direct_to_exit) * 26)
    score += int(min(bfs_to_exit, ROWS * 2) * 4)

    objectives: List[Position] = [START_PLAYER]
    objectives.extend(p for p in path[1:-1] if p in crystals)
    objectives.append(exit_pos)

    for i in range(len(objectives) - 1):
        a, b = objectives[i], objectives[i + 1]
        d_man = manhattan_distance(a, b)
        d_bfs = _shortest_grid_distance(a, b, walls)
        if d_bfs > d_man:
            score += int((d_bfs - d_man) * 14)

        min_r, max_r = min(a[0], b[0]), max(a[0], b[0])
        min_c, max_c = min(a[1], b[1]), max(a[1], b[1])
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) in walls:
                    score += 6

    for wr, wc in walls:
        touched = [
            path_index[(wr + dr, wc + dc)]
            for dr, dc in DIRECTIONS.values()
            if (wr + dr, wc + dc) in path_index
        ]
        if len(touched) >= 2 and (max(touched) - min(touched)) >= 3:
            score += 12

    decoy_branches = 0
    for pr, pc in path[:-1]:
        if any(
            0 <= pr + dr < ROWS
            and 0 <= pc + dc < COLS
            and (pr + dr, pc + dc) not in walls
            and (pr + dr, pc + dc) not in path_index
            for dr, dc in DIRECTIONS.values()
        ):
            decoy_branches += 1
    score += min(decoy_branches, int(len(path) * 0.7)) * 5

    turns = 0
    for i in range(2, len(path)):
        dr1 = path[i - 1][0] - path[i - 2][0]
        dc1 = path[i - 1][1] - path[i - 2][1]
        dr2 = path[i][0] - path[i - 1][0]
        dc2 = path[i][1] - path[i - 1][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    score += turns * 3

    return score


def generate_walls(path: List[Position]) -> Set[Position]:
    path_set = set(path)
    available = [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if (r, c) not in path_set
    ]
    _, wall_min, wall_max, _ = get_level_settings(LEVEL_NUMBER)
    factor = DIFFICULTY_SETTINGS[DIFFICULTY]["wall_factor"]
    scaled_min = max(0, round(wall_min * factor))
    scaled_max = max(scaled_min, round(wall_max * factor))
    wall_count = min(random.randint(scaled_min, scaled_max), len(available))
    return set(random.sample(available, wall_count))


def create_random_path() -> List[Position]:
    settings = DIFFICULTY_SETTINGS[DIFFICULTY]
    total_tiles = ROWS * COLS
    _, _, _, crystal_count = get_level_settings(LEVEL_NUMBER)
    target_len = int(total_tiles * 0.56 * settings["path_factor"])
    target_len = max(target_len, crystal_count + 5)
    target_len = min(target_len, total_tiles - 2)

    for _ in range(12):
        path = _find_winding_path_with_walls(set(), target_len)
        if path is not None:
            return path
    return create_fallback_path(min(target_len, total_tiles))


def has_solution() -> bool:
    if not GUARANTEED_PATH:
        return False

    state = create_initial_state()
    for i in range(len(GUARANTEED_PATH) - 1):
        current = GUARANTEED_PATH[i]
        dest = GUARANTEED_PATH[i + 1]
        if state.player != current:
            return False

        matched: Optional[Direction] = None
        for name, (dr, dc) in DIRECTIONS.items():
            if (current[0] + dr, current[1] + dc) == dest:
                matched = name
                break

        if matched is None or matched not in get_legal_moves(state):
            return False

        nxt = move_state(state, matched)
        if nxt is None or nxt.status == "lost":
            return False
        state = nxt

    return is_goal_state(state)


def generate_new_level(advance_level: bool = True) -> GameState:
    global LEVEL_NUMBER, CRYSTAL_COUNT, START_CRYSTALS, START_KEYS
    global EXIT_TILE, GUARANTEED_PATH, WALLS, LOCKED_GATES
    global ONE_WAY_TILES, CRUMBLING_TILES, SPIKE_TILES, SPECIAL_TILES, MAX_MOVES

    target_level = LEVEL_NUMBER + 1 if advance_level else LEVEL_NUMBER
    configure_board(target_level)
    _, wall_min, wall_max, crystal_count = get_level_settings(target_level)
    CRYSTAL_COUNT = crystal_count

    settings = DIFFICULTY_SETTINGS[DIFFICULTY]
    total_tiles = ROWS * COLS

    scaled_wall_min = max(2, round(wall_min * settings["wall_factor"]))
    scaled_wall_max = max(
        scaled_wall_min,
        min(int(total_tiles * 0.34), round(wall_max * settings["wall_factor"])),
    )

    best: Optional[Tuple[int, List[Position], Set[Position], Set[Position], Position]] = None

    for _ in range(24):
        wall_count = random.randint(scaled_wall_min, scaled_wall_max)
        walkable_budget = total_tiles - wall_count
        min_path_len = max(crystal_count + 4, int(walkable_budget * 0.52))
        max_path_len = min(
            walkable_budget - 2,
            max(min_path_len + 2, int(walkable_budget * 0.72 * settings["path_factor"])),
        )
        target_path_len = random.randint(
            min(min_path_len, max_path_len), max(min_path_len, max_path_len)
        )

        walls = _generate_interrupting_walls(
            wall_count, max(target_path_len + 2, int(walkable_budget * 0.85))
        )
        path = _find_winding_path_with_walls(walls, target_path_len)
        if path is None or len(path) < crystal_count + 2:
            continue

        exit_pos = path[-1]
        crystals = _pick_distributed_crystals(path, crystal_count)
        if len(crystals) < crystal_count:
            continue

        score = _score_puzzle_layout(path, walls, crystals, exit_pos)
        if best is None or score > best[0]:
            best = (score, path, walls, crystals, exit_pos)

    if best is not None:
        _, path, new_walls, new_crystals, new_exit = best
    else:
        path = create_random_path()
        if len(path) < CRYSTAL_COUNT + 2:
            path = create_fallback_path(CRYSTAL_COUNT + 2)
        new_exit = path[-1]
        new_crystals = _pick_distributed_crystals(path, CRYSTAL_COUNT)
        new_walls = generate_walls(path)

    special_cands = [p for p in path[1:-1] if p not in new_crystals]
    new_special = {random.choice(special_cands)} if special_cands else set()

    new_keys: Set[Position] = set()
    new_gates: Set[Position] = set()
    if target_level >= 2 and len(path) >= 6:
        gate_cands: List[Tuple[int, int]] = []
        min_gate_idx = max(3, int(len(path) * 0.52))
        for idx in range(min_gate_idx, len(path) - 1):
            if path[idx] in new_crystals:
                continue
            gr, gc = path[idx]
            adj_walls = sum(
                1
                for dr, dc in DIRECTIONS.values()
                if not (0 <= gr + dr < ROWS and 0 <= gc + dc < COLS)
                or (gr + dr, gc + dc) in new_walls
            )
            gate_cands.append((adj_walls, idx))

        if gate_cands:
            gate_cands.sort(reverse=True)
            top_slice = gate_cands[: min(3, len(gate_cands))]
            _, gate_idx = random.choice(top_slice)
            key_indices = [
                idx for idx in range(1, gate_idx - 1) if path[idx] not in new_crystals
            ]
            if key_indices:
                desired_keys = 2 if target_level >= 6 and len(key_indices) >= 2 else 1
                limit = max(desired_keys, (len(key_indices) * 3 + 3) // 4)
                chosen_key_indices = random.sample(
                    key_indices[: min(len(key_indices), limit)], desired_keys
                )
                for key_idx in chosen_key_indices:
                    new_keys.add(path[key_idx])
                new_gates.add(path[gate_idx])

    new_one_way: Dict[Position, Direction] = {}
    if target_level >= 3 and len(path) >= 6:
        desired_arrows = 3 if target_level >= 8 else (2 if target_level >= 5 else 1)
        arrow_indices = [
            idx
            for idx in range(1, len(path) - 1)
            if path[idx] not in new_crystals
            and path[idx] not in new_keys
            and path[idx] not in new_gates
        ]
        random.shuffle(arrow_indices)
        for idx in arrow_indices:
            if len(new_one_way) >= desired_arrows:
                break
            cur, nxt = path[idx], path[idx + 1]
            diff = (nxt[0] - cur[0], nxt[1] - cur[1])
            for dir_name, vec in DIRECTIONS.items():
                if vec == diff:
                    new_one_way[cur] = dir_name
                    break

    new_crumbling: Dict[Position, int] = {}
    if target_level >= 4 and len(path) >= 7:
        desired_crumbles = 3 if target_level >= 8 else (2 if target_level >= 6 else 1)
        crumble_indices = [
            idx
            for idx in range(2, len(path) - 1)
            if path[idx] not in new_crystals
            and path[idx] not in new_keys
            and path[idx] not in new_gates
            and path[idx] not in new_one_way
        ]
        random.shuffle(crumble_indices)
        max_slack = 3 if DIFFICULTY == "EASY" else (2 if DIFFICULTY == "MEDIUM" else 1)
        if target_level >= 7 and max_slack > 1:
            max_slack -= 1
        min_slack = 0 if (DIFFICULTY == "HARD" or target_level >= 8) else 1
        for idx in crumble_indices[:desired_crumbles]:
            slack = random.randint(min(min_slack, max_slack), max_slack)
            new_crumbling[path[idx]] = idx + slack

    new_spikes: Dict[Position, int] = {}
    if target_level >= 5 and len(path) >= 7:
        desired_spikes = 3 if target_level >= 8 else (2 if target_level >= 6 else 1)
        spike_indices = [
            idx
            for idx in range(1, len(path) - 1)
            if path[idx] not in new_crystals
            and path[idx] not in new_keys
            and path[idx] not in new_gates
            and path[idx] not in new_one_way
            and path[idx] not in new_crumbling
        ]
        random.shuffle(spike_indices)
        for idx in spike_indices[:desired_spikes]:
            # Safe on GUARANTEED_PATH step idx -> armed on opposite parity
            new_spikes[path[idx]] = (idx + 1) % 2

        if target_level >= 6:
            path_set = set(path)
            off_cands = [
                (r, c)
                for r in range(ROWS)
                for c in range(COLS)
                if (r, c) not in path_set
                and (r, c) not in new_walls
                and any(
                    (r + dr, c + dc) in path_set for dr, dc in DIRECTIONS.values()
                )
            ]
            if off_cands:
                extra_count = 2 if target_level >= 9 else 1
                for pos in random.sample(off_cands, min(extra_count, len(off_cands))):
                    new_spikes[pos] = random.choice((0, 1))

    if DIFFICULTY == "EASY":
        move_buffer = max(3, 6 - (target_level // 4))
    elif DIFFICULTY == "MEDIUM":
        move_buffer = max(2, 4 - (target_level // 4))
    else:
        move_buffer = max(1, 2 - (target_level // 5))
    computed_max_moves = len(path) - 1 + move_buffer

    START_CRYSTALS = frozenset(new_crystals)
    START_KEYS = frozenset(new_keys)
    EXIT_TILE = new_exit
    GUARANTEED_PATH = list(path)
    WALLS = new_walls
    LOCKED_GATES = new_gates
    ONE_WAY_TILES = new_one_way
    CRUMBLING_TILES = new_crumbling
    SPIKE_TILES = new_spikes
    SPECIAL_TILES = new_special
    MAX_MOVES = computed_max_moves
    LEVEL_NUMBER = target_level

    if not has_solution():
        GUARANTEED_PATH = create_fallback_path(max(CRYSTAL_COUNT + 2, len(path)))
        EXIT_TILE = GUARANTEED_PATH[-1]
        START_CRYSTALS = frozenset(
            random.sample(GUARANTEED_PATH[1:-1], CRYSTAL_COUNT)
        )
        START_KEYS = frozenset()
        LOCKED_GATES = set()
        ONE_WAY_TILES = {}
        CRUMBLING_TILES = {}
        SPIKE_TILES = {}
        MAX_MOVES = len(GUARANTEED_PATH) - 1 + move_buffer
        WALLS = generate_walls(GUARANTEED_PATH)
        SPECIAL_TILES = set()

        if not has_solution():
            WALLS = set()
            if not has_solution():
                raise RuntimeError("Could not generate a solvable level.")

    return create_initial_state()


register_game_env(sys.modules[__name__])


# ---------------------------------------------------------------------------
# PYGAME DRAWING HELPERS
# ---------------------------------------------------------------------------

RectTuple = Tuple[int, int, int, int]


def point_in_rect(px: int, py: int, rect: Optional[RectTuple]) -> bool:
    if rect is None:
        return False
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def compute_board_layout() -> Tuple[int, int]:
    avail_w = WIDTH - 2 * BOARD_SIDE_MARGIN
    avail_h = BOARD_BOTTOM_LIMIT - BOARD_Y
    tile_size = min(MAX_TILE_SIZE, avail_w // max(1, COLS), avail_h // max(1, ROWS))
    return tile_size, COLS * tile_size


def draw_text_center(
    screen: pygame.Surface,
    text: object,
    font: pygame.font.Font,
    color: Tuple[int, int, int],
    cx: int,
    cy: int,
) -> None:
    surf = font.render(str(text), True, color)
    screen.blit(surf, surf.get_rect(center=(cx, cy)))


def draw_text_left(
    screen: pygame.Surface,
    text: object,
    font: pygame.font.Font,
    color: Tuple[int, int, int],
    x: int,
    y: int,
) -> None:
    surf = font.render(str(text), True, color)
    screen.blit(surf, (x, y))


def draw_star(
    screen: pygame.Surface, cx: int, cy: int, color: Tuple[int, int, int]
) -> None:
    pts: List[Tuple[float, float]] = []
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        radius = 17.0 if i % 2 == 0 else 8.0
        pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, WHITE, pts, 1)


def get_game_over_explanation(state: GameState) -> str:
    if state.status == "won":
        return "All crystals collected; exit reached!"
    if state.status == "lost":
        if state.player == EXIT_TILE and len(state.crystals) > 0:
            return "Exit reached before all crystals were collected."
        if MAX_MOVES > 0 and len(state.vanished) >= MAX_MOVES:
            return "World collapsed! Move limit reached before escaping."
        return "No moves remain; vanished tiles, hazards, gates, or walls trapped you."
    return "Keep moving and plan your route carefully."


@dataclass
class UIState:
    game_state: GameState
    tile_size: int
    board_width: int
    level_has_solution: bool

    moves: int = 0
    started_at: Optional[float] = None
    final_game_time: float = 0.0

    instructions_open: bool = True
    instruction_scroll: int = 0

    game_paused: bool = False
    pause_started_at: Optional[float] = None
    total_paused_time: float = 0.0

    menu_open: Optional[str] = None

    ai_moves: List[Direction] = None  # type: ignore[assignment]
    ai_move_index: int = 0
    ai_playing: bool = False
    ai_next_move_time: int = 0
    selected_algorithm: str = "A*"
    ai_was_used: bool = False
    ai_nodes_explored: Optional[int] = 0
    ai_execution_time: float = 0.0
    ai_status_message: str = ""

    show_comparison: bool = False
    comparison_results: List[Dict[str, object]] = None  # type: ignore[assignment]

    hints_remaining: int = MAX_HINTS_PER_LEVEL
    hint_next_tile: Optional[Position] = None
    hint_message: str = ""

    mouse_pos: Tuple[int, int] = (-1, -1)

    def __post_init__(self) -> None:
        if self.ai_moves is None:
            self.ai_moves = []
        if self.comparison_results is None:
            self.comparison_results = []


def get_menu_options(menu_name: str, ui: UIState) -> List[Tuple[str, bool]]:
    playing = ui.game_state.status == "playing"
    if menu_name == "Game":
        return [("Exit", True)]
    if menu_name == "AI Tools":
        can_run_ai = playing and not ui.ai_playing
        return [
            ("Play BFS", can_run_ai),
            ("Play DFS", can_run_ai),
            ("Play IDDFS", can_run_ai),
            ("Play Greedy", can_run_ai),
            ("Play A*", can_run_ai),
            ("Play CSP + AC-3", can_run_ai),
            ("AI Hint", can_run_ai),
            ("Compare Algorithms", ui.game_state.status == "won"),
        ]
    if menu_name == "Difficulty":
        return [(name, True) for name in ("Easy", "Medium", "Hard")]
    if menu_name == "Help":
        return [("Instructions / Controls", True)]
    return []


# ---------------------------------------------------------------------------
# MAIN PYGAME CONTROLLER & RENDERER
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Vanishing World — AI Puzzle-Survival Game (Pygame)")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("segoeui", 27, bold=True)
    font_main = pygame.font.SysFont("segoeui", 19, bold=True)
    font_small = pygame.font.SysFont("segoeui", 15, bold=True)
    font_tiny = pygame.font.SysFont("segoeui", 13)

    init_state = generate_new_level(advance_level=False)
    t_size, b_width = compute_board_layout()
    ui = UIState(
        game_state=init_state,
        tile_size=t_size,
        board_width=b_width,
        level_has_solution=has_solution(),
    )

    regions: Dict[str, object] = {
        "menu_rects": {},
        "dropdown_rects": {},
        "hint_btn": None,
        "restart_btn": (18, 514, 112, 32),
        "pause_btn": (18, 554, 125, 32),
        "next_btn": (WIDTH - 143, 554, 125, 32),
        "board_rect": None,
        "cmp_return_rect": (280, 552, 340, 32),
    }

    def calc_elapsed() -> float:
        if ui.started_at is None:
            return 0.0
        now = time.perf_counter()
        paused_now = 0.0
        if ui.game_paused and ui.pause_started_at is not None:
            paused_now = max(0.0, now - ui.pause_started_at)
        return max(0.0, now - ui.started_at - ui.total_paused_time - paused_now)

    def get_elapsed() -> float:
        if ui.game_state.status in ("won", "lost"):
            return max(0.0, ui.final_game_time)
        return calc_elapsed()

    def sync_layout() -> None:
        ui.tile_size, ui.board_width = compute_board_layout()

    def restart_game() -> None:
        ui.game_paused = False
        ui.pause_started_at = None
        ui.total_paused_time = 0.0
        ui.comparison_results = []
        ui.show_comparison = False
        clear_search_result_cache()
        ui.hints_remaining = MAX_HINTS_PER_LEVEL
        ui.hint_next_tile = None
        ui.hint_message = ""
        ui.game_state = create_initial_state()
        sync_layout()
        ui.moves = 0
        ui.started_at = None
        ui.final_game_time = 0.0
        ui.ai_moves = []
        ui.ai_move_index = 0
        ui.ai_playing = False
        ui.ai_next_move_time = 0
        ui.selected_algorithm = "A*"
        ui.ai_was_used = False
        ui.ai_nodes_explored = 0
        ui.ai_execution_time = 0.0
        ui.ai_status_message = ""

    def next_level() -> None:
        if ui.game_state.status != "won" and ui.level_has_solution:
            return
        clear_search_result_cache()
        ui.game_state = generate_new_level(advance_level=True)
        ui.level_has_solution = has_solution()
        restart_game()

    def toggle_pause() -> None:
        if ui.game_state.status != "playing":
            return
        if not ui.game_paused:
            ui.game_paused = True
            ui.pause_started_at = time.perf_counter()
            return
        paused_sec = (
            max(0.0, time.perf_counter() - ui.pause_started_at)
            if ui.pause_started_at is not None
            else 0.0
        )
        paused_ms = int(paused_sec * 1000)
        if ui.ai_playing:
            ui.ai_next_move_time += paused_ms
        ui.total_paused_time += paused_sec
        ui.game_paused = False
        ui.pause_started_at = None

    def change_difficulty(new_diff: str) -> None:
        if not set_difficulty(new_diff):
            return
        ui.game_state = generate_new_level(advance_level=False)
        ui.level_has_solution = has_solution()
        restart_game()

    def handle_move(direction: Direction) -> None:
        if ui.game_state.status != "playing":
            return
        nxt = move_state(ui.game_state, direction)
        if nxt is None:
            return
        if ui.started_at is None:
            ui.started_at = time.perf_counter()
        ui.game_state = nxt
        ui.moves += 1
        ui.hint_next_tile = None
        ui.hint_message = ""
        if ui.game_state.status in ("won", "lost"):
            ui.final_game_time = calc_elapsed()

    def request_hint() -> None:
        if ui.game_state.status != "playing":
            ui.hint_message = "Hints are available only during gameplay."
            ui.hint_next_tile = None
            return
        if ui.hints_remaining <= 0:
            ui.hint_next_tile = None
            ui.hint_message = "No hints remaining for this level."
            return
        if ui.ai_playing:
            ui.hint_next_tile = None
            ui.hint_message = "Pause AI playback or restart before requesting a hint."
            return

        path: Optional[List[Direction]] = None
        try:
            path, _ = astar_search(ui.game_state, get_search_node_budget("A*"))
        except SearchLimitReached:
            path = None

        if not path:
            ui.hint_next_tile = None
            ui.hint_message = "A* found no route within its search budget."
            return

        direction = path[0]
        suggested = move_state(ui.game_state, direction)
        if suggested is None or suggested.status == "lost":
            ui.hint_next_tile = None
            ui.hint_message = "No safe hint is available from this state."
            return

        ui.hint_next_tile = suggested.player
        ui.hints_remaining -= 1
        if len(suggested.keys) < len(ui.game_state.keys):
            ui.hint_message = f"Move {direction}: picks up a key ({len(suggested.keys)} left) to unlock gates."
        elif len(suggested.crystals) < len(ui.game_state.crystals):
            ui.hint_message = f"Move {direction}: this move collects a crystal."
        elif suggested.player in CRUMBLING_TILES:
            ui.hint_message = f"Move {direction}: crosses a crumbling tile before it collapses."
        elif suggested.player in SPIKE_TILES:
            ui.hint_message = f"Move {direction}: steps safely across a retracted spike trap."
        elif len(suggested.crystals) == 0:
            ui.hint_message = f"Move {direction}: all crystals collected; head to exit!"
        else:
            ui.hint_message = f"Move {direction}: first step on an A* route to goal."

    def start_ai(search_name: str, search_fn: SearchFn) -> None:
        restart_game()
        ui.ai_was_used = True
        ui.selected_algorithm = search_name
        t0 = time.perf_counter()
        solution: Optional[List[Direction]] = None
        nodes: Optional[int] = None
        search_status: Optional[str] = None
        try:
            solution, nodes = search_fn(ui.game_state, get_search_node_budget(search_name))
        except SearchLimitReached as err:
            nodes = err.nodes_explored
            search_status = "Search limit reached"
        ui.ai_execution_time = (time.perf_counter() - t0) * 1000.0
        ui.ai_nodes_explored = nodes

        record_search_result(
            search_name,
            solution,
            nodes,
            ui.ai_execution_time,
            "Completed" if solution is not None else (search_status or "No solution found"),
        )
        if solution is None:
            ui.ai_status_message = (
                "Search limit reached; try another algorithm or play manually."
                if search_status
                else "No winning route was found by this algorithm."
            )
        else:
            ui.ai_status_message = ""

        ui.ai_moves = solution or []
        ui.ai_move_index = 0
        ui.ai_playing = len(ui.ai_moves) > 0
        ui.ai_next_move_time = pygame.time.get_ticks()

    def perform_menu_action(action: str) -> None:
        ui.menu_open = None
        if action == "Exit":
            ui.instructions_open = True
            ui.instruction_scroll = 0
        elif action == "Play BFS":
            if ui.game_state.status == "playing":
                start_ai("BFS", bfs_search)
        elif action == "Play DFS":
            if ui.game_state.status == "playing":
                start_ai("DFS", dfs_search)
        elif action == "Play IDDFS":
            if ui.game_state.status == "playing":
                start_ai("IDDFS", iddfs_search)
        elif action == "Play Greedy":
            if ui.game_state.status == "playing":
                start_ai("Greedy", greedy_search)
        elif action == "Play A*":
            if ui.game_state.status == "playing":
                start_ai("A*", astar_search)
        elif action == "Play CSP + AC-3":
            if ui.game_state.status == "playing":
                start_ai("CSP + AC-3", csp_search)
        elif action == "AI Hint":
            request_hint()
        elif action == "Compare Algorithms":
            if ui.game_state.status == "won":
                ui.show_comparison = True
                if not ui.comparison_results:
                    ui.comparison_results = compare_algorithms(create_initial_state())
        elif action in ("Easy", "Medium", "Hard"):
            change_difficulty(action.upper())
        elif action == "Instructions / Controls":
            ui.instructions_open = True

    def draw_board(board_x: int) -> None:
        gate_locked = len(ui.game_state.keys) > 0
        next_step = len(ui.game_state.vanished) + 1
        ts = ui.tile_size
        exit_inset = max(3, ts // 8)
        player_r = max(4, ts // 4)
        crystal_hw = max(4, ts // 5)
        crystal_hh = max(5, ts // 4)
        hint_inset = max(4, ts // 10)

        for r in range(ROWS):
            for c in range(COLS):
                x = board_x + c * ts
                y = BOARD_Y + r * ts
                pos = (r, c)
                deadline = CRUMBLING_TILES.get(pos)
                spike_parity = SPIKE_TILES.get(pos)

                tile_color = TILE_COLOR
                if pos in ui.game_state.vanished:
                    tile_color = VANISHED_COLOR
                elif pos in WALLS:
                    tile_color = WALL_COLOR
                elif pos in LOCKED_GATES:
                    tile_color = GATE_FILL_LOCKED if gate_locked else GATE_FILL_OPEN
                elif deadline is not None:
                    if next_step > deadline:
                        tile_color = CRUMBLE_DEAD
                    elif (deadline - len(ui.game_state.vanished)) <= 2:
                        tile_color = CRUMBLE_URGENT
                    else:
                        tile_color = CRUMBLE_FILL
                elif spike_parity is not None:
                    armed_next = (next_step % 2) == spike_parity
                    tile_color = SPIKE_ARMED_BG if armed_next else SPIKE_SAFE_BG

                pygame.draw.rect(screen, tile_color, (x, y, ts, ts))
                pygame.draw.rect(screen, GRID_COLOR, (x + 1, y + 1, ts - 2, ts - 2), 2)

                if (
                    deadline is not None
                    and pos not in ui.game_state.vanished
                    and pos not in WALLS
                ):
                    remaining = deadline - len(ui.game_state.vanished)
                    if remaining <= 0:
                        pygame.draw.line(
                            screen, CRUMBLE_URGENT, (x + 8, y + 8), (x + ts - 8, y + ts - 8), 2
                        )
                        pygame.draw.line(
                            screen, CRUMBLE_URGENT, (x + ts - 8, y + 8), (x + 8, y + ts - 8), 2
                        )
                    else:
                        pygame.draw.line(
                            screen,
                            CRUMBLE_CRACK,
                            (x + 6, y + ts - 8),
                            (x + ts // 2, y + ts // 2 + 2),
                            2,
                        )
                        pygame.draw.line(
                            screen,
                            CRUMBLE_CRACK,
                            (x + ts // 2, y + ts // 2 + 2),
                            (x + ts - 7, y + 7),
                            2,
                        )
                        draw_text_center(
                            screen,
                            f"T{remaining}",
                            font_tiny,
                            WHITE if remaining <= 2 else CRUMBLE_CRACK,
                            x + ts // 2,
                            y + max(11, ts // 4),
                        )

                if (
                    spike_parity is not None
                    and pos not in ui.game_state.vanished
                    and pos not in WALLS
                ):
                    armed_next = (next_step % 2) == spike_parity
                    cx, cy = x + ts // 2, y + ts // 2
                    off = max(6, ts // 4)
                    sz = max(4, ts // 9)
                    if armed_next:
                        for sx, sy in (
                            (cx - off, cy - off),
                            (cx + off, cy - off),
                            (cx - off, cy + off),
                            (cx + off, cy + off),
                        ):
                            pts = [
                                (sx, sy - sz),
                                (sx - sz, sy + sz),
                                (sx + sz, sy + sz),
                            ]
                            pygame.draw.polygon(screen, SPIKE_ARMED_FG, pts)
                    else:
                        vent_r = max(2, ts // 14)
                        for sx, sy in (
                            (cx - off, cy - off),
                            (cx + off, cy - off),
                            (cx - off, cy + off),
                            (cx + off, cy + off),
                        ):
                            pygame.draw.circle(screen, SPIKE_SAFE_FG, (sx, sy), vent_r)

                one_way_dir = ONE_WAY_TILES.get(pos)
                if one_way_dir and pos not in ui.game_state.vanished and pos not in WALLS:
                    angle = {"RIGHT": 0, "DOWN": 90, "LEFT": 180, "UP": -90}[one_way_dir]
                    rad = math.radians(angle)
                    cx, cy = x + ts // 2, y + ts // 2
                    rr = max(6, int(ts * 0.26))
                    pts = [
                        (cx + rr * math.cos(rad), cy + rr * math.sin(rad)),
                        (cx + rr * 0.8 * math.cos(rad + 2.3), cy + rr * 0.8 * math.sin(rad + 2.3)),
                        (cx - rr * 0.25 * math.cos(rad), cy - rr * 0.25 * math.sin(rad)),
                        (cx + rr * 0.8 * math.cos(rad - 2.3), cy + rr * 0.8 * math.sin(rad - 2.3)),
                    ]
                    pygame.draw.polygon(screen, (150, 200, 245), pts)

                if pos in LOCKED_GATES and pos not in ui.game_state.vanished:
                    pad = max(4, int(ts * 0.14))
                    if gate_locked:
                        pygame.draw.rect(
                            screen,
                            GATE_BORDER_LOCKED,
                            (x + pad, y + pad, ts - pad * 2, ts - pad * 2),
                            max(2, ts // 16),
                        )
                        draw_text_center(
                            screen,
                            "G",
                            font_tiny if ts < 40 else font_small,
                            GATE_TEXT_LOCKED,
                            x + ts // 2,
                            y + ts // 2 + 1,
                        )
                    else:
                        pygame.draw.rect(
                            screen, PLAYER_COLOR, (x + pad, y + pad, ts - pad * 2, ts - pad * 2), 2
                        )

                if ui.hint_next_tile == pos and pos not in ui.game_state.vanished and pos not in WALLS:
                    pygame.draw.rect(
                        screen,
                        HINT_OUTLINE,
                        (x + hint_inset, y + hint_inset, ts - hint_inset * 2, ts - hint_inset * 2),
                        max(1, ts // 16),
                        border_radius=5,
                    )

                if pos == EXIT_TILE and pos not in ui.game_state.vanished:
                    pygame.draw.rect(
                        screen,
                        EXIT_COLOR,
                        (x + exit_inset, y + exit_inset, ts - exit_inset * 2, ts - exit_inset * 2),
                    )
                    draw_text_center(
                        screen,
                        "E",
                        font_small if ts < 40 else font_main,
                        WALL_COLOR,
                        x + ts // 2,
                        y + ts // 2 + 1,
                    )

                if pos in ui.game_state.keys and pos not in ui.game_state.vanished:
                    cx, cy = x + ts // 2, y + ts // 2
                    badge_r = max(7, int(ts * 0.25))
                    pygame.draw.circle(screen, KEY_COLOR, (cx, cy), badge_r)
                    pygame.draw.circle(screen, WHITE, (cx, cy), badge_r, max(1, ts // 22))
                    draw_text_center(
                        screen,
                        "K",
                        font_tiny if ts < 40 else font_small,
                        WALL_COLOR,
                        cx,
                        cy + 1,
                    )

                if pos in ui.game_state.crystals:
                    cx, cy = x + ts // 2, y + ts // 2
                    pts = [(cx, cy - crystal_hh), (cx + crystal_hw, cy), (cx, cy + crystal_hh), (cx - crystal_hw, cy)]
                    pygame.draw.polygon(screen, CRYSTAL_COLOR, pts)
                    pygame.draw.polygon(screen, WHITE, pts, max(1, ts // 20))

                if pos == ui.game_state.player:
                    cx, cy = x + ts // 2, y + ts // 2
                    pygame.draw.circle(screen, PLAYER_COLOR, (cx, cy), player_r)
                    pygame.draw.circle(screen, WHITE, (cx, cy), player_r, max(1, ts // 20))

    def draw_results_panel() -> None:
        panel_x, panel_y, panel_w, panel_h = 425, 125, 425, 405
        pygame.draw.rect(screen, PANEL_COLOR, (panel_x, panel_y, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(screen, PANEL_LINE, (panel_x, panel_y, panel_w, panel_h), 2, border_radius=10)

        collected = len(START_CRYSTALS) - len(ui.game_state.crystals)
        status = ui.game_state.status
        outcome = ("AI WON!" if ui.ai_was_used else "YOU WON!") if status == "won" else "LEVEL FAILED"
        outcome_color = PLAYER_COLOR if status == "won" else RED

        draw_text_center(screen, "FINAL RESULTS", font_main, WHITE, panel_x + panel_w // 2, panel_y + 28)
        draw_text_center(screen, outcome, font_main, outcome_color, panel_x + panel_w // 2, panel_y + 62)
        draw_text_center(
            screen,
            get_game_over_explanation(ui.game_state),
            font_tiny,
            MUTED_TEXT,
            panel_x + panel_w // 2,
            panel_y + 91,
        )

        for i in range(3):
            draw_star(
                screen,
                panel_x + panel_w // 2 + (i - 1) * 48,
                panel_y + 119,
                GOLD if i < collected else EMPTY_STAR,
            )

        rows_data: List[Tuple[str, str]] = [
            ("Player type", "AI" if ui.ai_was_used else "Human"),
            ("Outcome", "Won" if status == "won" else "Lost"),
            ("Moves made", str(ui.moves)),
            ("Elapsed time", f"{get_elapsed():.2f} seconds"),
            ("Crystals collected", f"{collected}/{len(START_CRYSTALS)}"),
        ]
        if ui.ai_was_used:
            rows_data.extend(
                [
                    ("Algorithm", ui.selected_algorithm),
                    ("Solution length", str(len(ui.ai_moves))),
                    ("Search time", f"{ui.ai_execution_time:.3f} ms"),
                    (
                        "Nodes explored",
                        str(ui.ai_nodes_explored)
                        if ui.ai_nodes_explored is not None
                        else "Not reported",
                    ),
                ]
            )

        for idx, (lbl, val) in enumerate(rows_data):
            row_y = panel_y + 145 + idx * 27
            if idx % 2 == 0:
                pygame.draw.rect(
                    screen, ROW_STRIPE, (panel_x + 10, row_y - 12, panel_w - 20, 27), border_radius=4
                )
            draw_text_left(screen, lbl, font_tiny, MUTED_TEXT, panel_x + 18, row_y - 6)
            draw_text_left(screen, val, font_tiny, WHITE, panel_x + 210, row_y - 6)

    def draw_menu_bar() -> None:
        menu_rects: Dict[str, RectTuple] = {}
        dropdown_rects: Dict[str, Tuple[RectTuple, bool]] = {}

        pygame.draw.rect(screen, MENU_BG, (0, 0, WIDTH, MENU_BAR_HEIGHT))
        pygame.draw.line(screen, PANEL_LINE, (0, MENU_BAR_HEIGHT - 1), (WIDTH, MENU_BAR_HEIGHT - 1), 1)

        x = 14
        mx, my = ui.mouse_pos
        for label in MENU_ITEMS:
            tw = font_small.size(label)[0]
            rect: RectTuple = (x - 7, 1, tw + 20, MENU_BAR_HEIGHT - 2)
            menu_rects[label] = rect
            if label == ui.menu_open or point_in_rect(mx, my, rect):
                pygame.draw.rect(screen, MENU_HOVER, rect, border_radius=4)
            draw_text_left(screen, label, font_small, WHITE, x + 3, 5)
            x += tw + 35

        regions["menu_rects"] = menu_rects
        regions["dropdown_rects"] = dropdown_rects

        if not ui.menu_open:
            return
        options = get_menu_options(ui.menu_open, ui)
        m_rect = menu_rects.get(ui.menu_open)
        if not m_rect or not options:
            return

        dw = 235 if ui.menu_open in ("AI Tools", "Game") else 185
        rh = 30
        dd_rect: RectTuple = (m_rect[0], MENU_BAR_HEIGHT, dw, rh * len(options) + 6)
        pygame.draw.rect(screen, PANEL_COLOR, dd_rect, border_radius=5)
        pygame.draw.rect(screen, PANEL_LINE, dd_rect, 1, border_radius=5)

        for idx, (label, enabled) in enumerate(options):
            row: RectTuple = (dd_rect[0] + 3, dd_rect[1] + 3 + idx * rh, dw - 6, rh)
            dropdown_rects[label] = (row, enabled)
            if enabled and point_in_rect(mx, my, row):
                pygame.draw.rect(screen, MENU_HOVER, row, border_radius=3)
            draw_text_left(
                screen,
                label,
                font_small,
                WHITE if enabled else MENU_DISABLED,
                row[0] + 9,
                row[1] + 6,
            )

    def draw_instructions_screen() -> None:
        screen.fill(BACKGROUND)
        draw_text_center(screen, "VANISHING WORLD", font_title, CRYSTAL_COLOR, WIDTH // 2, 35)
        draw_text_center(screen, "HOW TO PLAY", font_main, WHITE, WIDTH // 2, 75)

        sections = (
            (
                "OBJECTIVE",
                (
                    "Collect ALL crystals on the board and reach the exit (E)",
                    "before the world collapses or you run out of legal moves.",
                ),
            ),
            (
                "MOVEMENT",
                (
                    "Use the arrow keys (or WASD / click adjacent tiles) to move.",
                    "You can move up, down, left, or right.",
                ),
            ),
            (
                "VANISHING TILES & MOVE LIMIT",
                (
                    "Every tile you leave disappears permanently.",
                    "Each level has a collapse move limit shown below the board.",
                    "Plan your route carefully to avoid trapping yourself.",
                ),
            ),
            (
                "KEYS, LOCKED GATES & ONE-WAY TILES",
                (
                    "Locked Gates (G) stay sealed until ALL Keys (K) are collected.",
                    "Level 6+ places multiple Keys before gates; grab every key first.",
                    "One-Way Arrow Tiles force your next step in the arrow direction.",
                ),
            ),
            (
                "CRUMBLING TILES & SPIKE TRAPS (Lv 4+)",
                (
                    "Crumbling Tiles (T#) collapse permanently once # turns pass.",
                    "Spike Traps alternate every turn: red spikes block your next move,",
                    "while dim floor vents mean the next step onto that tile is safe.",
                ),
            ),
            (
                "WALLS AND BOUNDARIES",
                (
                    "Walls interrupt direct paths and create tricky forks.",
                    "Look ahead for dead-end traps before committing to a route.",
                ),
            ),
            (
                "WINNING AND LOSING",
                (
                    "WIN: Collect all crystals and reach the exit within the move limit.",
                    "LOSE: Reach the exit early, exceed the move limit, or get trapped.",
                ),
            ),
            (
                "AI CONTROLS",
                (
                    "1 - Breadth-First Search (BFS)",
                    "2 - Depth-First Search (DFS)",
                    "3 - Iterative Deepening DFS (IDDFS)",
                    "4 - Greedy Best-First Search",
                    "5 - A* Search",
                    "6 - CSP with AC-3 and Backtracking",
                    "The AI searches for a solution and plays it.",
                ),
            ),
            (
                "OTHER CONTROLS",
                (
                    "H - Ask A* for a hint (3 hints per level).",
                    "P - Pause or resume the game.",
                    "R - Restart the current game.",
                    "C - Open the algorithm comparison screen after winning.",
                    "N - Generate a next level, if enabled.",
                ),
            ),
        )

        clip_rect = pygame.Rect(35, 105, WIDTH - 70, 345)
        screen.set_clip(clip_rect)
        y = 120 - ui.instruction_scroll
        for heading, lines in sections:
            draw_text_center(screen, heading, font_main, EXIT_COLOR, WIDTH // 2, y)
            y += 28
            for line in lines:
                draw_text_center(screen, line, font_small, WHITE, WIDTH // 2, y)
                y += 24
            y += 12
        screen.set_clip(None)

        draw_text_center(
            screen,
            "Scroll with mouse wheel or Up/Down arrows (ESC or Close to start)",
            font_small,
            CRYSTAL_COLOR,
            WIDTH // 2,
            468,
        )

        hover = point_in_rect(ui.mouse_pos[0], ui.mouse_pos[1], CLOSE_BUTTON_RECT)
        pygame.draw.rect(
            screen,
            BUTTON_HOVER_COLOR if hover else BUTTON_COLOR,
            CLOSE_BUTTON_RECT,
            border_radius=8,
        )
        draw_text_center(
            screen,
            "CLOSE",
            font_main,
            WHITE,
            CLOSE_BUTTON_RECT[0] + CLOSE_BUTTON_RECT[2] // 2,
            CLOSE_BUTTON_RECT[1] + CLOSE_BUTTON_RECT[3] // 2,
        )

    def draw_comparison_screen() -> None:
        screen.fill(COMP_BG)
        draw_text_center(screen, "AI ALGORITHM COMPARISON", font_title, WHITE, WIDTH // 2, 35)
        results = ui.comparison_results

        if not results:
            draw_text_center(screen, "No comparison results available.", font_main, RED, WIDTH // 2, 180)
            draw_text_center(screen, "Press C to return to the game", font_small, COMP_PROMPT, WIDTH // 2, 565)
            return

        successful = [
            r for r in results if r.get("moves") is not None and r.get("status") == "Completed"
        ]
        measured_nodes = [r for r in results if r.get("nodes") is not None]
        shortest_moves = min((int(r["moves"]) for r in successful), default=None)  # type: ignore[arg-type]
        comp_nodes = [r for r in successful if r.get("nodes") is not None]
        comp_times = [r for r in successful if r.get("time") is not None]
        fewest_nodes = min((int(r["nodes"]) for r in comp_nodes), default=None)  # type: ignore[arg-type]
        fastest_time = min((float(r["time"]) for r in comp_times), default=None)  # type: ignore[arg-type]

        draw_text_center(
            screen,
            "Green indicates the best value among completed searches.",
            font_tiny,
            BEST_GREEN,
            WIDTH // 2,
            60,
        )

        table_top = 82
        cols_cfg = (
            ("Algorithm", 115),
            ("Moves", 325),
            ("Nodes", 435),
            ("Time (ms)", 555),
            ("Status", 730),
        )
        for heading, cx in cols_cfg:
            draw_text_center(screen, heading, font_small, CRYSTAL_COLOR, cx, table_top)

        for idx, res in enumerate(results):
            y = table_top + 33 + idx * 32
            st = str(res.get("status") or "Unknown")
            is_short = shortest_moves is not None and res.get("moves") == shortest_moves and st == "Completed"
            name_col = BEST_GREEN if is_short else WHITE
            draw_text_center(screen, res["name"], font_tiny, name_col, cols_cfg[0][1], y)
            draw_text_center(
                screen,
                str(res["moves"]) if res.get("moves") is not None else "-",
                font_tiny,
                name_col,
                cols_cfg[1][1],
                y,
            )

            nodes_col = (
                BEST_GREEN
                if fewest_nodes is not None and res.get("nodes") == fewest_nodes and st == "Completed"
                else WHITE
            )
            draw_text_center(
                screen,
                str(res["nodes"]) if res.get("nodes") is not None else "-",
                font_tiny,
                nodes_col,
                cols_cfg[2][1],
                y,
            )

            tval = res.get("time")
            time_col = (
                BEST_GREEN
                if fastest_time is not None and tval == fastest_time and st == "Completed"
                else WHITE
            )
            draw_text_center(
                screen,
                f"{float(tval):.2f}" if isinstance(tval, (int, float)) else "-",
                font_tiny,
                time_col,
                cols_cfg[3][1],
                y,
            )

            st_short = "Not completed"
            if st == "Completed":
                st_short = "Completed"
            elif "limit" in st.lower():
                st_short = "Limit reached"
            elif "skipped" in st.lower():
                st_short = "Skipped"
            elif "no solution" in st.lower():
                st_short = "No solution"
            draw_text_center(
                screen,
                st_short,
                font_tiny,
                BEST_GREEN if st == "Completed" else MUTED_TEXT,
                cols_cfg[4][1],
                y,
            )

        draw_text_center(screen, "NODES EXPLORED", font_main, CRYSTAL_COLOR, WIDTH // 2, 295)
        chart_x, chart_y, chart_w, bar_h, bar_gap = 265, 320, 455, 17, 8
        max_nodes = max([int(r["nodes"]) for r in measured_nodes] + [1])  # type: ignore[arg-type]

        for idx, res in enumerate(results):
            y = chart_y + idx * (bar_h + bar_gap)
            draw_text_center(screen, res["name"], font_tiny, WHITE, 155, y + bar_h // 2)
            n_val = res.get("nodes")
            if n_val is None:
                draw_text_center(screen, "Not measured", font_tiny, MUTED_TEXT, chart_x + 65, y + bar_h // 2)
                continue
            n_int = int(n_val)  # type: ignore[arg-type]
            bar_w = int((n_int / max_nodes) * chart_w)
            if n_int > 0:
                bar_w = max(bar_w, 2)
            is_best = fewest_nodes is not None and n_int == fewest_nodes and res.get("status") == "Completed"
            pygame.draw.rect(
                screen,
                BEST_GREEN if is_best else CRYSTAL_COLOR,
                (chart_x, y, bar_w, bar_h),
                border_radius=3,
            )
            draw_text_center(
                screen,
                n_int,
                font_tiny,
                WHITE,
                min(chart_x + bar_w + 30, WIDTH - 45),
                y + bar_h // 2,
            )

        summary_y = 475
        if successful:
            s_names = [str(r["name"]) for r in successful if r.get("moves") == shortest_moves]
            draw_text_center(
                screen,
                f"Shortest completed solution: {shortest_moves} moves ({', '.join(s_names)})",
                font_tiny,
                BEST_GREEN,
                WIDTH // 2,
                summary_y + 60,
            )
        else:
            draw_text_center(screen, "No completed search found a solution.", font_tiny, MUTED_TEXT, WIDTH // 2, summary_y)

        if fewest_nodes is not None:
            f_names = [str(r["name"]) for r in comp_nodes if r.get("nodes") == fewest_nodes]
            draw_text_center(
                screen,
                f"Fewest nodes among completed searches: {', '.join(f_names)}",
                font_tiny,
                WHITE,
                WIDTH // 2,
                summary_y + 20,
            )

        if fastest_time is not None:
            t_names = [str(r["name"]) for r in comp_times if r.get("time") == fastest_time]
            draw_text_center(
                screen,
                f"Fastest completed search: {', '.join(t_names)}",
                font_tiny,
                WHITE,
                WIDTH // 2,
                summary_y + 39,
            )

        draw_text_center(screen, "Press C to return to the game", font_small, COMP_PROMPT, WIDTH // 2, 570)

    def draw_main_screen() -> None:
        screen.fill(BACKGROUND)
        status = ui.game_state.status
        game_over = status in ("won", "lost")

        draw_text_center(screen, "VANISHING WORLD", font_title, WHITE, WIDTH // 2, 43)
        draw_text_center(screen, "Collect crystals and reach the exit", font_small, WHITE, WIDTH // 2, 70)
        draw_text_center(screen, f"LEVEL {LEVEL_NUMBER}", font_main, EXIT_COLOR, WIDTH // 2, 91)
        draw_text_center(
            screen,
            f"Board: {ROWS} x {COLS}    Difficulty: {DIFFICULTY}",
            font_small,
            CRYSTAL_COLOR,
            WIDTH // 2,
            113,
        )

        board_x = (
            BOARD_SIDE_MARGIN
            if (game_over or not ui.level_has_solution)
            else (WIDTH - ui.board_width) // 2
        )
        regions["board_rect"] = (board_x, BOARD_Y, ui.board_width, ROWS * ui.tile_size)
        draw_board(board_x)
        board_center = board_x + ui.board_width // 2

        restart_btn: RectTuple = (18, HEIGHT - 14 - 32 * 2 - 8, 112, 32)
        pause_btn: RectTuple = (18, HEIGHT - 14 - 32, 125, 32)
        next_btn: RectTuple = (WIDTH - 18 - 125, HEIGHT - 14 - 32, 125, 32)
        regions["restart_btn"] = restart_btn
        regions["pause_btn"] = pause_btn
        regions["next_btn"] = next_btn

        next_allowed = status == "won" or not ui.level_has_solution
        controls = (
            (restart_btn, "Restart", True),
            (pause_btn, "Resume" if ui.game_paused else "Pause", status == "playing" or ui.game_paused),
            (next_btn, "Next Level", next_allowed and not ui.ai_playing),
        )
        mx, my = ui.mouse_pos
        for rect, label, enabled in controls:
            fill = MENU_BG
            if not enabled:
                fill = DISABLED_FILL
            elif point_in_rect(mx, my, rect):
                fill = MENU_HOVER
            pygame.draw.rect(screen, fill, rect, border_radius=6)
            pygame.draw.rect(screen, PANEL_LINE, rect, 1, border_radius=6)
            draw_text_center(
                screen,
                label,
                font_small,
                WHITE if enabled else MENU_DISABLED,
                rect[0] + rect[2] // 2,
                rect[1] + rect[3] // 2,
            )

        moves_left = max(0, MAX_MOVES - ui.moves) if MAX_MOVES > 0 else 999
        moves_color = RED if moves_left <= 1 else (EXIT_COLOR if moves_left <= 3 else WHITE)
        moves_label = f"Moves: {ui.moves}/{MAX_MOVES}" if MAX_MOVES > 0 else f"Moves: {ui.moves}"
        draw_text_center(screen, moves_label, font_main, moves_color, board_center - 80, 465)
        draw_text_center(
            screen,
            f"Crystals left: {len(ui.game_state.crystals)}",
            font_main,
            CRYSTAL_COLOR,
            board_center + 85,
            465,
        )

        if status == "playing" and ui.level_has_solution:
            hint_btn: RectTuple = (board_center - 75, 546, 150, 32)
            regions["hint_btn"] = hint_btn
            hover_h = point_in_rect(mx, my, hint_btn)
            pygame.draw.rect(screen, MENU_HOVER if hover_h else MENU_BG, hint_btn, border_radius=7)
            pygame.draw.rect(screen, PANEL_LINE, hint_btn, 1, border_radius=7)
            draw_text_center(
                screen,
                f"Hint ({ui.hints_remaining} left)",
                font_small,
                CRYSTAL_COLOR,
                hint_btn[0] + hint_btn[2] // 2,
                hint_btn[1] + hint_btn[3] // 2,
            )
            if ui.hint_message:
                draw_text_center(screen, ui.hint_message, font_tiny, WHITE, WIDTH // 2, 591)
        else:
            regions["hint_btn"] = None

        if not ui.level_has_solution:
            draw_text_center(screen, "NO SOLUTION", font_title, RED, WIDTH // 2, 490)
            draw_text_center(screen, "This level cannot be completed.", font_small, WHITE, WIDTH // 2, 520)
            draw_text_center(screen, "Press N to generate the next level", font_main, EXIT_COLOR, WIDTH // 2, 560)
            return

        if status == "won":
            draw_results_panel()
            draw_text_center(screen, "LEVEL CLEARED!", font_main, PLAYER_COLOR, WIDTH // 2, 555)
        elif status == "lost":
            draw_results_panel()
            draw_text_center(screen, "LEVEL FAILED!", font_main, RED, WIDTH // 2, 555)

        if status == "playing":
            if ui.ai_playing:
                draw_text_center(screen, "AI is playing...", font_small, CRYSTAL_COLOR, WIDTH // 2, 510)
            elif ui.ai_was_used and ui.ai_status_message:
                draw_text_center(screen, ui.ai_status_message, font_tiny, RED, WIDTH // 2, 510)
            else:
                draw_text_center(screen, "Use arrow keys or WASD to move", font_small, WHITE, WIDTH // 2, 520)

    def draw_pause_overlay() -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 10, 18, 190))
        screen.blit(overlay, (0, 0))
        px, py, pw, ph = WIDTH // 2 - 190, HEIGHT // 2 - 105, 380, 210
        pygame.draw.rect(screen, PANEL_COLOR, (px, py, pw, ph), border_radius=14)
        pygame.draw.rect(screen, PANEL_LINE, (px, py, pw, ph), 2, border_radius=14)
        draw_text_center(screen, "GAME PAUSED", font_title, WHITE, WIDTH // 2, py + 45)
        draw_text_center(screen, "P  -  Resume game", font_main, CRYSTAL_COLOR, WIDTH // 2, py + 95)
        draw_text_center(screen, "Your progress is preserved.", font_small, WHITE, WIDTH // 2, py + 125)
        draw_text_center(screen, "Press P again to continue.", font_tiny, MUTED_TEXT, WIDTH // 2, py + 165)

    running = True
    while running:
        now_ticks = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                ui.mouse_pos = event.pos
            elif event.type == pygame.MOUSEWHEEL and ui.instructions_open:
                ui.instruction_scroll = max(
                    0, min(INSTRUCTION_MAX_SCROLL, ui.instruction_scroll - event.y * 25)
                )
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                ui.mouse_pos = event.pos

                if ui.instructions_open:
                    if point_in_rect(mx, my, CLOSE_BUTTON_RECT):
                        ui.instructions_open = False
                    continue

                if ui.show_comparison:
                    if point_in_rect(mx, my, regions["cmp_return_rect"]):  # type: ignore[arg-type]
                        ui.show_comparison = False
                    continue

                m_rects: Dict[str, RectTuple] = regions["menu_rects"]  # type: ignore[assignment]
                d_rects: Dict[str, Tuple[RectTuple, bool]] = regions["dropdown_rects"]  # type: ignore[assignment]

                menu_hit = False
                for label, rect in m_rects.items():
                    if point_in_rect(mx, my, rect):
                        ui.menu_open = None if ui.menu_open == label else label
                        menu_hit = True
                        break
                if menu_hit:
                    continue

                dd_hit = False
                for label, (rect, enabled) in d_rects.items():
                    if point_in_rect(mx, my, rect):
                        if enabled:
                            perform_menu_action(label)
                        else:
                            ui.menu_open = None
                        dd_hit = True
                        break
                if dd_hit:
                    continue

                if ui.menu_open is not None:
                    ui.menu_open = None
                    continue

                if point_in_rect(mx, my, regions["hint_btn"]):  # type: ignore[arg-type]
                    request_hint()
                elif point_in_rect(mx, my, regions["restart_btn"]):  # type: ignore[arg-type]
                    ui.show_comparison = False
                    restart_game()
                elif point_in_rect(mx, my, regions["pause_btn"]):  # type: ignore[arg-type]
                    if ui.game_state.status == "playing" or ui.game_paused:
                        toggle_pause()
                elif point_in_rect(mx, my, regions["next_btn"]):  # type: ignore[arg-type]
                    if not ui.ai_playing and (ui.game_state.status == "won" or not ui.level_has_solution):
                        ui.show_comparison = False
                        next_level()
                elif (
                    not ui.game_paused
                    and ui.game_state.status == "playing"
                    and not ui.ai_playing
                    and point_in_rect(mx, my, regions["board_rect"])  # type: ignore[arg-type]
                ):
                    bx, by, _, _ = regions["board_rect"]  # type: ignore[misc]
                    col = (mx - bx) // ui.tile_size
                    row = (my - by) // ui.tile_size
                    pr, pc = ui.game_state.player
                    dr, dc = row - pr, col - pc
                    clicked_dir: Optional[Direction] = None
                    if (dr, dc) == (-1, 0):
                        clicked_dir = "UP"
                    elif (dr, dc) == (1, 0):
                        clicked_dir = "DOWN"
                    elif (dr, dc) == (0, -1):
                        clicked_dir = "LEFT"
                    elif (dr, dc) == (0, 1):
                        clicked_dir = "RIGHT"
                    if clicked_dir:
                        handle_move(clicked_dir)

            elif event.type == pygame.KEYDOWN:
                if ui.instructions_open:
                    if event.key == pygame.K_ESCAPE:
                        ui.instructions_open = False
                    elif event.key == pygame.K_DOWN:
                        ui.instruction_scroll = min(INSTRUCTION_MAX_SCROLL, ui.instruction_scroll + 25)
                    elif event.key == pygame.K_UP:
                        ui.instruction_scroll = max(0, ui.instruction_scroll - 25)
                    elif event.key == pygame.K_PAGEDOWN:
                        ui.instruction_scroll = min(INSTRUCTION_MAX_SCROLL, ui.instruction_scroll + 150)
                    elif event.key == pygame.K_PAGEUP:
                        ui.instruction_scroll = max(0, ui.instruction_scroll - 150)
                    continue

                if event.key == pygame.K_p and not ui.show_comparison:
                    toggle_pause()
                    continue

                if ui.game_paused:
                    continue

                if event.key == pygame.K_F1:
                    change_difficulty("EASY")
                    continue
                if event.key == pygame.K_F2:
                    change_difficulty("MEDIUM")
                    continue
                if event.key == pygame.K_F3:
                    change_difficulty("HARD")
                    continue

                if event.key == pygame.K_r:
                    ui.show_comparison = False
                    restart_game()
                    continue

                if event.key == pygame.K_n:
                    if not ui.ai_playing and (ui.game_state.status == "won" or not ui.level_has_solution):
                        ui.show_comparison = False
                        next_level()
                    continue

                if event.key == pygame.K_c:
                    if ui.game_state.status == "won":
                        ui.show_comparison = not ui.show_comparison
                        if ui.show_comparison and not ui.comparison_results:
                            ui.comparison_results = compare_algorithms(create_initial_state())
                    continue

                if (
                    event.key == pygame.K_h
                    and ui.game_state.status == "playing"
                    and not ui.show_comparison
                ):
                    request_hint()
                    continue

                if (
                    ui.game_state.status == "playing"
                    and not ui.show_comparison
                ):
                    if event.key == pygame.K_1:
                        start_ai("BFS", bfs_search)
                        continue
                    if event.key == pygame.K_2:
                        start_ai("DFS", dfs_search)
                        continue
                    if event.key == pygame.K_3:
                        start_ai("IDDFS", iddfs_search)
                        continue
                    if event.key == pygame.K_4:
                        start_ai("Greedy", greedy_search)
                        continue
                    if event.key == pygame.K_5:
                        start_ai("A*", astar_search)
                        continue
                    if event.key == pygame.K_6:
                        start_ai("CSP + AC-3", csp_search)
                        continue

                if (
                    ui.game_state.status == "playing"
                    and not ui.show_comparison
                    and not ui.ai_playing
                ):
                    key_map = {
                        pygame.K_UP: "UP",
                        pygame.K_w: "UP",
                        pygame.K_DOWN: "DOWN",
                        pygame.K_s: "DOWN",
                        pygame.K_LEFT: "LEFT",
                        pygame.K_a: "LEFT",
                        pygame.K_RIGHT: "RIGHT",
                        pygame.K_d: "RIGHT",
                    }
                    if event.key in key_map:
                        handle_move(key_map[event.key])

        if (
            not ui.instructions_open
            and not ui.game_paused
            and not ui.show_comparison
            and ui.level_has_solution
            and ui.game_state.status == "playing"
        ):
            if ui.ai_playing and now_ticks >= ui.ai_next_move_time:
                if ui.ai_move_index < len(ui.ai_moves):
                    handle_move(ui.ai_moves[ui.ai_move_index])
                    ui.ai_move_index += 1
                    ui.ai_next_move_time = now_ticks + AI_MOVE_DELAY_MS
                if ui.ai_move_index >= len(ui.ai_moves) or ui.game_state.status != "playing":
                    ui.ai_playing = False

        if ui.instructions_open:
            draw_instructions_screen()
        elif ui.show_comparison:
            draw_comparison_screen()
        else:
            draw_main_screen()
            if ui.game_paused:
                draw_pause_overlay()
            draw_menu_bar()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
