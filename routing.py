"""
routing.py

Unified routing service for the simulation grid, replacing routing_accel.py.

Tiers, in order of preference:
  1. LRU route cache (exact start/goal pairs)
  2. GPU/CPU torch flow fields (batched Jacobi relaxation per goal; a cached
     field answers every route to that goal in O(path length))
  3. Numba-compiled A* (module-level njit kernel, warmed up at startup)
  4. Pure-Python A* (guaranteed fallback)

Every tier increments a counter on `RoutingStats` so the UI can prove which
tier is actually serving routes; accelerator failures are logged and surfaced
instead of being silently swallowed.

Cost model matches the legacy world.find_route:
cost to enter a tile = tile.movement_cost * (0.75 if tile.road else 1.0).
"""

from __future__ import annotations

import heapq
import importlib.util
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("rlss.routing")

_NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if _NUMBA_AVAILABLE:
    try:
        from numba import njit, prange
    except Exception:  # pragma: no cover - broken install
        _NUMBA_AVAILABLE = False

if not _NUMBA_AVAILABLE:  # pragma: no cover - runtime fallback
    prange = range

    def njit(*args, **kwargs):
        def _decorate(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return _decorate

if _TORCH_AVAILABLE:
    try:
        import torch
        import torch.nn.functional as F
    except Exception:  # pragma: no cover - broken install
        _TORCH_AVAILABLE = False

INF = 1e18


# ---------------------------------------------------------------------------
# Numba A* kernel
# ---------------------------------------------------------------------------
# Flat module-level function: no closures / nonlocal (rejected by nopython
# mode), heap sized 4*H*W (a cell can be re-pushed once per incoming edge on
# a 4-connected grid), heap operations inlined.
@njit(cache=True)
def _astar_core(sx, sy, gx, gy, walkable, cost, max_steps, h_scale):  # pragma: no cover - jit
    H, W = walkable.shape
    empty = np.empty((0, 2), dtype=np.int32)
    if sx < 0 or sx >= W or sy < 0 or sy >= H or gx < 0 or gx >= W or gy < 0 or gy >= H:
        return empty
    if not walkable[sy, sx] or not walkable[gy, gx]:
        return empty

    g = np.full((H, W), INF, dtype=np.float64)
    parent = np.full((H, W), -1, dtype=np.int32)
    g[sy, sx] = 0.0

    heap_cap = 4 * H * W
    heap_f = np.empty(heap_cap, dtype=np.float64)
    heap_g = np.empty(heap_cap, dtype=np.float64)
    heap_i = np.empty(heap_cap, dtype=np.int32)
    size = 0

    # push start
    heap_f[0] = (abs(sx - gx) + abs(sy - gy)) * h_scale
    heap_g[0] = 0.0
    heap_i[0] = sy * W + sx
    size = 1

    found = False
    steps = 0
    while size > 0 and steps < max_steps:
        # --- pop root (inlined sift-down) ---
        c = heap_g[0]
        idx = heap_i[0]
        size -= 1
        last_f = heap_f[size]
        last_g = heap_g[size]
        last_i = heap_i[size]
        pos = 0
        while True:
            left = 2 * pos + 1
            if left >= size:
                break
            child = left
            right = left + 1
            if right < size:
                if heap_f[right] < heap_f[left] or (
                    heap_f[right] == heap_f[left] and heap_g[right] < heap_g[left]
                ):
                    child = right
            if heap_f[child] < last_f or (heap_f[child] == last_f and heap_g[child] < last_g):
                heap_f[pos] = heap_f[child]
                heap_g[pos] = heap_g[child]
                heap_i[pos] = heap_i[child]
                pos = child
            else:
                break
        if size > 0:
            heap_f[pos] = last_f
            heap_g[pos] = last_g
            heap_i[pos] = last_i
        # --- end pop ---

        x = idx % W
        y = idx // W
        if c > g[y, x]:
            continue
        if x == gx and y == gy:
            found = True
            break
        steps += 1
        for i in range(4):
            if i == 0:
                nx_, ny_ = x - 1, y
            elif i == 1:
                nx_, ny_ = x + 1, y
            elif i == 2:
                nx_, ny_ = x, y - 1
            else:
                nx_, ny_ = x, y + 1
            if nx_ < 0 or nx_ >= W or ny_ < 0 or ny_ >= H:
                continue
            if not walkable[ny_, nx_]:
                continue
            new_g = c + cost[ny_, nx_]
            if new_g < g[ny_, nx_]:
                g[ny_, nx_] = new_g
                parent[ny_, nx_] = idx
                if size >= heap_cap:
                    continue  # heap saturated; extremely unlikely, degrade gracefully
                # --- push (inlined sift-up) ---
                fv = new_g + (abs(nx_ - gx) + abs(ny_ - gy)) * h_scale
                pos = size
                size += 1
                while pos > 0:
                    p = (pos - 1) // 2
                    if heap_f[p] < fv or (heap_f[p] == fv and heap_g[p] < new_g):
                        break
                    heap_f[pos] = heap_f[p]
                    heap_g[pos] = heap_g[p]
                    heap_i[pos] = heap_i[p]
                    pos = p
                heap_f[pos] = fv
                heap_g[pos] = new_g
                heap_i[pos] = ny_ * W + nx_
                # --- end push ---

    if not found:
        return empty

    # Count path length, then fill output start->goal.
    length = 1
    cx, cy = gx, gy
    while cx != sx or cy != sy:
        pidx = parent[cy, cx]
        if pidx < 0:
            return empty
        cy = pidx // W
        cx = pidx % W
        length += 1
    out = np.empty((length, 2), dtype=np.int32)
    cx, cy = gx, gy
    for i in range(length - 1, -1, -1):
        out[i, 0] = cx
        out[i, 1] = cy
        pidx = parent[cy, cx]
        cy = pidx // W
        cx = pidx % W
    return out



# ---------------------------------------------------------------------------
# Parallel batch A* (reuses per-thread buffers via a visit-stamp trick)
# ---------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _astar_batch(sxs, sys_, gxs, gys, walkable, cost, h_scale, max_len, out, lens, nchunks,
                 pool_g, pool_stamp, pool_parent, pool_hf, pool_hg, pool_hi, stamp_ctr):  # pragma: no cover - jit
    """Batch A*.  Chunk c serves requests c, c+nchunks, c+2*nchunks, ... (round-robin
    for load balance) using pool row c, which persists across calls; the per-row
    visit stamp keeps searches isolated, so each path depends only on its own
    request -- results are identical for any chunking or pool reuse."""
    n = sxs.shape[0]
    H, W = walkable.shape
    for c in prange(nchunks):
        g = pool_g[c]
        stamp = pool_stamp[c]
        parent = pool_parent[c]
        heap_f = pool_hf[c]
        heap_g = pool_hg[c]
        heap_i = pool_hi[c]
        heap_cap = heap_f.shape[0]
        cur = stamp_ctr[c]
        for r in range(c, n, nchunks):
            lens[r] = 0
            sx = sxs[r]
            sy = sys_[r]
            gx = gxs[r]
            gy = gys[r]
            if sx < 0 or sx >= W or sy < 0 or sy >= H or gx < 0 or gx >= W or gy < 0 or gy >= H:
                continue
            if not walkable[sy, sx] or not walkable[gy, gx]:
                continue
            cur += 1
            sidx = sy * W + sx
            gidx = gy * W + gx
            stamp[sidx] = cur
            g[sidx] = 0.0
            parent[sidx] = -1
            heap_f[0] = (abs(sx - gx) + abs(sy - gy)) * h_scale
            heap_g[0] = 0.0
            heap_i[0] = sidx
            hsize = 1
            found = False
            while hsize > 0:
                cg = heap_g[0]
                idx = heap_i[0]
                hsize -= 1
                last_f = heap_f[hsize]
                last_g = heap_g[hsize]
                last_i = heap_i[hsize]
                pos = 0
                while True:
                    left = 2 * pos + 1
                    if left >= hsize:
                        break
                    child = left
                    right = left + 1
                    if right < hsize:
                        if heap_f[right] < heap_f[left] or (heap_f[right] == heap_f[left] and heap_g[right] < heap_g[left]):
                            child = right
                    if heap_f[child] < last_f or (heap_f[child] == last_f and heap_g[child] < last_g):
                        heap_f[pos] = heap_f[child]
                        heap_g[pos] = heap_g[child]
                        heap_i[pos] = heap_i[child]
                        pos = child
                    else:
                        break
                if hsize > 0:
                    heap_f[pos] = last_f
                    heap_g[pos] = last_g
                    heap_i[pos] = last_i
                if cg > g[idx]:
                    continue
                if idx == gidx:
                    found = True
                    break
                x = idx % W
                y = idx // W
                for k in range(4):
                    if k == 0:
                        nx_ = x - 1
                        ny_ = y
                    elif k == 1:
                        nx_ = x + 1
                        ny_ = y
                    elif k == 2:
                        nx_ = x
                        ny_ = y - 1
                    else:
                        nx_ = x
                        ny_ = y + 1
                    if nx_ < 0 or nx_ >= W or ny_ < 0 or ny_ >= H:
                        continue
                    if not walkable[ny_, nx_]:
                        continue
                    nidx = ny_ * W + nx_
                    new_g = cg + cost[ny_, nx_]
                    if stamp[nidx] != cur or new_g < g[nidx]:
                        stamp[nidx] = cur
                        g[nidx] = new_g
                        parent[nidx] = idx
                        if hsize >= heap_cap:
                            continue
                        fv = new_g + (abs(nx_ - gx) + abs(ny_ - gy)) * h_scale
                        pos = hsize
                        hsize += 1
                        while pos > 0:
                            pp = (pos - 1) // 2
                            if heap_f[pp] < fv or (heap_f[pp] == fv and heap_g[pp] < new_g):
                                break
                            heap_f[pos] = heap_f[pp]
                            heap_g[pos] = heap_g[pp]
                            heap_i[pos] = heap_i[pp]
                            pos = pp
                        heap_f[pos] = fv
                        heap_g[pos] = new_g
                        heap_i[pos] = nidx
            if not found:
                continue
            length = 1
            node = gidx
            while node != sidx:
                node = parent[node]
                length += 1
            if length > max_len:
                lens[r] = -1
                continue
            node = gidx
            for i in range(length - 1, -1, -1):
                out[r, i, 0] = node % W
                out[r, i, 1] = node // W
                node = parent[node]
            lens[r] = length
        stamp_ctr[c] = cur


def astar_python(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walkable: np.ndarray,
    cost: np.ndarray,
    max_steps: int = 2000,
) -> List[Tuple[int, int]]:
    """Guaranteed pure-Python A* fallback on the routing arrays."""
    H, W = walkable.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
        return []
    if not walkable[sy, sx] or not walkable[gy, gx]:
        return []
    # Admissible heuristic: manhattan distance x the cheapest entry cost
    # (roads cost < 1.0, so a plain manhattan heuristic overestimates and
    # produces suboptimal paths).
    h_scale = float(cost[walkable].min()) if walkable.any() else 1.0
    dist: Dict[Tuple[int, int], float] = {start: 0.0}
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    heap: List[Tuple[float, float, Tuple[int, int]]] = [((abs(sx - gx) + abs(sy - gy)) * h_scale, 0.0, start)]
    steps = 0
    while heap and steps < max_steps:
        _, c, current = heapq.heappop(heap)
        if c > dist.get(current, float("inf")):
            continue
        if current == goal:
            break
        steps += 1
        cx, cy = current
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H) or not walkable[ny, nx]:
                continue
            new_cost = c + float(cost[ny, nx])
            neighbor = (nx, ny)
            if new_cost < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(heap, (new_cost + (abs(nx - gx) + abs(ny - gy)) * h_scale, new_cost, neighbor))
    if goal not in came_from:
        return []
    path: List[Tuple[int, int]] = []
    node: Optional[Tuple[int, int]] = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Torch flow-field router
# ---------------------------------------------------------------------------
def _select_torch_device() -> "torch.device":
    """Resolve the routing accelerator from RLSS_ROUTING_DEVICE.

    Accepted values: auto (default), cuda, cuda:N, mps, cpu, gpu.
    Unsupported requests fall back to the best available device.
    """
    requested = os.environ.get("RLSS_ROUTING_DEVICE", "auto").strip().lower()
    has_mps = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "gpu"} or requested.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(requested if requested.startswith("cuda:") else "cuda")
        logger.warning("RLSS_ROUTING_DEVICE=%s requested, but CUDA is unavailable", requested)
    elif requested == "mps":
        if has_mps:
            return torch.device("mps")
        logger.warning("RLSS_ROUTING_DEVICE=mps requested, but MPS is unavailable")
    elif requested not in {"auto", ""}:
        logger.warning("Unknown RLSS_ROUTING_DEVICE=%s; using auto selection", requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


def _torch_backend_name(device: "torch.device") -> str:
    return f"torch-{device.type}"


class GPUGridRouter:
    """
    Batched shortest-path flow fields on a weighted 4-connected grid via
    Jacobi relaxation.  One converged field per goal answers *every* route
    to that goal in O(path length) on the CPU.

    Fixes over the legacy implementation:
      - goal anchoring via a single scatter (no per-goal Python loop inside
        the iteration)
      - no per-iteration ``.item()`` device sync: convergence is checked
        every 16 iterations only
      - LRU-capped field cache with on-demand batched computation, so any
        goal (not just building tiles) is servable
      - runs on CUDA when available, but also works (and is still fast) on
        CPU tensors
    """

    DX = (0, 0, -1, 1)  # 0 = up, 1 = down, 2 = left, 3 = right
    DY = (-1, 1, 0, 0)

    def __init__(self, walkable: np.ndarray, cost: np.ndarray, max_fields: int = 768) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for GPUGridRouter")
        self.H, self.W = walkable.shape
        self.device = _select_torch_device()
        self.device_name = str(self.device)
        self.walkable_np = walkable
        self.walkable_t = torch.from_numpy(walkable).to(self.device)
        self.cost_t = torch.from_numpy(cost.astype(np.float32, copy=False)).to(self.device)
        self.max_fields = max_fields
        # goal -> (dist ndarray, next_dir ndarray), LRU ordered
        self._cache: "OrderedDict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    # -- field computation ------------------------------------------------
    def _compute_fields(self, goals: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
        valid = [
            (gx, gy)
            for gx, gy in goals
            if 0 <= gx < self.W and 0 <= gy < self.H and self.walkable_np[gy, gx]
        ]
        if not valid:
            return {}
        B = len(valid)
        H, W = self.H, self.W

        b_idx = torch.arange(B, device=self.device)
        gx_idx = torch.tensor([g[0] for g in valid], device=self.device, dtype=torch.long)
        gy_idx = torch.tensor([g[1] for g in valid], device=self.device, dtype=torch.long)

        dist = torch.full((B, H, W), 1e9, dtype=torch.float32, device=self.device)
        dist[b_idx, gy_idx, gx_idx] = 0.0

        walkable_b = self.walkable_t.unsqueeze(0)
        pad = (1, 1, 1, 1)
        padded_cost = F.pad(self.cost_t.unsqueeze(0), pad, value=1e9)
        cost_up = padded_cost[:, :H, 1:W + 1]
        cost_down = padded_cost[:, 2:H + 2, 1:W + 1]
        cost_left = padded_cost[:, 1:H + 1, :W]
        cost_right = padded_cost[:, 1:H + 1, 2:W + 2]

        max_iters = 2 * (H + W)
        check_every = 16
        for it in range(max_iters):
            padded = F.pad(dist, pad, value=1e9)
            best = torch.minimum(
                torch.minimum(
                    padded[:, :H, 1:W + 1] + cost_up,
                    padded[:, 2:H + 2, 1:W + 1] + cost_down,
                ),
                torch.minimum(
                    padded[:, 1:H + 1, :W] + cost_left,
                    padded[:, 1:H + 1, 2:W + 2] + cost_right,
                ),
            )
            best = torch.where(walkable_b, best, torch.full_like(best, 1e9))
            new_dist = torch.minimum(dist, best)
            new_dist[b_idx, gy_idx, gx_idx] = 0.0
            if (it + 1) % check_every == 0:
                if bool(torch.equal(new_dist, dist)):
                    dist = new_dist
                    break
            dist = new_dist

        # next-step direction: argmin over neighbour dist + neighbour entry cost
        padded = F.pad(dist, pad, value=1e9)
        candidates = torch.stack(
            (
                padded[:, :H, 1:W + 1] + cost_up,
                padded[:, 2:H + 2, 1:W + 1] + cost_down,
                padded[:, 1:H + 1, :W] + cost_left,
                padded[:, 1:H + 1, 2:W + 2] + cost_right,
            ),
            dim=1,
        )
        next_dir = torch.argmin(candidates, dim=1).to(torch.int8)
        unreachable = dist >= 1e8
        next_dir = torch.where(unreachable, torch.full_like(next_dir, -1), next_dir)
        next_dir[b_idx, gy_idx, gx_idx] = -1

        dist_np = dist.cpu().numpy()
        next_np = next_dir.cpu().numpy()
        return {valid[b]: (dist_np[b], next_np[b]) for b in range(B)}

    def ensure_destinations(self, goals: Iterable[Tuple[int, int]], batch_size: int = 256) -> None:
        missing = [g for g in dict.fromkeys(goals) if g not in self._cache]
        for i in range(0, len(missing), batch_size):
            fields = self._compute_fields(missing[i:i + batch_size])
            for goal, payload in fields.items():
                self._cache[goal] = payload
        self._evict()

    def _evict(self) -> None:
        while len(self._cache) > self.max_fields:
            self._cache.popitem(last=False)

    def has_destination(self, goal: Tuple[int, int]) -> bool:
        return goal in self._cache

    def cached_field_count(self) -> int:
        return len(self._cache)

    def invalidate(self) -> None:
        self._cache.clear()

    # -- queries ----------------------------------------------------------
    def find_route(self, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = 2000) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]
        field_pair = self._cache.get(goal)
        if field_pair is None:
            return []
        self._cache.move_to_end(goal)
        _, nxt = field_pair
        sx, sy = start
        gx, gy = goal
        if not (0 <= sx < self.W and 0 <= sy < self.H):
            return []
        if not self.walkable_np[sy, sx] or nxt[sy, sx] == -1:
            return []
        path = [start]
        cx, cy = sx, sy
        for _ in range(max_steps):
            d = int(nxt[cy, cx])
            if d < 0:
                break
            cx += self.DX[d]
            cy += self.DY[d]
            path.append((cx, cy))
            if cx == gx and cy == gy:
                return path
        return []

    def route_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        if start == goal:
            return 0.0
        field_pair = self._cache.get(goal)
        if field_pair is None:
            return float("inf")
        dist, _ = field_pair
        sx, sy = start
        if 0 <= sx < self.W and 0 <= sy < self.H:
            return float(dist[sy, sx])
        return float("inf")


# ---------------------------------------------------------------------------
# Route service
# ---------------------------------------------------------------------------
@dataclass
class RoutingStats:
    """Per-tier telemetry, surfaced in both frontends."""
    # gpu_* = torch flow-field tier; cuda_* = CUDA A* kernel (cuda_routing.py)

    cache_hits: int = 0
    gpu_hits: int = 0
    numba_hits: int = 0
    python_hits: int = 0
    failures: int = 0
    field_batches: int = 0
    invalidations: int = 0
    gpu_error: Optional[str] = None
    numba_error: Optional[str] = None
    last_batch_ms: float = 0.0
    gpu_fields: int = 0
    gpu_device: str = "cpu"
    backend: str = "python"
    cuda_hits: int = 0              # A* searches served by the CUDA kernel
    cuda_mode: str = "off"          # off | cuda | hybrid (see cuda_routing.py)
    cuda_device: str = "none"
    cuda_note: Optional[str] = None

    def total(self) -> int:
        return self.cache_hits + self.gpu_hits + self.numba_hits + self.cuda_hits + self.python_hits

    def summary(self) -> str:
        total = max(1, self.total())
        parts = [
            f"RT[{self.backend}]",
            f"dev:{self.gpu_device}",
            f"gpu:{self.gpu_hits}",
            f"nb:{self.numba_hits}",
            f"cuda[{self.cuda_mode}]:{self.cuda_hits}",
            f"py:{self.python_hits}",
            f"cache:{100 * self.cache_hits // total}%",
            f"fields:{self.gpu_fields}",
        ]
        if self.last_batch_ms:
            parts.append(f"{self.last_batch_ms:.1f}ms")
        if self.gpu_error:
            parts.append(f"GPU-ERR:{self.gpu_error[:40]}")
        if self.numba_error:
            parts.append(f"NB-ERR:{self.numba_error[:40]}")
        return " ".join(parts)


ROUTE_CACHE_LIMIT = 30000
BATCH_MAX_PATH = 1024
_STAMP_RESET_AT = 2_000_000_000


def _make_astar_pool(rows: int, size: int) -> tuple:
    """Per-thread A* scratch memory (g, stamp, parent, heap f/g/idx, stamp counters)."""
    heap_cap = 4 * size
    return (
        np.empty((rows, size), dtype=np.float64),
        np.zeros((rows, size), dtype=np.int32),
        np.empty((rows, size), dtype=np.int32),
        np.empty((rows, heap_cap), dtype=np.float64),
        np.empty((rows, heap_cap), dtype=np.float64),
        np.empty((rows, heap_cap), dtype=np.int32),
        np.zeros(rows, dtype=np.int32),
    )


class RouteService:
    """Owns the routing arrays, accelerators, caches, and telemetry."""

    def __init__(self, walkable: np.ndarray, cost: np.ndarray, vcost: Optional[np.ndarray] = None,
                 gpu_mode: Optional[str] = None) -> None:
        self.walkable = np.ascontiguousarray(walkable, dtype=np.bool_)
        self.cost = np.ascontiguousarray(cost, dtype=np.float32)
        self._cost64 = self.cost.astype(np.float64)
        self._h_scale = float(self._cost64[self.walkable].min()) if self.walkable.any() else 1.0
        self.vcost = np.ascontiguousarray(vcost if vcost is not None else cost, dtype=np.float32)
        self._vcost64 = self.vcost.astype(np.float64)
        self._h_scale_v = float(self._vcost64[self.walkable].min()) if self.walkable.any() else 1.0
        self.stats = RoutingStats()
        self._route_cache: "OrderedDict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], ...]]" = OrderedDict()
        self._array_cache: "OrderedDict[Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray]" = OrderedDict()
        self._batch_ok = False
        # Leave one hardware thread for the render process; the sim thread itself
        # just waits on the kernel, so it does not need a core of its own.
        self._n_threads = max(1, (os.cpu_count() or 2) - 1)
        self._numba_ok = False
        self._astar_pool: Optional[tuple] = None
        self._gpu: Optional[GPUGridRouter] = None
        self._cuda = None                # cuda_routing.CudaAStar when enabled
        self._cuda_split = None          # cuda_routing.AdaptiveSplit (hybrid mode)
        self._cuda_auto = False
        self._cuda_batches = 0
        self._init_numba()
        self._init_gpu()
        self._init_cuda(gpu_mode)
        self._refresh_backend()

    # -- accelerator init -------------------------------------------------
    def _init_numba(self) -> None:
        if not _NUMBA_AVAILABLE:
            self.stats.numba_error = "numba not installed"
            return
        try:
            # Warm up (JIT-compile) on a tiny grid so cost isn't paid mid-sim.
            small_w = np.ones((4, 4), dtype=np.bool_)
            small_c = np.ones((4, 4), dtype=np.float64)
            _astar_core(0, 0, 3, 3, small_w, small_c, 64, 1.0)
            self._numba_ok = True
            try:
                out = np.zeros((2, 16, 2), dtype=np.int16)
                lens = np.zeros(2, dtype=np.int32)
                z = np.zeros(2, dtype=np.int32)
                _astar_batch(z, z, z + 3, z + 3, small_w, small_c, 1.0, 16, out, lens, 2,
                             *_make_astar_pool(2, small_w.size))
                self._batch_ok = bool(lens[0] > 0)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Numba batch A* warmup failed")
                self._batch_ok = False
                self.stats.numba_error = f"batch: {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - must not kill the sim
            logger.exception("Numba A* warmup failed")
            self.stats.numba_error = f"{type(exc).__name__}: {exc}"

    def _init_gpu(self) -> None:
        if not _TORCH_AVAILABLE:
            self.stats.gpu_error = "torch not installed"
            return
        try:
            self._gpu = GPUGridRouter(self.walkable, self.cost)
            self.stats.gpu_device = self._gpu.device_name
        except Exception as exc:  # noqa: BLE001 - must not kill the sim
            logger.exception("GPU router init failed")
            self._gpu = None
            self.stats.gpu_error = f"{type(exc).__name__}: {exc}"

    def _init_cuda(self, gpu_mode: Optional[str]) -> None:
        """Optional CUDA A* (exactly the same paths as the CPU kernel; see cuda_routing.py)."""
        try:
            import cuda_routing
        except Exception as exc:  # noqa: BLE001
            self.stats.cuda_note = f"unavailable: {type(exc).__name__}: {exc}"
            return
        requested = (gpu_mode or os.environ.get("RLSS_GPU") or "auto").strip().lower()
        mode = cuda_routing.resolve_mode(requested)
        if mode == "off" or not self._batch_ok:
            if requested not in ("off", "auto"):
                self.stats.cuda_note = "no usable CUDA device" if self._batch_ok else "CPU batch kernel unavailable"
            return
        try:
            self._cuda = cuda_routing.CudaAStar()
            # Compile + validate on a tiny grid now, not mid-simulation.
            small_w = np.ones((4, 4), dtype=np.bool_)
            small_c = np.ones((4, 4), dtype=np.float64)
            z = np.zeros(2, dtype=np.int32)
            _, lens = self._cuda.run(z, z, z + 3, z + 3, small_w, small_c, 1.0, -1, 16)
            if int(lens[0]) != 7:
                raise RuntimeError(f"CUDA A* self-test returned length {int(lens[0])}")
            self._cuda.invalidate()
        except Exception as exc:  # noqa: BLE001 - must not kill the sim
            logger.exception("CUDA A* init failed")
            self._cuda = None
            self.stats.cuda_note = f"init failed: {type(exc).__name__}: {exc}"
            return
        self.stats.cuda_mode = mode
        self.stats.cuda_device = self._cuda.device_name
        self._cuda_auto = requested == "auto"
        if mode == "hybrid":
            self._cuda_split = cuda_routing.AdaptiveSplit()

    def _disable_cuda(self, note: str) -> None:
        self._cuda = None
        self._cuda_split = None
        self.stats.cuda_mode = "off"
        self.stats.cuda_note = note

    def _astar_tier2(self, sx, sy, gx, gy, cost64, h_scale, kind: int):
        """Run one A* batch on the CPU kernel, the GPU kernel, or both concurrently
        (GPU kernels launch asynchronously, the CPU share runs meanwhile).  Every
        search is independent and both kernels are identical, so where a search
        runs never changes its path."""
        m = sx.shape[0]
        cuda_n = 0
        if self._cuda is not None:
            if self._cuda_split is None:
                cuda_n = m
            elif m >= 8:
                cuda_n = int(round(m * self._cuda_split.gpu_share))
        cpu_n = m - cuda_n
        out = np.empty((m, BATCH_MAX_PATH, 2), dtype=np.int16)
        lens = np.zeros(m, dtype=np.int32)
        handle = None
        t0 = time.perf_counter()
        if cuda_n:
            try:
                handle = self._cuda.launch(sx[cpu_n:], sy[cpu_n:], gx[cpu_n:], gy[cpu_n:],
                                           self.walkable, cost64, h_scale, kind, BATCH_MAX_PATH)
            except Exception as exc:  # noqa: BLE001
                logger.exception("CUDA A* launch failed")
                self._disable_cuda(f"launch failed: {type(exc).__name__}: {exc}")
                cuda_n, cpu_n = 0, m
        t1 = time.perf_counter()
        if cpu_n:
            pool = self._astar_pool_for(self.walkable.size)
            _astar_batch(sx[:cpu_n], sy[:cpu_n], gx[:cpu_n], gy[:cpu_n], self.walkable, cost64, h_scale,
                         BATCH_MAX_PATH, out[:cpu_n], lens[:cpu_n], int(min(cpu_n, self._n_threads)), *pool)
        t2 = time.perf_counter()
        if handle is not None:
            try:
                g_out, g_lens = self._cuda.collect(handle)
                out[cpu_n:] = g_out
                lens[cpu_n:] = g_lens
            except Exception as exc:  # noqa: BLE001 - recompute that share on the CPU
                logger.exception("CUDA A* collect failed")
                self._disable_cuda(f"collect failed: {type(exc).__name__}: {exc}")
                pool = self._astar_pool_for(self.walkable.size)
                _astar_batch(sx[cpu_n:], sy[cpu_n:], gx[cpu_n:], gy[cpu_n:], self.walkable, cost64, h_scale,
                             BATCH_MAX_PATH, out[cpu_n:], lens[cpu_n:], int(min(cuda_n, self._n_threads)), *pool)
                return out, lens, m
            t3 = time.perf_counter()
            split = self._cuda_split
            if split is not None:
                # The GPU ran from launch to collect; the CPU kernel ran t1..t2.
                split.record(cuda_n, t3 - t0, cpu_n, t2 - t1)
                self._cuda_batches += 1
                if self._cuda_auto and self._cuda_batches >= 20 and split.gpu_share < 0.1:
                    self._disable_cuda(f"auto: CUDA A* slower than the CPU kernel here ({split.summary()}); CPU only")
        return out, lens, cpu_n

    def _refresh_backend(self) -> None:
        if self._gpu is not None:
            self.stats.gpu_device = self._gpu.device_name
            self.stats.backend = _torch_backend_name(self._gpu.device)
        else:
            self.stats.gpu_device = "none"
            self.stats.gpu_fields = 0
            self.stats.backend = "numba" if self._numba_ok else "python"

    def _astar_pool_for(self, size: int) -> tuple:
        pool = self._astar_pool
        if pool is None or pool[0].shape[1] != size or pool[0].shape[0] < self._n_threads:
            pool = self._astar_pool = _make_astar_pool(self._n_threads, size)
        elif int(pool[6].max()) > _STAMP_RESET_AT:
            pool[1].fill(0)
            pool[6].fill(0)
        return pool

    # -- cache / invalidation ---------------------------------------------
    def invalidate(self, walkable: Optional[np.ndarray] = None, cost: Optional[np.ndarray] = None, vcost: Optional[np.ndarray] = None) -> None:
        """Clear all cached routes/fields; call when walkability or costs change."""
        if vcost is not None:
            self.vcost = np.ascontiguousarray(vcost, dtype=np.float32)
            self._vcost64 = self.vcost.astype(np.float64)
        if walkable is not None:
            self.walkable = np.ascontiguousarray(walkable, dtype=np.bool_)
        if cost is not None:
            self.cost = np.ascontiguousarray(cost, dtype=np.float32)
            self._cost64 = self.cost.astype(np.float64)
            self._h_scale = float(self._cost64[self.walkable].min()) if self.walkable.any() else 1.0
        self._h_scale_v = float(self._vcost64[self.walkable].min()) if self.walkable.any() else 1.0
        self._route_cache.clear()
        self._array_cache.clear()
        self.stats.invalidations += 1
        if self._cuda is not None:
            self._cuda.invalidate()
        if self._gpu is not None:
            try:
                self._gpu = GPUGridRouter(self.walkable, self.cost, max_fields=self._gpu.max_fields)
                self.stats.gpu_device = self._gpu.device_name
            except Exception as exc:  # noqa: BLE001
                logger.exception("GPU router rebuild failed")
                self._gpu = None
                self.stats.gpu_error = f"{type(exc).__name__}: {exc}"
        self._refresh_backend()

    def precompute_goals(self, goals: Iterable[Tuple[int, int]]) -> None:
        """Batch-build flow fields for popular destinations (buildings, parks...)."""
        if self._gpu is None:
            return
        try:
            t0 = time.perf_counter()
            self._gpu.ensure_destinations(goals)
            self.stats.field_batches += 1
            self.stats.last_batch_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.gpu_fields = self._gpu.cached_field_count()
            self._refresh_backend()
        except Exception as exc:  # noqa: BLE001
            logger.exception("GPU field precompute failed")
            self._gpu = None
            self.stats.gpu_error = f"{type(exc).__name__}: {exc}"
            self._refresh_backend()

    # -- routing ----------------------------------------------------------
    def find_route(self, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = 2000) -> List[Tuple[int, int]]:
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        if start == goal:
            return [start]

        key = (start, goal)
        cached = self._route_cache.get(key)
        if cached is not None:
            self._route_cache.move_to_end(key)
            self.stats.cache_hits += 1
            return list(cached)

        path: List[Tuple[int, int]] = []
        # A* tiers treat the limit as a node-expansion budget; 2000 is too
        # small for far goals on this grid, so give them the whole grid.
        expand_budget = max(max_steps, self.walkable.size)

        # Tier 1: GPU flow field
        if self._gpu is not None and self._gpu.has_destination(goal):
            try:
                path = self._gpu.find_route(start, goal, max_steps=max_steps)
                if path:
                    self.stats.gpu_hits += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("GPU route query failed")
                self._gpu = None
                self.stats.gpu_error = f"{type(exc).__name__}: {exc}"
                self._refresh_backend()
                path = []

        # Tier 2: Numba A*
        if not path and self._numba_ok:
            try:
                arr = _astar_core(
                    start[0], start[1], goal[0], goal[1],
                    self.walkable, self._cost64, expand_budget, self._h_scale,
                )
                if arr.shape[0]:
                    path = [(int(x), int(y)) for x, y in arr]
                    self.stats.numba_hits += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("Numba route query failed")
                self._numba_ok = False
                self.stats.numba_error = f"{type(exc).__name__}: {exc}"
                path = []

        # Tier 3: pure Python
        if not path:
            path = astar_python(start, goal, self.walkable, self.cost, expand_budget)
            if path:
                self.stats.python_hits += 1

        if not path:
            self.stats.failures += 1
            return []

        self._route_cache[key] = tuple(path)
        if len(self._route_cache) > ROUTE_CACHE_LIMIT:
            self._route_cache.popitem(last=False)
        return path

    def batch_routes(self, requests: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]], max_steps: int = 2000) -> List[List[Tuple[int, int]]]:
        """Serve many (start, goal) requests, precomputing flow fields for
        goals requested more than once so the GPU tier answers them."""
        if self._gpu is not None and requests:
            goal_counts: Dict[Tuple[int, int], int] = {}
            for _, goal in requests:
                g = (int(goal[0]), int(goal[1]))
                goal_counts[g] = goal_counts.get(g, 0) + 1
            hot = [g for g, n in goal_counts.items() if n >= 2 and not self._gpu.has_destination(g)]
            if hot:
                self.precompute_goals(hot)
        return [self.find_route(s, g, max_steps=max_steps) for s, g in requests]

    # -- array API (hot path used by Population) ------------------------------
    # kind 0 = pedestrian grid, kind 1 = vehicle grid (roads strongly preferred).
    def find_route_array(self, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = 2000, kind: int = 0) -> np.ndarray:
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        key = (start, goal, kind)
        cached = self._array_cache.get(key)
        if cached is not None:
            self._array_cache.move_to_end(key)
            self.stats.cache_hits += 1
            return cached
        return self.batch_routes_arrays([(start, goal)], max_steps=max_steps, kinds=[kind])[0]

    def batch_routes_arrays(self, requests: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]], max_steps: int = 2000,
                            kinds: Optional[Sequence[int]] = None) -> List[np.ndarray]:
        """Serve many routes at once as (n, 2) int16 arrays (start..goal inclusive).

        Cache hits are answered immediately; misses go through a parallel
        numba A* batch (or the legacy single-route tiers as a fallback)."""
        results: List[Optional[np.ndarray]] = [None] * len(requests)
        miss: Dict[int, List[Tuple[int, Tuple]]] = {0: [], 1: []}
        seen: Dict[Tuple, int] = {}
        dup: List[Tuple[int, int]] = []
        cache = self._array_cache
        for i, (s_, g_) in enumerate(requests):
            kind = 0 if kinds is None else int(kinds[i])
            key = ((int(s_[0]), int(s_[1])), (int(g_[0]), int(g_[1])), kind)
            hit = cache.get(key)
            if hit is not None:
                results[i] = hit
                self.stats.cache_hits += 1
                continue
            if key[0] == key[1]:
                results[i] = np.array([key[0]], dtype=np.int16)
                continue
            first = seen.get(key)
            if first is not None:
                dup.append((i, first))
                continue
            seen[key] = i
            miss[kind].append((i, key))
        for kind, items in miss.items():
            if not items:
                continue
            computed = self._compute_arrays([k for _, k in items], max_steps, kind)
            for (i, key), arr in zip(items, computed):
                results[i] = arr
                if arr.shape[0]:
                    cache[key] = arr
        while len(cache) > ROUTE_CACHE_LIMIT:
            cache.popitem(last=False)
        for i, first in dup:
            results[i] = results[first]
        return [r if r is not None else np.empty((0, 2), dtype=np.int16) for r in results]

    def _compute_arrays(self, keys, max_steps: int, kind: int = 0) -> List[np.ndarray]:
        n = len(keys)
        out_arrays: List[Optional[np.ndarray]] = [None] * n
        remaining = list(range(n))
        cost64 = self._cost64 if kind == 0 else self._vcost64
        h_scale = self._h_scale if kind == 0 else self._h_scale_v
        cost32 = self.cost if kind == 0 else self.vcost
        # Tier 1: GPU flow fields (pedestrian grid only) if the goal is already resident.
        if self._gpu is not None and kind == 0:
            still = []
            for i in remaining:
                s_, g_ = keys[i][0], keys[i][1]
                path = []
                if self._gpu.has_destination(g_):
                    try:
                        path = self._gpu.find_route(s_, g_, max_steps=max_steps)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("GPU route query failed")
                        self._gpu = None
                        self.stats.gpu_error = f"{type(exc).__name__}: {exc}"
                        self._refresh_backend()
                if path:
                    self.stats.gpu_hits += 1
                    out_arrays[i] = np.asarray(path, dtype=np.int16)
                else:
                    still.append(i)
            remaining = still
        # Tier 2: parallel numba A*
        if remaining and self._batch_ok:
            try:
                m = len(remaining)
                sx = np.fromiter((keys[i][0][0] for i in remaining), dtype=np.int32, count=m)
                sy = np.fromiter((keys[i][0][1] for i in remaining), dtype=np.int32, count=m)
                gx = np.fromiter((keys[i][1][0] for i in remaining), dtype=np.int32, count=m)
                gy = np.fromiter((keys[i][1][1] for i in remaining), dtype=np.int32, count=m)
                out, lens, cpu_n = self._astar_tier2(sx, sy, gx, gy, cost64, h_scale, kind)
                still = []
                for j, i in enumerate(remaining):
                    length = int(lens[j])
                    if length > 0:
                        out_arrays[i] = out[j, :length].copy()
                        if j < cpu_n:
                            self.stats.numba_hits += 1
                        else:
                            self.stats.cuda_hits += 1
                    else:
                        still.append(i)
                remaining = still
            except Exception as exc:  # noqa: BLE001
                logger.exception("Numba batch route failed")
                self._batch_ok = False
                self.stats.numba_error = f"batch: {type(exc).__name__}: {exc}"
        # Tier 3: single-route python fallback
        for i in remaining:
            s_, g_ = keys[i][0], keys[i][1]
            path = astar_python(s_, g_, self.walkable, cost32, max(max_steps, self.walkable.size))
            if path:
                self.stats.python_hits += 1
                out_arrays[i] = np.asarray(path, dtype=np.int16)
            else:
                self.stats.failures += 1
                out_arrays[i] = np.empty((0, 2), dtype=np.int16)
        return [a if a is not None else np.empty((0, 2), dtype=np.int16) for a in out_arrays]

    def route_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        if self._gpu is not None:
            d = self._gpu.route_distance(start, goal)
            if d != float("inf"):
                return d
        path = self.find_route(start, goal)
        if len(path) < 2:
            return 0.0
        return float(sum(self.cost[y, x] for x, y in path[1:]))


def make_entry_cost_grid(tiles: List[List[Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build routing arrays from the world tile grid: (walkable, pedestrian cost, vehicle cost).

    Vehicles strongly prefer roads: off-road ground costs 4x, so cars only cut
    across lots for the last tile or two (driveways)."""
    H = len(tiles)
    W = len(tiles[0])
    walkable = np.empty((H, W), dtype=np.bool_)
    cost = np.empty((H, W), dtype=np.float32)
    vcost = np.empty((H, W), dtype=np.float32)
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            walkable[y, x] = tile.is_walkable()
            c = tile.movement_cost
            if tile.road:
                c *= 0.75
                vcost[y, x] = c
            else:
                vcost[y, x] = max(c, 1.0) * 4.0
            cost[y, x] = float(c)
    return walkable, cost, vcost
