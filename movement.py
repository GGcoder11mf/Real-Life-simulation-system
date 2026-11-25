"""
movement.py

Compiled vehicle driving.  One call advances one vehicle through a sub-step,
stopping at red signals, writing a waypoint per tile entry (and a repeated
point to mark a wait).  Identical rules to the old Python loop, ~10x faster.
"""

from __future__ import annotations

import numpy as np

from routing import njit
from world import SIGNAL_CYCLE, SIGNAL_GREEN

MAX_PTS = 96
_PTS = np.zeros((MAX_PTS, 2), dtype=np.int16)
_TIMES = np.zeros(MAX_PTS, dtype=np.float32)


@njit(cache=True)
def _drive(route, i, px, py, acc, dt, seconds, sig, t_abs, cycle, green, out_pts, out_t):
    n = route.shape[0]
    cap = out_pts.shape[0] - 2
    out_pts[0, 0] = px
    out_pts[0, 1] = py
    out_t[0] = 0.0
    k = 1
    t = -acc
    blocked = False
    while i < n and k < cap:
        x = route[i, 0]
        y = route[i, 1]
        jid = sig[y, x]
        if jid != 0 and sig[py, px] != jid:
            axis = 0 if x == px else 1
            t0 = t if t > 0.0 else 0.0
            phase = (t_abs + t0 + jid * 37.0) % cycle
            d = (phase - (0.0 if axis == 0 else cycle / 2.0)) % cycle
            wait = 0.0 if d < green else cycle - d
            if wait > 0.0:
                t_go = t0 + wait
                if t_go >= seconds:
                    out_pts[k, 0] = px
                    out_pts[k, 1] = py
                    out_t[k] = seconds
                    k += 1
                    blocked = True
                    break
                out_pts[k, 0] = px
                out_pts[k, 1] = py
                out_t[k] = t_go
                k += 1
                t = t_go
        t_new = t + dt
        if t_new > seconds:
            break
        px = x
        py = y
        out_pts[k, 0] = x
        out_pts[k, 1] = y
        out_t[k] = t_new if t_new > 0.0 else 0.0
        k += 1
        t = t_new
        i += 1
    return k, i, px, py, blocked, t


def signal_array(world) -> np.ndarray:
    grid = world.signal_grid
    cached = getattr(world, "_sig_np", None)
    if cached is None or getattr(world, "_sig_src", None) is not grid:
        cached = np.asarray(grid, dtype=np.int32) if grid else np.zeros((world.height, world.width), dtype=np.int32)
        world._sig_np = cached
        world._sig_src = grid
    return cached


def drive_vehicle(route: np.ndarray, i: int, px: int, py: int, acc: float, dt: float,
                  seconds: float, sig: np.ndarray, t_abs: float):
    """Returns (pts int16 (k,2), times float32 (k,), new_i, px, py, acc_out, finished)."""
    route = np.ascontiguousarray(route, dtype=np.int16) if route.dtype != np.int16 or not route.flags.c_contiguous else route
    k, i2, nx, ny, blocked, t = _drive(route, i, px, py, acc, dt, seconds, sig, t_abs,
                                       SIGNAL_CYCLE, SIGNAL_GREEN, _PTS, _TIMES)
    n = route.shape[0]
    acc_out = 0.0 if (blocked or i2 >= n) else max(0.0, seconds - t)
    return _PTS[:k].copy(), _TIMES[:k].copy(), i2, nx, ny, acc_out, i2 >= n


# ---------------------------------------------------------------------------
# Batched driving: every vehicle of a sub-step in one compiled call.
# A vehicle's sub-step depends only on its own route/position/acc/mode, the
# signal grid and the clock, so computing all of them up front and applying
# the results in mover order is exactly equivalent to driving one at a time.
# ---------------------------------------------------------------------------
_SLICE = MAX_PTS - 1   # > max tiles one sub-step can consume (cap - 1) + the start-tile skip


@njit(cache=True)
def _drive_batch(buf, starts, lens, full, pxs, pys, accs, modes, tph, seconds, sig, t_abs, cycle, green,
                 out_pts, out_t, out_k, out_i, out_px, out_py, out_acc, out_state):  # pragma: no cover - jit
    for v in range(starts.shape[0]):
        L = lens[v]
        route = buf[starts[v]:starts[v] + L]
        px = pxs[v]
        py = pys[v]
        i = 0
        if i < L and route[0, 0] == px and route[0, 1] == py:
            i = 1
        if i >= L:
            out_state[v] = 0          # nothing left to drive
            continue
        tv = tph[modes[v]]
        dt = 3600.0 / (tv if tv > 1.0 else 1.0)
        k, i2, nx, ny, blocked, t = _drive(route, i, px, py, accs[v], dt, seconds, sig, t_abs,
                                           cycle, green, out_pts[v], out_t[v])
        finished = full[v] and i2 >= L
        out_k[v] = k
        out_i[v] = i2
        out_px[v] = nx
        out_py[v] = ny
        if blocked or finished:
            out_acc[v] = 0.0
        else:
            rem = seconds - t
            out_acc[v] = rem if rem > 0.0 else 0.0
        out_state[v] = 2 if finished else 1


def drive_batch(vehicles, mode_tph: np.ndarray, sig: np.ndarray, t_abs: float, seconds: float = 120.0):
    """Drive many vehicles (Person objects) for one sub-step.

    Returns per vehicle, in input order: None when the route is already used up,
    else (pts int16 (k,2), times float32 (k,), new_route_i, x, y, acc, finished)."""
    V = len(vehicles)
    starts = np.empty(V, dtype=np.int64)
    lens = np.empty(V, dtype=np.int64)
    full = np.empty(V, dtype=np.bool_)
    pxs = np.empty(V, dtype=np.int64)
    pys = np.empty(V, dtype=np.int64)
    accs = np.empty(V, dtype=np.float64)
    modes = np.empty(V, dtype=np.int64)
    i0s = [0] * V
    pieces = []
    off = 0
    for v, p in enumerate(vehicles):
        route = p._route
        i0 = p._route_i
        rem = route.shape[0] - i0
        if rem < 0:
            rem = 0
        L = rem if rem <= _SLICE else _SLICE
        piece = route[i0:i0 + L]
        pieces.append(piece)
        starts[v] = off
        lens[v] = L
        full[v] = L == rem
        off += L
        loc = p.location
        pxs[v] = loc[0]
        pys[v] = loc[1]
        accs[v] = p._acc
        modes[v] = p.mode
        i0s[v] = i0
    buf = np.concatenate(pieces).astype(np.int16, copy=False) if off else np.zeros((1, 2), dtype=np.int16)
    out_pts = np.empty((V, MAX_PTS, 2), dtype=np.int16)
    out_t = np.empty((V, MAX_PTS), dtype=np.float32)
    out_k = np.zeros(V, dtype=np.int64)
    out_i = np.zeros(V, dtype=np.int64)
    out_px = np.zeros(V, dtype=np.int64)
    out_py = np.zeros(V, dtype=np.int64)
    out_acc = np.zeros(V, dtype=np.float64)
    out_state = np.zeros(V, dtype=np.int8)
    _drive_batch(buf, starts, lens, full, pxs, pys, accs, modes, mode_tph, seconds, sig, t_abs,
                 SIGNAL_CYCLE, SIGNAL_GREEN, out_pts, out_t, out_k, out_i, out_px, out_py, out_acc, out_state)
    results = []
    ks = out_k.tolist()
    iis = out_i.tolist()
    xs = out_px.tolist()
    ys = out_py.tolist()
    acc_l = out_acc.tolist()
    st = out_state.tolist()
    for v in range(V):
        s = st[v]
        if s == 0:
            results.append(None)
            continue
        k = ks[v]
        results.append((out_pts[v, :k], out_t[v, :k], i0s[v] + iis[v], xs[v], ys[v], acc_l[v], s == 2))
    return results
