"""
cuda_routing.py

CUDA (numba.cuda) batch A* for the route service.

Each GPU thread runs one complete A* search with exactly the same algorithm as
routing._astar_batch: same cost model, same heuristic, same binary heap with the
same (f, g) tie-breaking and sift order.  A path therefore depends only on its
own request, so the GPU returns bit-identical routes to the CPU kernel, and any
split of a batch between GPU and CPU gives identical simulation results.

Scratch memory (g, visit stamp, parent, heap) lives in per-thread rows of a
persistent device pool; the visit-stamp trick avoids clearing it between
searches.  Grids are uploaded once and re-uploaded only after invalidation.

Modes (env RLSS_GPU or RouteService(gpu_mode=...)):
  off     - CPU numba only
  cuda    - every A* batch on the GPU
  hybrid  - each batch split between GPU and CPU threads, run concurrently,
            the split adapted to their measured throughput
  auto    - hybrid when a CUDA device is usable, else off   (default)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on the machine
    from numba import cuda
    _CUDA_IMPORT_OK = True
except Exception:  # noqa: BLE001
    cuda = None
    _CUDA_IMPORT_OK = False

GPU_MODES = ("off", "cuda", "hybrid", "auto")
_THREADS_PER_BLOCK = 32
_STAMP_RESET_AT = 2_000_000_000


def cuda_available() -> bool:
    if not _CUDA_IMPORT_OK:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def resolve_mode(mode: Optional[str]) -> str:
    mode = (mode or os.environ.get("RLSS_GPU") or "auto").strip().lower()
    if mode not in GPU_MODES:
        logger.warning("Unknown RLSS_GPU mode %r; using 'auto'", mode)
        mode = "auto"
    if mode == "auto":
        return "hybrid" if cuda_available() else "off"
    if mode != "off" and not cuda_available():
        logger.warning("RLSS_GPU=%s requested but no usable CUDA device; routing stays on the CPU", mode)
        return "off"
    return mode


if _CUDA_IMPORT_OK:  # pragma: no cover - jit

    @cuda.jit(cache=True)
    def _astar_cuda(sxs, sys_, gxs, gys, n, walkable, cost, h_scale, max_len, out, lens,
                    pool_g, pool_stamp, pool_parent, pool_hf, pool_hg, pool_hi, stamp_ctr):
        r = cuda.grid(1)
        if r >= n:
            return
        H = walkable.shape[0]
        W = walkable.shape[1]
        g = pool_g[r]
        stamp = pool_stamp[r]
        parent = pool_parent[r]
        heap_f = pool_hf[r]
        heap_g = pool_hg[r]
        heap_i = pool_hi[r]
        heap_cap = heap_f.shape[0]
        lens[r] = 0
        sx = sxs[r]
        sy = sys_[r]
        gx = gxs[r]
        gy = gys[r]
        if sx < 0 or sx >= W or sy < 0 or sy >= H or gx < 0 or gx >= W or gy < 0 or gy >= H:
            return
        if not walkable[sy, sx] or not walkable[gy, gx]:
            return
        cur = stamp_ctr[r] + 1
        stamp_ctr[r] = cur
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
            return
        length = 1
        node = gidx
        while node != sidx:
            node = parent[node]
            length += 1
        if length > max_len:
            lens[r] = -1
            return
        node = gidx
        for i in range(length - 1, -1, -1):
            out[r, i, 0] = node % W
            out[r, i, 1] = node // W
            node = parent[node]
        lens[r] = length


class CudaAStar:
    """Device-resident grids + per-thread scratch pool for batched A* on the GPU."""

    def __init__(self, max_rows: int = 128) -> None:
        if not cuda_available():
            raise RuntimeError("no usable CUDA device")
        self.device_name = cuda.get_current_device().name
        if isinstance(self.device_name, bytes):
            self.device_name = self.device_name.decode("ascii", "replace")
        self.max_rows = int(max_rows)
        self._grids = {}          # kind -> (walkable_dev, cost_dev, source ids)
        self._pool = None
        self._pool_size = -1
        self._lock = threading.Lock()

    # -- device memory --------------------------------------------------------
    def upload(self, kind: int, walkable: np.ndarray, cost64: np.ndarray) -> None:
        self._grids[kind] = (
            cuda.to_device(np.ascontiguousarray(walkable, dtype=np.bool_)),
            cuda.to_device(np.ascontiguousarray(cost64, dtype=np.float64)),
            (id(walkable), id(cost64)),
        )

    def _ensure(self, kind: int, walkable: np.ndarray, cost64: np.ndarray):
        entry = self._grids.get(kind)
        if entry is None or entry[2] != (id(walkable), id(cost64)):
            self.upload(kind, walkable, cost64)
            entry = self._grids[kind]
        size = walkable.size
        if self._pool is None or self._pool_size != size:
            rows = self.max_rows
            heap_cap = 4 * size
            self._pool = (
                cuda.device_array((rows, size), dtype=np.float64),
                cuda.to_device(np.zeros((rows, size), dtype=np.int32)),
                cuda.device_array((rows, size), dtype=np.int32),
                cuda.device_array((rows, heap_cap), dtype=np.float64),
                cuda.device_array((rows, heap_cap), dtype=np.float64),
                cuda.device_array((rows, heap_cap), dtype=np.int32),
                cuda.to_device(np.zeros(rows, dtype=np.int32)),
            )
            self._pool_size = size
            self._stamp_used = 0
        return entry

    def invalidate(self) -> None:
        self._grids.clear()

    # -- search -----------------------------------------------------------------
    def launch(self, sx: np.ndarray, sy: np.ndarray, gx: np.ndarray, gy: np.ndarray,
               walkable: np.ndarray, cost64: np.ndarray, h_scale: float, kind: int, max_len: int):
        """Queue A* for m requests on the GPU and return at once (kernels run
        asynchronously); pass the handle to collect() for the results."""
        m = int(sx.shape[0])
        parts = []
        if m == 0:
            return (m, max_len, None, parts)
        self._lock.acquire()
        try:
            walk_d, cost_d, _ = self._ensure(kind, walkable, cost64)
            pool = self._pool
            if self._stamp_used > _STAMP_RESET_AT:
                pool[1].copy_to_device(np.zeros(pool[1].shape, dtype=np.int32))
                pool[6].copy_to_device(np.zeros(pool[6].shape, dtype=np.int32))
                self._stamp_used = 0
            stream = cuda.stream()
            rows = self.max_rows
            for lo in range(0, m, rows):
                hi = min(m, lo + rows)
                n = hi - lo
                req = np.ascontiguousarray(np.stack([sx[lo:hi], sy[lo:hi], gx[lo:hi], gy[lo:hi]]).astype(np.int32))
                d_req = cuda.to_device(req, stream=stream)
                d_out = cuda.device_array((n, max_len, 2), dtype=np.int16, stream=stream)
                d_lens = cuda.device_array(n, dtype=np.int32, stream=stream)
                blocks = (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
                _astar_cuda[blocks, _THREADS_PER_BLOCK, stream](
                    d_req[0], d_req[1], d_req[2], d_req[3], n, walk_d, cost_d, float(h_scale), max_len,
                    d_out, d_lens, *pool)
                parts.append((lo, hi, d_out, d_lens))
                self._stamp_used += 1
        except Exception:
            self._lock.release()
            raise
        return (m, max_len, stream, parts)

    def collect(self, handle) -> Tuple[np.ndarray, np.ndarray]:
        """Wait for a launch() and return (out int16 (m, max_len, 2), lens int32 (m,))."""
        m, max_len, stream, parts = handle
        out = np.zeros((m, max_len, 2), dtype=np.int16)
        lens = np.zeros(m, dtype=np.int32)
        if stream is None:
            return out, lens
        try:
            stream.synchronize()
            for lo, hi, d_out, d_lens in parts:
                lens[lo:hi] = d_lens.copy_to_host()
                out[lo:hi] = d_out.copy_to_host()
        finally:
            self._lock.release()
        return out, lens

    def run(self, sx, sy, gx, gy, walkable, cost64, h_scale, kind, max_len):
        return self.collect(self.launch(sx, sy, gx, gy, walkable, cost64, h_scale, kind, max_len))


class AdaptiveSplit:
    """Tracks GPU and CPU A* throughput and picks the GPU share of a batch.

    The split only decides where each search runs; every search returns the
    same path on either device, so it never affects simulation results."""

    def __init__(self, gpu_share: float = 0.1) -> None:
        self.gpu_share = gpu_share
        self._gpu_rate: Optional[float] = None    # searches per second
        self._cpu_rate: Optional[float] = None

    @staticmethod
    def _ema(old: Optional[float], new: float) -> float:
        return new if old is None else old * 0.8 + new * 0.2

    def record(self, gpu_n: int, gpu_s: float, cpu_n: int, cpu_s: float) -> None:
        if gpu_n >= 8 and gpu_s > 0:
            self._gpu_rate = self._ema(self._gpu_rate, gpu_n / gpu_s)
        if cpu_n >= 8 and cpu_s > 0:
            self._cpu_rate = self._ema(self._cpu_rate, cpu_n / cpu_s)
        if self._gpu_rate and self._cpu_rate:
            share = self._gpu_rate / (self._gpu_rate + self._cpu_rate)
            # Keep a sliver on each side so both stay measured and the split can
            # follow load changes; the EMA lets a fast GPU ramp its share up.
            self.gpu_share = min(0.95, max(0.01, share))

    def summary(self) -> str:
        g = f"{self._gpu_rate:.0f}/s" if self._gpu_rate else "?"
        c = f"{self._cpu_rate:.0f}/s" if self._cpu_rate else "?"
        return f"gpu {g} cpu {c} share {self.gpu_share:.2f}"


