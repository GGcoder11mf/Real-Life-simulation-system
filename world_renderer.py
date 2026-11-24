"""
world_renderer.py

Procedural city renderer (no image textures).

* The ground, roads, buildings, trees and water are painted into cached
  *chunks* (512-px surfaces) at the current zoom, so a frame is a handful of
  blits.  Missing chunks are first shown as a cheap coarse render and refined
  a few per frame, so panning/zooming never stalls.
* Citizens, cars, taxis, buses and metro trains are drawn per frame from the
  simulation's movement waypoints, interpolated along the real route, with
  lane/sidewalk offsets so people walk on pavements and cars keep right.
* Night is rendered with a multiply tint plus additive street-lamp, window and
  headlight glows; traffic lights cycle at intersections.

Zoom range is 2..96 px per tile; a citizen is ~0.3 tile tall, so at maximum
zoom individual walkers are clearly visible on the sidewalks.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

from sim_protocol import BUILDING_CODES, TERRAIN_KEYS
from world import signal_state

ZOOM_STEPS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96]
CHUNK_PIXELS = 512
MAX_CHUNKS = 150

BUILDING_NAMES = {code: name for name, code in BUILDING_CODES.items()}
FACADE = {
    "house": (214, 178, 134), "apartment": (196, 150, 128), "shop": (236, 190, 120),
    "office": (120, 160, 200), "factory": (150, 120, 112), "warehouse": (134, 128, 120),
    "school": (232, 204, 110), "college": (206, 140, 96), "police": (110, 140, 190),
    "hospital": (236, 240, 244), "station": (156, 150, 200), "park_facility": (112, 168, 100),
    "court": (214, 206, 180), "prison": (118, 120, 128),
}
GROUND = {
    "water": (36, 98, 160), "park": (70, 140, 82), "rural": (150, 138, 92), "suburban": (112, 150, 100),
    "urban": (118, 122, 130), "industrial": (112, 104, 118), "road": (54, 56, 62), "rail": (54, 56, 62),
}
HOUSE_WALLS = [(232, 218, 192), (242, 240, 232), (204, 172, 140), (176, 198, 216), (216, 152, 122), (192, 204, 172), (226, 196, 150)]
HOUSE_ROOFS = [(150, 70, 58), (96, 98, 112), (122, 92, 66), (70, 102, 86), (170, 112, 62), (84, 84, 92)]
APT_WALLS = [(178, 122, 100), (204, 200, 190), (162, 170, 180), (216, 192, 152), (150, 150, 162), (190, 150, 130)]
GLASS = [(98, 152, 198), (92, 172, 172), (132, 142, 162), (178, 162, 102), (72, 112, 162), (120, 130, 150)]
SHOP_COLORS = [(236, 190, 120), (218, 120, 110), (150, 190, 210), (200, 210, 150), (230, 160, 190), (240, 220, 150)]
AWNINGS = [(214, 70, 70), (60, 130, 190), (60, 160, 110), (230, 170, 50), (150, 80, 170)]
ASPHALT = (54, 56, 62)
SIDEWALK = (150, 150, 152)
SKIN = [(244, 214, 186), (226, 178, 140), (178, 126, 92), (120, 82, 58)]
HAIR = [(40, 30, 26), (92, 62, 38), (150, 110, 60), (30, 30, 34), (170, 170, 172)]
CAR_COLORS = [(188, 52, 48), (52, 92, 170), (230, 230, 232), (36, 38, 42), (96, 100, 108), (60, 130, 90), (220, 170, 50), (140, 60, 150)]
TAXI_COLOR = (250, 200, 36)
BUS_COLOR = (44, 150, 196)


def shade(color: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
    return (max(0, min(255, int(color[0] * f))), max(0, min(255, int(color[1] * f))), max(0, min(255, int(color[2] * f))))


def mix(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


class WorldRenderer:
    def __init__(self) -> None:
        self.version = -1
        self.chunks: "OrderedDict[Tuple[int, int, int, int], pygame.Surface]" = OrderedDict()
        self.coarse: Dict[Tuple[int, int, int, int], pygame.Surface] = {}
        self.sprite_cache: Dict[Tuple[Any, ...], pygame.Surface] = {}
        self.glow_cache: Dict[int, pygame.Surface] = {}
        self.static: Dict[str, Any] = {}
        self.W = self.H = 0

    # ------------------------------------------------------------------
    # Static data
    # ------------------------------------------------------------------
    def set_static(self, static: Dict[str, Any]) -> None:
        self.static = static
        self.W, self.H = static["width"], static["height"]
        self.version = static["version"]
        self.chunks.clear()
        self.coarse.clear()
        H, W = self.H, self.W
        terr = static["terrain"]
        self.terr = terr
        road_code = TERRAIN_KEYS.index("road")
        rail_code = TERRAIN_KEYS.index("rail")
        self.water_code = TERRAIN_KEYS.index("water")
        self.road = (terr == road_code) | (terr == rail_code)
        self.rail = terr == rail_code
        pad = np.pad(self.road, 1, constant_values=False)
        self.nN = pad[:-2, 1:-1]
        self.nS = pad[2:, 1:-1]
        self.nW = pad[1:-1, :-2]
        self.nE = pad[1:-1, 2:]
        arms = self.nN.astype(np.int8) + self.nS + self.nW + self.nE
        self.arms = arms
        self.inter = self.road & (arms >= 3)
        yy, xx = np.mgrid[0:H, 0:W]
        self.hash = (((xx * 73856093) ^ (yy * 19349663) ^ 0x9E3779B9) & 0xFFFFFFFF).astype(np.uint32)
        self.bld_type = static["building_type"]
        self.floors = static.get("floors", np.ones((H, W), dtype=np.uint8))
        self.tier = static.get("tier", np.zeros((H, W), dtype=np.uint8))
        self.bid = static["building_id"]
        # Python lists are much faster than numpy for scalar access in painters.
        self.l_terr = terr.tolist()
        self.l_road = self.road.tolist()
        self.l_rail = self.rail.tolist()
        self.l_n = self.nN.tolist()
        self.l_s = self.nS.tolist()
        self.l_w = self.nW.tolist()
        self.l_e = self.nE.tolist()
        self.l_arms = arms.tolist()
        self.l_hash = self.hash.tolist()
        self.l_bt = self.bld_type.tolist()
        self.l_floors = self.floors.tolist()
        self.l_tier = self.tier.tolist()
        self.l_bid = self.bid.tolist()
        self.l_elev = static["elevation"].tolist()
        self.ground_lut = np.array([GROUND.get(k, (90, 90, 90)) for k in TERRAIN_KEYS], dtype=np.int16)
        self.inter_tiles = list(zip(*np.nonzero(self.inter)))
        sig = static.get("signals")
        if sig is not None and len(sig):
            self.signal_boxes = sig[:, :4].astype(np.int32)
            self.signal_ids = sig[:, 4].astype(np.int32)
        else:
            self.signal_boxes = np.zeros((0, 4), dtype=np.int32)
            self.signal_ids = np.zeros(0, dtype=np.int32)
        self.lamp_tiles = [
            (int(x), int(y)) for y, x in zip(*np.nonzero(self.road & ~self.inter & ((xx + yy) % 3 == 0) & (arms <= 2)))
        ]
        self.lamp_arr = np.array(self.lamp_tiles, dtype=np.int32).reshape(-1, 2) if self.lamp_tiles else np.zeros((0, 2), dtype=np.int32)
        self.inter_arr = np.array([(int(x), int(y)) for y, x in self.inter_tiles], dtype=np.int32).reshape(-1, 2) if self.inter_tiles else np.zeros((0, 2), dtype=np.int32)
        bld_ys, bld_xs = np.nonzero(self.bld_type)
        self.bld_arr = np.stack([bld_xs, bld_ys], axis=1).astype(np.int32) if bld_xs.size else np.zeros((0, 2), dtype=np.int32)
        self.transit = static.get("transit_lines", [])
        bs = static.get("buildings")
        self.blds: List[Tuple[Any, ...]] = []
        self.bld_by_row: Dict[int, List[int]] = {}
        if bs is not None:
            for i in range(len(bs["id"])):
                x, y = int(bs["x"][i]), int(bs["y"][i])
                hsh = int(self.hash[y, x])
                self.blds.append((x, y, int(bs["w"][i]), int(bs["h"][i]), int(bs["facing"][i]), int(bs["code"][i]),
                                  int(bs["floors"][i]), int(bs["tier"][i]), int(bs["id"][i]), hsh))
                self.bld_by_row.setdefault(y, []).append(i)
        self.bld_arr = np.array([(b[0], b[1]) for b in self.blds], dtype=np.int32).reshape(-1, 2) if self.blds else np.zeros((0, 2), dtype=np.int32)

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------
    @staticmethod
    def chunk_tiles(ts: int) -> int:
        return max(8, CHUNK_PIXELS // ts)

    def visible_chunks(self, x0: float, y0: float, tw: float, th: float, ts: int) -> List[Tuple[int, int]]:
        ct = self.chunk_tiles(ts)
        cx0 = max(0, int(math.floor(x0 / ct)))
        cy0 = max(0, int(math.floor(y0 / ct)))
        cx1 = min((self.W - 1) // ct, int(math.floor((x0 + tw) / ct)))
        cy1 = min((self.H - 1) // ct, int(math.floor((y0 + th) / ct)))
        return [(cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)]

    def draw_ground(self, dest: pygame.Surface, rect: pygame.Rect, x0: float, y0: float, ts: int, tw: float, th: float,
                    time_budget_ms: float = 14.0) -> None:
        ct = self.chunk_tiles(ts)
        started = time.perf_counter()
        order = self.visible_chunks(x0, y0, tw, th, ts)
        # centre-out build order
        ccx, ccy = (x0 + tw / 2) / ct, (y0 + th / 2) / ct
        order.sort(key=lambda c: (c[0] + 0.5 - ccx) ** 2 + (c[1] + 0.5 - ccy) ** 2)
        for cx, cy in order:
            key = (cx, cy, ts, self.version)
            surf = self.chunks.get(key)
            if surf is None:
                if (time.perf_counter() - started) * 1000.0 < time_budget_ms:
                    surf = self.paint_chunk(cx, cy, ts)
                    self.chunks[key] = surf
                    self.coarse.pop(key, None)
                    while len(self.chunks) > MAX_CHUNKS:
                        self.chunks.popitem(last=False)
                else:
                    surf = self.coarse.get(key)
                    if surf is None:
                        surf = self.paint_coarse(cx, cy, ts)
                        self.coarse[key] = surf
                        if len(self.coarse) > 400:
                            self.coarse.clear()
            else:
                self.chunks.move_to_end(key)
            px = rect.x + int(round((cx * ct - x0) * ts))
            py = rect.y + int(round((cy * ct - y0) * ts))
            dest.blit(surf, (px, py))

    def paint_roads_simple(self, surf: pygame.Surface, tx0: int, ty0: int, tx1: int, ty1: int, ts: int) -> None:
        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                if self.l_road[ty][tx]:
                    px, py = (tx - tx0) * ts, (ty - ty0) * ts
                    surf.fill((70, 72, 80), (px, py, ts, ts))

    def paint_ground(self, surf: pygame.Surface, tx: int, ty: int, px: int, py: int, ts: int, key: str) -> None:
        hsh = self.l_hash[ty][tx]
        if key == "water":
            wave = (hsh >> 3) % 3
            col = (70, 130, 190)
            y1 = py + int(ts * (0.25 + 0.25 * wave))
            pygame.draw.line(surf, col, (px + int(ts * 0.15), y1), (px + int(ts * 0.45), y1), 1)
            pygame.draw.line(surf, col, (px + int(ts * 0.55), y1 + ts // 4), (px + int(ts * 0.85), y1 + ts // 4), 1)
            edge = max(2, ts // 8)
            sand = (214, 200, 150)
            if ty > 0 and self.l_terr[ty - 1][tx] != self.water_code:
                surf.fill(sand, (px, py, ts, edge))
            if ty + 1 < self.H and self.l_terr[ty + 1][tx] != self.water_code:
                surf.fill(sand, (px, py + ts - edge, ts, edge))
            if tx > 0 and self.l_terr[ty][tx - 1] != self.water_code:
                surf.fill(sand, (px, py, edge, ts))
            if tx + 1 < self.W and self.l_terr[ty][tx + 1] != self.water_code:
                surf.fill(sand, (px + ts - edge, py, edge, ts))
        elif key == "park":
            if ts >= 16 and hsh % 9 == 0:
                pygame.draw.circle(surf, (236, 210, 90), (px + ts // 2, py + ts // 2), max(1, ts // 16))
        elif key == "rural":
            if ts >= 16:
                orient = ((tx // 3) + (ty // 3)) % 2
                col = (136, 124, 82)
                step = max(4, ts // 4)
                for k in range(0, ts, step * 2):
                    if orient:
                        surf.fill(col, (px, py + k, ts, max(1, step // 3)))
                    else:
                        surf.fill(col, (px + k, py, max(1, step // 3), ts))
        elif key == "suburban":
            if ts >= 16 and hsh % 7 == 0:
                pygame.draw.circle(surf, (210, 120, 150), (px + ts // 3, py + ts // 3), max(1, ts // 14))
        elif key == "urban":
            if ts >= 12:
                pygame.draw.rect(surf, (104, 108, 116), (px, py, ts, ts), 1)
        elif key == "industrial":
            if ts >= 12 and hsh % 3 == 0:
                surf.fill((250, 214, 70), (px + ts // 6, py + ts // 2, ts * 2 // 3, max(1, ts // 24)))

    def paint_trees(self, surf: pygame.Surface, tx: int, ty: int, px: int, py: int, ts: int) -> None:
        hsh = self.l_hash[ty][tx]
        count = 1 + hsh % 3
        for i in range(count):
            hx = ((hsh >> (i * 5)) & 31) / 31.0
            hy = ((hsh >> (i * 5 + 2)) & 31) / 31.0
            x = px + int(ts * (0.2 + 0.6 * hx))
            y = py + int(ts * (0.25 + 0.55 * hy))
            r = max(2, int(ts * (0.13 + 0.05 * ((hsh >> (i + 9)) & 3) / 3)))
            pygame.draw.ellipse(surf, (40, 80, 50), (x - r, y + r // 2, r * 2, max(2, r)))
            pygame.draw.circle(surf, (34, 104, 56), (x, y), r)
            pygame.draw.circle(surf, (58, 140, 76), (x - r // 3, y - r // 3), max(1, r * 2 // 3))

    def paint_road(self, surf: pygame.Surface, tx: int, ty: int, px: int, py: int, ts: int) -> None:
        n, s, w, e = self.l_n[ty][tx], self.l_s[ty][tx], self.l_w[ty][tx], self.l_e[ty][tx]
        arms = self.l_arms[ty][tx]
        surf.fill(ASPHALT, (px, py, ts, ts))
        if ts < 8:
            return
        side = max(2, int(ts * 0.14))
        if not n:
            surf.fill(SIDEWALK, (px, py, ts, side))
            surf.fill((110, 110, 112), (px, py + side - 1, ts, 1))
        if not s:
            surf.fill(SIDEWALK, (px, py + ts - side, ts, side))
            surf.fill((110, 110, 112), (px, py + ts - side, ts, 1))
        if not w:
            surf.fill(SIDEWALK, (px, py, side, ts))
        if not e:
            surf.fill(SIDEWALK, (px + ts - side, py, side, ts))
        if ts >= 24 and arms <= 2 and (tx + ty) % 4 == 1 and not self.l_bt[ty][tx]:
            hsh = self.l_hash[ty][tx]
            r = max(3, int(ts * 0.11))
            if not n and (hsh & 1):
                cy_ = py + side // 2 + 1
            elif not s:
                cy_ = py + ts - side // 2 - 1
            else:
                cy_ = None
            if cy_ is not None and (n or s):
                cx_ = px + ts // 2
                pygame.draw.circle(surf, (34, 100, 54), (cx_, cy_), r)
                pygame.draw.circle(surf, (58, 138, 74), (cx_ - r // 3, cy_ - r // 3), max(1, r * 2 // 3))
        if self.l_rail[ty][tx] and ts >= 10:
            # station platform
            surf.fill((70, 110, 170), (px + ts // 4, py + ts // 4, ts // 2, ts // 2))
            pygame.draw.rect(surf, (230, 240, 255), (px + ts // 4, py + ts // 4, ts // 2, ts // 2), 1)
            return
        if ts >= 12:
            horiz = (w or e) and not (n or s)
            vert = (n or s) and not (w or e)
            dash = max(2, ts // 5)
            line = (232, 214, 96)
            if horiz:
                for k in range(0, ts, dash * 2):
                    surf.fill(line, (px + k, py + ts // 2 - 1, dash, max(1, ts // 32 + 1)))
            elif vert:
                for k in range(0, ts, dash * 2):
                    surf.fill(line, (px + ts // 2 - 1, py + k, max(1, ts // 32 + 1), dash))
        if ts >= 20 and arms == 2 and ((n and s) or (w and e)):
            # zebra crossing next to intersections
            ni = self.neighbour_is_intersection(tx, ty)
            if ni:
                zebra = (226, 226, 226)
                step = max(2, ts // 8)
                if n and s:
                    for k in range(side + 1, ts - side - 1, step * 2):
                        if ni == "n":
                            surf.fill(zebra, (px + k, py + 1, step, ts // 5))
                        elif ni == "s":
                            surf.fill(zebra, (px + k, py + ts - ts // 5 - 1, step, ts // 5))
                else:
                    for k in range(side + 1, ts - side - 1, step * 2):
                        if ni == "w":
                            surf.fill(zebra, (px + 1, py + k, ts // 5, step))
                        elif ni == "e":
                            surf.fill(zebra, (px + ts - ts // 5 - 1, py + k, ts // 5, step))

    def neighbour_is_intersection(self, tx: int, ty: int) -> str:
        if ty > 0 and self.inter[ty - 1, tx]:
            return "n"
        if ty + 1 < self.H and self.inter[ty + 1, tx]:
            return "s"
        if tx > 0 and self.inter[ty, tx - 1]:
            return "w"
        if tx + 1 < self.W and self.inter[ty, tx + 1]:
            return "e"
        return ""

    # ------------------------------------------------------------------
    # Buildings
    # ------------------------------------------------------------------
    def person_sprite(self, size: int, shirt: Tuple[int, int, int], skin: int, hair: int, phase: int, flip: bool, idle: bool) -> pygame.Surface:
        key = ("p", size, shirt, skin, hair, phase, flip, idle)
        spr = self.sprite_cache.get(key)
        if spr is not None:
            return spr
        w, h = max(5, int(size * 0.55)), size
        spr = pygame.Surface((w, h), pygame.SRCALPHA)
        cx = w // 2
        swing = 0 if idle else int(math.sin(phase / 8.0 * math.tau) * max(1, size * 0.16))
        head_r = max(1, int(size * 0.17))
        body_top = int(size * 0.30)
        body_h = int(size * 0.36)
        leg_len = h - (body_top + body_h)
        pants = shade(shirt, 0.45)
        # legs
        pygame.draw.line(spr, pants, (cx, body_top + body_h), (cx - swing, h - 1), max(1, size // 10))
        pygame.draw.line(spr, shade(pants, 0.8), (cx, body_top + body_h), (cx + swing, h - 1), max(1, size // 10))
        # arms
        pygame.draw.line(spr, shade(shirt, 0.85), (cx, body_top + 1), (cx + swing, body_top + body_h - 1), max(1, size // 11))
        # torso
        pygame.draw.rect(spr, shirt, (cx - max(1, size // 8), body_top, max(2, size // 4), body_h), border_radius=max(1, size // 14))
        # head
        pygame.draw.circle(spr, SKIN[skin % len(SKIN)], (cx, head_r + 1), head_r)
        pygame.draw.circle(spr, HAIR[hair % len(HAIR)], (cx, head_r), head_r, max(1, head_r // 2))
        if flip:
            spr = pygame.transform.flip(spr, True, False)
        self.sprite_cache[key] = spr
        if len(self.sprite_cache) > 6000:
            self.sprite_cache.clear()
        return spr

    def vehicle_sprite(self, kind: str, length: int, color: Tuple[int, int, int], heading: int) -> pygame.Surface:
        """heading: 0 east, 1 south, 2 west, 3 north."""
        key = ("v", kind, length, color, heading)
        spr = self.sprite_cache.get(key)
        if spr is not None:
            return spr
        L = max(6, length)
        Wd = max(4, int(L * (0.42 if kind == "bus" else 0.5)))
        base = pygame.Surface((L, Wd), pygame.SRCALPHA)
        body = pygame.Rect(0, 0, L, Wd)
        pygame.draw.rect(base, shade(color, 0.55), body.inflate(0, 0), border_radius=max(1, Wd // 3))
        pygame.draw.rect(base, color, body.inflate(-2, -2), border_radius=max(1, Wd // 3))
        glass = (60, 86, 112)
        if kind == "bus":
            for i in range(5):
                gx = int(L * (0.12 + i * 0.16))
                base.fill(glass, (gx, 1, max(1, int(L * 0.11)), max(1, Wd - 3)))
            base.fill((230, 230, 210), (L - 3, 1, 2, Wd - 2))
        else:
            base.fill(glass, (int(L * 0.58), 2, int(L * 0.18), Wd - 4))
            base.fill(glass, (int(L * 0.18), 2, int(L * 0.12), Wd - 4))
            base.fill(shade(color, 1.15 if kind != "taxi" else 1.0), (int(L * 0.32), 2, int(L * 0.26), Wd - 4))
            base.fill((255, 244, 200), (L - 2, 1, 2, 2))
            base.fill((255, 244, 200), (L - 2, Wd - 3, 2, 2))
            base.fill((200, 40, 40), (0, 1, 1, 2))
            base.fill((200, 40, 40), (0, Wd - 3, 1, 2))
            if kind == "taxi":
                base.fill((255, 255, 255), (int(L * 0.42), Wd // 2 - 1, max(2, int(L * 0.12)), 2))
        spr = pygame.transform.rotate(base, -90 * heading) if heading else base
        self.sprite_cache[key] = spr
        return spr

    def glow(self, radius: int, level: float = 1.0) -> pygame.Surface:
        radius = max(2, radius)
        q = max(1, min(10, int(round(level * 10))))
        key = (radius, q)
        g = self.glow_cache.get(key)
        if g is None:
            g = pygame.Surface((radius * 2, radius * 2))
            lv = q / 10.0
            for r in range(radius, 0, -1):
                v = int(255 * lv * (1 - r / radius) ** 2)
                pygame.draw.circle(g, (v, v, int(v * 0.8)), (radius, radius), r)
            self.glow_cache[key] = g
        return g

    # ------------------------------------------------------------------
    # Moving agents
    # ------------------------------------------------------------------
    def agent_positions(self, frame: Dict[str, Any], u: float) -> Dict[str, Any]:
        """Positions (tile units) of all citizens at fraction `u` through the frame's motion.

        Travellers are placed by timestamp along their real waypoints, so vehicles
        that waited at a red light hold at the stop line and move off on green.
        Returns arrays aligned with frame['ids']: x, y, heading (0 E, 1 S, 2 W,
        3 N), moving, still (travelling but momentarily stopped) and a unit lateral
        vector used to place walkers on pavements and cars in their lane."""
        n = frame["ids"].shape[0]
        x = frame["x"].astype(np.float32) + 0.5
        y = frame["y"].astype(np.float32) + 0.5
        heading = np.zeros(n, dtype=np.int8)
        moving = np.zeros(n, dtype=bool)
        still = np.zeros(n, dtype=bool)
        lat_full = np.zeros((n, 2), dtype=np.float32)
        mv_ids = frame["mv_ids"]
        if mv_ids.shape[0]:
            ids = frame["ids"]
            idx = np.clip(np.searchsorted(ids, mv_ids), 0, n - 1)
            ok = ids[idx] == mv_ids
            pts = frame["mv_pts"].astype(np.float32)
            times = frame["mv_t"].astype(np.float64)
            off = frame["mv_off"]
            M = mv_ids.shape[0]
            N = pts.shape[0]
            starts = off[:-1].astype(np.int64)
            lens = (off[1:] - off[:-1]).astype(np.int64)
            ends = starts + lens - 1
            tq = float(min(1.0, max(0.0, u))) * 120.0
            mid = np.repeat(np.arange(M), lens)
            key = mid * 1000.0 + times
            q = np.arange(M) * 1000.0 + tq
            a = np.clip(np.searchsorted(key, q, side="right") - 1, starts, ends)
            b = np.minimum(a + 1, ends)
            ta, tb = times[a], times[b]
            frac = np.where(tb > ta, (tq - ta) / np.maximum(tb - ta, 1e-6), 0.0).clip(0.0, 1.0).astype(np.float32)
            # last non-zero movement direction at/before (or after) each point
            steps = np.zeros_like(pts)
            if N > 1:
                steps[:-1] = pts[1:] - pts[:-1]
            steps[ends] = 0.0
            nz = (steps[:, 0] != 0) | (steps[:, 1] != 0)
            idxs = np.arange(N)
            fwd = np.maximum.accumulate(np.where(nz, idxs, -1))
            bwd = np.minimum.accumulate(np.where(nz, idxs, N)[::-1])[::-1]
            pt_start = np.repeat(starts, lens)
            pt_end = np.repeat(ends, lens)
            use_f = fwd >= pt_start
            use_b = (~use_f) & (bwd <= pt_end)
            head = np.zeros((N, 2), dtype=np.float32)
            head[:, 0] = 1.0
            head[use_f] = steps[fwd[use_f]]
            head[use_b] = steps[bwd[use_b]]
            px = pts[a, 0] + (pts[b, 0] - pts[a, 0]) * frac
            py = pts[a, 1] + (pts[b, 1] - pts[a, 1]) * frac
            ha, hb = head[a], head[b]
            hv = ha * (1 - frac)[:, None] + hb * frac[:, None]

            def perp(v: np.ndarray) -> np.ndarray:
                return np.stack([-v[:, 1], v[:, 0]], axis=1)

            lat = perp(ha) * (1 - frac)[:, None] + perp(hb) * frac[:, None]
            lat = lat / np.maximum(np.linalg.norm(lat, axis=1, keepdims=True), 1e-4)
            sel = idx[ok]
            x[sel] = px[ok] + 0.5
            y[sel] = py[ok] + 0.5
            moving[sel] = True
            same = (pts[a, 0] == pts[b, 0]) & (pts[a, 1] == pts[b, 1])
            still[sel] = same[ok]
            h = hv[ok]
            hd = np.where(np.abs(h[:, 0]) >= np.abs(h[:, 1]), np.where(h[:, 0] >= 0, 0, 2), np.where(h[:, 1] >= 0, 1, 3))
            heading[sel] = hd.astype(np.int8)
            lat_full[sel] = lat[ok]
        return {"x": x, "y": y, "heading": heading, "moving": moving, "still": still, "lat": lat_full}

    def draw_agents(self, dest: pygame.Surface, frame: Dict[str, Any], agents: Dict[str, Any], x0: float, y0: float,
                    tw: float, th: float, ts: int, colors: np.ndarray, sim_seconds: float, selected_id: Optional[int]) -> int:
        n = frame["ids"].shape[0]
        outdoors = frame["indoors"] == 0
        ax, ay = agents["x"], agents["y"]
        vis = outdoors & (ax >= x0 - 2) & (ax < x0 + tw + 2) & (ay >= y0 - 2) & (ay < y0 + th + 2)
        sel = np.nonzero(vis)[0]
        if sel.size == 0:
            return 0
        sx = (ax[sel] - x0) * ts
        sy = (ay[sel] - y0) * ts
        mode = frame["mode"][sel]
        moving = agents["moving"][sel]
        vehicle = moving & ((mode == 2) | (mode == 3) | (mode == 4))
        w, h = dest.get_size()
        if ts < 14:
            # Dots: vehicles are bigger and brighter than pedestrians.
            arr = pygame.surfarray.pixels3d(dest)
            try:
                cols = colors[sel]
                size = max(1, min(3, ts // 3))
                for dx in range(size):
                    for dy in range(size):
                        xx = sx.astype(np.int32) + dx
                        yy = sy.astype(np.int32) + dy
                        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                        arr[xx[ok], yy[ok]] = cols[ok]
                if ts >= 5:
                    v = np.nonzero(vehicle)[0]
                    if v.size:
                        vx = sx[v].astype(np.int32)
                        vy = sy[v].astype(np.int32)
                        taxi = mode[v] == 4
                        vc = np.where(taxi[:, None], np.array(TAXI_COLOR, dtype=np.uint8)[None, :], np.array((255, 255, 255), dtype=np.uint8)[None, :])
                        for dx in range(2):
                            for dy in range(2):
                                xx = vx + dx
                                yy = vy + dy
                                ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                                arr[xx[ok], yy[ok]] = vc[ok]
            finally:
                del arr
            return int(sel.size)
        # Close-up: individual figures and vehicles.
        ids = frame["ids"][sel].astype(np.int64)
        lat = agents["lat"][sel]
        heading = agents["heading"][sel]
        order = np.argsort(sy)
        shirts = colors[sel]
        person_h = max(7, int(ts * 0.38))
        car_len = max(8, int(ts * 0.60))
        bus_len = max(12, int(ts * 1.1))
        bus_cells: set = set()
        # Vehicles waiting at a red light queue up behind each other.
        still = agents["still"][sel]
        queue_rank = np.zeros(sel.size, dtype=np.int32)
        waiting = np.nonzero(vehicle & still)[0]
        if waiting.size > 1:
            keys = (np.floor(ay[sel][waiting]).astype(np.int64) * self.W + np.floor(ax[sel][waiting]).astype(np.int64)) * 4 + heading[waiting]
            order = np.argsort(keys, kind="stable")
            sk = keys[order]
            first = np.r_[True, sk[1:] != sk[:-1]]
            gstart = np.maximum.accumulate(np.where(first, np.arange(sk.size), 0))
            queue_rank[waiting[order]] = np.arange(sk.size) - gstart
        drawn = 0
        limit = 6000
        selected_pos = None
        for i in order.tolist():
            if drawn >= limit:
                break
            pid = int(ids[i])
            px_ = float(sx[i])
            py_ = float(sy[i])
            if px_ < -40 or py_ < -40 or px_ > w + 40 or py_ > h + 40:
                continue
            m = int(mode[i])
            if vehicle[i]:
                hd = int(heading[i])
                # keep to the right-hand lane
                off = 0.17 * ts
                px_ += float(lat[i, 0]) * off
                py_ += float(lat[i, 1]) * off
                qr = int(queue_rank[i])
                if qr:
                    px_ -= (1, 0, -1, 0)[hd] * qr * 0.62 * ts
                    py_ -= (0, 1, 0, -1)[hd] * qr * 0.62 * ts
                if m == 3:
                    cell = (int(px_) // max(8, ts // 2), int(py_) // max(8, ts // 2), hd)
                    if cell in bus_cells:
                        continue
                    bus_cells.add(cell)
                    spr = self.vehicle_sprite("bus", bus_len, BUS_COLOR, hd)
                elif m == 4:
                    spr = self.vehicle_sprite("taxi", car_len, TAXI_COLOR, hd)
                else:
                    spr = self.vehicle_sprite("car", car_len, CAR_COLORS[pid % len(CAR_COLORS)], hd)
                dest.blit(spr, (int(px_ - spr.get_width() / 2), int(py_ - spr.get_height() / 2)))
                if pid == selected_id:
                    selected_pos = (int(px_), int(py_))
                drawn += 1
                continue
            idle = (not bool(moving[i])) or bool(still[i])
            if moving[i]:
                side = 1 if (pid & 1) else -1
                px_ += float(lat[i, 0]) * 0.36 * ts * side
                py_ += float(lat[i, 1]) * 0.36 * ts * side
            else:
                px_ += (((pid * 2654435761) >> 7) % 100 / 100.0 - 0.5) * ts * 0.6
                py_ += (((pid * 40503) >> 5) % 100 / 100.0 - 0.5) * ts * 0.6
            hd = int(heading[i])
            flip = hd == 2 or hd == 3
            phase = int((sim_seconds * 1.6 + pid * 0.37) * 8) % 8
            shirt = (int(shirts[i][0]), int(shirts[i][1]), int(shirts[i][2]))
            shirt = (shirt[0] // 24 * 24, shirt[1] // 24 * 24, shirt[2] // 24 * 24)
            spr = self.person_sprite(person_h, shirt, pid % 4, (pid >> 2) % 5, phase, flip, idle)
            dest.blit(spr, (int(px_ - spr.get_width() / 2), int(py_ - spr.get_height() + person_h * 0.12)))
            if pid == selected_id:
                selected_pos = (int(px_), int(py_ - person_h / 2))
            drawn += 1
        if selected_pos is not None:
            pygame.draw.circle(dest, (255, 234, 122), selected_pos, max(8, int(ts * 0.3)), 2)
        return drawn

    # ------------------------------------------------------------------
    # Dynamic scenery: metro, traffic lights, night lights
    # ------------------------------------------------------------------
    def draw_metro(self, dest: pygame.Surface, x0: float, y0: float, ts: int, sim_minutes: float, w: int, h: int) -> None:
        if ts < 6:
            return
        for li, line in enumerate(self.transit):
            stops = line["stops"]
            if len(stops) < 2:
                continue
            ax, ay = stops[0]
            bx, by = stops[-1]
            p0 = ((ax + 0.5 - x0) * ts, (ay + 0.5 - y0) * ts)
            p1 = ((bx + 0.5 - x0) * ts, (by + 0.5 - y0) * ts)
            colr = (214, 70, 70) if li % 2 == 0 else (70, 120, 214)
            pygame.draw.line(dest, shade(colr, 0.5), p0, p1, max(2, ts // 5))
            pygame.draw.line(dest, colr, p0, p1, max(1, ts // 10))
            length = math.hypot(bx - ax, by - ay)
            if length <= 0:
                continue
            period = 70.0
            for trip in range(3):
                phase = ((sim_minutes + trip * period / 3.0 + li * 11) % period) / period
                s = phase * 2 if phase < 0.5 else (1 - phase) * 2
                cars = 3
                for c in range(cars):
                    t = s - c * (0.9 / max(1.0, length))
                    if t < 0 or t > 1:
                        continue
                    cxp = ax + (bx - ax) * t
                    cyp = ay + (by - ay) * t
                    sx = (cxp + 0.5 - x0) * ts
                    sy = (cyp + 0.5 - y0) * ts
                    if -40 < sx < w + 40 and -40 < sy < h + 40:
                        horizontal = abs(bx - ax) >= abs(by - ay)
                        rw = max(4, int(ts * 0.82)) if horizontal else max(3, int(ts * 0.4))
                        rh = max(3, int(ts * 0.4)) if horizontal else max(4, int(ts * 0.82))
                        rect = pygame.Rect(int(sx - rw / 2), int(sy - rh / 2), rw, rh)
                        pygame.draw.rect(dest, (226, 230, 238), rect, border_radius=max(1, ts // 10))
                        pygame.draw.rect(dest, colr, rect, 1, border_radius=max(1, ts // 10))
                        dest.fill((70, 96, 130), rect.inflate(-rw // 2 if horizontal else 0, -rh // 2 if not horizontal else 0))

    def draw_signals(self, dest: pygame.Surface, x0: float, y0: float, tw: float, th: float, ts: int, sim_seconds: float) -> None:
        """Traffic lights, driven by the same timing function the simulation uses."""
        if ts < 14 or self.signal_boxes.shape[0] == 0:
            return
        boxes = self.signal_boxes
        vis = (boxes[:, 2] >= x0 - 1) & (boxes[:, 0] < x0 + tw + 1) & (boxes[:, 3] >= y0 - 1) & (boxes[:, 1] < y0 + th + 1)
        colors = [(60, 220, 90), (250, 200, 50), (230, 60, 60)]
        r = max(2, ts // 12)
        for idx in np.nonzero(vis)[0].tolist():
            x_lo, y_lo, x_hi, y_hi = boxes[idx].tolist()
            jid = int(self.signal_ids[idx])
            ns = colors[signal_state(jid, 0, sim_seconds)]
            ew = colors[signal_state(jid, 1, sim_seconds)]
            left = (x_lo - x0) * ts + ts * 0.08
            right = (x_hi + 1 - x0) * ts - ts * 0.08
            top = (y_lo - y0) * ts + ts * 0.08
            bottom = (y_hi + 1 - y0) * ts - ts * 0.08
            for cx, cy, col in ((left, top, ns), (right, bottom, ns), (right, top, ew), (left, bottom, ew)):
                pygame.draw.circle(dest, (24, 24, 28), (int(cx), int(cy)), r + 2)
                pygame.draw.circle(dest, col, (int(cx), int(cy)), r)

    def paint_coarse(self, cx: int, cy: int, ts: int) -> pygame.Surface:
        ct = self.chunk_tiles(ts)
        tx0, ty0 = cx * ct, cy * ct
        tx1, ty1 = min(self.W, tx0 + ct), min(self.H, ty0 + ct)
        terr = self.terr[ty0:ty1, tx0:tx1]
        noise = ((self.hash[ty0:ty1, tx0:tx1] >> 3) & 31).astype(np.float32) / 31.0
        factor = np.where((terr == self.water_code) | self.road[ty0:ty1, tx0:tx1], 1.0, 0.95 + 0.10 * noise)
        rgb = np.clip(self.ground_lut[terr].astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)
        codes = self.bld_type[ty0:ty1, tx0:tx1]
        if codes.any():
            lut = np.array([(150, 150, 150)] + [FACADE.get(BUILDING_NAMES.get(c, ""), (180, 180, 180)) for c in range(1, 16)], dtype=np.uint8)
            mask = codes > 0
            rgb[mask] = lut[np.minimum(codes[mask], 15)]
        small = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))
        return pygame.transform.scale(small, ((tx1 - tx0) * ts, (ty1 - ty0) * ts)).convert()

    def paint_chunk(self, cx: int, cy: int, ts: int) -> pygame.Surface:
        ct = self.chunk_tiles(ts)
        tx0, ty0 = cx * ct, cy * ct
        tx1, ty1 = min(self.W, tx0 + ct), min(self.H, ty0 + ct)
        surf = self.paint_coarse(cx, cy, ts)
        if ts < 6:
            self.paint_roads_simple(surf, tx0, ty0, tx1, ty1, ts)
            self.paint_buildings_simple(surf, tx0, ty0, tx1, ty1, ts)
            return surf
        for ty in range(ty0, ty1):
            row = self.l_terr[ty]
            for tx in range(tx0, tx1):
                px, py = (tx - tx0) * ts, (ty - ty0) * ts
                if self.l_road[ty][tx]:
                    self.paint_road(surf, tx, ty, px, py, ts)
                elif ts >= 10:
                    code = self.l_bt[ty][tx]
                    if code:
                        self.paint_lot_ground(surf, tx, ty, px, py, ts, code)
                    else:
                        self.paint_ground(surf, tx, ty, px, py, ts, TERRAIN_KEYS[row[tx]])
        # Buildings, painter order (far to near) so tall roofs overlap what is behind them.
        todo = []
        for row in range(max(0, ty0 - 4), min(self.H, ty1 + 3)):
            for idx in self.bld_by_row.get(row, ()):
                b = self.blds[idx]
                if b[0] + b[2] >= tx0 - 1 and b[0] <= tx1 and b[1] + b[3] >= ty0 - 1 and b[1] <= ty1 + 2:
                    todo.append(b)
        todo.sort(key=lambda b: (b[1] + b[3], b[0]))
        for b in todo:
            self.paint_building(surf, b, tx0, ty0, ts)
        if ts >= 10:
            for ty in range(ty0, ty1):
                for tx in range(tx0, tx1):
                    if not self.l_bt[ty][tx] and TERRAIN_KEYS[self.l_terr[ty][tx]] == "park":
                        self.paint_trees(surf, tx, ty, (tx - tx0) * ts, (ty - ty0) * ts, ts)
        return surf

    def paint_buildings_simple(self, surf: pygame.Surface, tx0: int, ty0: int, tx1: int, ty1: int, ts: int) -> None:
        for b in self.blds:
            x, y, w, h, facing, code, f, tier, bid, hsh = b
            if x + w < tx0 or x > tx1 or y + h < ty0 or y > ty1 + 2:
                continue
            name = BUILDING_NAMES.get(code, "house")
            base = self.building_palette(name, hsh, tier)[0]
            col = mix(base, (255, 255, 255), min(0.5, f / 28.0))
            inset = 1 if ts >= 3 else 0
            lift = int(min(ts * 0.5, f * 0.12 * ts / 2))
            px, py = (x - tx0) * ts + inset, (y - ty0) * ts + inset
            surf.fill(shade(base, 0.6), (px, py + h * ts - ts // 3, w * ts - inset, max(1, ts // 3)))
            surf.fill(col, (px, py - lift, w * ts - inset, h * ts - inset))

    # ------------------------------------------------------------------
    # Lots and buildings
    # ------------------------------------------------------------------
    def building_palette(self, name: str, hsh: int, tier: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """(wall colour, roof colour) varied per building."""
        k = (hsh >> 6)
        if name == "house":
            return HOUSE_WALLS[k % len(HOUSE_WALLS)], HOUSE_ROOFS[(k >> 3) % len(HOUSE_ROOFS)]
        if name == "apartment":
            wall = APT_WALLS[k % len(APT_WALLS)]
            return wall, shade(wall, 0.8)
        if name == "office":
            glass = GLASS[k % len(GLASS)]
            glass = mix(glass, (60, 80, 110), tier / 255.0 * 0.4)
            return glass, shade(glass, 0.75)
        if name == "shop":
            wall = SHOP_COLORS[k % len(SHOP_COLORS)]
            return wall, shade(wall, 0.8)
        base = FACADE.get(name, (200, 200, 200))
        return shade(base, 0.9 + ((k & 7) / 40.0)), shade(base, 1.0)

    def lot_ground_color(self, code: int, hsh: int) -> Tuple[int, int, int]:
        name = BUILDING_NAMES.get(code, "house")
        v = 0.94 + ((hsh >> 4) & 15) / 160.0
        if name in ("house", "apartment", "park_facility", "school", "college"):
            return shade((88, 152, 94), v)
        if name in ("factory", "warehouse", "prison"):
            return shade((118, 112, 120), v)
        return shade((126, 128, 134), v)

    def paint_lot_ground(self, surf: pygame.Surface, tx: int, ty: int, px: int, py: int, ts: int, code: int) -> None:
        hsh = self.l_hash[ty][tx]
        name = BUILDING_NAMES.get(code, "house")
        surf.fill(self.lot_ground_color(code, hsh), (px, py, ts, ts))
        if ts < 16:
            return
        if name in ("shop", "office", "hospital", "police", "court", "station"):
            # parking bays
            if hsh % 2 == 0:
                for k in range(3):
                    surf.fill((220, 220, 220), (px + int(ts * (0.18 + 0.28 * k)), py + int(ts * 0.72), max(1, ts // 40 + 1), int(ts * 0.22)))
        elif name in ("factory", "warehouse") and hsh % 3 == 0:
            surf.fill((236, 200, 60), (px + ts // 8, py + ts - ts // 5, ts * 3 // 4, max(1, ts // 30 + 1)))
        elif name in ("house", "apartment") and hsh % 5 == 0:
            pygame.draw.circle(surf, (218, 120, 150), (px + int(ts * 0.8), py + int(ts * 0.8)), max(1, ts // 18))

    def body_rect(self, b: Tuple[Any, ...], ts: int) -> Tuple[float, float, float, float]:
        """Ground footprint of the building mass inside its lot, relative to the lot origin."""
        x, y, w, h, facing, code, f, tier, bid, hsh = b
        name = BUILDING_NAMES.get(code, "house")
        lw, lh = w * ts, h * ts
        if name == "house":
            bw, bh = lw * (0.56 + ((hsh >> 12) & 3) * 0.04), lh * 0.5
            set_back = ts * 0.12
            if facing == 0:
                return (lw - bw) / 2, set_back, bw, bh
            if facing == 2:
                return (lw - bw) / 2, lh - bh - set_back, bw, bh
            if facing == 1:
                return lw - bw - set_back, (lh - bh) / 2, bw, bh
            return set_back, (lh - bh) / 2, bw, bh
        m = ts * (0.06 if name in ("factory", "warehouse") else 0.1 if name != "apartment" else 0.12)
        return m, m, lw - 2 * m, lh - 2 * m

    def building_rects(self, b: Tuple[Any, ...], ts: int) -> Tuple[pygame.Rect, pygame.Rect]:
        """(roof rect, front-wall rect) relative to the lot origin; walls rise to the north."""
        bx, by, bw, bh = self.body_rect(b, ts)
        f = b[6]
        lift = min(ts * 1.0, f * ts * 0.06)
        wall_h = ts * 0.15 + lift
        roof = pygame.Rect(int(bx), int(by - wall_h), max(2, int(bw)), max(2, int(bh)))
        wall = pygame.Rect(int(bx), int(by - wall_h + bh), max(2, int(bw)), max(2, int(wall_h)))
        return roof, wall

    def paint_building(self, surf: pygame.Surface, b: Tuple[Any, ...], tx0: int, ty0: int, ts: int) -> None:
        x, y, w, h, facing, code, f, tier, bid, hsh = b
        name = BUILDING_NAMES.get(code, "house")
        ox, oy = (x - tx0) * ts, (y - ty0) * ts
        if ts < 10:
            wall_c, roof_c = self.building_palette(name, hsh, tier)
            roof, wall = self.building_rects(b, ts)
            surf.fill(shade(wall_c, 0.7), wall.move(ox, oy))
            surf.fill(roof_c, roof.move(ox, oy))
            return
        wall_c, roof_c = self.building_palette(name, hsh, tier)
        roof, wall = self.building_rects(b, ts)
        bx, by, bw, bh = self.body_rect(b, ts)
        lw, lh = w * ts, h * ts
        roof = roof.move(ox, oy)
        wall = wall.move(ox, oy)
        # driveway / front path toward the street
        path = (172, 170, 166)
        dw = max(2, int(ts * (0.2 if name == "house" else 0.3)))
        if name != "park_facility":
            if facing == 0:
                surf.fill(path, (ox + int(bx + bw * 0.35), oy, dw, int(by)))
            elif facing == 2:
                surf.fill(path, (ox + int(bx + bw * 0.35), oy + int(by + bh), dw, int(lh - by - bh)))
            elif facing == 1:
                surf.fill(path, (ox + int(bx + bw), oy + int(by + bh * 0.4), int(lw - bx - bw), dw))
            else:
                surf.fill(path, (ox, oy + int(by + bh * 0.4), int(bx), dw))
        # trees in the yard margins
        if name in ("house", "apartment", "school", "college") and ts >= 16:
            for i in range(1 + (hsh >> 3) % 2):
                side = ((hsh >> (5 + i)) & 1)
                tx_ = ox + (int(bx * 0.45) if side == 0 else int(bx + bw + (lw - bx - bw) * 0.5))
                ty_ = oy + int(lh * (0.3 + 0.4 * (((hsh >> (9 + i)) & 7) / 7.0)))
                r = max(2, int(ts * 0.12))
                pygame.draw.ellipse(surf, (40, 80, 50), (tx_ - r, ty_ + r // 2, r * 2, max(2, r)))
                pygame.draw.circle(surf, (36, 108, 58), (tx_, ty_), r)
                pygame.draw.circle(surf, (60, 144, 78), (tx_ - r // 3, ty_ - r // 3), max(1, r * 2 // 3))
        # fence for houses
        if name == "house" and ts >= 24 and (hsh >> 2) & 1:
            pygame.draw.rect(surf, (214, 206, 190), (ox + 2, oy + 2, lw - 4, lh - 4), 1)
        # shadows
        surf.fill((60, 60, 60), (wall.x + int(ts * 0.1), wall.bottom - 1, wall.w, max(1, int(ts * 0.1))), special_flags=pygame.BLEND_RGB_SUB)
        surf.fill((48, 48, 48), (wall.right, roof.y + int(wall.h * 0.35), max(1, int(wall.h * 0.28)), roof.h + int(wall.h * 0.65)), special_flags=pygame.BLEND_RGB_SUB)
        # front wall
        surf.fill(shade(wall_c, 0.74), wall)
        # roof
        surf.fill(roof_c, roof)
        edge = shade(roof_c, 0.65)
        pygame.draw.rect(surf, edge, roof, 1)
        self.paint_roof_details(surf, name, roof, wall, hsh, f, tier, ts, roof_c)
        if name == "shop":
            awn = AWNINGS[(hsh >> 9) % len(AWNINGS)]
            for k in range(0, wall.w, max(3, ts // 6)):
                surf.fill(awn if (k // max(3, ts // 6)) % 2 == 0 else (240, 240, 240), (wall.x + k, wall.y + wall.h // 2, max(2, ts // 6), max(2, int(ts * 0.1))))
        if wall.h >= 5 and ts >= 12:
            glassy = name in ("office",)
            for rect in self.window_rects(b, ts, wall.move(-ox, -oy)):
                col = (176, 214, 238) if glassy else (44, 60, 82)
                surf.fill(col, rect.move(ox, oy))
            door_w = max(2, ts // 9)
            surf.fill((70, 50, 38), (wall.centerx - door_w // 2, wall.bottom - max(3, ts // 6), door_w, max(3, ts // 6)))
        pygame.draw.rect(surf, shade(wall_c, 0.55), wall, 1)

    def paint_roof_details(self, surf: pygame.Surface, name: str, roof: pygame.Rect, wall: pygame.Rect, hsh: int, f: int, tier: int, ts: int, roof_c: Tuple[int, int, int] = (120, 120, 120)) -> None:
        if name == "house":
            if (hsh >> 14) & 1:
                pygame.draw.line(surf, shade(roof_c, 0.7), (roof.x, roof.centery), (roof.right - 1, roof.centery), max(1, ts // 24))
                surf.fill(shade(roof_c, 1.15), (roof.x + 1, roof.y + 1, roof.w - 2, max(1, roof.h // 2 - 1)))
            else:
                pygame.draw.line(surf, shade(roof_c, 0.7), (roof.centerx, roof.y), (roof.centerx, roof.bottom - 1), max(1, ts // 24))
            if (hsh >> 15) & 1:
                surf.fill((92, 60, 50), (roof.x + roof.w // 5, roof.y + 1, max(2, ts // 12), max(2, ts // 9)))
        elif name == "apartment":
            rim = shade(roof_c, 0.8)
            pygame.draw.rect(surf, rim, roof.inflate(-4, -4), 1)
            for i in range(2):
                surf.fill((150, 154, 160), (roof.x + int(roof.w * (0.18 + 0.4 * i)), roof.y + roof.h // 3, max(2, ts // 8), max(2, ts // 10)))
            if wall.h > ts * 0.5:   # balconies
                for k in range(0, wall.w - 3, max(4, ts // 4)):
                    surf.fill((70, 72, 78), (wall.x + k + 1, wall.y + wall.h // 2, max(2, ts // 8), max(1, ts // 24)))
        elif name == "office":
            if f >= 14 and tier > 140:
                pygame.draw.circle(surf, (230, 230, 120), roof.center, max(3, min(roof.w, roof.h) // 3), 1)
                self.blit_label(surf, "H", roof.center, (230, 230, 120))
            else:
                for i in range(2):
                    surf.fill((136, 142, 150), (roof.x + int(roof.w * (0.2 + 0.4 * i)), roof.y + roof.h // 3, max(2, ts // 8), max(2, ts // 10)))
            pygame.draw.line(surf, (210, 220, 235), (roof.x + 2, roof.y + 2), (roof.right - 3, roof.y + 2), 1)
        elif name == "shop":
            surf.fill((250, 250, 250), (roof.x + roof.w // 4, roof.y + roof.h // 3, roof.w // 2, max(2, ts // 10)))
        elif name == "factory":
            n = max(2, roof.w // max(6, ts // 3))
            for i in range(n):
                sx = roof.x + int(roof.w * i / n)
                pygame.draw.polygon(surf, (96, 90, 88), [(sx, roof.bottom - 1), (sx + roof.w // (2 * n), roof.y + 1), (sx + roof.w // n, roof.bottom - 1)])
            surf.fill((74, 74, 78), (roof.right - ts // 5, roof.y - ts // 5, max(2, ts // 9), ts // 3 + 2))
            if (hsh >> 8) & 1:
                pygame.draw.circle(surf, (190, 194, 200), (roof.x + roof.w // 5, roof.bottom + ts // 8), max(3, ts // 7))
        elif name == "warehouse":
            for k in range(0, wall.w - 2, max(5, ts // 3)):
                surf.fill((92, 98, 106), (wall.x + k + 2, wall.bottom - max(3, ts // 5), max(3, ts // 5), max(3, ts // 5)))
        elif name == "hospital":
            cx, cy = roof.center
            arm = max(2, ts // 7)
            surf.fill((214, 50, 50), (cx - arm, cy - arm // 3, arm * 2, max(2, arm * 2 // 3)))
            surf.fill((214, 50, 50), (cx - arm // 3, cy - arm, max(2, arm * 2 // 3), arm * 2))
        elif name == "police":
            surf.fill((40, 70, 150), (roof.x, roof.y + roof.h // 3, roof.w, max(2, roof.h // 5)))
        elif name == "school":
            surf.fill((200, 70, 70), (roof.x + roof.w // 3, roof.y + 1, max(2, roof.w // 3), max(2, roof.h // 4)))
            pygame.draw.line(surf, (200, 200, 200), (roof.right - 3, roof.y), (roof.right - 3, roof.y - ts // 3), 1)
        elif name == "college":
            surf.fill((200, 190, 160), (roof.centerx - ts // 8, roof.y - ts // 4, max(3, ts // 4), ts // 3))
            pygame.draw.circle(surf, (250, 250, 240), (roof.centerx, roof.y - ts // 8), max(2, ts // 12))
        elif name == "station":
            surf.fill((70, 110, 170), (roof.x + 2, roof.y + 2, roof.w - 4, max(2, roof.h // 3)))
        elif name == "park_facility":
            pygame.draw.circle(surf, (60, 130, 70), roof.center, max(3, roof.w // 3))
        elif name == "prison":
            for k in range(0, roof.w, max(3, ts // 6)):
                pygame.draw.line(surf, (80, 80, 88), (roof.x + k, roof.y), (roof.x + k, roof.bottom - 1), 1)
        elif name == "court":
            for k in range(1, 5):
                pygame.draw.line(surf, (235, 232, 220), (wall.x + k * wall.w // 5, wall.y), (wall.x + k * wall.w // 5, wall.bottom - 1), max(1, ts // 24))

    def blit_label(self, surf: pygame.Surface, text: str, center: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        key = ("label", text, color)
        img = self.sprite_cache.get(key)
        if img is None:
            if not pygame.font.get_init():
                return
            img = pygame.font.SysFont("consolas", 12, bold=True).render(text, True, color)
            self.sprite_cache[key] = img
        surf.blit(img, (center[0] - img.get_width() // 2, center[1] - img.get_height() // 2))

    def window_rects(self, b: Tuple[Any, ...], ts: int, wall: pygame.Rect) -> List[pygame.Rect]:
        code, floors = b[5], b[6]
        name = BUILDING_NAMES.get(code, "house")
        rows = max(1, min(floors, int((wall.h - ts * 0.12) // max(3, ts * 0.11)), 12))
        cols = max(1, int(wall.w // max(4, ts * 0.17)))
        if name == "house":
            rows, cols = 1, 2
        if name in ("factory", "warehouse"):
            rows = 1
            cols = max(1, cols // 2)
        out = []
        top = wall.y + max(1, int(ts * 0.03))
        usable = wall.h - max(3, int(ts * 0.14))
        for r in range(rows):
            for c in range(cols):
                cw = wall.w / cols
                rh = usable / rows
                ww = max(1, int(cw * 0.55))
                wh = max(1, int(rh * 0.5))
                out.append(pygame.Rect(wall.x + int(c * cw + (cw - ww) / 2), top + int(r * rh + (rh - wh) / 2), ww, wh))
        return out

    def draw_night(self, dest: pygame.Surface, x0: float, y0: float, tw: float, th: float, ts: int, darkness: float,
                   hour_f: float, frame: Dict[str, Any], agents: Dict[str, Any]) -> None:
        """Additive glows: street lamps, lit windows and vehicle headlights."""
        if darkness <= 0.05 or ts < 6:
            return
        w, h = dest.get_size()
        intensity = min(1.0, darkness * 1.3)
        if ts >= 10 and self.lamp_arr.shape[0]:
            arr = self.lamp_arr
            vis = (arr[:, 0] >= x0 - 2) & (arr[:, 0] < x0 + tw + 2) & (arr[:, 1] >= y0 - 2) & (arr[:, 1] < y0 + th + 2)
            radius = max(6, int(ts * 1.5))
            g = self.glow(radius, 0.75 * intensity)
            for tx, ty in arr[vis].tolist():
                dest.blit(g, (int((tx + 0.5 - x0) * ts) - radius, int((ty + 0.5 - y0) * ts) - radius), special_flags=pygame.BLEND_RGB_ADD)
        if ts >= 16 and self.bld_arr.shape[0]:
            arr = self.bld_arr
            vis = np.nonzero((arr[:, 0] >= x0 - 4) & (arr[:, 0] < x0 + tw + 1) & (arr[:, 1] >= y0 - 5) & (arr[:, 1] < y0 + th + 2))[0]
            hour = int(hour_f) % 24
            for idx in vis.tolist():
                b = self.blds[idx]
                x, y, bw, bh, facing, code, f, tier, bid, hsh = b
                name = BUILDING_NAMES.get(code, "house")
                if name in ("house", "apartment"):
                    prob = 0.75 if (17 <= hour <= 23 or hour <= 1) else 0.18
                elif name in ("hospital", "police", "station", "prison"):
                    prob = 0.85
                else:
                    prob = 0.6 if 7 <= hour <= 20 else 0.1
                ox = (x - x0) * ts
                oy = (y - y0) * ts
                roof, wall = self.building_rects(b, ts)
                for i, rect in enumerate(self.window_rects(b, ts, wall)):
                    if ((hsh >> (i % 16)) ^ (i * 2654435761 >> 5)) % 100 < prob * 100:
                        dest.fill((int(240 * intensity), int(200 * intensity), int(110 * intensity)), rect.move(int(ox), int(oy)), special_flags=pygame.BLEND_RGB_ADD)
        if ts >= 14:
            mv = agents["moving"]
            mode = frame["mode"]
            sel = np.nonzero(mv & ((mode == 2) | (mode == 4)) & (frame["indoors"] == 0))[0]
            if sel.size:
                radius = max(6, int(ts * 0.9))
                g = self.glow(radius, 0.6 * intensity)
                for i in sel[:600].tolist():
                    sx = (agents["x"][i] - x0) * ts
                    sy = (agents["y"][i] - y0) * ts
                    if -radius < sx < w + radius and -radius < sy < h + radius:
                        dest.blit(g, (int(sx) - radius, int(sy) - radius), special_flags=pygame.BLEND_RGB_ADD)
