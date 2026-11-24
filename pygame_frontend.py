"""
pygame_frontend.py

High-performance pygame renderer for the life simulation.

Architecture
------------
* The simulation runs in a separate process (see sim_host.py).  This process
  only ever renders the most recent snapshot, so frame rate does not depend on
  population size or tick cost.
* The terrain/buildings layer is pre-rendered once per zoom level into a
  cached surface; each frame is then a single sub-rectangle blit.
* Citizens are stored as numpy arrays and drawn with vectorised pixel writes
  (zoomed out) or per-person sprites (zoomed in), only for the visible area.
* Positions are interpolated between snapshots for smooth motion.
* All text is cached as pre-rendered surfaces.

Usage:  python pygame_frontend.py [--preset large|small|classic] [--seed N]
                                   [--world-seed N] [--load FILE]
                                   [--bench-frames N]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if "--bench-frames" in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from sim_host import SimClient
from sim_protocol import ACTIVITIES, BUILDING_CODES, TERRAIN_KEYS
from world_renderer import ZOOM_STEPS, WorldRenderer
from world import BUILDING_TYPES

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 920
TARGET_FPS = 60
MIN_TILE = ZOOM_STEPS[0] if False else 2
MAX_TILE = 96

PANEL_BG = (18, 24, 32)
PANEL_BG_ALT = (23, 31, 42)
PANEL_BORDER = (72, 95, 122)
TEXT = (232, 237, 243)
TEXT_DIM = (164, 176, 192)
TEXT_WARN = (246, 205, 97)
TEXT_DANGER = (242, 111, 111)
TEXT_GOOD = (116, 213, 138)
ACCENT = (86, 184, 255)
ACCENT_2 = (255, 153, 84)
ACCENT_3 = (149, 218, 112)

TERRAIN_COLORS = {
    "water": (33, 92, 158),
    "park": (59, 132, 80),
    "rural": (137, 126, 87),
    "suburban": (118, 140, 112),
    "urban": (83, 102, 122),
    "industrial": (121, 98, 127),
    "road": (88, 91, 97),
    "rail": (153, 93, 65),
}
TEXTURE_ALIASES = {
    "road": "road", "rail": "rail", "water": "water", "park": "grass", "rural": "dirt",
}
BUILDING_TEXTURES = {
    "house": ["house1", "house2"], "apartment": ["apartment1", "apartment2"],
    "shop": ["shop1", "shop2"], "office": ["office1", "office2"], "factory": ["factory1"],
    "school": ["school1"], "college": ["college1"], "police": ["police1"], "court": ["court1"],
    "prison": ["prison1"], "hospital": ["hospital1"], "park_facility": ["park_facility1"],
    "station": ["station1"], "warehouse": ["warehouse1"],
}
BUILDING_COLORS = {
    "house": (217, 189, 130), "apartment": (200, 206, 215), "shop": (255, 195, 115),
    "office": (147, 188, 228), "factory": (166, 133, 128), "warehouse": (143, 126, 116),
    "school": (248, 214, 117), "college": (242, 163, 104), "police": (115, 168, 214),
    "hospital": (128, 211, 184), "station": (174, 171, 220), "park_facility": (122, 187, 108),
    "court": (190, 180, 148), "prison": (122, 122, 132),
}
ACTIVITY_COLORS = {
    "sleep": (110, 120, 200), "rest": (210, 214, 222), "work": (255, 190, 80),
    "commute_work": (255, 120, 80), "commute_home": (255, 120, 80), "commute_school": (255, 120, 80),
    "study": (130, 220, 130), "study_self": (130, 220, 130), "socialize": (255, 120, 200),
    "exercise": (90, 230, 210), "errand": (240, 235, 110), "clinic": (255, 90, 90),
    "job_search": (190, 150, 255), "break": (255, 225, 150), "other": (160, 160, 160),
}
INFECTION_COLORS = [(235, 240, 245), (255, 214, 84), (255, 70, 70), (110, 200, 255)]

HEATMAPS = ["off", "density", "happiness", "stress", "health", "infection", "traffic", "property", "crime", "wealth"]
COLOR_MODES = ["activity", "happiness", "stress", "health", "infection", "age"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

MENU_DEFS = [
    ("View", [("Citizen", "tab:citizen"), ("Building Inspector", "tab:building"), ("Metrics", "tab:metrics"), ("Charts", "tab:charts"),
              ("City", "tab:city"), ("Health & Env", "tab:health"), ("News", "tab:news"),
              ("Companies", "tab:companies"), ("Timeline", "tab:timeline"),
              ("Relationships", "tab:relationships"), ("Help", "tab:help"), ("Hide Panel", "tab:none")]),
    ("Map", [("Zoom In", "map:zoom_in"), ("Zoom Out", "map:zoom_out"), ("Center Citizen", "map:center"),
             ("Cycle Heatmap", "map:heat"), ("Cycle Citizen Colors", "map:color"),
             ("Toggle Minimap", "map:minimap"), ("Toggle Smooth Motion", "map:smooth")]),
    ("Sim", [("Pause/Resume", "sim:pause"), ("Faster", "sim:faster"), ("Slower", "sim:slower"),
             ("Save", "sim:save"), ("Load", "sim:load")]),
]


def lerp_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


def make_ramp(stops: List[Tuple[int, int, int]], n: int = 256) -> np.ndarray:
    lut = np.zeros((n, 3), dtype=np.uint8)
    seg = len(stops) - 1
    for i in range(n):
        pos = i / (n - 1) * seg
        k = min(seg - 1, int(pos))
        lut[i] = lerp_color(stops[k], stops[k + 1], pos - k)
    return lut


RAMP_HEAT = make_ramp([(20, 30, 90), (40, 130, 190), (90, 200, 120), (250, 220, 70), (240, 70, 50)])
RAMP_GOOD = make_ramp([(215, 60, 55), (240, 190, 70), (90, 205, 110)])       # low=bad .. high=good
RAMP_BAD = make_ramp([(90, 205, 110), (240, 190, 70), (215, 60, 55)])        # low=good .. high=bad
RAMP_VALUE = make_ramp([(30, 40, 80), (80, 120, 200), (170, 210, 120), (250, 240, 130)])


def fmt_num(value: float) -> str:
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:.2f}B"
    if a >= 1e6:
        return f"{value / 1e6:.2f}M"
    if a >= 1e4:
        return f"{value / 1e3:.1f}k"
    if a >= 100:
        return f"{value:.0f}"
    if a >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


class Frontend:
    def __init__(self, client: SimClient, bench_frames: int = 0, shot_prefix: str = "") -> None:
        self.client = client
        self.bench_frames = bench_frames
        self.shot_prefix = shot_prefix
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Life Sim")
        self.clock = pygame.time.Clock()
        self.running = True
        self.status_message = "WASD/arrows pan, wheel zoom, Space pause, H heatmap, V colors, F1 help."
        self.fonts: Dict[str, pygame.font.Font] = {
            "ui": pygame.font.SysFont("segoeui", 18),
            "small": pygame.font.SysFont("consolas", 15),
            "tiny": pygame.font.SysFont("consolas", 13),
            "title": pygame.font.SysFont("verdana", 26, bold=True),
        }
        self.text_cache: Dict[Tuple[Any, ...], pygame.Surface] = {}
        self.bg_cache: Dict[Tuple[int, int], pygame.Surface] = {}
        # camera in tile coordinates (float centre)
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.tile_size = 10
        self.zoom_initialised = False
        self.active_tab: Optional[str] = "citizen"
        self.active_menu: Optional[str] = None
        self.heatmap = 0
        self.color_mode = 0
        self.show_minimap = True
        self.smooth = True
        self.news_scroll = 0
        self.timeline_scroll = 0
        self.drag_pos: Optional[Tuple[int, int]] = None
        # static world
        self.static_version = -1
        self.layers_version = -1
        self.daily_version = -1
        self.static: Dict[str, Any] = {}
        self.layers: Dict[str, Any] = {}
        self.daily: Dict[str, Any] = {}
        self.world_surface: Optional[pygame.Surface] = None
        self.world_surface_key: Tuple[int, int] = (0, 0)
        self.minimap_base: Optional[pygame.Surface] = None
        self.minimap_scale = 1
        # textures
        self.texture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")
        self.terrain_tex: Dict[str, pygame.Surface] = {}
        self.sprites_people: List[pygame.Surface] = []
        self.sprites_buildings: Dict[str, List[pygame.Surface]] = {}
        self.scaled_cache: Dict[Tuple[Any, ...], pygame.Surface] = {}
        self.wr = WorldRenderer()
        self.delay = 0.3
        self.sim_minutes = 0.0
        self.last_agents: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        self.terrain_lut = np.array(
            [TERRAIN_COLORS.get(key, (90, 90, 90)) for key in TERRAIN_KEYS], dtype=np.int16
        )
        # per-frame caches
        self.interp_key: Tuple[int, int] = (-1, -1)
        self.interp_map: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.overlay_cache: Dict[str, Any] = {}
        self.tint_cache: Dict[Tuple[Any, ...], pygame.Surface] = {}
        self.frame_times: List[float] = []
        self.fps_display = 0.0
        self.draw_ms = 0.0
        self.rain_seed = 0
        self.last_frame_tick = -1
        self.person_arrays_key = -1
        self.person_colors: Optional[np.ndarray] = None
        self.hit_rects: List[Tuple[pygame.Rect, str]] = []

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------
    def load_textures(self) -> None:
        """Textures are gone: the world is rendered procedurally (see world_renderer.py)."""
        return

    def scaled(self, kind: str, surf: pygame.Surface, size: int, idx: int = 0) -> pygame.Surface:
        key = (kind, idx, size)
        cached = self.scaled_cache.get(key)
        if cached is None:
            cached = pygame.transform.smoothscale(surf, (size, size))
            self.scaled_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def text(self, font_key: str, string: str, color: Tuple[int, int, int] = TEXT) -> pygame.Surface:
        key = (font_key, string, color)
        surf = self.text_cache.get(key)
        if surf is None:
            if len(self.text_cache) > 3000:
                self.text_cache.clear()
            surf = self.fonts[font_key].render(string, True, color)
            self.text_cache[key] = surf
        return surf

    def blit_text(self, font_key: str, string: str, pos: Tuple[int, int], color: Tuple[int, int, int] = TEXT) -> int:
        surf = self.text(font_key, string, color)
        self.screen.blit(surf, pos)
        return surf.get_width()

    def fit(self, font_key: str, string: str, width: int) -> str:
        font = self.fonts[font_key]
        if font.size(string)[0] <= width:
            return string
        lo, hi = 0, len(string)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.size(string[:mid] + "...")[0] <= width:
                lo = mid
            else:
                hi = mid - 1
        return string[:lo] + "..."

    @staticmethod
    def line_color(text: str) -> Tuple[int, int, int]:
        lower = text.lower()
        if any(w in lower for w in ("bankruptcy", "convicted", "crime", "laid off", "meltdown", "failed", "died", "outbreak", "deficit")):
            return TEXT_DANGER
        if any(w in lower for w in ("hired", "graduated", "goodwill", "refund", "promotion", "raise", "recovered", "vaccin")):
            return TEXT_GOOD
        if any(w in lower for w in ("warning", "stress", "debt", "rent", "tax")):
            return TEXT_WARN
        return TEXT

    # ------------------------------------------------------------------
    # Data intake
    # ------------------------------------------------------------------
    def sync_data(self) -> None:
        c = self.client
        with c.lock:
            if c.static_version != self.static_version and c.static is not None:
                self.static = c.static
                self.static_version = c.static_version
                self.wr.set_static(self.static)
                self.world_surface = None
                self.minimap_base = None
                self.overlay_cache.clear()
                self.interp_key = (-1, -1)
                if not self.zoom_initialised:
                    self.cam_x = self.static["width"] / 2.0
                    self.cam_y = self.static["height"] / 2.0
            if c.layers_version != self.layers_version and c.layers is not None:
                self.layers = c.layers
                self.layers_version = c.layers_version
                for key in [k for k in self.overlay_cache if k.startswith("layer:")]:
                    del self.overlay_cache[key]
            if c.daily_version != self.daily_version and c.daily is not None:
                self.daily = c.daily
                self.daily_version = c.daily_version

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def layout(self) -> Dict[str, Any]:
        width, height = self.screen.get_size()
        header_h, footer_h = 52, 36
        map_rect = pygame.Rect(10, header_h + 14, width - 20, height - header_h - footer_h - 26)
        panel_w = min(560, max(340, width // 3))
        panel_rect = pygame.Rect(map_rect.right - panel_w - 8, map_rect.y + 8, panel_w, min(map_rect.height - 16, 760))
        return {
            "w": width, "h": height,
            "header": pygame.Rect(10, 8, width - 20, header_h),
            "footer": pygame.Rect(10, height - footer_h - 6, width - 20, footer_h),
            "map": map_rect,
            "panel": panel_rect,
        }

    def view_geometry(self, lay: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Return (x0, y0, tiles_w, tiles_h) of the visible region in tile units."""
        ts = self.tile_size
        map_rect = lay["map"]
        tw = map_rect.width / ts
        th = map_rect.height / ts
        W, H = self.static["width"], self.static["height"]
        if tw >= W:
            x0 = (W - tw) / 2.0
            self.cam_x = W / 2.0
        else:
            self.cam_x = min(max(self.cam_x, tw / 2.0), W - tw / 2.0)
            x0 = self.cam_x - tw / 2.0
        if th >= H:
            y0 = (H - th) / 2.0
            self.cam_y = H / 2.0
        else:
            self.cam_y = min(max(self.cam_y, th / 2.0), H - th / 2.0)
            y0 = self.cam_y - th / 2.0
        return x0, y0, tw, th

    # ------------------------------------------------------------------
    # World surface (pre-rendered terrain + buildings)
    # ------------------------------------------------------------------
    def person_color_table(self, frame: Dict[str, Any]) -> np.ndarray:
        key = (frame["tick"], self.color_mode)
        if self.person_arrays_key == key and self.person_colors is not None:
            return self.person_colors
        mode = COLOR_MODES[self.color_mode]
        n = frame["population"]
        if mode == "activity":
            lut = np.array([ACTIVITY_COLORS.get(a, (200, 200, 200)) for a in ACTIVITIES] + [(160, 160, 160)] * 2, dtype=np.uint8)
            colors = lut[frame["activity"]]
        elif mode == "happiness":
            colors = RAMP_GOOD[frame["happy"]]
        elif mode == "stress":
            colors = RAMP_BAD[frame["stress"]]
        elif mode == "health":
            colors = RAMP_GOOD[frame["health"]]
        elif mode == "infection":
            lut = np.array(INFECTION_COLORS + [(200, 200, 200)] * 4, dtype=np.uint8)
            colors = lut[frame["infection"]]
        else:
            colors = RAMP_VALUE[np.clip(frame["age"].astype(np.int32) * 3, 0, 255)]
        self.person_colors = colors
        self.person_arrays_key = key
        return colors

    def heatmap_surface(self, frame: Dict[str, Any]) -> Optional[pygame.Surface]:
        mode = HEATMAPS[self.heatmap]
        if mode == "off":
            return None
        W, H = self.static["width"], self.static["height"]
        per_person = mode in ("density", "happiness", "stress", "health", "infection", "traffic")
        key = f"live:{mode}:{frame['tick']}" if per_person else f"layer:{mode}:{self.layers_version}"
        cached = self.overlay_cache.get(key)
        if cached is not None:
            return cached
        for k in [k for k in self.overlay_cache if k.startswith("live:")]:
            del self.overlay_cache[k]
        if mode == "traffic":
            pts = frame["mv_pts"].astype(np.int64)
            lens = np.diff(frame["mv_off"])
            veh = np.repeat(np.isin(frame["mv_mode"], (2, 3, 4)), lens) if pts.shape[0] else np.zeros(0, dtype=bool)
            counts = np.zeros(W * H, dtype=np.float32)
            if pts.shape[0] and veh.any():
                v = pts[veh]
                counts = np.bincount(v[:, 1] * W + v[:, 0], minlength=W * H).astype(np.float32)
            blurred = self.blur(counts.reshape(H, W))
            norm = np.clip(blurred / 2.5, 0, 1)
            present = blurred > 0.03
            lut = RAMP_BAD
            rgb = lut[(norm * 255).astype(np.uint8)]
            surf = pygame.Surface((W, H), pygame.SRCALPHA)
            px = pygame.surfarray.pixels3d(surf)
            px[:] = rgb.swapaxes(0, 1)
            del px
            pa = pygame.surfarray.pixels_alpha(surf)
            pa[:] = np.where(present, 200, 0).astype(np.uint8).T
            del pa
            self.overlay_cache[key] = surf
            return surf
        if per_person:
            flat = frame["y"].astype(np.int64) * W + frame["x"].astype(np.int64)
            counts = np.bincount(flat, minlength=W * H).astype(np.float32).reshape(H, W)
            if mode == "density":
                grid = counts
                blurred = self.blur(grid)
                norm = np.clip(np.log1p(blurred) / max(0.5, np.log1p(max(1.0, np.percentile(blurred[blurred > 0], 98) if (blurred > 0).any() else 1.0))), 0, 1)
                lut = RAMP_HEAT
                present = blurred > 0.02
            else:
                src = {"happiness": frame["happy"], "stress": frame["stress"], "health": frame["health"],
                       "infection": (frame["infection"] == 2).astype(np.uint8) * 255}[mode]
                sums = np.bincount(flat, weights=src.astype(np.float32), minlength=W * H).astype(np.float32).reshape(H, W)
                bs, bc = self.blur(sums), self.blur(counts)
                present = bc > 0.05
                mean = np.where(present, bs / np.maximum(bc, 1e-3), 0) / 255.0
                if mode == "infection":
                    norm = np.clip(mean * 6.0, 0, 1)
                    present = bs > 0.3
                    lut = RAMP_HEAT
                else:
                    norm = np.clip(mean, 0, 1)
                    lut = RAMP_BAD if mode == "stress" else RAMP_GOOD
                    if mode == "stress":
                        norm = np.clip((norm - 0.2) / 0.6, 0, 1)
                    else:
                        norm = np.clip((norm - 0.2) / 0.6, 0, 1)
        else:
            lay = self.layers.get({"property": "property_value", "crime": "crime", "wealth": "wealth"}[mode])
            if lay is None:
                return None
            grid = lay.astype(np.float32)
            lo, hi = float(np.percentile(grid, 2)), float(np.percentile(grid, 98))
            norm = np.clip((grid - lo) / max(1e-6, hi - lo), 0, 1)
            lut = RAMP_BAD if mode == "crime" else RAMP_VALUE
            present = np.ones_like(norm, dtype=bool)
        rgb = lut[(norm * 255).astype(np.uint8)]
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        px = pygame.surfarray.pixels3d(surf)
        px[:] = rgb.swapaxes(0, 1)
        del px
        pa = pygame.surfarray.pixels_alpha(surf)
        pa[:] = np.where(present, 175, 0).astype(np.uint8).T
        del pa
        self.overlay_cache[key] = surf
        return surf

    @staticmethod
    def blur(grid: np.ndarray) -> np.ndarray:
        out = grid.copy()
        for axis in (0, 1):
            out = out + np.roll(grid if axis == 0 else out, 1, axis=axis) + np.roll(grid if axis == 0 else out, -1, axis=axis)
        return out / 3.0

    # ------------------------------------------------------------------
    # Atmosphere
    # ------------------------------------------------------------------
    def daylight_multiplier(self, hour_f: float) -> Tuple[int, int, int]:
        h = hour_f % 24
        if 7.5 <= h <= 17.5:
            return (255, 255, 255)
        night = (92, 108, 160)
        dusk = (255, 190, 150)
        if 5.0 <= h < 7.5:
            t = (h - 5.0) / 2.5
            return lerp_color(night, dusk, min(1.0, t * 2)) if t < 0.5 else lerp_color(dusk, (255, 255, 255), (t - 0.5) * 2)
        if 17.5 < h <= 20.5:
            t = (h - 17.5) / 3.0
            return lerp_color((255, 255, 255), dusk, min(1.0, t * 2)) if t < 0.5 else lerp_color(dusk, night, (t - 0.5) * 2)
        return night

    def tint_surface(self, size: Tuple[int, int], color: Tuple[int, int, int]) -> pygame.Surface:
        q = tuple(int(c) // 6 * 6 for c in color)
        key = (size, q)
        surf = self.tint_cache.get(key)
        if surf is None:
            if len(self.tint_cache) > 40:
                self.tint_cache.clear()
            surf = pygame.Surface(size)
            surf.fill(q)
            self.tint_cache[key] = surf
        return surf

    def draw_weather(self, rect: pygame.Rect, frame: Dict[str, Any]) -> None:
        cond = frame["weather_condition"]
        tints = {
            "cloudy": (180, 190, 205, 26), "rain": (100, 136, 190, 36), "storm": (65, 88, 128, 60),
            "fog": (205, 212, 218, 70), "heatwave": (255, 139, 76, 36), "cold_snap": (130, 176, 255, 30),
        }
        tint = tints.get(cond)
        if tint:
            key = ("wx", rect.size, cond)
            surf = self.tint_cache.get(key)
            if surf is None:
                surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                surf.fill(tint)
                self.tint_cache[key] = surf
            self.screen.blit(surf, rect.topleft)
        if cond in ("rain", "storm"):
            count = 160 if cond == "rain" else 320
            t = time.monotonic()
            rng = np.random.default_rng(7)
            bx = rng.random(count)
            by = rng.random(count)
            speed = 0.9 + rng.random(count) * 0.8
            fx = ((bx + t * 0.05 * speed) % 1.0) * rect.width
            fy = ((by + t * speed) % 1.0) * rect.height
            slant = 6 if cond == "storm" else 3
            for i in range(count):
                x = rect.x + int(fx[i])
                y = rect.y + int(fy[i])
                pygame.draw.line(self.screen, (170, 195, 235), (x, y), (x - slant, y + 9), 1)

    # ------------------------------------------------------------------
    # Main draw
    # ------------------------------------------------------------------
    def draw_background(self, w: int, h: int) -> None:
        bg = self.bg_cache.get((w, h))
        if bg is None:
            self.bg_cache.clear()
            bg = pygame.Surface((w, h))
            for i in range(0, h, 3):
                blend = i / max(1, h)
                pygame.draw.rect(bg, (int(14 + 12 * blend), int(20 + 18 * blend), int(28 + 24 * blend)), (0, i, w, 3))
            self.bg_cache[(w, h)] = bg
        self.screen.blit(bg, (0, 0))

    def draw_panel(self, rect: pygame.Rect, title: str, alt: bool = False) -> pygame.Rect:
        pygame.draw.rect(self.screen, PANEL_BG_ALT if alt else PANEL_BG, rect, border_radius=12)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=2, border_radius=12)
        self.blit_text("ui", title, (rect.x + 14, rect.y + 8))
        pygame.draw.line(self.screen, ACCENT, (rect.x + 12, rect.y + 36), (rect.right - 12, rect.y + 36), 2)
        return pygame.Rect(rect.x + 14, rect.y + 46, rect.width - 28, rect.height - 58)

    def draw(self) -> None:
        t0 = time.perf_counter()
        self.hit_rects = []
        self.sync_data()
        lay = self.layout()
        w, h = lay["w"], lay["h"]
        self.draw_background(w, h)
        frame, prev, frame_time, interval = self.client.snapshot()
        if self.client.fatal:
            self.draw_error(lay)
        elif not self.static or frame is None:
            self.draw_loading(lay)
        else:
            if not self.zoom_initialised:
                self.init_zoom(lay)
            self.draw_header(lay["header"], frame)
            cur, u = self.display_state()
            self.draw_map(lay, frame, cur or frame, u)
            self.draw_panel_tab(lay, frame)
            self.draw_footer(lay["footer"], frame)
        self.draw_menu_bar(lay["header"])
        pygame.display.flip()
        self.draw_ms = (time.perf_counter() - t0) * 1000.0

    def init_zoom(self, lay: Dict[str, Any]) -> None:
        W, H = self.static["width"], self.static["height"]
        rect = lay["map"]
        fit = min(rect.width / W, rect.height / H)
        fits = [z for z in ZOOM_STEPS if z <= fit]
        self.tile_size = fits[-1] if fits else ZOOM_STEPS[0]
        self.zoom_initialised = True

    def draw_loading(self, lay: Dict[str, Any]) -> None:
        rect = lay["map"]
        body = self.draw_panel(rect, "City Map", alt=True)
        angle = time.monotonic() * 4
        cx, cy = body.centerx, body.centery - 20
        for i in range(12):
            a = angle + i * math.tau / 12
            shade = int(60 + 190 * (i / 12))
            pygame.draw.circle(self.screen, (shade // 3, shade // 2, shade), (int(cx + math.cos(a) * 34), int(cy + math.sin(a) * 34)), 6)
        msg = self.client.status
        surf = self.text("ui", msg)
        self.screen.blit(surf, (cx - surf.get_width() // 2, cy + 64))
        sub = self.text("small", "The simulation runs in a separate process; the window stays responsive.", TEXT_DIM)
        self.screen.blit(sub, (cx - sub.get_width() // 2, cy + 92))

    def draw_error(self, lay: Dict[str, Any]) -> None:
        body = self.draw_panel(lay["map"], "Simulation process error", alt=True)
        y = body.y
        for line in (self.client.fatal or "").splitlines()[-30:]:
            self.blit_text("tiny", self.fit("tiny", line, body.width), (body.x, y), TEXT_DANGER)
            y += 16

    def draw_header(self, rect: pygame.Rect, frame: Dict[str, Any]) -> None:
        pygame.draw.rect(self.screen, (9, 13, 19), rect, border_radius=12)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=2, border_radius=12)
        left = rect.x + 400
        self.blit_text("ui", "Life Sim", (rect.x + 14, rect.y + 14))
        wd = WEEKDAYS[frame["weekday"]]
        self.blit_text("small", f"{wd} {frame['calendar_label']}  |  {frame['weather']}", (left, rect.y + 7))
        m = frame["metrics"]
        sub = (f"Pop {frame['population']:,}  |  Unemp {m.get('unemployment', 0) * 100:.1f}%  |  Happy {m.get('avg_happiness', 0):.2f}"
               f"  |  Stress {m.get('avg_stress', 0):.2f}  |  Infectious {int(m.get('infectious', 0))}  |  Traffic {m.get('traffic_speed', 1) * 100:.0f}%")
        self.blit_text("tiny", sub, (left, rect.y + 29), TEXT_DIM)

    def draw_footer(self, rect: pygame.Rect, frame: Dict[str, Any]) -> None:
        pygame.draw.rect(self.screen, (9, 13, 19), rect, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=10)
        status = frame.get("status") or self.status_message
        if self.client.status and self.client.status.startswith(("Saved", "Load")):
            status = self.client.status
        if frame["paused"]:
            status += "  [PAUSED]"
        self.blit_text("small", self.fit("small", status, rect.width // 2 - 20), (rect.x + 12, rect.y + 9))
        speed = 120.0 / max(0.03, frame.get("tick_duration", 1.2))
        info = (f"FPS {self.fps_display:4.0f}  draw {self.draw_ms:4.1f}ms  |  sim x{speed:.0f} ({frame.get('step_ms', 0):4.0f}ms/step)"
                f"  |  zoom {self.tile_size}px  heat:{HEATMAPS[self.heatmap]}  color:{COLOR_MODES[self.color_mode]}")
        surf = self.text("tiny", info, TEXT_DIM)
        self.screen.blit(surf, (rect.right - surf.get_width() - 12, rect.y + 11))

    def display_state(self) -> Tuple[Optional[Dict[str, Any]], float]:
        """Pick the frame to display and how far through its motion we are.

        Playback is delayed by ~1.6 frame intervals so the hourly decision step
        (and any pipe jitter) never shows up as stutter."""
        frames = self.client.timeline()
        if not frames:
            return None, 1.0
        last_t, last = frames[-1]
        if last.get("paused") or len(frames) < 2:
            return last, 1.0
        gaps = [frames[i + 1][0] - frames[i][0] for i in range(max(0, len(frames) - 8), len(frames) - 1)]
        gaps = [g for g in gaps if g < 2.5] or [0.2]
        interval = sorted(gaps)[len(gaps) // 2]
        self.delay += (min(2.5, max(0.04, interval * 1.7)) - self.delay) * 0.05
        target = time.monotonic() - self.delay
        if target >= last_t:
            return last, 1.0
        if target <= frames[0][0]:
            return frames[0][1], 0.0
        for k in range(len(frames) - 1):
            t0, t1 = frames[k][0], frames[k + 1][0]
            if t0 <= target < t1:
                return frames[k + 1][1], (target - t0) / max(1e-6, t1 - t0)
        return last, 1.0

    def draw_map(self, lay: Dict[str, Any], latest: Dict[str, Any], cur: Dict[str, Any], u: float) -> None:
        rect = lay["map"]
        pygame.draw.rect(self.screen, PANEL_BG_ALT, rect.inflate(8, 8), border_radius=10)
        x0, y0, tw, th = self.view_geometry(lay)
        ts = self.tile_size
        old_clip = self.screen.get_clip()
        self.screen.set_clip(rect)
        self.screen.fill((14, 19, 26), rect)
        self.wr.draw_ground(self.screen, rect, x0, y0, ts, tw, th)
        heat = self.heatmap_surface(cur)
        if heat is not None:
            ix0, iy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
            ix1 = min(self.static["width"], int(math.ceil(x0 + tw)))
            iy1 = min(self.static["height"], int(math.ceil(y0 + th)))
            if ix1 > ix0 and iy1 > iy0:
                sub = heat.subsurface(pygame.Rect(ix0, iy0, ix1 - ix0, iy1 - iy0))
                scaled = pygame.transform.scale(sub, ((ix1 - ix0) * ts, (iy1 - iy0) * ts))
                self.screen.blit(scaled, (rect.x + int(round((ix0 - x0) * ts)), rect.y + int(round((iy0 - y0) * ts))))
        dest = self.screen.subsurface(rect)
        sim_min = cur["day_index"] * 1440 + cur["hour"] * 60 + cur["minute"] - (0.0 if cur["paused"] else 2.0 * (1.0 - u))
        self.sim_minutes = sim_min
        self.wr.draw_metro(dest, x0, y0, ts, sim_min, rect.w, rect.h)
        agents = self.wr.agent_positions(cur, u)
        self.last_agents = (cur, agents)
        colors = self.person_color_table(cur)
        visible = self.wr.draw_agents(dest, cur, agents, x0, y0, tw, th, ts, colors, sim_min * 60.0, cur.get("selected_id"))
        self.wr.draw_signals(dest, x0, y0, tw, th, ts, sim_min * 60.0)
        self.draw_selection(rect, cur, agents, x0, y0)
        hour_f = (sim_min / 60.0) % 24.0
        mult = self.daylight_multiplier(hour_f)
        if mult != (255, 255, 255):
            self.screen.blit(self.tint_surface(rect.size, mult), rect.topleft, special_flags=pygame.BLEND_RGB_MULT)
        darkness = max(0.0, min(1.0, (1.0 - sum(mult) / 765.0) / 0.38))
        self.wr.draw_night(dest, x0, y0, tw, th, ts, darkness, hour_f, cur, agents)
        self.draw_weather(rect, cur)
        self.screen.set_clip(old_clip)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect.inflate(8, 8), width=2, border_radius=10)
        if self.show_minimap:
            self.draw_minimap(rect, cur, x0, y0, tw, th)
        self.draw_map_info(rect, cur, visible)
        self.hit_rects.append((rect, "map"))

    def draw_selection(self, rect: pygame.Rect, frame: Dict[str, Any], agents: Dict[str, Any], x0: float, y0: float) -> None:
        pid = frame.get("selected_id")
        if pid is None:
            return
        ts = self.tile_size
        route = frame.get("selected_route") or []
        if len(route) > 1 and ts >= 4:
            pts = [(rect.x + int((px - x0 + 0.5) * ts), rect.y + int((py - y0 + 0.5) * ts)) for px, py in route[::max(1, len(route) // 120)]]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (255, 234, 122), False, pts, 2)
        ids = frame["ids"]
        i = int(np.searchsorted(ids, pid))
        if i >= len(ids) or ids[i] != pid:
            return
        pulse = (math.sin(time.monotonic() * 5.0) + 1.0) * 0.5
        cx = rect.x + int((float(agents["x"][i]) - x0) * ts)
        cy = rect.y + int((float(agents["y"][i]) - y0) * ts)
        radius = max(6, int(ts * (0.5 + pulse * 0.2)))
        indoors = bool(frame["indoors"][i])
        pygame.draw.circle(self.screen, (255, 234, 122), (cx, cy), radius, 2)
        if indoors:
            self.blit_text("tiny", "indoors", (cx + radius + 4, cy - 7), (255, 234, 122))
        else:
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), max(2, radius // 4))

    def draw_minimap(self, rect: pygame.Rect, frame: Dict[str, Any], x0: float, y0: float, tw: float, th: float) -> None:
        W, H = self.static["width"], self.static["height"]
        scale = max(1, min(3, 220 // max(W, H // 1 or 1) + 1))
        scale = 2 if W * 2 <= 300 else 1
        if self.minimap_base is None or self.minimap_scale != scale:
            terr = self.static["terrain"]
            rgb = np.clip(self.terrain_lut[terr], 0, 255).astype(np.uint8)
            small = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))
            self.minimap_base = pygame.transform.scale(small, (W * scale, H * scale)).convert()
            self.minimap_scale = scale
        mm = self.minimap_base
        pos = (rect.x + 10, rect.bottom - mm.get_height() - 10)
        key = ("mm", frame["tick"], self.heatmap)
        dyn = self.overlay_cache.get("minimap_dyn")
        if dyn is None or dyn[0] != key:
            surf = mm.copy()
            arr = pygame.surfarray.pixels3d(surf)
            xs = frame["x"].astype(np.int32) * scale
            ys = frame["y"].astype(np.int32) * scale
            sel = frame["infection"] == 2
            arr[xs, ys] = (255, 255, 255)
            if sel.any():
                arr[xs[sel], ys[sel]] = (255, 60, 60)
            del arr
            self.overlay_cache["minimap_dyn"] = (key, surf)
            dyn = self.overlay_cache["minimap_dyn"]
        self.screen.blit(dyn[1], pos)
        pygame.draw.rect(self.screen, PANEL_BORDER, (pos[0] - 1, pos[1] - 1, mm.get_width() + 2, mm.get_height() + 2), 1)
        vr = pygame.Rect(pos[0] + int(max(0, x0) * scale), pos[1] + int(max(0, y0) * scale), int(min(tw, W) * scale), int(min(th, H) * scale))
        pygame.draw.rect(self.screen, (255, 234, 122), vr, 1)
        self.hit_rects.append((pygame.Rect(pos[0], pos[1], mm.get_width(), mm.get_height()), "minimap"))

    def draw_map_info(self, rect: pygame.Rect, frame: Dict[str, Any], visible: int) -> None:
        mouse = pygame.mouse.get_pos()
        if not rect.collidepoint(mouse) or not self.static:
            return
        tx, ty = self.screen_to_tile(mouse)
        W, H = self.static["width"], self.static["height"]
        if not (0 <= tx < W and 0 <= ty < H):
            return
        st = self.static
        terrain = TERRAIN_KEYS[int(st["terrain"][ty, tx])]
        bcode = int(st["building_type"][ty, tx])
        names = {v: k for k, v in BUILDING_CODES.items()}
        district = st["district_names"].get(int(st["district_map"][ty, tx]), "?")
        parts = [f"({tx},{ty}) {terrain}", f"{names[bcode]}" if bcode else "no building", district, f"{visible:,} citizens in view"]
        text = "  |  ".join(parts)
        surf = self.text("tiny", text, TEXT)
        box = pygame.Rect(rect.x + 10, rect.y + 8, surf.get_width() + 16, 24)
        s = pygame.Surface(box.size, pygame.SRCALPHA)
        s.fill((8, 12, 18, 190))
        self.screen.blit(s, box.topleft)
        self.screen.blit(surf, (box.x + 8, box.y + 4))

    def screen_to_tile(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        lay = self.layout()
        x0, y0, _, _ = self.view_geometry(lay)
        rect = lay["map"]
        return int(math.floor(x0 + (pos[0] - rect.x) / self.tile_size)), int(math.floor(y0 + (pos[1] - rect.y) / self.tile_size))

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------
    def draw_panel_tab(self, lay: Dict[str, Any], frame: Dict[str, Any]) -> None:
        tab = self.active_tab
        if not tab:
            return
        rect = lay["panel"]
        titles = {"building": "Building", "citizen": "Citizen", "metrics": "City Metrics", "charts": "Trends", "city": "City & Economy",
                  "health": "Health & Environment", "news": "News", "companies": "Companies",
                  "timeline": "Timeline", "relationships": "Relationships", "help": "Help"}
        body = self.draw_panel(rect, titles.get(tab, tab))
        if tab == "charts" or tab == "city":
            pass
        draw = getattr(self, f"tab_{tab}", None)
        old = self.screen.get_clip()
        self.screen.set_clip(body)
        if draw:
            draw(body, frame)
        self.screen.set_clip(old)
        close = pygame.Rect(rect.right - 40, rect.y + 8, 28, 24)
        pygame.draw.rect(self.screen, (33, 41, 54), close, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, close, width=1, border_radius=6)
        self.blit_text("small", "X", (close.x + 8, close.y + 2))
        self.hit_rects.insert(0, (close, "close"))
        self.hit_rects.insert(1, (rect, "panel"))

    def text_block(self, body: pygame.Rect, lines: List[str], line_height: int = 21, start: int = 0) -> None:
        y = body.y
        max_lines = max(1, body.height // line_height)
        for line in lines[start:start + max_lines]:
            self.blit_text("small", self.fit("small", line, body.width), (body.x, y), self.line_color(line))
            y += line_height

    def tab_citizen(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        self.text_block(body, frame["selected_lines"])

    def tab_building(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        b = frame.get("building")
        if not b:
            self.text_block(body, ["Shift+click a building on the map,", "or press B to inspect the building the", "selected citizen is standing in."])
            return
        self.blit_text("ui", self.fit("ui", b["name"], body.width), (body.x, body.y), ACCENT)
        y = body.y + 26
        for line in b["lines"]:
            self.blit_text("tiny", self.fit("tiny", line, body.width), (body.x, y), self.line_color(line) if line else TEXT)
            y += 16
        y += 6
        rows = list(reversed(b["rows"]))
        avail = body.bottom - y - 8
        row_h = max(9, min(24, avail // max(1, len(rows))))
        label_w = min(190, body.width // 2)
        bar_x = body.x + label_w + 4
        bar_w = max(40, body.width - label_w - 90)
        # roof
        pygame.draw.polygon(self.screen, PANEL_BORDER, [(bar_x, y), (bar_x + bar_w // 2, y - 7), (bar_x + bar_w, y)])
        for row in rows:
            cap = max(1, row["capacity"])
            rect = pygame.Rect(bar_x, y, bar_w, row_h - 1)
            pygame.draw.rect(self.screen, (28, 38, 52), rect)
            assigned = min(1.0, row["assigned"] / cap) if row["kind"] in ("work", "home") else 0.0
            if assigned:
                pygame.draw.rect(self.screen, (60, 110, 160), (rect.x, rect.y, int(rect.width * assigned), rect.height))
            present = min(1.0, row["present"] / cap)
            if present:
                gold = (255, 90, 90) if row["present"] > cap else (255, 200, 90)
                pygame.draw.rect(self.screen, gold, (rect.x, rect.y + rect.height // 3, int(rect.width * present), max(2, rect.height // 3)))
            pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1)
            if row_h >= 11:
                self.blit_text("tiny", self.fit("tiny", f"{row['floor']:>2} {row['label']}", label_w), (body.x, y - 1), TEXT_DIM)
                if row["kind"] in ("work", "home"):
                    tail = f"{row['present']}/{row['assigned']}"          # present now / assigned staff or residents
                else:
                    tail = f"{row['present']}/{cap}"                      # lobbies and civic floors: present / capacity
                over = row["present"] > cap
                self.blit_text("tiny", tail + (" !" if over else ""), (rect.right + 6, y - 1), TEXT_DANGER if over else TEXT)
            y += row_h
        pygame.draw.rect(self.screen, PANEL_BORDER, (bar_x - 4, y, bar_w + 8, 3))
        if row_h >= 11:
            self.blit_text("tiny", "blue=assigned  gold=present now (red = over capacity)  work/home floors: present/assigned, lobbies: present/capacity", (body.x, y + 6), TEXT_DIM)

    def tab_relationships(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        self.text_block(body, frame["relationship_lines"])

    def tab_news(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        self.text_block(body, frame["news_lines"], start=self.news_scroll)

    def tab_timeline(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        self.text_block(body, frame["timeline_lines"], start=self.timeline_scroll)

    def tab_companies(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        self.text_block(body, frame["company_lines"] + [""] + frame["market_lines"])

    def tab_help(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        self.text_block(body, [
            "Left click: select citizen / re-center on minimap",
            "Right/middle drag: pan   Wheel: zoom at cursor",
            "WASD / arrows: pan (hold)   [ ]: zoom",
            "Space: pause   + / -: sim speed",
            "H: cycle heat map (density, mood, traffic, ...)",
            "+ / -: sim speed (default is slow so you can watch)",
            "Zoom to 48-96 px to see walkers on the sidewalks",
            "V: cycle citizen color mode",
            "M: minimap   I: smooth motion",
            "Tab: next citizen   C: center on citizen",
            "Shift+click / B: inspect a building (floors, owner, staff)",
            "0: building inspector   Tall buildings = more floors",
            "1-9: panels   F5 save   F9 load   Q quit",
            "Night/day tint and weather are rendered live.",
        ])

    def tab_metrics(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        m = frame["metrics"]
        rows = [
            ("Population", f"{frame['population']:,}"), ("Median age", f"{m.get('median_age', 0):.0f}"),
            ("Seniors 65+", f"{m.get('pct_seniors', 0) * 100:.1f}%"), ("Dependency ratio", f"{m.get('dependency_ratio', 0):.2f}"),
            ("Unemployment", f"{m.get('unemployment', 0) * 100:.1f}%"), ("Labor participation", f"{m.get('labor_participation', 0) * 100:.0f}%"),
            ("Vacancies", f"{m.get('vacancies', 0):.0f}"), ("Avg salary", f"${m.get('avg_salary', 0):,.0f}"),
            ("Median net worth", f"${m.get('median_net_worth', 0):,.0f}"), ("Wealth Gini", f"{m.get('gini', 0):.3f}"),
            ("Top 10% share", f"{m.get('top10_share', 0) * 100:.0f}%"), ("Poverty rate", f"{m.get('poverty_rate', 0) * 100:.1f}%"),
            ("Avg debt", f"${m.get('avg_debt', 0):,.0f}"), ("Rent burden", f"{m.get('rent_burden', 0) * 100:.0f}% of income"),
            ("Housing occupancy", f"{m.get('housing_occupancy', 0) * 100:.0f}%"), ("Happiness", f"{m.get('avg_happiness', 0):.2f}"),
            ("Stress", f"{m.get('avg_stress', 0):.2f}"), ("High-stress share", f"{m.get('pct_high_stress', 0) * 100:.1f}%"),
            ("Health", f"{m.get('avg_health', 0):.2f}"), ("Crime / 1000 / day", f"{m.get('crime_per_1k', 0):.2f}"),
            ("Incarceration", f"{m.get('incarceration_rate', 0) * 100:.2f}%"), ("Open cases", f"{m.get('open_cases', 0):.0f}"),
            ("Avg commute", f"{m.get('avg_commute_min', 0):.0f} min"), ("Traffic speed", f"{m.get('traffic_speed', 1) * 100:.0f}%"),
            ("Air quality (AQI)", f"{m.get('aqi', 0):.0f}"), ("GDP / capita / day", f"${m.get('gdp_per_capita', 0):,.1f}"),
            ("Birth / death rate", f"{m.get('birth_rate', 0):.1f} / {m.get('death_rate', 0):.1f}"),
            ("College share", f"{m.get('college_share', 0) * 100:.0f}%"), ("Married share", f"{m.get('married_share', 0) * 100:.0f}%"),
            ("Companies", f"{m.get('companies', 0):.0f}"), ("Market cap", f"${fmt_num(m.get('market_cap', 0))}"),
            ("Treasury", f"${m.get('treasury', 0):,.0f}"), ("Inflation", f"{m.get('inflation', 0) * 100:.1f}%"),
            ("Property owners", f"{m.get('landlord_count', 0):.0f}"), ("Renter share", f"{m.get('renter_share', 0) * 100:.0f}%"),
            ("Owner-occupied", f"{m.get('owner_occupied_share', 0) * 100:.0f}%"), ("Avg rent", f"${m.get('avg_rent', 0):,.0f}"),
            ("Corporate landlords", f"{m.get('corporate_unit_share', 0) * 100:.0f}% of rentals"),
            ("Top-10 landlords", f"{m.get('top10_landlord_unit_share', 0) * 100:.0f}% of rentals"),
            ("Landlord income/mo", f"${m.get('landlord_income_month', 0):,.0f}"), ("Rent arrears (hh)", f"{m.get('arrears_households', 0):.0f}"),
            ("Floors avg / max", f"{m.get('avg_floors', 0):.1f} / {m.get('tallest_building', 0):.0f}"),
        ]
        col_w = body.width // 3
        y = body.y
        row_h = 36
        per_col = max(1, body.height // row_h)
        for i, (label, value) in enumerate(rows):
            col, row = divmod(i, per_col)
            if col > 2:
                break
            x = body.x + col * col_w
            yy = body.y + row * row_h
            self.blit_text("tiny", label, (x, yy), TEXT_DIM)
            self.blit_text("ui", self.fit("ui", value, col_w - 10), (x, yy + 14), TEXT)

    def sparkline(self, rect: pygame.Rect, series: List[float], label: str, color: Tuple[int, int, int], fmt: str = "{:.2f}") -> None:
        pygame.draw.rect(self.screen, (13, 18, 25), rect, border_radius=6)
        self.blit_text("tiny", label, (rect.x + 6, rect.y + 3), TEXT_DIM)
        if not series:
            return
        value = fmt.format(series[-1])
        surf = self.text("tiny", value, TEXT)
        self.screen.blit(surf, (rect.right - surf.get_width() - 6, rect.y + 3))
        data = series[-120:]
        lo, hi = min(data), max(data)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        plot = pygame.Rect(rect.x + 6, rect.y + 20, rect.width - 12, rect.height - 26)
        n = len(data)
        pts = [(plot.x + int(i / max(1, n - 1) * plot.width), plot.bottom - int((v - lo) / (hi - lo) * plot.height)) for i, v in enumerate(data)]
        if n > 1:
            pygame.draw.lines(self.screen, color, False, pts, 2)
        else:
            pygame.draw.circle(self.screen, color, pts[0], 2)

    def tab_charts(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        hist = self.daily.get("history", {}) if self.daily else {}
        charts = [
            ("population", "Population", ACCENT, "{:,.0f}"), ("unemployment", "Unemployment", TEXT_WARN, "{:.1%}"),
            ("avg_happiness", "Happiness", TEXT_GOOD, "{:.2f}"), ("avg_stress", "Stress", TEXT_DANGER, "{:.2f}"),
            ("gini", "Wealth Gini", ACCENT_2, "{:.3f}"), ("poverty_rate", "Poverty", TEXT_DANGER, "{:.1%}"),
            ("crime_per_1k", "Crime /1k", TEXT_DANGER, "{:.2f}"), ("infectious", "Infectious", (255, 90, 90), "{:.0f}"),
            ("traffic_speed", "Traffic speed", ACCENT, "{:.0%}"), ("aqi", "Air quality", ACCENT_3, "{:.0f}"),
            ("treasury", "Treasury", TEXT_GOOD, "${:,.0f}"), ("avg_salary", "Avg salary", ACCENT_2, "${:,.0f}"),
            ("rent_burden", "Rent burden", TEXT_WARN, "{:.0%}"), ("inflation", "Inflation", TEXT_WARN, "{:.1%}"),
            ("avg_health", "Health", TEXT_GOOD, "{:.2f}"), ("median_net_worth", "Median net worth", ACCENT, "${:,.0f}"),
        ]
        cols = 2
        cw = (body.width - 8) // cols
        ch = 64
        for i, (key, label, color, fmt) in enumerate(charts):
            col, row = i % cols, i // cols
            rect = pygame.Rect(body.x + col * (cw + 8), body.y + row * (ch + 6), cw, ch)
            if rect.bottom > body.bottom:
                break
            self.sparkline(rect, hist.get(key, []), label, color, fmt)

    def bars(self, rect: pygame.Rect, labels: List[str], values: List[float], color: Tuple[int, int, int], title: str) -> None:
        self.blit_text("tiny", title, (rect.x, rect.y), TEXT_DIM)
        top = rect.y + 18
        row_h = max(13, (rect.height - 18) // max(1, len(labels)))
        vmax = max(values) if values and max(values) > 0 else 1
        for i, (lab, val) in enumerate(zip(labels, values)):
            y = top + i * row_h
            if y + row_h > rect.bottom:
                break
            self.blit_text("tiny", lab[:11], (rect.x, y), TEXT)
            bw = int((rect.width - 150) * val / vmax)
            pygame.draw.rect(self.screen, color, (rect.x + 90, y + 2, max(1, bw), row_h - 4), border_radius=3)
            self.blit_text("tiny", f"{val:,.0f}", (rect.x + 94 + bw, y), TEXT_DIM)

    def tab_city(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        if not self.daily:
            return
        half = body.height // 2
        pyramid = self.daily["age_pyramid"]
        labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        totals = [a + b for a, b in zip(pyramid["men"], pyramid["women"])]
        self.bars(pygame.Rect(body.x, body.y, body.width, half // 2 + 30), labels, totals, ACCENT, "Age distribution")
        wlabels = ["<0", "0-500", "-1.5k", "-4k", "-10k", "-25k", "-60k", "-150k", "-400k", "400k+"]
        self.bars(pygame.Rect(body.x, body.y + half // 2 + 36, body.width, half // 2 + 20), wlabels, self.daily["wealth_hist"], ACCENT_2, "Net worth (adults)")
        sectors = self.daily["sector_jobs"]
        self.bars(pygame.Rect(body.x, body.y + half + 56, body.width, body.height - half - 56), list(sectors.keys()), list(sectors.values()), ACCENT_3, "Jobs by sector")

    def tab_health(self, body: pygame.Rect, frame: Dict[str, Any]) -> None:
        lines = list(frame["dynamics_lines"]) + [""] + list(frame["district_lines"])
        self.text_block(body, lines)
        hist = self.daily.get("history", {}) if self.daily else {}
        rect = pygame.Rect(body.x, body.bottom - 70, body.width, 66)
        self.sparkline(rect, hist.get("infectious", []), "Infectious over time", (255, 90, 90), "{:.0f}")

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def menu_layout(self, header: pygame.Rect) -> Tuple[List[Tuple[str, pygame.Rect]], Optional[pygame.Rect], List[Tuple[str, pygame.Rect, str]]]:
        x = header.x + 120
        buttons = []
        for label, _ in MENU_DEFS:
            rect = pygame.Rect(x, header.y + 12, 28 + len(label) * 9, 28)
            buttons.append((label, rect))
            x += rect.width + 6
        entries: List[Tuple[str, pygame.Rect, str]] = []
        menu_rect = None
        if self.active_menu:
            for label, rect in buttons:
                if label == self.active_menu:
                    items = dict(MENU_DEFS)[label]
                    menu_rect = pygame.Rect(rect.x, rect.bottom + 4, 210, len(items) * 28 + 8)
                    for i, (name, action) in enumerate(items):
                        entries.append((name, pygame.Rect(menu_rect.x + 4, menu_rect.y + 4 + i * 28, menu_rect.width - 8, 24), action))
        return buttons, menu_rect, entries

    def draw_menu_bar(self, header: pygame.Rect) -> None:
        buttons, menu_rect, entries = self.menu_layout(header)
        # Menu buttons sit inside the header, positioned right of the title.
        for label, rect in buttons:
            active = self.active_menu == label
            pygame.draw.rect(self.screen, (24, 34, 46) if active else (16, 22, 30), rect, border_radius=8)
            pygame.draw.rect(self.screen, ACCENT if active else PANEL_BORDER, rect, width=1, border_radius=8)
            self.blit_text("small", label, (rect.x + 14, rect.y + 5))
        if menu_rect:
            pygame.draw.rect(self.screen, (14, 19, 26), menu_rect, border_radius=10)
            pygame.draw.rect(self.screen, PANEL_BORDER, menu_rect, width=1, border_radius=10)
            for name, rect, _ in entries:
                pygame.draw.rect(self.screen, (22, 30, 40), rect, border_radius=6)
                self.blit_text("small", name, (rect.x + 10, rect.y + 3))

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def zoom(self, delta: int, anchor: Optional[Tuple[int, int]] = None) -> None:
        if not self.static:
            return
        idx = min(range(len(ZOOM_STEPS)), key=lambda i: abs(ZOOM_STEPS[i] - self.tile_size))
        new = ZOOM_STEPS[max(0, min(len(ZOOM_STEPS) - 1, idx + delta))]
        if new == self.tile_size:
            return
        lay = self.layout()
        if anchor is not None and lay["map"].collidepoint(anchor):
            x0, y0, _, _ = self.view_geometry(lay)
            wx = x0 + (anchor[0] - lay["map"].x) / self.tile_size
            wy = y0 + (anchor[1] - lay["map"].y) / self.tile_size
            self.tile_size = new
            nx = (anchor[0] - lay["map"].x) / new
            ny = (anchor[1] - lay["map"].y) / new
            self.cam_x = wx - nx + lay["map"].width / new / 2.0
            self.cam_y = wy - ny + lay["map"].height / new / 2.0
        else:
            self.tile_size = new

    def execute(self, action: str) -> None:
        kind, _, arg = action.partition(":")
        if kind == "tab":
            self.active_tab = None if arg == "none" else arg
        elif action == "map:zoom_in":
            self.zoom(1)
        elif action == "map:zoom_out":
            self.zoom(-1)
        elif action == "map:center":
            self.center_selected()
        elif action == "map:heat":
            self.heatmap = (self.heatmap + 1) % len(HEATMAPS)
        elif action == "map:color":
            self.color_mode = (self.color_mode + 1) % len(COLOR_MODES)
            self.person_arrays_key = -1
        elif action == "map:minimap":
            self.show_minimap = not self.show_minimap
        elif action == "map:smooth":
            self.smooth = not self.smooth
        elif action == "sim:pause":
            self.client.send("pause")
        elif action == "sim:faster":
            self.client.send("speed", True)
        elif action == "sim:slower":
            self.client.send("speed", False)
        elif action == "sim:save":
            self.client.send("save")
        elif action == "sim:load":
            self.client.send("load")

    def center_selected(self) -> None:
        frame, _, _, _ = self.client.snapshot()
        if frame and frame.get("selected_location"):
            self.cam_x, self.cam_y = frame["selected_location"]

    def pick_person(self, pos: Tuple[int, int]) -> None:
        lay = self.layout()
        x0, y0, _, _ = self.view_geometry(lay)
        rect = lay["map"]
        wx = x0 + (pos[0] - rect.x) / self.tile_size
        wy = y0 + (pos[1] - rect.y) / self.tile_size
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        if shift or self.last_agents is None:
            self.inspect_tile(int(wx), int(wy))
            return
        frame, agents = self.last_agents
        d = (agents["x"] - wx) ** 2 + (agents["y"] - wy) ** 2
        d = np.where(frame["indoors"] == 0, d, 1e9)
        i = int(np.argmin(d))
        if d[i] <= max(0.45, (16.0 / self.tile_size) ** 2):
            pid = int(frame["ids"][i])
            self.client.send("select", pid)
            self.active_tab = "citizen"
            self.status_message = f"Selected citizen {pid} at {int(frame['x'][i])},{int(frame['y'][i])}"
        else:
            self.inspect_tile(int(wx), int(wy))

    def inspect_tile(self, tx: int, ty: int) -> None:
        W, H = self.static["width"], self.static["height"]
        if 0 <= tx < W and 0 <= ty < H:
            bid = int(self.static["building_id"][ty, tx])
            if bid:
                self.client.send("inspect", bid)
                self.active_tab = "building"
                self.status_message = f"Inspecting building {bid} at {tx},{ty}"
                return
            self.status_message = f"Tile {tx},{ty} (no building; Shift+click a building to inspect)"

    def handle_click(self, pos: Tuple[int, int]) -> None:
        buttons, menu_rect, entries = self.menu_layout(self.layout()["header"])
        for label, rect in buttons:
            if rect.collidepoint(pos):
                self.active_menu = None if self.active_menu == label else label
                return
        if menu_rect and menu_rect.collidepoint(pos):
            for _, rect, action in entries:
                if rect.collidepoint(pos):
                    self.execute(action)
            self.active_menu = None
            return
        self.active_menu = None
        for rect, tag in self.hit_rects:
            if rect.collidepoint(pos):
                if tag == "close":
                    self.active_tab = None
                    return
                if tag == "panel":
                    return
                if tag == "minimap":
                    scale = self.minimap_scale
                    self.cam_x = (pos[0] - rect.x) / scale
                    self.cam_y = (pos[1] - rect.y) / scale
                    return
                if tag == "map":
                    self.pick_person(pos)
                    return

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                self.handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_click(event.pos)
                elif event.button in (2, 3):
                    self.drag_pos = event.pos
                elif event.button == 4:
                    self.zoom(1, event.pos)
                elif event.button == 5:
                    self.zoom(-1, event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
                self.drag_pos = None
            elif event.type == pygame.MOUSEMOTION and self.drag_pos is not None:
                dx = event.pos[0] - self.drag_pos[0]
                dy = event.pos[1] - self.drag_pos[1]
                self.cam_x -= dx / self.tile_size
                self.cam_y -= dy / self.tile_size
                self.drag_pos = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                pass

    def handle_key(self, key: int) -> None:
        tabs = {pygame.K_1: "citizen", pygame.K_2: "metrics", pygame.K_3: "charts", pygame.K_4: "city",
                pygame.K_5: "health", pygame.K_6: "news", pygame.K_7: "companies", pygame.K_8: "timeline", pygame.K_9: "relationships"}
        if key in (pygame.K_q, pygame.K_ESCAPE):
            self.running = False
        elif key == pygame.K_SPACE:
            self.client.send("pause")
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.client.send("speed", True)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.client.send("speed", False)
        elif key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
            self.zoom(-1)
        elif key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
            self.zoom(1)
        elif key == pygame.K_c:
            self.center_selected()
        elif key == pygame.K_b:
            frame, _, _, _ = self.client.snapshot()
            if frame and frame.get("selected_location"):
                self.inspect_tile(*frame["selected_location"])
        elif key == pygame.K_0:
            self.active_tab = "building"
        elif key == pygame.K_TAB:
            self.client.send("cycle")
            self.active_tab = "citizen"
        elif key == pygame.K_h:
            self.execute("map:heat")
        elif key == pygame.K_v:
            self.execute("map:color")
        elif key == pygame.K_m:
            self.show_minimap = not self.show_minimap
        elif key == pygame.K_i:
            self.smooth = not self.smooth
        elif key == pygame.K_F1:
            self.active_tab = None if self.active_tab == "help" else "help"
        elif key == pygame.K_F5:
            self.client.send("save")
        elif key == pygame.K_F9:
            self.client.send("load")
        elif key == pygame.K_PAGEDOWN:
            self.news_scroll += 1
            self.timeline_scroll += 1
        elif key == pygame.K_PAGEUP:
            self.news_scroll = max(0, self.news_scroll - 1)
            self.timeline_scroll = max(0, self.timeline_scroll - 1)
        elif key in tabs:
            self.active_tab = tabs[key]

    def held_keys(self, dt: float) -> None:
        if not self.static:
            return
        keys = pygame.key.get_pressed()
        speed = 28.0 * dt * (14.0 / self.tile_size) ** 0.5 * (2.5 if keys[pygame.K_LSHIFT] else 1.0)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.cam_x -= speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.cam_x += speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.cam_y -= speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.cam_y += speed

    # ------------------------------------------------------------------
    def run(self) -> None:
        last = time.perf_counter()
        frames = 0
        bench: List[float] = []
        try:
            while self.running:
                now = time.perf_counter()
                dt = now - last
                last = now
                self.frame_times.append(dt)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                self.fps_display = len(self.frame_times) / max(1e-6, sum(self.frame_times))
                self.handle_events()
                self.held_keys(dt)
                if self.bench_frames:
                    self.scripted_bench_step(frames)
                self.draw()
                if self.bench_frames:
                    if self.static and self.client.frame is not None:
                        frames += 1
                        bench.append(self.draw_ms)
                        if self.shot_prefix and frames in (40, 100, 170, 230):
                            pygame.image.save(self.screen, f"{self.shot_prefix}{frames}.png")
                    if frames >= self.bench_frames + 60 or self.client.fatal:
                        if self.client.fatal:
                            print(self.client.fatal)
                        break
                else:
                    frames += 1
                self.clock.tick(TARGET_FPS if not self.bench_frames else 0)
        finally:
            if self.bench_frames and bench:
                data = sorted(bench[60:] or bench)
                avg = sum(data) / len(data)
                frame = self.client.frame or {}
                print(f"BENCH population={frame.get('population')} world={self.static.get('width')}x{self.static.get('height')}"
                      f" tile={self.tile_size}px frames={len(data)}")
                print(f"  draw ms: avg={avg:.2f} p50={data[len(data) // 2]:.2f} p95={data[int(len(data) * 0.95)]:.2f} max={data[-1]:.2f}")
                print(f"  => {1000.0 / avg:.0f} fps potential; sim frames received={self.client.frames_received}; "
                      f"sim step={frame.get('step_ms', 0):.1f}ms")
            self.client.close()
            pygame.quit()

    def scripted_bench_step(self, frame_no: int) -> None:
        """Pan and zoom while benchmarking so every code path is exercised."""
        if not self.static:
            return
        if frame_no == 40:
            self.heatmap = 1
        if frame_no == 80:
            self.heatmap = 0
            self.active_tab = "charts"
        if frame_no == 120:
            self.active_tab = "metrics"
        if frame_no in (100, 140, 180, 220, 260, 300):
            self.zoom(1 if frame_no < 220 else -1)
        if frame_no == 30:
            self.cam_x, self.cam_y = self.static["width"] / 2.0, self.static["height"] / 2.0
        self.cam_x += math.sin(frame_no * 0.05) * 0.4
        self.cam_y += math.cos(frame_no * 0.04) * 0.2



def main() -> None:
    parser = argparse.ArgumentParser(description="Life Sim visual frontend")
    parser.add_argument("--preset", default="large", choices=["large", "small", "classic"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--world-seed", type=int, default=None)
    parser.add_argument("--load", default=None, help="load a saved simulation file")
    parser.add_argument("--tick-ms", type=float, default=None,
                        help="wall-clock milliseconds per 2 simulated minutes (default 1200; lower = faster)")
    parser.add_argument("--bench-frames", type=int, default=0, help="headless render benchmark")
    parser.add_argument("--shot", default="", help="with --bench-frames: save screenshots with this path prefix")
    args = parser.parse_args()
    config = {
        "preset": args.preset, "seed": args.seed, "world_seed": args.world_seed,
        "load_path": args.load,
        "tick_duration": args.tick_ms / 1000.0 if args.tick_ms else None,
    }
    pygame.init()
    pygame.font.init()
    client = SimClient(config)
    Frontend(client, bench_frames=args.bench_frames, shot_prefix=args.shot).run()


if __name__ == "__main__":
    main()
