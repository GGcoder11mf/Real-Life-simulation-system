"""
life.py

Deep per-person simulation layered on top of the citizen objects.

    BodyStore   struct-of-arrays anatomy for every citizen: 16 body parts
                (bones, organs, skin, brain), a 10-locus diploid genome,
                16 conditions (genetic, acquired, mental).  Daily ageing,
                healing and disease progression are numpy-vectorised, so
                the cost does not scale with Python object count.
    Hospitals   beds, staff-driven care quality, triage by acuity, billing.
    LifeSystem  conversations, friendship, romance, marriage, pregnancy and
                birth (with genetic inheritance), bereavement, schools
                (grades, teachers, bullying) and the injury API used by the
                crime, workplace and traffic systems.

Everything here is deterministic given the engine seed and is saved/loaded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from world import WorldMap, clamp

# ----------------------------------------------------------------------
# Anatomy
# ----------------------------------------------------------------------
PARTS = ("brain", "eyes", "ears", "heart", "lungs", "liver", "kidneys", "stomach",
         "skin", "skull", "spine", "ribs", "l_arm", "r_arm", "l_leg", "r_leg")
PART_INDEX = {name: i for i, name in enumerate(PARTS)}
NP = len(PARTS)
BONES = [PART_INDEX[n] for n in ("skull", "spine", "ribs", "l_arm", "r_arm", "l_leg", "r_leg")]
VITAL = [PART_INDEX[n] for n in ("brain", "heart", "lungs", "liver", "kidneys")]
# base daily healing rate per part (fraction of the missing integrity)
HEAL_RATE = np.array([0.0015, 0.001, 0.001, 0.003, 0.004, 0.012, 0.004, 0.01,
                      0.10, 0.012, 0.012, 0.014, 0.016, 0.016, 0.016, 0.016], dtype=np.float32)
# how strongly permanent scarring lowers the healing ceiling per unit of damage
SCAR_FACTOR = np.array([0.20, 0.15, 0.15, 0.12, 0.12, 0.08, 0.12, 0.05,
                        0.06, 0.03, 0.05, 0.02, 0.02, 0.02, 0.03, 0.03], dtype=np.float32)

# injury patterns: cause -> (part index list, weights)
def _pattern(**kw: float) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.array([PART_INDEX[k] for k in kw], dtype=np.int32)
    w = np.array(list(kw.values()), dtype=np.float64)
    return idx, w / w.sum()


INJURY_PATTERNS = {
    "assault": _pattern(skull=2, brain=1, ribs=3, l_arm=2, r_arm=2, skin=4, eyes=1, stomach=1),
    "weapon": _pattern(skin=4, ribs=2, heart=1, lungs=2, liver=1, l_arm=2, r_arm=2, stomach=1),
    "accident": _pattern(l_leg=3, r_leg=3, spine=2, ribs=2, skull=1, brain=1, skin=3, l_arm=1, r_arm=1),
    "workplace": _pattern(l_arm=3, r_arm=3, skin=3, eyes=1, lungs=2, spine=2, l_leg=1, r_leg=1),
    "fall": _pattern(l_leg=2, r_leg=2, spine=2, skull=1, l_arm=2, r_arm=2, ribs=1),
    "duty": _pattern(skin=3, ribs=2, l_arm=2, r_arm=2, skull=1, l_leg=1, r_leg=1),
    "bullying": _pattern(skin=3, ribs=1, l_arm=1, r_arm=1, eyes=1),
    "fire": _pattern(skin=5, lungs=3, eyes=1, l_arm=1, r_arm=1),
}

# ----------------------------------------------------------------------
# Conditions (genetic, acquired, mental)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CondDef:
    name: str
    locus: int          # genome locus (-1 = acquired)
    mode: str           # rec | dom | add | x | acq
    onset_age: float
    onset_rate: float   # daily chance once eligible (scaled by allele count for "add")
    progress: float     # daily severity gain when untreated
    part: int           # part damaged
    damage: float       # integrity damage per day per unit severity
    lethal: float       # daily death hazard at severity 1.0
    mental: bool = False


CONDS: Tuple[CondDef, ...] = (
    CondDef("cystic fibrosis", 0, "rec", 0, 1.0, 0.0006, PART_INDEX["lungs"], 0.0009, 0.0010),
    CondDef("sickle cell disease", 1, "rec", 1, 1.0, 0.0004, PART_INDEX["kidneys"], 0.0006, 0.0006),
    CondDef("huntington disease", 2, "dom", 42, 0.004, 0.0010, PART_INDEX["brain"], 0.0010, 0.0010),
    CondDef("type 1 diabetes", 3, "add", 6, 0.00012, 0.0004, PART_INDEX["kidneys"], 0.0005, 0.0005),
    CondDef("hypertension", 4, "add", 34, 0.00028, 0.0004, PART_INDEX["heart"], 0.0004, 0.0006),
    CondDef("asthma", 5, "add", 3, 0.00018, 0.0002, PART_INDEX["lungs"], 0.0002, 0.0003),
    CondDef("hemophilia", 6, "x", 0, 1.0, 0.0, PART_INDEX["skin"], 0.0, 0.0),
    CondDef("colon cancer", 7, "add", 45, 0.00007, 0.0050, PART_INDEX["stomach"], 0.0020, 0.0040),
    CondDef("breast cancer", 8, "add", 40, 0.00006, 0.0050, PART_INDEX["ribs"], 0.0010, 0.0035),
    CondDef("myopia", 9, "add", 6, 0.0003, 0.0001, PART_INDEX["eyes"], 0.0001, 0.0),
    # acquired / mental (locus -1)
    CondDef("fracture", -1, "acq", 0, 0.0, 0.0, PART_INDEX["l_arm"], 0.0, 0.0),
    CondDef("concussion", -1, "acq", 0, 0.0, 0.0, PART_INDEX["brain"], 0.0, 0.0),
    CondDef("chronic pain", -1, "acq", 0, 0.0, 0.0, PART_INDEX["spine"], 0.0, 0.0),
    CondDef("depression", 9, "acq", 14, 0.0, 0.0, PART_INDEX["brain"], 0.0, 0.0, True),
    CondDef("anxiety disorder", -1, "acq", 8, 0.0, 0.0, PART_INDEX["brain"], 0.0, 0.0, True),
    CondDef("PTSD", -1, "acq", 0, 0.0, 0.0, PART_INDEX["brain"], 0.0, 0.0, True),
)
NC = len(CONDS)
C = {c.name: i for i, c in enumerate(CONDS)}
NL = 10
# disease allele frequencies per locus
ALLELE_FREQ = np.array([0.035, 0.05, 0.003, 0.10, 0.18, 0.15, 0.02, 0.07, 0.07, 0.30], dtype=np.float64)
GENETIC_IDX = [i for i, c in enumerate(CONDS) if c.mode in ("rec", "dom", "add", "x")]
ADD_RISK = np.array([0.0, 1.0, 2.6], dtype=np.float32)

MENTAL_IDX = [C["depression"], C["anxiety disorder"], C["PTSD"]]
FRACTURE, CONCUSSION, CHRONIC_PAIN = C["fracture"], C["concussion"], C["chronic pain"]
HEMOPHILIA = C["hemophilia"]


class BodyStore:
    """SoA anatomy + genome + conditions, indexed through a pid -> slot map."""

    def __init__(self, capacity: int = 1024) -> None:
        self.cap = 0
        self.slot: Dict[int, int] = {}
        self.free: List[int] = []
        self.pid = np.zeros(0, dtype=np.int32)
        self.alive = np.zeros(0, dtype=bool)
        self.male = np.zeros(0, dtype=bool)
        self.parts = np.ones((0, NP), dtype=np.float32)
        self.ceiling = np.ones((0, NP), dtype=np.float32)
        self.geno = np.zeros((0, NL), dtype=np.uint8)
        self.cond = np.zeros((0, NC), dtype=np.float32)
        self.pain = np.zeros(0, dtype=np.float32)
        self.trauma = np.zeros(0, dtype=np.float32)
        self.preg = np.zeros(0, dtype=np.int16)          # days pregnant (0 = not)
        self.preg_father = np.zeros(0, dtype=np.int32)
        self.care = np.ones(0, dtype=np.float32)         # medical care multiplier
        self.treated = np.zeros(0, dtype=np.float32)     # share of disease progression suppressed
        self.scars = np.zeros(0, dtype=np.uint8)
        self.nutrition = np.zeros(0, dtype=np.float32)
        self.grow(capacity)

    # -- storage ---------------------------------------------------------
    def grow(self, new_cap: int) -> None:
        old = self.cap
        if new_cap <= old:
            return
        def ext(arr: np.ndarray, fill: Any = 0) -> np.ndarray:
            shape = (new_cap,) + arr.shape[1:]
            out = np.full(shape, fill, dtype=arr.dtype)
            out[:old] = arr
            return out
        self.pid = ext(self.pid, -1)
        self.alive = ext(self.alive)
        self.male = ext(self.male)
        self.parts = ext(self.parts, 1.0)
        self.ceiling = ext(self.ceiling, 1.0)
        self.geno = ext(self.geno)
        self.cond = ext(self.cond)
        self.pain = ext(self.pain)
        self.trauma = ext(self.trauma)
        self.preg = ext(self.preg)
        self.preg_father = ext(self.preg_father)
        self.care = ext(self.care, 1.0)
        self.treated = ext(self.treated)
        self.scars = ext(self.scars)
        self.nutrition = ext(self.nutrition, 0.8)
        self.free.extend(range(new_cap - 1, old - 1, -1))
        self.cap = new_cap

    def alloc(self, pid: int) -> int:
        if not self.free:
            self.grow(max(1024, self.cap * 2))
        s = self.free.pop()
        self.slot[pid] = s
        self.pid[s] = pid
        self.alive[s] = True
        self.parts[s] = 1.0
        self.ceiling[s] = 1.0
        self.cond[s] = 0.0
        self.pain[s] = 0.0
        self.trauma[s] = 0.0
        self.preg[s] = 0
        self.preg_father[s] = 0
        self.care[s] = 1.0
        self.treated[s] = 0.0
        self.scars[s] = 0
        self.nutrition[s] = 0.8
        return s

    def release(self, pid: int) -> None:
        s = self.slot.pop(pid, None)
        if s is None:
            return
        self.alive[s] = False
        self.pid[s] = -1
        self.free.append(s)

    def live_slots(self) -> np.ndarray:
        return np.nonzero(self.alive)[0]

    # -- genetics --------------------------------------------------------
    @staticmethod
    def random_genome(rng: np.random.Generator) -> np.ndarray:
        a = (rng.random(NL) < ALLELE_FREQ).astype(np.uint8)
        b = (rng.random(NL) < ALLELE_FREQ).astype(np.uint8)
        return a | (b << 1)

    @staticmethod
    def inherit(mother: np.ndarray, father: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Each parent passes one of its two alleles per locus; rare mutation."""
        pick_m = rng.integers(0, 2, NL).astype(np.uint8)
        pick_f = rng.integers(0, 2, NL).astype(np.uint8)
        a = (mother >> pick_m) & 1
        b = (father >> pick_f) & 1
        mut = rng.random(NL) < 0.002
        a = np.where(mut, 1 - a, a).astype(np.uint8)
        return a | (b << 1)

    def alleles(self, slots: np.ndarray) -> np.ndarray:
        g = self.geno[slots]
        return (g & 1) + (g >> 1)

    # -- injuries ---------------------------------------------------------
    def injure(self, slot: int, cause: str, severity: float, rng: np.random.Generator) -> Tuple[List[str], float]:
        """Apply an injury; returns (description fragments, total damage)."""
        idx, w = INJURY_PATTERNS.get(cause, INJURY_PATTERNS["accident"])
        severity = float(max(0.02, min(1.0, severity)))
        n_hit = 1 + int(severity > 0.35) + int(severity > 0.7)
        hits = rng.choice(idx, size=n_hit, replace=False, p=w) if n_hit < len(idx) else idx
        bleed = 1.6 if self.male[slot] and self.geno[slot, 6] and self._expressed_hemophilia(slot) else 1.0
        total = 0.0
        notes: List[str] = []
        for part in hits:
            d = float(min(0.95, severity * rng.uniform(0.5, 1.0) * bleed))
            self.parts[slot, part] = max(0.0, self.parts[slot, part] - d)
            self.ceiling[slot, part] = max(0.25, self.ceiling[slot, part] - d * SCAR_FACTOR[part])
            total += d
            name = PARTS[part]
            if part in BONES and d > 0.3:
                self.cond[slot, FRACTURE] = max(self.cond[slot, FRACTURE], d)
                notes.append(f"fractured {name.replace('_', ' ')}")
            elif part == PART_INDEX["skull"] or (part == PART_INDEX["brain"] and d > 0.2):
                self.cond[slot, CONCUSSION] = max(self.cond[slot, CONCUSSION], d)
                notes.append("suffered a concussion")
            elif part == PART_INDEX["skin"] and d > 0.25:
                self.scars[slot] = min(255, self.scars[slot] + 1)
                notes.append("was left with a scar")
            elif d > 0.3:
                notes.append(f"damaged {name.replace('_', ' ')}")
        self.pain[slot] = min(1.0, self.pain[slot] + total * 0.5)
        if total > 0.7:
            self.cond[slot, CHRONIC_PAIN] = max(self.cond[slot, CHRONIC_PAIN], min(1.0, total * 0.4))
        return notes, total

    def _expressed_hemophilia(self, slot: int) -> bool:
        return self.cond[slot, HEMOPHILIA] > 0

    def add_trauma(self, slot: int, amount: float) -> None:
        self.trauma[slot] = min(1.0, self.trauma[slot] + amount)

    # -- aggregate views ---------------------------------------------------
    def vitality(self, slots: np.ndarray) -> np.ndarray:
        v = self.parts[slots][:, VITAL]
        return 0.5 * v.min(axis=1) + 0.5 * v.mean(axis=1)

    def mobility(self, slots: np.ndarray) -> np.ndarray:
        p = self.parts[slots]
        legs = (p[:, PART_INDEX["l_leg"]] + p[:, PART_INDEX["r_leg"]]) * 0.5
        return np.minimum(legs, 0.4 + 0.6 * p[:, PART_INDEX["spine"]])

    def acuity(self, slots: np.ndarray) -> np.ndarray:
        """0..1 need for hospital care."""
        p = self.parts[slots]
        organ = 1.0 - p[:, VITAL].min(axis=1)
        sev = (self.cond[slots][:, GENETIC_IDX] * np.array([CONDS[i].lethal * 600 for i in GENETIC_IDX], dtype=np.float32)).max(axis=1)
        frac = self.cond[slots, FRACTURE] * 0.8
        conc = self.cond[slots, CONCUSSION] * 0.9
        a = np.maximum.reduce([organ * 1.1, np.minimum(sev, 1.0), frac, conc])
        return np.clip(a + (self.preg[slots] >= 266) * 0.9, 0.0, 1.0)

    # -- daily vectorised update ----------------------------------------
    def tick(self, ages: np.ndarray, slots: np.ndarray, health: np.ndarray, stress: np.ndarray,
             rng: np.random.Generator) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
        """Heal, age, progress disease.  Returns (new-condition slots, [(slot, death cause)])."""
        n = slots.shape[0]
        if n == 0:
            return slots, []
        parts = self.parts[slots]
        ceil = self.ceiling[slots]
        care = self.care[slots][:, None]
        vigor = (0.35 + 0.65 * health)[:, None] * np.clip(1.15 - ages / 140.0, 0.4, 1.1)[:, None]
        heal = HEAL_RATE[None, :] * vigor * care
        parts = np.minimum(ceil, parts + (ceil - parts) * np.minimum(heal, 0.9))
        # ageing wear on organs / bones past 50
        wear = np.clip((ages - 50.0) / 1.2e6, 0.0, 5e-5).astype(np.float32)
        parts[:, VITAL] -= wear[:, None] * (0.5 + 0.5 * (1.0 - health))[:, None]
        parts[:, BONES] -= (wear * 0.6)[:, None]
        # conditions ---------------------------------------------------
        cond = self.cond[slots]
        geno = self.geno[slots]
        cnt = ((geno & 1) + (geno >> 1)).astype(np.int8)
        male = self.male[slots]
        newly = np.zeros(n, dtype=bool)
        treat = (1.0 - 0.85 * self.treated[slots])
        for i in GENETIC_IDX:
            c = CONDS[i]
            n_al = cnt[:, c.locus]
            if c.mode == "rec":
                elig = n_al == 2
                rate = np.where(elig, c.onset_rate, 0.0)
            elif c.mode == "dom":
                elig = n_al >= 1
                rate = np.where(elig, c.onset_rate, 0.0)
            elif c.mode == "x":
                elig = np.where(male, n_al >= 1, n_al == 2)
                rate = np.where(elig, 1.0, 0.0)
            else:
                rate = ADD_RISK[n_al] * c.onset_rate
            ok = (ages >= c.onset_age) & (cond[:, i] <= 0.0) & (rate > 0)
            if c.name in ("hypertension", "colon cancer", "breast cancer"):
                ok &= ages < 110
            if c.name == "breast cancer":
                ok &= ~male | (rng.random(n) < 0.01)
            roll = rng.random(n)
            hit = ok & (roll < rate * (1.0 + 0.6 * np.clip(stress - 0.5, 0, 1)))
            if hit.any():
                cond[hit, i] = np.maximum(cond[hit, i], 0.05)
                newly |= hit
            active = cond[:, i] > 0.0
            if c.progress > 0 and active.any():
                cond[:, i] = np.where(active, np.minimum(1.0, cond[:, i] + c.progress * treat * (1.0 + (ages > 60) * 0.5)), cond[:, i])
            if c.damage > 0 and active.any():
                d = np.where(active, cond[:, i] * c.damage * treat, 0.0).astype(np.float32)
                parts[:, c.part] -= d
                ceil[:, c.part] = np.maximum(0.2, ceil[:, c.part] - d * 0.4)
        # acquired conditions decay when healed
        fr = cond[:, FRACTURE]
        limb_ok = parts[:, BONES].min(axis=1)
        cond[:, FRACTURE] = np.where(fr > 0, np.where(limb_ok > 0.92, 0.0, np.maximum(0.0, fr - 0.01 * care[:, 0])), 0.0)
        conc = cond[:, CONCUSSION]
        cond[:, CONCUSSION] = np.where(conc > 0, np.maximum(0.0, conc - 0.03 * care[:, 0]), 0.0)
        pain = self.pain[slots]
        pain = np.clip(pain * 0.93 + cond[:, CHRONIC_PAIN] * 0.04 + (1.0 - parts.min(axis=1)) * 0.01, 0.0, 1.0)
        # mental conditions follow stress, trauma and genetics
        trauma = self.trauma[slots] * 0.995
        dep_pred = (cnt[:, CONDS[C["depression"]].locus] * 0.5 + 0.2)
        stress_hi = np.clip(stress - 0.55, 0.0, 1.0)
        dep_p = (0.0004 + stress_hi * 0.004 + trauma * 0.003) * dep_pred
        anx_p = 0.0003 + stress_hi * 0.003 + trauma * 0.002
        ptsd_p = np.where(trauma > 0.45, 0.01 * trauma, 0.0)
        for idx, p in ((C["depression"], dep_p), (C["anxiety disorder"], anx_p), (C["PTSD"], ptsd_p)):
            hit = (cond[:, idx] <= 0.0) & (ages >= 8) & (rng.random(n) < p)
            if hit.any():
                cond[hit, idx] = 0.2 + 0.4 * rng.random(int(hit.sum()))
                newly |= hit
            active = cond[:, idx] > 0.0
            relief = 0.004 + 0.012 * self.treated[slots]
            drift = np.where(stress > 0.5, 0.004, -relief)
            cond[:, idx] = np.where(active, np.clip(cond[:, idx] + drift, 0.0, 1.0), 0.0)
            cond[:, idx] = np.where(active & (cond[:, idx] < 0.02), 0.0, cond[:, idx])
        parts = np.clip(parts, 0.0, 1.0)
        self.parts[slots] = parts
        self.ceiling[slots] = ceil
        self.cond[slots] = cond
        self.pain[slots] = pain
        self.trauma[slots] = trauma
        # death -------------------------------------------------------------
        organ_min = parts[:, VITAL].min(axis=1)
        hazard = np.zeros(n, dtype=np.float32)
        for i in GENETIC_IDX:
            if CONDS[i].lethal > 0:
                hazard += cond[:, i] ** 2 * CONDS[i].lethal * treat * (1.0 + (ages > 70) * 0.5)
        hazard += np.where(organ_min < 0.10, 0.2, 0.0) + np.where(organ_min < 0.04, 0.6, 0.0)
        hazard += np.where(parts[:, PART_INDEX["brain"]] < 0.08, 0.5, 0.0)
        hazard /= np.maximum(0.6, self.care[slots])
        dies = rng.random(n) < np.minimum(hazard, 0.9)
        deaths: List[Tuple[int, str]] = []
        if dies.any():
            for j in np.nonzero(dies)[0]:
                deaths.append((int(slots[j]), self._death_cause(cond[j], parts[j])))
        return slots[newly], deaths

    @staticmethod
    def _death_cause(cond_row: np.ndarray, parts_row: np.ndarray) -> str:
        best, name = 0.0, "organ failure"
        for i in GENETIC_IDX:
            score = float(cond_row[i]) * CONDS[i].lethal
            if score > best:
                best, name = score, CONDS[i].name
        if best <= 0.0:
            worst = int(np.argmin(parts_row))
            name = f"{PARTS[worst].replace('_', ' ')} failure"
        return name

    # -- inspection ------------------------------------------------------
    def describe(self, pid: int) -> List[str]:
        s = self.slot.get(pid)
        if s is None:
            return []
        lines: List[str] = []
        p = self.parts[s]
        worst = np.argsort(p)[:4]
        lines.append("Body: " + ", ".join(f"{PARTS[i].replace('_', ' ')} {p[i] * 100:.0f}%" for i in worst))
        active = [CONDS[i].name + (f" {self.cond[s, i]:.0%}" if self.cond[s, i] < 0.99 else "")
                  for i in range(NC) if self.cond[s, i] > 0.0]
        if active:
            lines.append("Conditions: " + ", ".join(active))
        carrier = [CONDS[i].name for i in GENETIC_IDX
                   if CONDS[i].mode == "rec" and ((self.geno[s, CONDS[i].locus] & 1) + (self.geno[s, CONDS[i].locus] >> 1)) == 1]
        if carrier:
            lines.append("Carrier: " + ", ".join(carrier))
        extras = []
        if self.pain[s] > 0.15:
            extras.append(f"pain {self.pain[s]:.0%}")
        if self.scars[s]:
            extras.append(f"{int(self.scars[s])} scar(s)")
        if self.trauma[s] > 0.1:
            extras.append(f"trauma {self.trauma[s]:.0%}")
        if self.preg[s] > 0:
            extras.append(f"pregnant day {int(self.preg[s])}/270")
        if extras:
            lines.append("State: " + ", ".join(extras))
        return lines

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        live = self.live_slots()
        d: Dict[str, Any] = {"n": int(live.shape[0])}
        d["pid"] = self.pid[live].astype(np.int32).tobytes()
        for name in ("male", "parts", "ceiling", "geno", "cond", "pain", "trauma", "preg", "preg_father", "care", "treated", "scars", "nutrition"):
            d[name] = getattr(self, name)[live].tobytes()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BodyStore":
        store = cls(max(1024, int(d["n"]) * 2))
        n = int(d["n"])
        pids = np.frombuffer(d["pid"], dtype=np.int32)
        for pid in pids.tolist():
            store.alloc(pid)
        slots = np.array([store.slot[p] for p in pids.tolist()], dtype=np.int64)
        def load(name: str, dtype: Any, shape: Tuple[int, ...]) -> None:
            arr = np.frombuffer(d[name], dtype=dtype).reshape((n,) + shape)
            getattr(store, name)[slots] = arr
        load("male", bool, ())
        load("parts", np.float32, (NP,))
        load("ceiling", np.float32, (NP,))
        load("geno", np.uint8, (NL,))
        load("cond", np.float32, (NC,))
        load("pain", np.float32, ())
        load("trauma", np.float32, ())
        load("preg", np.int16, ())
        load("preg_father", np.int32, ())
        load("care", np.float32, ())
        load("treated", np.float32, ())
        load("scars", np.uint8, ())
        if "nutrition" in d:
            load("nutrition", np.float32, ())
        return store


# ----------------------------------------------------------------------
# Hospitals
# ----------------------------------------------------------------------
@dataclass
class HospitalState:
    building_id: int
    beds: int
    patients: Dict[int, int] = field(default_factory=dict)   # pid -> admit day
    admitted_total: int = 0
    discharged_total: int = 0
    deaths: int = 0
    births: int = 0
    revenue: float = 0.0


class Hospitals:
    COST_PER_DAY = 140.0
    INSURED_SHARE = 0.72            # paid by the public health fund

    def __init__(self, world: WorldMap) -> None:
        self.world = world
        self.hospitals: Dict[int, HospitalState] = {}
        self.waiting: List[int] = []
        self.public_spend_today = 0.0
        self.public_spend_total = 0.0
        self.refresh()

    def refresh(self) -> None:
        for b in self.world.building_by_type("hospital"):
            if b.building_id not in self.hospitals:
                self.hospitals[b.building_id] = HospitalState(b.building_id, beds=max(6, 5 * b.floors + b.capacity // 12))

    def staff(self, hid: int) -> int:
        b = self.world.buildings.get(hid)
        c = self.world.companies.get(b.company_id) if b is not None and b.company_id is not None else None
        return len(c.employee_ids) if c is not None else 0

    def quality(self, hid: int) -> float:
        st = self.hospitals[hid]
        staff = self.staff(hid)
        ratio = staff / max(1.0, st.beds * 0.55)
        occupancy = len(st.patients) / max(1, st.beds)
        return float(clamp(0.45 + 0.8 * min(ratio, 1.2) - 0.35 * max(0.0, occupancy - 0.85), 0.35, 1.6))

    @property
    def total_beds(self) -> int:
        return sum(h.beds for h in self.hospitals.values())

    @property
    def occupied(self) -> int:
        return sum(len(h.patients) for h in self.hospitals.values())


# ----------------------------------------------------------------------
# Schools
# ----------------------------------------------------------------------
GRADE_SUBJECTS = ("literacy", "numeracy", "science", "civics")


@dataclass
class SchoolRecord:
    grade: int = 0               # 1..12; 0 = not in school
    gpa: float = 0.6
    absences: int = 0
    bullied: float = 0.0
    bully: float = 0.0
    dropped: bool = False


class LifeSystem:
    """Orchestrates body, medicine, social bonds, births and schooling."""

    PREG_DAYS = 270

    def __init__(self, seed: int, world: WorldMap) -> None:
        self.world = world
        self.rng = random.Random(seed + 9100)
        self.np_rng = np.random.default_rng(seed + 9101)
        self.body = BodyStore()
        self.hospitals = Hospitals(world)
        self.school: Dict[int, SchoolRecord] = {}
        self.stats: Dict[str, float] = {k: 0.0 for k in (
            "births", "marriages", "divorces", "friendships", "conversations", "bullying",
            "injuries", "hospital_deaths", "graduates", "dropouts", "genetic_cases", "romances",
            "stillbirths", "complications", "widowed")}
        self.deaths_by_cause: Dict[str, int] = {}
        self.dating: Dict[int, int] = {}          # pid -> partner pid (unmarried couples)
        self.dating_since: Dict[int, int] = {}
        self.events: List[str] = []
        self.convo_log: List[str] = []            # latest conversations (UI)
        self.day = 0

    # ---------------------------------------------------------------- setup
    def register(self, person: Any, genome: Optional[np.ndarray] = None) -> int:
        s = self.body.alloc(person.person_id)
        self.body.male[s] = person.gender == "man"
        self.body.geno[s] = genome if genome is not None else self.body.random_genome(self.np_rng)
        h = float(clamp(person.health, 0.0, 1.0))
        # existing health seeds organ integrity so old/frail citizens start frail
        base = 0.82 + 0.18 * h
        age_loss = max(0.0, (person.age - 45) / 220.0)
        self.body.parts[s] = np.clip(base - age_loss * self.np_rng.random(NP), 0.35, 1.0).astype(np.float32)
        self.body.ceiling[s] = np.maximum(self.body.parts[s], 0.5)
        if person.age < 18:
            self.school[person.person_id] = SchoolRecord(grade=max(0, min(12, person.age - 5)) if person.age >= 5 else 0,
                                                         gpa=0.45 + 0.4 * person.genetics.cognition * self.rng.random() + 0.15)
        return s

    def register_all(self, population: Any) -> None:
        for person in population.people.values():
            if person.person_id not in self.body.slot:
                self.register(person)
        self.seed_conditions(population)

    def seed_conditions(self, population: Any) -> None:
        """Existing genetic conditions at day 0 (a population has prevalence, not a blank slate)."""
        slots = self.body.live_slots()
        if slots.shape[0] == 0:
            return
        ages = np.fromiter((population.people[int(self.body.pid[s])].age for s in slots), dtype=np.float32, count=slots.shape[0])
        cnt = self.body.alleles(slots)
        male = self.body.male[slots]
        for i in GENETIC_IDX:
            c = CONDS[i]
            n_al = cnt[:, c.locus]
            if c.mode == "rec":
                hit = (n_al == 2) & (ages >= c.onset_age)
            elif c.mode == "dom":
                hit = (n_al >= 1) & (ages >= c.onset_age)
            elif c.mode == "x":
                hit = np.where(male, n_al >= 1, n_al == 2)
            else:
                span = np.clip((ages - c.onset_age) / 40.0, 0.0, 1.0)
                hit = (self.np_rng.random(slots.shape[0]) < ADD_RISK[n_al] * c.onset_rate * 365 * span * 6)
            hit &= ages >= c.onset_age
            if hit.any():
                self.body.cond[slots[hit], i] = 0.05 + 0.5 * self.np_rng.random(int(hit.sum())) * np.clip(ages[hit] / 80.0, 0.2, 1.0)

    # ------------------------------------------------------------- injuries
    def injure(self, population: Any, pid: int, cause: str, severity: float, note: str = "") -> float:
        s = self.body.slot.get(pid)
        person = population.people.get(pid)
        if s is None or person is None:
            return 0.0
        notes, total = self.body.injure(s, cause, severity, self.np_rng)
        self.stats["injuries"] += 1
        person.health = clamp(person.health - total * 0.45, 0.0, 1.0)
        person.stress = clamp(person.stress + 0.1 + total * 0.2, 0.0, 1.0)
        if cause in ("assault", "weapon", "duty", "bullying", "fire"):
            self.body.add_trauma(s, 0.1 + 0.4 * severity)
        what = ", ".join(notes) if notes else "was hurt"
        person.add_memory("health", f"{note or cause.title()}: {what}.", -0.2 - total * 0.2)
        if total > 0.5 and not person.care_tile:
            self.hospitals.waiting.append(pid)
        return total

    def traumatize(self, population: Any, pid: int, amount: float, note: str) -> None:
        s = self.body.slot.get(pid)
        person = population.people.get(pid)
        if s is None or person is None:
            return
        self.body.add_trauma(s, amount)
        person.stress = clamp(person.stress + amount * 0.5, 0.0, 1.0)
        person.add_memory("mind", note, -0.15 - amount * 0.3)

    # --------------------------------------------------------- daily: body
    HAZARD_SECTORS = {"manufacturing": 0.00012, "construction": 0.0002, "logistics": 0.00008, "transport": 0.00005,
                      "food": 0.00003, "healthcare": 0.00003, "retail": 0.00002}

    def stage_hazards(self, population: Any, day_index: int) -> None:
        """Workplace accidents, road crashes and falls: the everyday sources of injury."""
        rng = self.rng
        people = population.people
        world = self.world
        for company in world.companies.values():
            rate = self.HAZARD_SECTORS.get(company.sector, 0.0)
            if rate <= 0 or not company.employee_ids:
                continue
            hits = rng.random() * 0 + len(company.employee_ids) * rate
            n = int(hits) + (1 if rng.random() < hits - int(hits) else 0)
            for _ in range(n):
                pid = company.employee_ids[int(rng.random() * len(company.employee_ids))]
                if pid in people and not people[pid].incarcerated:
                    self.injure(population, pid, "workplace", 0.1 + 0.55 * rng.random() ** 2, f"Accident at {company.name}")
        # road crashes scale with vehicle trips and congestion
        drivers = [p for p in people.values() if p.mode >= 2 and not p.incarcerated]
        expected = len(drivers) * 0.00012 * (1.0 + 2.0 * max(0.0, world.traffic_factor - 1.0)) * (1.6 if world.bad_weather else 1.0)
        n = int(expected) + (1 if rng.random() < expected - int(expected) else 0)
        for _ in range(n):
            p = drivers[int(rng.random() * len(drivers))]
            self.injure(population, p.person_id, "accident", 0.1 + 0.8 * rng.random() ** 2, "Road accident")
        elderly = [p for p in people.values() if p.age >= 70]
        if elderly:
            expected = len(elderly) * 0.0002
            n = int(expected) + (1 if rng.random() < expected - int(expected) else 0)
            for _ in range(n):
                p = elderly[int(rng.random() * len(elderly))]
                self.injure(population, p.person_id, "fall", 0.15 + 0.6 * rng.random(), "Fell at home")

    def stage_body(self, population: Any, day_index: int) -> List[str]:
        self.day = day_index
        self.stage_hazards(population, day_index)
        body = self.body
        slots = body.live_slots()
        n = slots.shape[0]
        out: List[str] = []
        if n == 0:
            return out
        people = population.people
        plist = [people.get(int(body.pid[s])) for s in slots]
        if any(p is None for p in plist):
            keep = [i for i, p in enumerate(plist) if p is not None]
            for i, p in enumerate(plist):
                if p is None:
                    body.release(int(body.pid[slots[i]]))
            slots = slots[keep]
            plist = [plist[i] for i in keep]
            n = len(plist)
        ages = np.fromiter((p.age for p in plist), dtype=np.float32, count=n)
        health = np.fromiter((p.health for p in plist), dtype=np.float32, count=n)
        stress = np.fromiter((p.stress for p in plist), dtype=np.float32, count=n)
        # treated share: medication access scales with money, plus hospital stay
        wealth = np.fromiter((p.cash + p.bank_balance for p in plist), dtype=np.float32, count=n)
        body.treated[slots] = np.clip(0.25 + wealth / 8000.0, 0.0, 0.9)
        # nutrition follows what households can afford to eat; hunger slows healing and wears health down
        body.nutrition[slots] = body.nutrition[slots] * 0.9 + 0.1 * np.clip(wealth / 350.0, 0.0, 1.0)
        health = health * (0.75 + 0.25 * body.nutrition[slots])
        for j in np.nonzero(body.nutrition[slots] < 0.2)[0].tolist():
            plist[j].health = clamp(plist[j].health - 0.0015, 0.0, 1.0)
        # hospital care multiplier is set by stage_hospitals; reset for the discharged
        in_care = np.zeros(n, dtype=bool)
        for st in self.hospitals.hospitals.values():
            if st.patients:
                for pid in st.patients:
                    sl = body.slot.get(pid)
                    if sl is not None:
                        body.care[sl] = 1.0 + 2.0 * min(1.4, self.hospitals.quality(st.building_id))
        new_slots, deaths = body.tick(ages, slots, health, stress, self.np_rng)
        for s in new_slots.tolist():
            p = people.get(int(body.pid[s]))
            if p is None:
                continue
            names = [CONDS[i].name for i in range(NC) if body.cond[s, i] > 0.0 and not self._announced(p, CONDS[i].name)]
            if names:
                self.stats["genetic_cases"] += sum(1 for i in range(NC) if CONDS[i].locus >= 0 and body.cond[s, i] > 0)
                p.add_memory("health", f"Was diagnosed with {names[0]}.", -0.25)
                p.stress = clamp(p.stress + 0.08, 0.0, 1.0)
        # reflect body state back onto the citizen's headline stats
        vit = body.vitality(slots)
        mob = body.mobility(slots)
        pain = body.pain[slots]
        for j in np.nonzero((vit + 0.12 < health) | (pain > 0.3) | (mob < 0.6))[0].tolist():
            p = plist[j]
            target = float(vit[j] * 0.8 + 0.2)
            if p.health > target:
                p.health = p.health - (p.health - target) * 0.35
            if pain[j] > 0.3:
                p.happiness = clamp(p.happiness - pain[j] * 0.04, 0.0, 1.0)
            if mob[j] < 0.6:
                p.energy = clamp(p.energy - (0.6 - mob[j]) * 0.05, 0.0, 1.0)
        dep = body.cond[slots][:, MENTAL_IDX].max(axis=1)
        for j in np.nonzero(dep > 0.0)[0].tolist():
            p = plist[j]
            p.happiness = clamp(p.happiness - dep[j] * 0.035, 0.0, 1.0)
            p.motivation = clamp(p.motivation - dep[j] * 0.02, 0.0, 1.0)
        if deaths:
            dead_ids = []
            for s, cause in deaths:
                pid = int(body.pid[s])
                p = people.get(pid)
                if p is None:
                    continue
                self.deaths_by_cause[cause] = self.deaths_by_cause.get(cause, 0) + 1
                out.append(f"{p.full_name}, {p.age}, died of {cause}.")
                dead_ids.append(pid)
            if dead_ids:
                population.remove_people(dead_ids)
        return out

    @staticmethod
    def _announced(person: Any, name: str) -> bool:
        return any(name in m.headline for m in person.memories[-6:])

    # ---------------------------------------------------------- hospitals
    def stage_hospitals(self, population: Any, day_index: int) -> List[str]:
        out: List[str] = []
        net = self.hospitals
        net.refresh()
        if not net.hospitals:
            return out
        body = self.body
        people = population.people
        net.public_spend_today = 0.0
        # discharge / bill current patients
        for st in net.hospitals.values():
            company = self.world.companies.get(self.world.buildings[st.building_id].company_id) if st.building_id in self.world.buildings else None
            qual = net.quality(st.building_id)
            for pid in list(st.patients):
                p = people.get(pid)
                sl = body.slot.get(pid)
                if p is None or sl is None:
                    del st.patients[pid]
                    continue
                stay = day_index - st.patients[pid]
                bill = net.COST_PER_DAY * (0.6 + 0.8 * qual)
                own = bill * (1.0 - net.INSURED_SHARE)
                paid = min(own, max(0.0, p.cash))
                p.cash -= paid
                p.debt += own - paid
                public = bill * net.INSURED_SHARE
                net.public_spend_today += public
                st.revenue += bill
                if company is not None:
                    company.cash += bill
                a = float(body.acuity(np.array([sl]))[0])
                if a < 0.22 or stay > 40:
                    del st.patients[pid]
                    st.discharged_total += 1
                    p.care_tile = None
                    body.care[sl] = 1.0
                    if stay > 1:
                        p.add_memory("health", "Was discharged from hospital.", 0.1)
        net.public_spend_total += net.public_spend_today
        # triage
        slots = body.live_slots()
        if slots.shape[0]:
            acu = body.acuity(slots)
            eligible = np.nonzero(acu > 0.5)[0]
            order = eligible[np.argsort(-acu[eligible])]
            admitted_now = set(pid for st in net.hospitals.values() for pid in st.patients)
            queue: List[int] = []
            for j in order[:400].tolist():
                pid = int(body.pid[slots[j]])
                if pid in admitted_now or pid not in people or people[pid].incarcerated:
                    continue
                queue.append(pid)
            net.waiting = queue
            for pid in queue:
                p = people[pid]
                best, best_d = None, 10 ** 9
                for st in net.hospitals.values():
                    if len(st.patients) >= st.beds:
                        continue
                    hb = self.world.buildings[st.building_id]
                    d = abs(hb.tile[0] - p.location[0]) + abs(hb.tile[1] - p.location[1])
                    if d < best_d:
                        best, best_d = st, d
                if best is None:
                    break
                best.patients[pid] = day_index
                best.admitted_total += 1
                p.care_tile = self.world.buildings[best.building_id].tile
                p.destination = p.care_tile
                p.add_memory("health", "Was admitted to hospital.", -0.15)
        return out

    # ------------------------------------------------- pregnancy and births
    def stage_pregnancy(self, population: Any, day_index: int) -> List[str]:
        out: List[str] = []
        body = self.body
        people = population.people
        # progress existing pregnancies
        preg_slots = np.nonzero(body.alive & (body.preg > 0))[0]
        births: List[Tuple[int, int]] = []
        for s in preg_slots.tolist():
            body.preg[s] += 1
            if body.preg[s] >= self.PREG_DAYS:
                births.append((int(body.pid[s]), int(body.preg_father[s])))
        for mid, fid in births:
            msg = self._give_birth(population, mid, fid, day_index)
            if msg:
                out.append(msg)
        # conceptions: married or committed couples
        rng = self.rng
        for pid, p in people.items():
            if p.gender != "woman" or p.spouse_id is None or not (20 <= p.age <= 42):
                continue
            s = body.slot.get(pid)
            if s is None or body.preg[s] > 0:
                continue
            if len(p.children_ids) >= 4 or p.incarcerated:
                continue
            partner = people.get(p.spouse_id)
            if partner is None or partner.incarcerated or partner.gender != "man":
                continue
            fert = clamp(1.0 - (p.age - 20) / 30.0, 0.05, 1.0) * (0.6 + 0.4 * p.health) * (1.0 - body.parts[s, VITAL].min() * 0.0)
            desire = clamp(1.0 - len(p.children_ids) * 0.28, 0.1, 1.0) * (0.7 if p.cash + p.bank_balance < 800 else 1.0)
            if rng.random() < 0.0024 * fert * desire:
                body.preg[s] = 1
                body.preg_father[s] = partner.person_id
                p.add_memory("life", "Learned they were expecting a child.", 0.4)
                partner.add_memory("life", "Learned they were going to be a parent.", 0.35)
                p.happiness = clamp(p.happiness + 0.08, 0.0, 1.0)
        return out

    def _give_birth(self, population: Any, mother_id: int, father_id: int, day_index: int) -> Optional[str]:
        people = population.people
        mother = people.get(mother_id)
        s = self.body.slot.get(mother_id)
        if mother is None or s is None:
            return None
        self.body.preg[s] = 0
        in_hospital = False
        for st in self.hospitals.hospitals.values():
            if mother_id in st.patients:
                in_hospital = True
                st.births += 1
        qual = 1.0 if not in_hospital else 1.6
        risk = 0.012 + max(0.0, mother.age - 34) * 0.004 + (1.0 - mother.health) * 0.02
        if self.rng.random() < risk / qual:
            self.stats["complications"] += 1
            self.injure(population, mother_id, "accident", 0.35, "Childbirth complications")
            if self.rng.random() < 0.35 / qual:
                self.stats["stillbirths"] += 1
                mother.add_memory("life", "Lost the baby during childbirth.", -0.8)
                mother.stress = clamp(mother.stress + 0.3, 0.0, 1.0)
                return f"{mother.full_name} lost a baby in childbirth."
        parents = [mother_id] + ([father_id] if father_id in people else [])
        child = population.create_child(population._person_counter, parents)
        population._person_counter += 1
        if father_id in people:
            people[father_id].add_memory("life", f"{child.full_name} was born.", 0.5)
        mother.add_memory("life", f"Gave birth to {child.full_name}.", 0.6)
        mother.happiness = clamp(mother.happiness + 0.1, 0.0, 1.0)
        mg = self.body.geno[s]
        fs = self.body.slot.get(father_id)
        fg = self.body.geno[fs] if fs is not None else self.body.random_genome(self.np_rng)
        cs = self.register(child, self.body.inherit(mg, fg, self.np_rng))
        self.body.male[cs] = child.gender == "man"
        child.health = clamp(0.8 + self.rng.uniform(-0.1, 0.1) - (0.1 if self.body.cond[s].max() > 0.5 else 0.0), 0.2, 1.0)
        if father_id in people:
            child.full_name = f"{child.full_name.split()[0]} {people[father_id].full_name.split()[-1]}" if " " in child.full_name else child.full_name
        self.stats["births"] += 1
        return f"{child.full_name} was born to {mother.full_name}."

    # ------------------------------------- conversations, friendship, romance
    TOPICS = ("work", "family", "money", "gossip", "romance", "health", "hobbies", "news", "complaints", "advice")

    def _topic(self, a: Any, b: Any, rel: Any, same_company: bool) -> str:
        r = self.rng.random()
        if same_company and r < 0.35:
            return "work"
        if rel.romance > 0.3 and r < 0.5:
            return "romance"
        if a.stress > 0.65 and r < 0.4:
            return "complaints"
        if a.cash + a.bank_balance < 300 and r < 0.3:
            return "money"
        if a.health < 0.5 and r < 0.35:
            return "health"
        if a.children_ids and b.children_ids and r < 0.4:
            return "family"
        return self.rng.choice(self.TOPICS)

    def converse(self, a: Any, b: Any, day_index: int) -> float:
        """One real conversation between two people; returns its quality (-1..1)."""
        rel_a = a.relationship_with(b.person_id)
        rel_b = b.relationship_with(a.person_id)
        same_co = a.job_company_id is not None and a.job_company_id == b.job_company_id
        topic = self._topic(a, b, rel_a, same_co)
        pa, pb = a.personality, b.personality
        compat = 1.0 - (abs(pa.sociability - pb.sociability) + abs(pa.empathy - pb.empathy)
                        + abs(pa.risk_tolerance - pb.risk_tolerance)) / 3.0
        mood = (a.happiness + b.happiness) * 0.5 - (a.stress + b.stress) * 0.25
        warmth = (pa.empathy + pb.empathy) * 0.5
        fit = {"work": 0.1 if same_co else -0.05, "gossip": 0.05, "complaints": -0.05 + warmth * 0.2,
               "advice": warmth * 0.15, "romance": 0.1 if rel_a.romance > 0.2 else -0.1}.get(topic, 0.05)
        quality = clamp((compat - 0.5) * 0.9 + mood * 0.5 + fit + self.rng.uniform(-0.25, 0.25), -1.0, 1.0)
        gain = quality * (0.02 + 0.03 * min(pa.sociability, pb.sociability))
        for rel, me, other in ((rel_a, a, b), (rel_b, b, a)):
            rel.affinity = clamp(rel.affinity + gain, -1.0, 1.0)
            rel.familiarity = clamp(rel.familiarity + 0.03, 0.0, 1.0)
            rel.trust = clamp(rel.trust + (gain * 0.6 if quality > 0 else gain * 1.2), 0.0, 1.0)
            if topic in ("advice", "complaints", "family", "health") and quality > 0:
                rel.trust = clamp(rel.trust + 0.015, 0.0, 1.0)
            if quality < -0.5:
                rel.rivalry = clamp(rel.rivalry + 0.05, 0.0, 1.0)
            rel.last_interaction_day = day_index
            rel.interaction_count += 1
            if topic == "advice" and quality > 0.2:
                me.stress = clamp(me.stress - 0.02, 0.0, 1.0)
            if topic == "gossip" and other.gossip_log:
                rel.gossip_knowledge = clamp(rel.gossip_knowledge + 0.05, 0.0, 1.0)
        for me in (a, b):
            me.happiness = clamp(me.happiness + quality * 0.01, 0.0, 1.0)
            me.stress = clamp(me.stress - max(0.0, quality) * 0.01, 0.0, 1.0)
        # romantic spark between eligible adults of compatible age
        if self._romance_eligible(a, b) and (topic == "romance" or quality > 0.35):
            spark = max(0.0, quality) * 0.07 * (0.5 + warmth)
            rel_a.romance = clamp(rel_a.romance + spark, 0.0, 1.0)
            rel_b.romance = clamp(rel_b.romance + spark * self.rng.uniform(0.6, 1.1), 0.0, 1.0)
        self.stats["conversations"] += 1
        if self.rng.random() < 0.01:
            tone = "warmly" if quality > 0.3 else "tensely" if quality < -0.3 else "casually"
            self.convo_log = (self.convo_log + [f"{a.full_name} and {b.full_name} talked about {topic} {tone}."])[-24:]
        return quality

    @staticmethod
    def _romance_eligible(a: Any, b: Any) -> bool:
        return (a.age >= 18 and b.age >= 18 and abs(a.age - b.age) <= 12 and a.spouse_id is None and b.spouse_id is None
                and b.person_id not in a.family_ids and b.person_id not in a.parent_ids and a.person_id not in b.parent_ids
                and a.gender != b.gender)

    def social_slice(self, population: Any, day_index: int, j: int, k: int) -> List[str]:
        """Each citizen has a few conversations a day with people they share a place or a circle with."""
        out: List[str] = []
        people = population.people
        world = self.world
        rng = self.rng
        for p in population._slice(j, k):
            if p.incarcerated or p.age < 4:
                continue
            if rng.random() > 0.25 + 0.45 * p.personality.sociability:
                continue
            other: Optional[Any] = None
            if p.spouse_id is not None and rng.random() < 0.35:
                other = people.get(p.spouse_id)
            if other is None:
                tile = world.tiles[p.location[1] % world.height][p.location[0] % world.width]
                occ = tile.occupants
                if len(occ) > 1 and rng.random() < 0.6:
                    ordered = sorted(occ)
                    oid = ordered[int(rng.random() * len(ordered))]
                    if oid != p.person_id:
                        other = people.get(oid)
            if other is None and p.social_circle:
                other = people.get(p.social_circle[int(rng.random() * len(p.social_circle))])
            if other is None or other.person_id == p.person_id or other.incarcerated:
                continue
            if (p.age < 18) != (other.age < 18) and p.person_id not in other.family_ids:
                continue                                   # kids talk to kids unless family
            self.converse(p, other, day_index)
            rel = p.relationship_with(other.person_id)
            msg = self._advance_bond(p, other, rel, day_index)
            if msg:
                out.append(msg)
        return out

    def _advance_bond(self, a: Any, b: Any, rel_a: Any, day_index: int) -> Optional[str]:
        rel_b = b.relationship_with(a.person_id)
        pid_a, pid_b = a.person_id, b.person_id
        # --- friendship
        if a.relation_roles.get(pid_b) is None and rel_a.affinity > 0.45 and rel_a.trust > 0.35 and rel_b.affinity > 0.4:
            a.set_relation_role(pid_b, "friend")
            b.set_relation_role(pid_a, "friend")
            if pid_b not in a.social_circle:
                a.social_circle.append(pid_b)
            if pid_a not in b.social_circle:
                b.social_circle.append(pid_a)
            self.stats["friendships"] += 1
            a.add_memory("social", f"Became friends with {b.full_name}.", 0.15, related_people=[pid_b])
            b.add_memory("social", f"Became friends with {a.full_name}.", 0.15, related_people=[pid_a])
            return None
        if rel_a.rivalry > 0.6 and a.relation_roles.get(pid_b) in (None, "friend"):
            a.set_relation_role(pid_b, "rival")
        # --- romance: dating -> marriage
        if self._romance_eligible(a, b):
            if (pid_a not in self.dating and pid_b not in self.dating and rel_a.romance > 0.5 and rel_b.romance > 0.45
                    and rel_a.trust > 0.3):
                self.dating[pid_a] = pid_b
                self.dating[pid_b] = pid_a
                self.dating_since[pid_a] = self.dating_since[pid_b] = day_index
                a.set_relation_role(pid_b, "partner")
                b.set_relation_role(pid_a, "partner")
                self.stats["romances"] += 1
                a.happiness = clamp(a.happiness + 0.06, 0.0, 1.0)
                b.happiness = clamp(b.happiness + 0.06, 0.0, 1.0)
                a.add_memory("romance", f"Started dating {b.full_name}.", 0.35)
                b.add_memory("romance", f"Started dating {a.full_name}.", 0.35)
                return f"{a.full_name} and {b.full_name} started dating."
        if self.dating.get(pid_a) == pid_b:
            since = self.dating_since.get(pid_a, day_index)
            if rel_a.romance < 0.25 or rel_a.rivalry > 0.5 or (day_index - since > 120 and self.rng.random() < 0.01):
                self._break_up(a, b)
                return f"{a.full_name} and {b.full_name} broke up."
            if (day_index - since >= 40 and rel_a.romance > 0.78 and rel_b.romance > 0.7
                    and rel_a.trust > 0.55 and a.age >= 20 and b.age >= 20 and self.rng.random() < 0.08):
                return self._marry(a, b)
        return None

    def _break_up(self, a: Any, b: Any) -> None:
        for x, y in ((a, b), (b, a)):
            self.dating.pop(x.person_id, None)
            self.dating_since.pop(x.person_id, None)
            if x.relation_roles.get(y.person_id) == "partner":
                x.relation_roles[y.person_id] = "ex"
            x.happiness = clamp(x.happiness - 0.08, 0.0, 1.0)
            x.stress = clamp(x.stress + 0.1, 0.0, 1.0)
            x.add_memory("romance", f"Broke up with {y.full_name}.", -0.4)

    def _marry(self, a: Any, b: Any) -> str:
        for x in (a, b):
            self.dating.pop(x.person_id, None)
            self.dating_since.pop(x.person_id, None)
        a.set_spouse(b)
        self.stats["marriages"] += 1
        a.happiness = clamp(a.happiness + 0.12, 0.0, 1.0)
        b.happiness = clamp(b.happiness + 0.12, 0.0, 1.0)
        # wedding: shared household when possible, shared name
        wife, husband = (a, b) if a.gender == "woman" else (b, a)
        if wife.full_name.split()[-1] != husband.full_name.split()[-1] and " " in wife.full_name:
            wife.full_name = f"{wife.full_name.split()[0]} {husband.full_name.split()[-1]}"
        a.add_memory("romance", f"Married {b.full_name}.", 0.8)
        b.add_memory("romance", f"Married {a.full_name}.", 0.8)
        return f"{a.full_name} and {b.full_name} got married."

    def stage_marriages(self, population: Any, day_index: int) -> List[str]:
        """Divorce of strained marriages and widowhood housekeeping."""
        out: List[str] = []
        for p in list(population.people.values()):
            if p.spouse_id is None or p.gender != "woman":
                continue
            other = population.people.get(p.spouse_id)
            if other is None:
                p.spouse_id = None
                continue
            rel = p.relationship_with(other.person_id)
            strain = (1.0 - rel.affinity) * 0.4 + p.stress * 0.2 + rel.rivalry * 0.5 - rel.trust * 0.4
            if p.debt > 20000:
                strain += 0.1
            if strain > 0.52 and self.rng.random() < 0.004:
                p.spouse_id = None
                other.spouse_id = None
                for x, y in ((p, other), (other, p)):
                    x.relation_roles[y.person_id] = "ex"
                    x.happiness = clamp(x.happiness - 0.15, 0.0, 1.0)
                    x.stress = clamp(x.stress + 0.15, 0.0, 1.0)
                    x.add_memory("romance", f"Divorced {y.full_name}.", -0.6)
                self.stats["divorces"] += 1
                out.append(f"{p.full_name} and {other.full_name} divorced.")
        return out

    # -------------------------------------------------------- bereavement
    def on_death(self, population: Any, person: Any) -> None:
        pid = person.person_id
        self.body.release(pid)
        self.school.pop(pid, None)
        if getattr(population, "_emigrating", False):
            self.dating.pop(pid, None)
            return
        partner = self.dating.pop(pid, None)
        if partner is not None:
            self.dating.pop(partner, None)
        for st in self.hospitals.hospitals.values():
            if pid in st.patients:
                del st.patients[pid]
                st.deaths += 1
                self.stats["hospital_deaths"] += 1
        people = population.people
        grieving = set(person.family_ids) | set(person.parent_ids) | set(person.children_ids)
        if person.spouse_id is not None:
            grieving.add(person.spouse_id)
        grieving |= {oid for oid, role in person.relation_roles.items() if role in ("friend", "partner")}
        for oid in grieving:
            o = people.get(oid)
            if o is None:
                continue
            closeness = o.relationship_with(pid).closeness() + (0.5 if oid == person.spouse_id or oid in person.children_ids else 0.2)
            o.stress = clamp(o.stress + 0.25 * min(1.0, closeness), 0.0, 1.0)
            o.happiness = clamp(o.happiness - 0.2 * min(1.0, closeness), 0.0, 1.0)
            o.add_memory("loss", f"{person.full_name} died.", -0.7 * min(1.0, closeness + 0.2))
            s = self.body.slot.get(oid)
            if s is not None and closeness > 0.7:
                self.body.add_trauma(s, 0.15)
            if oid == person.spouse_id:
                self.stats["widowed"] += 1

    # ----------------------------------------------------------- schooling
    def stage_schools(self, population: Any, day_index: int) -> List[str]:
        """Daily learning, bullying, truancy; yearly promotion and graduation."""
        out: List[str] = []
        people = population.people
        world = self.world
        rng = self.rng
        # teaching quality per school building
        quality: Dict[int, float] = {}
        for b in world.building_by_type("school"):
            c = world.companies.get(b.company_id) if b.company_id is not None else None
            staff = len(c.employee_ids) if c is not None else 0
            students = sum(1 for p in people.values() if p.current_enrollment_building == b.building_id and p.student_status == "school") if False else 0
            quality[b.building_id] = staff
        pupils_by_school: Dict[int, List[Any]] = {}
        schools = world.building_by_type("school")
        for p in people.values():
            if 5 <= p.age < 18 and p.person_id not in self.school and schools:
                self.school[p.person_id] = SchoolRecord(grade=max(1, p.age - 5), gpa=0.45 + 0.35 * p.genetics.cognition)
                if p.student_status is None:
                    nearest = min(schools, key=lambda b: abs(b.tile[0] - p.location[0]) + abs(b.tile[1] - p.location[1]))
                    p.student_status = "school"
                    p.current_enrollment_building = nearest.building_id
        for pid, rec in list(self.school.items()):
            p = people.get(pid)
            if p is None:
                self.school.pop(pid, None)
                continue
            if p.age >= 18:
                self._graduate(p, rec, out)
                continue
            if p.age < 5:
                continue
            if rec.grade == 0:
                rec.grade = max(1, p.age - 5)
            if p.student_status == "school" and not rec.dropped:
                pupils_by_school.setdefault(p.current_enrollment_building or -1, []).append(p)
        for bid, pupils in pupils_by_school.items():
            staff = quality.get(bid, 0)
            ratio = len(pupils) / max(1.0, staff)
            teach = clamp(1.25 - ratio / 30.0, 0.35, 1.2)
            c = world.companies.get(world.buildings[bid].company_id) if bid in world.buildings and world.buildings[bid].company_id is not None else None
            if c is not None and c.employee_ids:
                skill = sum(people[t].skills.get("teaching", 0.2) for t in c.employee_ids[:20] if t in people) / max(1, min(20, len(c.employee_ids)))
                teach = clamp(teach * (0.8 + 0.5 * skill), 0.3, 1.3)
            bullies = [p for p in pupils if self.school[p.person_id].bully > 0.5]
            for p in pupils:
                rec = self.school[p.person_id]
                if rng.random() < 0.04 + p.stress * 0.06 + (0.05 if rec.bullied > 0.5 else 0.0):
                    rec.absences += 1
                    continue
                gain = (0.0009 + 0.0014 * p.genetics.cognition + 0.0006 * p.work_traits.get("learning_drive", 0.5)) * teach
                target = clamp((0.25 + 0.45 * p.genetics.cognition + 0.25 * p.work_traits.get("learning_drive", 0.5)) * (0.7 + 0.3 * teach)
                               - (0.12 if rec.bullied > 0.5 else 0.0) - (0.1 if p.stress > 0.7 else 0.0)
                               - 0.01 * min(rec.absences, 12), 0.05, 0.98)
                rec.gpa += (target - rec.gpa) * 0.02
                p.skills["literacy"] = clamp(p.skills.get("literacy", 0.0) + gain * 0.8, 0.0, 1.0) if "literacy" in p.skills else p.skills.get("literacy", 0.0)
                rec.bullied = max(0.0, rec.bullied - 0.01)
                rec.bully = max(0.0, rec.bully - 0.002) if p.personality.empathy > 0.4 else rec.bully
                # bullying: low-empathy, impulsive pupils target quieter peers
                if bullies and rec.bully <= 0.5 and p.personality.sociability < 0.5 and rng.random() < 0.02:
                    attacker = bullies[int(rng.random() * len(bullies))]
                    if attacker.person_id != p.person_id:
                        self._bully(population, attacker, p, rec)
                elif rec.bully <= 0.5 and p.personality.empathy < 0.3 and p.personality.impulsiveness > 0.6 and rng.random() < 0.004:
                    rec.bully = 0.6
                    p.add_memory("school", "Started picking on classmates.", -0.1)
                if rec.gpa < 0.18 and p.age >= 15 and rng.random() < 0.002:
                    rec.dropped = True
                    p.student_status = None
                    self.stats["dropouts"] += 1
                    p.add_memory("school", "Dropped out of school.", -0.4)
        if day_index > 0 and day_index % 365 == 364:
            for pid, rec in self.school.items():
                p = people.get(pid)
                if p is not None and p.student_status == "school" and not rec.dropped and rec.grade < 12:
                    if rec.gpa >= 0.3:
                        rec.grade += 1
                    else:
                        p.add_memory("school", "Had to repeat the year.", -0.3)
        return out

    def _bully(self, population: Any, attacker: Any, victim: Any, rec: SchoolRecord) -> None:
        rec.bullied = min(1.0, rec.bullied + 0.25)
        self.stats["bullying"] += 1
        victim.stress = clamp(victim.stress + 0.06, 0.0, 1.0)
        victim.happiness = clamp(victim.happiness - 0.05, 0.0, 1.0)
        rel = victim.relationship_with(attacker.person_id)
        rel.rivalry = clamp(rel.rivalry + 0.2, 0.0, 1.0)
        rel.affinity = clamp(rel.affinity - 0.2, -1.0, 1.0)
        if self.rng.random() < 0.15:
            self.injure(population, victim.person_id, "bullying", 0.18 + 0.2 * self.rng.random(), f"Bullied by {attacker.full_name}")
        else:
            self.traumatize(population, victim.person_id, 0.05, f"Was bullied by {attacker.full_name}.")
        if self.rng.random() < 0.1:
            attacker.crime_tendency = clamp(attacker.crime_tendency + 0.01, 0.0, 1.0)

    def _graduate(self, p: Any, rec: SchoolRecord, out: List[str]) -> None:
        self.school.pop(p.person_id, None)
        if rec.dropped or rec.gpa < 0.25:
            p.education_level = "none" if p.education_level == "none" else p.education_level
            return
        if p.education_level in ("none",):
            p.education_level = "school"
        p.skills["teaching"] = p.skills.get("teaching", 0.0)
        self.stats["graduates"] += 1
        p.add_memory("school", f"Graduated school with a {rec.gpa * 4:.1f} GPA.", 0.4)
        if rec.gpa > 0.75:
            p.skills["research"] = clamp(p.skills.get("research", 0.0) + 0.05, 0.0, 1.0) if "research" in p.skills else p.skills.get("research", 0.0)

    # ------------------------------------------------------------ metrics
    def metrics(self, population: Any) -> Dict[str, float]:
        body = self.body
        slots = body.live_slots()
        net = self.hospitals
        pregnant = int((body.preg[slots] > 0).sum()) if slots.shape[0] else 0
        chronic_cols = [i for i in GENETIC_IDX if CONDS[i].name not in ("myopia", "asthma")]
        chronic = int(((body.cond[slots][:, chronic_cols] > 0).any(axis=1)).sum()) if slots.shape[0] else 0
        mental = int(((body.cond[slots][:, MENTAL_IDX] > 0).any(axis=1)).sum()) if slots.shape[0] else 0
        injured = int(((body.parts[slots].min(axis=1) < 0.7)).sum()) if slots.shape[0] else 0
        pupils = sum(1 for r in self.school.values() if r.grade > 0 and not r.dropped)
        gpa = float(np.mean([r.gpa for r in self.school.values() if r.grade > 0])) if self.school else 0.0
        married = sum(1 for p in population.people.values() if p.spouse_id is not None) // 2
        return {
            "Hospital beds": float(net.total_beds),
            "Beds occupied": float(net.occupied),
            "ER waiting": float(len(net.waiting)),
            "Pregnant now": float(pregnant),
            "Chronic/genetic cases": float(chronic),
            "Mental health cases": float(mental),
            "Injured citizens": float(injured),
            "Pupils": float(pupils),
            "Avg GPA": gpa * 4.0,
            "Married couples": float(married),
            "Dating couples": float(len(self.dating) // 2),
            "Births total": float(self.stats["births"]),
            "Divorces total": float(self.stats["divorces"]),
            "Bullying incidents": float(self.stats["bullying"]),
            "Public health spend/day": float(net.public_spend_today),
        }

    def summary_lines(self, population: Any) -> List[str]:
        m = self.metrics(population)
        lines = [f"{k}: {v:,.1f}" if v % 1 else f"{k}: {int(v):,}" for k, v in m.items()]
        top = sorted(self.deaths_by_cause.items(), key=lambda kv: -kv[1])[:3]
        if top:
            lines.append("Top causes of death: " + ", ".join(f"{k} ({v})" for k, v in top))
        return lines

    # -------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "body": self.body.to_dict(),
            "hospitals": {str(h.building_id): {"beds": h.beds, "patients": {str(k): v for k, v in h.patients.items()},
                                               "admitted": h.admitted_total, "discharged": h.discharged_total,
                                               "deaths": h.deaths, "births": h.births, "revenue": h.revenue}
                          for h in self.hospitals.hospitals.values()},
            "waiting": list(self.hospitals.waiting),
            "public_spend_total": self.hospitals.public_spend_total,
            "school": {str(k): [v.grade, v.gpa, v.absences, v.bullied, v.bully, v.dropped] for k, v in self.school.items()},
            "stats": dict(self.stats),
            "deaths_by_cause": dict(self.deaths_by_cause),
            "dating": {str(k): v for k, v in self.dating.items()},
            "dating_since": {str(k): v for k, v in self.dating_since.items()},
            "rng": self.rng.getstate(),
            "np_rng": _np_pack(self.np_rng.bit_generator.state),
            "day": self.day,
        }

    @classmethod
    def from_dict(cls, seed: int, world: WorldMap, data: Dict[str, Any]) -> "LifeSystem":
        life = cls(seed, world)
        life.body = BodyStore.from_dict(data["body"])
        life.hospitals.hospitals = {}
        for bid, h in data.get("hospitals", {}).items():
            st = HospitalState(int(bid), int(h["beds"]), {int(k): v for k, v in h["patients"].items()},
                               h["admitted"], h["discharged"], h["deaths"], h["births"], h["revenue"])
            life.hospitals.hospitals[int(bid)] = st
        life.hospitals.waiting = list(data.get("waiting", []))
        life.hospitals.public_spend_total = float(data.get("public_spend_total", 0.0))
        life.school = {int(k): SchoolRecord(int(v[0]), float(v[1]), int(v[2]), float(v[3]), float(v[4]), bool(v[5]))
                       for k, v in data.get("school", {}).items()}
        life.stats.update(data.get("stats", {}))
        life.deaths_by_cause = dict(data.get("deaths_by_cause", {}))
        life.dating = {int(k): int(v) for k, v in data.get("dating", {}).items()}
        life.dating_since = {int(k): int(v) for k, v in data.get("dating_since", {}).items()}
        state = data.get("rng")
        if state is not None:
            from world import coerce_rng_state
            st = coerce_rng_state(state)
            if st is not None:
                life.rng.setstate(st)
        np_state = data.get("np_rng")
        if np_state:
            life.np_rng.bit_generator.state = _np_state(np_state)
        life.day = int(data.get("day", 0))
        return life


def _np_pack(state: Dict[str, Any]) -> Dict[str, Any]:
    """128-bit PCG64 words do not fit msgpack ints; store them as strings."""
    out = dict(state)
    inner = dict(out["state"])
    inner["state"] = str(inner["state"])
    inner["inc"] = str(inner["inc"])
    out["state"] = inner
    return out


def _np_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """msgpack turns tuples/ints into lists/bytes; normalise a PCG64 state dict."""
    out = dict(state)
    inner = dict(out["state"])
    inner["state"] = int(inner["state"])
    inner["inc"] = int(inner["inc"])
    out["state"] = inner
    return out
