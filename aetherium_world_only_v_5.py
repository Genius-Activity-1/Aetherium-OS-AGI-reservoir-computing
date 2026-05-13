# Aetherium World-Only V5.4 (FULL)
# -------------------------------------------------
# Phase-field inspired model (rho, theta, phi_g) + Field Events + PNJ + Micro-factions
#
# V5.4 goal (post V5.3): keep factions emergence, but reduce flow saturation.
# Changes vs V5.3:
#   1) Flow desaturation: flow = tanh(grad / k)
#   2) Softer events: lower ev_phase_amp + weaker VORTEX_DEFECT amplitude
#   3) Lower thermal noise (theta_thermal)
#   4) Event spawn self-limiting when local flow already high
#
# Dependencies: numpy, matplotlib
# Run: python aetherium_world_only_v5_4_full.py

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def clamp_signed(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(theta: float) -> float:
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def softmax(x: np.ndarray, tau: float = 0.6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = (x - x.max()) / max(1e-9, tau)
    ex = np.exp(x)
    return ex / ex.sum()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Zone Phase State
# -----------------------------

@dataclass
class ZonePhase:
    rho: float
    theta: float
    phi_g: float

    psi_coherence: float = 0.0
    omg_flow: float = 0.0
    phi_stress: float = 0.0

    def clamp_all(self) -> None:
        self.rho = clamp(self.rho)
        self.theta = wrap_pi(self.theta)
        self.phi_g = clamp(self.phi_g)
        self.psi_coherence = clamp(self.psi_coherence)
        self.omg_flow = clamp(self.omg_flow)
        self.phi_stress = clamp(self.phi_stress)


# -----------------------------
# Field Events
# -----------------------------

FIELD_EVENT_KINDS = (
    "PHASE_SHOCK",
    "DENSITY_DROP",
    "DENSITY_PULSE",
    "GRAV_WELL",
    "COHERENCE_BLESS",
    "VORTEX_DEFECT",
)


@dataclass
class FieldEvent:
    eid: int
    zid: int
    kind: str
    intensity: float
    remaining: int

    def tick(self) -> None:
        self.remaining -= 1


# -----------------------------
# PNJ + Factions
# -----------------------------

PNJ_PROFILES = ("STOIC", "IMPULSIVE", "LEADER", "FRAGILE", "EXPLORER")

PROFILE = {
    "STOIC":     dict(sens_flow=0.70, sens_grav=0.90, learn=0.90, mood_vol=0.65, mobility=0.70, influence=1.00, support_gain=1.20),
    "IMPULSIVE": dict(sens_flow=1.25, sens_grav=1.00, learn=0.80, mood_vol=1.35, mobility=1.00, influence=0.95, support_gain=0.85),
    "LEADER":    dict(sens_flow=0.95, sens_grav=0.95, learn=1.00, mood_vol=0.90, mobility=0.85, influence=1.50, support_gain=1.10),
    "FRAGILE":   dict(sens_flow=1.55, sens_grav=1.15, learn=0.70, mood_vol=1.10, mobility=0.90, influence=0.80, support_gain=1.35),
    "EXPLORER":  dict(sens_flow=1.00, sens_grav=0.85, learn=0.90, mood_vol=0.95, mobility=1.55, influence=0.90, support_gain=0.95),
}


@dataclass
class PNJ:
    pid: int
    zid: int
    profile: str
    coherence: float
    energy: float
    mood: float
    scar: float
    faction_id: Optional[int] = None

    def clamp_all(self) -> None:
        self.coherence = clamp(self.coherence)
        self.energy = clamp(self.energy)
        self.mood = clamp_signed(self.mood)
        self.scar = clamp(self.scar)


@dataclass
class MicroFaction:
    fid: int
    zid: int
    doctrine: float
    cohesion: float
    leader_id: int
    members: Set[int]


# -----------------------------
# World V5.4
# -----------------------------

class WorldV54:
    def __init__(self, w: int = 4, h: int = 3, seed: int = 1234, out_dir: str = "out_v5_4"):
        random.seed(seed)
        np.random.seed(seed)

        self.w = w
        self.h = h
        self.seed = seed
        self.out_dir = out_dir

        self.zones: Dict[int, ZonePhase] = {}
        self.neighbors: Dict[int, List[int]] = {}

        self.events_by_zone: Dict[int, List[FieldEvent]] = {}
        self._eid = 0

        self.pnjs: Dict[int, PNJ] = {}
        self.pnjs_by_zone: Dict[int, List[int]] = {}
        self.factions: Dict[int, MicroFaction] = {}
        self.factions_by_zone: Dict[int, List[int]] = {}
        self._pid = 0
        self._fid = 0

        # -----------------
        # Field params
        # -----------------
        self.phase_diff = 0.035
        self.phase_grav_couple = 0.018
        self.theta_thermal = 0.008  # V5.4 lower noise (was 0.012 in V5.3)

        self.rho_relax = 0.018
        self.rho_grad_loss = 0.055
        self.rho_evap = 0.007
        self.rho_floor = 0.06
        self.rho_floor_push = 0.006
        self.rho_diff = 0.060

        self.poisson_iters = 6
        self.grav_avg_w = 0.62
        self.grav_rho_w = 0.30
        self.grav_leak = 0.012

        self.coherence_tau = 1.15
        self.flow_norm = 2.25  # V5.4: scale for tanh desaturation

        # -----------------
        # Event params
        # -----------------
        self.base_event_rate = 0.08
        self.event_intensity_lo = 0.25
        self.event_intensity_hi = 0.90

        self.ev_phase_amp = 0.52  # V5.4 softer shocks (was 0.65)
        self.ev_rho_amp = 0.22
        self.ev_grav_amp = 0.18

        # V5.4 vortex softer
        self.ev_vortex_center = 0.55 * math.pi
        self.ev_vortex_nb = 0.30 * math.pi

        # -----------------
        # PNJ params
        # -----------------
        self.p_alpha = 0.05
        self.p_beta = 0.040
        self.p_energy_gain = 0.035
        self.p_energy_loss = 0.045
        self.p_scar_gain = 0.020
        self.p_scar_heal = 0.010

        self.move_base = 0.012
        self.move_flow_boost = 0.060

        # -----------------
        # Factions
        # -----------------
        self.max_factions_per_zone = 3
        self.faction_min_size = 6
        self.faction_form_threshold = 0.24
        self.faction_form_acc_rate = 0.11
        self.faction_dissolve_threshold = 0.22
        self.faction_update_stride = 10

        # feedback
        self.fb_theta_align = 0.008
        self.fb_theta_noise = 0.010
        self.fb_rho_support = 0.006

        self._zone_form_acc: Dict[int, float] = {}

        self._init_world()

    # ---------- geometry
    def zid(self, x: int, y: int) -> int:
        return y * self.w + x

    def xy(self, zid: int) -> Tuple[int, int]:
        return zid % self.w, zid // self.w

    def _init_world(self) -> None:
        for y in range(self.h):
            for x in range(self.w):
                zid = self.zid(x, y)
                self.zones[zid] = ZonePhase(
                    rho=random.uniform(0.35, 0.70),
                    theta=random.uniform(-math.pi, math.pi),
                    phi_g=random.uniform(0.18, 0.40),
                )

        for zid in self.zones:
            x, y = self.xy(zid)
            nbs: List[int] = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                xx, yy = x + dx, y + dy
                if 0 <= xx < self.w and 0 <= yy < self.h:
                    nbs.append(self.zid(xx, yy))
            self.neighbors[zid] = nbs

        self.events_by_zone = {zid: [] for zid in self.zones}
        self.pnjs_by_zone = {zid: [] for zid in self.zones}
        self.factions_by_zone = {zid: [] for zid in self.zones}
        self._zone_form_acc = {zid: 0.0 for zid in self.zones}

        self.derive_observables()

    # -----------------------------
    # Field helpers
    # -----------------------------
    def _phase_grad(self, zid: int) -> float:
        z = self.zones[zid]
        nbs = self.neighbors[zid]
        if not nbs:
            return 0.0
        diffs = [abs(wrap_pi(self.zones[n].theta - z.theta)) for n in nbs]
        return float(np.mean(diffs))

    # -----------------------------
    # Field dynamics
    # -----------------------------
    def update_phase_field(self) -> None:
        new_theta: Dict[int, float] = {}
        for zid, z in self.zones.items():
            nbs = self.neighbors[zid]
            if not nbs:
                new_theta[zid] = z.theta
                continue

            diffs = [wrap_pi(self.zones[n].theta - z.theta) for n in nbs]
            lap = float(np.mean(diffs))

            # thermal noise scaled by local stress proxy (phi_g + grad)
            grad = self._phase_grad(zid)
            stress_proxy = clamp(0.55 * z.phi_g + 0.45 * (grad / math.pi))
            noise = self.theta_thermal * (0.4 + 0.6 * stress_proxy) * random.uniform(-1.0, 1.0)

            new_theta[zid] = wrap_pi(z.theta + self.phase_diff * lap - self.phase_grav_couple * z.phi_g + noise)

        for zid, th in new_theta.items():
            self.zones[zid].theta = th

    def update_density(self) -> None:
        rho0 = {zid: self.zones[zid].rho for zid in self.zones}

        for zid, z in self.zones.items():
            grad = self._phase_grad(zid) / math.pi
            z.rho += self.rho_relax * (1.0 - grad) - self.rho_grad_loss * grad
            z.rho -= self.rho_evap * z.rho
            z.rho += self.rho_floor_push * (0.30 - z.rho)
            z.rho = max(self.rho_floor, clamp(z.rho))

        # diffusion / transport
        new_rho: Dict[int, float] = {}
        for zid, z in self.zones.items():
            nbs = self.neighbors[zid]
            if not nbs:
                new_rho[zid] = z.rho
                continue
            m = float(np.mean([rho0[n] for n in nbs]))
            new_rho[zid] = max(self.rho_floor, clamp(z.rho + self.rho_diff * (m - rho0[zid])))

        for zid, val in new_rho.items():
            self.zones[zid].rho = val

    def update_gravity(self, iters: int | None = None) -> None:
        iters = self.poisson_iters if iters is None else iters
        for _ in range(iters):
            new_phi: Dict[int, float] = {}
            for zid, z in self.zones.items():
                nbs = self.neighbors[zid]
                if not nbs:
                    new_phi[zid] = z.phi_g
                    continue
                avg = float(np.mean([self.zones[n].phi_g for n in nbs]))
                val = self.grav_avg_w * avg + self.grav_rho_w * z.rho
                val -= self.grav_leak * val
                new_phi[zid] = clamp(val)
            for zid, val in new_phi.items():
                self.zones[zid].phi_g = val

    def derive_observables(self) -> None:
        for zid, z in self.zones.items():
            phases = [z.theta] + [self.zones[n].theta for n in self.neighbors[zid]]
            center = z.theta
            diffs = [wrap_pi(p - center) for p in phases]
            var = float(np.var(diffs))
            z.psi_coherence = clamp(1.0 / (1.0 + self.coherence_tau * var))

            grad = self._phase_grad(zid)
            # V5.4: desaturated flow
            z.omg_flow = float(np.tanh(grad / self.flow_norm))
            z.omg_flow = clamp(z.omg_flow)

            # stress: slightly flow-sensitive
            z.phi_stress = clamp(0.45 * z.phi_g + 0.55 * z.omg_flow)
            z.clamp_all()

    def tick_field(self) -> None:
        self.update_phase_field()
        self.update_density()
        self.update_gravity()
        self.derive_observables()

    # -----------------------------
    # Events
    # -----------------------------
    def _event_spawn_prob(self, zid: int) -> float:
        z = self.zones[zid]
        x = self.base_event_rate
        x += 0.10 * max(0.0, z.omg_flow - 0.25)
        x += 0.06 * max(0.0, 0.55 - z.psi_coherence)
        # V5.4: self-limiting when already highly turbulent
        if z.omg_flow > 0.80:
            x *= (1.0 - 0.55 * (z.omg_flow - 0.80) / 0.20)
        return clamp(x, 0.0, 0.35)

    def _choose_event_kind(self, zid: int) -> str:
        z = self.zones[zid]
        w_phase = 0.35 + 0.60 * z.omg_flow
        w_drop  = 0.18 + 0.35 * max(0.0, z.phi_g - 0.55)
        w_pulse = 0.10 + 0.25 * max(0.0, 0.50 - z.phi_g)
        w_well  = 0.16 + 0.35 * z.omg_flow
        w_bless = 0.08 + 0.35 * max(0.0, z.psi_coherence - 0.65)
        w_vort  = 0.10 + 0.30 * z.omg_flow

        kinds = [
            ("PHASE_SHOCK", w_phase),
            ("DENSITY_DROP", w_drop),
            ("DENSITY_PULSE", w_pulse),
            ("GRAV_WELL", w_well),
            ("COHERENCE_BLESS", w_bless),
            ("VORTEX_DEFECT", w_vort),
        ]
        total = sum(w for _, w in kinds)
        r = random.random() * total
        acc = 0.0
        for k, w in kinds:
            acc += w
            if r <= acc:
                return k
        return "PHASE_SHOCK"

    def _new_event(self, zid: int, kind: str) -> FieldEvent:
        self._eid += 1
        intensity = clamp(random.uniform(self.event_intensity_lo, self.event_intensity_hi))
        dur = int(6 + 20 * intensity)
        return FieldEvent(eid=self._eid, zid=zid, kind=kind, intensity=intensity, remaining=dur)

    def spawn_events(self) -> None:
        for zid in self.zones:
            if random.random() < self._event_spawn_prob(zid):
                self.events_by_zone[zid].append(self._new_event(zid, self._choose_event_kind(zid)))

    def apply_events(self) -> None:
        for zid in list(self.zones.keys()):
            if not self.events_by_zone[zid]:
                continue

            z = self.zones[zid]
            nbs = self.neighbors[zid]

            keep: List[FieldEvent] = []
            for e in self.events_by_zone[zid]:
                i = e.intensity

                if e.kind == "PHASE_SHOCK":
                    z.theta = wrap_pi(z.theta + self.ev_phase_amp * i * random.uniform(-1.0, 1.0))
                    for nb in nbs:
                        self.zones[nb].theta = wrap_pi(self.zones[nb].theta + 0.22 * self.ev_phase_amp * i * random.uniform(-1.0, 1.0))

                elif e.kind == "VORTEX_DEFECT":
                    # V5.4: softer defect seed
                    z.theta = wrap_pi(z.theta + self.ev_vortex_center * i * random.choice([-1.0, 1.0]))
                    for k, nb in enumerate(nbs):
                        sign = 1.0 if (k % 2 == 0) else -1.0
                        self.zones[nb].theta = wrap_pi(self.zones[nb].theta + sign * self.ev_vortex_nb * i)

                elif e.kind == "DENSITY_DROP":
                    z.rho = max(self.rho_floor, clamp(z.rho - self.ev_rho_amp * i))

                elif e.kind == "DENSITY_PULSE":
                    z.rho = clamp(z.rho + 0.80 * self.ev_rho_amp * i)

                elif e.kind == "GRAV_WELL":
                    z.phi_g = clamp(z.phi_g + self.ev_grav_amp * i)
                    z.rho = max(self.rho_floor, clamp(z.rho - 0.25 * self.ev_rho_amp * i))

                elif e.kind == "COHERENCE_BLESS":
                    # keep it mild, avoid freezing the field
                    if nbs:
                        mean_theta = math.atan2(
                            float(np.mean([math.sin(self.zones[n].theta) for n in nbs])),
                            float(np.mean([math.cos(self.zones[n].theta) for n in nbs])),
                        )
                        z.theta = wrap_pi(z.theta + 0.28 * self.ev_phase_amp * i * wrap_pi(mean_theta - z.theta))

                e.tick()
                if e.remaining > 0:
                    keep.append(e)

            self.events_by_zone[zid] = keep

        self.derive_observables()

    # -----------------------------
    # PNJs
    # -----------------------------
    def init_pnjs(self, n_per_zone: int = 15) -> None:
        self.pnjs.clear()
        self.pnjs_by_zone = {zid: [] for zid in self.zones}
        self._pid = 0

        for zid, z in self.zones.items():
            for _ in range(n_per_zone):
                self._pid += 1
                prof = random.choice(PNJ_PROFILES)
                p = PNJ(
                    pid=self._pid,
                    zid=zid,
                    profile=prof,
                    coherence=clamp(0.45 + 0.45 * z.psi_coherence),
                    energy=clamp(0.50 + 0.35 * z.rho - 0.10 * z.phi_g),
                    mood=clamp_signed(random.uniform(-0.15, 0.15) + 0.20 * (z.psi_coherence - 0.5)),
                    scar=clamp(random.uniform(0.05, 0.20) + 0.12 * z.omg_flow),
                    faction_id=None,
                )
                p.clamp_all()
                self.pnjs[p.pid] = p
                self.pnjs_by_zone[zid].append(p.pid)

    def _compute_faction_cohesion(self, member_ids: List[int]) -> float:
        if not member_ids:
            return 0.0
        moods = np.array([self.pnjs[pid].mood for pid in member_ids], dtype=float)
        Cs = np.array([self.pnjs[pid].coherence for pid in member_ids], dtype=float)
        mood_var = float(np.var(moods)) if len(moods) > 1 else 0.0
        mood_term = clamp(1.0 - mood_var)
        return clamp(0.55 * float(np.mean(Cs)) + 0.45 * mood_term)

    def _remove_from_faction(self, pid: int, fid: int) -> None:
        if fid not in self.factions:
            self.pnjs[pid].faction_id = None
            return
        f = self.factions[fid]
        if pid in f.members:
            f.members.remove(pid)
        self.pnjs[pid].faction_id = None

    def _faction_support(self, pid: int) -> float:
        p = self.pnjs[pid]
        if p.faction_id is None or p.faction_id not in self.factions:
            return 0.0
        f = self.factions[p.faction_id]
        return clamp(f.cohesion) * (0.5 + 0.5 * max(0.0, f.doctrine))

    def update_pnjs(self) -> None:
        for zid in self.zones:
            z = self.zones[zid]
            flow = z.omg_flow
            grav = z.phi_g
            coh = z.psi_coherence

            for pid in list(self.pnjs_by_zone[zid]):
                p = self.pnjs[pid]
                pp = PROFILE[p.profile]

                support = self._faction_support(pid) * pp["support_gain"]
                support = clamp(support, 0.0, 1.0)
                flow_eff = clamp(flow * (1.0 - 0.25 * support))

                p.coherence += self.p_alpha * pp["learn"] * (coh - p.coherence) - self.p_beta * pp["sens_flow"] * flow_eff
                p.energy += self.p_energy_gain * (z.rho - 0.45) - self.p_energy_loss * (0.7 * pp["sens_grav"] * grav + 0.3 * pp["sens_flow"] * flow_eff)
                p.mood += pp["mood_vol"] * (0.05 * (coh - 0.5) - 0.06 * (flow_eff - 0.22) - 0.05 * (grav - 0.35))

                p.scar += self.p_scar_gain * pp["sens_flow"] * max(0.0, flow_eff - 0.20)
                p.scar += 0.010 * max(0.0, 0.40 - p.energy)
                p.scar -= self.p_scar_heal * (0.4 + 0.6 * coh) * (0.6 + 0.4 * support)

                p.clamp_all()

    # -----------------------------
    # Mobility
    # -----------------------------
    def maybe_move_pnj(self, pid: int) -> None:
        p = self.pnjs[pid]
        zid = p.zid
        nbs = self.neighbors.get(zid, [])
        if not nbs:
            return

        z = self.zones[zid]
        flow = z.omg_flow
        pp = PROFILE[p.profile]

        move_p = (self.move_base + self.move_flow_boost * max(0.0, flow - 0.22)) * pp["mobility"]
        if p.profile == "EXPLORER":
            move_p += 0.012
        if p.energy < 0.33:
            move_p += 0.010

        if random.random() > clamp(move_p, 0.0, 0.25):
            return

        scores = []
        for nb in nbs:
            zn = self.zones[nb]
            score = (1.2 * zn.psi_coherence) + (1.0 - zn.omg_flow) + 0.25 * (zn.rho - zn.phi_g)
            scores.append(score)
        probs = softmax(np.array(scores, dtype=float), tau=0.6)
        new_zid = int(np.random.choice(nbs, p=probs))

        if new_zid == zid:
            return

        if p.faction_id is not None:
            self._remove_from_faction(pid, p.faction_id)

        self.pnjs_by_zone[zid].remove(pid)
        self.pnjs_by_zone[new_zid].append(pid)
        p.zid = new_zid

    # -----------------------------
    # Factions
    # -----------------------------
    def _faction_pressure(self, zid: int) -> Tuple[float, float, int]:
        fids = self.factions_by_zone.get(zid, [])
        if not fids:
            return 0.0, 0.0, 0
        prs = []
        cohes = []
        for fid in fids:
            if fid not in self.factions:
                continue
            f = self.factions[fid]
            infl = sum(PROFILE[self.pnjs[pid].profile]["influence"] for pid in f.members)
            prs.append(infl * f.doctrine * f.cohesion)
            cohes.append(f.cohesion)
        if not prs:
            return 0.0, 0.0, 0
        raw = sum(prs)
        pressure = math.tanh(raw / 9.0)
        return pressure, float(np.mean(cohes)), len(prs)

    def _maybe_form_faction(self, zid: int) -> None:
        if len(self.factions_by_zone[zid]) >= self.max_factions_per_zone:
            return

        z = self.zones[zid]
        flow = z.omg_flow
        coh = z.psi_coherence

        acc = self._zone_form_acc[zid]
        acc += self.faction_form_acc_rate * max(0.0, flow - self.faction_form_threshold)
        acc *= 0.985
        self._zone_form_acc[zid] = acc

        if acc < 1.0:
            return

        free = [pid for pid in self.pnjs_by_zone[zid] if self.pnjs[pid].faction_id is None]
        if len(free) < self.faction_min_size:
            return

        leader = max(
            free,
            key=lambda pid: PROFILE[self.pnjs[pid].profile]["influence"] * (0.3 + 0.7 * self.pnjs[pid].coherence),
        )
        leader_mood = self.pnjs[leader].mood

        base = clamp_signed((coh - 0.60) - 0.65 * (flow - 0.24))
        sign = 1.0 if leader_mood >= 0 else -1.0
        doctrine = clamp_signed(sign * base)

        free_sorted = sorted(free, key=lambda pid: abs(self.pnjs[pid].mood - leader_mood))
        members = set(free_sorted[: self.faction_min_size])
        members.add(leader)

        self._fid += 1
        fid = self._fid
        cohesion = self._compute_faction_cohesion(list(members))
        self.factions[fid] = MicroFaction(fid=fid, zid=zid, doctrine=doctrine, cohesion=cohesion, leader_id=leader, members=members)
        self.factions_by_zone[zid].append(fid)

        for pid in members:
            self.pnjs[pid].faction_id = fid

        self._zone_form_acc[zid] = 0.25

    def _update_factions_zone(self, zid: int) -> None:
        keep: List[int] = []
        for fid in list(self.factions_by_zone[zid]):
            if fid not in self.factions:
                continue
            f = self.factions[fid]

            f.members = {pid for pid in f.members if self.pnjs[pid].zid == zid}

            if f.leader_id not in f.members and f.members:
                f.leader_id = max(
                    list(f.members),
                    key=lambda pid: PROFILE[self.pnjs[pid].profile]["influence"] * (0.3 + 0.7 * self.pnjs[pid].coherence),
                )

            f.cohesion = self._compute_faction_cohesion(list(f.members))

            if len(f.members) < max(3, self.faction_min_size // 2) or f.cohesion < self.faction_dissolve_threshold:
                for pid in list(f.members):
                    if self.pnjs[pid].faction_id == fid:
                        self.pnjs[pid].faction_id = None
                del self.factions[fid]
            else:
                keep.append(fid)

        self.factions_by_zone[zid] = keep
        self._maybe_form_faction(zid)

    def update_factions(self) -> None:
        for zid in self.zones:
            self._update_factions_zone(zid)

    # -----------------------------
    # Feedback
    # -----------------------------
    def apply_faction_feedback(self) -> None:
        for zid, z in self.zones.items():
            pressure, avg_coh, nf = self._faction_pressure(zid)
            if nf == 0:
                continue

            if pressure > 0:
                nbs = self.neighbors[zid]
                if nbs:
                    mean_theta = math.atan2(
                        float(np.mean([math.sin(self.zones[n].theta) for n in nbs])),
                        float(np.mean([math.cos(self.zones[n].theta) for n in nbs])),
                    )
                    z.theta = wrap_pi(z.theta + self.fb_theta_align * pressure * (0.4 + 0.6 * avg_coh) * wrap_pi(mean_theta - z.theta))
                z.rho = clamp(z.rho + self.fb_rho_support * pressure * (0.4 + 0.6 * avg_coh))
            else:
                z.theta = wrap_pi(z.theta + self.fb_theta_noise * (-pressure) * (0.4 + 0.6 * avg_coh) * random.uniform(-1.0, 1.0))

        self.derive_observables()

    # -----------------------------
    # Tick
    # -----------------------------
    def tick_full(self, t: int) -> None:
        self.tick_field()
        self.spawn_events()
        self.apply_events()

        if self.pnjs:
            self.update_pnjs()
            for pid in list(self.pnjs.keys()):
                self.maybe_move_pnj(pid)
            if (t % self.faction_update_stride) == 0:
                self.update_factions()
            self.apply_faction_feedback()

    # -----------------------------
    # Plots
    # -----------------------------
    def _grid_mat(self, key: str) -> np.ndarray:
        mat = np.zeros((self.h, self.w), dtype=float)
        for zid, z in self.zones.items():
            x, y = self.xy(zid)
            mat[y, x] = float(getattr(z, key))
        return mat

    def plot_maps(self, out_prefix: str = "v5_4") -> None:
        mats = {
            "rho": self._grid_mat("rho"),
            "theta": self._grid_mat("theta"),
            "phi_g": self._grid_mat("phi_g"),
            "psi_coherence": self._grid_mat("psi_coherence"),
            "omg_flow": self._grid_mat("omg_flow"),
            "phi_stress": self._grid_mat("phi_stress"),
        }

        for name, mat in mats.items():
            plt.figure()
            plt.imshow(mat, origin="lower")
            plt.title(f"{name} map")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.colorbar()
            for y in range(self.h):
                for x in range(self.w):
                    plt.text(x, y, f"{mat[y, x]:.2f}", ha="center", va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(f"{self.out_dir}/{out_prefix}_{name}.png", dpi=160)
            plt.close()

    def plot_series(self, series: Dict[str, List[float]], out_name: str = "v5_4_series.png") -> None:
        plt.figure()
        for k, v in series.items():
            plt.plot(v, label=k)
        plt.title("V5.4 diagnostics")
        plt.xlabel("tick")
        plt.ylabel("value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/{out_name}", dpi=160)
        plt.close()


# -----------------------------
# Demo
# -----------------------------

def run_demo():
    ensure_dir("out_v5_4")

    w = WorldV54(w=4, h=3, seed=1234, out_dir="out_v5_4")
    w.init_pnjs(n_per_zone=15)

    T = 260
    series = {
        "avg_rho": [],
        "avg_phi_g": [],
        "avg_coh": [],
        "avg_flow": [],
        "avg_stress": [],
        "active_events": [],
        "avg_factions_per_zone": [],
        "avg_pressure": [],
    }

    for t in range(T):
        w.tick_full(t)

        zs = list(w.zones.values())
        series["avg_rho"].append(float(np.mean([z.rho for z in zs])))
        series["avg_phi_g"].append(float(np.mean([z.phi_g for z in zs])))
        series["avg_coh"].append(float(np.mean([z.psi_coherence for z in zs])))
        series["avg_flow"].append(float(np.mean([z.omg_flow for z in zs])))
        series["avg_stress"].append(float(np.mean([z.phi_stress for z in zs])))
        series["active_events"].append(float(np.mean([len(w.events_by_zone[zid]) for zid in w.zones])))

        nfs, prs = [], []
        for zid in w.zones:
            pz, _, nf = w._faction_pressure(zid)
            nfs.append(nf)
            prs.append(pz)
        series["avg_factions_per_zone"].append(float(np.mean(nfs)))
        series["avg_pressure"].append(float(np.mean(prs)))

        if t in (0, 40, 120, 200, T - 1):
            w.plot_maps(out_prefix=f"v5_4_t{t:04d}")

    w.plot_series(series)

    print("DONE V5.4.")
    print("Zones (final):")
    for zid, z in w.zones.items():
        print(
            zid,
            "rho", round(z.rho, 3),
            "phi_g", round(z.phi_g, 3),
            "coh", round(z.psi_coherence, 3),
            "flow", round(z.omg_flow, 3),
            "stress", round(z.phi_stress, 3),
            "events", len(w.events_by_zone[zid]),
        )

    print("PNJ", len(w.pnjs), "Factions", len(w.factions))


if __name__ == "__main__":
    run_demo()
