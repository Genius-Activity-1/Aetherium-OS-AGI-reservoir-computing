# Aetherium World-Only V5 (Phase Field)
# Cleaned + Debugged version (V5.4.5)
#
# Model: grid of zones with (rho, theta, phi_g)
# - theta: phase-like variable
# - rho: density-like variable with floor to avoid extinction
# - phi_g: gravity-like potential coupled to rho
# - psi_coherence: local circular coherence of theta
# - omg_flow: desaturated gradient proxy (tanh)
# - phi_stress: stress proxy from (phi_g, omg_flow)
#
# V5.4.4 -> V5.4.5 changes:
# - Fixed mass regulator eligibility logic so it can REMOVE mass when zones are capped at 1.0.
#   (The prior version excluded rho==1.0 from eligibility, so it could not decrease total mass.)
# - Kept deterministic per-instance RNG.
#
# Dependencies: numpy
# Run:
#   python aetherium_world_only_v5_phase_field.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# -----------------------------
# Utils
# -----------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(theta: float) -> float:
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


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
    event_count: int = 0


# -----------------------------
# World
# -----------------------------

class WorldV54:
    """Phase-field toy world.

    Determinism:
      Uses per-instance RNGs (self.rng) to remain deterministic even when worlds are ticked interleaved.

    Rho:
      - Conservative diffusion moves rho between neighbors.
      - Weak source/sink terms exist (evap, grav feed, recovery).
      - Mass regulator nudges total rho toward initial_mass to counteract floor/cap and event losses.

    NOTE on mass regulator:
      Eligibility must depend on the direction of correction:
        - If we need to REMOVE mass (step < 0), capped zones (rho==1) must be eligible.
        - If we need to ADD mass (step > 0), floored zones (rho==rho_floor) are typically ineligible.
    """

    def __init__(self, w: int = 4, h: int = 3, seed: int = 1234):
        self.w = w
        self.h = h
        self.seed = seed

        # Per-instance RNGs
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

        # Tuned parameters
        self.flow_norm = 2.3
        self.theta_thermal = 0.008

        self.ev_phase_amp = 0.5
        self.ev_vortex_amp = 0.55 * math.pi

        self.rho_floor = 0.06

        # Conservative rho diffusion rate (small)
        self.rho_diffuse_eps = 0.018

        # Weak non-conservative terms
        self.rho_evap = 0.0015
        self.rho_grav_feed = 0.006
        self.rho_recover_k = 0.006
        self.rho_recover_target = 0.35

        # Optional mass regulator (quasi-conservation helper)
        self.mass_regulator_k = 0.08  # 0 disables
        self.mass_regulator_max_step = 0.04

        # Containers
        self.zones: Dict[int, ZonePhase] = {}
        self.neighbors: Dict[int, List[int]] = {}

        self._init_world()
        self.derive_observables()

        # Diagnostics baseline
        self.initial_mass = self.total_rho()

    # ---------- geometry
    def zid(self, x: int, y: int) -> int:
        return y * self.w + x

    def xy(self, zid: int) -> tuple[int, int]:
        return zid % self.w, zid // self.w

    def _init_world(self) -> None:
        for y in range(self.h):
            for x in range(self.w):
                zid = self.zid(x, y)
                self.zones[zid] = ZonePhase(
                    rho=float(self.rng.uniform(0.4, 0.7)),
                    theta=float(self.rng.uniform(-math.pi, math.pi)),
                    phi_g=float(self.rng.uniform(0.2, 0.4)),
                )

        for zid in self.zones:
            x, y = self.xy(zid)
            nbs: List[int] = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                xx, yy = x + dx, y + dy
                if 0 <= xx < self.w and 0 <= yy < self.h:
                    nbs.append(self.zid(xx, yy))
            self.neighbors[zid] = nbs

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def total_rho(self) -> float:
        return float(sum(z.rho for z in self.zones.values()))

    # -----------------------------
    # Core dynamics
    # -----------------------------
    def update_phase(self) -> None:
        new_theta: Dict[int, float] = {}
        for zid, z in self.zones.items():
            nbs = self.neighbors[zid]
            if not nbs:
                new_theta[zid] = z.theta
                continue

            lap = sum(self.zones[n].theta - z.theta for n in nbs) / len(nbs)
            noise = self.rng.uniform(-1.0, 1.0) * self.theta_thermal * (0.25 + 0.75 * z.phi_stress)

            th = z.theta + 0.04 * lap - 0.015 * z.phi_g + noise
            new_theta[zid] = wrap_pi(th)

        for zid, th in new_theta.items():
            self.zones[zid].theta = th

    def _rho_conservative_diffusion(self) -> None:
        """Conservative transfer: symmetric pairwise exchange across edges.

        This preserves sum(rho) exactly *before* any clamping.
        """
        eps = self.rho_diffuse_eps
        if eps <= 0.0:
            return

        delta = {zid: 0.0 for zid in self.zones}

        for zid in self.zones:
            for nb in self.neighbors[zid]:
                if nb <= zid:
                    continue
                a = self.zones[zid]
                b = self.zones[nb]
                flow = eps * (a.rho - b.rho)
                delta[zid] -= flow
                delta[nb] += flow

        for zid, d in delta.items():
            self.zones[zid].rho += d

    def _mass_regulator(self, target_mass: float) -> None:
        """Gently correct total rho mass toward target_mass.

        We add/subtract a small amount distributed over zones that can move in the needed direction.
        """
        k = self.mass_regulator_k
        if k <= 0.0:
            return

        mass = self.total_rho()
        err = target_mass - mass
        if abs(err) < 1e-12:
            return

        # apply only a fraction of correction per tick, capped
        step = max(-self.mass_regulator_max_step, min(k * err, self.mass_regulator_max_step))
        if abs(step) < 1e-15:
            return

        # Eligibility depends on direction:
        #  - removing mass (step < 0): allow rho==1.0, just require above floor
        #  - adding mass (step > 0): allow rho==rho_floor, just require below cap
        eps = 1e-9
        if step < 0.0:
            eligible = [z for z in self.zones.values() if z.rho > self.rho_floor + eps]
        else:
            eligible = [z for z in self.zones.values() if z.rho < 1.0 - eps]

        if not eligible:
            return

        per = step / len(eligible)
        for z in eligible:
            z.rho = clamp(z.rho + per)
            z.rho = max(self.rho_floor, z.rho)

    def update_density(self) -> None:
        # 1) conservative diffusion first
        self._rho_conservative_diffusion()

        # 2) weak non-conservative local terms
        for zid, z in self.zones.items():
            nbs = self.neighbors[zid]
            if nbs:
                grad = sum(abs(self.zones[n].theta - z.theta) for n in nbs) / len(nbs)
            else:
                grad = 0.0

            # mild grad response
            z.rho += 0.020 * (1.0 - grad) - 0.015 * grad

            # weak evaporation + gravity feed
            z.rho -= self.rho_evap * z.rho
            z.rho += self.rho_grav_feed * z.phi_g

            # weak recovery toward target
            z.rho += self.rho_recover_k * (self.rho_recover_target - z.rho)

            z.rho = max(self.rho_floor, clamp(z.rho))

        # 3) gentle mass correction toward initial mass
        self._mass_regulator(self.initial_mass)

    def update_gravity(self, iters: int = 4) -> None:
        for _ in range(iters):
            for zid, z in self.zones.items():
                nbs = self.neighbors[zid]
                if not nbs:
                    continue
                avg = sum(self.zones[n].phi_g for n in nbs) / len(nbs)
                z.phi_g = clamp(0.6 * avg + 0.35 * z.rho)

    def derive_observables(self) -> None:
        for zid, z in self.zones.items():
            nbs = self.neighbors[zid]
            phases = [z.theta] + [self.zones[n].theta for n in nbs]

            center = z.theta
            diffs = [((p - center + math.pi) % (2.0 * math.pi)) - math.pi for p in phases]
            var = float(np.var(diffs))
            z.psi_coherence = clamp(1.0 / (1.0 + 1.35 * var))

            if nbs:
                grad = sum(abs(self.zones[n].theta - z.theta) for n in nbs) / len(nbs)
            else:
                grad = 0.0
            z.omg_flow = clamp(math.tanh(grad / self.flow_norm))

            z.phi_stress = clamp(0.5 * z.phi_g + 0.5 * z.omg_flow)

    # -----------------------------
    # Field events
    # -----------------------------
    def maybe_event(self) -> None:
        for zid, z in self.zones.items():
            if z.event_count > 8 and self.rng.random() < 0.8:
                continue

            limit = 1.0 - 0.65 * max(0.0, z.omg_flow - 0.65) / 0.35
            limit = clamp(limit, 0.35, 1.0)
            p = (0.04 + 0.15 * z.phi_stress) * limit

            if self.rng.random() > p:
                continue

            r = self.rng.random()
            if r < 0.55:
                kind = "PHASE_SHOCK"
            elif r < 0.90:
                kind = "VORTEX_DEFECT"
            else:
                kind = "DENSITY_DROP"

            z.event_count += 1

            if kind == "PHASE_SHOCK":
                z.theta = wrap_pi(z.theta + self.rng.uniform(-1.0, 1.0) * self.ev_phase_amp)

            elif kind == "DENSITY_DROP":
                # make density drop mostly conservative: move mass to neighbors (spillage)
                drop = self.rng.uniform(0.03, 0.09)
                before = z.rho
                z.rho = max(self.rho_floor, z.rho - drop)
                spilled = max(0.0, before - z.rho)

                nbs = self.neighbors.get(zid, [])
                if nbs and spilled > 0.0:
                    share = spilled / len(nbs)
                    for nb in nbs:
                        zz = self.zones[nb]
                        zz.rho = clamp(zz.rho + share)
                        zz.rho = max(self.rho_floor, zz.rho)

            elif kind == "VORTEX_DEFECT":
                z.theta = wrap_pi(z.theta + self.rng.choice([-1.0, 1.0]) * self.ev_vortex_amp)
                for nb in self.neighbors.get(zid, []):
                    self.zones[nb].theta = wrap_pi(
                        self.zones[nb].theta
                        + self.rng.choice([-1.0, 1.0]) * 0.18 * self.ev_vortex_amp
                    )

    def tick(self) -> None:
        self.update_phase()
        self.update_density()
        self.update_gravity()
        self.derive_observables()
        self.maybe_event()


# -----------------------------
# Tests
# -----------------------------

def _snapshot(w: WorldV54) -> List[tuple]:
    out: List[tuple] = []
    for zid in sorted(w.zones.keys()):
        z = w.zones[zid]
        out.append(
            (
                round(z.rho, 6),
                round(z.theta, 6),
                round(z.phi_g, 6),
                round(z.psi_coherence, 6),
                round(z.omg_flow, 6),
                round(z.phi_stress, 6),
                int(z.event_count),
            )
        )
    return out


def _run_tests() -> None:
    w0 = WorldV54(seed=1)
    assert hasattr(w0, "flow_norm")
    assert hasattr(w0, "rng")

    for _ in range(5):
        w0.tick()

    for z in w0.zones.values():
        assert 0.0 <= z.rho <= 1.0
        assert -math.pi <= z.theta <= math.pi
        assert 0.0 <= z.phi_g <= 1.0
        assert 0.0 <= z.psi_coherence <= 1.0
        assert 0.0 <= z.omg_flow <= 1.0
        assert 0.0 <= z.phi_stress <= 1.0

    # determinism: same seed should match
    w1 = WorldV54(seed=42)
    w2 = WorldV54(seed=42)
    for _ in range(10):
        w1.tick(); w2.tick()
    assert _snapshot(w1) == _snapshot(w2)

    # determinism: sequential ticking should match sequential ticking
    w3 = WorldV54(seed=99)
    w4 = WorldV54(seed=99)
    for _ in range(10):
        w3.tick()
    for _ in range(10):
        w4.tick()
    assert _snapshot(w3) == _snapshot(w4)

    # rho never below rho_floor
    w5 = WorldV54(seed=7)
    for _ in range(50):
        w5.tick()
    assert all(z.rho >= w5.rho_floor for z in w5.zones.values())

    # diffusion-only preserves total mass exactly (no sources/sinks)
    w6 = WorldV54(seed=11)
    w6.rho_evap = 0.0
    w6.rho_grav_feed = 0.0
    w6.rho_recover_k = 0.0
    w6.mass_regulator_k = 0.0

    m0 = w6.total_rho()
    for _ in range(20):
        w6.update_phase()
        w6._rho_conservative_diffusion()
    m1 = w6.total_rho()
    assert abs(m1 - m0) < 1e-9

    # mass regulator must be able to decrease total mass when above target (even if all at cap)
    w7 = WorldV54(seed=5)
    w7.mass_regulator_k = 1.0
    w7.mass_regulator_max_step = 0.25
    for z in w7.zones.values():
        z.rho = 1.0
    high = w7.total_rho()
    target = w7.initial_mass
    assert high > target
    w7._mass_regulator(target)
    after = w7.total_rho()
    assert after < high


# -----------------------------
# Demo
# -----------------------------

if __name__ == "__main__":
    _run_tests()

    w = WorldV54(seed=1234)
    for _ in range(300):
        w.tick()

    mass = w.total_rho()
    drift = (mass - w.initial_mass) / max(1e-9, w.initial_mass)

    print("DONE V5.4.5.")
    print(f"Total rho mass: {mass:.4f} | initial: {w.initial_mass:.4f} | drift: {drift*100:.2f}%")
    print("Zones (final):")
    for zid in sorted(w.zones.keys()):
        z = w.zones[zid]
        print(
            zid,
            "rho", round(z.rho, 3),
            "phi_g", round(z.phi_g, 3),
            "coh", round(z.psi_coherence, 3),
            "flow", round(z.omg_flow, 3),
            "stress", round(z.phi_stress, 3),
            "events", z.event_count,
        )
