"""
physics_engine.py
-----------------
Physics-based leak detection for three zones.

Zone A — Air intake (before turbocharger)
    Expected MAF from VE table + air density.
    If actual MAF << expected → Zone A flag.
    Secondary confirmation: Lambda high + N_turbo high.

Zone B — Charge-air path (between turbo and intake manifold)
    Expected MAP from compressor map.
    If actual MAP << expected → Zone B flag.
    Sub-location from boost_temp vs intercooler_outlet_temp delta.
    CASCADE RULE: If Zone B fires first and Zone C fires within
    CASCADE_WINDOW samples → Zone C suppressed as secondary EBP effect.

Zone C — Exhaust path
    Expected EBP from regression (fuel rate + RPM).
    If actual EBP << expected → Zone C flag.
    Sub-location from EGT pair comparison.
    CASCADE RULE: If Zone C fires first and Zone B fires within
    CASCADE_WINDOW samples → Zone B suppressed as secondary MAP effect.

Each module also runs:
  - Drift detection  : slow residual growth → warning, not alert
  - Edge-case filter : DPF regen, EGR, coolant, transient suppression
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

from simulator import (
    VE_TABLE_RPM, VE_TABLE_VE,
    TURBO_RPM_BREAKPOINTS, TURBO_PRESSURE_RATIO,
    EBP_COEFF_FUEL, EBP_COEFF_RPM, EBP_INTERCEPT,
    DISPLACEMENT_L, INTERCOOLER_EFFICIENCY,
    EGT_BASE_K, EGT_PER_RPM,
    N_TURBO_ENGINE_RPM, N_TURBO_SHAFT_RPM, LAMBDA_NOMINAL,
    _air_density, _ve_lookup, _turbo_pressure_ratio, _turbo_comp_temp_rise,
)

# ─── Detection Thresholds ─────────────────────────────────────────────────────
ZONE_A_MAF_THRESHOLD_PCT = 8.0        # % residual to flag
ZONE_B_MAP_THRESHOLD_PCT = 6.0        # % residual to flag
ZONE_C_EBP_THRESHOLD_PCT = 12.0       # % residual to flag (EBP drops on leak)
EGT_DELTA_THRESHOLD_C    = 8.0        # °C difference between EGT pair for sub-location
                                       # (reduced from 20 °C — median filter dampens raw signal)
# Zone B sub-location thresholds.
BOOST_IC_DELTA_THRESHOLD     = 8.0    # °C — retained for legacy compatibility
BOOST_IC_BEFORE_IC_THRESHOLD = 43.0  # °C — delta above this → leak is BEFORE intercooler

# ─── Zone A secondary confirmation thresholds ─────────────────────────────────
# Lambda and N_turbo must both deviate from nominal to confirm a Zone A flag.
ZONE_A_LAMBDA_CONFIRM_DELTA  = 0.06  # λ units above nominal → lean confirmation
ZONE_A_N_TURBO_CONFIRM_PCT   = 4.0   # % above expected N_turbo → confirm higher spin

# ─── Cascade suppression window ───────────────────────────────────────────────
# If the primary zone fires and the secondary zone fires within this many
# samples later, the secondary zone trigger is treated as a cascade consequence
# rather than an independent fault.
CASCADE_WINDOW = 20   # samples (~2 s at 100 ms/step)

# ─── Drift Detection ──────────────────────────────────────────────────────────
DRIFT_WINDOW  = 50          # samples (~5 s at 100 ms)
DRIFT_SLOPE_THRESHOLD = 0.008  # residual % per sample — slow creep


# ─── Confidence mapping (residual % → confidence %) ──────────────────────────
def _pct_to_confidence(residual_pct: float, threshold_pct: float) -> float:
    """Sigmoid-like mapping: at threshold → 50%, at 3× threshold → 95%."""
    ratio = residual_pct / max(threshold_pct, 0.01)
    conf  = 100.0 / (1.0 + np.exp(-3.0 * (ratio - 1.0)))
    return round(float(np.clip(conf, 0, 100)), 1)


def _n_turbo_expected(engine_rpm: float) -> float:
    """Expected turbocharger shaft speed [RPM] at given engine RPM."""
    return float(np.interp(engine_rpm, N_TURBO_ENGINE_RPM, N_TURBO_SHAFT_RPM))


@dataclass
class ZoneResult:
    zone:           str
    flag:           bool          = False
    residual_pct:   float         = 0.0
    expected:       float         = 0.0
    actual:         float         = 0.0
    confidence:     float         = 0.0
    sub_location:   str           = "unknown"
    drift:          bool          = False
    suppressed:     bool          = False
    suppression_reason: str       = ""
    sensor_name:    str           = ""
    # Cascade fields — populated by the PhysicsEngine orchestrator
    is_cascade:     bool          = False   # True if this flag is a secondary consequence
    cascade_effect: str           = ""      # e.g. "EBP_secondary_from_B"
    confirmed_by:   str           = ""      # e.g. "lambda_high+n_turbo_high"


class ResidualTracker:
    """Tracks a sliding window of residuals for drift detection."""

    def __init__(self, window: int = DRIFT_WINDOW):
        self._buf: deque = deque(maxlen=window)

    def push(self, residual: float):
        self._buf.append(residual)

    def is_drifting(self) -> bool:
        if len(self._buf) < self._buf.maxlen:
            return False
        xs = np.arange(len(self._buf))
        ys = np.array(self._buf)
        coeffs = np.polyfit(xs, ys, 1)
        slope = abs(coeffs[0])
        return slope >= DRIFT_SLOPE_THRESHOLD


class CausalTracker:
    """
    Tracks the sample index at which each zone first flagged within the
    current 'active' event window.  Used to determine which zone fired
    FIRST so that subsequent zone triggers can be identified as cascade
    consequences rather than independent faults.

    Design notes
    ------------
    - Recording only happens when a zone transitions False → True.
    - A flag is cleared (reset) when the zone returns to unflagged for at
      least CASCADE_WINDOW consecutive samples, so a new genuine fault in
      the same zone after a clear is treated as a fresh event.
    - `is_cascade(cause, effect)` returns True when:
        • cause zone fired before (or same sample as) effect zone, AND
        • the delay between them is ≤ CASCADE_WINDOW samples.
    """

    def __init__(self, window: int = CASCADE_WINDOW):
        self._window = window
        # sample index when zone first fired (-1 = not currently flagged)
        self._first_flag: dict = {"A": -1, "B": -1, "C": -1}
        # track previous flag state to detect transitions
        self._prev_flag: dict = {"A": False, "B": False, "C": False}
        # consecutive unflagged count per zone (for reset)
        self._unflagged_count: dict = {"A": 0, "B": 0, "C": 0}
        self._sample: int = 0

    def update(self, zone: str, flagged: bool):
        """Call once per sample per zone, BEFORE calling is_cascade()."""
        if flagged:
            self._unflagged_count[zone] = 0
            # Record the first sample at which this zone flags in this event
            if self._first_flag[zone] == -1:
                self._first_flag[zone] = self._sample
        else:
            self._unflagged_count[zone] += 1
            # Reset if unflagged for long enough → new event window
            if self._unflagged_count[zone] >= self._window:
                self._first_flag[zone] = -1
        self._prev_flag[zone] = flagged

    def tick(self):
        """Advance the sample counter. Call once per full update cycle."""
        self._sample += 1

    def is_cascade(self, cause_zone: str, effect_zone: str) -> bool:
        """
        Returns True if `cause_zone` fired before `effect_zone` AND the
        gap between them is within CASCADE_WINDOW samples.
        Both zones must currently be flagged (first_flag != -1).
        """
        t_cause  = self._first_flag[cause_zone]
        t_effect = self._first_flag[effect_zone]
        if t_cause == -1 or t_effect == -1:
            return False
        gap = t_effect - t_cause
        # cause must fire first (gap >= 0) and within window
        return 0 <= gap <= self._window


# ─── ZONE A MODULE ────────────────────────────────────────────────────────────

class ZoneADetector:
    """
    Detects air-intake leaks (pre-turbo, air filter, intake ducting).
    Uses VE table + air density to compute expected MAF.
    Secondary confirmation via Lambda (lean shift) and N_turbo (higher spin).
    """

    def __init__(self):
        self._tracker = ResidualTracker()

    def _expected_maf(self, rpm: float, intercooler_outlet_c: float, map_kpa: float) -> float:
        """Expected MAF in g/s given current operating point.

        Uses intercooler_outlet_c (not iat_c) for air density — this matches the
        simulator physics exactly and eliminates RPM-dependent bias that arose when
        iat_c (post-sensor noise) was used instead.
        """
        ve            = _ve_lookup(rpm)
        rho           = _air_density(intercooler_outlet_c, map_kpa)
        vol_flow_m3s  = (DISPLACEMENT_L / 1000.0) * (rpm / 60.0) / 2.0
        vol_flow_m3s *= ve
        return vol_flow_m3s * rho * 1000.0    # g/s

    def run(self, filt: dict, ecu: dict) -> ZoneResult:
        result = ZoneResult(zone="A", sensor_name="maf_gs")

        # Edge-case suppression
        if ecu.get("transient"):
            result.suppressed = True
            result.suppression_reason = "Engine transient — MAF unstable"
            return result
        if ecu.get("egr_pct", 0) > 40:
            result.suppressed = True
            result.suppression_reason = "High EGR — bypasses MAF path"
            return result

        rpm                  = filt["rpm"]
        intercooler_outlet_c = filt["intercooler_outlet_c"]
        map_kpa              = filt["map_kpa"]
        actual               = filt["maf_gs"]

        expected = self._expected_maf(rpm, intercooler_outlet_c, map_kpa)
        if expected <= 0:
            return result

        # Residual: positive means actual is LOWER than expected (leak symptom)
        residual_pct = ((expected - actual) / expected) * 100.0
        self._tracker.push(residual_pct)

        result.expected      = round(expected, 3)
        result.actual        = round(actual, 3)
        result.residual_pct  = round(residual_pct, 2)
        result.drift         = self._tracker.is_drifting()
        result.confidence    = _pct_to_confidence(residual_pct, ZONE_A_MAF_THRESHOLD_PCT)

        if residual_pct >= ZONE_A_MAF_THRESHOLD_PCT:
            result.flag = True
            result.sub_location = "pre_turbo_leak"

            # ── Secondary confirmation via Lambda and N_turbo ─────────────────
            # Zone A: unmetered air makes mixture lean (lambda rises) and the
            # turbo spins harder to pull air through the restricted path.
            lambda_actual   = filt.get("lambda_ratio", LAMBDA_NOMINAL)
            n_turbo_actual  = filt.get("n_turbo_rpm", 0.0)
            n_turbo_expect  = _n_turbo_expected(rpm)

            lambda_high  = (lambda_actual - LAMBDA_NOMINAL) >= ZONE_A_LAMBDA_CONFIRM_DELTA
            n_turbo_high = (n_turbo_actual > n_turbo_expect * (1.0 + ZONE_A_N_TURBO_CONFIRM_PCT / 100.0))

            if lambda_high and n_turbo_high:
                result.confirmed_by = "lambda_high+n_turbo_high"
                # Boost confidence slightly when both secondary signals agree
                result.confidence = min(100.0, result.confidence + 8.0)
            elif lambda_high:
                result.confirmed_by = "lambda_high"
                result.confidence = min(100.0, result.confidence + 4.0)
            elif n_turbo_high:
                result.confirmed_by = "n_turbo_high"
                result.confidence = min(100.0, result.confidence + 4.0)

        return result


# ─── ZONE B MODULE ────────────────────────────────────────────────────────────

class ZoneBDetector:
    """
    Detects charge-air leaks (turbo outlet → intercooler → intake manifold).
    Uses compressor map to compute expected MAP.
    Sub-location from boost_temp vs intercooler_outlet_temp delta.
    """

    def __init__(self):
        self._tracker = ResidualTracker()

    def _expected_map(self, rpm: float, ambient_kpa: float = 101.325) -> float:
        pr = _turbo_pressure_ratio(rpm)
        return pr * ambient_kpa

    def run(self, filt: dict, ecu: dict) -> ZoneResult:
        result = ZoneResult(zone="B", sensor_name="map_kpa")

        # Edge-case suppression
        if ecu.get("transient"):
            result.suppressed = True
            result.suppression_reason = "Engine transient — boost unstable"
            return result
        if ecu.get("coolant_temp_c", 88) < 60:
            result.suppressed = True
            result.suppression_reason = "Cold engine — turbo not at operating point"
            return result

        rpm     = filt["rpm"]
        actual  = filt["map_kpa"]
        expected = self._expected_map(rpm)

        residual_pct = ((expected - actual) / expected) * 100.0
        self._tracker.push(residual_pct)

        result.expected     = round(expected, 3)
        result.actual       = round(actual, 3)
        result.residual_pct = round(residual_pct, 2)
        result.drift        = self._tracker.is_drifting()
        result.confidence   = _pct_to_confidence(residual_pct, ZONE_B_MAP_THRESHOLD_PCT)

        if residual_pct >= ZONE_B_MAP_THRESHOLD_PCT:
            result.flag = True
            # Sub-location via boost→intercooler_outlet temperature delta
            boost_temp = filt.get("boost_temp_c", 0.0)
            ic_temp    = filt.get("intercooler_outlet_c", 0.0)
            temp_delta = boost_temp - ic_temp

            expected_comp_temp_rise = _turbo_comp_temp_rise(rpm)
            expected_baseline_delta = expected_comp_temp_rise * INTERCOOLER_EFFICIENCY
            dynamic_threshold = expected_baseline_delta + 5.0

            if temp_delta > dynamic_threshold:
                result.sub_location = "before_intercooler_hose_or_turbo_outlet"
            else:
                result.sub_location = "after_intercooler_hose_or_clamp"

        return result


# ─── ZONE C MODULE ────────────────────────────────────────────────────────────

class ZoneCDetector:
    """
    Detects exhaust path leaks (cracked manifold, blown gasket, loose connection).
    Uses fuel-rate + RPM regression to compute expected EBP.
    Sub-location from EGT pair comparison.
    """

    def __init__(self):
        self._tracker = ResidualTracker()

    def _expected_ebp(self, fuel_gs: float, rpm: float, dpf_regen: bool = False) -> float:
        """Expected EBP in kPa.

        When DPF regen is active the filter is being actively regenerated and
        exhaust backpressure rises naturally by ~35 %.  The model must account
        for this so the residual stays near zero during regen events and does
        not produce a false Zone C flag.
        """
        base = (EBP_COEFF_FUEL * fuel_gs
                + EBP_COEFF_RPM  * rpm
                + EBP_INTERCEPT)
        if dpf_regen:
            base *= 1.35   # mirrors the simulator's regen multiplier
        return base

    def run(self, filt: dict, ecu: dict) -> ZoneResult:
        result = ZoneResult(zone="C", sensor_name="ebp_kpa")

        # ── Edge-case suppression ─────────────────────────────────────────────
        dpf_regen = bool(ecu.get("dpf_regen", False))
        if dpf_regen:
            result.suppressed = True
            result.suppression_reason = "DPF regen active — EBP naturally elevated; Zone C suppressed"
            return result
        if ecu.get("coolant_temp_c", 88) < 60:
            result.suppressed = True
            result.suppression_reason = "Cold engine — EBP model not calibrated"
            return result

        fuel_gs  = filt["fuel_rate_gs"]
        rpm      = filt["rpm"]
        actual   = filt["ebp_kpa"]
        expected = self._expected_ebp(fuel_gs, rpm, dpf_regen=False)

        # For Zone C, leak DROPS EBP — actual is lower than expected
        residual_pct = ((expected - actual) / max(expected, 0.1)) * 100.0
        self._tracker.push(residual_pct)

        result.expected     = round(expected, 3)
        result.actual       = round(actual, 3)
        result.residual_pct = round(residual_pct, 2)
        result.drift        = self._tracker.is_drifting()
        result.confidence   = _pct_to_confidence(residual_pct, ZONE_C_EBP_THRESHOLD_PCT)

        if residual_pct >= ZONE_C_EBP_THRESHOLD_PCT:
            result.flag = True

            egt1  = filt.get("egt_1_c", 0.0)
            egt2  = filt.get("egt_2_c", 0.0)
            egt3  = filt.get("egt_3_c", 0.0)
            egt4  = filt.get("egt_4_c", 0.0)
            egt5  = filt.get("egt_5_c", 0.0)

            load = filt.get("load_pct", 0.0)
            egt_base_expect = EGT_BASE_K + EGT_PER_RPM * (rpm - 1000) + (load / 100.0) * 150 - 273.15

            drop_12 = egt1 - egt2    # normally ~150°C
            drop_23 = egt2 - egt3    # normally ~70°C
            drop_34 = egt3 - egt4    # normally ~80°C
            drop_45 = egt4 - egt5    # normally ~60°C

            if egt1 < (egt_base_expect - 15):
                result.sub_location = "exhaust_manifold"
            elif drop_12 > (150 + 15):
                result.sub_location = "between_manifold_and_turbine"
            elif drop_23 > (70 + 15):
                result.sub_location = "between_turbine_and_doc"
            elif drop_34 > (80 + 15):
                result.sub_location = "between_doc_and_dpf"
            elif drop_45 > (60 + 15):
                result.sub_location = "between_dpf_and_scr"
            else:
                result.sub_location = "general_exhaust_restriction"

        return result


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class PhysicsEngine:
    """
    Runs all three zone detectors and applies cascade suppression rules.

    Cascade Rules
    -------------
    Rule 1 — Zone B fires unambiguously first (gap > 0):
        Zone C EBP drop is a secondary consequence (less fuel burned →
        less exhaust mass → lower EBP). Zone C flag is suppressed.

    Rule 2 — Zone C fires unambiguously first (gap > 0):
        Zone B MAP drop is a secondary consequence (turbo loses exhaust
        energy → less boost → MAP drops). Zone B flag is suppressed.

    Rule 3 — Simultaneous (gap == 0, both cross threshold same sample):
        This happens when coupling is instantaneous in the simulator.
        Tie-break by NORMALIZED RESIDUAL: whichever zone's residual is
        larger relative to its own detection threshold is the root cause.
        Example: Zone C 0.65 → EBP residual 38.6% / 12% = 3.2 vs
                              MAP residual  7.8%  /  6% = 1.3 → C wins.

    Rule 4 — Neither fits window:
        Both flags preserved as independent faults.
    """

    def __init__(self):
        self.zone_a = ZoneADetector()
        self.zone_b = ZoneBDetector()
        self.zone_c = ZoneCDetector()
        self._causal = CausalTracker(window=CASCADE_WINDOW)

    def run(self, filt: dict, raw: dict) -> Tuple[ZoneResult, ZoneResult, ZoneResult]:
        """
        filt: filtered sensor values from pipeline
        raw:  original row (contains ECU flags)
        Returns: (zone_a_result, zone_b_result, zone_c_result)
        """
        ecu = {
            "dpf_regen":     bool(raw.get("dpf_regen", 0)),
            "egr_pct":       raw.get("egr_pct", 15.0),
            "coolant_temp_c": raw.get("coolant_temp_c", 88.0),
            "transient":     bool(raw.get("transient", 0)),
            "load_pct":      raw.get("load_pct", 50.0),
        }

        # ── 1. Run all three detectors independently ──────────────────────────
        ra = self.zone_a.run(filt, ecu)
        rb = self.zone_b.run(filt, ecu)
        rc = self.zone_c.run(filt, ecu)

        # ── 2. Update causal tracker ──────────────────────────────────────────
        # Suppressed zones (DPF regen, cold engine, EGR) are not genuine flags
        # and must not participate in causality ordering.
        self._causal.update("A", ra.flag and not ra.suppressed)
        self._causal.update("B", rb.flag and not rb.suppressed)
        self._causal.update("C", rc.flag and not rc.suppressed)

        # ── 3. Apply cascade suppression ─────────────────────────────────────
        b_active = rb.flag and not rb.suppressed
        c_active = rc.flag and not rc.suppressed

        if b_active and c_active:
            b_to_c = self._causal.is_cascade(cause_zone="B", effect_zone="C")
            c_to_b = self._causal.is_cascade(cause_zone="C", effect_zone="B")

            if b_to_c and not c_to_b:
                # ── Rule 1: Zone B unambiguously fired BEFORE Zone C ──────────
                rc.flag            = False
                rc.is_cascade      = True
                rc.cascade_effect  = "EBP_secondary_from_B"
                rc.suppressed      = True
                rc.suppression_reason = (
                    "Zone C EBP drop is a secondary cascade consequence of Zone B leak "
                    "(less fuel burned → less exhaust mass → lower EBP). "
                    "Zone B flagged first."
                )
                rb.cascade_effect = "secondary_EBP_drop_noted_in_Zone_C"

            elif c_to_b and not b_to_c:
                # ── Rule 2: Zone C unambiguously fired BEFORE Zone B ──────────
                rb.flag            = False
                rb.is_cascade      = True
                rb.cascade_effect  = "MAP_secondary_from_C"
                rb.suppressed      = True
                rb.suppression_reason = (
                    "Zone B MAP drop is a secondary cascade consequence of Zone C leak "
                    "(turbo loses exhaust energy → less boost → MAP drops). "
                    "Zone C flagged first."
                )
                rc.cascade_effect = "secondary_MAP_drop_noted_in_Zone_B"

            elif b_to_c and c_to_b:
                # ── Rule 3: Simultaneous flag (gap == 0) ─────────────────────
                b_ratio = rb.residual_pct / max(ZONE_B_MAP_THRESHOLD_PCT, 0.01)
                c_ratio = rc.residual_pct / max(ZONE_C_EBP_THRESHOLD_PCT, 0.01)

                if c_ratio > b_ratio:
                    # Zone C is the root cause
                    rb.flag            = False
                    rb.is_cascade      = True
                    rb.cascade_effect  = "MAP_secondary_from_C"
                    rb.suppressed      = True
                    rb.suppression_reason = (
                        "Zone B MAP drop is a secondary cascade consequence of Zone C leak "
                        "(turbo loses exhaust energy → less boost → MAP drops). "
                        "Simultaneous flags: Zone C normalized residual "
                        f"({c_ratio:.1f}x) exceeds Zone B ({b_ratio:.1f}x)."
                    )
                    rc.cascade_effect = "secondary_MAP_drop_noted_in_Zone_B"
                else:
                    # Zone B is the root cause
                    rc.flag            = False
                    rc.is_cascade      = True
                    rc.cascade_effect  = "EBP_secondary_from_B"
                    rc.suppressed      = True
                    rc.suppression_reason = (
                        "Zone C EBP drop is a secondary cascade consequence of Zone B leak "
                        "(less fuel burned → less exhaust mass → lower EBP). "
                        "Simultaneous flags: Zone B normalized residual "
                        f"({b_ratio:.1f}x) exceeds Zone C ({c_ratio:.1f}x)."
                    )
                    rb.cascade_effect = "secondary_EBP_drop_noted_in_Zone_C"
            # else: both flags outside window → independent faults, keep both

        # ── 4. Advance sample counter ─────────────────────────────────────────
        self._causal.tick()

        return ra, rb, rc


# ─── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from simulator import EngineSimulator
    from pipeline  import DataPipeline

    sim  = EngineSimulator(2000, 60)
    pipe = DataPipeline()
    pe   = PhysicsEngine()

    # Warm up steady state
    for _ in range(110):
        pipe.process(sim.step())

    print("=== Healthy mode ===")
    for _ in range(5):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
    print(f"  Zone A flag: {ra.flag}  residual={ra.residual_pct:.1f}%")
    print(f"  Zone B flag: {rb.flag}  residual={rb.residual_pct:.1f}%")
    print(f"  Zone C flag: {rc.flag}  residual={rc.residual_pct:.1f}%")
    print(f"  Lambda:   {pr.filt.get('lambda_ratio', 0):.4f}")
    print(f"  N_turbo:  {pr.filt.get('n_turbo_rpm', 0):.0f} RPM")

    print("\n=== Zone A leak 30% ===")
    sim.inject_leak("A", 0.30)
    for _ in range(40):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
    print(f"  Zone A flag: {ra.flag}  residual={ra.residual_pct:.1f}%  confirmed_by={ra.confirmed_by!r}")
    print(f"  Zone B flag: {rb.flag}  residual={rb.residual_pct:.1f}%")
    print(f"  Zone C flag: {rc.flag}  residual={rc.residual_pct:.1f}%")
    print(f"  Lambda:   {pr.filt.get('lambda_ratio', 0):.4f}  (expect > {LAMBDA_NOMINAL:.2f})")
    print(f"  N_turbo:  {pr.filt.get('n_turbo_rpm', 0):.0f} RPM")
    sim.clear_leak()
    pe = PhysicsEngine()

    print("\n=== Zone B leak 30% (expect Zone C cascade-suppressed) ===")
    for _ in range(50):
        pipe.process(sim.step())
    sim.inject_leak("B", 0.30)
    for _ in range(40):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
    print(f"  Zone B flag: {rb.flag}  residual={rb.residual_pct:.1f}%  sub={rb.sub_location}")
    print(f"  Zone C flag: {rc.flag}  is_cascade={rc.is_cascade}  reason={rc.suppression_reason[:60]!r}")
    print(f"  Zone B cascade_effect: {rb.cascade_effect!r}")
    sim.clear_leak()
    pe = PhysicsEngine()

    print("\n=== Zone C leak 30% (expect Zone B cascade-suppressed) ===")
    for _ in range(50):
        pipe.process(sim.step())
    sim.inject_leak("C", 0.30)
    for _ in range(40):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
    print(f"  Zone C flag: {rc.flag}  residual={rc.residual_pct:.1f}%  sub={rc.sub_location}")
    print(f"  Zone B flag: {rb.flag}  is_cascade={rb.is_cascade}  reason={rb.suppression_reason[:60]!r}")
    print(f"  Zone C cascade_effect: {rc.cascade_effect!r}")
