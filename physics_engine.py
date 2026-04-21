"""
physics_engine.py
-----------------
Physics-based leak detection for three zones.

Zone A — Air intake (before turbocharger)
    Expected MAF from VE table + air density.
    If actual MAF << expected → Zone A flag.

Zone B — Charge-air path (between turbo and intake manifold)
    Expected MAP from compressor map.
    If actual MAP << expected → Zone B flag.
    Sub-location from boost_temp vs intercooler_outlet_temp delta.

Zone C — Exhaust path
    Expected EBP from regression (fuel rate + RPM).
    If actual EBP << expected → Zone C flag.
    Sub-location from EGT pair comparison.

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
    _air_density, _ve_lookup, _turbo_pressure_ratio,
)

# ─── Detection Thresholds ─────────────────────────────────────────────────────
ZONE_A_MAF_THRESHOLD_PCT = 8.0        # % residual to flag
ZONE_B_MAP_THRESHOLD_PCT = 6.0        # % residual to flag
ZONE_C_EBP_THRESHOLD_PCT = 12.0       # % residual to flag (EBP drops on leak)
EGT_DELTA_THRESHOLD_C    = 20.0       # °C difference between EGT pair for sub-location
BOOST_IC_DELTA_THRESHOLD = 8.0        # °C difference for Zone B sub-location

# ─── Drift Detection ──────────────────────────────────────────────────────────
DRIFT_WINDOW  = 50          # samples (~5 s at 100 ms)
DRIFT_SLOPE_THRESHOLD = 0.008  # residual % per sample — slow creep (true drift); sudden leaks have higher slope but still alert

# ─── Confidence mapping (residual % → confidence %) ──────────────────────────
def _pct_to_confidence(residual_pct: float, threshold_pct: float) -> float:
    """Sigmoid-like mapping: at threshold → 50%, at 3× threshold → 95%."""
    ratio = residual_pct / max(threshold_pct, 0.01)
    conf  = 100.0 / (1.0 + np.exp(-3.0 * (ratio - 1.0)))
    return round(float(np.clip(conf, 0, 100)), 1)


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


# ─── ZONE A MODULE ────────────────────────────────────────────────────────────

class ZoneADetector:
    """
    Detects air-intake leaks (pre-turbo, air filter, intake ducting).
    Uses VE table + air density to compute expected MAF.
    """

    def __init__(self):
        self._tracker = ResidualTracker()

    def _expected_maf(self, rpm: float, iat_c: float, map_kpa: float) -> float:
        """Expected MAF in g/s given current operating point."""
        ve            = _ve_lookup(rpm)
        rho           = _air_density(iat_c, map_kpa)
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

        rpm     = filt["rpm"]
        iat_c   = filt["iat_c"]
        map_kpa = filt["map_kpa"]
        actual  = filt["maf_gs"]

        expected = self._expected_maf(rpm, iat_c, map_kpa)
        if expected <= 0:
            return result

        # Residual: negative means actual is LOWER than expected (leak symptom)
        residual_pct = ((expected - actual) / expected) * 100.0
        self._tracker.push(residual_pct)

        result.expected      = round(expected, 3)
        result.actual        = round(actual, 3)
        result.residual_pct  = round(residual_pct, 2)
        result.drift         = self._tracker.is_drifting()
        result.confidence    = _pct_to_confidence(residual_pct, ZONE_A_MAF_THRESHOLD_PCT)

        # Drift is informational — never suppresses a genuine residual flag
        if residual_pct >= ZONE_A_MAF_THRESHOLD_PCT:
            result.flag         = True
            result.sub_location = "intake_duct_or_air_filter"

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

        # Drift is informational — never suppresses a genuine residual flag
        if residual_pct >= ZONE_B_MAP_THRESHOLD_PCT:
            result.flag = True
            # Sub-location: compare boost_temp and intercooler_outlet
            boost_temp = filt.get("boost_temp_c", 0)
            ic_temp    = filt.get("intercooler_outlet_c", 0)
            delta      = boost_temp - ic_temp
            if delta < BOOST_IC_DELTA_THRESHOLD:
                # Intercooler is working normally → leak is BEFORE intercooler
                result.sub_location = "before_intercooler_hose_or_turbo_outlet"
            else:
                # Intercooler not cooling much → leak is AFTER intercooler
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

    def _expected_ebp(self, fuel_gs: float, rpm: float) -> float:
        return (EBP_COEFF_FUEL * fuel_gs
                + EBP_COEFF_RPM * rpm
                + EBP_INTERCEPT)

    def run(self, filt: dict, ecu: dict) -> ZoneResult:
        result = ZoneResult(zone="C", sensor_name="ebp_kpa")

        # Edge-case suppression
        if ecu.get("dpf_regen"):
            result.suppressed = True
            result.suppression_reason = "DPF regen active — EBP naturally elevated"
            return result
        if ecu.get("coolant_temp_c", 88) < 60:
            result.suppressed = True
            result.suppression_reason = "Cold engine — EBP model not calibrated"
            return result

        fuel_gs  = filt["fuel_rate_gs"]
        rpm      = filt["rpm"]
        actual   = filt["ebp_kpa"]
        expected = self._expected_ebp(fuel_gs, rpm)

        # For Zone C, leak DROPS EBP — actual is lower than expected
        residual_pct = ((expected - actual) / max(expected, 0.1)) * 100.0
        self._tracker.push(residual_pct)

        result.expected     = round(expected, 3)
        result.actual       = round(actual, 3)
        result.residual_pct = round(residual_pct, 2)
        result.drift        = self._tracker.is_drifting()
        result.confidence   = _pct_to_confidence(residual_pct, ZONE_C_EBP_THRESHOLD_PCT)

        # Drift is informational — never suppresses a genuine residual flag
        if residual_pct >= ZONE_C_EBP_THRESHOLD_PCT:
            result.flag = True
            # Sub-location: which EGT is dropping abnormally
            egt1 = filt.get("egt_1_c", 0)
            egt2 = filt.get("egt_2_c", 0)
            delta = egt1 - egt2
            if delta > EGT_DELTA_THRESHOLD_C:
                result.sub_location = "upstream_bank_cylinder_exhaust_ports"
            elif delta < -EGT_DELTA_THRESHOLD_C:
                result.sub_location = "downstream_bank_DPF_or_catalyst"
            else:
                result.sub_location = "general_exhaust_restriction"

        return result


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class PhysicsEngine:
    """Runs all three zone detectors and returns their results."""

    def __init__(self):
        self.zone_a = ZoneADetector()
        self.zone_b = ZoneBDetector()
        self.zone_c = ZoneCDetector()

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
        }
        ra = self.zone_a.run(filt, ecu)
        rb = self.zone_b.run(filt, ecu)
        rc = self.zone_c.run(filt, ecu)
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

    print("\n=== Zone B leak 25% ===")
    sim.inject_leak("B", 0.25)
    for _ in range(30):
        raw = sim.step()
        pr  = pipe.process(raw)
        ra, rb, rc = pe.run(pr.filt, pr.raw)
    print(f"  Zone B flag: {rb.flag}  residual={rb.residual_pct:.1f}%  sub={rb.sub_location}")
    print(f"  Zone B confidence: {rb.confidence}%")
