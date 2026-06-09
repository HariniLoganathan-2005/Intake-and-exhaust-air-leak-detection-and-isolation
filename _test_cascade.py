import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from simulator import EngineSimulator
from pipeline import DataPipeline
from physics_engine import PhysicsEngine, ZONE_B_MAP_THRESHOLD_PCT, ZONE_C_EBP_THRESHOLD_PCT
from ml_engine import MLEngine, MLResult
from fusion import fuse

print("=== EXACT SCENARIO FROM DASHBOARD: Zone C 0.65, location=manifold_to_turbine ===")
sim  = EngineSimulator(2000, 60)
pipe = DataPipeline()
pe   = PhysicsEngine()
ml   = MLEngine()
try:
    ml.load(); ml_ok = True
except Exception:
    ml_ok = False

for _ in range(110):
    pipe.process(sim.step())

sim.inject_leak("C", 0.65, c_location="manifold_to_turbine")

for _ in range(40):
    raw = sim.step()
    pr  = pipe.process(raw)
    ra, rb, rc = pe.run(pr.filt, pr.raw)
    ml_r = ml.run(pr.filt) if ml_ok else MLResult()
    d = fuse(ra, rb, rc, ml_r, raw['timestamp'])

print(f"Zone A: flag={ra.flag}  res={ra.residual_pct:.1f}%")
print(f"Zone B: flag={rb.flag}  res={rb.residual_pct:.1f}%  is_cascade={rb.is_cascade}")
print(f"Zone C: flag={rc.flag}  res={rc.residual_pct:.1f}%  is_cascade={rc.is_cascade}")
b_ratio = rb.residual_pct / ZONE_B_MAP_THRESHOLD_PCT
c_ratio = rc.residual_pct / ZONE_C_EBP_THRESHOLD_PCT
print(f"Normalized ratios: B={b_ratio:.2f}x  C={c_ratio:.2f}x  (higher = root cause)")
print()
print(f"DECISION: status={d.status}  zone={d.zone}  sub={d.sub_location}")
print(f"  cascade_zones={d.cascade_zones}")
print(f"  confirmed_by={d.confirmed_by}")
print(f"  action_has_cascade_note={'[Note:' in d.action}")
print()

# Also test a range of Zone C severities
print("=== Zone C severities: when does secondary MAP cross Zone B threshold? ===")
for sev in [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90]:
    sim2  = EngineSimulator(2000, 60)
    pipe2 = DataPipeline()
    pe2   = PhysicsEngine()
    for _ in range(110):
        pipe2.process(sim2.step())
    sim2.inject_leak("C", sev, c_location="manifold_to_turbine")
    for _ in range(40):
        raw2 = sim2.step()
        pr2  = pipe2.process(raw2)
        ra2, rb2, rc2 = pe2.run(pr2.filt, pr2.raw)
    d2 = fuse(ra2, rb2, rc2, MLResult(), raw2['timestamp'])
    print(f"  sev={sev:.2f}: C_res={rc2.residual_pct:.1f}%  B_res={rb2.residual_pct:.1f}%  "
          f"detected_zone={d2.zone}  B_cascade={rb2.is_cascade}")
