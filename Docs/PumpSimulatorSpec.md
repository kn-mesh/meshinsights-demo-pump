# OEM Pump Demo Simulation Specification

## Goal
Simulate 90 days of 5‑minute telemetry for three pump health narratives and make their signatures easy for experts to recognize in time‑series and operating‑map plots. The simulator emits raw signals only; downstream pipelines compute derived KPIs.

---

## Use Case
- Asset: End‑suction centrifugal pump moving solvent (batch transfer; throttled discharge control).
- Cadence: 5‑minute samples for 90 days.
- Raw signals in CSV:
  - `Ps_kPa` — suction pressure (kPa). Reference ≈ 0 kPa (atmosphere). More negative ⇒ worse suction.
  - `Pd_kPa` — discharge pressure (kPa).
  - `Q_m3h` — flow (m³/h).
  - `I_A` — motor current (A).
  - Tags: `batch_id`, `recipe`.

Tag semantics
- `batch_id`: Unique identifier of a contiguous production run on a given pump (e.g., `000123_B0042`). Assigned only during scheduled active windows; idle timestamps between/around batches keep an empty `batch_id` by design.
- `recipe`: Process recipe tied to the batch window. In this demo, it maps to the targeted flow setpoint (`Q_set`) and is encoded like `R65`, `R80`, `R95`. During healthy false‑positive epochs, the recipe is updated to reflect the epoch’s shifted setpoint so stratifying by recipe explains reversible shifts. Idle periods have an empty `recipe`.

Derived KPIs (computed downstream, not in the simulator CSV):
- `dP_kPa = Pd_kPa − Ps_kPa`
  - **Definition**: differential (pump) head in kilopascals; the hydraulic pressure increase provided by the pump across the suction and discharge ports. Calculated as discharge minus suction pressure.
- `Eff = (Q_m3h · dP_kPa) / I_A`
  - **Definition**: efficiency proxy (unitless) computed as the hydraulic power proxy (flow × differential pressure) divided by electrical current. Note this is a relative proxy used for comparisons and trend detection; downstream systems should convert to consistent power units if absolute efficiency is required.

Operating pattern:
- 2–4 batches/day; each batch runs 2–6 hours.
- During a batch, control targets `Q_set ∈ {65, 80, 95}` m³/h by recipe.
- Off‑shift idle with near‑zero flow/current and small noise.

---

## Recognition Guide (matches simulator behavior)

### 1) Cavitation
Narrative: Poor suction (plugged strainer, vapor bubbles) causes efficiency loss, noisier head, and intermittent flow dips.

Visual cues:
- Suction drift: `Ps_kPa` trends downward after ≈ day 18–26. Over ~4–6 weeks, median becomes ≈ 8–15% more negative vs. pre‑event baseline.
- Noisier head: `dP_kPa` variance increases; p95–p05 band ≈ 1.5–2.0× baseline, especially during active hours.
- Intermittent events: 5–15‑minute flow dips of 5–12% at unchanged `Q_set`, accompanied by `I_A` spikes of ≈ 8–15%.
- Chronic effect: modest flow fade accumulating to ≈ 1–3% by late period. Total head decline bounded to ≈ 8–15% from pre‑cavitation baseline (not runaway).

Operating map (Q vs dP):
- Cloud shifts left/down and becomes wider; noticeably higher vertical spread at a given Q.

KPIs to validate:
- Weekly median `Ps_kPa` more negative; weekly var(`dP_kPa`) ≈ 1.5–2.0× baseline.
- `I_A / (Q · dP)` increases ≈ 10–20% vs. baseline.

### 2) Impeller Wear (Hydraulic Efficiency Loss)
Narrative: Smooth, monotonic loss of hydraulic head/efficiency with stable variance.

Visual cues:
- Suction stable: `Ps_kPa` ≈ flat.
- Head decline: `dP_kPa` decreases 5–20% over the full 90‑day window; approximate linear trend; variance ~unchanged.
- Load/flow coupling (two modes simulated):
  - Flow fades ≈ 2–8% with a smaller `I_A` creep ≈ 2–6%; or
  - Flow ≈ steady while `I_A` creeps ≈ 5–10%.

Operating map (Q vs dP):
- Cloud shifts downward roughly parallel to the OEM curve, with similar spread.

KPIs to validate:
- Weekly `dP` at `Q_set` declines steadily; efficiency proxy falls ≈ 10–25% by day 90.

### 3) Healthy (False Positive — Process/Control)
Narrative: Apparent anomalies come from recipe/ambient changes; behavior reverts when the recipe returns.

Visual cues:
- Epochal shifts: 7–14‑day epochs with `Q_set` steps of ±5–10% and corresponding `dP` shifts of ±3–6%.
- Reversibility: When the recipe toggles back, values revert; no monotonic long‑term decline.
- Attribution: Stratifying by `recipe` collapses efficiency and ratios back to on‑curve, distinguishing from true faults.

Operating map (Q vs dP):
- Multiple tight clouds on the OEM curve; aggregated view appears messy if not stratified by recipe.

KPIs to validate:
- KPIs toggle by epoch and revert; net trend across 90 days remains near baseline when recipes cycle.

---

## Data Schema
CSV columns emitted by the simulator:
```csv
timestamp_utc, pump_id, batch_id, recipe, Ps_kPa, Pd_kPa, Q_m3h, I_A
```
- Uniform 5‑minute spacing.
- Idle periods filled with small noise (typical Q≈0.05 m³/h, I≈0.1 A).
- Active/idle can be inferred from `Q_m3h` and `batch_id`.

Downstream (pipeline) should compute:
- `dP_kPa = Pd_kPa − Ps_kPa`
- `Eff = (Q_m3h · dP_kPa) / I_A`

---

## Simulation Recipe (high‑level)

OEM curve:
- `dP_expected(Q) = a − b·Q − c·Q²` (quadratic head‑flow relation).

Baseline (healthy):
- Batch schedule: 2–4/day, 2–6 h each, `Q_set ∈ {65, 80, 95}` m³/h.
- Active flow: `Q_m3h ~ N(Q_set, (0.03·Q_set)²)`; head noise ≈ 1–2% of expected.
- Suction: `Ps_kPa` ≈ 0 ± 2 kPa active; idle near 0 with tighter noise.
- Current: chosen so downstream `Eff` centers ≈ 1.0 during active periods; idle current near zero.

Class overlays (key parameters):
- Cavitation (starts ≈ day 18–26): variance factor ×1.5–2.0; bounded head decline ≈ 8–15%; chronic flow fade ≈ 1–3%; intermittent 5–12% Q dips lasting 5–15 minutes with 8–15% `I_A` spikes; suction drifts more negative over weeks.
- Impeller Wear: head declines 5–20% over 90 days with steady variance; either flow fades 2–8% with 2–6% current creep, or flow steady with 5–10% current creep.
- Healthy (FP): epochs 7–14 days; `Q_set` ±5–10%, `dP` ±3–6%; `recipe` updated for epoch; behavior reverts when recipe returns.

Labels per pump:
- `class ∈ {CAVITATION, IMPELLER_WEAR, HEALTHY_FP}`, with `class_start_utc` and `class_end_utc` bounding the simulation window.

---

## Sanity Checks (quick validation)
- Cavitation: median `Ps_kPa` more negative by ≈ 8–15% from pre‑event; var(`dP_kPa`) ≈ 1.5–2.0× baseline; intermittent Q dips with I spikes present.
- Impeller Wear: `dP` at `Q_set` ↓ 5–20% over 90 days; variance ~unchanged; current/flow creep matches one of the two modes.
- Healthy: epoch‑wise shifts visible and reversible; stratification by `recipe` normalizes KPIs; by day 90, overall trend near baseline when recipes cycle.
