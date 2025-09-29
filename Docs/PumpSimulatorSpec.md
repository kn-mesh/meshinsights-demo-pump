# OEM Pump Demo Simulation Specification

## Goal
Simulate **90 days of telemetry data** (data every 5 minutes) using **Python** for **three types of devices**, each type corresponding to one of the three classification categories below.  
The simulated dataset must allow us to demonstrate how MeshInsights can outperform traditional alarm engines by classifying machine states more accurately and triggering precise actions.

---

## Use case
- **Asset:** End-suction centrifugal pump moving solvent from storage to reactor (chemical batch transfer; throttled discharge control).
- **Cadence:** 5-minute rollups for **90 days**.
- **Signals (raw & derived):**
  - `Ps_kPa` (suction pressure, kPa; reference ~0 = atmosphere; more negative = worse suction).
  - `Pd_kPa` (discharge pressure, kPa).
  - `Q_m3h` (flow, m³/h).
  - `I_A` (motor current, A).
  - **Derived:** `dP_kPa = Pd_kPa - Ps_kPa`; **efficiency proxy** `Eff = (Q_m3h * dP_kPa) / I_A`.
  - Optional tags: `batch_id`, `recipe`, `valve_pos_pct`, `site_temp_C`.

- **Operating pattern:**
  - 2–4 batches/day; each batch runs 2–6 hours.
  - During a batch, the control loop targets a flow setpoint `Q_set` (varies by recipe: e.g., 65, 80, 95 m³/h).
  - Off-shift = pump mostly idle (zero or tiny recirculation flow).

---

## Classifications (device types)

### 1. Cavitation
Previously called “Suction Restriction / Emerging Cavitation.”  
This is simplified to **Cavitation** for clarity. The device simulates cavitation-like behavior as suction conditions deteriorate.

- **Narrative:** Pump struggles to pull fluid due to poor suction (plugged strainer, vapor bubbles, etc.).
- **Data characteristics (90 days):**
  - `Ps_kPa`: downward drift during active hours (median −8–15% vs baseline over 4–6 weeks).
  - `dP_kPa`: declining median; **variance increases** (p95–p05 band 1.5–2× baseline).
  - `Q_m3h`: short dips (5–15 min) at same `Q_set`.
  - `I_A`: p95 increases; spikes when Q dips.
- **Operating map (Q vs dP):** cloud shifts left/down and widens.
- **Weekly rollups:** median(Ps) ↓; var(dP) ↑; I/Q ↑ 10–20%.

### 2. Impeller Wear / Hydraulic Efficiency Loss
- **Narrative:** Impeller wears down over time; smooth, monotonic efficiency decline.
- **Data characteristics (90 days):**
  - `Ps_kPa`: stable.
  - `Pd_kPa`: gradual decline (−0.5 to −1.5 kPa/week).
  - `dP_kPa`: 5–20% drop over 90 days; variance unchanged.
  - `Q_m3h`: slightly lower (2–10%) or flat while I_A creeps up (5–10%).
- **Operating map (Q vs dP):** cloud shifts downward parallel to OEM band.
- **Weekly rollups:** dP@Q_set ↓ steadily; Eff ↓ 10–25%.

### 3. Healthy (False Positive – Process/Control)
- **Narrative:** Apparent anomalies caused by recipe/ambient changes, not faults.
- **Data characteristics (90 days):**
  - Epochal shifts tied to recipe or ambient changes.
  - No monotonic decline; values revert when recipe returns.
  - `I_A` tracks load; ratios normalize when stratified by recipe.
- **Operating map (Q vs dP):** multiple tight clouds on-curve, messy if aggregated.
- **Weekly rollups:** toggles by epoch; U-shaped KPIs tied to recipe/ambient.

---

## Data schema
```csv
timestamp_utc, pump_id, batch_id, recipe, Ps_kPa, Pd_kPa, Q_m3h, I_A, dP_kPa, Eff
```
- Uniform 5-min spacing (fill idle with Q≈0, small noise).
- Active/idle inferred from Q.

---

## Simulation recipe

### Global constants
- OEM band: dP_expected(Q) = a − bQ − cQ².
- Baseline ranges:
  - Q_set ∈ {65, 80, 95} m³/h.
  - Baseline dP_expected(Q_set) ≈ 250–300 kPa.
  - Ps_active ≈ −5 to +5 kPa.
  - I_A tuned so Eff ≈ 0.9–1.1.

### Steps
1. Generate calendar: 90 days × 5-min, with 2–4 batches/day.
2. Baseline (Healthy):
   - Q_m3h ~ N(Q_set, (0.03·Q_set)²).
   - dP ~ dP_expected(Q) + ε, ε ~ N(0, σ²) with σ ≈ 1–2% of expected.
   - Ps ~ 0 ± 2; Pd = dP + Ps.
   - I_A set to yield Eff ~ 0.95–1.05.
   - Idle: Q≈0, dP≈0, I≈0 (small noise).
3. Overlay per class:
   - **Cavitation:** start day ~20, Ps drift −0.2 to −0.5 kPa/day, dP variance ×1.5–2.0, occasional Q dips (−5–10%) with I spikes (+8–15%).
   - **Impeller Wear:** dP decline 5–20% over 90d (linear), variance steady, I_A increase 5–10% (or Q drop 2–10%).
   - **Healthy:** create epochs (7–14 days) with Q_set ±5–10%, dP shift +3–6%, revert after epoch.
4. Rollups: compute min/max/p05/p95 within windows if needed.
5. Labels: class ∈ {CAVITATION, IMPELLER_WEAR, HEALTHY_FP} per pump or period.

---

## Label schema
```csv
pump_id, class, class_start_utc, class_end_utc
```

---

## Sanity checks
- **Cavitation:** Ps median ↓ 8–15% by day 60; var(dP) 1.5–2.0× baseline.
- **Impeller Wear:** dP@Q_set ↓ 5–20% over 90 days; variance ~1.0.
- **Healthy:** per-epoch shifts but overall return to baseline by day 90.
