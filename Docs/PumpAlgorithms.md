# Pump Algorithms — Simplified (3 KPIs Max, No Smoothing/Active Mask/Residuals)

Goal: Keep it valuable but simple based on the simulator. Use at most three high‑value KPIs, shared across types where possible, computed directly from raw 5‑minute data. No active/idle masks, no smoothing, no operating‑map residuals.

Inputs (per row)
- `timestamp_utc, pump_id, batch_id, recipe, Ps_kPa, Pd_kPa, Q_m3h, I_A`
- Derived: `dP_kPa = Pd_kPa − Ps_kPa`

Scoping choices (simple mode)
- Stratify by `pump_id` and `recipe`. Idle rows naturally drop because idle has an empty `recipe` in the simulator.
- Aggregation window: calendar week is the primary unit; use the first 14 days per recipe as the baseline period.

---

## Universal KPIs (used for all types)
These three KPIs are computed per pump × recipe × week and interpreted differently for each class. They require only raw fields and `dP_kPa`.

1) Head Spread Ratio (HNI)
- What: Relative head noise (spread) this week vs baseline.
- Calc: `HNI_w = (p95(dP_kPa)_w − p05(dP_kPa)_w) / (p95(dP_kPa)_baseline − p05(dP_kPa)_baseline)`
- Value: Cavitation widens the head band ≈1.5–2.0×; wear keeps spread about the same; healthy stays near baseline.
- Timeframe: weekly; baseline is first 14 days per recipe. If baseline spread < small epsilon (e.g., 1 kPa), clamp denominator to epsilon.

2) Head Trend Slope (HTS)
- What: Long‑term change in head level.
- Calc: For each recipe, compute weekly median `dP_kPa`. Trend = percent change from baseline to last week or the OLS slope converted to 90‑day percent. Report `HTS_%` (negative = decline).
- Value: Wear manifests as a smooth 5–20% decline; cavitation has limited monotonic decline; healthy reverts with recipe changes, net small trend.
- Timeframe: weekly medians across 90 days; baseline = first 14 days per recipe.

3) Intermittent Event Rate (IER)
- What: Rate of short flow dips with coincident current spikes (5–15 minutes) using raw 5‑minute data.
- Calc per pump × recipe × week:
  - Let `Q_med_w = median(Q_m3h in week, same recipe)` and `I_med_w = median(I_A in week, same recipe)`.
  - A “dip” is two consecutive rows with `Q_m3h ≤ 0.95 * Q_med_w`.
  - A “confirmed event” is a dip where either of the two rows has `I_A ≥ 1.08 * I_med_w`.
  - Count unique events (collapse overlapping sequences) and compute `IER = events / hours_in_recipe_week`, where `hours_in_recipe_week = (#rows for recipe that week) * 5 / 60`.
- Value: Cavitation produces intermittent dips with load spikes; wear does not; healthy remains low.
- Timeframe: 5‑minute samples, aggregated weekly.

---

## Per‑Type Use (max 3 KPIs each)

Cavitation
- HNI: flag if `HNI ≥ 1.5` on any week.
- IER: flag if `IER ≥ 0.2 events/hour` on any of two distinct weeks.
- HTS: supportive only; expect small to moderate decline (typically > −15% bounded). Do not use as a primary flag.

Impeller Wear
- HTS: flag wear if `HTS_% ≤ −5%` total over 90 days (per recipe), ideally monotonic or mostly negative weeks.
- HNI: require stability `HNI ≤ 1.2` on most weeks (to rule out cavitation).
- IER: require low events `IER < 0.1 events/hour` on most weeks.

Healthy (False Positive)
- HTS: healthy if `|HTS_%| < 3%` across recipes (no net decline).
- HNI: `HNI ≤ 1.2` on most weeks.
- IER: `IER < 0.1 events/hour` on most weeks.

Interpretation note
- Recipes: Since simulator encodes setpoint changes as `recipe`, aggregate and decide per recipe, then take the majority/median decision across recipes for the pump. Recipe shifts that revert will keep HTS small and HNI/IER low → Healthy.

---

## Computation Steps (minimal)
1) Compute `dP_kPa = Pd_kPa − Ps_kPa` per row.
2) Group by `pump_id` × `recipe` × calendar week.
3) For each group, compute: weekly median `dP_kPa`, `Q_med_w`, `I_med_w`, weekly `p95(dP_kPa)` and `p05(dP_kPa)`, `#rows`.
4) Event scan within the group to count `IER` using the thresholds above.
5) Baseline per recipe = first 14 days of data for that recipe. Store baseline spread `(p95−p05)` and baseline median `dP_kPa`.
6) Compute `HNI` and `HTS_%` per recipe; summarize to pump‑level by median across recipes.

---

## Thresholds Summary (aligned to simulator)
- Cavitation: `HNI ≥ 1.5` and/or `IER ≥ 0.2/hr` on recurring weeks.
- Wear: `HTS_% ≤ −5%` over 90 days with `HNI ≤ 1.2` and `IER < 0.1/hr`.
- Healthy: `|HTS_%| < 3%`, `HNI ≤ 1.2`, `IER < 0.1/hr` across most weeks.

---

## Notes & Trade‑offs
- No smoothing: thresholds use weekly medians only; short spikes may slightly inflate HNI/IER but still track simulator behavior.
- No explicit active mask: relying on `recipe` naturally excludes idle since simulator leaves `recipe` empty when idle.
- Keep units consistent (kPa, m³/h, A). Trends are percent‑based and unit‑free.
