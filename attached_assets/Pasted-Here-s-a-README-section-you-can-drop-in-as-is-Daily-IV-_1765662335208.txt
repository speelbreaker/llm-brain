Here’s a README section you can drop in as-is.

## Daily IV Sanity Check (for Humans & Ops Bot)

This checklist is for anyone (or any ops bot) that wants to quickly verify that IV calibration and the synthetic universe are in a healthy state **before trusting new backtests or live decisions**.

### 1. Run a fresh live calibration

**UI steps**

1. Open the **Calibration vs Deribit** tab.
2. Select:
   - Underlying: `BTC`
   - Source: `Live API`
   - DTE Range: weekly band, e.g. `3–10` days
3. Click **Run Calibration (With Policy)**.

**Pass criteria (weekly band)**

From the results card:

- **Option Type metrics**
  - `Count ≥ 50`
  - `MAE % ≤ 20%`
  - `Vega-Weighted MAE % ≤ 15%`
- **Term Structure (weekly band row)**
  - `MAE % ≤ 25%`
  - `Vega-Wtd MAE % ≤ 20%`
- **Skew Fit Analysis**
  - `Max absolute diff ≤ 0.10`  
  (if between `0.10` and `0.20`, treat as “drift but not broken”; above `0.20` is a warning)

If any of these are badly violated (huge MAE, tiny sample size, or crazy skew diff), flag IV as **suspicious** and avoid using this run to update multipliers.

> **For ops bot:**  
> Parse the latest BTC live calibration run. If `count < 50` or `mae_pct > 25` or `vega_weighted_mae_pct > 20`, mark the run as `UNHEALTHY` and do not auto-update any multipliers from it.

---

### 2. Sanity-check the recommended IV multipliers

In the **IV Calibration Update Policy** box:

- Note:
  - `Recommended` multiplier
  - `Smoothed` multiplier
  - `Current Applied Multipliers` table

**Pass criteria**

- All applied multipliers are within a **reasonable band**, e.g.:
  - `0.70 ≤ multiplier ≤ 1.50`
- Day-over-day drift is small:
  - `|smoothed_today – smoothed_yesterday| ≤ 0.20`  
    (large jumps often mean data or calibration issues)
- Status is either:
  - `Applied` (with enough samples & vega), or  
  - `Not Applied (Sample size too small)` — which is OK as long as existing multipliers are in-range.

> **For ops bot:**  
> - Compare `smoothed` vs previous day.  
> - If multiplier is outside `[0.7, 1.5]` **or** daily change `> 0.2`, raise an `IV_DRIFT_WARNING`.

---

### 3. Confirm skew is being applied (not stuck at 1.0)

Still in the calibration tab:

- Check **Skew Fit Analysis**:
  - “Current Ratio” should **not** be permanently `1.0000` at all deltas if you’ve previously applied skew.
  - “Recommended Ratio” should be reasonably close:
    - `|Diff| ≤ 0.10` is good.
    - `0.10 < |Diff| ≤ 0.20` means “market skew moved, consider reapplying soon”.
    - `> 0.20` → something is off (market regime change or skew not updating).

> **For ops bot:**  
> - Read current ratios from the calibration store (not static settings).  
> - If all current ratios == 1.0 for BTC/ETH but we have past successful skew calibrations, raise a `SKEW_NOT_APPLIED` warning.

---

### 4. Quick Greg smoke test (optional but recommended)

Run a **short backtest** for GregBot – VRP Harvester on BTC:

- Underlying: `BTC_USDC`
- Dates: e.g. last 5–7 days
- Mode: your normal synthetic mode (e.g. Live IV, Synthetic Grid)
- Use the **current applied multipliers** (don’t override).

**What to look for**

You don’t need perfect performance, just sanity:

- `num_trades` is **non-zero** (Greg is alive).
- `win_rate` and `net_profit_pct` are not completely insane given recent price action.
- If you run two tiny tests with multipliers `0.9` and `1.1`:
  - At `1.1` you should generally see **more trades and higher VRP scores** than at `0.9`.

> **For ops bot (future)**:  
> - Optionally schedule a daily mini backtest with the current multipliers and record:
>   - `num_trades`, `net_profit_pct`, `median_greg_score`.  
> - Only raise an alert if `num_trades == 0` for several days **and** VRP is high (indicating a logic bug, not just a quiet market).

---

### 5. When NOT to trust the day’s IV

Mark the day as **IV_UNTRUSTWORTHY** (for human + bot) if **any** of the following hold:

- Live calibration:
  - `count < 30` **and** `mae_pct > 30%`
  - or `vega_weighted_mae_pct > 30%`
- Recommended multiplier:
  - outside `[0.5, 2.0]`
  - or changed by more than `0.3` vs yesterday
- Skew:
  - `Max abs diff > 0.25`
  - or current ratios stuck at `1.0` when previous calibrations weren’t
- Backtest smoke test:
  - zero trades for Greg over a period where VRP is clearly high.

In that case, **do not** auto-apply the day’s calibration; investigate data or Deribit conditions first.
