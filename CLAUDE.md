# Gainline — rugby union model + betting research

Owner: Wyatt (actuary; wants concise, direct answers with concrete numbers). Odds shown in **American** format.

## Goal of this phase
Backtest every model against real closing odds from The Odds API, tune or rebuild models to beat the market,
and if no model edge survives, find situational angles that do. Models in scope: match result / win probability
(Tests, clubs, Super Rugby), handicap (margin), total points, try scorer.

## Hard rules
- **Bet only systems with ≥10% backtested ROI** on out-of-sample data, with a bootstrap CI and a sensible n.
- Strict walk-forward. Tune ≤2024, validate 2025, test 2026. Never touch test data while tuning. Calibration is always fitted on earlier seasons only.
- Report every result against the market: Brier (model vs no-vig close vs blend), ROI at best price and at average price, CLV (did we beat the close?).
- Count every angle/variant tried and correct for multiple testing (holdout confirmation, or Bonferroni on the p-value). Log rejects in model/README.md.
- Staking: ¼-Kelly, 5% EV floor. Bets tab uses Bet = ½ of ¼-Kelly, Lean = ¼ of ¼-Kelly.
- Wyatt prefers the **2-way moneyline (draw-no-bet)**: EV = nd·(bp·o − 1).
- Never commit `.env` (ODDS_API_KEY). Estimate API credits (`--estimate`) and tell Wyatt before spending.
- The public site carries no betting content; the Bets tab is private.

## Setup
`bash setup.sh` (venv, deps, re-clones seanyboi/rugbydata for player data, creates .env). Then put the key in `.env`.
All scripts run from `model/` (paths are relative: `../data`, `../matches.parquet`).
Sanity check first: `cd model && python3 verify_site.py` → 23/23 pass.

## Layout
- `model/` — everything. Read `model/README.md` first: it is the full decision log (what was tried, kept, rejected, and why).
  - Data: `build_data.py` → `games.parquet` (3,365 games 2018–Nov 2026); pre-match states `SI/SC/SR.parquet`; player data.
  - Tests/clubs: `v6.py` (harness: elo, probit_fit, score), `features.py` (attack/defence → totals), `lineup*.py`.
  - Super Rugby: `sr.py`. Try scorer: `tries.py` (+ `tries_calib.json`). Grades: `grades.py`.
  - Production: `produce.py` → `site.json`; `bets.py` (Bets tab rules); `verify_site.py` (23 checks).
  - Odds: `odds_pull.py` (Odds API historical closing lines, cached, `--estimate`, `--budget`) → `odds_hist.csv`;
    `backtest.py` (model vs market, bets rule, bootstrap vs 10% bar, `--books`); `odds_log.csv` + `log_round.py` (forward log).
  - `preds_oos.csv` — 2,320 out-of-sample predictions since 2023 to join to odds.
- `site/` — `mobile_template.html` + `teamjs.js`; rebuild by substituting `__TEAMJS__` and `__DATA__` (site.json).
- `bt_sr/` — first real-odds backtest (see below).

## Current models (out-of-sample)
| Model | Method | Brier / miss |
|---|---|---|
| Tests | margin-Elo 1995+, K .08, HFA 4, cap 60, probit σ≈13.9 | 0.158 |
| Clubs | K .06, HFA Top14 9 / Prem 7 / URC 6 / Euro 6, cap 40, 20% regression, squad logistic | 0.1725 |
| Super Rugby | K .03, HFA 1 +2 cross-border +6 Drua in Fiji | 0.1695 (2026) |
| Try scorer | Poisson team tries × position share × player mult (k=8), logistic recal | 0.1142 |
| Totals | attack/defence state | barely beats flat average |

Rejected as predictors (hurt test): blowout dampening, per-team HFA, red-card down-weighting, xMargin ratings,
kicking %, referee, SR squad features, Test lineup value. Weather untested (run `weather_pull.py` locally).
These may still work as **betting angles vs the market** — test them that way, separately.

## What we know vs the market so far
`bt_sr/bt_sr.py`: Super Rugby 2014–19, 788 test games, average closing 1X2 odds (free GitHub data).
Model Brier 0.1878 vs market **0.1828**; blend 0.1832 (model adds nothing). Bets rule: 118 bets, −0.8% ROI,
109 of them long underdogs. **Fails the bar.** The model's "edges" cluster on big dogs → likely over-confidence there.

## Plan
1. **Pull odds.** `python3 odds_pull.py --estimate` → report credits → pull (h2h, spreads, totals; uk, eu, us).
   Check which rugbyunion_* keys and years exist (historical data starts mid-2020). Fix team-name aliases.
   Try-scorer props are probably not in the historical API: confirm. If not, the try model is tested on the forward log only.
2. **Baseline backtest** every market: `backtest.py` for ML (DNB), handicap (cover % vs line, ROI at −110 equivalents), totals.
   Split by competition, favourite/dog, line size, home/away. Measure CLV using opening vs closing snapshots.
3. **Market-aware models.** Treat the closing line as the baseline and model the residual
   (result − closing margin, total − closing total) with our features. If the residual model has no out-of-sample skill, the market is efficient there.
4. **New versions to try:** shrink toward the market (fit the blend weight instead of 50/50); a fix for underdog over-confidence;
   opening-line value (bet early lines that our model says will move — score by CLV); team-sheet timing (Thursday/Friday announcements vs line moves);
   totals model rebuilt on the residual vs the closing total.
5. **Situational angles** (each vs closing line, with multiple-testing control): Test-window club depletion, travel/jet lag,
   short turnaround, rotated XVs (changes vs prior week), dead rubbers / qualified teams, European rounds after Test windows,
   derbies, weather, referee, promoted/relegation-threatened sides, first round after the break. `angles.py` is a starting point.
6. **Ship only what clears ≥10% ROI** on test. Update `bets.py`, `produce.py`, `verify_site.py`, `model/README.md`.

## Open items
- About 10 Prem 2025–26 games are missing from the source.
- No 2026–27 club squad data (player data ends Sep 2026).
- Bets tab on the site uses Wyatt's PA-book board from 18 Nov (prices in `bets.py` MKT, decimal).
