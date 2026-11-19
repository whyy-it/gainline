# Rugby union margin-Elo model
Run order: `build_data.py` (assembles games.parquet) -> `model.py` (grid-tunes params.json) -> `price.py` (ratings + pricing).
Data: seanyboi/rugbydata (GitHub, 2018-Sep 2026) + hand-entered 2026 gap fill (Nations Championship R1-5, Sep tests, 2026-27 Prem/Top14/URC to 8 Nov).
Model: pred margin = R_home - R_away + HFA; R += K*(clip(margin,±cap) - pred). Win prob ~ Normal(pred, sigma).
Intl: K=0.10, HFA=4, cap=60, sigma=16.6. Club: K=0.06, HFA Top14 9 / Prem 7 / URC 6 / Europe 6, cap 40, 20% season regression.
To update: add new results to add_rows() in build_data.py, update NC6 / CLUB_FIX odds in price.py, rerun.

## v2 (variables test)
features.py builds pre-match state: Elo, attack/defence, rest, travel, form, team-specific HFA.
lineup.py builds squad features from player logs (core = share of usual top-23 in the matchday squad; exp = avg prior apps).
evaluate_v2.py = walk-forward Brier/log-loss by season. Result: only squad strength adds real signal (clubs);
internationals stay on v1 Elo with tier-1 sigma 16.6. Squad features need team sheets at bet time.

## Try-scorer model (tries.py)
lambda = team expected tries (Poisson on Elo margin) x position share x player multiplier (EB, k=4) x 1.106 calibration.
P(anytime) = 1 - exp(-lambda). Test 2025-26: Brier 0.1138 vs 0.1186 position-only. tryscorer_preview.csv = round 6 preview.

## Fine-tuning round (tune.py), 17 Nov
Strict split: tune on 2024 season, score once on 2025-26+.
- Match model extras (blowout dampening, new-team K, per-team HFA, probit calibration): better on validation,
  WORSE on test (intl 0.163 -> 0.178, club 0.176 -> 0.177). Overfit -> rejected; v1 params kept.
- Totals: attack/defence rate 0.02-0.12 all within 0.1 pt MAE; kept 0.05. Totals beat a flat average by only 0.5-0.7 pts.
- Try model: added team expected points to the team-tries Poisson -> test Brier 0.1142 -> 0.1133. Adopted.

## v6 experiments (v6.py, lineup_value.py), 17 Nov. Split: tune 2024, test 2025-26.
KEPT
- Long international history (1995-2022 tier-1 Tests from the 1871-2024 file) + K=0.08: Test Brier 0.1624 -> 0.1584 (val 0.186 -> 0.177).
- Club lineup strength (player-value of matchday 23 + rotation gap vs last 10 games + core share): club Brier 0.1763 -> 0.1724 (val 0.171 -> 0.164).
REJECTED (worse or no change on test)
- Red-card down-weighting (both), performance/xMargin ratings (both), goal-kicking %, referee tendencies (778 games, r = -0.01),
  lineup value for internationals (only 14% of Tests have player data; val worse).
- Weather: not reachable from this environment.

## Player grades, weather, odds log (17 Nov)
- grades.py + grade_weights_pa.json: "points added". Scoring credited at face value (try 5, conversion 2, penalty 3),
  try assist 2.5, cards set by convention (yellow -3.5, red -8). Non-scoring actions are valued by a sign-constrained ridge
  of match margin on team stat differences (4,532 matches, trained pre-Aug 2024): line break +1.62, turnover won +1.05,
  turnover lost -1.04, lineout won +0.58, penalty conceded -0.52, 10 m carried +0.5, tackle +0.06, missed tackle -0.04.
  Kicks excluded (teams that lead kick more). Predictive check: lineup strength from these grades scores club Brier
  0.1621 val / 0.1737 test vs hand weights 0.1642 / 0.1724, i.e. equal within noise. The model keeps hand weights; the site uses points added.
- weather_pull.py / weather_test.py: run on your own machine (Open-Meteo is blocked from Claude's sandbox).
  pip install pandas pyarrow requests; python3 weather_pull.py; python3 weather_test.py. Keep weather only if it lowers the miss on val AND test.
- log_round.py + odds_log.csv: round 6 opening lines seeded. Log closing lines and results each round; `report` gives
  model vs market Brier and how often the line moved toward the model.

## Super Rugby Pacific (sr.py, sr_params.json), 18 Nov
477 matches 2022-2026. Tuned on 2025, tested on 2026 (83 games): Brier 0.1695, 76% winners, avg margin miss 14.0.
K=0.03 (ratings move slowly), season regression 20%, cap 40.
Home advantage: 1 pt domestic, +2 when the visitors crossed a border, +6 extra for the Fijian Drua at home in Fiji.
Rejected: squad/lineup features (too little history; hurt validation). Totals barely beat the league average (11.9 vs 12.1 miss), avg total 55.8.
Try model now includes Super Rugby (2026 test: 0.1240 vs 0.1274 position-only).
Bug fix: tries.py cleaning dropped any match with blank points (all 2026 files). Test set grew 9.7k -> 25.6k rows;
true overall try Brier 0.1145 vs 0.1182 baseline (earlier 0.1133 was on the smaller, biased subset).

## Audit, 18 Nov (produce.py -> bets.py -> verify_site.py)
Data: no duplicates, no swapped home/away Tests, no impossible scores. Removed 1 bad row (Wasps 0-42 London Irish, Jan 2023,
after Wasps folded). Two suspicious shutouts verified real (Sharks 0-35 Cardiff 2022, Perpignan 0-28 Montpellier 2025).
Known gap: ~10 of 93 Prem 2025-26 games missing from the source.
Leakage: every feature is computed from games before the match; calibration always fitted on earlier seasons.
Calibration on unseen seasons: Tests 0.158 (draw = half), clubs 0.1725, Super Rugby 0.1695 - all well calibrated within noise.
Try model: over-confident above 40% -> added logistic recalibration (fitted 2024-25; position, Test, replacement terms).
  Test Brier 0.1145 -> 0.1142; 40-60% band now 46% predicted vs 45% actual.
Club squad features cannot be used for 2026-27 yet (player data ends Sep 2026): club projections are rating-only for now.
produce.py rebuilds every site number; bets.py picks bets by written rules; verify_site.py re-implements the Test ratings
from scratch and checks all 23 site checks (ratings, probabilities, scores, odds, EV, stakes, coverage).

## Historical odds backtest (18 Nov)
odds_pull.py  - pulls closing odds (5 min before kickoff) for every rugbyunion_* key on The Odds API, 1 snapshot per kickoff
                slot, cached; `--estimate` first to see credits. Needs ODDS_API_KEY. Writes odds_hist.csv.
preds_oos.csv - the model's out-of-sample predictions for 2,320 games since 2023 (each season calibrated only on earlier ones).
backtest.py   - model vs closing market Brier, margin miss vs the line, and the bets.py rule graded at the best price
                (2-way ML = draw-no-bet), with 95% bootstrap ROI interval and a verdict against the 10% bar.
                `--books draftkings,fanduel,...` to grade at PA books only.

## Backtest vs real closing odds: Super Rugby 2014-19 (18 Nov)
Data: github.com/lukepothier/bet results.csv, 1,331 Super Rugby games 2010-19 with average closing 1X2 odds (no dates;
season ends located by known finals). bt_sr.py: same Elo method, tuned on 2012-13, probit walk-forward, test 2014-19 (788 games).
Brier model 0.1878 vs market 0.1828 vs 50/50 blend 0.1832 -> market better; model adds nothing to the market.
Bets rule (blend, EV>=5%, gap>=3, DNB priced from 1X2): 118 bets, win 29.7%, avg odds 4.46, flat ROI -0.8%
(90% CI -29% to +32%), P(ROI>=10%) 26%. Fires mostly on long underdogs (109 of 118). Fails the 10% bar.
Caveats: older era (18-team, SA/ARG/JPN sides), average not best prices (3-way vig 8.9%), moneyline only.
Tests and clubs still untested against odds -> needs odds_pull.py run locally with the Odds API key.
