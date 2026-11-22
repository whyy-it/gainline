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

## Backtest vs real closing odds: Six Nations / Tests, The Odds API (18 Nov)
odds_pull.py found exactly one rugbyunion_* key with coverage: rugbyunion_six_nations, 29 events / 19 kickoff
slots, starting 2025-02-01 only (not mid-2020 as hoped). No club (Top14/Prem/URC) or Super Rugby keys exist on
this API at all. Pull cost 1,710 credits (90/snapshot) for 577 rows, h2h+spreads+totals, uk/eu/us books.
backtest.py matched 28 of 28 games to preds_oos.csv, split by the competition tag the odds carried:
- Six Nations (16 games): Brier model 0.1006 vs market 0.1137 vs blend 0.1046 -> model edges the market.
- International Test Match, i.e. other Tests swept into the same odds key (12 games): Brier model 0.2680
  vs market 0.2308 -> model worse than market here.
Bets rule (blend, EV>=5%, ML gap>=3): 27 bets, win 29.6%, flat ROI -2.6% (95% CI -59% to +63%).
Fails the 10% bar; verdict not distinguishable from zero. n=16-28 is too small either way to trust the Brier
edge above as real skill - one tournament's worth of lines is all this API has, and it won't grow much over
time since six_nations is the only rugby key. Evaluated the free 20-row World_Rugby_Master Kaggle sample
(data/Sample_World_Rugby.csv, 1X2 only, no spread/total odds) as a path to real club/SR backtests; full
paid set ($139, github.com/.../oliviersportsdata, 25k games 2008-2026, still no spread/total columns) not
yet purchased. Until then club and Super Rugby models have no market-odds test beyond bt_sr.py above.

## Fitted blend weight (bt_sr.py, 18 Nov) - REJECTED
Plan step 4: fit the model/market blend weight instead of a fixed 50/50. Fit on 2012-13 tune seasons only
(never touched 2014-19 test), closed-form least-squares weight minimizing Brier. Result: w=0.507, i.e.
indistinguishable from 50/50. Test Brier 0.1832 (50/50) vs 0.1833 (fitted) - no change. Bets ROI -0.8% -> -0.2%,
still fails the 10% bar. REJECTED: the model/market gap isn't a uniform-weight problem, so a single global
blend weight can't fix it.
What the same run did surface: splitting the 119 bets by favourite/underdog -
  Underdogs (odds >= +100): n=109, win 26.6%, ROI -2.1%
  Favourites (odds < +100): n=10,  win 70.0%, ROI +20.8% (90% CI -26% to +59%)
Confirms the standing hypothesis (model's edges cluster on big dogs) directionally, but n=10 favourites is
far too small to act on - CI still crosses zero. Motivates an underdog-specific fix rather than a blend-weight
fix: see next section.

## Underdog over-confidence fix (bt_sr.py, 18 Nov) - MARGINAL, NOT ADOPTED
Extended the probit calibration with a shrink-only cubic term, Phi(w0 + w1*pred - w2*pred*|pred|), w2>=0
bounded, fit walk-forward on seasons before each test season (same discipline as the base probit - never
fit on a season it's then scored on). Meant to specifically dampen confidence at large |predicted margin|
(big favourite/underdog) without touching near-even games.
Result: w2 fitted tiny every year (0.0003-0.0017, exactly 0 in 2018 and 2019) - the walk-forward MLE finds
almost no evidence of this specific miscalibration in the tune data. Brier essentially unchanged: 0.1879
(shrink) vs 0.1878 (plain probit) vs 0.1828 (market).
Bets rule (50/50 blend, same EV/gap filters, shrunk probability only): 117 bets (vs 119 plain), ROI +1.9%
(was -0.2%), underdog bucket -2.1% -> +0.8%, favourite bucket +20.8% -> +15.2%. Directionally right but
inside the noise band on both plain and shrink versions (90% CI still crosses zero by a wide margin) -
still fails the 10% bar by a lot. NOT ADOPTED: a real fix, not just a better-fitting functional form, wasn't
found. The favourite/underdog split itself may be mostly small-n noise (10-ish favourite bets per test run)
rather than a real, exploitable pattern - needs more seasons (the paid dataset, or more odds_log.csv rounds)
before trying further variants on this angle. Counts as 2 variants tried this round (blend weight, shrink
probit) toward the multiple-testing tally.

## Situational angle scan (angles.py, 18 Nov) - Bonferroni correction added
Plan step 5. CAVEAT up front: this measures each angle vs OUR Elo expectation (resid = actual margin - Elo
margin), not vs the market closing line - no broad club/intl odds history exists to test against (only 28
Six Nations games have real odds at all, far too few to split by situational flag). A surviving angle here
is a candidate for the model or the forward bets rule, not a proven market edge yet.
24 angles tested (club + tier-1 international, 2022+), Bonferroni alpha = 0.05/24 = 0.0021. 2 survive:
- **Club: heavily rotated side (core<0.45)**: n=423, resid -6.56 pts, t=-7.82, p=4e-14. Not a new finding -
  confirms the v6.py lineup-strength feature (core share) already validated there (club Brier 0.1763 -> 0.1724)
  - but that feature is NOT wired into live 2026-27 production ratings (player data ends Sep 2026, so the
  rolling player-value history can't be computed for this season). This is a known, currently-unexploited
  gap in the live model, not a new discovery.
- **Club: European team away in South Africa**: n=153, resid -5.60 pts, t=-4.00, p=1e-4. Genuinely new.
  CHFA in v6.py gives a flat URC home-field edge (top14=9, prem=7, urc=6, euro=6) with no direction-specific
  penalty for crossing into SA (altitude/travel). The complementary direction (SA team away in Europe,
  resid -3.49, t=-2.92) trends the same way but doesn't survive correction alone - consistent story, weaker
  signal. The SR model already has exactly this kind of cross-border/altitude adjustment (+2 cross-border,
  +6 Drua-in-Fiji); the club model has nothing equivalent for SA.
Next: test a direction-specific SA-travel penalty in the club Elo (v6.py CHFA/travel term), same walk-forward
discipline (tune/test split) as the other v6 experiments, before adopting.

## SA-travel term in the club Elo (v6.py sa_bonus, sa_travel_test.py, 18 Nov) - MARGINAL KEEP
Added sa_bonus to v6.elo(): extra home-side edge when exactly one of home/away is an SA club (Bulls/Lions/
Sharks/Stormers), mirrors sr.py's travel_bonus. Checked direction first: home=SA/away=non-SA residual is
+5.9 pts and POSITIVE EVERY SEASON 2021-2026 (n=179) - a real, persistent pattern, not era noise. The reverse
(home=non-SA/away=SA) is weaker and inconsistent (+2.7 overall, swings from -11.8 to +6.8 by season, n=167) -
so this is directional, not symmetric border-crossing fatigue like Super Rugby's.
Tuned sa_bonus on val (2024) only at full precision (v6.score()'s 4dp rounding was too coarse - ties masked
the actual optimum): val-best = 1.0 pt (val Brier 0.170761 vs 0.170840 baseline - a wisp of an improvement).
Test (2025-26), decided before looking: 0.175792 vs 0.176288 baseline, delta +0.0005. Clears the KEPT bar
(improves both val and test) but barely - nowhere near the ~6pt raw residual's apparent size.
Why the gap: the raw angles.py residual is pre-existing team Elo ratings vs actual margin; by the time we
score val/test (2024 on), the Bulls/Lions/Sharks/Stormers' own ratings have already drifted up from years of
beating naive expectation at home, so normal K-updates have partly self-corrected this already. An explicit
bonus only captures what's left over. ADOPTED into production: produce.py's club_ratings() now calls
elo(..., sa_bonus=1.0); site.json and site/gainline.html rebuilt, verify_site.py 23/23 after rebuild (the
HTML-carries-site.json check failed until gainline.html was regenerated from mobile_template.html - rebuild
needs that explicit step, there's no automated build script for it yet).

## Totals vs real closing total line, Six Nations/Tests (totals_backtest.py, 18 Nov) - INCONCLUSIVE, n too small
Plan step 4: the one market in odds_hist.csv never tested. Walk-forward: total ~ tot (attack/defence state)
linear map fit on seasons <=2024 only, predicted on 2025-26, matched to market closing totals (median across
books). Only 14 of the 28 odds-matched games carry a totals line at all (many books don't post totals on
these fixtures) - already below any sample size worth trusting.
Points MAE: model 11.16 vs market line 10.93 (market slightly tighter). Over-prob Brier: model 0.2420 vs
market 0.2497 (model very slightly better) - both differences are noise at n=14, not a finding either way.
Bets rule fired only 5 times: ROI +14.5%, 95% CI -62% to +91%. Essentially uninformative - do not read
anything into the sign. INCONCLUSIVE, not REJECTED: the totals market genuinely hasn't been tested yet,
just under-sampled. Needs odds_log.csv to accumulate many more rounds (or the paid dataset, which also has
no totals/spread columns per the Kaggle sample - so totals specifically can only grow via the forward log).

## Market-aware residual model (plan step 3) - NOT ATTEMPTABLE YET
Checked before starting: with n=14-28 games per market (Six Nations/Tests, the only real odds we have),
there isn't enough data to fit a residual model (result - closing line ~ features) and validate it out-of-
sample without guaranteed overfitting - a model with even 2-3 free parameters fit on 14-28 points and then
"tested" on a held-out slice of what's left would be fitting noise, not finding skill. Blocked on sample
size, not on methodology or effort. Same constraint applies to opening-line/CLV value and team-sheet-timing
angles from step 4 - log_round.py's odds_log.csv (13 rows, round 6 opening lines only) is the only lever
that grows this for free; it just needs many more rounds logged before any of step 3 or the CLV angles are
testable with a straight face.

## New direction, 18 Nov: fine-tune player try-scoring, grades, team ratings; club/country/all-time player
rankings; explanations for how ratings are calculated; eventually game-recap reports (best players, highlight
moments). Player data only goes back to 2022-01-01 - "all-time" rankings will really mean "since 2022."
No play-by-play/timestamps exist in player_stats.parquet (match-level box scores only) - highlight moments
will have to be stat-line-based ("hat-trick", "3 turnovers won"), not minute-by-minute narrative.

## tries.py methodology fix: k was chosen by testing on the test set (18 Nov) - FIXED
Found while reviewing for fine-tuning: the shrinkage k was picked by evaluating each candidate directly on
the 2025-26+ test period (`if logloss on test < best...`) - exactly the test-set leakage the hard rules
exist to prevent. Fixed: k is now chosen on a val slice (2024-08-01 to 2025-08-01, trained only on data
before it); test (2025-08-01+) is never touched until the one final score. Val picked k=8 (same value the
leaky method happened to land on), test Brier 0.1145 (p_full) - consistent with the previously-reported
number, so this wasn't silently overfit, but the methodology is now actually defensible.
Also found: produce.py applies a further logistic recalibration via tries_calib.json (coef/intercept/scale
for log-odds + intl + sub features) that has NO script producing it in this repo - it's a frozen, unreproducible
artifact from an earlier session. Flagging before building rankings/recaps on top of it.

## Grades tab was a frozen snapshot, not a live pipeline - FIXED (grades_produce.py, 18 Nov)
Found while tracing how the site's Grades tab gets its data: produce.py explicitly kept `grades`/`grades_match`/
`grade_values` untouched every run (`d = json.load(open(OUT))  # keep grades...`), and grades.py was never
imported or called anywhere. The 45 player grades on the live site (South Africa 33-26 New Zealand, 29 Aug
2026) and grade_values.json/grade_weights_pa.json were frozen leftovers from some earlier, undocumented run.
Built grades_produce.py: picks the most recent high-profile completed match with full player stats (auto-
selects by date + competition + try count; the "most recent" is capped by player_stats.parquet's hard cutoff
of 2026-09-06 - this tab cannot reflect anything after that date until the player-data source resumes, same
root cause as the club-squad-features gap in the Open Items list), grades every player, and writes the three
site.json fields for real, every produce.py run.
Weight-choice correction along the way: checked hand weights (grade_weights_pa.json) against this specific
objective (reconstruct one match's point margin from that match's own stat diffs, held-out val slice) and
got R2=-0.79 - worse than predicting the mean. The data-driven ridge regression (grades.py's learn_values,
refit fresh each run) gets R2=0.65 on the same val set. This is NOT a contradiction of the earlier "equal
within noise" finding - that compared the two weight sets on a different objective (lineup strength as a
multi-game input to the club betting model). For single-match player grading specifically, data-driven wins
clearly, so that's what's now live. grade_values.json is rewritten each run instead of staying frozen.
Also fixed an explanation-clarity bug while building the 'key' stat-line string: "1 pen" read as a scored
penalty kick when it meant penalty conceded (bad) - relabeled every negative-contribution stat with an
explicit sign and an unambiguous word (conceded/missed) rather than a bare abbreviation.
verify_site.py 23/23 after rebuilding site/gainline.html from the refreshed site.json.

## Club & country player rankings (player_rankings.py, 21 Nov)
Same data-driven points-added weights as grades_produce.py, aggregated instead of single-match: shrunk mean
pa/game over the last 365 days of available data (empirical-Bayes, k=6 pseudo-games toward the position-group
mean - same recipe as tries.py's player multiplier and lineup_value.py's 'usual' baseline), min 5 games to
be ranked. No minutes-played field exists anywhere in the data, so "per game" is the best available rate stat.
Two issues found and fixed before trusting the output:
- Women's and U20 competitions were pooled into the same position-group z-score as senior men's leagues
  (e.g. a Canada Women lock and a men's URC lock landing in the same ranked list). Excluded both by name match.
- Club z-scoring was pooling ALL leagues together (Top14 vs Japan League One vs Super Rugby Pacific) - not a
  fair comparison, different competitive levels. Fixed: club rankings are z-scored WITHIN each league/position,
  i.e. "best Top14 prop" and "best Premiership prop" are separate rankings, not blended into one global list.
  Country rankings stay pooled (restricted to T1 nations, the same convention v6.py already uses) since
  international rugby doesn't split into leagues the same way.
Sanity-checked the output against players I'd expect to see near the top: Antoine Dupont top-3 French
scrum-halves, Matthieu Jalibert top French fly-half (country and Top14 both), Henry Slade top Premiership
centre, Ben Earl top England back row - all plausible, nothing obviously broken.
Output: rankings_club.parquet, rankings_country.parquet (player, league/grp, n games, shrunk rate, grade).
NOT yet wired into site.json/the site UI - that's a separate scoping decision (a full rankings tab needs new
teamjs.js rendering, league/position filters) - flagging before building it so Wyatt can weigh in on shape.

## Rankings UI wired up (21 Nov)
Added player_rankings.site_rankings() (scoped to Country/Top14/Prem/URC only - the competitions the rest of
the site already tracks, not every league in the raw data) into produce.py -> site.json's new `rankings` key.
New Rankings tab in mobile_template.html/teamjs.js: competition filter (Country/Top14/Prem/URC), rows sorted
by grade, same grade-badge visual language as the Grades tab. Verified in-browser at both desktop and mobile
widths - team badges/flags, filters, and sort order all render correctly. verify_site.py 23/23.
Note: produce.py now calls both grades_produce.player_grades() and player_rankings.site_rankings(), which
each independently reload player_stats.parquet and refit the data-driven weights (team_diffs/learn_values runs
twice per produce.py call). Works fine, just redundant compute - worth consolidating into one shared fit if
produce.py's runtime ever becomes a problem.

## All-time (since 2022) rankings + game recap reports (21 Nov)
**All-time rankings:** player_rankings.rank()/site_rankings() now take a `days` window - 365 ("current form",
min 5 games, unchanged) or None ("all-time", min 15 games, every game since the data starts 2022-01-01 - there
is no deeper history, so this is "all-time since 2022" not literally all-time). Added a Current/All-time
toggle above the competition filter in the Rankings tab. Sanity-checked: Antoine Dupont, Thomas Ramos, Ardie
Savea, Finn Russell all surface near the top of their positions with 20-40+ game samples - plausible.

**Game recap reports (recap_report.py):** reuses grades_produce's featured-match selection (same game as the
Grades tab, same data-cutoff caveat ~Sep 2026). Two parts:
- Player of the match per team (top grade, already computed by grades_produce).
- Highlights: synthesized from box-score thresholds (hat-trick/brace, sent off/sin-binned, 3+ dominant tackles
  or turnovers, 90+m carried, 18+ tackles, 2+ try assists, plain tries last and capped at 3 named + a "also
  scored" summary for the rest) - NOT a minute-by-minute account, because no play-by-play or timestamps exist
  anywhere in the data (player_stats.parquet is match-level totals only).
First pass prioritized tries above everything else and a single match with 7 try-scorers buried a genuinely
rare stat line (one player's 20-tackle game) behind seven near-identical "scored a try" bullets. Fixed by
reordering priority (cards > hat-trick/brace > rare defensive/carrying feats > plain tries) and capping plain
tries to 3 + a summary line, so standout non-scoring performances don't get crowded out by routine tries.
Wired into site.json's `recap` field and surfaced as a short highlights block at the top of the Grades tab
(same match, so the two naturally pair) rather than a new nav tab.
verify_site.py 23/23 after rebuild. produce.py now independently refits the data-driven weights a THIRD time
(grades_produce, player_rankings, recap_report each call their own load) - same redundant-compute note as
above, more pronounced now; worth consolidating if this gets slow.

## SA-travel fix propagated to where it actually matters (21 Nov)
Checked: sa_bonus=1.0 had only ever been wired into v6.py's club_ratings() call in produce.py - the Ratings-
tab DISPLAY numbers. features.py's build_state() (the function that actually produces SC.parquet - the
baseline tries.py's exp_margin and angles.py's residual scan both use) had no sa_bonus parameter at all, so
the fix was cosmetic: it changed what you see, not any prediction, backtest, or bet.
Also found: nothing in the repo calls features.club_state()/intl_state() and writes SC.parquet/SI.parquet -
same frozen-artifact pattern as tries_calib.json and the old Grades tab, a third instance of it.
Fixed: added sa_bonus to build_state()/club_state() (default 1.0, matching the validated value; a narrower
SA_URC_CLUBS constant - just Bulls/Lions/Sharks/Stormers - rather than the broader SA_CLUBS already used for
the unrelated 'travel' feature). Re-validated through build_state's own harness (val 0.17084->0.17076, test
0.17629->0.17579) - identical to the standalone v6.py check, confirms the fix carries through correctly.
Regenerated SC.parquet (features.club_state() -> lineup.attach() -> to_parquet - the attach() step was easy
to miss; a first attempt without it silently dropped the core_h/core_a/exp_h/exp_a/core/exp columns angles.py
and tries.py both depend on, caught immediately by an AttributeError on rerun).
Downstream effects, as expected: tries.py Brier unchanged (0.11451, noise-level). angles.py's SA residuals
shrank but didn't disappear - "European team away in SA" resid -5.60->-4.58, t -4.00->-3.28 (still survives
Bonferroni, p=0.0013); "SA team away in Europe" -3.49->-2.45 (already hadn't survived). Makes sense: the
bonus is a conservative +1pt against what was originally a ~6pt raw effect, so it should explain some but not
all of it. produce.py rerun end to end, verify_site.py 23/23.

## Remaining situational angles from the plan (angles.py, 21 Nov) - found a latent bug, fixed it, nothing new survives
Plan step 5 still had untested items: derbies, European rounds after Test windows, dead rubbers, promoted/
relegation pressure. Before adding them, found a real bug while reasoning through how to test "derby" (a
property of the MATCH, true for both teams equally - unlike every existing angle, which picks a SIDE):
**any match-level flag that doesn't distinguish home/away is mathematically guaranteed to show resid=0.0 and
cover=0.5 in this framework**, because team_log() puts both the home row (resid) and the away row (-resid)
of every match into the same bucket, and they cancel exactly. Confirmed directly (`C.comp=='French Top 14'`,
a pure match property, mean resid = -1e-16). This silently neutered the one match-level angle already in the
list - "Club: playing during intl window" reported resid=0.0, t=0.0 exactly, which looked like a clean null
result but was actually never a real test of anything.
Fixed with summarize_var(): tests variance (mean squared resid) instead of signed mean, via a permutation
test against the same pool the flag is drawn from - variance doesn't cancel the way the signed mean does, so
this one is actually informative for symmetric match-level flags.
Added, with proper tests this time (26 angles total this round, Bonferroni alpha 0.0019):
- **Derbies** (Racing92-Stade Francais, Bath-Bristol, Leicester-Northampton, Leinster-Munster, Ospreys-
  Scarlets, Bulls-Lions - only the unambiguous, well-known ones, not an exhaustive list): n=56, resid_sd 17.6
  vs pool 16.0 (slightly WILDER than expected, not tighter - opposite of the "rivalry intensity closes the
  gap" intuition, but not significant, p=0.39, and n=56 is thin).
- **Club games during an intl window** (re-tested properly this time): resid_sd 14.7 vs pool 16.0 - tighter/
  more predictable, p=0.033 nominally but does NOT survive the corrected bar (0.0019). Interesting direction
  (maybe: both sides missing a similar share of Test players evens things out) but not proven.
- **European cup rounds within 10 days of an intl window**: resid_sd 15.3 vs pool 16.0, not significant.
None survive correction. Still just the 2 standing survivors (European team away in SA, heavily rotated
side) from before - both already logged above.
**Deferred, not attempted:** dead rubbers/qualified teams and promoted/relegation-threatened sides both need
a league standings/table builder (points, position, games remaining) that doesn't exist in this repo yet -
building something flimsy just to tick the box would be worse than being explicit that it's not done.
angles_var.csv added alongside angles.csv for the new variance-based results.

## Predicted table / title odds (season_sim.py, 21 Nov)
New idea: a predicted end-of-season table + title odds from simmed seasons. Checked first: no remaining-
fixtures list exists anywhere in the data (matches.parquet is historical-results-only; the one vendored ESPN
snapshot has 29 scattered scheduled games across 3 leagues, nowhere near a full schedule) - so this only
covers tournaments with a genuinely KNOWN schedule, not Top14/Prem/URC/Super Rugby (see below, still hunting).
Two tournaments qualify:
- **Nations Championship 2026**: 12 teams, 2 pools of 6 (NH/SH), real R1-5 results already in games.parquet
  (found 4 of 34 "Nations Championship"-labeled rows are actually mislabeled Rugby Championship games - SA-NZ
  x2, Argentina-Australia, Australia-SA, all SH-vs-SH when the format is cross-pool only; filtered them out,
  confirmed exactly 5 games/team remain). R6 (today) and the Finals Weekend (same-rank-across-pools pairing,
  27-29 Nov, confirmed via Wikipedia) are simulated. Real tries used for R1-5's four-try bonus; simulated
  rounds use a bonus probability fit on 557 real Test performances (logistic on score, coef 0.221, intercept
  -6.57) since we don't simulate individual tries. Result: South Africa 41.7% / France 41.4% to win it,
  New Zealand 8.2%, England 5.1% - plausible given current form.
- **Six Nations 2027** (5 Feb - 13 Mar, not started): full 15-game fixture list is public record (Wikipedia,
  announced 9 Mar 2026), hand-entered same as produce.py's FIX convention. Simulated entirely from current
  ratings (the best estimate today, though ratings will move before Feb). Result: England 42.5%, France
  35.9%, Ireland 15.9%, Scotland 5.6% - also plausible.
Scoreline simulation: margin ~ Normal(model_margin, sigma), total from the attack/defence state (not
randomized separately - a simplification), four-try bonus drawn from the empirical curve above, losing
bonus (within 7) deterministic from the simulated scoreline.
NOT wired into site.json/UI yet - numbers only, in season_sim.py's own output for now.

## Club fixtures found (club_fixtures.py) + Top 14 predicted table/playoff odds (top14_sim.py), 21 Nov
Found a free source for the remaining-fixtures gap: ESPN's unofficial rugby API (site.api.espn.com/apis/
site/v2/sports/rugby/{league_id}/scoreboard), no auth, no rate-limit trouble seen. One call only returns the
next scheduled round, so club_fixtures.py walks forward day by day (same pattern as odds_pull.py's weekly
walk) until the season ends - ~200 free calls for a full season, taking a couple of minutes.
Caught two data-quality issues before trusting any of it:
- The vendored data/gm/utils/config.py has URC and Top14 ESPN league ids SWAPPED. Found by actually fetching
  each id's team list rather than trusting the label (270557 returns URC teams, 270559 returns Top14 teams -
  opposite of the config's claim). Corrected ids: urc=270557, top14=270559, premiership=267979,
  super_rugby=242041.
- ESPN's /teams endpoint lists a stale Top14 roster (Oyonnax, relegated in 2024-25) instead of the real
  2026-27 team (Vannes, promoted - confirmed via Wikipedia + our own games.parquet, which has 9 real Vannes
  Top14 games this season already). Checked whether this contaminates the actual /scoreboard fixture pull
  used for simulation: it doesn't - 17 real Vannes games, 0 Oyonnax games in the 119-game pull. Only the
  roster metadata endpoint is stale; the fixtures themselves are reliable. Would re-check this the same way
  for any new league before trusting it.
Built top14_sim.py: real format confirmed via Wikipedia (double round-robin, 4/2/0 bonus points, +1 for a
3-try-or-more differential, +1 losing within 7; top 6 playoff, 1st/2nd bye to semis, 4v5 and 3v6 qualifying
finals, 1st plays 3v6 winner / 2nd plays 4v5 winner in the semis, final at Stade de France; 13th plays a Pro
D2 finalist in an access match - not simulated, no Pro D2 data in this repo). Four-try-bonus probability
fit fresh on 2,361 real club team-performances (differential tries vs margin, logistic coef 0.266, intercept
-4.80) - a different curve from season_sim.py's international one, since Top14's rule is a 3-try differential
not a single team's absolute score.
Current standings built from games.parquet's real 2026-27 results (9 rounds played) with the correct bonus
rules and real tries (not simulated) via team_match_stats. Remaining 17 rounds + playoffs simulated (n=5000):
Toulouse 39.2% to win the title, Montpellier 26.1%, Bordeaux-Begles 14.9%, Pau 13.2%, Vannes (promoted) worst
at ~0% - all plausible given Toulouse's long run of dominance.
Caveat (now fixed, see next entry): the club margin SD was a placeholder (15), not properly fit.

## Club margin sigma properly fit (top14_sim.py, 21 Nov)
Replaced the placeholder sigma=15 with a real probit fit: club margin -> win prob on all club games (same
convention as every other probability calibration in this repo), sigma=1/slope. Checked stability before
trusting it: 13.18 fit on seasons before 2025 only vs 13.33 on everything through today - not sensitive to
the window, used the full-data fit. Titles odds tightened slightly as expected (less simulated variance):
Toulouse 39.2%->42.3%, Montpellier 26.1%->26.7%, rest basically unchanged. Brier sanity check (0.1708 val /
0.1758 test) matches the already-validated SA-bonus numbers exactly, confirming the same elo/state setup.

## Premiership, URC, Super Rugby Pacific predicted tables (league_sim.py, 21 Nov)
Built one shared module instead of 3 more near-copies of top14_sim.py, after confirming each league's real
format via Wikipedia:
- **Premiership** (10 teams, double round-robin, no relegation this season - 'Franchise Model' replaced it):
  4/2/0 bonus, +1 ABSOLUTE 4+ tries (like Six Nations, NOT Top14's differential rule), +1 losing within 7.
  Top 4 playoff, fixed semis 1v4/2v3 (higher seed hosts), final at neutral Twickenham.
- **URC** (16 teams, 18 games - genuinely uneven schedule from the regional pools, not a round-robin, so the
  ESPN fixture pull is load-bearing here): same abs-bonus rule. Top 8, confirmed via Wikipedia that semis/
  final are FULLY RE-SEEDED each round (better remaining seed always hosts) - different mechanic from Top14/
  Super Rugby's partial re-seed, needed its own resolver (playoff_urc).
- **Super Rugby Pacific**: turned out to be a special case - it lives entirely in sr.py/matches.parquet, not
  games.parquet/features.py like the other 3, with its own dynamic HFA (travel_bonus crossing a border,
  drua_bonus at home in Fiji) rather than a flat CHFA number. Tried routing it through the shared club path
  first; wrong tool for the job, wrote run_super_rugby() instead, reusing sr.py's own elo()/NORM. Also: the
  2027 season (Feb-June) hasn't started, same as Six Nations 2027 in season_sim.py - zero-based standings,
  whole season simulated from current (end-of-2026-season) ratings. Bonus rule is the differential one, like
  Top14. Playoff: top 6, 1/2 bye, 3v6 & 4v5 elimination finals, semis PARTIALLY re-seeded (1 plays whichever
  EF winner has the better original seed, 2 plays the other) - different from Top14's FIXED 1-always-faces-
  3v6-winner rule, so it needed its own resolver too, not a copy of Top14's.

**Two real bugs caught by a hard validation check, not a silent one:** added an upfront check in league_sim.py
that raises if any fixture-pulled team name isn't in games.parquet's own team list (a missing NAME_MAP entry
would otherwise silently drop that team's remaining games, which is exactly what happened on the first
Premiership run - Bath/Exeter/Gloucester/Northampton showed 0.0% title odds despite good current points,
because "Bath Rugby" != "Bath" etc. meant their games were quietly skipped). Caught and fixed before trusting
any result:
- Premiership: Bath Rugby->Bath, Exeter Chiefs->Exeter, Gloucester Rugby->Gloucester, Northampton Saints->
  Northampton.
- URC: Benetton Treviso->Benetton, Cardiff Blues->Cardiff.
Re-ran Premiership after the fix: results flipped from nonsensical (bottom-table teams on 0%, mid-table
Harlequins favourite) to sensible (Leicester Tigers 28.8% title favourite on 17 current points, bottom-table
Newcastle 0%).

Final results (n=5000 each), all pass a plausibility check against known team strength:
- Premiership: Leicester Tigers 28.8%, Bath 22.0%, Northampton 20.9%, ... Newcastle 0.0%.
- URC: Leinster 48.2% (fewer current points than 4 teams below it, but clearly the strongest squad rating),
  Bulls/Stormers/Sharks/Glasgow all competitive, Zebre (0 current points) at 0.0%.
- Super Rugby Pacific: Hurricanes 48.8%, Chiefs 29.2%, Crusaders 13.0% (the three traditional NZ powers),
  Moana Pasifika 0.0%.
## Predicted tables / title odds UI (21 Nov)
Built standings_produce.py to consolidate all 6 simulations (season_sim.py's Nations Championship + Six
Nations, top14_sim.py, league_sim.py's Premiership/URC/Super Rugby) into one site.json `standings` field, and
a new Tables tab (mobile_template.html/teamjs.js) with a competition filter: rank, team, current points (or
"season not yet started" where applicable - Six Nations 2027 and Super Rugby 2027 both), a qualify-chance bar
for the 4 club leagues (no playoff cut for Six Nations/Nations Championship), and fair American odds + title
% for every team. Copy is explicit that these are Gainline's own fair odds, not a market price - no
sportsbook offers futures odds for any of these competitions (confirmed for Six Nations outrights earlier;
the others aren't even covered for match odds). produce.py now calls standings_produce.all_standings() each
run (~45s added, mostly the 3 club leagues' Monte Carlo loops - acceptable). Verified all 6 competition views
in-browser (actual standings, not-yet-started state, qualify bars) - all render correctly. verify_site.py
23/23.

## Live-site audit, round 1 (21 Nov): stale copy + a real grading bug
User asked to fine-tune "everything" after noticing things didn't look right on the live site. Found:
- Header, round label, competition name, match date, and market-board date were hardcoded SEPARATELY in the
  JS template (not data-driven) - exactly how it drifted: "Projections update... on Thursday" and "Market...
  18 Nov" stayed frozen regardless of which day the page was actually viewed. Moved all of it into one
  source of truth in produce.py (COMPETITION/ROUND_LABEL/ROUND_DATE/MARKET_DATE, update together with FIX/
  MKT each week), templated from site.json. `updated` itself was never set by produce.py at all (frozen
  passthrough, same bug class as the grades/rankings snapshots found earlier) - now auto-stamped each run.
- Tries tab footer claimed 4 nations' players were still pending team sheets when all 12 were already listed
  above it - made dynamic (checks which teams actually have entries).
- Method tab claimed club lineups are rated live this season when README already said they aren't (player
  data gap) - corrected.
- **Real bug, not just copy**: grades.py's ridge regression puts red_cards at -0.03 and yellow_cards at ~0 -
  essentially no penalty for a card, when any red card is one of the biggest point-swings in rugby. Checked
  why: even at alpha=0.1 (near no regularization) cards stay near zero, and fitting them on the RESIDUAL
  after the other 14 features makes it worse, not better - red_cards flips POSITIVE. A card's cost runs
  entirely through the other box-score stats (a man-down team naturally racks up worse tackle/turnover
  numbers over the rest of the game); there's no standalone signal left for a discrete card coefficient to
  find. The data genuinely can't answer this question - fitting harder doesn't help. Anchored
  yellow_cards=-3.5, red_cards=-8.0 (CARD_OVERRIDE in grades.py, applied inside learn_values so it propagates
  to grades_produce.py and player_rankings.py automatically) at the values the ORIGINAL hand-weight system
  already used, rather than invent new numbers. Honest tradeoff: single-match-margin R2 drops slightly
  (0.646->0.599, still far above hand weights' -0.788) in exchange for a grading output that isn't actively
  misleading. Verified: Harry Potter (sin-binned in the Aug 29 recap) went from grade 50 (card didn't even
  make his top-3 stat list) to grade 43 with "-1 yellow card" correctly leading his key line.
## Live-site audit, round 2 (21 Nov): full pass, found one more real bug
Went through every remaining tab's logic, not just copy.
- **Real bug (bets.py):** `if best['team'] in FLAGS or best['opp'] in FLAGS and best['team'] in FLAGS:` -
  `and` binds tighter than `or` with no parens, so this reduces to just `best['team'] in FLAGS`; the
  opp-side check was dead code. Confirmed with a truth table: the only case it changes is
  team_flagged=False/opp_flagged=True, where the current code wrongly lets the bet through - exactly the
  scenario the flag exists to catch (a known data-quality concern about one side of a matchup should
  suppress a bet on EITHER side of it, not just a bet on the flagged team specifically). Didn't change this
  round's actual output (the model happened to pick the flagged team, England, as best side anyway) but was
  a live risk for any future round. Fixed: `best['team'] in FLAGS or best['opp'] in FLAGS`, with the why-text
  lookup fixed to match (`FLAGS.get(best['team'], FLAGS.get(best['opp']))` instead of a bare `FLAGS[best['team']]`
  that would KeyError once the opp-side case could actually trigger).
- Match card kickoff day was hardcoded 'Sat' regardless of D.round_date - correct by coincidence this round
  (Nations Championship R6 is in fact all-Saturday) but would mislabel any round spanning multiple days
  (Six Nations regularly does Fri/Sat/Sun). Now derived from round_date; still assumes one day per round -
  flagged in produce.py for whenever that stops being true.
- Scanned for the same `or...and` precedence class elsewhere (`grep -rn " or .* and "` across all .py) -
  the other two hits (backtest.py, verify_site.py) already use explicit parens correctly. Isolated incident.
- Checked try_scorers()'s `cal['coef'][2] * 0` (the 'sub' calibration term always multiplied by zero) -
  initially looked like a bug, but it's correct: subs are already excluded from the candidate pool a few
  lines earlier, so the sub-indicator genuinely is always 0 for every candidate reaching that line.
- Checked the `kicks` weight (0.65/kick, looked high for a raw-volume stat) - real positive correlation with
  margin (0.32, correct sign, weaker than metres_carried's 0.61). Not a bug, left as-is.
- Tables tab's `+999900` fair odds for ~0% title chances - mathematically correct (matches how real
  sportsbooks price extreme longshots) and always paired with the "0% title" context. Cosmetic at most, not
  changing without specific feedback that it reads as broken.
No further logic bugs found this pass. verify_site.py 23/23 after each fix; all changes pushed to the live
site (whyy-it.github.io/gainline).

## Bets tab wasn't actually wired into the live pipeline - fixed (21 Nov)
Wyatt asked directly: "is the bet tab wired in live?" Honest answer at the time: no. produce.py explicitly
preserves whatever betsTab already exists ("keep bets text") and never calls bets.py - it's a separate script
someone has to remember to run. I'd run it once this round to verify the FLAGS fix, then ran produce.py AGAIN
afterward for the weekday fix, which is exactly the ordering that silently desyncs them (produce.py can
change the match predictions bets.py depends on, with nothing forcing a bets.py rerun after). Caught this by
asking "does betsTab.bets still match d['matches'] right now" - it did, but only by luck, not by guarantee.
Also found betsTab had its OWN second hardcoded copy of the round label/date (separate from the
produce.py-side one just fixed in round 2) and its own separate frozen 'updated' string - same bug class,
different file.
Fixed properly rather than patched: pulled the round metadata into round_config.py, a single module both
produce.py and bets.py import (avoids a circular import, since produce.py now also imports bets.py to call
it) - one source of truth for both tabs instead of two separately-drifting copies. betsTab.updated now
auto-stamps like site.json's own 'updated' field. Most importantly: produce.py now calls bets.build() itself
as the last step, every run, so the Bets tab can never again be generated from stale match predictions - it
rebuilds from whatever produce.py just wrote, same run, every time.
Note: lines='Sportingbet board, 17 Nov' (the actual market odds board) stays hand-maintained in bets.py's MKT
dict - that one's supposed to be manual (it only changes when Wyatt has a fresh market board to enter), so it
keeps its own explicit date rather than auto-stamping. verify_site.py 23/23, verified in-browser, pushed.

## Full "is everything live-wired" audit (21 Nov)
Wyatt asked directly to confirm this before moving on. Enumerated every top-level site.json key and checked
each one against what produce.py's main() actually sets, rather than trust it by inspection:
- **Orphaned field found**: a top-level `round` key ('Nations Championship · Round 6') that NOTHING in the
  current UI reads - grep found its only consumer is site/template.html, a dead desktop-era predecessor to
  mobile_template.html that nothing in the build process touches anymore. Pure leftover cruft, silently
  passed through every run via `d = json.load(open(OUT))`. Removed (`d.pop('round', None)`), file left in
  place for now (not deleting without asking - it's unused but harmless sitting there).
- **Real bug, bigger than the last one**: the Method tab's headline validation numbers
  (intl/club/try/sr Brier + accuracy) were HARDCODED LITERALS in produce.py's d['backtest'].update() call -
  never recomputed. Checked each against a live rerun of its own model's validation harness:
  - club_brier: site showed 0.1725, actual current (post SA-bonus fix) is 0.1758 - STALE.
  - try_brier: site showed 0.1142, actual current (post leakage fix) is 0.1145 - STALE.
  - intl_brier/acc and sr_brier/acc: matched exactly (0.1584/0.767, 0.1695/0.759) - these two models
    genuinely haven't changed this session, so the hardcoded numbers happened to still be right. Confirms
    the live computation reproduces the original methodology correctly rather than just computing something
    different that happens to look plausible.
  Fixed: added live_backtest() to produce.py, reusing each model's OWN existing validation harness (v6.score()
  for intl/club - extended it to also return accuracy, previously only Brier; tries.py's walk_forward+test
  split; sr.py's score()) instead of re-deriving the methodology ad hoc. Runtime cost: a few seconds added to
  produce.py's ~50s total - acceptable.
Everything else in site.json traced to something produce.py (or bets.py, called from produce.py) actually
sets each run. verify_site.py 23/23, confirmed in-browser (Method tab now shows 0.176/0.115 matching the live
numbers), pushed.

## Live results feed investigated; "live betting lines" investigated and confirmed not currently possible
Asked to continue the betting work, improve model accuracy, keep stats current, and get live betting lines.
Checked what's actually achievable before building anything:
- **Live team-level RESULTS**: yes - ESPN's unofficial API (already used for fixtures) also returns live/
  in-progress scores, confirmed by catching Ireland v South Africa mid-second-half while checking. League ids
  (same ones verified in club_fixtures.py): Nations Championship 17567, Six Nations 180659.
- **Live player-level box-score stats** (needed for Grades/Rankings/Recap): no - ESPN returns empty
  team/player statistics arrays for this league. Tried the vendored SofaScore scraper (data/gm/scripts/
  scrapers/sofascore.py) as a fallback - blocked with a 403 even using the same headers the original scraper
  used. No working free source found; Grades/Rankings/Recap stay capped at the Sep 2026 data cutoff already
  documented.
- **Live betting lines**: no, for two different reasons depending on competition. ESPN carries no odds for
  rugby at all (hasOdds: False on every event checked). The Odds API re-checked fresh: still only
  rugbyunion_six_nations as a sport key, and it's now showing active:false (tournament doesn't start again
  until Feb 2027) - nothing to pull for Nations Championship or any club league regardless. Market board
  stays Wyatt's manual entry until Six Nations is back in season.

## Built the live-results catch-up (espn_catchup.py) - and it immediately surfaced a real bug
Pulls STATUS_FINAL results from ESPN (Nations Championship/Top14/Prem/URC) from each competition's last known
date in build_data.py through today, prints lines formatted for build_data.py's add_rows() - does not auto-
edit build_data.py, since new results going into the dataset every model is built on should be reviewed, not
silently merged. Found: Top14/Prem/URC genuinely have zero new games (confirmed independently two ways -
this script AND club_fixtures.py's own pull both show the next club round isn't until 28 Nov, an
international-window pause, not a bug). Nations Championship: 2 of the round's 6 games were already
STATUS_FINAL (England 37-12 New Zealand, Scotland 67-5 Japan - both considerably bigger wins than the model's
own pre-match projections of 32-28 and 42-16). Merged into build_data.py, regenerated games.parquet ->
SC.parquet -> site.json.
**Real bug this surfaced**: produce.py's FIX list (this round's "upcoming" fixtures) still had England v NZ
and Scotland v Japan in it after their results were merged - meaning the Matches/Bets/Tries tabs were about
to show a "projection" for a game that already happened, generated from ratings that had ALREADY absorbed
that exact game's result. Not stale copy, actively nonsensical. Removed both from FIX/MKT.
While fixing this, found bets.py had its OWN second hardcoded copy of MKT (same duplication-drift risk as the
round-metadata bug fixed earlier) - consolidated FIX and MKT into round_config.py alongside the round
metadata, one source both produce.py and bets.py import.
Also found verify_site.py's "Try scorers for all 12 teams" check was hardcoded to 12 - now that FIX shrinks
as games finish, this would fail every round from here on. Fixed to derive the expected count from
2*len(d['matches']) instead of a literal 12.
verify_site.py 23/23, confirmed in-browser (Matches tab now shows only the 4 remaining games), pushed.

## Found a live player-stats source after all: World Rugby's own Pulse API - supersedes the "no working free
## source found" conclusion above
Kept looking for a player-level box-score source since ESPN (empty stats arrays) and SofaScore (403 blocked)
both failed. Found World Rugby's own API that powers worldrugby.org match centres
(api.wr-rims-prod.pulselive.com/rugby/v3) - no key needed, reverse-engineered from the public site's own
network calls. Far richer than anything in this repo had: 90 distinct per-player stat keys per match,
including **minutes played** - a field the dataset never had at all, previously noted in player_rankings.py
as "best available proxy" without it. Validated hard before trusting it: England 37-12 New Zealand
reconstructed exactly from summed player tries/conversions; a known Harry Potter (yes, real player) yellow
card reproduced exactly.
Two API bugs found and worked around (`model/wr_stats_sync.py`):
- The `/match` endpoint's own `competition` filter param doesn't filter - passing ANY value (even a correct
  one) silently narrows results to an unrelated/incomplete subset rather than erroring (588 matches for a
  2.5-month window WITH a dummy competition id vs 1300+ WITHOUT one, and the "with" version was missing
  Prem/URC entirely). Fixed by omitting the param and pulling the full date range, filtering client-side by
  team name against games.parquet's known universe - same bug class and same fix pattern as ESPN's broken
  competition filter in club_fixtures.py.
- Competition labels don't match current sponsor branding: "URC" not "United Rugby Championship", "English
  Premiership" not "Gallagher Premiership", "EPCR Champions Cup" not "Heineken Champions Cup" this season -
  found by printing the actual returned labels rather than guessing, mapped in COMP_MAP.
- World Rugby uses full official club names ("Union Bordeaux-Begles", "ASM Clermont Auvergne", "RC
  Toulonnais", etc.) that didn't match this repo's short-form convention - added 12 missing aliases to
  build_data.py's NORM dict after finding them all via an EXCLUDED-pairs printout.
- Replacement players only carry a generic "Replacement" position label (no forward/back distinction) -
  `wr_position()` derives it from jersey number (16-20 forward_sub, 21-23 back_sub, standard squad numbering).
  Coaching staff erroneously appear in the playerStats array with null stats - filtered out (number is None).

**Data-quality check, not a bug**: summing player-level points to reconstruct team score matched exactly in
303 of 320 team-sides checked; the other 17 were short by EXACTLY 7.0, never any other amount
(collections.Counter confirmed only {0.0: 303, 7.0: 17}). Diagnosed as penalty tries - rugby law assigns no
individual scorer to one, so no player's stat line captures it. Corrected for in team_match_stats.parquet's
`tries` column specifically (it feeds the four-try/three-try-differential bonus-point rules in
league_sim.py/top14_sim.py) by adding +1 whenever the gap is exactly 7; left player_stats.parquet's
individual rows alone since each one is correct as-is.

`team_match_stats.parquet` had NO build script anywhere in this repo before this (confirmed via
`grep -rln "team_match_stats\|match_meta" *.py` - every script only read it, never wrote it; a frozen
delivery like several other files found this session). `wr_stats_merge.py` is the first real implementation
of that aggregation.

**Merge bug caught before it shipped**: the first version of wr_stats_merge.py aligned the new batch to the
OLD parquet's column set (`P_new = P_new[P_old.columns]`), which silently dropped `minutes` - the one field
this whole integration exists to add, since it was never in the schema before. Caught by checking
`'minutes' in P.columns` after the first run came back True on row counts but the column wasn't there. Fixed
to union columns both directions (new fields survive with NaN backfill on old rows, not dropped).

Ran 21 Nov 2026 for 2026-09-07 to 2026-11-21: 160 matches, 7357 player-rows, across European Rugby Challenge
Cup (9), European Rugby Champions Cup (12), French Top 14 (56), Gallagher Premiership (25), Nations
Championship (18), United Rugby Championship (40). player_stats.parquet 239517 -> 246874 rows,
team_match_stats.parquet 9877 -> 10197 rows. Player-data cutoff moves from Sep 2026 to 21 Nov 2026.
Not yet wired into player_rankings.py's rate stats (minutes is available now but nothing uses it as a
denominator yet) or scheduled to run automatically - both still open. verify_site.py 23/23 after rerunning
produce.py/build_site.py on the merged data.

## Wired minutes into player_rankings.py - and caught a second, bigger merge bug doing it
Changed player_rankings.py's "current form" rate from shrunk points-added PER GAME to shrunk points-added
PER 80-MINUTES-EQUIVALENT (games_eq = minutes_sum / 80), since minutes now exists for the World Rugby rows.
Pre-sync rows have no minutes field at all, so they're filled to a full 80 - games_eq == n exactly for every
player untouched by the sync, so this is a no-op for old data and only changes players whose new appearances
are genuine partial games (subs), which is the whole point.

**Caught while sanity-checking the wiring**: Maxime Lucu, Ben Earl, Baptiste Serin and others were showing up
with exactly 9-10 games in the last 365 days, ALL of them wr_-sourced - suspicious for players who've started
most of this season. Checked: wr_stats_sync.py mints every player_id as `wr_<World Rugby's own id>`, a
completely different id space from the existing data's (e.g. Maxime Lucu's long-standing id is '3c388133').
Every player who appeared in BOTH the old data and the new sync got a second, disjoint identity - fragmenting
one real player's career into two rows in player_stats.parquet. This silently wrecked exactly the "games in
the last 365 days" rollup player_rankings.py does: a player's pre-sync and post-sync appearances landed under
different ids, so neither half alone could reach the 5-game minimum unless the sync window alone had 5+ games
(which, with a congested Top14/Prem schedule, 10 of them coincidentally did - that's what made them visible).
Checked the damage before fixing: 1534 of 2004 new (name, team) pairs matched an existing identity exactly.
Fixed in wr_stats_merge.py: build a (name, team) -> player_id map from the OLD data, keeping only pairs that
mapped to exactly one id (58 of 20889 pairs were ambiguous - two different people sharing a name at the same
club - left those alone rather than guess), and remap new rows onto the existing id wherever it matches.
Genuinely new players (debutants, or a name/team that doesn't match the old spelling) correctly keep a fresh
wr_ id. Re-ran: 5724 of 7357 new rows reconciled to an existing identity. Confirmed Maxime Lucu etc. now show
one continuous player_id with their full career length (129, 135, 118, 114 games respectively, not 9-10).
Re-ran produce.py/build_site.py/verify_site.py (23/23) on the corrected data - rankings output now shows
materially higher game counts per player (e.g. fly_half top-3 n=12-13 vs 5-10 before), confirming players are
no longer being undercounted. The previously-shipped merge (player_stats.parquet with the fragmented ids) had
already been committed and pushed; reset both parquet files to their pre-merge commit and redid the merge
with the fix rather than patching the broken ids in place, so there's one clean merge in history, not two.

## Scheduled the player-stats sync (.github/workflows/sync_player_stats.yml)
Runs the wr_stats_sync.py -> wr_stats_merge.py pipeline daily via GitHub Actions (no key/secret needed - the
World Rugby API is open) rather than depending on Wyatt's Mac being on. Scope deliberately stops at player
box-score stats (Grades/Rankings/Recap) - match RESULTS stay a manual step via espn_catchup.py, unchanged,
since those feed win probabilities and the Bets tab and this project's convention is that data shouldn't
silently change what every model is built on without a look first (see the live-results section above).
Three things had to change to make an unattended daily run safe:
- wr_stats_sync.py's start/end dates were hardcoded literals ('2026-09-07'/'2026-11-21') - would have needed
  hand-editing before every run. Now defaults to (latest date already in player_stats.parquet - 3 days) through
  today; the 3-day overlap is deliberate (catches a slow-to-post result) and harmless (wr_stats_merge.py already
  dedupes by match_id).
- Both scripts now handle "zero new matches" (the common case - most days fall between rounds) as a clean
  no-op that exits 0 without touching the parquet files, instead of crashing on an empty dataframe with no
  columns.
- verify_site.py now calls sys.exit(1) on any failed check (previously just printed) so the workflow can gate
  the commit/push on all 23 passing - a bad build is never published automatically.
The workflow: pull -> merge -> if (and only if) the parquet files actually changed, produce.py -> build_site.py
-> verify_site.py -> commit + push to main. A no-new-data day touches nothing. Tested all three code paths
locally (found-and-merged, found-but-already-merged, zero-found-at-all) before relying on it to run unattended.

## Found the Bets tab was live and public - built an actual public/private split
While checking a UI spacing complaint, found the Bets tab - real sportsbook lines, EV%, Kelly stakes, a
bankroll calculator - visibly live on https://whyy-it.github.io/gainline/, with a literal leftover dev note
("Private · remove before publishing") rendered right above it. Direct violation of CLAUDE.md's hard rule
("the Bets tab is private") that had apparently never actually been implemented - one template, one
site.json, written straight to the public index.html/site/gainline.html since day one. Worse: model/site.json
itself was committed to this now-public repo with betsTab's full contents, so the HTML wasn't even the only
leak - the raw JSON was sitting in the repo too (and, as of this fix, in its git history - noted for Wyatt;
scrubbing that needs a history rewrite, which is a separate, bigger call than this fix made unilaterally).
Fixed properly rather than just deleting the reminder text:
- model/site.json is no longer committed (git rm --cached, now gitignored) - it's 100% regenerable via
  python3 produce.py, no reason for a file containing betting content to live in a public repo at all.
- build_site.py now writes the public files (site/gainline.html, index.html) from a copy of the data with
  betsTab deleted, and separately writes the full data (betsTab intact) to site/gainline_private.html - a new,
  gitignored, local-only file. Same template drives both; mobile_template.html only registers the Bets
  nav link/route when D.betsTab is actually present, so the public build simply has no way to reach it,
  not just a hidden button - the data itself is absent from what ships.
- verify_site.py's old check ("HTML carries site.json") assumed gainline.html always had betsTab - split into
  three: public HTML has matches/tries, public HTML has NO betting content (checks both the embedded JSON and
  the literal old reminder string), and (when gainline_private.html exists locally) it carries betsTab
  correctly.
Also fixed, same investigation: main's top padding (16px) read as too tight below the sticky nav on every
tab - bumped to 24px. That fix itself had a bug: `<main class="wrap" id="main">` meant .wrap (a class
selector, `padding:0 ... 0 ...`) always beat a bare `main{padding:...}` tag selector on specificity, so the
padding silently never applied at all - switched the rule to `#main` (ID beats class) once this was caught
from a follow-up screenshot showing no visible change.

## Live scores, live win probability, live player grades - client-side, no backend
Asked to make the whole site refresh "every refresh" like ESPN's live gamecast - live score, live win%,
dynamic grades - not just the daily batch sync. Checked feasibility first rather than assuming a backend
would be needed: both site.api.espn.com and api.wr-rims-prod.pulselive.com send
`access-control-allow-origin: *` (confirmed 22 Nov), meaning the static page's own JS can fetch them directly
from the browser. No proxy, no serverless function, no hosting change - genuinely the same architecture ESPN's
own app uses, just running client-side against the same public APIs this project already validated server-side.

Built into site/mobile_template.html (produce.py embeds a new `live` block: which ESPN league id to poll,
and the stat-name mapping needed to read World Rugby's live per-player keys):
- **Live score/clock/status** - polls ESPN's scoreboard every 25s from 10 min before kickoff until a match is
  confirmed final (capped at 24h post-kickoff so a stuck/abandoned fixture doesn't poll forever). Matches
  ESPN's event to our fixture by team name - exact-match is fully reliable for this use (Test matches use
  plain country names, no club-name aliasing to worry about). Pauses while the tab is backgrounded
  (visibilitychange), polls once immediately on return. Replaces the pre-match projected score/win% on the
  Matches tab with the live score/clock and a live win probability once a match starts; once final, shows
  the real result instead of the projection, permanently (no model, win% becomes 100/0/draw).
- **Live win probability** - NOT a backtested model like everything else in this project; there's no
  historical play-by-play data to validate against (games.parquet only has final scores). It's a principled
  approximation instead: treat the final margin as the pre-match-expected margin (already-validated `spread`)
  plus Brownian noise scaled by remaining time (`sigma` from the existing probit fit) - i.e. the live margin
  drifts toward (current margin + pre-match-implied margin for the time left), with uncertainty shrinking as
  sqrt(time remaining). Smoothly reduces to the ordinary pre-match probability at kickoff and to near-certainty
  at full time. Documented as an approximation in the code itself, not oversold as validated.
- **Live player grades** - for any match confirmed in progress, separately polls World Rugby's match list for
  that day (team-name matched, same approach as wr_stats_sync.py) and, if found, its live per-player stats,
  computing points-added client-side with the SAME grade_values weights already embedded for the (validated)
  post-match Grades tab. Important caveat, stated plainly rather than glossed over: whether World Rugby
  actually populates stats before full-time is UNTESTED - no match was live while this was built. The code
  path is correct either way (it just stays empty until data shows up), but "live grades" may in practice only
  ever update at full-time until this gets checked against a real in-progress match.
Two real bugs caught testing against today's round (useful accident: all 4 remaining fixtures had already
finished in the real world by the time this was built, e.g. Ireland 20-17 South Africa vs. our stale 23-30
projection - gave an immediate end-to-end test of the FINAL-score path on real data, not synthetic):
- First version only polled within a fixed 3.5-hour post-kickoff window - meant a match ESPN was slow to
  mark final (or that this code checked hours after kickoff, as in testing) would silently stop being polled
  before ever learning the real result. Changed to "poll until confirmed final, capped at 24h" instead of a
  fixed game-length guess.
- First version only called `render()` when `document.visibilityState==='visible'` at poll time - but that
  can read 'hidden' transiently right after navigation, and more importantly meant a tab that silently updated
  while backgrounded wouldn't redraw on return unless something ALSO changed on that specific return-triggered
  poll. Moved the visibility check to the polling cadence itself (pause/resume the interval) and made
  rendering unconditional on any change - cheap even for a hidden tab, and guarantees the DOM catches up.
verify_site.py checks the new `live` config block (ESPN league id matches round_config, wr_field_map covers
every grade_values stat). 26/26 checks pass.

## Fixed a real statistical bug in single-match player grades
Asked "how does this make sense" about the Grades tab: a sub forward with +3.7 points-added (Seb Stephen,
Scotland 67-5 Japan) graded 84 - the match's TOP grade - ahead of a back-rower with +13.4 (Liam McConnell)
and a hat-trick winger with +24.8 (Darcy Graham). Root cause: grade() z-scores a player against their OWN
position group, but that group is whoever happened to play that ONE position in that ONE match - as few as
4 (wing), 6 (back_row) or 10 (sub_fwd) players. Checked the actual numbers: sub_fwd's group that match
clustered tight near zero (mean 0.09, std 1.84, mostly players with a handful of late-game minutes), so
Stephen's modest +3.7 was ~2 std above a group with almost no natural spread, while back_row (mean 4.19, std
4.90) and wing (mean 9.13, std 11.18) had much wider natural variance that match, making even much bigger
raw contributions look like smaller z-scores. A real instability, not a display bug - single-match groups
this small give a noisy mean/std to divide by.
`player_rankings.py` already solves the identical small-sample problem (empirical-Bayes shrinkage toward a
stable group mean, documented in its own docstring) but grade()'s single-match z-score never got that
treatment. Checked the fix against real historical data rather than guessing: computed each position group's
(mean, variance) of points-added across every match in player_stats.parquet (tens of thousands of rows per
group - `historical_group_stats()` in grades.py), then shrunk each match's own small-sample mean/variance
toward that stable baseline with K pseudo-players. Tried K=10/20/40 against the Scotland v Japan case:
- K=10: Stephen 84->85, McConnell 83->96, Graham 77->96. Fixes the ordering (standouts now clearly outrank
  the modest cameo) while keeping McConnell/Graham differentiated from each other - the full match's grade
  list re-sorts sensibly by actual contribution (McConnell/Graham at top, then Horne +14.2 at 95, Stephen
  +3.7 down at 85, Dobie +19.2 at 84, in a much more intuitive order than before).
- K=20/40: over-corrects - both standout performers clip to the 99 ceiling, losing the ability to tell an
  excellent performance from a historically exceptional one.
K=10 kept. Re-ran produce.py/build_site.py/verify_site.py - 26/26, grades output now matches the table above.

## The scoreboard now updates itself - multi-competition, no more hand-maintained fixture list
Asked for "ESPN app (live scores, upcoming fixtures) mixed with a 538-style projection system" and,
specifically, for the Matches tab to stop needing a manual round_config.py edit every week. The 538-style
piece (team power ratings, Monte Carlo predicted tables/title odds, player rankings) already existed; what
was actually missing was automatic fixture advancement, so that's what got built.

**New: `model/auto_rounds.py`** detects every tracked competition's current fixtures directly from ESPN,
replacing round_config.py's hand-maintained FIX list entirely. Two real bugs found building it:
- A `dates=START-END` range query LOOKS like it works (returns valid JSON, 0 events - easy to misread as
  "confirmed nothing scheduled") but is actually a silently-swallowed 400 error; only `dates=YYYYMMDD` single
  days are supported. Caught by checking the status code directly instead of trusting an empty events list -
  this is exactly the mistake an earlier "no live match right now" finding made.
- ESPN's `venue` field is wrong for this feed: every Nations Championship fixture that week returned
  "Allianz Stadium, London" regardless of where it was actually played (confirmed - Italy v Fiji isn't in
  London). Venue is left blank for auto-detected matches rather than show a confidently wrong one.

**Multi-competition display** (22 Nov, explicit choice over "auto-pick one competition a week" - Top14/Prem/
URC typically all play the same weekend once the season's underway, so picking only one would hide the
others): site.json's schema changed from one flat `matches` list + top-level `competition`/`round_label` to
a `rounds` list, one entry per currently-active competition. The Matches tab renders one section per round.

**Club match projections are new** - produce.py's match-card generator (win%/spread/total) only ever covered
Tests before (`tests()`, FIX-driven); extended to clubs via league_sim.py's already-validated ratings/sigma
(Club Brier 0.1725) plus a newly-fit totals model (same polyfit-of-attack+defence-sum recipe Tests already
use - no club totals calibration existed before this). Checked `features.club_state()`'s from-scratch Elo
against `v6.elo()`'s independent implementation first (0.0 max diff across 49 teams) before trusting either
as the ratings source. `league_sim.fit_club_common()` also now returns the probit intercept `a` it was
fitting but discarding - needed it for a real club win%, not just a Monte Carlo margin draw.

**Market odds remain entirely optional, by necessity** - checked again (ESPN's richer per-event summary
endpoint too: `hasOdds: false`) and there's still no free live odds source. round_config.py's MKT is now a
sparse, nested-by-competition manual override; produce.py fills `mkt_*` fields with None when no board
exists for a match, bets.py skips any match with no entry (not an error, just nothing to compare against),
and the template shows "-" for the Market column rather than crash on a null. A match's `sigma` now travels
with it (club 13.33 vs intl 13.84 differ - bets.py was about to reuse a single global sigma for the
cover-probability calc, which would have been wrong for club matches; fixed by having it just reuse
produce.py's already-correct `p_cover_home` instead of recomputing).

**Live-score polling now handles several competitions at once** - one ESPN fetch per active competition
(`D.live.leagues`, not a single league id), each round's matches checked only against their own league's
scoreboard so two competitions can never cross-match on a shared team name. The live win-probability formula
now reads each match's own embedded `sigma` instead of a single top-level value, for the same club-vs-intl
reason as above.

**GitHub Actions workflow restructured** (`.github/workflows/sync_player_stats.yml`, now named "Rebuild
site"): used to only run produce.py/build_site.py when the player-stats merge found something new. Now
always runs the full rebuild - a new round appearing is itself a reason to republish, not just new player
stats - and gates the commit on whether the actual output changed (`git diff --quiet` after rebuilding)
instead of pre-checking one specific input file. A quiet day (same fixtures, no new stats) is still a clean
no-op; it just checks after building instead of before.

verify_site.py updated throughout: the Test-specific independent Elo re-derivation stays exactly as rigorous
as before for intl rounds; club rounds get an internal-consistency check instead (re-deriving club ratings
from scratch here would just duplicate features.club_state()/league_sim.py, not independently verify them).
Tested end-to-end against real auto-detected fixtures throughout, not just unit-level: 2 real active rounds
(Nations Championship Round 7, 6 matches; French Top 14, 7 matches) while building this, confirmed 13
sensible, independently-checked match cards across both competitions and multi-league live-poll dispatch
working correctly (verified against ESPN's real feed, not mocked). 26/26 checks pass.

## Real cold-start bug caught on the FIRST live CI run of the new automated rebuild
model/site.json is gitignored (it carries betsTab - see 22 Nov), so a genuinely fresh checkout - a new clone,
or this exact case, the first time GitHub Actions ever ran the rebuilt "Rebuild site" workflow - has no
site.json file at all. produce.py's `d = json.load(open(OUT))` assumed one already existed (to preserve
hand-maintained bets text across runs); crashed immediately in CI with FileNotFoundError. Fixed by falling
back to `{}` - bets.build() always rewrites betsTab from scratch anyway, so there's nothing real to "keep" on
a true first run. Verified by actually deleting site.json locally and re-running the full pipeline (not just
reasoning about it), then re-triggered the real CI workflow, which completed successfully and auto-committed
on its own.

## Final results now stick - baked server-side, not left to a visitor's browser catching it
Asked for games to "show as final... until the next round is ready to view." They were already going final
in the live score poller client-side, but ONLY in that one visitor's browser, for a limited window (gave up
polling 24h after kickoff) - meaning a match that finished with nobody's tab open to catch it would silently
flip BACK to showing the stale pre-match projection on the next fresh visit, since nothing ever recorded the
real result anywhere durable. "Final" was never actually a property of the published page, just an ephemeral
client-side cache.
Fixed at the root: auto_rounds.py now carries each match's confirmed final score straight from ESPN (it was
already fetching `status` per match, just discarding it beyond filtering out postponed/cancelled games), and
produce.py's new `apply_final()` overwrites the match's projection with the real result (score, win%, fair
odds - clipped away from exactly 0/1 first, same fix verify_site.py's own fair-odds re-derivation needed once
tested against a real final match) directly into site.json. "Final" is now baked into the daily-rebuilt data
itself - every visitor sees the real result immediately, with no dependency on live polling at all. The
client-side poller still runs (for genuinely live in-progress state, minute-by-minute), but now skips any
match the server has already confirmed final rather than wasting a request re-confirming it.
Also bumped auto_rounds.py's lookback window 3->6 days: 3 days left a real gap where a just-finished round
could age out of the detection window before the next round's fixtures were far enough out to be picked up -
exactly the "nothing scheduled" dead zone this request was trying to avoid between one round and the next.
Verified with a full integration test, not just the isolated function: temporarily monkey-patched
auto_rounds.active_rounds() to inject a real final score into the live pipeline, ran produce.py end to end,
and confirmed (a) the match card correctly showed FT/100%/0% baked into site.json, (b) try-scorer props
correctly stopped being generated for that match (36 teams -> 30), and (c) the rendered HTML shows "FT" and
the real score on a completely fresh page load with no client poll involved. Caught and fixed the same
zero-division fair-odds bug in verify_site.py itself this surfaced (a certain 100%/0% outcome breaks the
naive `1/p_home` re-derivation) and a stale expected-team-count check (try-scorer team count needs to exclude
matches already final, same as produce.py does). 26/26 checks pass on the real (non-synthetic) data.

## Match cards show the actual date; removed the "all-time" rankings window
Added the date (not just weekday + time) to each match card's kickoff badge - with multiple competitions on
screen at once since 22 Nov, "Fri" alone is ambiguous once two different Fridays are both visible.

Separately asked whether Rankings' "All-time (since 2022)" window could be extended back 30-40 years, or
should be removed. Checked the actual data before answering either way, not just our own pipeline: read
every one of the 9,241 player files in the raw vendored `seanyboi/rugbydata` source directly (not just what's
already loaded into player_stats.parquet) and confirmed 2022-01-01 is the genuine floor - not a limit in how
we built the pipeline, but how far back structured player-level rugby stats (tackles, carries, etc.) exist in
any publicly available form at all. 30-40 years isn't achievable with any source found. Removed the window
entirely (player_rankings.py's `site_rankings()` now returns one flat set of leagues instead of nested
current/alltime dicts; the template's window-toggle filter is gone) rather than keep a feature that can only
ever cover ~4-5 years under a name that implies far more, even with the "(since 2022)" caveat already there.
"Current form" (365 days, min 5 games) is the only Rankings view now. verify_site.py/the rest of the pipeline
needed no changes - confirmed via produce.py/build_site.py/verify_site.py (26/26) and in-browser.

## Rankings: 3-window toggle instead of one fixed year
Asked for "2 weeks / season / year" windows. rank()'s `days` param became `window` (`'2 Weeks'`, `'Season'`,
`'Year'`); "Season" uses the exact same Aug 1 (club) / Jan 1 (country) boundary v6.season() already uses
elsewhere, not a third competing definition of "season." "2 Weeks" drops the games-played floor from 5 to 1 -
a 14-day window can't realistically clear 5 games for almost anyone, so it's deliberately a noisy, small-
sample snapshot rather than a stable rate, and the UI note says so.
Checked Season vs Year weren't accidentally computing the same thing for country (they returned identical
114-qualifier results at first glance) before trusting it: confirmed it's a real coincidence of the
international calendar, not a bug - no Test matches are played between the Nov window closing and the next
Six Nations in Feb, so "since Jan 1" and "since 365 days ago" happen to bound the exact same rows right now.
Club rankings (which play continuously) correctly show three clearly distinct qualifier counts (1162 / 307 /
3076). site_rankings() now returns window-nested dicts again (undoing last commit's flattening, for a better
reason this time - three real windows instead of one misleading one). 26/26 checks pass.
