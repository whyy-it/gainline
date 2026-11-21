"""Test the SA-travel angle from angles.py: SA club at home vs non-SA away beats Elo by ~5.9 pts (n=179,
every season 2021-26 positive); the reverse direction (SA away in Europe) is weaker/noisier (+2.7, inconsistent
by season). Try adding a direction-agnostic sa_bonus to the club HFA (mirrors sr.py's travel_bonus for Super
Rugby border crossings). Same split as v6.py: pick sa_bonus on val (2024) only at full precision (v6.score()
rounds to 4dp, too coarse here), report test (2025-26) with the val-chosen value - never touch test to choose."""
import numpy as np, warnings
from scipy.stats import norm
from v6 import elo, CHFA, season, probit_fit
from model import G
warnings.filterwarnings('ignore')

C = G[G.kind == 'club'].reset_index(drop=True)
y = (C.hs > C.as_).astype(float).values + 0.5 * (C.hs == C.as_).values
ss = np.array([season(d, 'club') for d in C.date])

def brier(pred, seas):
    sel = np.isin(ss, seas); fit = (ss < min(seas)) & (ss >= min(seas) - 3)
    w = probit_fit(pred[fit], y[fit]); p = norm.cdf(w[0] + w[1] * pred[sel])
    return float(np.mean((p - y[sel]) ** 2)), int(sel.sum())

GRID = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
rows = []
for b in GRID:
    pred, _ = elo(C, 'club', 0.06, 5, 40, 0.2, hfa_map=CHFA, sa_bonus=b)
    v, vn = brier(pred, [2024]); t, tn = brier(pred, [2025, 2026])
    rows.append((b, v, vn, t, tn))
    print(f'  sa_bonus={b:4.1f}  val={v:.6f} (n={vn})   test={t:.6f} (n={tn}) [test held out from selection]')

best = min(rows, key=lambda r: r[1])
base = rows[0]
print(f'\nChosen on val only: sa_bonus={best[0]}  (val {best[1]:.6f} vs baseline val {base[1]:.6f})')
print(f'\nTEST (2025-26), decided before looking here:')
print(f'  baseline (sa_bonus=0):      test Brier {base[3]:.6f}  (n={base[4]})')
print(f'  sa_bonus={best[0]} (val-chosen): test Brier {best[3]:.6f}  (n={best[4]})')
d = base[3] - best[3]
print(f'  delta: {d:+.6f} ({"better" if d > 0 else "worse" if d < 0 else "no change"})')
print(f'\nFor comparison, the raw Elo-residual angle (no probability calibration) was ~6 pts of MARGIN and highly')
print(f'significant (angles.py). The probability-calibration gain here is much smaller - the adaptive Elo ratings')
print(f'already partly absorb this pattern season over season via normal K-updates; an explicit bonus only adds')
print(f'the residual on top of what rating drift has already learned.')
