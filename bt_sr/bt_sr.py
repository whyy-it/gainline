"""Walk-forward backtest of the Super Rugby method + Bets rules against real closing 1X2 odds, 2010-2019.
Data: lukepothier/bet results.csv (average closing odds, rows newest-first, no dates).
Season ends located by the known finals. Tune on 2011-13, test 2014-19 (never seen when tuned)."""
import numpy as np, pandas as pd, itertools, json
from scipy.optimize import minimize
from scipy.stats import norm

d = pd.read_csv('sr_odds_2011_19.csv').iloc[::-1].reset_index(drop=True)
d.columns = ['home', 'away', 'hs', 'as_', 'oh', 'od', 'oa']
ENDS = [168, 292, 417, 543, 669, 794, 936, 1078, 1205, 1331]
d['season'] = np.searchsorted(ENDS, d.index) + 2010
C = {**{t: 'NZ' for t in 'Blues Chiefs Crusaders Highlanders Hurricanes'.split()},
     **{t: 'AU' for t in 'Brumbies Waratahs Reds Rebels Force'.split()},
     **{t: 'SA' for t in 'Bulls Cheetahs Kings Lions Sharks Stormers'.split()}, 'Jaguares': 'AR', 'Sunwolves': 'JP'}
d['travel'] = [int(C[h] != C[a]) for h, a in zip(d.home, d.away)]
y = (d.hs > d.as_).astype(float).values + 0.5 * (d.hs == d.as_).values

# market: no-vig 3-way, then draw-no-bet
ih, idr, ia = 1 / d.oh, 1 / d.od, 1 / d.oa
d['vig'] = ih + idr + ia - 1
d['mkt'] = ih / (ih + ia)                               # DNB home prob, no vig
d['dnb_h'] = d.oh * (d.od - 1) / d.od                   # synthetic DNB price (back side, hedge draw)
d['dnb_a'] = d.oa * (d.od - 1) / d.od

def elo(K, hfa, tb, cap=40, reg=0.2):
    R = {}; pred = np.zeros(len(d)); last = None
    for j, r in enumerate(d.itertuples()):
        if last is not None and r.season != last:
            mu = np.mean(list(R.values())); R = {t: mu + (v - mu) * (1 - reg) for t, v in R.items()}
        last = r.season
        p = R.get(r.home, 0) - R.get(r.away, 0) + hfa + tb * r.travel; pred[j] = p
        e = np.clip(r.hs - r.as_, -cap, cap) - p
        R[r.home] = R.get(r.home, 0) + K * e; R[r.away] = R.get(r.away, 0) - K * e
    return pred

def pfit(x, t):
    f = lambda w: -np.mean(t * np.log(np.clip(norm.cdf(w[0] + w[1] * x), 1e-6, 1)) + (1 - t) * np.log(np.clip(1 - norm.cdf(w[0] + w[1] * x), 1e-6, 1)))
    return minimize(f, [0, .06], method='Nelder-Mead').x

def walk(pred, seasons):
    """probit fitted only on seasons before each one (from 2011; 2010 = burn-in)"""
    p = np.full(len(d), np.nan); W = {}
    for s in seasons:
        fit = (d.season < s) & (d.season >= 2011); w = pfit(pred[fit], y[fit]); W[s] = w
        sel = d.season == s; p[sel] = norm.cdf(w[0] + w[1] * pred[sel])
    return p, W

# tune on 2012-13 (probit for each fitted on earlier seasons)
best = None
for K, hfa, tb in itertools.product([0.02, 0.03, 0.04, 0.06], [1, 3, 5], [0, 2, 4]):
    pr = elo(K, hfa, tb); p, _ = walk(pr, [2012, 2013]); s = d.season.isin([2012, 2013])
    b = np.mean((p[s] - y[s]) ** 2)
    if best is None or b < best[0]: best = (b, K, hfa, tb)
print('tuned on 2012-13:', best)
_, K, hfa, tb = best
pred = elo(K, hfa, tb); TEST = list(range(2014, 2020)); p, W = walk(pred, TEST)
T = d.season.isin(TEST).values
d['model'] = p; d['pred'] = pred

bm = np.mean((p[T] - y[T]) ** 2); bk = np.mean((d.mkt.values[T] - y[T]) ** 2)
bl = np.mean((0.5 * p[T] + 0.5 * d.mkt.values[T] - y[T]) ** 2)
print(f'TEST 2014-19 n={T.sum()}  Brier model {bm:.4f}  market {bk:.4f}  50/50 blend {bl:.4f}   avg vig {d.vig[T].mean():.3f}')
for s in TEST:
    m = (d.season == s).values
    print(s, m.sum(), round(np.mean((p[m] - y[m]) ** 2), 4), round(np.mean((d.mkt.values[m] - y[m]) ** 2), 4))

# ---- bets rule (2-way ML = draw-no-bet) ----
rows = []
for j in np.where(T)[0]:
    r = d.iloc[j]; w = W[r.season]
    mkt_margin = (norm.ppf(np.clip(r.mkt, 1e-4, 1 - 1e-4)) - w[0]) / w[1]       # market's implied margin on model's scale
    for side, mp, kp, o, won in (('H', r.model, r.mkt, r.dnb_h, r.hs > r.as_), ('A', 1 - r.model, 1 - r.mkt, r.dnb_a, r.as_ > r.hs)):
        bp = 0.5 * mp + 0.5 * kp; ev = bp * o - 1
        gap = (r.pred - mkt_margin) * (1 if side == 'H' else -1)
        if ev < 0.05 or gap < 3: continue
        tier = 'Lean' if abs(mkt_margin) >= 20 else 'Bet'
        k = max((bp * o - 1) / (o - 1), 0) * 0.25 * (0.5 if tier == 'Bet' else 0.25)
        res = 0.0 if r.hs == r.as_ else (o - 1 if won else -1.0)
        rows.append(dict(season=r.season, home=r.home, away=r.away, side=side, odds=round(o, 3), blend=bp, ev=ev, gap=gap, tier=tier, stake=k, res=res,
                         mkt_margin=mkt_margin))
B = pd.DataFrame(rows); B.to_csv('bets_sr_2014_19.csv', index=False)

def summ(b, label):
    if len(b) == 0: print(label, 'no bets'); return
    roi = b.res.mean(); rng = np.random.default_rng(0)
    bs = [rng.choice(b.res.values, len(b)).mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [5, 95]); p10 = np.mean(np.array(bs) >= 0.10)
    kel = (b.stake * b.res).sum()
    print(f'{label:28s} n={len(b):4d}  win {np.mean(b.res > 0):.1%}  avg odds {b.odds.mean():.2f}  flat ROI {roi:+.1%}  90% CI [{lo:+.1%}, {hi:+.1%}]  P(ROI>=10%) {p10:.0%}  Kelly bankroll {kel:+.1%}')
summ(B, 'All (Bet+Lean)'); summ(B[B.tier == 'Bet'], 'Bet tier only')
summ(B[B.odds >= 2], 'Underdogs (>= +100)'); summ(B[B.odds < 2], 'Favourites (< +100)')
for s in TEST: summ(B[B.season == s], f'  {s}')
# no-blend raw model edge for comparison
raw = []
for j in np.where(T)[0]:
    r = d.iloc[j]
    for mp, o, won in ((r.model, r.dnb_h, r.hs > r.as_), (1 - r.model, r.dnb_a, r.as_ > r.hs)):
        if mp * o - 1 >= 0.05: raw.append(0.0 if r.hs == r.as_ else (o - 1 if won else -1.0))
print(f'Raw model EV>=5% (no blend, no gap rule): n={len(raw)} flat ROI {np.mean(raw):+.1%}')
json.dump(dict(K=K, hfa=hfa, travel=tb, brier_model=bm, brier_market=bk, brier_blend=bl), open('bt_sr_summary.json', 'w'), indent=1)
