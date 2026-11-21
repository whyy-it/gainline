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

def fit_blend(model_p, mkt_p, t):
    """Closed-form least-squares weight on [model_p vs mkt_p] minimizing Brier, clipped to a true blend (0-1)."""
    a = model_p - mkt_p; b = mkt_p - t
    w = -np.sum(a * b) / np.sum(a * a)
    return float(np.clip(w, 0.0, 1.0))

def walk(pred, seasons):
    """probit fitted only on seasons before each one (from 2011; 2010 = burn-in)"""
    p = np.full(len(d), np.nan); W = {}
    for s in seasons:
        fit = (d.season < s) & (d.season >= 2011); w = pfit(pred[fit], y[fit]); W[s] = w
        sel = d.season == s; p[sel] = norm.cdf(w[0] + w[1] * pred[sel])
    return p, W

def pfit_shrink(x, t):
    """Probit w0 + w1*x - w2*x*|x|, w2 bounded >=0 (shrink-only): dampens confidence specifically at
    large |predicted margin| (big favourite/underdog), leaving near-even games ~unchanged."""
    f = lambda w: -np.mean(t * np.log(np.clip(norm.cdf(w[0] + w[1] * x - w[2] * x * np.abs(x)), 1e-6, 1)) +
                            (1 - t) * np.log(np.clip(1 - norm.cdf(w[0] + w[1] * x - w[2] * x * np.abs(x)), 1e-6, 1)))
    return minimize(f, [0, .06, 0], method='L-BFGS-B', bounds=[(None, None), (None, None), (0, None)]).x

def walk_shrink(pred, seasons):
    p = np.full(len(d), np.nan); W = {}
    for s in seasons:
        fit = (d.season < s) & (d.season >= 2011); w = pfit_shrink(pred[fit], y[fit]); W[s] = w
        sel = d.season == s; p[sel] = norm.cdf(w[0] + w[1] * pred[sel] - w[2] * pred[sel] * np.abs(pred[sel]))
    return p, W

# tune on 2012-13 (probit for each fitted on earlier seasons)
best = None
for K, hfa, tb in itertools.product([0.02, 0.03, 0.04, 0.06], [1, 3, 5], [0, 2, 4]):
    pr = elo(K, hfa, tb); p, _ = walk(pr, [2012, 2013]); s = d.season.isin([2012, 2013])
    b = np.mean((p[s] - y[s]) ** 2)
    if best is None or b < best[0]: best = (b, K, hfa, tb)
print('tuned on 2012-13:', best)
_, K, hfa, tb = best

# fit the blend weight on the SAME tune seasons (2012-13) used for K/hfa/tb - never on 2014-19 test
pred_tune = elo(K, hfa, tb); p_tune, _ = walk(pred_tune, [2012, 2013]); s_tune = d.season.isin([2012, 2013]).values
W_BLEND = fit_blend(p_tune[s_tune], d.mkt.values[s_tune], y[s_tune])
print(f'blend weight fitted on 2012-13 (model vs market): {W_BLEND:.3f}  (0.5 = even blend, 0 = pure market)')

pred = elo(K, hfa, tb); TEST = list(range(2014, 2020)); p, W = walk(pred, TEST)
T = d.season.isin(TEST).values
d['model'] = p; d['pred'] = pred

bm = np.mean((p[T] - y[T]) ** 2); bk = np.mean((d.mkt.values[T] - y[T]) ** 2)
bl = np.mean((0.5 * p[T] + 0.5 * d.mkt.values[T] - y[T]) ** 2)
bl_fit = np.mean((W_BLEND * p[T] + (1 - W_BLEND) * d.mkt.values[T] - y[T]) ** 2)
print(f'TEST 2014-19 n={T.sum()}  Brier model {bm:.4f}  market {bk:.4f}  50/50 blend {bl:.4f}  fitted-weight blend {bl_fit:.4f}   avg vig {d.vig[T].mean():.3f}')
for s in TEST:
    m = (d.season == s).values
    print(s, m.sum(), round(np.mean((p[m] - y[m]) ** 2), 4), round(np.mean((d.mkt.values[m] - y[m]) ** 2), 4))

# ---- bets rule (2-way ML = draw-no-bet), run at a given blend weight w ----
def make_bets(w):
    rows = []
    for j in np.where(T)[0]:
        r = d.iloc[j]; ww = W[r.season]
        mkt_margin = (norm.ppf(np.clip(r.mkt, 1e-4, 1 - 1e-4)) - ww[0]) / ww[1]   # market's implied margin on model's scale
        for side, mp, kp, o, won in (('H', r.model, r.mkt, r.dnb_h, r.hs > r.as_), ('A', 1 - r.model, 1 - r.mkt, r.dnb_a, r.as_ > r.hs)):
            bp = w * mp + (1 - w) * kp; ev = bp * o - 1
            gap = (r.pred - mkt_margin) * (1 if side == 'H' else -1)
            if ev < 0.05 or gap < 3: continue
            tier = 'Lean' if abs(mkt_margin) >= 20 else 'Bet'
            k = max((bp * o - 1) / (o - 1), 0) * 0.25 * (0.5 if tier == 'Bet' else 0.25)
            res = 0.0 if r.hs == r.as_ else (o - 1 if won else -1.0)
            rows.append(dict(season=r.season, home=r.home, away=r.away, side=side, odds=round(o, 3), blend=bp, ev=ev, gap=gap, tier=tier, stake=k, res=res,
                             mkt_margin=mkt_margin))
    return pd.DataFrame(rows)

def summ(b, label):
    if len(b) == 0: print(label, 'no bets'); return
    roi = b.res.mean(); rng = np.random.default_rng(0)
    bs = [rng.choice(b.res.values, len(b)).mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [5, 95]); p10 = np.mean(np.array(bs) >= 0.10)
    kel = (b.stake * b.res).sum()
    print(f'{label:28s} n={len(b):4d}  win {np.mean(b.res > 0):.1%}  avg odds {b.odds.mean():.2f}  flat ROI {roi:+.1%}  90% CI [{lo:+.1%}, {hi:+.1%}]  P(ROI>=10%) {p10:.0%}  Kelly bankroll {kel:+.1%}')

B50 = make_bets(0.5)
print('\n-- 50/50 blend (old version) --')
summ(B50, 'All (Bet+Lean)'); summ(B50[B50.tier == 'Bet'], 'Bet tier only')

B = make_bets(W_BLEND)
print(f'\n-- fitted {W_BLEND:.3f}/{1 - W_BLEND:.3f} blend (new version) --')
summ(B, 'All (Bet+Lean)'); summ(B[B.tier == 'Bet'], 'Bet tier only')
summ(B[B.odds >= 2], 'Underdogs (>= +100)'); summ(B[B.odds < 2], 'Favourites (< +100)')
for s in TEST: summ(B[B.season == s], f'  {s}')
B.to_csv('bets_sr_2014_19.csv', index=False)
# no-blend raw model edge for comparison
raw = []
for j in np.where(T)[0]:
    r = d.iloc[j]
    for mp, o, won in ((r.model, r.dnb_h, r.hs > r.as_), (1 - r.model, r.dnb_a, r.as_ > r.hs)):
        if mp * o - 1 >= 0.05: raw.append(0.0 if r.hs == r.as_ else (o - 1 if won else -1.0))
print(f'Raw model EV>=5% (no blend, no gap rule): n={len(raw)} flat ROI {np.mean(raw):+.1%}')

# ---- underdog over-confidence fix: shrink the probit at large |margin|, walk-forward, test on 2014-19 ----
p_shrink, W_shrink = walk_shrink(pred, TEST)
d['model_shrink'] = p_shrink
bm_shrink = np.mean((p_shrink[T] - y[T]) ** 2)
print(f'\nUnderdog-shrink probit: Brier {bm_shrink:.4f}  vs plain-probit model {bm:.4f}  vs market {bk:.4f}')
for s in TEST: print(f'  {s} w2 (shrink strength): {W_shrink[s][2]:.5f}')

def make_bets_shrink(w):
    rows = []
    for j in np.where(T)[0]:
        r = d.iloc[j]; ww = W[r.season]  # gap/tier still keyed off the plain-probit margin inversion, for consistency
        mkt_margin = (norm.ppf(np.clip(r.mkt, 1e-4, 1 - 1e-4)) - ww[0]) / ww[1]
        for side, mp, kp, o, won in (('H', r.model_shrink, r.mkt, r.dnb_h, r.hs > r.as_), ('A', 1 - r.model_shrink, 1 - r.mkt, r.dnb_a, r.as_ > r.hs)):
            bp = w * mp + (1 - w) * kp; ev = bp * o - 1
            gap = (r.pred - mkt_margin) * (1 if side == 'H' else -1)
            if ev < 0.05 or gap < 3: continue
            tier = 'Lean' if abs(mkt_margin) >= 20 else 'Bet'
            k = max((bp * o - 1) / (o - 1), 0) * 0.25 * (0.5 if tier == 'Bet' else 0.25)
            res = 0.0 if r.hs == r.as_ else (o - 1 if won else -1.0)
            rows.append(dict(season=r.season, home=r.home, away=r.away, side=side, odds=round(o, 3), blend=bp, ev=ev, gap=gap, tier=tier, stake=k, res=res,
                             mkt_margin=mkt_margin))
    return pd.DataFrame(rows)

Bsh = make_bets_shrink(0.5)
print(f'\n-- underdog-shrink probit, 50/50 blend (new version) -- ({len(Bsh)} bets vs {len(B)} plain-probit)')
summ(Bsh, 'All (Bet+Lean)'); summ(Bsh[Bsh.tier == 'Bet'], 'Bet tier only')
summ(Bsh[Bsh.odds >= 2], 'Underdogs (>= +100)'); summ(Bsh[Bsh.odds < 2], 'Favourites (< +100)')
for s in TEST: summ(Bsh[Bsh.season == s], f'  {s}')
Bsh.to_csv('bets_sr_2014_19_shrink.csv', index=False)

json.dump(dict(K=K, hfa=hfa, travel=tb, brier_model=bm, brier_market=bk, brier_blend_5050=bl, blend_weight_fitted=W_BLEND,
                brier_blend_fitted=bl_fit, brier_model_shrink=bm_shrink,
                roi_plain=float(B.res.mean()), roi_shrink=float(Bsh.res.mean()) if len(Bsh) else None,
                n_bets_plain=len(B), n_bets_shrink=len(Bsh)), open('bt_sr_summary.json', 'w'), indent=1)
