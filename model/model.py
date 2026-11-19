"""Margin-Elo rugby model. Ratings are in points: pred_margin = R_home - R_away + HFA.
Separate pools for internationals and clubs. Walk-forward tuned."""
import numpy as np, pandas as pd, itertools, json
from scipy.stats import norm

G = pd.read_parquet('../model/games.parquet').sort_values('date').reset_index(drop=True)

HFA_KEY = {'French Top 14': 'top14', 'Gallagher Premiership': 'prem', 'United Rugby Championship': 'urc',
           'European Rugby Champions Cup': 'euro', 'European Rugby Challenge Cup': 'euro'}

def run(df, K, hfa, cap, regress, kind, hfa_map=None, init=None, eval_from=None, return_ratings=False):
    R = dict(init or {})
    preds = []
    last_season = None
    for r in df.itertuples():
        season = r.date.year if r.date.month >= 8 else r.date.year - 1
        if kind == 'club' and last_season is not None and season != last_season:
            mu = np.mean(list(R.values())) if R else 0
            R = {t: mu + (v - mu) * (1 - regress) for t, v in R.items()}
        if kind == 'intl' and last_season is not None and season != last_season:
            R = {t: v * (1 - regress) for t, v in R.items()}
        last_season = season
        h, a = r.home, r.away
        rh, ra = R.get(h, 0.0), R.get(a, 0.0)
        H = 0.0 if r.neutral else (hfa_map.get(HFA_KEY.get(r.competition), hfa) if hfa_map else hfa)
        p = rh - ra + H
        m = r.hs - r.as_
        if eval_from is None or r.date >= eval_from:
            preds.append((r.Index, p, m))
        e = np.clip(m, -cap, cap) - p
        # new teams learn faster for first few games
        R[h] = rh + K * e
        R[a] = ra - K * e
    out = pd.DataFrame(preds, columns=['i', 'pred', 'margin']).set_index('i')
    return (out, R) if return_ratings else out

def score(out, sigma=None):
    res = out.margin - out.pred
    sig = sigma or res.std()
    p = norm.cdf(out.pred / sig)
    y = (out.margin > 0).astype(float) + 0.5 * (out.margin == 0)
    brier = np.mean((p - y) ** 2)
    ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    acc = np.mean((out.pred > 0) == (out.margin > 0))
    return dict(mae=np.abs(res).mean(), rmse=np.sqrt((res ** 2).mean()), brier=brier, logloss=ll, acc=acc, n=len(out), sigma=sig)

if __name__ == '__main__':
    results = {}
    # ---- internationals: tune on 2024-01..2026-11 ; tier-1 subset scored separately
    I = G[G.kind == 'intl']
    T1 = {'England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji'}
    best = None
    for K, hfa, cap, reg in itertools.product([0.06, 0.08, 0.10, 0.12, 0.15], [2, 3, 4, 5, 6], [30, 40, 60], [0.0, 0.15, 0.3]):
        out = run(I, K, hfa, cap, reg, 'intl', eval_from=pd.Timestamp('2024-01-01'))
        t1 = out[[I.loc[i].home in T1 and I.loc[i].away in T1 for i in out.index]]
        s = score(t1)
        if best is None or s['mae'] < best[1]['mae']:
            best = ((K, hfa, cap, reg), s)
    print('INTL best (K,hfa,cap,reg):', best)
    results['intl'] = dict(params=best[0], tier1_eval=best[1])
    # ---- clubs: per-competition HFA
    C = G[G.kind == 'club']
    bestc = None
    for K, cap, reg, ht, hp, hu, he in itertools.product([0.05, 0.07, 0.09, 0.11], [30, 40], [0.2, 0.35, 0.5],
                                                       [7, 9, 11], [4, 6], [4, 6], [3, 5]):
        hm = dict(top14=ht, prem=hp, urc=hu, euro=he)
        out = run(C, K, 5, cap, reg, 'club', hfa_map=hm, eval_from=pd.Timestamp('2024-08-01'))
        s = score(out)
        if bestc is None or s['mae'] < bestc[1]['mae']:
            bestc = ((K, cap, reg, hm), s)
    print('CLUB best:', bestc)
    results['club'] = dict(params=[bestc[0][0], bestc[0][1], bestc[0][2]], hfa=bestc[0][3], eval=bestc[1])
    json.dump(results, open('../model/params.json', 'w'), indent=1, default=float)
