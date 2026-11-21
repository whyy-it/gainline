"""v6 experiments harness. Same split as tune.py: validation = 2024 season, test = 2025-26 onward.
Win prob = probit(a + b*pred) calibrated on the 3 seasons before each split."""
import numpy as np, pandas as pd, warnings
from scipy.optimize import minimize
from scipy.stats import norm
from model import G
warnings.filterwarnings('ignore')
T1 = {'England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji'}
KEY = {'French Top 14':'top14','Gallagher Premiership':'prem','United Rugby Championship':'urc',
       'European Rugby Champions Cup':'euro','European Rugby Challenge Cup':'euro'}
CHFA = dict(top14=9, prem=7, urc=6, euro=6)
SA_CLUBS = {'Bulls', 'Lions', 'Sharks', 'Stormers'}

def season(d, kind): return d.year if kind == 'intl' else (d.year if d.month >= 8 else d.year - 1)

def intl_games(long_history=False):
    I = G[G.kind == 'intl'].copy()
    if not long_history: return I.reset_index(drop=True)
    k = pd.read_csv('../data/kyejones25-alt_rugby-sql-analysis/data/internationals.csv')
    k['date'] = pd.to_datetime(k.date)
    k = k[(k.date >= '1995-07-01') & (k.date < '2022-06-01')]
    k = pd.DataFrame(dict(date=k.date, kind='intl', competition=k.competition, city=k.city, home=k.home_team, away=k.away_team,
                          hs=k.home_score.astype(float), as_=k.away_score.astype(float), neutral=k.neutral.astype(bool)))
    I = I[~((I.date < '2022-06-01') & I.home.isin(T1) & I.away.isin(T1))]
    return pd.concat([k, I], ignore_index=True).sort_values('date').reset_index(drop=True)

def elo(df, kind, K, hfa, cap, regress, weight=None, target=None, hfa_map=None, sa_bonus=0.0):
    """weight: per-row multiplier on K. target: per-row margin to learn from (default actual margin).
    sa_bonus: extra home-side edge when exactly one of home/away is an SA club (crossing into/out of SA
    altitude+travel), added on top of hfa_map - mirrors sr.py's travel_bonus for Super Rugby border crossings."""
    R = {}; preds = np.zeros(len(df)); last = None
    W = np.ones(len(df)) if weight is None else np.asarray(weight, float)
    TG = (df.hs - df.as_).values if target is None else np.asarray(target, float)
    for j, r in enumerate(df.itertuples()):
        s = season(r.date, kind)
        if last is not None and s != last:
            mu = np.mean(list(R.values())) if (kind == 'club' and R) else 0.0
            R = {t: mu + (v - mu) * (1 - regress) for t, v in R.items()}
        last = s
        H = 0.0 if r.neutral else (hfa_map[KEY[r.competition]] if hfa_map else hfa)
        if not r.neutral and sa_bonus and ((r.home in SA_CLUBS) != (r.away in SA_CLUBS)):
            H += sa_bonus
        p = R.get(r.home, 0) - R.get(r.away, 0) + H; preds[j] = p
        e = np.clip(TG[j], -cap, cap) - p
        R[r.home] = R.get(r.home, 0) + K * W[j] * e; R[r.away] = R.get(r.away, 0) - K * W[j] * e
    return preds, R

def probit_fit(x, y):
    f = lambda w: -np.mean(y * np.log(np.clip(norm.cdf(w[0] + w[1] * x), 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - norm.cdf(w[0] + w[1] * x), 1e-6, 1)))
    return minimize(f, [0, 0.06], method='Nelder-Mead').x

def score(df, kind, pred, mask=None, extra=None):
    """Brier on val (2024) and test (2025+). extra: optional DataFrame of extra features -> logistic stack."""
    y = (df.hs > df.as_).astype(float).values + 0.5 * (df.hs == df.as_).values
    ss = np.array([season(d, kind) for d in df.date]); m = np.ones(len(df), bool) if mask is None else mask
    out = {}
    for name, seas in (('val', [2024]), ('test', [2025, 2026])):
        sel = m & np.isin(ss, seas); fit = m & (ss < min(seas)) & (ss >= min(seas) - 3)
        if extra is None:
            w = probit_fit(pred[fit], y[fit]); p = norm.cdf(w[0] + w[1] * pred[sel])
        else:
            from sklearn.linear_model import LogisticRegression
            X = np.c_[pred, extra.values]
            lr = LogisticRegression(max_iter=2000).fit(X[fit], y[fit] > 0.5); p = lr.predict_proba(X[sel])[:, 1]
        out[name] = round(float(np.mean((p - y[sel]) ** 2)), 4); out[name + '_n'] = int(sel.sum())
    return out

def t1_mask(df): return (df.home.isin(T1) & df.away.isin(T1)).values
