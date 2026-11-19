"""Walk-forward test: train on everything before each test season, predict that season. Compare Brier vs v1 Elo."""
import numpy as np, pandas as pd, warnings
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from features import intl_state, club_state
warnings.filterwarnings('ignore')

T1 = {'England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji'}

def season_of(d, kind):
    return d.year if kind == 'intl' else (d.year if d.month >= 8 else d.year - 1)

def brier(p, y): return float(np.mean((p - y) ** 2))
def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def evaluate(S, kind, feats_sets, burn, tests, mask_fn=None):
    S = S.copy(); S['season'] = [season_of(d, kind) for d in S.date]
    S['y'] = (S.margin > 0).astype(float) + 0.5 * (S.margin == 0)
    out = {k: [] for k in feats_sets}; out['v1_elo'] = []; ys = []
    for ts in tests:
        tr = S[(S.season >= burn) & (S.season < ts)]; te = S[S.season == ts]
        if mask_fn is not None: te = te[mask_fn(te)]
        if len(te) == 0: continue
        ys.append(te.y.values)
        sig = (tr.margin - tr.elo).std()
        out['v1_elo'].append(1 - norm.cdf(0.5, te.elo, sig))
        for name, (feats, model) in feats_sets.items():
            if model == 'lin':
                m = LinearRegression().fit(tr[feats], tr.margin.clip(-50, 50))
                mu = m.predict(te[feats]); s = (tr.margin.clip(-50, 50) - m.predict(tr[feats])).std()
                out[name].append(1 - norm.cdf(0.5, mu, s))
            elif model == 'logit':
                m = LogisticRegression(C=1.0, max_iter=1000).fit(tr[feats], (tr.margin > 0))
                out[name].append(m.predict_proba(te[feats])[:, 1])
            elif model == 'gbm':
                m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40)
                m.fit(tr[feats], tr.margin.clip(-50, 50)); mu = m.predict(te[feats])
                s = (tr.margin.clip(-50, 50) - m.predict(tr[feats])).std() * 1.05
                out[name].append(1 - norm.cdf(0.5, mu, s))
    y = np.concatenate(ys)
    res = {k: dict(brier=brier(np.concatenate(v), y), logloss=ll(np.concatenate(v), y)) for k, v in out.items()}
    return pd.DataFrame(res).T.sort_values('brier'), len(y), y.mean()

BASE = ['elo']
ALL = ['elo', 'ad', 'rest', 'travel', 'form', 'team_hfa']
SETS = {
    'elo_lin': (BASE, 'lin'),
    '+att/def': (['elo', 'ad'], 'lin'),
    '+rest': (['elo', 'rest'], 'lin'),
    '+travel': (['elo', 'travel'], 'lin'),
    '+form': (['elo', 'form'], 'lin'),
    '+team_hfa': (['elo', 'team_hfa'], 'lin'),
    'all_lin': (ALL, 'lin'),
    'all_logit': (ALL, 'logit'),
    'all_gbm': (ALL, 'gbm'),
}

if __name__ == '__main__':
    pd.set_option('display.width', 200)
    SI, *_ = intl_state(); SC, *_ = club_state()
    r, n, base = evaluate(SC, 'club', SETS, burn=2020, tests=[2023, 2024, 2025, 2026])
    print(f'CLUB out-of-sample, {n} games, home win rate {base:.3f}'); print(r.round(4))
    t1 = lambda te: te.home.isin(T1) & te.away.isin(T1)
    r, n, base = evaluate(SI, 'intl', SETS, burn=2022, tests=[2024, 2025, 2026], mask_fn=t1)
    print(f'\nINTL tier-1 out-of-sample, {n} games, home win rate {base:.3f}'); print(r.round(4))
