"""Fine-tuning with a strict split: params tuned on validation seasons, scored once on the held-out test.
Match model upgrades tested:
  - margin transform: update on sign(m)*|m|^alpha (diminishing returns on blowouts)
  - new-team boost: larger K for a team's first N games
  - per-team home advantage learned online (shrunk)
  - probit calibration: P(win) = Phi(a + b*pred) instead of fixed sigma
Totals: tune attack/defence learning rate + shrink to league mean.
"""
import numpy as np, pandas as pd, json, warnings
from scipy.optimize import minimize
from scipy.stats import norm
from model import G
warnings.filterwarnings('ignore')
KEY = {'French Top 14':'top14','Gallagher Premiership':'prem','United Rugby Championship':'urc',
       'European Rugby Champions Cup':'euro','European Rugby Challenge Cup':'euro'}

def season(d, kind): return d.year if kind == 'intl' else (d.year if d.month >= 8 else d.year - 1)

def elo2(df, kind, K, hfa, alpha, regress, newK, team_hfa_w, hfa_lg=None):
    R, n, th = {}, {}, {}
    preds = np.zeros(len(df)); last = None
    for j, r in enumerate(df.itertuples()):
        s = season(r.date, kind)
        if last is not None and s != last:
            mu = np.mean(list(R.values())) if (kind == 'club' and R) else 0.0
            R = {t: mu + (v - mu) * (1 - regress) for t, v in R.items()}
        last = s
        h, a = r.home, r.away
        base = (hfa_lg[KEY[r.competition]] if hfa_lg else hfa)
        th_h = th.get(h, (0., 0)); team_adj = team_hfa_w * th_h[0] / (th_h[1] + 8)
        H = 0.0 if r.neutral else base + team_adj
        p = R.get(h, 0) - R.get(a, 0) + H
        preds[j] = p
        m = r.hs - r.as_
        e = np.sign(m) * abs(m) ** alpha - np.sign(p) * abs(p) ** alpha
        kh = K * (1 + newK * (n.get(h, 0) < 8)); ka = K * (1 + newK * (n.get(a, 0) < 8))
        R[h] = R.get(h, 0) + kh * e; R[a] = R.get(a, 0) - ka * e
        n[h] = n.get(h, 0) + 1; n[a] = n.get(a, 0) + 1
        if not r.neutral:
            s_, c_ = th_h; th[h] = (s_ + (m - p), c_ + 1)
    return preds

def probit_fit(pred, y):
    f = lambda w: -np.mean(y * np.log(np.clip(norm.cdf(w[0] + w[1] * pred), 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - norm.cdf(w[0] + w[1] * pred), 1e-6, 1)))
    return minimize(f, [0, 0.06], method='Nelder-Mead').x

def brier(p, y): return float(np.mean((p - y) ** 2))

def evaluate(kind, mask_eval, val_seasons, test_seasons, x, hfa_lg_keys=False):
    df = G[G.kind == kind].reset_index(drop=True)
    y = (df.hs > df.as_).astype(float).values + 0.5 * (df.hs == df.as_).values
    if hfa_lg_keys:
        K, alpha, regress, newK, tw, h1, h2, h3, h4 = x
        pr = elo2(df, kind, K, 0, alpha, regress, newK, tw, dict(top14=h1, prem=h2, urc=h3, euro=h4))
    else:
        K, hfa, alpha, regress, newK, tw = x
        pr = elo2(df, kind, K, hfa, alpha, regress, newK, tw)
    ss = np.array([season(d, kind) for d in df.date]); m = mask_eval(df)
    out = {}
    for name, seas in (('val', val_seasons), ('test', test_seasons)):
        sel = m & np.isin(ss, seas)
        fit_sel = m & (ss < min(seas)) & (ss >= min(seas) - 3)    # calibrate on the 3 seasons before
        w = probit_fit(pr[fit_sel], y[fit_sel])
        out[name] = brier(norm.cdf(w[0] + w[1] * pr[sel]), y[sel]); out[name + '_n'] = int(sel.sum())
    return out, pr

T1 = {'England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji'}
if __name__ == '__main__':
    res = {}
    # ---------- internationals
    mI = lambda d: (d.home.isin(T1) & d.away.isin(T1)).values
    v1 = [0.10, 4, 1.0, 0.0, 0.0, 0.0]
    base, _ = evaluate('intl', mI, [2024], [2025, 2026], v1)
    obj = lambda x: evaluate('intl', mI, [2024], [2025, 2026], x)[0]['val'] if (0.02 < x[0] < 0.4 and 0.4 < x[2] <= 1.2 and 0 <= x[3] < 0.6 and 0 <= x[4] < 3 and 0 <= x[5] <= 1) else 1
    best = None
    for start in ([0.10, 4, 0.8, 0.1, 0.5, 0.3], [0.15, 3, 0.7, 0.2, 1.0, 0.0], [0.08, 5, 1.0, 0.0, 0.0, 0.5]):
        r = minimize(obj, start, method='Nelder-Mead', options=dict(maxiter=120, xatol=1e-3, fatol=1e-5))
        if best is None or r.fun < best.fun: best = r
    tuned, _ = evaluate('intl', mI, [2024], [2025, 2026], best.x)
    res['intl'] = dict(v1=base, tuned=tuned, params=dict(zip(['K','hfa','alpha','regress','newK','team_hfa_w'], best.x.round(3))))
    print('INTL', res['intl'])
    # ---------- clubs
    mC = lambda d: np.ones(len(d), bool)
    v1c = [0.06, 1.0, 0.2, 0.0, 0.0, 9, 7, 6, 6]
    basec, _ = evaluate('club', mC, [2024], [2025, 2026], v1c, True)
    objc = lambda x: evaluate('club', mC, [2024], [2025, 2026], x, True)[0]['val'] if (0.02 < x[0] < 0.3 and 0.4 < x[1] <= 1.2 and 0 <= x[2] < 0.7 and 0 <= x[3] < 3 and 0 <= x[4] <= 1) else 1
    r = minimize(objc, [0.08, 0.8, 0.25, 0.5, 0.3, 9, 7, 6, 5], method='Nelder-Mead', options=dict(maxiter=200, xatol=1e-3, fatol=1e-5))
    tunedc, _ = evaluate('club', mC, [2024], [2025, 2026], r.x, True)
    res['club'] = dict(v1=basec, tuned=tunedc, params=dict(zip(['K','alpha','regress','newK','team_hfa_w','hfa_top14','hfa_prem','hfa_urc','hfa_euro'], r.x.round(3))))
    print('CLUB', res['club'])
    json.dump(res, open('../model/tune_results.json', 'w'), indent=1, default=float)
