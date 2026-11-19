"""Fit final ratings, run backtest diagnostics, price upcoming fixtures vs market."""
import json, numpy as np, pandas as pd
from scipy.stats import norm
from model import G, run, score

P = json.load(open('../model/params.json'))
IK, IH, ICAP, IREG = P['intl']['params']
CK, CCAP, CREG = 0.06, 40, 0.2
CHFA = dict(top14=9, prem=7, urc=6, euro=6)

I = G[G.kind == 'intl']; C = G[G.kind == 'club']
oi, RI = run(I, IK, IH, ICAP, IREG, 'intl', eval_from=pd.Timestamp('2024-01-01'), return_ratings=True)
oc, RC = run(C, CK, 5, CCAP, CREG, 'club', hfa_map=CHFA, eval_from=pd.Timestamp('2024-08-01'), return_ratings=True)
T1 = ['England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji']
oi1 = oi[[I.loc[i].home in T1 and I.loc[i].away in T1 for i in oi.index]]
SIG_I = (oi1.margin - oi1.pred).std(); SIG_C = (oc.margin - oc.pred).std()

def probs(pred, sig):
    pd_ = norm.pdf(0, pred, sig)          # density at 0 ~ P(draw) per 1-pt bucket
    ph = 1 - norm.cdf(0.5, pred, sig); pa = norm.cdf(-0.5, pred, sig)
    return ph, 1 - ph - pa, pa

def cover(pred, sig, line):   # P(home margin + line > 0), line from home perspective (e.g. -9.5)
    return 1 - norm.cdf(-line, pred, sig)

def kelly(p, dec):
    b = dec - 1; f = (p * b - (1 - p)) / b
    return max(f, 0) / 4

def calib(out, sig, bins=(0, .2, .35, .5, .65, .8, 1.01)):
    p = 1 - norm.cdf(0.5, out.pred, sig); y = (out.margin > 0).astype(float)
    df = pd.DataFrame(dict(p=p, y=y)); df['b'] = pd.cut(df.p, bins)
    return df.groupby('b', observed=True).agg(n=('y', 'size'), pred=('p', 'mean'), actual=('y', 'mean')).round(3)

# ---------- upcoming internationals (Sportingbet ZA board, pulled 17 Nov) ----------
NC6 = [  # home, away, 1, X, 2, home_line, home_hcp_odds, away_hcp_odds
    ('England', 'New Zealand', 2.25, 17.0, 1.68, +2.5, 1.82, 1.88),
    ('Scotland', 'Japan', 1.008, 51.0, 23.0, -30.5, 1.82, 1.88),
    ('Ireland', 'South Africa', 4.33, 19.0, 1.24, +9.5, 1.90, 1.80),
    ('Italy', 'Fiji', 1.47, 16.5, 2.80, -5.5, 1.90, 1.80),
    ('France', 'Argentina', 1.062, 29.0, 10.5, -19.5, 1.88, 1.82),
    ('Wales', 'Australia', 2.85, 17.5, 1.44, +5.5, 1.83, 1.87),
]
rows = []
for h, a, o1, ox, o2, ln, oh, oa in NC6:
    pred = RI[h] - RI[a] + IH
    ph, pdw, pa = probs(pred, SIG_I)
    inv = 1/o1 + 1/ox + 1/o2
    mkt_ph, mkt_pa = (1/o1)/inv, (1/o2)/inv
    # market-implied margin from handicap (fair line = -ln)
    pc = cover(pred, SIG_I, ln)
    cands = [('ML ' + h, ph, o1), ('ML ' + a, pa, o2), (f'{h} {ln:+.1f}', pc, oh), (f'{a} {-ln:+.1f}', 1 - pc, oa)]
    for lab, p, dec in cands:
        ev = p * dec - 1
        rows.append(dict(match=f'{h} v {a}', bet=lab, model_p=p, odds=dec, ev=ev, qk=kelly(p, dec),
                         model_margin=pred, mkt_margin=-ln, min_odds_5pct=1.05 / p if p > 0 else np.nan))
B = pd.DataFrame(rows)

# ---------- club fixtures (no market yet: fair prices + 5%-EV min odds) ----------
CLUB_FIX = [('Bordeaux-Begles','Montpellier','top14'),('Stade Francais','Vannes','top14'),('La Rochelle','Perpignan','top14'),
            ('Racing 92','Pau','top14'),('Bayonne','Castres','top14'),('Lyon','Toulouse','top14'),('Clermont Auvergne','Toulon','top14'),
            ('Bath','Bristol Bears','prem'),('Gloucester','Newcastle','prem'),('Harlequins','Leicester Tigers','prem'),
            ('Saracens','Northampton','prem'),('Sale Sharks','Exeter','prem')]
crow = []
for h, a, lg in CLUB_FIX:
    pred = RC[h] - RC[a] + CHFA[lg]
    ph, pdw, pa = probs(pred, SIG_C)
    crow.append(dict(league=lg, match=f'{h} v {a}', fair_line=f'{h} {-pred:+.1f}', p_home=ph, p_away=pa,
                     fair_home=1/ph, fair_away=1/pa, bet_home_at=1.05/ph, bet_away_at=1.05/pa))
CF = pd.DataFrame(crow)

if __name__ == '__main__':
    pd.set_option('display.width', 220)
    print('SIGMA intl', round(SIG_I, 2), ' club', round(SIG_C, 2))
    print('INTL tier-1 backtest 2024+:', score(oi1)); print(calib(oi1, SIG_I))
    print('CLUB backtest 2024-25+:', score(oc)); print(calib(oc, SIG_C))
    ri = pd.Series(RI)[T1 + ['Georgia','Samoa','Tonga','Portugal','Spain','Uruguay','United States of America','Romania','Canada','Chile']].sort_values(ascending=False)
    print(ri.round(1))
    print(B.round(3).to_string())
    rc = pd.Series(RC); print(rc.sort_values(ascending=False).head(30).round(1))
    print(CF.round(2).to_string())
    json.dump(dict(RI={k: float(v) for k, v in RI.items()}, RC={k: float(v) for k, v in RC.items()},
                   SIG_I=float(SIG_I), SIG_C=float(SIG_C)), open('../model/ratings.json', 'w'))
    B.to_csv('../model/intl_bets.csv', index=False); CF.to_csv('../model/club_fair.csv', index=False)
