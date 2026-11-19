"""Situational angle scan. Each angle picks a SIDE in a game; we measure that side vs the pre-match Elo line.
resid = side's actual margin - Elo-expected margin. cover = side beats the Elo line (proxy for ATS; no odds history).
Breakeven at -110 is 52.4%."""
import numpy as np, pandas as pd
from scipy.stats import norm

SC = pd.read_parquet('../model/SC.parquet'); SC['kind'] = 'club'
SI = pd.read_parquet('../model/SI.parquet'); SI['kind'] = 'intl'
T1 = {'England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji'}
SA = {'Bulls', 'Lions', 'Sharks', 'Stormers'}
FR = {'Toulouse','Bordeaux-Begles','La Rochelle','Racing 92','Stade Francais','Toulon','Clermont Auvergne','Montpellier','Pau',
      'Castres','Bayonne','Lyon','Perpignan','Vannes','US Montauban','Brive','Oyonnax'}

def team_log(S):
    """long table: one row per team-game with pre-match info and that game's result vs Elo."""
    rows = []
    for i, r in S.iterrows():
        for side, sgn in (('home', 1), ('away', -1)):
            t = r.home if side == 'home' else r.away; o = r.away if side == 'home' else r.home
            rows.append(dict(i=i, date=r.date, team=t, opp=o, side=side, exp=sgn * r.elo, margin=sgn * r.margin,
                             core=r.core_h if side == 'home' else r.core_a, comp=r.comp, neutral=r.neutral))
    L = pd.DataFrame(rows).sort_values(['team', 'date'])
    L['resid'] = L.margin - L.exp
    g = L.groupby('team')
    L['prev_resid'] = g.resid.shift(1); L['prev_margin'] = g.margin.shift(1); L['prev_exp'] = g.exp.shift(1)
    L['prev_date'] = g.date.shift(1); L['rest'] = (L.date - L.prev_date).dt.days
    L['prev_comp'] = g.comp.shift(1)
    return L

def summarize(name, x, sig=16.6):
    x = x.dropna(subset=['resid'])
    if len(x) < 25: return None
    cover = (x.resid > 0).mean() + 0.5 * (x.resid == 0).mean()
    p_exp = 1 - norm.cdf(0.5, x.exp, sig); su = (x.margin > 0).mean()
    t = x.resid.mean() / (x.resid.std() / np.sqrt(len(x)))
    by = x.groupby(x.date.apply(lambda d: d.year if d.month >= 8 else d.year - 1)).resid.apply(lambda r: (r > 0).mean())
    return dict(angle=name, n=len(x), resid=x.resid.mean(), t=t, cover=cover, su=su, su_exp=p_exp.mean(),
                seasons_pos=f'{(by > 0.5).sum()}/{len(by)}')

def run():
    C = team_log(SC[SC.date >= '2022-08-01']); I = team_log(SI[SI.date >= '2022-01-01'])
    I = I[I.team.isin(T1) & I.opp.isin(T1)]
    A = []
    big_nov = lambda d: d.month == 11 or (d.month in (2, 3))   # international windows
    # ---- club
    A.append(summarize('Club: SA team away in Europe', C[(C.team.isin(SA)) & (C.side == 'away') & ~C.opp.isin(SA)]))
    A.append(summarize('Club: European team away in SA', C[(~C.team.isin(SA)) & (C.side == 'away') & C.opp.isin(SA)]))
    A.append(summarize('Club: short rest (<=6d) vs rested opp', C[(C.rest <= 6)]))
    A.append(summarize('Club: heavily rotated side (core<0.45)', C[C.core < 0.45]))
    A.append(summarize('Club: full-strength side (core>0.80)', C[C.core > 0.80]))
    A.append(summarize('Club: off upset win (prev resid>+15, was dog)', C[(C.prev_resid > 15) & (C.prev_exp < 0) & (C.prev_margin > 0)]))
    A.append(summarize('Club: off blowout loss (prev resid<-20)', C[C.prev_resid < -20]))
    A.append(summarize('Top 14: away dog of 10+', C[(C.comp == 'French Top 14') & (C.side == 'away') & (C.exp <= -10)]))
    A.append(summarize('Top 14: home fav of 10+', C[(C.comp == 'French Top 14') & (C.side == 'home') & (C.exp >= 10)]))
    A.append(summarize('Top 14: away side, any', C[(C.comp == 'French Top 14') & (C.side == 'away')]))
    A.append(summarize('Prem: away dog of 10+', C[(C.comp == 'Gallagher Premiership') & (C.side == 'away') & (C.exp <= -10)]))
    A.append(summarize('URC: away dog of 10+', C[(C.comp == 'United Rugby Championship') & (C.side == 'away') & (C.exp <= -10)]))
    A.append(summarize('Euro cups: French team away', C[C.comp.str.contains('European') & C.team.isin(FR) & (C.side == 'away')]))
    A.append(summarize('Euro cups: French team home', C[C.comp.str.contains('European') & C.team.isin(FR) & (C.side == 'home')]))
    A.append(summarize('Club: playing during intl window', C[C.date.apply(big_nov) & (C.comp.isin(['French Top 14','Gallagher Premiership','United Rugby Championship']))]))
    A.append(summarize('Club: first game after Europe (league)', C[C.prev_comp.fillna('').str.contains('European') & ~C.comp.str.contains('European')]))
    # ---- internationals (tier-1 v tier-1)
    SH = {'New Zealand','Australia','South Africa','Argentina','Fiji'}
    A.append(summarize('Intl: SH team touring NH (Nov)', I[I.team.isin(SH) & (I.side == 'away') & ~I.opp.isin(SH) & (I.date.dt.month.isin([10, 11]))]))
    A.append(summarize('Intl: NH team touring SH (Jun-Jul)', I[~I.team.isin(SH) & (I.side == 'away') & I.opp.isin(SH) & (I.date.dt.month.isin([6, 7]))]))
    A.append(summarize('Intl: off upset win', I[(I.prev_resid > 10) & (I.prev_exp < 0) & (I.prev_margin > 0)]))
    A.append(summarize('Intl: off loss by 15+', I[I.prev_margin <= -15]))
    A.append(summarize('Intl: home dog', I[(I.side == 'home') & (I.exp < 0) & ~I.neutral]))
    A.append(summarize('Intl: road fav', I[(I.side == 'away') & (I.exp > 0) & ~I.neutral]))
    A.append(summarize('Intl: big dog (10+)', I[I.exp <= -10]))
    A.append(summarize('Intl: big fav (10+)', I[I.exp >= 10]))
    return pd.DataFrame([a for a in A if a]).sort_values('t', ascending=False)

if __name__ == '__main__':
    pd.set_option('display.width', 220)
    R = run(); print(R.round(3).to_string(index=False)); R.to_csv('../model/angles.csv', index=False)
