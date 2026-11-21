"""Situational angle scan. Each angle picks a SIDE in a game; we measure that side vs the pre-match Elo line.
resid = side's actual margin - Elo-expected margin. cover = side beats the Elo line (proxy for ATS; no odds history).
Breakeven at -110 is 52.4%."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import norm

SC = pd.read_parquet('../model/SC.parquet'); SC['kind'] = 'club'
SI = pd.read_parquet('../model/SI.parquet'); SI['kind'] = 'intl'
T1 = {'England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji'}
SA = {'Bulls', 'Lions', 'Sharks', 'Stormers'}
FR = {'Toulouse','Bordeaux-Begles','La Rochelle','Racing 92','Stade Francais','Toulon','Clermont Auvergne','Montpellier','Pau',
      'Castres','Bayonne','Lyon','Perpignan','Vannes','US Montauban','Brive','Oyonnax'}
# Well-established derbies only, within the 3 leagues this site tracks - not an exhaustive rivalry list,
# just the handful that are unambiguous (avoids a shaky/arbitrary long list).
DERBIES = [frozenset(p) for p in [
    ('Racing 92', 'Stade Francais'),            # Le Classique Parisien
    ('Bath', 'Bristol Bears'),                  # West Country derby
    ('Leicester Tigers', 'Northampton'),         # East Midlands derby
    ('Leinster', 'Munster'),                    # Ireland's biggest derby
    ('Ospreys', 'Scarlets'),                    # Welsh regional derby
    ('Bulls', 'Lions'),                         # South African (Pretoria-Johannesburg) derby
]]

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

def summarize_var(name, x, pool, n_perm=4000):
    """For MATCH-LEVEL flags that don't distinguish home/away (derby, playing during an intl window, etc.):
    summarize()'s mean resid and cover are mathematically forced to ~0 and 0.5 for these, since every match
    contributes a home row and an away row with resid_home = -resid_away exactly - no real effect, however
    large, can show up in a signed mean. (Found while adding new angles - it silently neutered the existing
    "Club: playing during intl window" result.) Variance doesn't cancel this way: test whether |resid| runs
    higher (more chaotic/upset-prone) or lower (tighter, more predictable) than a same-size random sample of
    the pool it's drawn from - a permutation test, so it reports a real p-value comparable to the others.
    pool: the resid array of the broader population (e.g. all club games) this flag is a subset of."""
    x = x.dropna(subset=['resid'])
    if len(x) < 25: return None
    v = (x.resid ** 2).mean()
    rng = np.random.default_rng(0)
    perm = np.array([(rng.choice(pool, len(x)) ** 2).mean() for _ in range(n_perm)])
    p = 2 * min(np.mean(perm >= v), np.mean(perm <= v))
    return dict(angle=name, n=len(x), resid_sd=v ** 0.5, pool_sd=(pool ** 2).mean() ** 0.5, p_two_sided=min(p, 1.0))

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

def run_var():
    """Match-level (symmetric, non-side-specific) angles - tested on variance, not mean, per summarize_var's
    docstring. One row per match (home side only) so each match counts once, not twice."""
    C = team_log(SC[SC.date >= '2022-08-01'])
    C = C[C.side == 'home'].copy()   # one row per match now; home/away no longer meaningful for these flags
    big_nov = lambda d: d.month == 11 or (d.month in (2, 3))
    pool = C.resid.dropna().values
    is_derby = [frozenset((h, a)) in DERBIES for h, a in zip(C.team, C.opp)]
    CLUBLGS = ['French Top 14', 'Gallagher Premiership', 'United Rugby Championship']
    V = []
    V.append(summarize_var('Derby games (tighter or wilder than Elo expects)', C[pd.Series(is_derby, index=C.index)], pool))
    V.append(summarize_var('Club: during an intl window (Nov/Feb/Mar)', C[C.date.apply(big_nov) & C.comp.isin(CLUBLGS)], pool))
    V.append(summarize_var('Euro cup round within 10 days of an intl window', C[C.comp.str.contains('European', na=False) &
             C.date.apply(lambda d: big_nov(d) or big_nov(d + pd.Timedelta(days=10)) or big_nov(d - pd.Timedelta(days=10)))], pool))
    return pd.DataFrame([v for v in V if v])

if __name__ == '__main__':
    pd.set_option('display.width', 220)
    R = run()
    Rv = run_var()
    # multiple-testing correction (CLAUDE.md hard rule): Bonferroni across EVERY angle tried this scan,
    # mean-based and variance-based together (they're all draws from the same "how many things did we try" tally)
    n_angles = len(R) + len(Rv)
    alpha = 0.05 / n_angles
    from scipy.stats import t as tdist
    R['p_two_sided'] = 2 * (1 - tdist.cdf(R.t.abs(), R.n - 1))
    R['survives_bonferroni'] = R.p_two_sided < alpha
    Rv['survives_bonferroni'] = Rv.p_two_sided < alpha
    print(R.round(4).to_string(index=False))
    print(f'\n--- match-level angles (variance test - see summarize_var docstring for why) ---')
    print(Rv.round(4).to_string(index=False))
    survivors = R.survives_bonferroni.sum() + Rv.survives_bonferroni.sum()
    print(f'\n{n_angles} angles tested -> Bonferroni alpha {alpha:.4f}. Survive: {survivors} of {n_angles}.')
    if R.survives_bonferroni.any():
        print(R[R.survives_bonferroni][['angle', 'n', 'resid', 't', 'p_two_sided']].to_string(index=False))
    if Rv.survives_bonferroni.any():
        print(Rv[Rv.survives_bonferroni][['angle', 'n', 'resid_sd', 'pool_sd', 'p_two_sided']].to_string(index=False))
    print('\nCAVEAT: resid/t (and resid_sd) are vs our OWN Elo expectation, not the market closing line (no broad '
          'odds history exists for clubs/internationals - see model/README.md). A surviving angle is a candidate '
          'for the model or for the forward bets rule, not a proven market edge; it still needs validating against '
          'real closing lines as odds_log.csv accumulates rounds, or against the paid odds dataset if purchased.')
    R.to_csv('../model/angles.csv', index=False)
    Rv.to_csv('../model/angles_var.csv', index=False)
