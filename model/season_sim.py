"""Monte Carlo tournament simulation: predicted final table + title odds, for the two tournaments with a
genuinely known fixture list (a real remaining-games list, not fabricated):
  - Nations Championship 2026: Rounds 1-5 are real results (games.parquet); Round 6 (today, 21 Nov) and the
    Finals Weekend (27-29 Nov, same-rank-across-pools pairing) are simulated from our Test ratings.
  - Six Nations 2027 (5 Feb - 13 Mar): not started yet; full 15-game fixture list is public record (announced
    9 Mar 2026), hand-entered below same as produce.py's FIX list. Simulated entirely from current ratings -
    the best estimate available today, though ratings will of course move before Feb 2027.
No play-by-play/minutes data exists (see README), so a simulated scoreline is margin + total drawn from the
model's own distributions, not simulated tries - the four-try bonus is awarded probabilistically from an
empirical score->P(4+ tries) curve fit on 557 real Test team-performances (logistic, score coef 0.221,
intercept -6.57), not guessed. Club competitions (Top14/Prem/URC/Super Rugby) are NOT done here - no reliable
remaining-fixtures source was found (see README); this only covers tournaments with a genuinely known schedule.
"""
import numpy as np, pandas as pd, warnings
from scipy.stats import norm
import features as F
from v6 import intl_games, elo, probit_fit, T1, season
warnings.filterwarnings('ignore')

HFA_I = 4.0
NH = {'England', 'France', 'Ireland', 'Italy', 'Scotland', 'Wales'}
SH = {'Argentina', 'Australia', 'Fiji', 'Japan', 'New Zealand', 'South Africa'}
# four-try bonus probability given a team's score, fit on 557 real Test performances (team_match_stats.tries
# vs games.parquet score, International Test Match only): P = sigmoid(-6.574 + 0.2206*score)
TRY_BONUS = lambda score: 1 / (1 + np.exp(6.574 - 0.2206 * score))

SIX_NATIONS_2027 = [  # announced 9 Mar 2026 (wikipedia), round, home, away
    (1, 'Ireland', 'England'), (1, 'Scotland', 'Italy'), (1, 'France', 'Wales'),
    (2, 'Italy', 'Ireland'), (2, 'Scotland', 'Wales'), (2, 'England', 'France'),
    (3, 'Wales', 'Ireland'), (3, 'England', 'Italy'), (3, 'France', 'Scotland'),
    (4, 'Scotland', 'Ireland'), (4, 'Italy', 'France'), (4, 'Wales', 'England'),
    (5, 'Italy', 'Wales'), (5, 'England', 'Scotland'), (5, 'Ireland', 'France'),
]

def fit():
    """Same fit as produce.py's tests(), generalized to predict ANY team pair, not just this round's FIX."""
    I = intl_games(True)
    pr, R = elo(I, 'intl', 0.08, HFA_I, 60, 0.0)
    y = (I.hs > I.as_).astype(float).values + 0.5 * (I.hs == I.as_).values
    ss = np.array([season(d, 'intl') for d in I.date])
    m = (I.home.isin(T1) & I.away.isin(T1)).values & (ss >= 2023)
    w = probit_fit(pr[m], y[m])
    S, _, ATT, DEF = F.build_state(I, 'intl', 0.08, lambda r: HFA_I, 60, 0.0, Ka=0.08)
    hist = S[S.home.isin(T1) & S.away.isin(T1) & (S.date >= '2023-01-01')]
    b = np.polyfit(hist.tot, hist.total, 1)
    mu_pts = I[['hs', 'as_']].values.mean()
    sigma = 1 / w[1]
    return dict(R=R, ATT=ATT, DEF=DEF, w=w, b=b, mu_pts=mu_pts, sigma=sigma)

def predict(home, away, F_):
    R, ATT, DEF, w, b, mu_pts = F_['R'], F_['ATT'], F_['DEF'], F_['w'], F_['b'], F_['mu_pts']
    mg = R[home] - R[away] + HFA_I
    eh = mu_pts + ATT.get(home, 0) - DEF.get(away, 0) + HFA_I / 2
    ea = mu_pts + ATT.get(away, 0) - DEF.get(home, 0) - HFA_I / 2
    tot = np.polyval(b, eh + ea)
    return mg, tot

def sim_game(home, away, F_, rng):
    mg, tot = predict(home, away, F_)
    margin = rng.normal(mg, F_['sigma'])
    hs = round((tot + margin) / 2); as_ = round((tot - margin) / 2)
    hs, as_ = max(hs, 0), max(as_, 0)
    diff = hs - as_
    pts_h = pts_a = 0
    if diff > 0: pts_h += 4
    elif diff < 0: pts_a += 4
    else: pts_h += 2; pts_a += 2
    if rng.random() < TRY_BONUS(hs): pts_h += 1
    if rng.random() < TRY_BONUS(as_): pts_a += 1
    if 0 < -diff <= 7: pts_h += 1       # losing bonus (within 7 - the actual scoreline, not the model's margin)
    if 0 < diff <= 7: pts_a += 1
    return dict(home=home, away=away, hs=hs, as_=as_, pts_h=pts_h, pts_a=pts_a)

def nc_actual_standings():
    """R1-5 already played - use REAL tries (team_match_stats, joined by date+team) for the four-try bonus,
    not the simulated proxy, since we actually know what happened."""
    g = pd.read_parquet('games.parquet')
    nc = g[g.competition == 'Nations Championship']
    cross = nc[((nc.home.isin(NH)) & (nc.away.isin(SH))) | ((nc.home.isin(SH)) & (nc.away.isin(NH)))]
    tm = pd.read_parquet('team_match_stats.parquet'); tm = tm[tm.ok]
    tries = tm.set_index(['date', 'team']).tries.to_dict()
    pts = {t: 0 for t in NH | SH}; pd_ = {t: 0 for t in NH | SH}
    for r in cross.itertuples():
        diff = r.hs - r.as_
        ph = pa = 0
        if diff > 0: ph += 4
        elif diff < 0: pa += 4
        else: ph += 2; pa += 2
        if tries.get((r.date, r.home), 0) >= 4: ph += 1
        if tries.get((r.date, r.away), 0) >= 4: pa += 1
        if 0 < -diff <= 7: ph += 1
        if 0 < diff <= 7: pa += 1
        pts[r.home] += ph; pts[r.away] += pa
        pd_[r.home] += diff; pd_[r.away] -= diff
    return pts, pd_

NC_R6 = [('England', 'New Zealand'), ('Scotland', 'Japan'), ('Ireland', 'South Africa'),
         ('Italy', 'Fiji'), ('France', 'Argentina'), ('Wales', 'Australia')]   # today's known fixtures

def run_nc(n=10000, seed=0):
    F_ = fit(); rng = np.random.default_rng(seed)
    base_pts, base_pd = nc_actual_standings()
    champ_count = {t: 0 for t in NH | SH}
    rank_count = {t: [0] * 6 for t in NH | SH}
    for _ in range(n):
        pts = dict(base_pts); pdiff = dict(base_pd)
        for h, a in NC_R6:
            r = sim_game(h, a, F_, rng)
            pts[h] += r['pts_h']; pts[a] += r['pts_a']; pdiff[h] += r['hs'] - r['as_']; pdiff[a] += r['as_'] - r['hs']
        nh_rank = sorted(NH, key=lambda t: (-pts[t], -pdiff[t]))
        sh_rank = sorted(SH, key=lambda t: (-pts[t], -pdiff[t]))
        for i, t in enumerate(nh_rank): rank_count[t][i] += 1
        for i, t in enumerate(sh_rank): rank_count[t][i] += 1
        final = sim_game(nh_rank[0], sh_rank[0], F_, rng)   # Championship Final, 1st vs 1st
        champ = nh_rank[0] if final['hs'] > final['as_'] else (sh_rank[0] if final['hs'] < final['as_'] else None)
        if champ: champ_count[champ] += 1
    return {t: dict(rank_dist=[c / n for c in rank_count[t]], p_champion=champ_count[t] / n) for t in NH | SH}

def run_six_nations(n=10000, seed=0):
    F_ = fit(); rng = np.random.default_rng(seed)
    teams = list(NH)
    rank_count = {t: [0] * 6 for t in teams}
    champ_count = {t: 0 for t in teams}; gs_count = {t: 0 for t in teams}
    for _ in range(n):
        pts = {t: 0 for t in teams}; pdiff = {t: 0 for t in teams}; wins = {t: 0 for t in teams}
        for rnd, h, a in SIX_NATIONS_2027:
            r = sim_game(h, a, F_, rng)
            pts[h] += r['pts_h']; pts[a] += r['pts_a']; pdiff[h] += r['hs'] - r['as_']; pdiff[a] += r['as_'] - r['hs']
            if r['hs'] > r['as_']: wins[h] += 1
            elif r['as_'] > r['hs']: wins[a] += 1
        for t in teams:
            if wins[t] == 5: pts[t] += 3; gs_count[t] += 1   # Grand Slam bonus
        rank = sorted(teams, key=lambda t: (-pts[t], -pdiff[t]))
        for i, t in enumerate(rank): rank_count[t][i] += 1
        champ_count[rank[0]] += 1
    return {t: dict(rank_dist=[c / n for c in rank_count[t]], p_champion=champ_count[t] / n,
                    p_grand_slam=gs_count[t] / n) for t in teams}

def amer(p):
    if p <= 0: return '+999900'
    dec = 1 / p
    return f'+{round((dec - 1) * 100)}' if dec >= 2 else f'{round(-100 / (dec - 1))}'

if __name__ == '__main__':
    pd.set_option('display.width', 200)
    print('=== Nations Championship 2026 (R1-5 actual, R6 + Finals Weekend simulated, n=10000) ===')
    nc = run_nc()
    for t, d in sorted(nc.items(), key=lambda x: -x[1]['p_champion']):
        print(f"{t:14s} P(champion) {d['p_champion']:.1%}  fair odds {amer(d['p_champion'])}   rank dist {[round(x,2) for x in d['rank_dist']]}")
    print('\n=== Six Nations 2027 (not yet started, n=10000) ===')
    sn = run_six_nations()
    for t, d in sorted(sn.items(), key=lambda x: -x[1]['p_champion']):
        print(f"{t:10s} P(title) {d['p_champion']:.1%}  fair odds {amer(d['p_champion'])}  P(Grand Slam) {d['p_grand_slam']:.1%}  rank dist {[round(x,2) for x in d['rank_dist']]}")
