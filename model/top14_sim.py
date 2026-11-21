"""Top 14 predicted table + playoff odds, Monte Carlo. Format confirmed via Wikipedia (21 Nov 2026):
14 teams, double round-robin. Bonus: 4/win, 2/draw, 0/loss, +1 for winning by 3+ more tries than the
opponent OR losing by <=7 points. Playoffs: top 6 qualify. 1st and 2nd get byes to the semifinal. Qualifying
finals: 4th v 5th, 3rd v 6th. Semis: 1st v (3v6 winner), 2nd v (4v5 winner). Winners meet in the Final
(Stade de France). 14th relegated outright; 13th plays a Pro D2 finalist in an access match (not simulated
here - needs Pro D2 data this repo doesn't have).
Remaining fixtures: club_fixtures.py (ESPN, verified reliable - see its docstring for the Oyonnax/Vannes
data-quality check done before trusting it). Four-try-bonus probability: fit on 2,361 real club team-
performances (differential tries vs opponent, by margin) - logistic, coef 0.266, intercept -4.80 - a
DIFFERENT curve from season_sim.py's international one (Top14's rule is a 3-try DIFFERENTIAL, not a single
team's absolute score threshold). Game-margin sigma is probit-fit on all club games (fit_sigma()), same
convention as the rest of this repo's probability calibration - 13.33, stable whether fit on pre-2025 data
only (13.18) or everything through today.
"""
import numpy as np, pandas as pd, warnings
import features as F
from v6 import CHFA, season
from club_fixtures import pull, ESPN_LEAGUE_IDS
from datetime import datetime
warnings.filterwarnings('ignore')

COMP = 'French Top 14'
TRY_DIFF_BONUS = lambda margin: 1 / (1 + np.exp(4.803 - 0.26587 * margin))   # P(this side's try-diff >= 3), by ITS OWN margin

def fit_club():
    S, R, ATT, DEF = F.club_state()   # sa_bonus defaults to 1.0 (see README, 18/21 Nov)
    return dict(R=R, HFA=CHFA['top14'])

def predict(home, away, F_):
    return F_['R'][home] - F_['R'][away] + F_['HFA']

def current_standings():
    """Actual bonus points through today from games.parquet, Top14-specific rules, real tries."""
    g = pd.read_parquet('games.parquet')
    t14 = g[(g.competition == COMP) & (g.date >= '2026-08-01')]
    tm = pd.read_parquet('team_match_stats.parquet'); tm = tm[tm.ok]
    tries = tm.set_index(['date', 'team']).tries.to_dict()
    teams = sorted(set(t14.home) | set(t14.away))
    pts = {t: 0 for t in teams}; pdiff = {t: 0 for t in teams}; played = {t: 0 for t in teams}
    for r in t14.itertuples():
        diff = r.hs - r.as_
        th, ta = tries.get((r.date, r.home), 0), tries.get((r.date, r.away), 0)
        ph = pa = 0
        if diff > 0: ph += 4
        elif diff < 0: pa += 4
        else: ph += 2; pa += 2
        if th - ta >= 3: ph += 1
        if ta - th >= 3: pa += 1
        if 0 < -diff <= 7: ph += 1
        if 0 < diff <= 7: pa += 1
        pts[r.home] += ph; pts[r.away] += pa
        pdiff[r.home] += diff; pdiff[r.away] -= diff
        played[r.home] += 1; played[r.away] += 1
    return pts, pdiff, played, teams

def fit_sigma():
    """Probit-fit club margin -> win prob on ALL available club games (same one-shot convention produce.py's
    tests() uses for Tests), sigma = 1/slope. Checked stability: 13.18 fit on seasons <2025 only vs 13.33 on
    everything through today - not sensitive to the window, so the full-data fit is used."""
    from v6 import elo, probit_fit
    G = pd.read_parquet('games.parquet'); C = G[G.kind == 'club'].reset_index(drop=True)
    pred, _ = elo(C, 'club', 0.06, 5, 40, 0.2, hfa_map=CHFA, sa_bonus=1.0)
    y = (C.hs > C.as_).astype(float).values + 0.5 * (C.hs == C.as_).values
    w = probit_fit(pred, y)
    return 1 / w[1]

def sim_margin(home, away, F_, rng, sigma):
    mg = predict(home, away, F_)
    return rng.normal(mg, sigma)

def sim_bonus_points(margin, rng):
    diff = round(margin)
    ph = pa = 0
    if diff > 0: ph += 4
    elif diff < 0: pa += 4
    else: ph += 2; pa += 2
    if rng.random() < TRY_DIFF_BONUS(margin): ph += 1
    if rng.random() < TRY_DIFF_BONUS(-margin): pa += 1
    if 0 < -diff <= 7: ph += 1
    if 0 < diff <= 7: pa += 1
    return ph, pa, diff

def run(n=5000, seed=0, fixtures_csv='fixtures_top14.csv'):
    F_ = fit_club()
    sigma = fit_sigma()
    base_pts, base_pd, played, teams = current_standings()
    fx = pd.read_csv(fixtures_csv)
    rng = np.random.default_rng(seed)
    rank_count = {t: [0] * len(teams) for t in teams}
    champ_count = {t: 0 for t in teams}; final_count = {t: 0 for t in teams}; top6_count = {t: 0 for t in teams}
    for _ in range(n):
        pts = dict(base_pts); pdiff = dict(base_pd)
        for r in fx.itertuples():
            if r.home not in pts or r.away not in pts: continue   # safety: unknown team name slipped through
            margin = sim_margin(r.home, r.away, F_, rng, sigma)
            ph, pa, diff = sim_bonus_points(margin, rng)
            pts[r.home] += ph; pts[r.away] += pa; pdiff[r.home] += diff; pdiff[r.away] -= diff
        rank = sorted(teams, key=lambda t: (-pts[t], -pdiff[t]))
        for i, t in enumerate(rank): rank_count[t][i] += 1
        for t in rank[:6]: top6_count[t] += 1
        s1, s2, s3, s4, s5, s6 = rank[:6]
        qf1 = sim_margin(s4, s5, F_, rng, sigma); w45 = s4 if qf1 > 0 else s5
        qf2 = sim_margin(s3, s6, F_, rng, sigma); w36 = s3 if qf2 > 0 else s6
        sf1 = sim_margin(s1, w36, F_, rng, sigma); wsf1 = s1 if sf1 > 0 else w36
        sf2 = sim_margin(s2, w45, F_, rng, sigma); wsf2 = s2 if sf2 > 0 else w45
        final_count[wsf1] += 1; final_count[wsf2] += 1
        fin = sim_margin(wsf1, wsf2, F_, rng, sigma); champ = wsf1 if fin > 0 else wsf2
        champ_count[champ] += 1
    return {t: dict(rank_dist=[c / n for c in rank_count[t]], p_top6=top6_count[t] / n,
                    p_final=final_count[t] / n, p_champion=champ_count[t] / n,
                    current_pts=base_pts[t], played=played[t]) for t in teams}

def amer(p):
    if p <= 0: return '+999900'
    dec = 1 / p
    return f'+{round((dec - 1) * 100)}' if dec >= 2 else f'{round(-100 / (dec - 1))}'

if __name__ == '__main__':
    import os
    if not os.path.exists('fixtures_top14.csv'):
        print('pulling fixtures (one-time, ~2 min, free)...')
        pull('top14', datetime(2026, 11, 21), datetime(2027, 6, 15)).to_csv('fixtures_top14.csv', index=False)
    pd.set_option('display.width', 200)
    R = run()
    print(f"{'team':18s} {'pld':>4s} {'pts':>4s}  {'P(top6)':>8s} {'P(final)':>8s} {'P(champ)':>8s}  fair odds")
    for t, d in sorted(R.items(), key=lambda x: -x[1]['p_champion']):
        print(f"{t:18s} {d['played']:4d} {d['current_pts']:4d}  {d['p_top6']:7.1%} {d['p_final']:8.1%} {d['p_champion']:8.1%}  {amer(d['p_champion'])}")
