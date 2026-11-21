"""Generalized predicted-table/playoff-odds Monte Carlo, built after top14_sim.py so the other 3 leagues
share one core instead of 3 near-duplicate copies (less risk of a rule inconsistency between them). Formats
confirmed via Wikipedia, 21 Nov 2026:

  Premiership (10 teams, double round-robin=18 games, no relegation this season - 'Franchise Model'):
    bonus: 4/2/0, +1 ABSOLUTE 4+ tries, +1 losing within 7. Playoffs: top 4, fixed semis 1v4 & 2v3 (higher
    seed hosts), final at Twickenham (neutral).
  URC (16 teams, 18 games - NOT a full round-robin, regional pools mean an uneven schedule - fixture pull is
    load-bearing here, not optional): bonus: 4/2/0, +1 ABSOLUTE 4+ tries, +1 losing within 7. Playoffs: top 8,
    QF seeded 1v8/2v7/3v6/4v5 (assumed standard - not explicitly confirmed), SF and final FULLY RE-SEEDED each
    round (best remaining seed always hosts, plays the worst remaining seed).
  Super Rugby Pacific (10 teams - Rebels folded, matches produce.py's super_rugby() which already drops them
    -, ~16 games, 1 bye each): bonus: 4/2/0, +1 for a 3-try DIFFERENTIAL (like Top14, not absolute), +1
    losing within 7. Playoffs: top 6, 1st/2nd bye, 3v6 and 4v5 elimination finals, semis PARTIALLY re-seeded
    (1 plays whichever EF winner has the BETTER original seed, 2 plays the other - different from Top14's
    FIXED 1-always-plays-3v6-winner rule, which is why this needed its own resolver, not top14_sim.py's).

Bonus-point curves (both already fit on real data, not guessed):
  ABSOLUTE: P(team's OWN score >= 4-try threshold) - logistic on score, coef 0.2206, intercept -6.574
            (557 real Test performances - see season_sim.py).
  DIFFERENTIAL: P(try diff >= 3) - logistic on margin, coef 0.26587, intercept -4.8031 (2,361 real club
            team-performances - see top14_sim.py).
Club margin sigma: 13.33, probit-fit on all club games (top14_sim.py's fit_sigma()), shared across all 4
leagues - it's a club-wide calibration, not league-specific.
"""
import numpy as np, pandas as pd, warnings
import features as F
from v6 import CHFA, elo, probit_fit
from club_fixtures import pull, ESPN_LEAGUE_IDS
from datetime import datetime
warnings.filterwarnings('ignore')

ABS_BONUS = lambda score: 1 / (1 + np.exp(6.574 - 0.2206 * score))         # season_sim.py's curve
DIFF_BONUS = lambda margin: 1 / (1 + np.exp(4.803 - 0.26587 * margin))     # top14_sim.py's curve

LEAGUES = {   # super_rugby isn't here - it's special-cased in run_super_rugby() (different module/HFA rule)
    'premiership': dict(comp='Gallagher Premiership', hfa_key='prem', bonus='abs', n_playoff=4),
    'urc':         dict(comp='United Rugby Championship', hfa_key='urc', bonus='abs', n_playoff=8),
}

def fit_club_common():
    S, R, ATT, DEF = F.club_state()
    G = pd.read_parquet('games.parquet'); C = G[G.kind == 'club'].reset_index(drop=True)
    pred, _ = elo(C, 'club', 0.06, 5, 40, 0.2, hfa_map=CHFA, sa_bonus=1.0)
    y = (C.hs > C.as_).astype(float).values + 0.5 * (C.hs == C.as_).values
    w = probit_fit(pred, y)
    return dict(R=R, sigma=1 / w[1])

def predict_margin(home, away, F_, hfa):
    return F_['R'].get(home, 0) - F_['R'].get(away, 0) + hfa

def bonus_prob(kind, side_score_or_margin):
    return ABS_BONUS(side_score_or_margin) if kind == 'abs' else DIFF_BONUS(side_score_or_margin)

def sim_margin(home, away, F_, hfa, rng):
    return rng.normal(predict_margin(home, away, F_, hfa), F_['sigma'])

def sim_bonus_points(margin, bonus_kind, rng, tot_guess=45):
    """tot_guess: rough expected combined score, only used for the 'abs' bonus curve which needs a single
    team's SCORE (not just margin) - not separately simulated per-league here (would need each league's own
    attack/defence totals model); a fixed-ish total is a simplification, logged honestly."""
    diff = round(margin)
    ph = pa = 0
    if diff > 0: ph += 4
    elif diff < 0: pa += 4
    else: ph += 2; pa += 2
    if bonus_kind == 'abs':
        hs, as_ = (tot_guess + margin) / 2, (tot_guess - margin) / 2
        if rng.random() < ABS_BONUS(hs): ph += 1
        if rng.random() < ABS_BONUS(as_): pa += 1
    else:
        if rng.random() < DIFF_BONUS(margin): ph += 1
        if rng.random() < DIFF_BONUS(-margin): pa += 1
    if 0 < -diff <= 7: ph += 1
    if 0 < diff <= 7: pa += 1
    return ph, pa, diff

def current_standings(comp, bonus_kind):
    g = pd.read_parquet('games.parquet')
    cg = g[(g.competition == comp) & (g.date >= '2026-08-01')]
    tm = pd.read_parquet('team_match_stats.parquet'); tm = tm[tm.ok]
    tries = tm.set_index(['date', 'team']).tries.to_dict()
    teams = sorted(set(cg.home) | set(cg.away))
    pts = {t: 0 for t in teams}; pdiff = {t: 0 for t in teams}; played = {t: 0 for t in teams}
    for r in cg.itertuples():
        diff = r.hs - r.as_
        th, ta = tries.get((r.date, r.home), 0), tries.get((r.date, r.away), 0)
        ph = pa = 0
        if diff > 0: ph += 4
        elif diff < 0: pa += 4
        else: ph += 2; pa += 2
        if bonus_kind == 'abs':
            if th >= 4: ph += 1
            if ta >= 4: pa += 1
        else:
            if th - ta >= 3: ph += 1
            if ta - th >= 3: pa += 1
        if 0 < -diff <= 7: ph += 1
        if 0 < diff <= 7: pa += 1
        pts[r.home] += ph; pts[r.away] += pa
        pdiff[r.home] += diff; pdiff[r.away] -= diff
        played[r.home] += 1; played[r.away] += 1
    return pts, pdiff, played, teams

def playoff_prem(seeds, F_, hfa, rng):
    """fixed semis 1v4, 2v3, higher seed hosts; final at neutral Twickenham."""
    s1, s2, s3, s4 = seeds[:4]
    m1 = sim_margin(s1, s4, F_, hfa, rng); w1 = s1 if m1 > 0 else s4
    m2 = sim_margin(s2, s3, F_, hfa, rng); w2 = s2 if m2 > 0 else s3
    mf = sim_margin(w1, w2, F_, 0.0, rng); champ = w1 if mf > 0 else w2   # neutral final, no HFA
    return w1, w2, champ

def playoff_urc(seeds, F_, hfa, rng):
    """top 8, re-seeded every round, better remaining seed always hosts (confirmed via Wikipedia)."""
    rank = {t: i for i, t in enumerate(seeds[:8])}   # 0 = best
    pairs = [(seeds[0], seeds[7]), (seeds[1], seeds[6]), (seeds[2], seeds[5]), (seeds[3], seeds[4])]
    qf_winners = []
    for h, a in pairs:   # better seed (lower rank) hosts
        m = sim_margin(h, a, F_, hfa, rng); qf_winners.append(h if m > 0 else a)
    qf_winners.sort(key=lambda t: rank[t])
    sf_pairs = [(qf_winners[0], qf_winners[3]), (qf_winners[1], qf_winners[2])]
    sf_winners = []
    for h, a in sf_pairs:
        m = sim_margin(h, a, F_, hfa, rng); sf_winners.append(h if m > 0 else a)
    sf_winners.sort(key=lambda t: rank[t])
    h, a = sf_winners
    mf = sim_margin(h, a, F_, hfa, rng); champ = h if mf > 0 else a
    return sf_winners, champ

def run_super_rugby(n=5000, seed=0):
    """Special-cased: Super Rugby Pacific lives in sr.py/matches.parquet, not games.parquet/features.py like
    the other 3 leagues, and has its own dynamic HFA (travel_bonus crossing a border, drua_bonus at home in
    Fiji - not a flat CHFA number). The 2027 season (Feb-June) hasn't started yet, same situation as Six
    Nations 2027 in season_sim.py - zero-based standings, the whole season simulated from current ratings."""
    import sr
    df = sr.games()
    _, R = sr.elo(df, 0.03, 1, 40, 0.2, travel_bonus=2, drua_bonus=6)
    R.pop('Rebels', None)
    G = pd.read_parquet('games.parquet'); C = G[G.kind == 'club'].reset_index(drop=True)
    sigma = fit_club_common()['sigma']   # reuse the club-wide sigma fit - no separate SR fit exists
    teams = sorted(R.keys())
    fx = pd.read_csv('fixtures_super_rugby.csv')
    fx['home'] = fx.home.map(lambda t: sr.NORM.get(t, t)); fx['away'] = fx.away.map(lambda t: sr.NORM.get(t, t))
    unknown = (set(fx.home) | set(fx.away)) - set(teams)
    if unknown:
        raise ValueError(f'super_rugby: fixtures reference unknown teams - add to sr.NORM: {unknown}')

    def margin(home, away, rng):
        travel = int(sr.country(home) != sr.country(away))
        drua_home = int(home == 'Fijian Drua')
        H = 1 + 2 * travel + 6 * drua_home
        return rng.normal(R[home] - R[away] + H, sigma)

    rng = np.random.default_rng(seed)
    rank_count = {t: [0] * len(teams) for t in teams}
    champ_count = {t: 0 for t in teams}; final_count = {t: 0 for t in teams}; qual_count = {t: 0 for t in teams}
    for _ in range(n):
        pts = {t: 0 for t in teams}; pdiff = {t: 0 for t in teams}
        for r in fx.itertuples():
            m = margin(r.home, r.away, rng)
            ph, pa, diff = sim_bonus_points(m, 'diff', rng)
            pts[r.home] += ph; pts[r.away] += pa; pdiff[r.home] += diff; pdiff[r.away] -= diff
        rank = sorted(teams, key=lambda t: (-pts[t], -pdiff[t]))
        for i, t in enumerate(rank): rank_count[t][i] += 1
        for t in rank[:6]: qual_count[t] += 1
        rank6 = {t: i for i, t in enumerate(rank[:6])}
        s1, s2, s3, s4, s5, s6 = rank[:6]
        m36 = margin(s3, s6, rng); w36 = s3 if m36 > 0 else s6
        m45 = margin(s4, s5, rng); w45 = s4 if m45 > 0 else s5
        better, worse = sorted([w36, w45], key=lambda t: rank6[t])
        msf1 = margin(s1, better, rng); wsf1 = s1 if msf1 > 0 else better
        msf2 = margin(s2, worse, rng); wsf2 = s2 if msf2 > 0 else worse
        final_count[wsf1] += 1; final_count[wsf2] += 1
        mf = margin(wsf1, wsf2, rng); champ = wsf1 if mf > 0 else wsf2
        champ_count[champ] += 1
    return {t: dict(rank_dist=[c / n for c in rank_count[t]], p_qualify=qual_count[t] / n,
                    p_final=final_count[t] / n, p_champion=champ_count[t] / n,
                    current_pts=0, played=0) for t in teams}

def run(league, n=5000, seed=0):
    cfg = LEAGUES[league]
    F_ = fit_club_common()
    hfa = CHFA[cfg['hfa_key']] if cfg['hfa_key'] else 1.0   # Super Rugby's own HFA (1pt) isn't in CHFA (that's europe/urc/top14/prem only)
    base_pts, base_pd, played, teams = current_standings(cfg['comp'], cfg['bonus'])
    fx = pd.read_csv(f'fixtures_{league}.csv')
    unknown = (set(fx.home) | set(fx.away)) - set(teams)
    if unknown:
        raise ValueError(f'{league}: fixtures.csv has team names not in games.parquet - likely a missing '
                         f'NAME_MAP entry in club_fixtures.py, would silently drop their games: {unknown}')
    rng = np.random.default_rng(seed)
    n_teams = len(teams)
    rank_count = {t: [0] * n_teams for t in teams}
    champ_count = {t: 0 for t in teams}; final_count = {t: 0 for t in teams}; qual_count = {t: 0 for t in teams}
    for _ in range(n):
        pts = dict(base_pts); pdiff = dict(base_pd)
        for r in fx.itertuples():
            if r.home not in pts or r.away not in pts: continue
            margin = sim_margin(r.home, r.away, F_, hfa, rng)
            ph, pa, diff = sim_bonus_points(margin, cfg['bonus'], rng)
            pts[r.home] += ph; pts[r.away] += pa; pdiff[r.home] += diff; pdiff[r.away] -= diff
        rank = sorted(teams, key=lambda t: (-pts[t], -pdiff[t]))
        for i, t in enumerate(rank): rank_count[t][i] += 1
        for t in rank[:cfg['n_playoff']]: qual_count[t] += 1
        seeds = rank[:cfg['n_playoff']]
        if league == 'premiership': w1, w2, champ = playoff_prem(seeds, F_, hfa, rng); final_count[w1] += 1; final_count[w2] += 1
        else: finalists, champ = playoff_urc(seeds, F_, hfa, rng); final_count[finalists[0]] += 1; final_count[finalists[1]] += 1
        champ_count[champ] += 1
    return {t: dict(rank_dist=[c / n for c in rank_count[t]], p_qualify=qual_count[t] / n,
                    p_final=final_count[t] / n, p_champion=champ_count[t] / n,
                    current_pts=base_pts[t], played=played[t]) for t in teams}

def amer(p):
    if p <= 0: return '+999900'
    dec = 1 / p
    return f'+{round((dec - 1) * 100)}' if dec >= 2 else f'{round(-100 / (dec - 1))}'

if __name__ == '__main__':
    import sys, os
    league = sys.argv[1] if len(sys.argv) > 1 else 'premiership'
    if not os.path.exists(f'fixtures_{league}.csv'):
        print(f'pulling {league} fixtures (one-time, free, a couple minutes)...')
        pull(league, datetime(2026, 11, 21), datetime(2027, 7, 1)).to_csv(f'fixtures_{league}.csv', index=False)
    pd.set_option('display.width', 200)
    R = run_super_rugby() if league == 'super_rugby' else run(league)
    print(f"{'team':20s} {'pld':>4s} {'pts':>4s}  {'P(qualify)':>10s} {'P(final)':>8s} {'P(champ)':>8s}  fair odds")
    for t, d in sorted(R.items(), key=lambda x: -x[1]['p_champion']):
        print(f"{t:20s} {d['played']:4d} {d['current_pts']:4d}  {d['p_qualify']:9.1%} {d['p_final']:8.1%} {d['p_champion']:8.1%}  {amer(d['p_champion'])}")
