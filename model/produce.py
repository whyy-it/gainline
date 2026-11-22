"""One script that rebuilds every number on Gainline from the final models.
Run order: python3 build_data.py && python3 produce.py   -> writes site.json
Final models (each validated on unseen seasons, see README):
  Tests        margin-Elo on 1995+ history, K .08, HFA 4, probit win prob calibrated on 2023+ tier-1 games
  Totals       attack/defence state (long history, rate .08), linear map fitted on 2023+ tier-1 games
  Clubs        margin-Elo K .06, HFA Top14 9 / Prem 7 / URC 6 / Europe 6, 20% season regression
  Super Rugby  margin-Elo K .03, HFA 1 +2 cross-border +6 Drua in Fiji, 20% regression
  Try scorer   team tries (Poisson on margin, expected points) x position share x player multiplier (k=8),
               then logistic recalibration (fitted 2024-25)

22 Nov: the round(s) shown on the Matches tab are no longer a hand-maintained FIX list - auto_rounds.py
detects every tracked competition's current fixtures directly from ESPN each run, so the scoreboard advances
on its own (see model/README.md). Multiple competitions can be live at once (Top14/Prem/URC overlap most
weekends) - site.json now carries a `rounds` list, one entry per active competition, instead of one flat
`matches` list. round_config.py's only remaining job is the OPTIONAL market-odds override (MKT) - there's
still no free live odds source, so Market/Bets stay manual whenever Wyatt wants them, and simply absent
otherwise (no error, no stale hand-maintained fixture list to go wrong)."""
import json, numpy as np, pandas as pd, warnings
from datetime import date
from scipy.stats import norm
warnings.filterwarnings('ignore')
import features as F
from v6 import intl_games, elo, probit_fit, season, T1, G, CHFA, KEY, score, t1_mask
import league_sim as LS
import auto_rounds as AR
from round_config import MKT, MARKET_DATE, ESPN_LEAGUE_IDS
from wr_stats_sync import FIELD_MAP   # client needs this to read World Rugby's live per-player stat keys
import sr as SR

OUT = '../model/site.json'
HFA_I = 4.0

def amer(dec): return f'+{round((dec - 1) * 100)}' if dec >= 2 else f'{round(-100 / (dec - 1))}'

def fit_intl():
    """Same fit as before 22 Nov's rewrite, just no longer tied to a hardcoded FIX list - generalized to
    predict ANY Tier-1 pair, same recipe season_sim.py already uses for its own (unrelated) simulation."""
    I = intl_games(True)
    pr, R = elo(I, 'intl', 0.08, HFA_I, 60, 0.0)
    y = (I.hs > I.as_).astype(float).values + 0.5 * (I.hs == I.as_).values
    ss = np.array([season(d, 'intl') for d in I.date])
    m = (I.home.isin(T1) & I.away.isin(T1)).values & (ss >= 2023)
    w = probit_fit(pr[m], y[m])
    S, _, ATT, DEF = F.build_state(I, 'intl', 0.08, lambda r: HFA_I, 60, 0.0, Ka=0.08)
    hist = S[S.home.isin(T1) & S.away.isin(T1) & (S.date >= '2023-01-01')]
    totb = np.polyfit(hist.tot, hist.total, 1)
    mu_pts = I[['hs', 'as_']].values.mean()
    return dict(R=R, a=float(w[0]), b=float(w[1]), sigma=1 / w[1], ATT=ATT, DEF=DEF, totb=totb, mu_pts=mu_pts, hfa=HFA_I)

def fit_club():
    """Club equivalent of fit_intl() - never existed before 22 Nov because the Matches tab never needed to
    predict a club game (club ratings/tables used league_sim.py's own Monte Carlo instead). Reuses
    league_sim.fit_club_common() for R/sigma/win% (already validated - Club Brier 0.1725) and fits the same
    totals-model recipe Tests already use (polyfit of attack+defence sum against real totals) since no club
    totals calibration existed either."""
    F_ = LS.fit_club_common()
    S, _, ATT, DEF = F.club_state()
    hist = S[S.date >= '2023-01-01']
    totb = np.polyfit(hist.tot, hist.total, 1)
    C = G[G.kind == 'club']
    mu_pts = C[['hs', 'as_']].values.mean()
    return dict(R=F_['R'], a=F_['a'], b=F_['b'], sigma=F_['sigma'], ATT=ATT, DEF=DEF, totb=totb, mu_pts=mu_pts)

def project(home, away, competition, kind, fit):
    hfa = fit['hfa'] if kind == 'intl' else CHFA[KEY[competition]]
    R, a, b, sigma, ATT, DEF, totb, mu_pts = fit['R'], fit['a'], fit['b'], fit['sigma'], fit['ATT'], fit['DEF'], fit['totb'], fit['mu_pts']
    mg = R.get(home, 0) - R.get(away, 0) + hfa
    ph = norm.cdf(a + b * (mg - 0.5)); pa = 1 - norm.cdf(a + b * (mg + 0.5))
    eh = mu_pts + ATT.get(home, 0) - DEF.get(away, 0) + hfa / 2
    ea = mu_pts + ATT.get(away, 0) - DEF.get(home, 0) - hfa / 2
    tot = float(np.polyval(totb, eh + ea))
    row = dict(p_home=float(ph), p_away=float(pa), p_draw=max(1 - ph - pa, 0),
              proj_home=round((tot + mg) / 2), proj_away=round((tot - mg) / 2), spread=-mg, total=tot,
              fair_home=amer(1 / ph), fair_away=amer(1 / pa), sigma=float(sigma))
    # sigma travels with the match (not just the old single top-level d['probit']) because the live win-prob
    # poller needs the RIGHT one per match now - club sigma (13.33) and intl sigma (13.84) differ, and several
    # competitions can be on screen at once (22 Nov)
    # Market comparison is optional - no free live odds source exists (checked again 22 Nov via ESPN's richer
    # per-event endpoint: hasOdds is still false). Only present when Wyatt has manually entered a board for
    # this specific (competition, home team) in round_config.MKT; otherwise these stay None and the template/
    # bets.py both treat that as "nothing to compare against" rather than an error.
    mkt = MKT.get(competition, {}).get(home)
    if mkt:
        o1, ox, o2, ln, oh, oa = mkt
        row.update(mkt_home=amer(o1), mkt_away=amer(o2), mkt_spread=ln, mdh=o1, mda=o2,
                  p_cover_home=1 - norm.cdf(-ln, mg, sigma))
    else:
        row.update(mkt_home=None, mkt_away=None, mkt_spread=None, mdh=None, mda=None, p_cover_home=None)
    return row, mg

def club_ratings():
    C = G[G.kind == 'club'].reset_index(drop=True)
    _, R = elo(C, 'club', 0.06, 5, 40, 0.2, hfa_map=CHFA, sa_bonus=1.0)
    LG = {}
    for comp, lg in (('French Top 14', 'Top 14'), ('Gallagher Premiership', 'Prem'), ('United Rugby Championship', 'URC')):
        cur = C[(C.competition == comp) & (C.date >= '2026-08-01')]
        for t in pd.concat([cur.home, cur.away]).unique(): LG[t] = lg
    return [dict(team=t, rating=round(float(R[t]), 2), league=LG[t]) for t in LG]

def super_rugby():
    df = SR.games(); _, R = SR.elo(df, 0.03, 1, 40, 0.2, travel_bonus=2, drua_bonus=6)
    R.pop('Rebels', None); mu = np.mean(list(R.values()))
    return sorted([dict(team=t, rating=round(float((v - mu) * 0.8), 2)) for t, v in R.items()], key=lambda x: -x['rating'])

def live_backtest():
    """The Method tab's headline numbers were hardcoded literals, never recomputed - found stale 21 Nov
    (club/try had both drifted from the actual current model after this session's fixes: club 0.1725 vs the
    real 0.1758 post-SA-bonus, try 0.1142 vs the real 0.11451 post-leakage-fix). Each model already has its
    own validation harness; this just calls them and reads off the test split instead of trusting a number
    someone typed in once."""
    I = intl_games(True)
    pr, _ = elo(I, 'intl', 0.08, HFA_I, 60, 0.0)
    si = score(I, 'intl', pr, mask=t1_mask(I))
    C = G[G.kind == 'club'].reset_index(drop=True)
    pc, _ = elo(C, 'club', 0.06, 5, 40, 0.2, hfa_map=CHFA, sa_bonus=1.0)
    sc = score(C, 'club', pc)
    import tries as TR
    tp = TR.load(); P, _, _ = TR.walk_forward(tp, test_from='2025-08-01', k=8)
    te = P[P.date >= '2025-08-01']; q = te.p_full.clip(1e-4, 1 - 1e-4)
    try_brier = float(np.mean((q - te.y) ** 2))
    df = SR.games(); ps, _ = SR.elo(df, 0.03, 1, 40, 0.2, travel_bonus=2, drua_bonus=6)
    ssr = SR.score(df, ps)
    return dict(intl_brier=si['test'], intl_acc=si['test_acc'], club_brier=sc['test'], try_brier=round(try_brier, 4),
                sr_brier=ssr['test'], sr_acc=ssr['test_acc'])

def try_scorers(R, intl_matches):
    """Test-only (no club equivalent - see CLAUDE.md's model table). intl_matches: list of (home, away, mg)
    for this cycle's international round(s) - empty when none are on, which is fine, just no tries that week."""
    if not intl_matches: return [], None
    import tries as TR
    p = TR.load(); P, mt, share = TR.walk_forward(p, test_from='2027-01-01', k=8)
    cal = json.load(open('../model/tries_calib.json'))
    P = P.sort_values('date'); last = P.groupby('player_id').tail(1).set_index('player_id')
    mult_now = (last.cum_t + last.tries + 8) / (last.cum_e + last.exp_t + 8)
    names = P.groupby('player_id').name.last()
    nat = P[(P.kind == 'intl') & (P.date >= '2025-01-01')]
    rows = []
    for h, a, mg, tot in intl_matches:
        for team, sgn, pts in ((h, 1, (tot + mg) / 2), (a, -1, (tot - mg) / 2)):
            em = sgn * mg
            lam = mt.predict(pd.DataFrame(dict(em2=[np.clip(em, -40, 40)], intl=[1], pts=[pts])))[0]
            st = nat[(nat.team == team) & ~nat.grp.str.startswith('sub')]
            if st.empty: continue
            ap = st.groupby('player_id').agg(n=('match_id', 'nunique'), grp=('grp', lambda x: x.mode().iloc[0]), lastd=('date', 'max'))
            ap = ap[(ap.n >= 2) & (ap.lastd >= pd.Timestamp('2025-06-01'))]
            ap['mult'] = mult_now.reindex(ap.index).fillna(1)
            raw = (1 - np.exp(-cal['scale'] * lam * ap.grp.map(share) * ap.mult)).clip(1e-4, .999)
            z = cal['intercept'] + cal['coef'][0] * np.log(raw / (1 - raw)) + cal['coef'][1] * 1 + cal['coef'][2] * 0
            ap['p'] = 1 / (1 + np.exp(-z))
            ap['name'] = names.reindex(ap.index); ap['team'] = team; ap['team_tries'] = lam
            rows.append(ap.sort_values('p', ascending=False).head(3))
    if not rows: return [], None
    X = pd.concat(rows)
    return [dict(team=r.team, name=r['name'], pos=r.grp.replace('_', ' '), p=float(r.p), fair=amer(1 / r.p), bet_at=amer(1.05 / r.p))
            for _, r in X.iterrows()], X

def main():
    active = AR.active_rounds()
    summary = ', '.join(f"{r['competition']} ({len(r['matches'])})" for r in active)
    print(f'{len(active)} active competition round(s): {summary}')
    need_intl = any(r['kind'] == 'intl' for r in active)
    need_club = any(r['kind'] == 'club' for r in active)
    fit = {}
    if need_intl: fit['intl'] = fit_intl()
    if need_club: fit['club'] = fit_club()

    rounds, intl_for_tries = [], []
    for r in active:
        f = fit[r['kind']]
        matches = []
        for m in r['matches']:
            proj, mg = project(m['home'], m['away'], r['competition'], r['kind'], f)
            matches.append(dict(home=m['home'], away=m['away'], venue=m['venue'], ko_utc=m['ko_utc'],
                                competition=r['competition'], **proj))
            if r['kind'] == 'intl': intl_for_tries.append((m['home'], m['away'], mg, proj['total']))
        rounds.append(dict(competition=r['competition'], round_label=r['round_label'], kind=r['kind'], matches=matches))

    R_intl = fit['intl']['R'] if need_intl else {}
    T1L = sorted([(t, R_intl.get(t, 0)) for t in T1], key=lambda x: -x[1])
    tries, _ = try_scorers(R_intl, intl_for_tries)
    import grades_produce as GP
    import player_rankings as PR
    import recap_report as RC
    import standings_produce as SP
    graded, grades_match, grade_values = GP.player_grades()
    rankings = PR.site_rankings()
    game_recap = RC.recap()
    standings = SP.all_standings()
    try:
        d = json.load(open(OUT))      # keep bets text (betsTab is hand-maintained, see bets.py)
    except FileNotFoundError:
        # site.json is gitignored (it carries betsTab - see README, 22 Nov), so a fresh checkout - a new
        # clone, or CI - has none at all. Found this the hard way: the first real CI run of the "Rebuild
        # site" workflow crashed right here. Starting from {} is correct either way - bets.build() always
        # rewrites betsTab from scratch regardless, there's nothing real to "keep" on a true first run.
        d = {}
    d.pop('round', None); d.pop('matches', None); d.pop('competition', None); d.pop('round_label', None)
    d.pop('round_date', None)   # all replaced by `rounds` (22 Nov) - drop the old single-round fields outright
    # rather than leave a stale copy no code reads anymore (same "orphaned field" mistake found 21 Nov, see README)
    d.setdefault('backtest', {})
    d.update(updated=date.today().strftime('%-d %b'), rounds=rounds, market_date=MARKET_DATE,
             intl=[dict(team=t, rating=round(float(v), 2)) for t, v in T1L], club=club_ratings(),
             sr=super_rugby(), tries=tries, grades=graded, grades_match=grades_match,
             grade_values={k: round(float(v), 4) for k, v in grade_values.items()}, rankings=rankings,
             recap=game_recap, standings=standings,
             # Client-side live polling needs these (site/mobile_template.html - CORS is open on both APIs,
             # confirmed 22 Nov, so the static page fetches live scores/stats itself, no backend needed): one
             # ESPN league id per active competition (there can be several concurrently now), and how to read
             # World Rugby's live per-player stat keys into our schema.
             live=dict(leagues=[dict(competition=r['competition'], espn_league=ESPN_LEAGUE_IDS.get(r['competition']))
                                for r in active],
                       wr_field_map={k: FIELD_MAP[k] for k in list(grade_values) + ['points']}))
    d['backtest'].update(live_backtest())
    if need_intl:
        d['probit'] = dict(a=fit['intl']['a'], b=fit['intl']['b'], sigma=fit['intl']['sigma'])
    json.dump(d, open(OUT, 'w'), default=float)
    for rnd in rounds:
        print(f"-- {rnd['competition']} ({rnd['round_label']}) --")
        for m in rnd['matches']:
            print(f"{m['home']:9s} {m['proj_home']:2d}-{m['proj_away']:2d} {m['away']:13s} p={m['p_home']:.3f}/{m['p_away']:.3f} spread {m['spread']:+.1f} total {m['total']:.1f}")
    print('tries', len(tries), sorted(set(t['team'] for t in tries)))
    import bets   # always rebuild betsTab from these same fresh matches/probit - a stale run order (produce.py
    bets.build()  # changing predictions, betsTab not regenerated after) was a real gap found 21 Nov.

if __name__ == '__main__':
    main()
