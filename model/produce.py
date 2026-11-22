"""One script that rebuilds every number on Gainline from the final models.
Run order: python3 build_data.py && python3 produce.py   -> writes site.json
Final models (each validated on unseen seasons, see README):
  Tests        margin-Elo on 1995+ history, K .08, HFA 4, probit win prob calibrated on 2023+ tier-1 games
  Totals       attack/defence state (long history, rate .08), linear map fitted on 2023+ tier-1 games
  Clubs        margin-Elo K .06, HFA Top14 9 / Prem 7 / URC 6 / Europe 6, 20% season regression
  Super Rugby  margin-Elo K .03, HFA 1 +2 cross-border +6 Drua in Fiji, 20% regression
  Try scorer   team tries (Poisson on margin, expected points) x position share x player multiplier (k=8),
               then logistic recalibration (fitted 2024-25)"""
import json, numpy as np, pandas as pd, warnings
from datetime import date, datetime, timezone
from scipy.stats import norm
warnings.filterwarnings('ignore')
import features as F
from v6 import intl_games, elo, probit_fit, season, T1, G, CHFA, score, t1_mask
import sr as SR

OUT = '../model/site.json'
from round_config import COMPETITION, ROUND_LABEL, ROUND_DATE, MARKET_DATE, FIX, MKT, ESPN_LEAGUE_IDS   # shared with bets.py
from wr_stats_sync import FIELD_MAP   # client needs this to read World Rugby's live per-player stat keys
HFA_I = 4.0
_MON = {m: i + 1 for i, m in enumerate(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}

def ko_utc(ko):
    """Every ko string in round_config.FIX is hand-entered in GMT (UTC+0 - fine year-round since GMT, unlike
    BST, never shifts), e.g. '16:40 GMT'. ROUND_DATE gives day+month for the whole round but no year (see its
    docstring - assumes the round is "this week"), so infer year from today, correcting for the one real edge
    case: a round that straddles New Year's (round in Dec, checked in Jan, or vice versa)."""
    _, day, mon = ROUND_DATE.split()
    today = date.today()
    year = today.year
    if _MON[mon] == 12 and today.month == 1: year -= 1
    elif _MON[mon] == 1 and today.month == 12: year += 1
    h, m = map(int, ko.split()[0].split(':'))
    return datetime(year, _MON[mon], int(day), h, m, tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')

def amer(dec): return f'+{round((dec - 1) * 100)}' if dec >= 2 else f'{round(-100 / (dec - 1))}'

def tests():
    I = intl_games(True)
    pr, R = elo(I, 'intl', 0.08, HFA_I, 60, 0.0)
    y = (I.hs > I.as_).astype(float).values + 0.5 * (I.hs == I.as_).values
    ss = np.array([season(d, 'intl') for d in I.date])
    m = (I.home.isin(T1) & I.away.isin(T1)).values & (ss >= 2023)
    w = probit_fit(pr[m], y[m])
    # totals: attack/defence state incl. this round's fixtures (pre-match rows)
    fx = pd.DataFrame([dict(date=pd.Timestamp('2026-11-21'), kind='intl', competition='Nations Championship', city=None,
                            home=h, away=a, hs=0., as_=0., neutral=False) for h, a, *_ in FIX])
    S, *_ = F.build_state(pd.concat([I, fx], ignore_index=True), 'intl', 0.08, lambda r: HFA_I, 60, 0.0, Ka=0.08)
    hist = S.iloc[:-len(FIX)]; hist = hist[hist.home.isin(T1) & hist.away.isin(T1) & (hist.date >= '2023-01-01')]
    b = np.polyfit(hist.tot, hist.total, 1)
    up = S.tail(len(FIX)).reset_index(drop=True)
    return R, w, b, up

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

def try_scorers(R, up):
    import tries as TR
    from sklearn.linear_model import LogisticRegression
    p = TR.load(); P, mt, share = TR.walk_forward(p, test_from='2027-01-01', k=8)
    cal = json.load(open('../model/tries_calib.json'))
    P = P.sort_values('date'); last = P.groupby('player_id').tail(1).set_index('player_id')
    mult_now = (last.cum_t + last.tries + 8) / (last.cum_e + last.exp_t + 8)
    names = P.groupby('player_id').name.last()
    nat = P[(P.kind == 'intl') & (P.date >= '2025-01-01')]
    rows = []
    for (h, a, *_), (_, s) in zip(FIX, up.iterrows()):
        mg = R[h] - R[a] + HFA_I
        for team, sgn, pts in ((h, 1, (s.tot + s.ad) / 2), (a, -1, (s.tot - s.ad) / 2)):
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
    X = pd.concat(rows)
    return [dict(team=r.team, name=r['name'], pos=r.grp.replace('_', ' '), p=float(r.p), fair=amer(1 / r.p), bet_at=amer(1.05 / r.p))
            for _, r in X.iterrows()], X

def main():
    R, w, b, up = tests()
    matches = []
    for (h, a, venue, ko), (_, s) in zip(FIX, up.iterrows()):
        mg = R[h] - R[a] + HFA_I
        ph = norm.cdf(w[0] + w[1] * (mg - 0.5)); pa = 1 - norm.cdf(w[0] + w[1] * (mg + 0.5))
        tot = float(np.polyval(b, s.tot)); o1, ox, o2, ln, oh, oa = MKT[h]
        sig = 1 / w[1]
        matches.append(dict(home=h, away=a, venue=venue, ko=ko, ko_utc=ko_utc(ko), p_home=ph, p_away=pa, p_draw=max(1 - ph - pa, 0),
                            proj_home=round((tot + mg) / 2), proj_away=round((tot - mg) / 2), spread=-mg, total=tot,
                            fair_home=amer(1 / ph), fair_away=amer(1 / pa), mkt_home=amer(o1), mkt_away=amer(o2),
                            mkt_spread=ln, mdh=o1, mda=o2, p_cover_home=1 - norm.cdf(-ln, mg, sig)))
    T1L = sorted([(t, R[t]) for t in T1], key=lambda x: -x[1])
    tries, _ = try_scorers(R, up)
    import grades_produce as GP
    import player_rankings as PR
    import recap_report as RC
    import standings_produce as SP
    graded, grades_match, grade_values = GP.player_grades()
    rankings = PR.site_rankings()
    game_recap = RC.recap()
    standings = SP.all_standings()
    d = json.load(open(OUT))          # keep bets text (betsTab is hand-maintained, see bets.py)
    d.pop('round', None)   # orphaned top-level field from an old desktop template (site/template.html, dead
    # since mobile_template.html replaced it) - nothing in the current UI reads it; found 21 Nov while
    # auditing for anything NOT actually regenerated each run.
    d.update(updated=date.today().strftime('%-d %b'), competition=COMPETITION, round_label=ROUND_LABEL,
             round_date=ROUND_DATE, market_date=MARKET_DATE, matches=matches,
             intl=[dict(team=t, rating=round(float(v), 2)) for t, v in T1L], club=club_ratings(),
             sr=super_rugby(), tries=tries, grades=graded, grades_match=grades_match,
             grade_values={k: round(float(v), 4) for k, v in grade_values.items()}, rankings=rankings,
             recap=game_recap, standings=standings,
             # Client-side live polling needs these (site/mobile_template.html - CORS is open on both APIs,
             # confirmed 22 Nov, so the static page fetches live scores/stats itself, no backend needed):
             # which ESPN scoreboard to poll for this round's matches, and how to read World Rugby's live
             # per-player stat keys into our schema (same mapping wr_stats_sync.py uses for the post-match sync).
             live=dict(espn_league=ESPN_LEAGUE_IDS.get(COMPETITION),
                       wr_field_map={k: FIELD_MAP[k] for k in list(grade_values) + ['points']}))
    d['backtest'].update(live_backtest())
    d['probit'] = dict(a=float(w[0]), b=float(w[1]), sigma=float(1 / w[1]))
    json.dump(d, open(OUT, 'w'), default=float)
    for m in matches:
        print(f"{m['home']:9s} {m['proj_home']:2d}-{m['proj_away']:2d} {m['away']:13s} p={m['p_home']:.3f}/{m['p_away']:.3f} spread {m['spread']:+.1f} total {m['total']:.1f} cover {m['p_cover_home']:.3f}")
    print('tries', len(tries), sorted(set(t['team'] for t in tries)))
    import bets   # always rebuild betsTab from these same fresh matches/probit - a stale run order (produce.py
    bets.build()  # changing predictions, betsTab not regenerated after) was a real gap found 21 Nov.

if __name__ == '__main__':
    main()
