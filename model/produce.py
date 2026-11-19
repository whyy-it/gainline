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
from scipy.stats import norm
warnings.filterwarnings('ignore')
import features as F
from v6 import intl_games, elo, probit_fit, season, T1, G, CHFA
import sr as SR

OUT = '../model/site.json'
FIX = [('England', 'New Zealand', 'Twickenham', '14:10 GMT'), ('Scotland', 'Japan', 'Murrayfield', '14:10 GMT'),
       ('Ireland', 'South Africa', 'Aviva Stadium', '16:40 GMT'), ('Italy', 'Fiji', 'Udine', '16:40 GMT'),
       ('France', 'Argentina', 'Stade de France', '20:10 GMT'), ('Wales', 'Australia', 'Principality Stadium', '20:10 GMT')]
# market board (decimal): 1, X, 2, home handicap, home hcp odds, away hcp odds  -- Sportingbet ZA, 17 Nov
MKT = {'England': (2.0, 1e9, 1.7692, 0.5, 1.9434, 1.8621),
       'Scotland': (1.005, 1e9, 18.0, -27.5, 1.8772, 1.9259),
       'Ireland': (4.2, 1e9, 1.2083, 9.5, 1.8929, 1.9259),
       'Italy': (1.4, 1e9, 2.85, -5.5, 1.9434, 1.8621),
       'France': (1.0286, 1e9, 10.8, -19.5, 1.9259, 1.8929),
       'Wales': (2.75, 1e9, 1.4255, 5.5, 1.8929, 1.9259)}
HFA_I = 4.0

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
    _, R = elo(C, 'club', 0.06, 5, 40, 0.2, hfa_map=CHFA)
    LG = {}
    for comp, lg in (('French Top 14', 'Top 14'), ('Gallagher Premiership', 'Prem'), ('United Rugby Championship', 'URC')):
        cur = C[(C.competition == comp) & (C.date >= '2026-08-01')]
        for t in pd.concat([cur.home, cur.away]).unique(): LG[t] = lg
    return [dict(team=t, rating=round(float(R[t]), 2), league=LG[t]) for t in LG]

def super_rugby():
    df = SR.games(); _, R = SR.elo(df, 0.03, 1, 40, 0.2, travel_bonus=2, drua_bonus=6)
    R.pop('Rebels', None); mu = np.mean(list(R.values()))
    return sorted([dict(team=t, rating=round(float((v - mu) * 0.8), 2)) for t, v in R.items()], key=lambda x: -x['rating'])

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
        matches.append(dict(home=h, away=a, venue=venue, ko=ko, p_home=ph, p_away=pa, p_draw=max(1 - ph - pa, 0),
                            proj_home=round((tot + mg) / 2), proj_away=round((tot - mg) / 2), spread=-mg, total=tot,
                            fair_home=amer(1 / ph), fair_away=amer(1 / pa), mkt_home=amer(o1), mkt_away=amer(o2),
                            mkt_spread=ln, mdh=o1, mda=o2, p_cover_home=1 - norm.cdf(-ln, mg, sig)))
    T1L = sorted([(t, R[t]) for t in T1], key=lambda x: -x[1])
    tries, _ = try_scorers(R, up)
    d = json.load(open(OUT))          # keep grades / grade values / bets text
    d.update(matches=matches, intl=[dict(team=t, rating=round(float(v), 2)) for t, v in T1L], club=club_ratings(),
             sr=super_rugby(), tries=tries)
    d['backtest'].update(intl_brier=0.1584, intl_acc=0.767, club_brier=0.1725, try_brier=0.1142, sr_brier=0.1695, sr_acc=0.759)
    d['probit'] = dict(a=float(w[0]), b=float(w[1]), sigma=float(1 / w[1]))
    json.dump(d, open(OUT, 'w'), default=float)
    for m in matches:
        print(f"{m['home']:9s} {m['proj_home']:2d}-{m['proj_away']:2d} {m['away']:13s} p={m['p_home']:.3f}/{m['p_away']:.3f} spread {m['spread']:+.1f} total {m['total']:.1f} cover {m['p_cover_home']:.3f}")
    print('tries', len(tries), sorted(set(t['team'] for t in tries)))

if __name__ == '__main__':
    main()
