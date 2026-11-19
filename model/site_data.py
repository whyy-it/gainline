"""Build JSON for the demo site: match projections, try scorers, power ratings, post-match player grades."""
import json, glob, numpy as np, pandas as pd
from scipy.stats import norm
import features as F
from model import G

R = json.load(open('../model/ratings.json'))
SIG = R['SIG_I']
NC6 = [('England','New Zealand','Twickenham',2.25,1.68,2.5,'14:10 GMT'),('Scotland','Japan','Murrayfield',1.008,23.0,-30.5,'14:10 GMT'),
       ('Ireland','South Africa','Aviva Stadium',4.33,1.24,9.5,'16:40 GMT'),('Italy','Fiji','Udine',1.47,2.80,-5.5,'16:40 GMT'),
       ('France','Argentina','Stade de France',1.062,10.5,-19.5,'20:10 GMT'),('Wales','Australia','Principality Stadium',2.85,1.44,5.5,'20:10 GMT')]

def amer(d):
    return f'+{round((d-1)*100)}' if d >= 2 else f'{round(-100/(d-1))}'

# ---- projections: margin from Elo, total from attack/defence state
fx = pd.DataFrame([dict(date=pd.Timestamp('2026-11-21'), kind='intl', competition='Nations Championship', city=None,
                        home=h, away=a, hs=0., as_=0., neutral=False) for h, a, *_ in NC6])
I = pd.concat([G[G.kind == 'intl'], fx], ignore_index=True)
S, *_ = F.build_state(I, 'intl', F.IK, lambda r: F.IH, F.ICAP, F.IREG)
hist = S.iloc[:-6]; T1 = {'England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji'}
h1 = hist[hist.home.isin(T1) & hist.away.isin(T1) & (hist.date >= '2023-01-01')]
# total: blend att/def total with tier-1 league average, fitted by OLS
X = np.c_[np.ones(len(h1)), h1.tot]; b = np.linalg.lstsq(X, h1.total, rcond=None)[0]
sig_tot = (h1.total - X @ b).std()
matches = []
for (h, a, venue, o1, o2, ln, ko), (_, s) in zip(NC6, S.tail(6).iterrows()):
    m = R['RI'][h] - R['RI'][a] + F.IH
    tot = b[0] + b[1] * s.tot
    hs, as_ = (tot + m) / 2, (tot - m) / 2
    ph = 1 - norm.cdf(0.5, m, SIG); pa = norm.cdf(-0.5, m, SIG)
    ou_line = round(tot * 2) / 2 + 0.5
    matches.append(dict(home=h, away=a, venue=venue, ko=ko, p_home=ph, p_draw=1 - ph - pa, p_away=pa,
                        proj_home=round(hs), proj_away=round(as_), spread=-m, total=tot, ou_line=ou_line,
                        p_over=1 - norm.cdf(ou_line, tot, sig_tot),
                        fair_home=amer(1/ph), fair_away=amer(1/pa),
                        mkt_home=amer(o1), mkt_away=amer(o2), mkt_spread=ln,
                        p_cover_home=1 - norm.cdf(-ln, m, SIG)))

# ---- try scorers
X_ = pd.read_csv('../model/tryscorer_preview.csv')
tries = [dict(team=r.team, name=r['name'], pos=r.grp.replace('_', ' '), p=r.p, fair=amer(1/r.p), bet_at=amer(1.05/r.p))
         for _, r in X_.iterrows()]

# ---- power ratings
ri = sorted([(t, R['RI'][t]) for t in T1], key=lambda x: -x[1])
cr = pd.read_csv('../model/club_ratings.csv', index_col=0).rating_pts
LEAGUE = {}
for t in ['Toulouse','Bordeaux-Begles','La Rochelle','Racing 92','Stade Francais','Toulon','Clermont Auvergne','Montpellier','Pau','Castres','Bayonne','Lyon','Perpignan','Vannes']: LEAGUE[t] = 'Top 14'
for t in ['Bath','Bristol Bears','Exeter','Gloucester','Harlequins','Leicester Tigers','Newcastle','Northampton','Sale Sharks','Saracens']: LEAGUE[t] = 'Prem'
for t in ['Leinster','Bulls','Stormers','Sharks','Glasgow Warriors','Munster','Ulster','Connacht','Benetton','Edinburgh','Lions','Cardiff','Ospreys','Scarlets','Dragons','Zebre']: LEAGUE[t] = 'URC'
club = [dict(team=t, rating=float(v), league=LEAGUE.get(t, '')) for t, v in cr.items() if t in LEAGUE]

# ---- post-match player grades (SA v NZ, Cape Town, 29 Aug 2026)
cols = ['player_id','name','team','position','tries','try_assists','line_breaks','defenders_beaten','metres_carried','offloads',
        'carries','tackles_made','tackles_missed','turnovers_won','penalties_conceded','yellow_cards','red_cards','lineouts_won',
        'dominant_tackles','game_date','competition','match_id']
L = []
for f in glob.glob('../data/rugbydata/data/v3/player/*.parquet'):
    try:
        x = pd.read_parquet(f, columns=cols)
    except Exception:
        continue
    x = x[x.competition.isin(['International Test Match','Six Nations','The Rugby Championship','Rugby World Cup'])]
    if len(x): L.append(x)
P = pd.concat(L).drop_duplicates(['player_id','match_id'])
num = cols[4:19]
P[num] = P[num].apply(pd.to_numeric, errors='coerce').fillna(0)
P = P[(P.tries <= 4) & (P.tackles_made <= 40) & (P.metres_carried <= 300)]
W = dict(tries=6, try_assists=4, line_breaks=3, defenders_beaten=1, metres_carried=0.06, offloads=1, carries=0.3,
         tackles_made=0.45, tackles_missed=-1.2, turnovers_won=3, penalties_conceded=-2, yellow_cards=-6, red_cards=-15,
         lineouts_won=0.8, dominant_tackles=0.6)
P['raw'] = sum(P[k] * w for k, w in W.items())
from tries import GROUP
P['grp'] = P.position.map(GROUP).fillna('sub_back')
st = P.groupby('grp').raw.agg(['mean', 'std'])
P['grade'] = (60 + 12 * (P.raw - P.grp.map(st['mean'])) / P.grp.map(st['std'])).clip(30, 99).round()
M = P[P.match_id == 'e1440199'].copy()
M['key'] = M.apply(lambda r: ', '.join(f'{int(r[k])} {lab}' for k, lab in [('tries','T'),('line_breaks','LB'),('metres_carried','m'),
                   ('tackles_made','tkl'),('turnovers_won','TO')] if r[k] > 0), axis=1)
grades = [dict(team=r.team, name=r['name'], pos=r.grp.replace('_', ' '), grade=int(r.grade), key=r.key)
          for _, r in M.sort_values('grade', ascending=False).iterrows()]

out = dict(updated='17 Nov 2026', round='Nations Championship · Round 6 · Sat 21 Nov', matches=matches, tries=tries,
           intl=[dict(team=t, rating=float(v)) for t, v in ri], club=club, grades=grades,
           grades_match='South Africa 33–26 New Zealand · Cape Town · 29 Aug 2026',
           backtest=dict(intl_brier=0.1636, intl_acc=0.748, club_brier=0.1795, try_brier=0.1138))
json.dump(out, open('../model/site.json', 'w'), default=float)
print(json.dumps(matches[0], default=float, indent=0)[:600]); print(len(grades), grades[:3]); print('total fit', b, sig_tot)
