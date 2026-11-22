"""Independent cross-check of every number on Gainline.
Re-implements the Test rating loop from scratch (no shared code), re-derives probabilities, scores, odds,
EVs and stakes, and checks the HTML actually carries site.json. Prints PASS/FAIL per check, exits non-zero
on any failure so sync_player_stats.yml's scheduled run won't commit/push a bad build."""
import sys, json, re, pathlib, numpy as np, pandas as pd
from scipy.stats import norm

d = json.load(open('../model/site.json'))
html = open('../site/gainline.html').read()
fails = []
def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail and not ok else ''))
    if not ok: fails.append(name)

_club_sigma = []
def club_sigma():   # lazy + memoized - only worth the fit cost if a club round actually has a market board
    if not _club_sigma:
        import league_sim as LS
        _club_sigma.append(LS.fit_club_common()['sigma'])
    return _club_sigma[0]

# 1. Public HTML (what's actually committed/served - CLAUDE.md: "the Bets tab is private") carries every
# non-betting number from site.json, and genuinely carries NO betting content - not just a hidden tab, the
# data itself must be absent (see build_site.py, 22 Nov). The private local build, if present, must still
# carry betsTab correctly - gated separately since it's gitignored and won't exist on a fresh clone.
m = re.search(r'const D = (\{.*?\});\n', html, re.S)
emb = json.loads(m.group(1)) if m else None
check('Public HTML carries rounds/tries', emb is not None and emb['rounds'] == d['rounds'] and emb['tries'] == d['tries'])
check('Public HTML has no betting content', emb is not None and 'betsTab' not in emb and 'remove before publishing' not in html.lower())
priv = pathlib.Path('../site/gainline_private.html')
if priv.exists():
    pm = re.search(r'const D = (\{.*?\});\n', priv.read_text(), re.S)
    pemb = json.loads(pm.group(1)) if pm else None
    check('Private HTML carries betsTab matching site.json', pemb is not None and pemb.get('betsTab', {}).get('bets') == d['betsTab']['bets'])
else:
    print('SKIP Private HTML check (site/gainline_private.html not built - run build_site.py)')

# 2. Independent Test ratings: fresh Elo loop over the same inputs
G = pd.read_parquet('../model/games.parquet')
k = pd.read_csv('../data/kyejones25-alt_rugby-sql-analysis/data/internationals.csv', parse_dates=['date'])
T1 = ['England','France','Ireland','Italy','Scotland','Wales','New Zealand','Australia','South Africa','Argentina','Japan','Fiji']
k = k[(k.date >= '1995-07-01') & (k.date < '2022-06-01')]
rows = [(r.date, r.home_team, r.away_team, float(r.home_score), float(r.away_score), bool(r.neutral)) for r in k.itertuples()]
gi = G[G.kind == 'intl']; gi = gi[~((gi.date < '2022-06-01') & gi.home.isin(T1) & gi.away.isin(T1))]
rows += [(r.date, r.home, r.away, r.hs, r.as_, bool(r.neutral)) for r in gi.itertuples()]
rows.sort(key=lambda x: x[0])
R = {}
for dt, h, a, hs, as_, neu in rows:
    p = R.get(h, 0) - R.get(a, 0) + (0 if neu else 4)
    e = max(-60, min(60, hs - as_)) - p
    R[h] = R.get(h, 0) + .08 * e; R[a] = R.get(a, 0) - .08 * e
site_r = {x['team']: x['rating'] for x in d['intl']}
mx = max(abs(R[t] - site_r[t]) for t in T1)
check('Test ratings match an independent re-implementation', mx < 0.02, f'max diff {mx:.3f}')

# 3. Match cards: margins, probabilities, scores, odds. Tests get the full independent re-derivation (the
# from-scratch Elo loop above); club rounds (22 Nov: several competitions can be active at once, see
# produce.py) get an internal-consistency check instead - re-deriving club ratings from scratch here too
# would just duplicate features.club_state()/league_sim.py rather than independently verify them.
w = d.get('probit')
all_matches = [(rnd, mt) for rnd in d['rounds'] for mt in rnd['matches']]
def fair_odds(p):   # same clipping as produce.py's amer_safe - a final match's p_home is exactly 1/0/0.5
    p = min(max(p, 1e-4), 1 - 1e-4); fh = 1 / p
    return f'+{round((fh-1)*100)}' if fh >= 2 else f'{round(-100/(fh-1))}'

for rnd, mt in all_matches:
    h, a = mt['home'], mt['away']
    ok = abs(mt['p_home'] + mt['p_draw'] + mt['p_away'] - 1) < 1e-9
    ok &= pd.Timestamp(mt['ko_utc']).tzinfo is not None   # a real, timezone-aware instant (ESPN's own date field)
    if mt.get('final'):
        # apply_final() (produce.py, 23 Nov) overwrites proj_home/away with the REAL score once ESPN confirms
        # a match is over - the pre-match total/spread derivation and fair-odds-from-model-prob checks below
        # don't apply anymore, so check the final-specific invariant instead: the embedded result is self-
        # consistent (win/loss/draw correctly reflected, fair odds correctly derived from the now-certain prob).
        hs, as_ = mt['proj_home'], mt['proj_away']
        ok &= (mt['p_home'] == 1.0 if hs > as_ else (mt['p_home'] == 0.0 if hs < as_ else mt['p_home'] == 0.5))
        ok &= fair_odds(mt['p_home']) == mt['fair_home'] and fair_odds(mt['p_away']) == mt['fair_away']
        check(f"Match card {h} v {a} ({rnd['competition']}) - final result baked in correctly", bool(ok))
        continue
    ok &= mt['proj_home'] == round((mt['total'] - mt['spread']) / 2) and mt['proj_away'] == round((mt['total'] + mt['spread']) / 2)
    ok &= fair_odds(mt['p_home']) == mt['fair_home']
    if mt['mkt_spread'] is not None:   # only present when round_config.MKT has a board for this match
        sigma = w['sigma'] if rnd['kind'] == 'intl' else club_sigma()
        pc = 1 - norm.cdf(-mt['mkt_spread'], -mt['spread'], sigma)
        ok &= abs(pc - mt['p_cover_home']) < 1e-3
    if rnd['kind'] == 'intl':
        mg = R[h] - R[a] + 4
        ok &= abs(-mt['spread'] - mg) < 0.05
        ph = norm.cdf(w['a'] + w['b'] * (mg - .5)); pa = 1 - norm.cdf(w['a'] + w['b'] * (mg + .5))
        ok &= abs(ph - mt['p_home']) < 1e-3 and abs(pa - mt['p_away']) < 1e-3
        check(f'  win % ordering matches spread sign ({h})', (mt['p_home'] > mt['p_away']) == (mg > 0))
    check(f"Match card {h} v {a} ({rnd['competition']})", bool(ok))

# 3b. Live-polling config (site/mobile_template.html fetches ESPN/World Rugby directly client-side, 22 Nov -
# this is the data it needs to find and read them; wrong here means live scores/grades silently never work)
from round_config import ESPN_LEAGUE_IDS
live = d.get('live', {})
want = {rnd['competition'] for rnd in d['rounds']}
have = {l['competition'] for l in live.get('leagues', [])}
check('Live config: one ESPN league entry per active round', want == have, f'{want} vs {have}')
check('Live config: every league id matches round_config',
      all(l['espn_league'] == ESPN_LEAGUE_IDS.get(l['competition']) for l in live.get('leagues', [])))
check('Live config: wr_field_map covers every grade_values stat + points',
      set(live.get('wr_field_map', {})) == set(d['grade_values']) | {'points'})

# 4. Bets: EV, stakes, rules
for b in d['betsTab']['bets']:
    nd = b.get('nd', 1.0); ev = nd * (b['blend'] * b['dec'] - 1); qk = nd * max((b['blend'] * (b['dec'] - 1) - (1 - b['blend'])) / (b['dec'] - 1), 0) / 4
    ok = abs(ev - b['ev']) < 1e-9 and abs(qk - b['qk']) < 1e-9 and abs(b['blend'] - (b['model'] + b['mkt']) / 2) < 1e-9
    ok &= b['ev'] >= 0.05 and (b['kind'] != 'ML' or b['gap'] >= 3)
    check(f"Bet {b['bet']} arithmetic and rules", bool(ok))

# 5. Try scorers
ok = all(0 < t['p'] < 1 for t in d['tries'])
for t in d['tries']:
    f = 1 / t['p']; s = f'+{round((f-1)*100)}' if f >= 2 else f'{round(-100/(f-1))}'
    ok &= s == t['fair']
check('Try probabilities in range, odds consistent', bool(ok))
n_intl_teams = sum(2 * len(rnd['matches']) for rnd in d['rounds'] if rnd['kind'] == 'intl')   # try scorer is
# Test-only (no club equivalent, see CLAUDE.md) - 0 when no intl round is active, which is fine, not a bug
n_intl_teams -= sum(2 for rnd in d['rounds'] if rnd['kind'] == 'intl' for mt in rnd['matches'] if mt.get('final'))
# a match already confirmed final (23 Nov) correctly gets no try-scorer props computed for it - see produce.py
if n_intl_teams:
    check(f'Try scorers for all {n_intl_teams} teams this round', len(set(t['team'] for t in d['tries'])) == n_intl_teams)
    tt = pd.DataFrame(d['tries']).groupby('team').p.sum()
    check('No team implausibly loaded (top-3 sum < 1.8)', bool((tt < 1.8).all()), tt.round(2).to_dict())
else:
    check('No intl round active -> tries list correctly empty', len(d['tries']) == 0)

# 6. Ratings tables
lg = pd.Series([c['league'] for c in d['club']]).value_counts().to_dict()
check('Club leagues complete (Top 14 14, Prem 10, URC 16)', lg == {'URC': 16, 'Top 14': 14, 'Prem': 10}, lg)
check('Super Rugby 11 teams, centred', len(d['sr']) == 11 and abs(sum(x['rating'] for x in d['sr'])) < 0.1)
check('Test ratings sorted', [x['rating'] for x in d['intl']] == sorted([x['rating'] for x in d['intl']], reverse=True))

print('\n%d checks failed' % len(fails) if fails else '\nAll checks passed')
sys.exit(1 if fails else 0)
