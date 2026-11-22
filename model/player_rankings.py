"""Club & country player rankings. Builds on grades_produce.py's validated data-driven weights (R2=0.65
single-match, vs hand weights' -0.79 - see README) but aggregates across games instead of one match.
Three windows (23 Nov): "2 Weeks", "Season" (since 1 Aug for clubs, since 1 Jan for country - the same
season-boundary convention v6.season() already uses elsewhere, not a third competing definition), and "Year"
(365 days - this is the old, only, "current form" window, just one of three now). Each is shrunk mean points-
added per 80-minutes-equivalent, empirical-Bayes shrunk toward the position group's mean with k pseudo-(80-
min-windows) - same recipe tries.py and lineup_value.py already use elsewhere in this repo, for consistency.
"2 Weeks" drops the games-played floor to 1 (from 5) - a 14-day window genuinely can't clear 5 games for
almost anyone, so this is deliberately a noisier, smaller-sample snapshot, not a stable rate.
Minutes-played only exists for the World Rugby sync rows added 21 Nov (wr_stats_sync.py); every older row has
no minutes field at all, so those are filled to a full 80 - same as the old implicit assumption, before
minutes existed, that every appearance in player_stats.parquet was a full match. Means a Sep-2026-and-earlier
appearance and a post-sync one with minutes=80 are treated identically; only the new partial-game rows (subs
coming on, players going off early) get properly weighted down instead of counted as a full game played.
Club and country are scored SEPARATELY: a player's club form and Test form can differ, and the eligible
player pool differs a lot (every league vs only capped internationals), so z-scoring them together would
be apples-to-oranges."""
import re
import numpy as np, pandas as pd
from tries import GROUP
from grades import team_diffs, learn_values, FEATS
from v6 import T1

INTL_COMPS = {'International Test Match', 'The Rugby Championship', 'Six Nations', 'Rugby World Cup',
              'Autumn Nations Series', 'Autumn Nations Cup', 'Nations Championship'}
EXCLUDE_COMP = re.compile(r"women|u[\s-]?20", re.I)   # different talent pools - not comparable to senior men's
MIN_GAMES = 5
K_SHRINK = 6  # pseudo-games toward the position-group mean

def load_graded():
    p = pd.read_parquet('player_stats.parquet')
    tm = pd.read_parquet('team_match_stats.parquet'); tm = tm[tm.ok]
    p = p[p.set_index(['match_id', 'team']).index.isin(tm.set_index(['match_id', 'team']).index)].copy()
    for f in FEATS + ['points']: p[f] = p[f].fillna(0)
    p['minutes'] = p['minutes'].fillna(80.0)   # pre-sync rows have no minutes field at all - assume a full match,
    # matching the old (pre-minutes) implicit behaviour exactly so this only changes rows that newly have real data
    p['grp'] = p.position.map(GROUP)
    p = p[p.grp.notna()]
    comp = p.competition.fillna(p.set_index(['match_id', 'team']).index.map(
        tm.set_index(['match_id', 'team']).comp).to_series(index=p.index))
    p = p[~comp.fillna('').str.contains(EXCLUDE_COMP)]   # drop women's/U20 - different talent pool, not comparable
    comp = comp[p.index]
    p['league'] = comp
    p['bucket'] = np.where(comp.isin(INTL_COMPS), 'country', 'club')
    p = p[(p.bucket == 'club') | p.team.isin(T1)]   # country bucket: tier-1 nations only, same convention as v6.py
    _, m, X = team_diffs()
    vals, r2 = learn_values(m, X, train_end=str(p.date.max() + pd.Timedelta(days=1)))  # fit on ALL history - this is a
    # retrospective valuation (what's an action worth), not a forecast, so no walk-forward need here; R2 already
    # validated out-of-sample in grades_produce.py's fit_report.
    p['pa'] = sum(p[f] * v for f, v in vals.items()) + 0.5 * p.points
    return p, vals

def _season_start(max_date, bucket):
    """Same boundary v6.season() uses elsewhere - club seasons run Aug-Jun (northern hemisphere leagues
    dominate what this site tracks), country/Test windows are just the calendar year."""
    if bucket == 'club':
        y = max_date.year if max_date.month >= 8 else max_date.year - 1
        return pd.Timestamp(y, 8, 1)
    return pd.Timestamp(max_date.year, 1, 1)

WINDOWS = ('2 Weeks', 'Season', 'Year')

def rank(p, bucket, window='Year', min_games=None):
    """club: z-scored within (league, position) - a Top14 prop isn't comparable to a Japan League One prop
    by raw rate, different competitive pools. country: z-scored within position only (already T1-only, one pool)."""
    x = p[p.bucket == bucket].copy()
    key = ['league', 'grp'] if bucket == 'club' else ['grp']
    maxd = x.date.max()
    since = {'2 Weeks': maxd - pd.Timedelta(days=14), 'Season': _season_start(maxd, bucket),
            'Year': maxd - pd.Timedelta(days=365)}[window]
    if min_games is None: min_games = 1 if window == '2 Weeks' else MIN_GAMES
    recent = x[x.date >= since]
    g = recent.groupby(['player_id'] + key).agg(name=('name', 'last'), team=('team', 'last'),
                                                 n=('pa', 'size'), pa_sum=('pa', 'sum'),
                                                 min_sum=('minutes', 'sum')).reset_index()
    g = g[g.n >= min_games]   # sample-size bar stays games played, not minutes - still 5 independent matches
    g['games_eq'] = g.min_sum / 80.0   # minutes-equivalent games; == n exactly for pre-sync players (minutes=80 always)
    gk = g.groupby(key)
    grp_mean = gk.apply(lambda d: d.pa_sum.sum() / d.games_eq.sum(), include_groups=False)
    g['grp_mean_pg'] = g.set_index(key).index.map(grp_mean)
    g['rate'] = (g.pa_sum + K_SHRINK * g.grp_mean_pg) / (g.games_eq + K_SHRINK)   # shrunk points-added per 80 min
    st = g.groupby(key).rate.agg(['mean', 'std'])
    g['z'] = ((g.rate - g.set_index(key).index.map(st['mean'])) / g.set_index(key).index.map(st['std'])).fillna(0.0)
    g['grade'] = (60 + 12 * g.z).clip(30, 99).round().astype(int)
    return g.sort_values('rate', ascending=False)

LEAGUE_LABEL = {'French Top 14': 'Top 14', 'Gallagher Premiership': 'Prem', 'United Rugby Championship': 'URC'}

def _format(R):
    return [dict(grp=r.grp, pos=r.grp.replace('_', ' '), name=r['name'], team=r.team, n=int(r.n),
                rate=round(float(r.rate), 2), grade=int(r.grade))
           for _, r in R.sort_values('grade', ascending=False).iterrows()]

def site_rankings():
    """Scoped to the competitions the rest of the site already tracks (Country/Top14/Prem/URC) rather than
    every league in the data (Pro D2, Japan League One, Mitre 10 Cup, etc. are out of scope for this site).
    Three windows - see module docstring. (There used to also be an "all-time (since 2022)" window - removed
    23 Nov after confirming 2022-01-01 is a genuine data floor, not a pipeline limit: structured player-level
    rugby stats don't exist publicly further back than that in any source checked.)"""
    p, _ = load_graded()
    out = {}
    for window in WINDOWS:
        R_country = rank(p, 'country', window=window)
        R_club = rank(p, 'club', window=window)
        out[window] = {'Country': _format(R_country)}
        for full, label in LEAGUE_LABEL.items():
            out[window][label] = _format(R_club[R_club.league == full])
    return out

if __name__ == '__main__':
    pd.set_option('display.width', 200)
    p, vals = load_graded()
    print(f'rows {len(p)}  club {len(p[p.bucket=="club"])}  country {len(p[p.bucket=="country"])}')
    R_country = rank(p, 'country')
    print(f'\n=== COUNTRY top 5 per position (min {MIN_GAMES} games, last 365 days, T1 nations only) ===')
    for grp, x in R_country.groupby('grp'):
        print(f'\n{grp}:'); print(x.head(5)[['name', 'team', 'n', 'rate', 'grade']].to_string(index=False))

    R_club = rank(p, 'club')
    print(f'\n=== CLUB top 3 per position, Top 14 and Premiership only (min {MIN_GAMES} games) ===')
    for lg in ['French Top 14', 'Gallagher Premiership']:
        print(f'\n--- {lg} ---')
        for grp, x in R_club[R_club.league == lg].groupby('grp'):
            print(f'{grp}:'); print(x.head(3)[['name', 'team', 'n', 'rate', 'grade']].to_string(index=False))
    R_club.to_parquet('rankings_club.parquet'); R_country.to_parquet('rankings_country.parquet')
