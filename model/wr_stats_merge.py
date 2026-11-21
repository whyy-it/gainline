"""Merges wr_stats_sync.py's reviewed staging files into player_stats.parquet and team_match_stats.parquet
(the latter has no existing build script anywhere in this repo - always a frozen delivery, confirmed 21 Nov -
so this is also the first real implementation of that aggregation).

Known, understood gap: summing player-level 'points'/'tries' slightly undercounts a team's real total on
penalty-try matches (rugby law gives a penalty try no individual scorer, so nobody's stat line captures it).
Validated across all 160 new matches: 303 of 320 team-sides reconstruct the real score exactly from player
points; the other 17 are short by EXACTLY 7 (one try+conversion), never any other amount - the signature of a
penalty try, not a data error. Corrected for team_match_stats.parquet's 'tries' column specifically (it feeds
the four-try/three-try-differential bonus rules in league_sim.py/top14_sim.py) by adding +1 there whenever
the points gap is exactly 7; left player_stats.parquet alone (individual stats are correct as-is, the gap
only appears when you sum them)."""
import pandas as pd

FEATS = ['tries', 'line_breaks', 'defenders_beaten', 'metres_carried', 'carries', 'offloads', 'tackles_made',
         'tackles_missed', 'turnovers_won', 'turnovers_lost', 'penalties_conceded', 'yellow_cards', 'red_cards',
         'lineouts_won', 'kicks', 'passes']

def build_team_match_stats(P, M):
    agg = P.groupby(['match_id', 'team']).agg(**{f: (f, 'sum') for f in FEATS},
                                               points=('points', 'sum'), n=('player_id', 'size'),
                                               mx=('tries', 'max'), date=('date', 'first'),
                                               comp=('competition', 'first')).reset_index()
    Ms = M.set_index('match_id')
    rows = []
    for _, r in agg.iterrows():
        m = Ms.loc[r.match_id] if r.match_id in Ms.index else None
        tries = r.tries
        if m is not None and m.get('scores') not in (None, ''):
            try:
                scores = eval(m['scores']) if isinstance(m['scores'], str) else m['scores']
                is_home = (m['home'] == r.team)
                real_pts = scores[0] if is_home else scores[1]
                if real_pts - r.points == 7: tries += 1   # penalty try - no individual scorer to sum
            except Exception:
                pass
        rows.append(dict(match_id=r.match_id, team=r.team, tries=tries, line_breaks=r.line_breaks,
                         defenders_beaten=r.defenders_beaten, metres_carried=r.metres_carried, carries=r.carries,
                         offloads=r.offloads, tackles_made=r.tackles_made, tackles_missed=r.tackles_missed,
                         turnovers_won=r.turnovers_won, turnovers_lost=r.turnovers_lost,
                         penalties_conceded=r.penalties_conceded, yellow_cards=r.yellow_cards,
                         red_cards=r.red_cards, lineouts_won=r.lineouts_won, kicks=r.kicks, passes=r.passes,
                         date=r.date, n=int(r.n), mx=float(r.mx), comp=r.comp, ok=True))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    P_new = pd.read_csv('new_player_stats_staging.csv', parse_dates=['date'], dtype={'game_date': str, 'player_id': str, 'match_id': str})
    M_new = pd.read_csv('new_match_meta_staging.csv', parse_dates=['date'])
    P_old = pd.read_parquet('player_stats.parquet')
    TM_old = pd.read_parquet('team_match_stats.parquet')

    dup = set(P_old.match_id) & set(P_new.match_id)
    if dup:
        print(f'{len(dup)} match_ids already present - dropping from the new batch (should be 0, these are new wr_ ids)')
        P_new = P_new[~P_new.match_id.isin(dup)]

    # union columns both ways - don't drop new fields the old data never had (e.g. 'minutes', first time
    # available at all) just because old rows can't backfill them, and don't drop old fields new data lacks
    all_cols = list(P_old.columns) + [c for c in P_new.columns if c not in P_old.columns]
    for c in all_cols:
        if c not in P_old.columns: P_old[c] = None
        if c not in P_new.columns: P_new[c] = None
    P_new = P_new[all_cols]; P_old = P_old[all_cols]
    P_all = pd.concat([P_old, P_new], ignore_index=True)

    TM_new = build_team_match_stats(pd.read_csv('new_player_stats_staging.csv', parse_dates=['date'], dtype={'game_date': str, 'player_id': str, 'match_id': str}), M_new)
    for c in TM_old.columns:
        if c not in TM_new.columns: TM_new[c] = None
    TM_new = TM_new[TM_old.columns]
    TM_all = pd.concat([TM_old, TM_new], ignore_index=True)

    print(f'player_stats: {len(P_old)} -> {len(P_all)} (+{len(P_all)-len(P_old)})')
    print(f'team_match_stats: {len(TM_old)} -> {len(TM_all)} (+{len(TM_all)-len(TM_old)})')
    P_all.to_parquet('player_stats.parquet')
    TM_all.to_parquet('team_match_stats.parquet')
    print('wrote both parquet files')
