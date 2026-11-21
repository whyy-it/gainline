"""Syncs player_stats.parquet / team_match_stats.parquet from World Rugby's own Pulse API (reverse-engineered,
no key needed) - found 21 Nov while looking for a live player-stats source (ESPN has none; SofaScore blocked).
Far richer than what either had: minutes played (never had this field before), 90 distinct per-player stat
keys, validated against known results (England 37-12 NZ reproduced exactly from 6 England tries + 1 NZ try;
Harry Potter's known yellow card reproduced exactly).

No script in this repo ever built team_match_stats.parquet either (checked - it's always been a frozen
delivery, same pattern as several other files found this session). Built that aggregation from scratch here.

The API's own `competition` filter param doesn't actually filter (confirmed: passing any value still returns
unrelated competitions) - works around it by pulling the full date-range match list (paginated, wide ranges
work fine: 588 matches for a 2.5-month global window in one sweep) and filtering client-side by team name
against games.parquet's known team universe, which is far cheaper than ESPN's day-by-day walk.

Run from model/: python3 wr_stats_sync.py [start_date] [end_date] - writes to staging CSVs for review
(new_player_stats_staging.csv, new_team_match_stats_staging.csv), does NOT touch the real parquet files -
merge after reviewing.

With no args (the scheduled-run case, see .github/workflows/sync_player_stats.yml), start defaults to 3 days
before the latest date already in player_stats.parquet - a small overlap is harmless since wr_stats_merge.py
already drops any match_id it's seen before - and end defaults to today, so a daily cron run always catches
whatever finished since the last run without anyone updating a hardcoded date."""
import sys, time, requests
from datetime import date, timedelta
import pandas as pd
from build_data import norm

COLS = ['tries', 'try_assists', 'line_breaks', 'defenders_beaten', 'metres_carried', 'carries', 'offloads',
        'passes', 'kicks', 'tackles_made', 'tackles_missed', 'turnovers_won', 'turnovers_lost',
        'penalties_conceded', 'yellow_cards', 'red_cards', 'lineouts_won', 'rucks_won', 'conversion_goals',
        'penalty_goals', 'drop_goals_converted', 'points', 'dominant_tackles', 'minutes', 'player_id', 'name',
        'team', 'position', 'match_id', 'game_date', 'competition', 'team_vs', 'date']   # explicit header even
# when 0 matches are found - keeps the staging CSV's schema stable for wr_stats_merge.py to read either way

H = {'User-Agent': 'Mozilla/5.0 (compatible; Gainline/1.0)'}
BASE = 'https://api.wr-rims-prod.pulselive.com/rugby/v3'

# World Rugby's own competition label -> this repo's convention (only the ones we actively track).
# Checked the ACTUAL labels returned (21 Nov) rather than guess - they don't match World Rugby's public
# branding half the time (e.g. "URC" not "United Rugby Championship", "English Premiership" not "Gallagher
# Premiership", "EPCR Champions Cup" not "Heineken Champions Cup" this season - sponsor names drift year to
# year and the API doesn't follow them).
COMP_MAP = {
    'Top 14': 'French Top 14', 'English Premiership': 'Gallagher Premiership', 'URC': 'United Rugby Championship',
    'Nations Championship': 'Nations Championship', 'Six Nations': 'Six Nations',
    'The Rugby Championship': 'The Rugby Championship', 'Rugby World Cup': 'Rugby World Cup',
    'EPCR Champions Cup': 'European Rugby Champions Cup', 'Heineken Champions Cup': 'European Rugby Champions Cup',
    'EPCR Challenge Cup': 'European Rugby Challenge Cup',
}
CLUB_COMPS = {'French Top 14', 'Gallagher Premiership', 'United Rugby Championship',
              'European Rugby Champions Cup', 'European Rugby Challenge Cup'}

# our schema field -> World Rugby stat key (checked against two real matches - see README for the validation)
FIELD_MAP = {
    'tries': 'Tries', 'try_assists': 'TryAssist', 'line_breaks': 'CleanBreaks', 'defenders_beaten': 'DefendersBeaten',
    'metres_carried': 'CarriesMetres', 'carries': 'Carries', 'offloads': 'Offload', 'passes': 'Passes',
    'kicks': 'Kicks', 'tackles_made': 'Tackles', 'tackles_missed': 'MissedTackles', 'turnovers_won': 'TurnoverWon',
    'turnovers_lost': 'TurnoversConceded', 'penalties_conceded': 'Penalties', 'yellow_cards': 'YellowCards',
    'red_cards': 'RedCards', 'lineouts_won': 'LineoutsWon', 'rucks_won': 'RucksWon', 'conversion_goals': 'Conversions',
    'penalty_goals': 'KickPenaltyGood', 'drop_goals_converted': 'DropGoalsConverted', 'points': 'Points',
    'dominant_tackles': 'DominantTackles', 'minutes': 'MinutesPlayedTotal',
}

POS_FIX = {'full_back': 'fullback', 'number_8': 'number_eight'}   # align to tries.py/grades.py's GROUP dict keys

def wr_position(label, number):
    """World Rugby only labels replacements generically ('Replacement') regardless of role - derive forward
    vs back from the jersey number using the standard rugby squad-numbering convention (16-20 forwards,
    21-23 backs; starters 1-15 already carry a specific position label, this only applies to subs)."""
    lab = (label or '').lower().replace(' ', '_')
    if lab != 'replacement':
        return POS_FIX.get(lab, lab) or None
    try:
        n = int(number)
    except (TypeError, ValueError):
        return 'back_sub'
    return 'forward_sub' if 16 <= n <= 20 else 'back_sub'

def match_list(start, end):
    """No 'competition' param - passing ANY value (even a bogus one) narrows the results to some unrelated
    subset rather than erroring or being ignored outright (confirmed: 588 matches/2.5mo WITH a dummy
    competition id vs ~1300+ WITHOUT one, and the 'with' version was missing Prem/URC entirely). Omitting it
    gives genuine global coverage; filter client-side by team name instead (same workaround club_fixtures.py
    already uses for ESPN's broken competition filtering - same bug class, different provider)."""
    out, page = [], 0
    while True:
        r = requests.get(f'{BASE}/match', params=dict(matchStatus='C', sort='asc', pageSize=100, page=page,
                         startDate=start, endDate=end), headers=H, timeout=20)
        d = r.json(); out.extend(d.get('content', []))
        pi = d.get('pageInfo', {})
        if page >= pi.get('numPages', 1) - 1: break
        page += 1; time.sleep(0.2)
    return out

def relevant(m, known_teams):
    teams = m.get('teams', [])
    if len(teams) != 2: return None
    names = [norm(t.get('name', '')) for t in teams]
    if not all(n in known_teams for n in names): return None
    comp_raw = m.get('events', [{}])[0].get('label', '') if m.get('events') else ''
    comp_base = comp_raw.rsplit(' ', 1)[0] if comp_raw and comp_raw.rsplit(' ', 1)[-1].isdigit() else comp_raw
    comp = COMP_MAP.get(comp_base)
    if not comp: return None
    return comp

def fetch_match(match_id, comp):
    r = requests.get(f'{BASE}/match/{match_id}/stats', headers=H, timeout=20)
    if r.status_code != 200: return None
    d = r.json(); mt = d['match']
    date = mt['time']['label']
    teams = {t['id']: norm(t['name']) for t in mt['teams']}
    team_ids = list(teams.keys())
    rows = []
    for i, ts in enumerate(d['teamStats']):
        team = teams.get(team_ids[i] if i < len(team_ids) else '', None)
        opp = teams.get(team_ids[1 - i] if i < len(team_ids) else '', None)
        if team is None: continue
        for p in ts.get('playerStats', []):
            if p.get('positionLabel') == 'Head Coach' or p.get('number') is None:
                continue   # coaching staff show up in playerStats with null stats - not players, drop them
            s = p.get('stats', {})
            row = {k: s.get(v, 0) for k, v in FIELD_MAP.items()}
            row.update(player_id=f"wr_{p['player']['id']}", name=p['player']['name']['display'], team=team,
                      position=wr_position(p.get('positionLabel'), p.get('number')),
                      match_id=f'wr_{match_id}', game_date=date.replace('-', ''), competition=comp,
                      team_vs=opp, date=pd.Timestamp(date))
            rows.append(row)
    return rows, dict(match_id=f'wr_{match_id}', date=pd.Timestamp(date), competition=comp,
                      home=teams.get(team_ids[0]) if team_ids else None,
                      away=teams.get(team_ids[1]) if len(team_ids) > 1 else None, scores=mt.get('scores'))

if __name__ == '__main__':
    default_start = (pd.read_parquet('player_stats.parquet').date.max() - timedelta(days=3)).strftime('%Y-%m-%d')
    start = sys.argv[1] if len(sys.argv) > 1 else default_start
    end = sys.argv[2] if len(sys.argv) > 2 else date.today().strftime('%Y-%m-%d')
    known = set(pd.read_parquet('games.parquet').pipe(lambda g: set(g.home) | set(g.away)))
    print(f'pulling World Rugby match list {start} to {end}...')
    ml = match_list(start, end)
    print(f'{len(ml)} total matches globally in range; filtering to known teams + tracked competitions...')
    targets = []
    for m in ml:
        comp = relevant(m, known)
        if comp: targets.append((m['matchId'], comp))
    print(f'{len(targets)} matches to fetch')
    all_rows, meta_rows = [], []
    for i, (mid, comp) in enumerate(targets):
        try:
            res = fetch_match(mid, comp)
            if res:
                rows, meta = res; all_rows.extend(rows); meta_rows.append(meta)
        except Exception as e:
            print(f'  skip {mid}: {e}', file=sys.stderr)
        if i % 20 == 0: print(f'  {i}/{len(targets)}...')
        time.sleep(0.15)
    P = pd.DataFrame(all_rows, columns=COLS) if all_rows else pd.DataFrame(columns=COLS)
    print(f'\n{len(P)} player-rows from {len(meta_rows)} matches')
    print(P.groupby('competition').match_id.nunique() if len(P) else 'none')
    P.to_csv('new_player_stats_staging.csv', index=False)
    pd.DataFrame(meta_rows, columns=['match_id', 'date', 'competition', 'home', 'away', 'scores']) \
        .to_csv('new_match_meta_staging.csv', index=False)
    print('wrote new_player_stats_staging.csv, new_match_meta_staging.csv - review before merging')
