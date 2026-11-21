"""Pulls the full remaining-season fixture list from ESPN's unofficial rugby API (free, no auth). One call
only returns the NEXT scheduled round, so this walks forward day by day (same pattern as odds_pull.py's
weekly walk) until the season ends. Reusable across leagues - only the ESPN_LEAGUE_IDS values differ.

CAUTION found 21 Nov: the vendored data/gm/utils/config.py has URC and Top14 ESPN ids SWAPPED (verified by
actually fetching each id's team list, not by trusting the label). Also found the /teams endpoint lists a
stale/wrong roster for Top14 (shows Oyonnax, who was relegated in 2024-25; real 2026-27 Top14 has Vannes,
confirmed via Wikipedia) - but the /scoreboard endpoint used here is NOT affected (17 real Vannes games, 0
Oyonnax games in the pull), so this is reliable for fixtures even though /teams isn't. Verify a new league's
pull the same way (spot-check team names) before trusting it blindly."""
import requests, time
from datetime import datetime, timedelta
import pandas as pd

ESPN_LEAGUE_IDS = {'urc': '270557', 'top14': '270559', 'premiership': '267979', 'super_rugby': '242041'}

# ESPN display name -> this repo's team name convention (only entries that actually differ)
NAME_MAP = {
    'Bordeaux Begles': 'Bordeaux-Begles', 'Castres Olympique': 'Castres', 'Montpellier Herault': 'Montpellier',
    'Stade Francais Paris': 'Stade Francais', 'Stade Toulousain': 'Toulouse',
    'Bristol Rugby': 'Bristol Bears', 'Newcastle Falcons': 'Newcastle',
    'Bath Rugby': 'Bath', 'Exeter Chiefs': 'Exeter', 'Gloucester Rugby': 'Gloucester',
    'Northampton Saints': 'Northampton',
    'Benetton Treviso': 'Benetton', 'Cardiff Blues': 'Cardiff',
}
def norm(name): return NAME_MAP.get(name, name)

def pull(league, start, end):
    """league: key into ESPN_LEAGUE_IDS. start/end: datetime. Returns a DataFrame of scheduled games found
    by walking every day in [start, end) - slow (one call/day) but free and no rate-limit issues seen."""
    espn_id = ESPN_LEAGUE_IDS[league]
    seen = {}
    d = start
    while d < end:
        try:
            r = requests.get(f'https://site.api.espn.com/apis/site/v2/sports/rugby/{espn_id}/scoreboard',
                             params=dict(dates=d.strftime('%Y%m%d')), timeout=15)
            for e in r.json().get('events', []):
                comp = e.get('competitions', [{}])[0]
                teams = comp.get('competitors', [])
                if len(teams) != 2: continue
                home = next((t for t in teams if t.get('homeAway') == 'home'), teams[0])
                away = next((t for t in teams if t.get('homeAway') == 'away'), teams[1])
                hn, an = home.get('team', {}).get('displayName'), away.get('team', {}).get('displayName')
                if not hn or not an or 'TBD' in (hn, an): continue   # placeholder playoff slots, not yet seeded
                seen[e['id']] = dict(date=e.get('date', '')[:10], home=norm(hn), away=norm(an),
                                     status=e.get('status', {}).get('type', {}).get('name'),
                                     notes=comp.get('notes', [{}])[0].get('headline', '') if comp.get('notes') else '')
        except Exception as ex:
            print('skip', d.date(), ex)
        d += timedelta(days=1)
    return pd.DataFrame(seen.values()).sort_values('date').reset_index(drop=True) if seen else pd.DataFrame()

if __name__ == '__main__':
    import sys
    league = sys.argv[1] if len(sys.argv) > 1 else 'top14'
    F = pull(league, datetime(2026, 11, 21), datetime(2027, 6, 15))
    print(f'{league}: {len(F)} fixtures found')
    print(F.to_string())
    F.to_csv(f'fixtures_{league}.csv', index=False)
