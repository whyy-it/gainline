"""Pulls completed ('STATUS_FINAL') results from ESPN's unofficial API for the competitions build_data.py
hand-maintains (same ESPN league ids verified working in club_fixtures.py - see that file's docstring for the
Oyonnax/Vannes data-quality check already done on these ids), from each competition's last-known date in
build_data.py through today. Prints formatted lines ready to paste into build_data.py's add_rows() - does NOT
edit build_data.py itself, since that file is meant to be reviewed before trusting new results into the
dataset every model is built on."""
import sys, requests
from datetime import datetime, timedelta
sys.path.insert(0, '.')
from build_data import norm

LEAGUES = {
    'Nations Championship': ('17567', '2026-11-16', 'intl'),
    'French Top 14': ('270559', '2026-11-09', 'club'),
    'Gallagher Premiership': ('267979', '2026-11-01', 'club'),
    'United Rugby Championship': ('270557', '2026-11-01', 'club'),
}

def pull(espn_id, start, end):
    out = []
    d = start
    while d <= end:
        try:
            r = requests.get(f'https://site.api.espn.com/apis/site/v2/sports/rugby/{espn_id}/scoreboard',
                             params=dict(dates=d.strftime('%Y%m%d')), timeout=15)
            for e in r.json().get('events', []):
                comp = e.get('competitions', [{}])[0]
                if comp.get('status', e.get('status', {})).get('type', {}).get('name') != 'STATUS_FINAL':
                    continue
                teams = comp.get('competitors', [])
                if len(teams) != 2: continue
                home = next((t for t in teams if t.get('homeAway') == 'home'), None)
                away = next((t for t in teams if t.get('homeAway') == 'away'), None)
                if not home or not away: continue
                out.append(dict(date=d.strftime('%Y-%m-%d'), home=norm(home['team']['displayName']),
                                away=norm(away['team']['displayName']), hs=int(home['score']), as_=int(away['score']),
                                neutral=bool(comp.get('neutralSite', False)),
                                city=comp.get('venue', {}).get('address', {}).get('city')))
        except Exception as ex:
            print(f'  skip {d.date()}: {ex}', file=sys.stderr)
        d += timedelta(days=1)
    return out

if __name__ == '__main__':
    today = datetime(2026, 11, 21)
    for comp, (espn_id, start_str, kind) in LEAGUES.items():
        start = datetime.strptime(start_str, '%Y-%m-%d')
        rows = pull(espn_id, start, today)
        print(f'\n=== {comp} ({len(rows)} new results, {start_str} to {today.date()}) ===')
        if kind == 'intl':
            for r in rows:
                print(f"        ('{r['date']}','{r['home']}',{r['hs']},{r['as_']},'{r['away']}',{int(r['neutral'])}),  # {r['city']}")
        else:
            for r in rows:
                print(f"{r['date']}|{r['home']}|{r['hs']}|{r['as_']}|{r['away']}")
