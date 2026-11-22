"""Auto-detects each tracked competition's current/upcoming round directly from ESPN - replaces hand-
maintaining round_config.py's FIX list every week (see README, 22 Nov: "we need this to update automatically").
Queries every competition in ESPN_LEAGUE_IDS (round_config.py - shared with the live-score poller), walking
day by day (same approach as club_fixtures.py's pull() - confirmed 22 Nov that ESPN's rugby scoreboard
endpoint only accepts a single `dates=YYYYMMDD`; a `START-END` range looks like it works (returns valid JSON,
0 events) but is actually a silently-swallowed 400 error - caught this by checking the status code directly,
not just trusting an empty events list), and groups whatever comes back into one "round" per competition -
the nearest cluster of fixtures within 3 days of the earliest not-yet-final match (covers a normal Fri-Sun or
Sat-only window for every competition tracked here).

"Round" is a cluster of dates, not a real official round NUMBER - ESPN doesn't expose one reliably enough
to trust, so round_label is the date range (e.g. "28-29 Nov") rather than "Round 11".

Also carries each match's final score when ESPN confirms it (23 Nov: a round used to only "go final" in the
visitor's own browser, via the client-side live poller - meaning a match that finished more than 24h ago, with
no one's tab open to catch it, would silently show the stale PRE-match projection again on the next fresh
visit, since the poller had given up trying and the server-side rebuild never recorded the real result at
all). produce.py now bakes a confirmed final result straight into site.json, so "final" is a property of the
published data itself, not something that depends on a visitor's browser having polled at the right moment.

Reuses club_fixtures.py's ESPN name aliasing (NOT build_data.py's - that one's for World Rugby's different
naming convention, a different provider with its own quirks)."""
from datetime import date, datetime, timedelta
import requests
from club_fixtures import norm
from round_config import ESPN_LEAGUE_IDS

KIND = {'Nations Championship': 'intl', 'Six Nations': 'intl', 'French Top 14': 'club',
        'Gallagher Premiership': 'club', 'United Rugby Championship': 'club'}

def _parse_dt(iso):
    return datetime.fromisoformat(iso.replace('Z', '+00:00'))

def _fetch_day(league_id, d):
    r = requests.get(f'https://site.api.espn.com/apis/site/v2/sports/rugby/{league_id}/scoreboard',
                     params=dict(dates=d.strftime('%Y%m%d')), timeout=15)
    if r.status_code != 200: return []   # a bad day shouldn't kill the whole pull - just treat it as empty
    return r.json().get('events', [])

def competition_round(comp, days_back=6, days_fwd=10):
    """One competition's current cluster of fixtures, or None if nothing's on. days_back needs to comfortably
    bridge from one round finishing to the next one being detected (bumped 3->6 on 23 Nov - 3 days left a
    real gap where a just-finished round could age out of the window before the next round's fixtures were
    far enough out to be picked up, showing "nothing scheduled" in between); days_fwd is the lookahead window."""
    today = date.today()
    seen = {}   # keyed by event id - a game can show up on more than one day's query near a UTC date boundary
    for i in range(-days_back, days_fwd + 1):
        d = today + timedelta(days=i)
        for ev in _fetch_day(ESPN_LEAGUE_IDS[comp], d):
            c = ev.get('competitions', [{}])[0]
            teams = c.get('competitors', [])
            if len(teams) != 2: continue
            home = next((t for t in teams if t.get('homeAway') == 'home'), teams[0])
            away = next((t for t in teams if t.get('homeAway') == 'away'), teams[1])
            hn, an = home.get('team', {}).get('displayName'), away.get('team', {}).get('displayName')
            st = c.get('status', {}).get('type', {}).get('name')
            if not hn or not an or 'TBD' in (hn, an) or st in ('STATUS_POSTPONED', 'STATUS_CANCELED'): continue
            # venue deliberately left blank, not read from ESPN: checked 22 Nov and it's wrong here -
            # "Allianz Stadium, London" was returned for every single Nations Championship fixture that week,
            # including ones obviously not played there (Italy v Fiji). Showing a confidently wrong venue is
            # worse than showing none; revisit with a cross-check (e.g. World Rugby's /match venue field,
            # already proven correct in wr_stats_sync.py) if this is wanted later.
            row = dict(home=norm(hn), away=norm(an), venue='', ko_utc=c.get('date'), status=st)
            if st == 'STATUS_FINAL':
                hs = next((t.get('score') for t in teams if t.get('homeAway') == 'home'), None)
                as_ = next((t.get('score') for t in teams if t.get('homeAway') == 'away'), None)
                if hs is not None and as_ is not None:
                    row['final_home'] = float(hs); row['final_away'] = float(as_)
            seen[ev['id']] = row
    rows = list(seen.values())
    if not rows: return None
    rows.sort(key=lambda r: r['ko_utc'])
    anchor = next((r['ko_utc'] for r in rows if r['status'] != 'STATUS_FINAL'), rows[-1]['ko_utc'])
    adt = _parse_dt(anchor)
    cluster = [r for r in rows if abs((_parse_dt(r['ko_utc']) - adt).total_seconds()) <= 3 * 86400]
    cluster.sort(key=lambda r: r['ko_utc'])
    dates = sorted({_parse_dt(r['ko_utc']).strftime('%-d %b') for r in cluster})
    label = dates[0] if len(dates) == 1 else f'{dates[0]} - {dates[-1]}'
    return dict(competition=comp, kind=KIND[comp], round_label=label,
               matches=[{k: v for k, v in r.items() if k != 'status'} for r in cluster])

def active_rounds():
    out = []
    for comp in ESPN_LEAGUE_IDS:
        try:
            r = competition_round(comp)
        except Exception as e:
            print(f'  auto_rounds: skip {comp}: {e}')
            continue
        if r: out.append(r)
    return out

if __name__ == '__main__':
    for r in active_rounds():
        print(f"{r['competition']} ({r['round_label']}): {len(r['matches'])} matches")
        for m in r['matches']:
            print(f"   {m['home']} v {m['away']} - {m['venue']} - {m['ko_utc']}")
