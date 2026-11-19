"""Pull historical CLOSING odds for rugby union from The Odds API (paid plans have historical data).
Run on your own machine:
    pip install requests pandas
    export ODDS_API_KEY=your_key
    python3 odds_pull.py --estimate          # lists rugby keys + estimated credit cost, spends almost nothing
    python3 odds_pull.py --budget 15000      # pulls, stops before spending more than this many credits
Writes odds_hist.csv (one row per match per bookmaker, decimal odds) and a cache folder so re-runs don't re-spend.

How it works: for each rugby union competition, list the historical events, then take ONE odds snapshot per
kickoff slot (5 minutes before the earliest kickoff in that slot). One snapshot returns every match in the slot,
which is far cheaper than per-event calls. Historical odds cost 10 credits per region per market per call."""
import argparse, json, os, time
from datetime import datetime, timedelta, timezone
import pandas as pd, requests

BASE = 'https://api.the-odds-api.com/v4'
try:
    from dotenv import load_dotenv; load_dotenv('../.env')
except ImportError: pass
KEY = os.environ.get('ODDS_API_KEY', '')
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'odds_cache'); os.makedirs(CACHE, exist_ok=True)
MARKETS = 'h2h,spreads,totals'
REGIONS = 'uk,eu,us'          # uk/eu have the deepest rugby books; us gives PA-legal books where listed
START = '2020-06-06T00:00:00Z'  # API's historical start; rugby coverage begins later and is reported per key

def get(path, **params):
    params['apiKey'] = KEY
    f = os.path.join(CACHE, (path + json.dumps(params, sort_keys=True)).replace('/', '_').replace(' ', '')[:200] + '.json')
    if os.path.exists(f): return json.load(open(f)), 0
    r = requests.get(BASE + path, params=params, timeout=60)
    if r.status_code != 200: raise RuntimeError(f'{r.status_code} {r.text[:200]}')
    used = int(r.headers.get('x-requests-last', 0) or 0)
    data = r.json(); json.dump(data, open(f, 'w')); time.sleep(0.25)
    return data, used

def rugby_keys():
    sports, _ = get('/sports', all='true')
    return [s['key'] for s in sports if s['key'].startswith('rugbyunion')]

def slots_for(key):
    """Walk weekly through history, collecting events + commence times (events endpoint costs 1 credit)."""
    events, spent, t = {}, 0, datetime.fromisoformat(START.replace('Z', '+00:00'))
    end = datetime.now(timezone.utc)
    while t < end:
        try:
            d, used = get(f'/historical/sports/{key}/events', date=t.strftime('%Y-%m-%dT%H:%M:%SZ'))
        except RuntimeError as e:
            if '422' in str(e) or '404' in str(e): t += timedelta(days=7); continue
            raise
        spent += used
        for ev in d.get('data', []): events[ev['id']] = ev
        t += timedelta(days=7)
    # group kickoffs into 3-hour slots
    ks = sorted({ev['commence_time'] for ev in events.values()})
    slots, cur = [], None
    for k in ks:
        kt = datetime.fromisoformat(k.replace('Z', '+00:00'))
        if cur is None or (kt - cur) > timedelta(hours=3):
            slots.append(kt); cur = kt
    return events, slots, spent

def parse(snapshot, key):
    rows = []
    for ev in snapshot.get('data', []):
        for bk in ev.get('bookmakers', []):
            r = dict(sport_key=key, event_id=ev['id'], commence_time=ev['commence_time'], home=ev['home_team'], away=ev['away_team'],
                     book=bk['key'], snapshot=snapshot.get('timestamp'))
            for mk in bk.get('markets', []):
                oc = {o['name']: o for o in mk['outcomes']}
                if mk['key'] == 'h2h':
                    r['ml_home'] = oc.get(ev['home_team'], {}).get('price'); r['ml_away'] = oc.get(ev['away_team'], {}).get('price')
                    r['ml_draw'] = oc.get('Draw', {}).get('price')
                elif mk['key'] == 'spreads':
                    h = oc.get(ev['home_team'], {}); a = oc.get(ev['away_team'], {})
                    r['spread_home'] = h.get('point'); r['spread_home_odds'] = h.get('price'); r['spread_away_odds'] = a.get('price')
                elif mk['key'] == 'totals':
                    r['total'] = oc.get('Over', {}).get('point'); r['over_odds'] = oc.get('Over', {}).get('price'); r['under_odds'] = oc.get('Under', {}).get('price')
            rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--estimate', action='store_true'); ap.add_argument('--budget', type=int, default=10000)
    a = ap.parse_args()
    if not KEY: raise SystemExit('Set ODDS_API_KEY first: export ODDS_API_KEY=...')
    keys = rugby_keys(); print('Rugby union keys:', keys)
    plan, spent = {}, 0
    for k in keys:
        ev, slots, s = slots_for(k); spent += s; plan[k] = (ev, slots)
        first = min((e['commence_time'] for e in ev.values()), default='-')
        print(f'{k:40s} events {len(ev):5d}  kickoff slots {len(slots):4d}  from {first[:10]}')
    cost_each = 10 * len(MARKETS.split(',')) * len(REGIONS.split(','))
    total = sum(len(s) for _, s in plan.values()) * cost_each
    print(f'\nEvent listing spent ~{spent} credits. Closing-odds pull needs ~{total} credits ({cost_each} per snapshot).')
    if a.estimate: return
    rows, used_total = [], spent
    for k, (ev, slots) in plan.items():
        for kt in slots:
            if used_total + cost_each > a.budget: print('Budget reached; re-run with a higher --budget to continue (cached calls are free).'); break
            snap_t = (kt - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
            try:
                d, used = get(f'/historical/sports/{k}/odds', date=snap_t, regions=REGIONS, markets=MARKETS, oddsFormat='decimal')
            except RuntimeError as e:
                print('skip', k, snap_t, e); continue
            used_total += used
            # keep only events that kick off within this slot (snapshot also lists later fixtures)
            for r in parse(d, k):
                ct = datetime.fromisoformat(r['commence_time'].replace('Z', '+00:00'))
                if timedelta(0) <= ct - kt <= timedelta(hours=3): rows.append(r)
        print(f'{k}: {len(rows)} rows so far, ~{used_total} credits used')
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'odds_hist.csv'), index=False)
    print('wrote odds_hist.csv', len(rows), 'rows')

if __name__ == '__main__':
    main()
