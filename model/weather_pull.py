"""Pull match-day weather for every game in games.parquet from Open-Meteo (free, no API key).
Run on your own machine:   python3 weather_pull.py        -> writes weather.csv (cached; safe to re-run)
Then:                      python3 weather_test.py        -> tests whether weather improves the totals model
Needs: pandas, pyarrow, requests  (pip install pandas pyarrow requests)"""
import json, os, time
import pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = 'https://geocoding-api.open-meteo.com/v1/search'
ARCHIVE = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST = 'https://api.open-meteo.com/v1/forecast'

# Home city per club, used when a game's city is missing
HOME_CITY = {'Bath': 'Bath', 'Bayonne': 'Bayonne', 'Benetton': 'Treviso', 'Biarritz Olympique': 'Biarritz', 'Bordeaux-Begles': 'Bordeaux',
    'Bristol Bears': 'Bristol', 'Brive': 'Brive-la-Gaillarde', 'Bulls': 'Pretoria', 'Cardiff': 'Cardiff', 'Castres': 'Castres',
    'Cheetahs': 'Bloemfontein', 'Clermont Auvergne': 'Clermont-Ferrand', 'Connacht': 'Galway', 'Dragons': 'Newport', 'Edinburgh': 'Edinburgh',
    'Exeter': 'Exeter', 'Glasgow Warriors': 'Glasgow', 'Gloucester': 'Gloucester', 'Harlequins': 'Twickenham', 'La Rochelle': 'La Rochelle',
    'Leicester Tigers': 'Leicester', 'Leinster': 'Dublin', 'Lions': 'Johannesburg', 'London Irish': 'Brentford', 'Lyon': 'Lyon',
    'Montpellier': 'Montpellier', 'Munster': 'Limerick', 'Newcastle': 'Newcastle upon Tyne', 'Northampton': 'Northampton', 'Ospreys': 'Swansea',
    'Oyonnax': 'Oyonnax', 'Pau': 'Pau', 'Perpignan': 'Perpignan', 'Racing 92': 'Nanterre', 'Sale Sharks': 'Salford', 'Saracens': 'Barnet',
    'Scarlets': 'Llanelli', 'Sharks': 'Durban', 'Stade Francais': 'Paris', 'Stormers': 'Cape Town', 'Toulon': 'Toulon', 'Toulouse': 'Toulouse',
    'US Montauban': 'Montauban', 'Ulster': 'Belfast', 'Vannes': 'Vannes', 'Wasps': 'Coventry', 'Worcester Warriors': 'Worcester', 'Zebre': 'Parma',
    'England': 'Twickenham', 'France': 'Saint-Denis', 'Ireland': 'Dublin', 'Italy': 'Rome', 'Scotland': 'Edinburgh', 'Wales': 'Cardiff',
    'New Zealand': 'Auckland', 'Australia': 'Sydney', 'South Africa': 'Johannesburg', 'Argentina': 'Buenos Aires', 'Japan': 'Tokyo', 'Fiji': 'Suva'}
# Closed or closable roofs: weather doesn't reach the pitch
ROOF = {('Racing 92', 'Nanterre'), ('Wales', 'Cardiff'), ('New Zealand', 'Dunedin')}

def geocode(city, cache):
    if city in cache: return cache[city]
    r = requests.get(GEO, params=dict(name=city, count=1), timeout=20).json()
    res = (r.get('results') or [None])[0]
    cache[city] = (res['latitude'], res['longitude']) if res else None
    time.sleep(0.2); return cache[city]

def daily(lat, lon, start, end, forecast=False):
    url = FORECAST if forecast else ARCHIVE
    p = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, timezone='auto',
             daily='precipitation_sum,wind_speed_10m_max,temperature_2m_max')
    d = requests.get(url, params=p, timeout=60).json()['daily']
    return pd.DataFrame(dict(date=pd.to_datetime(d['time']), precip_mm=d['precipitation_sum'],
                             wind_kmh=d['wind_speed_10m_max'], temp_c=d['temperature_2m_max']))

def main():
    g = pd.read_parquet(os.path.join(HERE, 'games.parquet'))
    g = g[g.date >= '2018-01-01'].copy()
    g['wcity'] = [c if isinstance(c, str) and c else HOME_CITY.get(h) for c, h in zip(g.city, g.home)]
    g = g[g.wcity.notna()]
    cpath = os.path.join(HERE, 'geocache.json')
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    out = []
    for city, grp in g.groupby('wcity'):
        ll = geocode(city, cache)
        if not ll: print('no geocode:', city); continue
        try:
            w = daily(ll[0], ll[1], grp.date.min().strftime('%Y-%m-%d'), grp.date.max().strftime('%Y-%m-%d'))
        except Exception as e:
            print('weather failed:', city, e); continue
        m = grp[['date', 'home', 'away', 'wcity']].merge(w, on='date', how='left'); out.append(m)
        print(f'{city:28s} {len(m):4d} games'); time.sleep(0.3)
    json.dump(cache, open(cpath, 'w'))
    W = pd.concat(out, ignore_index=True)
    W['roof'] = [int((h, c) in ROOF) for h, c in zip(W.home, W.wcity)]
    W.to_csv(os.path.join(HERE, 'weather.csv'), index=False)
    print('wrote weather.csv', len(W), 'rows')

if __name__ == '__main__':
    main()
