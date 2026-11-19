"""Super Rugby Pacific model. Same approach as Tests/clubs: margin-Elo, probit win prob, walk-forward.
Split: tune on 2025 season, test on 2026 (both unseen when scored)."""
import numpy as np, pandas as pd, warnings
from scipy.optimize import minimize
from scipy.stats import norm
warnings.filterwarnings('ignore')

NORM = {'Melbourne Rebels': 'Rebels', 'New South Wales Waratahs': 'Waratahs', 'Queensland Reds': 'Reds', 'Western Force': 'Force'}
HOME_CITIES = {'Blues': {'Auckland', 'Albany', 'Whangarei'}, 'Chiefs': {'Hamilton', 'Pukekohe', 'New Plymouth', 'Mount Maunganui', 'Rotorua', 'Tauranga'},
    'Crusaders': {'Christchurch', 'Nelson', 'Timaru'}, 'Highlanders': {'Dunedin', 'Invercargill', 'Queenstown'},
    'Hurricanes': {'Wellington', 'Palmerston North', 'Napier'}, 'Moana Pasifika': {'Auckland', 'Albany', 'Apia', 'Pukekohe'},
    'Fijian Drua': {'Lautoka', 'Suva', 'Ba'}, 'Brumbies': {'Canberra'}, 'Waratahs': {'Sydney'},
    'Reds': {'Brisbane', 'Townsville', 'Gold Coast', 'Bokarina'}, 'Rebels': {'Melbourne'}, 'Force': {'Perth'}}
NZ = {'Blues', 'Chiefs', 'Crusaders', 'Highlanders', 'Hurricanes', 'Moana Pasifika'}
AU = {'Brumbies', 'Waratahs', 'Reds', 'Rebels', 'Force'}
def country(t): return 'NZ' if t in NZ else 'AU' if t in AU else 'FJ'

def games():
    m = pd.read_parquet('../matches.parquet'); m = m[m.competition == 'Super Rugby Pacific'].copy()
    m = m[m.completed.astype(bool) & m.home_score.notna()]
    m['date'] = pd.to_datetime(m.date)
    m['home'] = m.home_team.replace(NORM); m['away'] = m.away_team.replace(NORM)
    m['hs'] = m.home_score.astype(float); m['as_'] = m.away_score.astype(float)
    m['neutral'] = [c not in HOME_CITIES.get(h, set()) for h, c in zip(m.home, m.city_played)]
    m['travel'] = [int(country(h) != country(a)) for h, a in zip(m.home, m.away)]       # away side crossed a border
    m['drua_home'] = ((m.home == 'Fijian Drua') & ~m.neutral).astype(int)
    m['final'] = (m.date.dt.month >= 6).astype(int)
    return m.sort_values('date').reset_index(drop=True)[['date', 'home', 'away', 'hs', 'as_', 'neutral', 'travel', 'drua_home', 'final', 'city_played', 'match_id']]

def elo(df, K, hfa, cap, regress, travel_bonus=0.0, drua_bonus=0.0, init=None):
    R = dict(init or {}); preds = np.zeros(len(df)); last = None
    for j, r in enumerate(df.itertuples()):
        if last is not None and r.date.year != last:
            mu = np.mean(list(R.values())); R = {t: mu + (v - mu) * (1 - regress) for t, v in R.items()}
        last = r.date.year
        H = 0.0 if r.neutral else hfa + travel_bonus * r.travel + drua_bonus * r.drua_home
        p = R.get(r.home, 0) - R.get(r.away, 0) + H; preds[j] = p
        e = np.clip(r.hs - r.as_, -cap, cap) - p
        R[r.home] = R.get(r.home, 0) + K * e; R[r.away] = R.get(r.away, 0) - K * e
    return preds, R

def probit_fit(x, y):
    f = lambda w: -np.mean(y * np.log(np.clip(norm.cdf(w[0] + w[1] * x), 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - norm.cdf(w[0] + w[1] * x), 1e-6, 1)))
    return minimize(f, [0, 0.06], method='Nelder-Mead').x

def score(df, pred, extra=None):
    y = (df.hs > df.as_).astype(float).values + 0.5 * (df.hs == df.as_).values; yr = df.date.dt.year.values
    out = {}
    for name, yrs in (('val', [2025]), ('test', [2026])):
        sel = np.isin(yr, yrs); fit = (yr < min(yrs)) & (yr >= 2023)    # burn-in 2022
        if extra is None:
            w = probit_fit(pred[fit], y[fit]); p = norm.cdf(w[0] + w[1] * pred[sel])
        else:
            from sklearn.linear_model import LogisticRegression
            X = np.c_[pred, extra]; p = LogisticRegression(max_iter=2000).fit(X[fit], y[fit] > .5).predict_proba(X[sel])[:, 1]
        out[name] = round(float(np.mean((p - y[sel]) ** 2)), 4)
        out[name + '_acc'] = round(float(np.mean((p > .5) == (y[sel] > .5))), 3)
        out[name + '_mae'] = round(float(np.abs((df.hs - df.as_).values[sel] - pred[sel]).mean()), 1)
    return out
