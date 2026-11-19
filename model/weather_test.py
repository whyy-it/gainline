"""Does match-day weather improve the totals (over/under) model?
Same split as the rest of the project: fit on the 3 seasons before, score on 2024 (validation) and 2025-26 (test).
Prints average miss on total points with and without weather. Run after weather_pull.py."""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
import features as F
from v6 import intl_games, T1, season

def run():
    W = pd.read_csv(os.path.join(HERE, 'weather.csv'), parse_dates=['date'])
    W.loc[W.roof == 1, ['precip_mm', 'wind_kmh']] = 0
    W['wet'] = (W.precip_mm >= 2).astype(int)
    key = W.set_index(['date', 'home', 'away'])
    rows = []
    for kind in ('club', 'intl'):
        if kind == 'intl':
            S, *_ = F.build_state(intl_games(True), 'intl', 0.08, lambda r: 4, 60, 0.0, Ka=0.08)
            S = S[S.home.isin(T1) & S.away.isin(T1)]
        else:
            S, *_ = F.build_state(F.G[F.G.kind == 'club'], 'club', 0.06, lambda r: F.CHFA[F.KEY[r.competition]], 40, 0.2, Ka=0.05)
        S = S.join(key[['precip_mm', 'wind_kmh', 'temp_c', 'wet']], on=['date', 'home', 'away'])
        S = S.dropna(subset=['precip_mm', 'wind_kmh', 'temp_c'])
        S['s'] = [season(d, kind) for d in S.date]
        for split, seas in (('val', [2024]), ('test', [2025, 2026])):
            tr = S[(S.s < min(seas)) & (S.s >= min(seas) - 3)]; te = S[S.s.isin(seas)]
            if len(te) == 0 or len(tr) < 50: continue
            for name, cols in (('no weather', ['tot']), ('+ rain', ['tot', 'precip_mm', 'wet']),
                               ('+ rain, wind, temp', ['tot', 'precip_mm', 'wet', 'wind_kmh', 'temp_c'])):
                X = np.c_[np.ones(len(tr)), tr[cols].values]; b = np.linalg.lstsq(X, tr.total, rcond=None)[0]
                pred = np.c_[np.ones(len(te)), te[cols].values] @ b
                rows.append(dict(kind=kind, split=split, model=name, mae=np.abs(te.total - pred).mean(), n=len(te),
                                 rain_coef=b[2] if len(b) > 2 else np.nan))
    R = pd.DataFrame(rows)
    print(R.pivot_table(index=['kind', 'model'], columns='split', values='mae').round(2))
    print('\nPoints per mm of rain (fitted):'); print(R.dropna().groupby(['kind', 'split']).rain_coef.mean().round(3))
    print('\nKeep weather only if it lowers the miss on BOTH val and test.')

if __name__ == '__main__':
    run()
