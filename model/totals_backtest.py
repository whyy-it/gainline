"""Totals model vs the real closing total line (Six Nations/Tests, The Odds API) - plan step 4, the one
untested market in odds_hist.csv. Walk-forward: fit total ~ tot (attack/defence state) on seasons <=2024 only,
predict on 2025+ games. Never touch 2025+ while fitting. Market total = median closing line across books;
market no-vig over prob from over/under odds, for a Brier-style comparison alongside the points-MAE one."""
import numpy as np, pandas as pd, re, unicodedata
import features as F
from v6 import intl_games, T1

HFA_I = 4.0

ALIAS = {}  # Test nations don't need club-style aliasing; names already match between odds_hist and intl_games
def key(t):
    t = unicodedata.normalize('NFKD', str(t)).encode('ascii', 'ignore').decode().lower().strip()
    return re.sub(r'\s+', ' ', t)

# ---- build the state history (same recipe as produce.py's tests(), full history so 'tot' has no leakage per row)
I = intl_games(True)
S, *_ = F.build_state(I, 'intl', 0.08, lambda r: HFA_I, 60, 0.0, Ka=0.08)
S = S[S.home.isin(T1) & S.away.isin(T1)].copy()
S['season'] = S.date.dt.year.where(S.date.dt.month >= 8, S.date.dt.year - 1)
S['hk'] = S.home.map(key); S['ak'] = S.away.map(key)
S['date_s'] = S.date.dt.strftime('%Y-%m-%d')

# ---- walk-forward linear map tot -> total, fit on seasons < test season only
def predict_total(test_seasons, fit_min_season=2015):
    out = np.full(len(S), np.nan)
    for s in test_seasons:
        fit = (S.season < s) & (S.season >= fit_min_season)
        if fit.sum() < 50: continue
        b = np.polyfit(S.tot[fit], S.total[fit], 1)
        sel = S.season == s
        out[sel] = np.polyval(b, S.tot[sel])
    return out

TEST_SEASONS = [2025, 2026]
S['model_total'] = predict_total(TEST_SEASONS)

# ---- load market totals (median closing line + no-vig over prob)
O = pd.read_csv('odds_hist.csv')
O = O.dropna(subset=['total'])
O['date'] = pd.to_datetime(O.commence_time).dt.tz_convert(None).dt.strftime('%Y-%m-%d')
O['hk'] = O.home.map(key); O['ak'] = O.away.map(key)
g = O.groupby(['date', 'hk', 'ak'])
C = g.agg(mkt_total=('total', 'median'), over_odds=('over_odds', 'median'), under_odds=('under_odds', 'median'),
          n_books=('book', 'nunique')).reset_index()

M = S.merge(C, left_on=['date_s', 'hk', 'ak'], right_on=['date', 'hk', 'ak'], how='inner')
if len(M) < 0.5 * len(C):  # allow a one-day tz slip, same trick as backtest.py
    S2 = S.copy(); S2['date_s'] = (S2.date + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    M = pd.concat([M, S2.merge(C, left_on=['date_s', 'hk', 'ak'], right_on=['date', 'hk', 'ak'], how='inner')]).drop_duplicates(['hk', 'ak', 'date_s'])
M = M.dropna(subset=['model_total'])
print(f'Matched {len(M)} of {len(C)} games with a market total to the model (2025-26 only, walk-forward fit).')
if len(M) == 0:
    raise SystemExit('No matches - check date/name alignment.')

# ---- accuracy: points MAE, model vs market line
mae_model = (M.model_total - M.total).abs().mean()
mae_mkt = (M.mkt_total - M.total).abs().mean()
print(f'\nPoints MAE  (lower better)   model {mae_model:.2f}   market line {mae_mkt:.2f}   n={len(M)}')

# ---- no-vig over-probability Brier (actual outcome = total > line)
io, iu = 1 / M.over_odds, 1 / M.under_odds
M['mkt_p_over'] = io / (io + iu)
M['over'] = (M.total > M.mkt_total).astype(float) + 0.5 * (M.total == M.mkt_total)
# model's own over-prob needs a scale for total-points variance; use the residual std from the fit seasons as sigma
resid_std = (S.total - S.model_total).std()
from scipy.stats import norm
M['model_p_over'] = 1 - norm.cdf(M.mkt_total, M.model_total, resid_std)
b_model = ((M.model_p_over - M.over) ** 2).mean(); b_mkt = ((M.mkt_p_over - M.over) ** 2).mean()
print(f'Over-prob Brier (lower better)  model {b_model:.4f}   market {b_mkt:.4f}   (sigma={resid_std:.1f})')

# ---- betting rule: flat-stake, EV>=5%, bet whichever side (over/under) the model + blend favors
M['blend_p_over'] = 0.5 * M.model_p_over + 0.5 * M.mkt_p_over
bets = []
for r in M.itertuples():
    for side, p, o in (('OVER', r.blend_p_over, r.over_odds), ('UNDER', 1 - r.blend_p_over, r.under_odds)):
        ev = p * o - 1
        if ev < 0.05: continue
        won = (r.total > r.mkt_total) if side == 'OVER' else (r.total < r.mkt_total)
        push = r.total == r.mkt_total
        res = 0.0 if push else (o - 1 if won else -1.0)
        bets.append(dict(date=r.date_s, match=f'{r.home} v {r.away}', side=side, line=r.mkt_total, odds=round(o, 3), ev=ev, res=res))
B = pd.DataFrame(bets)
if B.empty:
    print('\nNo totals bets met the EV>=5% rule.')
else:
    rng = np.random.default_rng(0)
    bs = [rng.choice(B.res.values, len(B)).mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'\nBetting rule: n={len(B)}  win {np.mean(B.res > 0):.1%}  flat ROI {B.res.mean():+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]')
    print(B.to_string(index=False))
    B.to_csv('totals_bets.csv', index=False)
