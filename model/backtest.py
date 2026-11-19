"""Backtest Gainline against historical closing odds.
Inputs: preds_oos.csv (model's out-of-sample predictions, shipped) + odds_hist.csv (from odds_pull.py).
Run:  python3 backtest.py            (or --books draftkings,fanduel,betmgm to grade at PA books only)
Reports, per competition:
  - Brier: model vs closing market vs 50/50 blend (the market is the bar to beat)
  - Spread: model's average miss vs the closing line's miss
  - Betting rule (same as bets.py): blended EV >= +5%, ML needs a 3-pt rating edge, 2-way ML = draw-no-bet,
    bets at the best price across the chosen books; ROI with 95% bootstrap interval and the 10%+ ROI bar."""
import argparse, os, re, unicodedata, numpy as np, pandas as pd
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))

ALIAS = {'newcastle red bulls': 'newcastle', 'newcastle falcons': 'newcastle', 'bristol': 'bristol bears', 'bath rugby': 'bath',
         'exeter chiefs': 'exeter', 'gloucester rugby': 'gloucester', 'leicester': 'leicester tigers', 'northampton saints': 'northampton',
         'sale': 'sale sharks', 'stade toulousain': 'toulouse', 'union bordeaux begles': 'bordeaux-begles', 'bordeaux begles': 'bordeaux-begles',
         'ubb': 'bordeaux-begles', 'asm clermont auvergne': 'clermont auvergne', 'clermont': 'clermont auvergne', 'stade rochelais': 'la rochelle',
         'rc toulon': 'toulon', 'rc toulonnais': 'toulon', 'section paloise': 'pau', 'castres olympique': 'castres', 'aviron bayonnais': 'bayonne',
         'lyon ou': 'lyon', 'lou rugby': 'lyon', 'usa perpignan': 'perpignan', 'montpellier herault': 'montpellier', 'stade francais paris': 'stade francais',
         'rc vannes': 'vannes', 'us montauban': 'us montauban', 'glasgow': 'glasgow warriors', 'cardiff rugby': 'cardiff', 'cardiff blues': 'cardiff',
         'benetton rugby': 'benetton', 'benetton treviso': 'benetton', 'zebre parma': 'zebre', 'dragons rfc': 'dragons', 'vodacom bulls': 'bulls',
         'dhl stormers': 'stormers', 'hollywoodbets sharks': 'sharks', 'emirates lions': 'lions', 'edinburgh rugby': 'edinburgh',
         'nsw waratahs': 'waratahs', 'new south wales waratahs': 'waratahs', 'queensland reds': 'reds', 'western force': 'force', 'act brumbies': 'brumbies',
         'fiji drua': 'fijian drua', 'usa': 'united states of america'}
def key(t):
    t = unicodedata.normalize('NFKD', str(t)).encode('ascii', 'ignore').decode().lower().strip()
    t = re.sub(r'\s+', ' ', t.replace('-', ' ')).strip()
    t = ALIAS.get(t, t)
    return t.replace(' ', '-') if t == 'bordeaux begles' else t

def load(books):
    P = pd.read_csv(os.path.join(HERE, 'preds_oos.csv'))
    O = pd.read_csv(os.path.join(HERE, 'odds_hist.csv'))
    if books: O = O[O.book.isin(books)]
    O['date'] = pd.to_datetime(O.commence_time).dt.tz_convert(None).dt.strftime('%Y-%m-%d')
    O['hk'] = O.home.map(key); O['ak'] = O.away.map(key)
    P['hk'] = P.home.map(key); P['ak'] = P.away.map(key)
    # consensus close (median across books) + best price per side
    g = O.groupby(['date', 'hk', 'ak'])
    C = g.agg(ml_home=('ml_home', 'median'), ml_draw=('ml_draw', 'median'), ml_away=('ml_away', 'median'),
              spread_home=('spread_home', 'median'), sh_odds=('spread_home_odds', 'median'), sa_odds=('spread_away_odds', 'median'),
              best_ml_home=('ml_home', 'max'), best_ml_away=('ml_away', 'max'),
              best_sh=('spread_home_odds', 'max'), best_sa=('spread_away_odds', 'max'), n_books=('book', 'nunique')).reset_index()
    M = P.merge(C, on=['date', 'hk', 'ak'], how='inner')
    # allow a one-day timezone slip
    if len(M) < 0.5 * len(C):
        P2 = P.copy(); P2['date'] = (pd.to_datetime(P2.date) + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
        M = pd.concat([M, P2.merge(C, on=['date', 'hk', 'ak'], how='inner')]).drop_duplicates(['date', 'hk', 'ak'])
    return M, P, C

def market_probs(r):
    ih, ia = 1 / r.ml_home, 1 / r.ml_away; idr = 1 / r.ml_draw if pd.notna(r.ml_draw) and r.ml_draw > 0 else 0
    s = ih + ia + idr; return ih / s, ia / s

def run(M):
    M = M.copy(); M['y'] = (M.hs > M.as_).astype(float) + 0.5 * (M.hs == M.as_)
    mk = M.apply(market_probs, axis=1, result_type='expand'); M['mkt_h'] = mk[0]; M['mkt_a'] = mk[1]
    M['mkt_h_nd'] = M.mkt_h / (M.mkt_h + M.mkt_a)
    M['mod_h_nd'] = M.p_home / (M.p_home + M.p_away)
    M['blend'] = .5 * M.mod_h_nd + .5 * M.mkt_h_nd
    out = []
    for comp, x in M.groupby('competition'):
        y = x.y; dec = x[x.y != .5]
        b = lambda p: np.mean((p - y) ** 2)
        sp = x.dropna(subset=['spread_home'])
        out.append(dict(competition=comp, games=len(x), brier_model=b(x.mod_h_nd), brier_market=b(x.mkt_h_nd), brier_blend=b(x.blend),
                        miss_model=np.abs((sp.hs - sp.as_) - sp.model_margin).mean() if len(sp) else np.nan,
                        miss_line=np.abs((sp.hs - sp.as_) + sp.spread_home).mean() if len(sp) else np.nan))
    print('\nAccuracy vs the closing line (lower is better)')
    print(pd.DataFrame(out).round(4).to_string(index=False))
    # ---- betting rule
    bets = []
    for r in M.itertuples():
        pd_ = max(r.p_draw, 0); sig = r.sigma
        cands = []
        if pd.notna(r.spread_home) and pd.notna(r.best_sh):
            pc = 1 - norm.cdf(-r.spread_home, r.model_margin, sig); t = 1 / r.sh_odds + 1 / r.sa_odds
            mk_h = (1 / r.sh_odds) / t
            cands += [('HCP', 'home', .5 * pc + .5 * mk_h, r.best_sh, 1.0, r.model_margin + r.spread_home),
                      ('HCP', 'away', .5 * (1 - pc) + .5 * (1 - mk_h), r.best_sa, 1.0, -(r.model_margin + r.spread_home))]
        if pd.notna(r.best_ml_home) and pd.notna(r.spread_home):
            gap = r.model_margin + r.spread_home
            cands += [('ML', 'home', .5 * r.mod_h_nd + .5 * r.mkt_h_nd, r.best_ml_home, 1 - pd_, gap),
                      ('ML', 'away', .5 * (1 - r.mod_h_nd) + .5 * (1 - r.mkt_h_nd), r.best_ml_away, 1 - pd_, -gap)]
        best = None
        for kind, side, bp, o, nd, gap in cands:
            ev = nd * (bp * o - 1)
            if ev < .05 or (kind == 'ML' and gap < 3): continue
            if best is None or ev > best[3]: best = (kind, side, bp, ev, o, gap)
        if best is None: continue
        kind, side, bp, ev, o, gap = best
        mg = r.hs - r.as_
        if kind == 'HCP':
            adj = mg + r.spread_home if side == 'home' else -(mg + r.spread_home)
            res = o - 1 if adj > 0 else (0 if adj == 0 else -1)
        else:
            won = mg > 0 if side == 'home' else mg < 0
            res = 0 if mg == 0 else (o - 1 if won else -1)
        bets.append(dict(date=r.date, competition=r.competition, match=f'{r.home} v {r.away}', bet=f'{kind} {side}', odds=o, ev=ev, profit=res))
    B = pd.DataFrame(bets)
    if B.empty: print('\nNo bets met the rule.'); return
    rng = np.random.default_rng(0)
    def ci(p):
        bs = [p[rng.integers(0, len(p), len(p))].mean() for _ in range(4000)]; return np.percentile(bs, [2.5, 97.5])
    print('\nBetting rule results (1 unit flat stakes)')
    rows = []
    for comp, x in list(B.groupby('competition')) + [('ALL', B)]:
        lo, hi = ci(x.profit.values)
        rows.append(dict(competition=comp, bets=len(x), win_rate=(x.profit > 0).mean(), roi=x.profit.mean(), ci_low=lo, ci_high=hi,
                         avg_ev_claimed=x.ev.mean()))
    R = pd.DataFrame(rows); print(R.round(3).to_string(index=False))
    allr = R[R.competition == 'ALL'].iloc[0]
    print(f"\nVerdict: ROI {allr.roi:+.1%} on {int(allr.bets)} bets (95% CI {allr.ci_low:+.1%} to {allr.ci_high:+.1%}).",
          'Clears the 10% bar with confidence.' if allr.ci_low >= .10 else
          'Positive but not proven.' if allr.ci_low > 0 else 'Not distinguishable from zero/negative: do not scale up.')
    B.to_csv(os.path.join(HERE, 'backtest_bets.csv'), index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--books', default='')
    a = ap.parse_args(); books = [b for b in a.books.split(',') if b]
    M, P, C = load(books)
    print(f'Matched {len(M)} of {len(C)} odds games to model predictions (unmatched usually = name spellings; check ALIAS).')
    run(M)
