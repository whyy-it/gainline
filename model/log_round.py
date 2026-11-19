"""Results + odds log: builds the history needed to test the model against the market.
Files: odds_log.csv (one row per match per snapshot: open / close) and results.csv.
Usage:
  python3 log_round.py lines snapshot.csv     # append a snapshot (stage = open or close), see columns below
  python3 log_round.py result 2026-11-21 England "New Zealand" 24 20
  python3 log_round.py report                 # model vs market Brier, closing-line value, record
snapshot.csv columns: date,home,away,stage,book,ml_home,ml_draw,ml_away,spread_home,spread_home_odds,spread_away_odds,total,over_odds,under_odds,model_p_home,model_margin,model_total
Odds are decimal; spread_home is the home team's line (e.g. 2.5 = home +2.5)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
LOG, RES = os.path.join(HERE, 'odds_log.csv'), os.path.join(HERE, 'results.csv')

def add_lines(path):
    new = pd.read_csv(path, parse_dates=['date']); new['logged_at'] = pd.Timestamp.now().round('min')
    old = pd.read_csv(LOG, parse_dates=['date']) if os.path.exists(LOG) else None
    out = new if old is None else pd.concat([old, new], ignore_index=True)
    out.to_csv(LOG, index=False); print(f'logged {len(new)} rows -> {LOG} ({len(out)} total)')

def add_result(date, home, away, hs, as_):
    row = pd.DataFrame([dict(date=pd.Timestamp(date), home=home, away=away, hs=int(hs), as_=int(as_))])
    old = pd.read_csv(RES, parse_dates=['date']) if os.path.exists(RES) else None
    out = row if old is None else pd.concat([old[~((old.date == row.date[0]) & (old.home == home))], row])
    out.to_csv(RES, index=False); print('result saved:', home, hs, '-', as_, away)

def novig(h, d, a):
    ih, idr, ia = 1 / h, (1 / d if d and d == d else 0), 1 / a
    s = ih + idr + ia; return ih / s, ia / s

def report():
    L = pd.read_csv(LOG, parse_dates=['date'])
    if not os.path.exists(RES): print('no results yet'); return
    R = pd.read_csv(RES, parse_dates=['date'])
    k = ['date', 'home', 'away']
    op = L[L.stage == 'open'].drop_duplicates(k, keep='last'); cl = L[L.stage == 'close'].drop_duplicates(k, keep='last')
    M = op.merge(R, on=k).merge(cl[k + ['ml_home', 'ml_draw', 'ml_away', 'spread_home']], on=k, how='left', suffixes=('', '_close'))
    if M.empty: print('no settled matches with logged lines'); return
    M['y'] = (M.hs > M.as_).astype(float) + 0.5 * (M.hs == M.as_)
    M['mkt_open'] = [novig(h, d, a)[0] for h, d, a in zip(M.ml_home, M.ml_draw, M.ml_away)]
    has_close = M.ml_home_close.notna()
    M['mkt_close'] = np.where(has_close, [novig(h, d, a)[0] if h == h else np.nan for h, d, a in zip(M.ml_home_close, M.ml_draw_close, M.ml_away_close)], np.nan)
    b = lambda p: float(np.mean((p - M.y) ** 2))
    print(f'Settled matches: {len(M)}')
    print(f'Brier  model {b(M.model_p_home):.4f} | market open {b(M.mkt_open):.4f}' + (f' | market close {np.mean((M.mkt_close[has_close]-M.y[has_close])**2):.4f}' if has_close.any() else ''))
    if has_close.any():
        # closing-line value: did the line move toward the model between open and close?
        mv = (M.mkt_close - M.mkt_open)[has_close]; side = np.sign(M.model_p_home - M.mkt_open)[has_close]
        print(f'Line moved toward the model in {np.mean(mv * side > 0):.0%} of {has_close.sum()} games (avg {np.mean(mv * side) * 100:+.1f} pts of win prob)')
    M['margin'] = M.hs - M.as_
    print(f'Margin miss  model {np.abs(M.margin - M.model_margin).mean():.1f} | market spread {np.abs(M.margin + M.spread_home).mean():.1f}')

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if cmd == 'lines': add_lines(sys.argv[2])
    elif cmd == 'result': add_result(*sys.argv[2:7])
    else: report()
