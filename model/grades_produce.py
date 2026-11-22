"""Reproduces the site's Grades tab (previously a frozen, unreproducible snapshot - see model/README.md).
Picks the most recent high-profile completed match with full player stats (player_stats.parquet is synced
daily via World Rugby's Pulse API - see README), grades every player in it, and writes grades/grades_match/
grade_values into site.json. Run after produce.py, or import player_grades() from it.

Weights: the README's hand-vs-data-driven finding ("equal within noise") was on a DIFFERENT objective -
lineup-strength as a multi-game betting feature predicting club match outcomes. For THIS product (grading
one player's performance in one specific match from that match's own box score), that's the wrong test.
Checked directly: hand weights get R2=-0.79 reconstructing a single match's point margin from its own stat
diffs (worse than predicting the mean); the data-driven ridge regression gets R2=0.65 on the same held-out
games. Uses the data-driven weights for the live Grades tab accordingly, refit here each run (not frozen) -
see model/README.md for the full writeup. grade_weights_pa.json / hand weights stay documented as the
validated choice for the lineup-strength betting feature specifically; this script does not touch that use.
"""
import json, numpy as np, pandas as pd
from tries import GROUP
from grades import team_diffs, learn_values, grade, historical_group_stats, FEATS

ABBR = {'line_breaks': 'LB', 'defenders_beaten': 'DB', 'metres_carried': 'm', 'carries': 'car', 'offloads': 'offld',
        'tackles_made': 'tkl', 'turnovers_won': 'TO won', 'lineouts_won': 'LO', 'kicks': 'kicks', 'dominant_tackles': 'dom tkl',
        'try_assists': 'assist', 'tackles_missed': 'tkl missed', 'turnovers_lost': 'TO conceded', 'penalties_conceded': 'pen conceded',
        'yellow_cards': 'yellow card', 'red_cards': 'red card', 'points': 'pts', 'tries': 'T'}
NEG_FEATS = {'tackles_missed', 'turnovers_lost', 'penalties_conceded', 'yellow_cards', 'red_cards'}

HIGH_PROFILE = {'International Test Match', 'The Rugby Championship', 'Six Nations', 'Rugby World Cup',
                'Autumn Nations Series', 'Autumn Nations Cup', 'Nations Championship'}

def fit_report(m, X, train_end='2024-08-01'):
    """Fit both weight sets and report val fit quality (R^2 on a held-out slice after train_end)."""
    vals, r2_train = learn_values(m, X, train_end)
    tr = (m.date < train_end).values; va = ~tr
    sign = np.array([1 if f in vals and vals[f] >= 0 else -1 for f in FEATS])  # just for reporting sign convention
    pred_va = sum(X[:, i] * vals[f] for i, f in enumerate(FEATS))[va]
    pts = np.where(m.points + m.points_o > 0, m.points - m.points_o, (m.tries - m.tries_o) * 7)
    r2_val = 1 - np.mean((pts[va] - pred_va) ** 2) / np.var(pts[va])
    hand = json.load(open('grade_weights_pa.json'))
    pred_hand_va = sum(X[:, i] * hand.get(f, 0.0) for i, f in enumerate(FEATS))[va]
    r2_hand_val = 1 - np.mean((pts[va] - pred_hand_va) ** 2) / np.var(pts[va])
    print(f'Data-driven weights: train R2={r2_train:.3f}  val R2={r2_val:.3f}  (n_train={tr.sum()}, n_val={va.sum()})')
    print(f'Hand weights:                        val R2={r2_hand_val:.3f}  (same val set)')
    return vals, hand

def key_string(row, vals, n=3):
    contrib = sorted(((abs(row[f] * vals.get(f, 0)), row[f], f) for f in FEATS if row.get(f, 0)), reverse=True)
    parts = []
    for _, v, f in contrib[:n]:
        if v == 0: continue
        label = ABBR.get(f, f); sign = '-' if f in NEG_FEATS else ''
        vs = f'{sign}{int(v)}' if abs(v - int(v)) < 1e-6 else f'{sign}{v:.1f}'
        parts.append(f'{vs} {label}')
    return ', '.join(parts) if parts else '-'

def pick_match():
    tm = pd.read_parquet('team_match_stats.parquet'); tm = tm[tm.ok]
    p = pd.read_parquet('player_stats.parquet')
    p = p[p.set_index(['match_id', 'team']).index.isin(tm.set_index(['match_id', 'team']).index)]
    maxd = p.date.max()
    window = p[p.date >= maxd - pd.Timedelta(days=5)].merge(tm[['match_id', 'team', 'comp']], on=['match_id', 'team'], how='left')
    hp = window[window.comp.isin(HIGH_PROFILE)]
    pool = hp if len(hp) else window
    tries_by_match = pool.groupby('match_id').tries.sum().sort_values(ascending=False)
    mid = tries_by_match.index[0]
    g = pool[pool.match_id == mid]
    teams = g.team.unique(); comp = g.comp.iloc[0]
    tm_row = tm[(tm.match_id == mid) & (tm.team == teams[0])].iloc[0]
    score = {t: int(g[g.team == t].points.sum()) if g[g.team == t].points.notna().all() and g[g.team == t].points.sum() > 0
                else int(g[g.team == t].tries.sum() * 7) for t in teams}
    label = f'{teams[0]} {score[teams[0]]}–{score[teams[1]]} {teams[1]} · {comp} · {tm_row.date.strftime("%d %b %Y")}'
    return mid, g.copy(), label

def player_grades():
    p, m, X = team_diffs()
    vals, hand = fit_report(m, X)
    hist = historical_group_stats(p, vals)   # stable per-position mean/variance - a single match's own group
    # can be as few as 4-10 players, too noisy to z-score against alone (see grades.py docstring, 22 Nov)
    mid, roster, label = pick_match()
    for f in FEATS + ['points']: roster[f] = roster[f].fillna(0)
    graded = grade(roster, vals, hist=hist)   # data-driven weights - validated for THIS (single-match) objective above
    out = []
    for _, r in graded.sort_values('grade', ascending=False).iterrows():
        out.append(dict(team=r.team, name=r['name'], pos=r.grp.replace('_', ' '), grade=int(r.grade),
                        pa=round(float(r.pa), 1), key=key_string(r, vals)))
    return out, label, vals

if __name__ == '__main__':
    out, label, vals = player_grades()
    print(f'\nFeatured match: {label}')
    print(f'{len(out)} players graded\n')
    for r in out[:10]: print(r)
    d = json.load(open('site.json'))
    d['grades'] = out; d['grades_match'] = label; d['grade_values'] = {k: round(float(v), 4) for k, v in vals.items()}
    json.dump(d, open('site.json', 'w'), default=float)
    json.dump({k: round(float(v), 4) for k, v in vals.items()}, open('grade_values.json', 'w'), indent=1)
    print('\nwrote grades / grades_match / grade_values into site.json, and refreshed grade_values.json')
