"""Data-driven player grades ("points added").
Step 1: learn what each non-scoring action is worth in match points: regress team margin on team stat differences
        (ridge, signs constrained: good actions >= 0, bad actions <= 0), trained on 2022-24 club + Test games.
Step 2: player points added = sum(value_k * stat_k) + s * points the player scored (s = credit for finishing).
Step 3: grade = 60 + 12 * z-score within position group (replacements graded vs replacements), clipped 30-99.
Validation: lineup strength built from these grades vs hand-weighted grades, scored on held-out club Brier."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from tries import GROUP

POS = ['line_breaks', 'defenders_beaten', 'metres_carried', 'carries', 'offloads', 'tackles_made', 'turnovers_won', 'lineouts_won', 'kicks', 'dominant_tackles', 'try_assists']
NEG = ['tackles_missed', 'turnovers_lost', 'penalties_conceded', 'yellow_cards', 'red_cards']
FEATS = POS + NEG

def team_diffs():
    p = pd.read_parquet('../model/player_stats.parquet')
    tm = pd.read_parquet('../model/team_match_stats.parquet'); okk = tm[tm.ok].set_index(['match_id', 'team']).index
    p = p[p.set_index(['match_id', 'team']).index.isin(okk)].copy()
    for f in FEATS + ['points', 'tries']: p[f] = p[f].fillna(0)
    agg = p.groupby(['match_id', 'team'])[FEATS + ['points', 'tries']].sum().reset_index()
    # pair the two teams of each match
    m = agg.merge(agg, on='match_id', suffixes=('', '_o')); m = m[m.team != m.team_o]
    m = m.merge(p.groupby('match_id').date.first().rename('date'), on='match_id')
    X = np.c_[[m[f] - m[f + '_o'] for f in FEATS]].T
    return p, m, X

def learn_values(m, X, train_end='2024-08-01'):
    # margin from scores: tries*5 + kicks; use team points where recorded, else tries*7 proxy
    pts = np.where(m.points + m.points_o > 0, m.points - m.points_o, (m.tries - m.tries_o) * 7)
    tr = (m.date < train_end).values
    sign = np.array([1] * len(POS) + [-1] * len(NEG))
    rg = Ridge(alpha=10, positive=True).fit(X[tr] * sign, pts[tr])
    vals = dict(zip(FEATS, rg.coef_ * sign))
    r2 = 1 - np.mean((pts[tr] - rg.predict(X[tr] * sign)) ** 2) / np.var(pts[tr])
    return vals, r2

def grade(p, vals, s=0.5):
    p = p.copy()
    p['pa'] = sum(p[f] * v for f, v in vals.items()) + s * p.points.fillna(0)
    p['grp'] = p.position.map(GROUP).fillna('sub_back')
    st = p.groupby('grp').pa.agg(['mean', 'std'])
    p['z'] = (p.pa - p.grp.map(st['mean'])) / p.grp.map(st['std'])
    p['grade'] = (60 + 12 * p.z).clip(30, 99).round()
    return p
