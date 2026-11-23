"""Data-driven player grades ("points added").
Step 1: learn what each non-scoring action is worth in match points: regress team margin on team stat differences
        (ridge, signs constrained: good actions >= 0, bad actions <= 0), trained on 2022-24 club + Test games.
Step 2: player points added = sum(value_k * stat_k) + s * points the player scored (s = credit for finishing).
Step 3: grade = 60 + 12 * z-score within position group, shrunk toward each group's historical mean/variance
        (clipped 30-99). Found 22 Nov: a single match's own position group can be tiny (as few as 4-10 players -
        sub_fwd, back_row, wing all run that low), so its own mean/std is too noisy to z-score against safely.
        Real case that exposed it: a Scotland 67-5 Japan blowout graded a sub forward with +3.7 points-added
        84 - the match's TOP grade - ahead of a back-rower with +13.4 and a winger with +24.8 (a hat-trick),
        purely because the sub-forward group that match happened to cluster near zero (mean 0.09, std 1.84)
        while the other groups' natural spread was wider. Empirical-Bayes shrinkage toward each group's
        mean/variance across ALL matches (K=10 pseudo-players - see README for how that was picked) fixes it:
        the three players above re-grade to 85/96/96, which actually tracks who did more in the game.
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

CARD_OVERRIDE = {'yellow_cards': -3.5, 'red_cards': -8.0}
# Checked (21 Nov): the ridge fit puts BOTH cards near zero regardless of alpha (0.1-50 tried), and fitting
# them on the residual after the other 14 features (to rule out shrinkage) makes it worse, not better -
# red_cards flips POSITIVE. A card's cost runs entirely through the other box-score stats (a team down a
# man naturally racks up worse tackle/turnover numbers over the rest of the game); there's no standalone
# signal left for a discrete card coefficient to find, so the data genuinely can't answer this question.
# Anchored here at domain-reasonable values instead of trusting a fit that's actively misleading.

def learn_values(m, X, train_end='2024-08-01'):
    # margin from scores: tries*5 + kicks; use team points where recorded, else tries*7 proxy
    pts = np.where(m.points + m.points_o > 0, m.points - m.points_o, (m.tries - m.tries_o) * 7)
    tr = (m.date < train_end).values
    sign = np.array([1] * len(POS) + [-1] * len(NEG))
    rg = Ridge(alpha=10, positive=True).fit(X[tr] * sign, pts[tr])
    vals = dict(zip(FEATS, rg.coef_ * sign))
    vals.update(CARD_OVERRIDE)
    r2 = 1 - np.mean((pts[tr] - rg.predict(X[tr] * sign)) ** 2) / np.var(pts[tr])
    return vals, r2

K_GRADE_SHRINK = 10   # pseudo-players toward the historical group mean/variance - picked by checking the
# Scotland v Japan case in the docstring above: K=10 fixes it (sub-forward drops from top grade 84 to 85-ish
# *relative* standing, i.e. no longer tops the match; the real standouts re-grade to 96) without over-shrinking
# into the 99 ceiling the way K=20-40 did when tried - see model/README.md, 22 Nov.

def historical_group_stats(p_all, vals, s=0.5):
    """Stable per-position-group (mean, var) of single-match points-added, computed across every match in the
    data - not just the one being graded. A single match's own group can be as few as 4-10 players (too noisy
    to z-score against on its own); this is what grade() shrinks toward."""
    p_all = p_all.copy()
    for f in FEATS + ['points']: p_all[f] = p_all[f].fillna(0)
    p_all['pa'] = sum(p_all[f] * v for f, v in vals.items()) + s * p_all.points
    p_all['grp'] = p_all.position.map(GROUP).fillna('sub_back')
    return p_all.groupby('grp').pa.agg(['mean', 'var'])

def grade(p, vals, s=0.5, hist=None, K=K_GRADE_SHRINK):
    p = p.copy()
    p['pa'] = sum(p[f] * v for f, v in vals.items()) + s * p.points.fillna(0)
    p['grp'] = p.position.map(GROUP).fillna('sub_back')
    match_st = p.groupby('grp').pa.agg(['mean', 'var', 'count']).fillna({'var': 0.0})   # pandas' sample var of
    # a single observation is NaN, not 0 - a position group with exactly 1 player in this match (common: some
    # groups are a single starter per side) would otherwise poison the whole shrunk blend below with NaN. 0 is
    # the right default here: with n this small the K=10 historical shrinkage already dominates regardless.
    if hist is not None:
        n = match_st['count']
        mean = (n * match_st['mean'] + K * hist['mean']) / (n + K)
        var = (n * match_st['var'] + K * hist['var']) / (n + K)
    else:   # no historical baseline available - fall back to the (noisier) single-match stats, same as before
        mean, var = match_st['mean'], match_st['var']
    p['z'] = (p.pa - p.grp.map(mean)) / p.grp.map(var) ** 0.5
    p['grade'] = (60 + 12 * p.z).clip(30, 99).round()
    return p
