"""Anytime try-scorer model.
lambda_player = team_expected_tries * position_share * player_multiplier ;  P(anytime) = 1 - exp(-lambda)
 - team_expected_tries: Poisson GLM on pre-match Elo expected margin + competition
 - position_share: share of a team's tries scored by one player in that slot (by position group, starter/bench)
 - player_multiplier: empirical-Bayes ratio of the player's actual tries to expected tries, shrunk with k pseudo-tries
Everything is computed walk-forward (pre-match only)."""
import os, numpy as np, pandas as pd, warnings
from sklearn.linear_model import PoissonRegressor
from build_data import norm
warnings.filterwarnings('ignore')

GROUP = {'left_wing': 'wing', 'right_wing': 'wing', 'wing': 'wing', 'fullback': 'fullback',
         'centre': 'centre', 'inside_centre': 'centre', 'outside_centre': 'centre', 'fly_half': 'fly_half',
         'scrum_half': 'scrum_half', 'flanker': 'back_row', 'blindside_flanker': 'back_row', 'openside_flanker': 'back_row',
         'number_eight': 'back_row', 'back_row': 'back_row', 'hooker': 'hooker', 'lock': 'lock', 'prop': 'prop',
         'tighthead_prop': 'prop', 'loosehead_prop': 'prop', 'back_sub': 'sub_back', 'forward_sub': 'sub_fwd',
         'front_row_sub': 'sub_fwd', 'utility back': 'sub_back'}

def load():
    p = pd.read_parquet('../model/player_full.parquet')
    p['team'] = p.team.map(norm); p['date'] = pd.to_datetime(p.game_date.astype(str))
    p['grp'] = p.position.map(GROUP)
    p = p[p.grp.notna() & p.tries.notna()]
    p['tries'] = p.tries.astype(float)
    # data cleaning: drop matches with impossible try counts
    tm = p.groupby(['match_id', 'team']).agg(t=('tries', 'sum'), pts=('points', 'sum'), mx=('tries', 'max'), n=('player_id', 'size')).reset_index()
    bad = tm[(tm.mx > 4) | ((tm.pts > 0) & (tm.t * 5 > tm.pts + 1)) | (tm.t > 16) | (tm.n < 18)]   # points can be blank (2026 files)
    p = p[~p.set_index(['match_id', 'team']).index.isin(bad.set_index(['match_id', 'team']).index)]
    # attach pre-match Elo expected margin
    SC = pd.read_parquet('../model/SC.parquet'); SI = pd.read_parquet('../model/SI.parquet')
    parts = [SC.assign(kind='club'), SI.assign(kind='intl')]
    if os.path.exists('../model/SR.parquet'):
        SRS = pd.read_parquet('../model/SR.parquet'); parts.append(SRS.assign(kind='club'))
    S = pd.concat(parts)
    em = {}
    for r in S.itertuples():
        for t, e in ((r.home, r.elo), (r.away, -r.elo)):
            for dd in (0, 1, -1):
                em.setdefault((t, r.date + pd.Timedelta(days=dd)), (e, r.kind, r.comp))
    v = [em.get((t, d)) for t, d in zip(p.team, p.date)]
    p['exp_margin'] = [x[0] if x else np.nan for x in v]; p['kind'] = [x[1] if x else None for x in v]
    p = p[p.exp_margin.notna()]
    # expected team points from attack/defence state (v4.1: improves test Brier 0.1143 -> 0.1136)
    ep = {}
    for r in S.itertuples():
        for t, v in ((r.home, (r.tot + r.ad) / 2), (r.away, (r.tot - r.ad) / 2)):
            for dd in (0, 1, -1):
                ep.setdefault((t, r.date + pd.Timedelta(days=dd)), v)
    p['exp_pts'] = [ep.get((t, d), np.nan) for t, d in zip(p.team, p.date)]
    p = p[p.exp_pts.notna()]
    return p.sort_values('date')

def team_table(p):
    return p.groupby(['match_id', 'team', 'date', 'kind', 'competition']).agg(t=('tries', 'sum'), em=('exp_margin', 'first'), pts=('exp_pts', 'first')).reset_index()

def fit_team(T):
    X = pd.get_dummies(T[['em']].assign(intl=(T.kind == 'intl').astype(int), em2=T.em.clip(-40, 40)), dtype=float)
    m = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(T[['em']].assign(em2=T.em.clip(-40, 40), intl=(T.kind == 'intl').astype(int), pts=T.pts)[['em2', 'intl', 'pts']], T.t); return m

def walk_forward(p, test_from='2025-08-01', k=8.0):
    T = team_table(p)
    tr_T = T[T.date < test_from]; mt = fit_team(tr_T)
    T['lam'] = mt.predict(T[['em']].assign(em2=T.em.clip(-40, 40), intl=(T.kind == 'intl').astype(int), pts=T.pts)[['em2', 'intl', 'pts']])
    p = p.merge(T[['match_id', 'team', 'lam']], on=['match_id', 'team'])
    # position shares from training period
    tr = p[p.date < test_from]
    tt = tr.groupby(['match_id', 'team']).tries.transform('sum')
    share = (tr.groupby('grp').tries.sum() / tr.groupby('grp').size()) / (tr.tries.sum() / tr.groupby(['match_id', 'team']).ngroups)
    p['pos_share'] = p.grp.map(share)   # expected fraction of team tries per player in this slot
    p['exp_t'] = p.lam * p.pos_share
    # running player multiplier (pre-match)
    p = p.sort_values(['player_id', 'date'])
    g = p.groupby('player_id')
    p['cum_t'] = g.tries.cumsum() - p.tries; p['cum_e'] = g.exp_t.cumsum() - p.exp_t
    p['mult'] = (p.cum_t + k) / (p.cum_e + k)
    p['lam_pos'] = p.exp_t; p['lam_full'] = p.exp_t * p.mult
    base_rate = tr.groupby('grp').tries.apply(lambda s: (s > 0).mean())
    p['p_posonly'] = p.grp.map(base_rate)
    p['p_team'] = 1 - np.exp(-p.lam_pos); p['p_full'] = 1 - np.exp(-p.lam_full)
    p['y'] = (p.tries > 0).astype(float)
    return p, mt, share

def scores(te):
    out = {}
    for c in ['p_posonly', 'p_team', 'p_full']:
        q = te[c].clip(1e-4, 1 - 1e-4)
        out[c] = dict(brier=np.mean((q - te.y) ** 2), logloss=-np.mean(te.y * np.log(q) + (1 - te.y) * np.log(1 - q)))
    return pd.DataFrame(out).T

if __name__ == '__main__':
    pd.set_option('display.width', 200)
    p = load(); print('rows', len(p), 'try rate', p.tries.gt(0).mean().round(3))
    best = None
    for k in [2, 4, 8, 16]:
        P_, mt, share = walk_forward(p, k=k)
        te = P_[P_.date >= '2025-08-01']; s = scores(te)
        print('k', k, s.loc['p_full'].round(5).to_dict())
        if best is None or s.loc['p_full', 'logloss'] < best[1].loc['p_full', 'logloss']: best = (k, s, P_)
    k, s, P_ = best; te = P_[P_.date >= '2025-08-01']
    print('\nbest k', k, ' test rows', len(te)); print(s.round(5))
    print('\nteam-tries GLM coef (margin, intl):', mt.coef_.round(4), 'intercept', round(mt.intercept_, 3))
    print('\nposition share multiplier:'); print(share.round(3).sort_values(ascending=False))
    te = te.assign(b=pd.cut(te.p_full, [0, .05, .1, .15, .2, .3, .4, .6, 1]))
    print('\ncalibration (test):'); print(te.groupby('b', observed=True).agg(n=('y', 'size'), pred=('p_full', 'mean'), actual=('y', 'mean')).round(3))
    te['dec'] = pd.qcut(te.p_full.rank(method='first'), 10, labels=False)
    print('\nhit rate by decile:'); print(te.groupby('dec').agg(pred=('p_full', 'mean'), actual=('y', 'mean')).round(3).T)
    P_.to_parquet('../model/tries_scored.parquet')
