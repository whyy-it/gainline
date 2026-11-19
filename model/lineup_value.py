"""Player-value lineup strength and goal-kicking quality (all pre-match, walk-forward).
player value  = shrunk running mean of a position-normalised contribution score (same weights as the site's grades)
lineup        = mean value of the matchday squad (starters weight 1, replacements 0.5)
usual_gap     = lineup value today minus the team's average lineup value over its last 10 games (rotation in points)
kick_pct      = team's goal-kicking success over the previous 10 games where recorded
"""
import numpy as np, pandas as pd
from tries import GROUP

W = dict(tries=6, try_assists=4, line_breaks=3, defenders_beaten=1, metres_carried=0.06, offloads=1, carries=0.3,
         tackles_made=0.45, tackles_missed=-1.2, turnovers_won=3, penalties_conceded=-2, yellow_cards=-6, red_cards=-15,
         lineouts_won=0.8, dominant_tackles=0.6)

def build(shrink=6):
    p = pd.read_parquet('../model/player_stats.parquet')
    tm = pd.read_parquet('../model/team_match_stats.parquet')
    ok = tm[tm.ok].set_index(['match_id', 'team']).index
    p = p[p.set_index(['match_id', 'team']).index.isin(ok)].copy()
    for k in W: p[k] = p[k].fillna(0)
    p['raw'] = sum(p[k] * w for k, w in W.items())
    p['grp'] = p.position.map(GROUP).fillna('sub_back')
    p['starter'] = ~p.grp.str.startswith('sub')
    st = p.groupby('grp').raw.agg(['mean', 'std'])
    p['z'] = (p.raw - p.grp.map(st['mean'])) / p.grp.map(st['std'])
    p = p.sort_values(['player_id', 'date'])
    g = p.groupby('player_id')
    p['n_prev'] = g.cumcount(); p['sum_prev'] = g.z.cumsum() - p.z
    p['val'] = p.sum_prev / (p.n_prev + shrink)            # shrunk to 0 (= average player)
    p['w'] = np.where(p.starter, 1.0, 0.5)
    # kicking: attempts implied by goals / pct where pct recorded
    p['goals'] = p.conversion_goals.fillna(0) + p.penalty_goals.fillna(0)
    pct = p.kick_percent_success.where(p.kick_percent_success > 1, p.kick_percent_success * 100)
    p['att'] = np.where(pct > 0, p.goals / (pct / 100), np.where(p.goals > 0, np.nan, 0))
    team = p.groupby(['match_id', 'team']).apply(lambda x: pd.Series(dict(
        date=x.date.iloc[0], lineup=np.average(x.val, weights=x.w), goals=x.goals.sum(),
        att=x.att.sum() if x.att.notna().all() else np.nan))).reset_index().sort_values(['team', 'date'])
    gt = team.groupby('team')
    team['usual'] = gt.lineup.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    team['usual_gap'] = team.lineup - team.usual
    team['kick_pct'] = (gt.goals.transform(lambda s: s.shift(1).rolling(10, min_periods=4).sum()) /
                        gt.att.transform(lambda s: s.shift(1).rolling(10, min_periods=4).sum()))
    return team

def attach(df, team):
    look = {}
    for r in team.itertuples():
        for dd in (0, 1, -1):
            look.setdefault((r.team, r.date + pd.Timedelta(days=dd)), r)
    out = pd.DataFrame(index=df.index)
    for side, col in (('h', 'home'), ('a', 'away')):
        rows = [look.get((t, d)) for t, d in zip(df[col], df.date)]
        for f in ('lineup', 'usual_gap', 'kick_pct'):
            out[f + '_' + side] = [getattr(r, f) if r is not None else np.nan for r in rows]
    X = pd.DataFrame(index=df.index)
    X['lineup'] = (out.lineup_h - out.lineup_a)
    X['gap'] = (out.usual_gap_h - out.usual_gap_a)
    X['kick'] = (out.kick_pct_h - out.kick_pct_a)
    X = X.replace([np.inf, -np.inf], np.nan); X['kick'] = X.kick.clip(-0.6, 0.6)
    for c in list(X.columns):
        X[c + '_miss'] = X[c].isna().astype(int); X[c] = X[c].fillna(0)
    return X
