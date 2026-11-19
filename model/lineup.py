"""Squad-strength features from player appearance logs (rotation proxy).
For a team on a match date:
  core  = share of today's matchday squad who are among the team's 23 most-used players over the prior 365 days
  exp   = mean prior appearances (for this team, since 2022) of today's squad, capped at 40
"""
import pandas as pd, numpy as np
from collections import defaultdict, Counter
from build_data import norm

A = pd.read_parquet('../model/apps.parquet')
A['team'] = A.team.map(norm); A['date'] = pd.to_datetime(A.game_date.astype(str))
A = A.drop_duplicates(['player_id', 'match_id'])

def squad_features():
    feats = {}
    for team, g in A.sort_values('date').groupby('team'):
        hist = []           # (date, set of players)
        apps = Counter()
        for (d, mid), sq in g.groupby(['date', 'match_id']):
            players = set(sq.player_id)
            recent = Counter()
            for hd, hp in hist:
                if (d - hd).days <= 365:
                    recent.update(hp)
            top = {p for p, _ in recent.most_common(23)}
            core = len(players & top) / max(len(players), 1) if top else np.nan
            exp = np.mean([min(apps[p], 40) for p in players]) if hist else np.nan
            feats[(team, d)] = (core, exp, len(hist))
            hist.append((d, players)); apps.update(players)
    return feats

def attach(S, feats, tol_days=1):
    def look(team, d):
        for k in range(0, tol_days + 1):
            for s in (k, -k):
                v = feats.get((team, d + pd.Timedelta(days=s)))
                if v: return v
        return (np.nan, np.nan, 0)
    hv = [look(h, d) for h, d in zip(S.home, S.date)]; av = [look(a, d) for a, d in zip(S.away, S.date)]
    S = S.copy()
    S['core_h'] = [x[0] for x in hv]; S['core_a'] = [x[0] for x in av]
    S['exp_h'] = [x[1] for x in hv]; S['exp_a'] = [x[1] for x in av]
    S['core'] = S.core_h - S.core_a; S['exp'] = S.exp_h - S.exp_a
    return S
