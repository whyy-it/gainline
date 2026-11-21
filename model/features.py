"""v2: richer state per match (computed pre-match, no leakage), then a walk-forward stacked model.
State variables:
  elo_diff      margin-Elo (v1)
  att/def       separate offence/defence ratings -> expected points each side (also gives totals)
  rest_diff     days since last match, home minus away (capped)
  travel        away side crossed hemispheres (SA <-> Europe, SH <-> NH)
  form          home/away avg Elo residual over last 3 games (momentum the ratings haven't absorbed)
  team_hfa      team-specific home edge (shrunk running mean of home residual)
  league dummies for HFA
"""
import numpy as np, pandas as pd, json
from model import G

SA_CLUBS = {'Bulls', 'Lions', 'Sharks', 'Stormers', 'Cheetahs', 'Pumas'}
SA_URC_CLUBS = {'Bulls', 'Lions', 'Sharks', 'Stormers'}   # the 4 validated in v6.py's sa_bonus (README, 18 Nov) -
# narrower than SA_CLUBS above (which also covers Cheetahs/Pumas for the unrelated 'travel' feature)
SH_NATIONS = {'New Zealand', 'Australia', 'South Africa', 'Argentina', 'Fiji', 'Samoa', 'Tonga', 'Uruguay', 'Chile',
              'Brazil', 'Paraguay', 'Namibia', 'Zimbabwe', 'Kenya'}
NH_HOSTS = {'England', 'France', 'Ireland', 'Italy', 'Scotland', 'Wales', 'Georgia', 'Portugal', 'Spain', 'Romania',
            'Japan', 'United States of America', 'Canada'}

def build_state(df, kind, K, hfa_fn, cap, regress, Ka=0.05, sa_bonus=0.0):
    R, ATT, DEF = {}, {}, {}
    last, resid_hist, hres = {}, {}, {}
    mu_pts = df[['hs', 'as_']].values.mean()
    rows, last_season = [], None
    for r in df.itertuples():
        season = r.date.year if r.date.month >= 8 else r.date.year - 1
        if last_season is not None and season != last_season:
            if kind == 'club':
                m = np.mean(list(R.values()))
                R = {t: m + (v - m) * (1 - regress) for t, v in R.items()}
                ATT = {t: v * (1 - regress) for t, v in ATT.items()}; DEF = {t: v * (1 - regress) for t, v in DEF.items()}
            else:
                R = {t: v * (1 - regress) for t, v in R.items()}
        last_season = season
        h, a = r.home, r.away
        H = 0.0 if r.neutral else hfa_fn(r)
        if not r.neutral and sa_bonus and ((h in SA_URC_CLUBS) != (a in SA_URC_CLUBS)): H += sa_bonus
        rh, ra = R.get(h, 0.), R.get(a, 0.)
        pred = rh - ra + H
        # attack/defence expected points (log-ish additive)
        eh = mu_pts + ATT.get(h, 0) - DEF.get(a, 0) + H / 2
        ea = mu_pts + ATT.get(a, 0) - DEF.get(h, 0) - H / 2
        rest_h = min((r.date - last.get(h, r.date - pd.Timedelta(days=21))).days, 21)
        rest_a = min((r.date - last.get(a, r.date - pd.Timedelta(days=21))).days, 21)
        if kind == 'club':
            travel = int((a in SA_CLUBS) != (h in SA_CLUBS)) * (1 if a in SA_CLUBS else -1) if not r.neutral else 0
        else:
            travel = int(a in SH_NATIONS and h in NH_HOSTS) - int(a in NH_HOSTS and h in SH_NATIONS) if not r.neutral else 0
        form_h = np.mean(resid_hist.get(h, [0])[-3:]); form_a = np.mean(resid_hist.get(a, [0])[-3:])
        th = hres.get(h, (0., 0))
        team_hfa = th[0] / (th[1] + 10) if not r.neutral else 0.0   # shrunk toward 0 with 10 pseudo-games
        rows.append(dict(i=r.Index, elo=pred, ad=eh - ea, tot=eh + ea, rest=rest_h - rest_a, travel=travel,
                         form=form_h - form_a, team_hfa=team_hfa, neutral=int(r.neutral), margin=r.hs - r.as_,
                         total=r.hs + r.as_, date=r.date, comp=r.competition, home=h, away=a))
        # updates
        m = r.hs - r.as_
        e = np.clip(m, -cap, cap) - pred
        R[h] = rh + K * e; R[a] = ra - K * e
        ATT[h] = ATT.get(h, 0) + Ka * (r.hs - eh); DEF[a] = DEF.get(a, 0) - Ka * (r.hs - eh)
        ATT[a] = ATT.get(a, 0) + Ka * (r.as_ - ea); DEF[h] = DEF.get(h, 0) - Ka * (r.as_ - ea)
        resid_hist.setdefault(h, []).append(e); resid_hist.setdefault(a, []).append(-e)
        if not r.neutral:
            s, n = hres.get(h, (0., 0)); hres[h] = (s + e, n + 1)
        last[h] = last[a] = r.date
    return pd.DataFrame(rows).set_index('i'), R, ATT, DEF

P = json.load(open('../model/params.json'))
IK, IH, ICAP, IREG = P['intl']['params']
CHFA = dict(top14=9, prem=7, urc=6, euro=6)
KEY = {'French Top 14': 'top14', 'Gallagher Premiership': 'prem', 'United Rugby Championship': 'urc',
       'European Rugby Champions Cup': 'euro', 'European Rugby Challenge Cup': 'euro'}

def intl_state():
    return build_state(G[G.kind == 'intl'], 'intl', IK, lambda r: IH, ICAP, IREG)
def club_state(sa_bonus=1.0):
    return build_state(G[G.kind == 'club'], 'club', 0.06, lambda r: CHFA[KEY[r.competition]], 40, 0.2, sa_bonus=sa_bonus)
