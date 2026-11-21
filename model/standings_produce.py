"""Consolidates all 6 tournament simulations (season_sim.py's two + league_sim.py's three + Super Rugby's
special case) into one site.json field. Each competition's entry: team list with current points (0 if the
competition hasn't started), games played, title odds (p_champion + fair American odds), and p_qualify
where a qualification cutoff is meaningful (club leagues' playoff line) - Six Nations/Nations Championship
don't get a p_qualify (Six Nations has no playoff; every Nations Championship pool team reaches Finals
Weekend by definition, so "qualifying" isn't an interesting cut there - only the final matters)."""
import season_sim as SS
import league_sim as LS
import top14_sim as T14

def amer(p): return LS.amer(p)

def _nc_block():
    d = SS.run_nc()
    base_pts, _ = SS.nc_actual_standings()
    rows = [dict(team=t, pts=base_pts[t], played=5, p_champion=round(v['p_champion'], 4),
                fair_odds=amer(v['p_champion'])) for t, v in d.items()]
    return dict(label='Nations Championship 2026', has_qualify=False,
                note='R1-5 actual results; R6 (21 Nov) and Finals Weekend (27-29 Nov) simulated.',
                teams=sorted(rows, key=lambda r: -r['p_champion']))

def _six_nations_block():
    d = SS.run_six_nations()
    rows = [dict(team=t, pts=0, played=0, p_champion=round(v['p_champion'], 4), fair_odds=amer(v['p_champion']),
                p_grand_slam=round(v['p_grand_slam'], 4)) for t, v in d.items()]
    return dict(label='Six Nations 2027', has_qualify=False,
                note='Tournament has not started (5 Feb - 13 Mar 2027); simulated from current ratings.',
                teams=sorted(rows, key=lambda r: -r['p_champion']))

def _club_block(d, label, note):
    rows = [dict(team=t, pts=v['current_pts'], played=v['played'],
                p_qualify=round(v.get('p_qualify', v.get('p_top6', 0)), 4),
                p_champion=round(v['p_champion'], 4), fair_odds=amer(v['p_champion']))
           for t, v in d.items()]
    return dict(label=label, has_qualify=True, note=note, teams=sorted(rows, key=lambda r: -r['p_champion']))

def all_standings():
    return {
        'Nations Championship': _nc_block(),
        'Six Nations': _six_nations_block(),
        'Top 14': _club_block(T14.run(), 'Top 14', 'Top 6 make the playoffs (1st/2nd bye to semis).'),
        'Premiership': _club_block(LS.run('premiership'), 'Premiership', 'Top 4 make the playoffs.'),
        'URC': _club_block(LS.run('urc'), 'URC', 'Top 8 make the playoffs.'),
        'Super Rugby Pacific': _club_block(LS.run_super_rugby(), 'Super Rugby Pacific',
            'Season has not started (Feb-June 2027); simulated from current ratings. Top 6 make the playoffs.'),
    }

if __name__ == '__main__':
    import json
    d = all_standings()
    for k, v in d.items():
        print(f'\n=== {v["label"]} ===')
        for r in v['teams'][:5]: print(' ', r)
