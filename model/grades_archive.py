"""Publishes player grades for EVERY match we have full player-stats data for (23 Nov: "you should be able to
go back to any game we have data on and see the player ratings" - previously only the single most-recent
high-profile match was ever graded, via grades_produce.py's pick_match()).

Embedding all ~5,000 matches' grades directly in site.json isn't viable - each match averages ~46 player
rows, so the full archive is ~227k rows; inlined that would be tens of MB on every single page load for a
feature almost no one opens on a given visit. Instead this publishes one small JSON file per match under
repo-root grades_archive/<match_id>.json (git/GitHub Pages serve thousands of small static files just fine -
no different from any other static asset) plus one lightweight index (grades_archive/index.json: match_id,
date, teams, scores, competition) that site/mobile_template.html fetches lazily, only when the Grades tab's
match picker is actually opened. Browsing a game fetches exactly one ~3-5KB file for that match - nothing
else changes. Lives at repo ROOT, not under site/, because index.html (what GitHub Pages actually serves) is
also at repo root - build_site.py substitutes a __ARCHIVE_BASE__ path prefix so the two files under site/
(gainline.html, gainline_private.html) can still reach it with a '../' relative path.

Regenerates everything every run rather than only new matches: grade_values/historical_group_stats are
refit on ALL current data each time (same as grades_produce.py), so an old match's grade can drift slightly
as the model improves - recomputing the full ~4,974-match archive takes ~165s (measured 23 Nov), trivial
for a scheduled background job, and avoids ever serving a stale file from an earlier model fit.

Run after produce.py (needs player_stats.parquet/team_match_stats.parquet, same as grades_produce.py) and
before build_site.py (grades_archive/ must exist for build_site.py's __ARCHIVE_BASE__ substitution to be
meaningful, though it'll substitute fine either way - the directory itself doesn't need to exist yet)."""
import json, pathlib, time
import pandas as pd
from grades import team_diffs, learn_values, historical_group_stats, grade, FEATS

OUT_DIR = pathlib.Path('../grades_archive')

def build():
    t0 = time.time()
    p_all, m, X = team_diffs()
    vals, _ = learn_values(m, X, train_end=str(p_all.date.max() + pd.Timedelta(days=1)))
    hist = historical_group_stats(p_all, vals)
    for f in FEATS + ['points']: p_all[f] = p_all[f].fillna(0)

    tm = pd.read_parquet('team_match_stats.parquet')
    tm_ok = tm[tm.ok]
    valid_ids = set(tm_ok.match_id)   # team_match_stats has one row PER TEAM per match, so match_id repeats -
    # dedupe before using as a lookup (a .loc on the raw frame would return a 2-row Series, not a scalar)
    comp_by_match = tm_ok.drop_duplicates('match_id').set_index('match_id')['comp']
    by_match = p_all.groupby('match_id')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for mid, roster in by_match:
        if mid not in valid_ids: continue   # not a validated two-sided match (grades_produce.py's bar)
        teams = roster.team.unique()
        if len(teams) != 2: continue
        score = {t: int(roster[roster.team == t].points.sum()) for t in teams}
        graded = grade(roster.copy(), vals, hist=hist)
        players = [dict(team=r.team, name=r['name'], pos=r.grp.replace('_', ' '), grade=int(r.grade),
                        pa=round(float(r.pa), 1)) for _, r in graded.sort_values('grade', ascending=False).iterrows()]
        date = roster.date.iloc[0]
        comp = comp_by_match.get(mid, '')
        if pd.isna(comp): comp = ''   # 21 of 9,666 team_match_stats rows have no comp recorded - a MISSING
        # VALUE, not a missing KEY, so the .get() fallback above doesn't catch it on its own. Matters here
        # specifically because a bare float NaN serializes to the literal token `NaN` in Python's json.dumps
        # (non-standard JSON - browsers' JSON.parse correctly rejects it), which silently broke every fetch
        # of this file client-side with no visible error until actually tested in a browser.
        t1, t2 = teams[0], teams[1]
        label = f'{t1} {score[t1]}–{score[t2]} {t2} · {comp} · {pd.Timestamp(date).strftime("%d %b %Y")}'
        (OUT_DIR / f'{mid}.json').write_text(json.dumps(dict(match=label, players=players), default=float, allow_nan=False))
        index.append(dict(match_id=mid, date=pd.Timestamp(date).strftime('%Y-%m-%d'), home=t1, away=t2,
                          home_score=score[t1], away_score=score[t2], competition=comp))
    index.sort(key=lambda r: r['date'], reverse=True)
    (OUT_DIR / 'index.json').write_text(json.dumps(index, allow_nan=False))
    print(f'grades_archive: {len(index)} matches published to {OUT_DIR} in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    build()
