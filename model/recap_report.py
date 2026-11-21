"""Game recap: final score, best players (from grades_produce's grading), and highlight moments synthesized
from box-score stats. No play-by-play/timestamps exist anywhere in the data (player_stats.parquet is match-
level totals only - see README), so "highlights" are standout STAT LINES ("scored a brace", "won 3 turnovers"),
not a minute-by-minute account of the game. Reuses grades_produce's featured-match selection so the recap and
the Grades tab are always about the same game."""
import pandas as pd
import grades_produce as GP

# (stat, threshold, priority - lower surfaces first, format). Rarer defensive/carrying feats outrank a
# routine single try (most matches have several of those; a 20-tackle game does not).
THRESH = [
    ('red_cards', 1, 0, lambda n, name, team: f'{name} ({team}) was sent off'),
    ('tries', 3, 0, lambda n, name, team: f'{name} ({team}) scored a hat-trick'),
    ('tries', 2, 1, lambda n, name, team: f'{name} ({team}) scored twice'),
    ('yellow_cards', 1, 2, lambda n, name, team: f'{name} ({team}) was sin-binned'),
    ('dominant_tackles', 3, 3, lambda n, name, team: f'{name} ({team}) made {int(n)} dominant tackles'),
    ('turnovers_won', 3, 3, lambda n, name, team: f'{name} ({team}) won {int(n)} turnovers'),
    ('metres_carried', 90, 4, lambda n, name, team: f'{name} ({team}) carried for {int(n)}m'),
    ('tackles_made', 18, 4, lambda n, name, team: f'{name} ({team}) made {int(n)} tackles'),
    ('try_assists', 2, 5, lambda n, name, team: f'{name} ({team}) set up {int(n)} tries'),
    ('tries', 1, 6, lambda n, name, team: f'{name} ({team}) scored a try'),
]
MAX_PLAIN_TRIES = 3   # cap routine single-try bullets; summarize the rest instead of flooding the list

def highlights(roster, max_n=8):
    seen, out, plain_try_scorers = set(), [], []
    for stat, bar, prio, fmt in THRESH:
        if stat not in roster.columns: continue
        for _, r in roster[roster[stat].fillna(0) >= bar].iterrows():
            if r.player_id in seen: continue   # one highlight per player, their single best (rarest) stat line
            seen.add(r.player_id)
            if stat == 'tries' and bar == 1:
                plain_try_scorers.append(r['name']); continue
            out.append((prio, -r[stat], fmt(r[stat], r['name'], r.team)))
    out.sort()
    out = [h[2] for h in out]
    if plain_try_scorers:
        shown, rest = plain_try_scorers[:MAX_PLAIN_TRIES], plain_try_scorers[MAX_PLAIN_TRIES:]
        out += [f'{n} scored a try' for n in shown]
        if rest: out.append(f'Also scored: {", ".join(rest)}')
    return out[:max_n]

def recap():
    out, label, vals = GP.player_grades()
    mid, roster, _ = GP.pick_match()
    teams = roster.team.unique()
    score = label.split('·')[0].strip()   # "Home SC–SC Away", already computed by pick_match
    potm = []
    for t in teams:
        top = [p for p in out if p['team'] == t][:1]
        if top: potm.append(dict(team=t, name=top[0]['name'], pos=top[0]['pos'], key=top[0]['key'], grade=top[0]['grade']))
    hl = highlights(roster)
    return dict(score=score, competition=label.split('·')[1].strip() if '·' in label else '',
                date=label.split('·')[-1].strip(), potm=potm, highlights=hl)

def render_text(r):
    lines = [r['score'], f"{r['competition']} · {r['date']}", '']
    lines.append('PLAYER OF THE MATCH')
    for p in r['potm']:
        lines.append(f"  {p['team']}: {p['name']} ({p['pos']}, grade {p['grade']}) - {p['key']}")
    lines.append(''); lines.append('HIGHLIGHTS')
    for h in r['highlights']: lines.append(f'  - {h}')
    return '\n'.join(lines)

if __name__ == '__main__':
    r = recap()
    print(render_text(r))
