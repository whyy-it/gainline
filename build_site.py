"""Rebuilds the site from site/mobile_template.html + model/site.json, writing to BOTH
site/gainline.html (the documented location, CLAUDE.md) and index.html (repo root, what GitHub Pages
actually serves - https://whyy-it.github.io/gainline/). Run this instead of writing to either file by hand;
the two drifted out of sync once already (21 Nov) when index.html was added as a one-off copy.

Public/private split (added 22 Nov after the Bets tab - real sportsbook lines, EV%, Kelly stakes, plus a
leftover "remove before publishing" note - turned up live on the public site): betsTab never goes into the
two committed/public files. It's stripped from the embedded data before writing them; the untouched full
data instead goes to site/gainline_private.html, a local-only file (gitignored, same as model/site.json
itself now is) for viewing the Bets tab. mobile_template.html's JS only shows the Bets nav link/section when
D.betsTab is actually present, so one template correctly renders both variants.

Run from the repo root: python3 build_site.py
"""
import pathlib, json

ROOT = pathlib.Path(__file__).parent
PRIVATE_OUT = ROOT / 'site' / 'gainline_private.html'

def build():
    tpl = (ROOT / 'site' / 'mobile_template.html').read_text()
    data_raw = (ROOT / 'model' / 'site.json').read_text()
    teamjs = (ROOT / 'site' / 'teamjs.js').read_text()

    public = json.loads(data_raw)
    public.pop('betsTab', None)
    out_public = tpl.replace('__DATA__', json.dumps(public)).replace('__TEAMJS__', teamjs)
    for dest in (ROOT / 'site' / 'gainline.html', ROOT / 'index.html'):
        dest.write_text(out_public)
        print(f'wrote {dest.relative_to(ROOT)} ({len(out_public)} bytes) - public, no betting content')

    out_private = tpl.replace('__DATA__', data_raw).replace('__TEAMJS__', teamjs)
    PRIVATE_OUT.write_text(out_private)
    print(f'wrote {PRIVATE_OUT.relative_to(ROOT)} ({len(out_private)} bytes) - PRIVATE, gitignored, not committed')

if __name__ == '__main__':
    build()
