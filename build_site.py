"""Rebuilds the site from site/mobile_template.html + model/site.json, writing to BOTH
site/gainline.html (the documented location, CLAUDE.md) and index.html (repo root, what GitHub Pages
actually serves - https://whyy-it.github.io/gainline/). Run this instead of writing to either file by hand;
the two drifted out of sync once already (21 Nov) when index.html was added as a one-off copy.
Run from the repo root: python3 build_site.py
"""
import pathlib

ROOT = pathlib.Path(__file__).parent

def build():
    tpl = (ROOT / 'site' / 'mobile_template.html').read_text()
    data = (ROOT / 'model' / 'site.json').read_text()
    teamjs = (ROOT / 'site' / 'teamjs.js').read_text()
    out = tpl.replace('__DATA__', data).replace('__TEAMJS__', teamjs)
    for dest in (ROOT / 'site' / 'gainline.html', ROOT / 'index.html'):
        dest.write_text(out)
        print(f'wrote {dest.relative_to(ROOT)} ({len(out)} bytes)')

if __name__ == '__main__':
    build()
