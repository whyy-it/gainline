#!/usr/bin/env bash
# One-time setup. Run from the repo root: bash setup.sh
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install -q pandas==2.3.3 numpy scipy scikit-learn pyarrow requests python-dotenv
# Player-level data (365 MB, too big to ship): only data/v3/player is used
[ -d data/rugbydata ] || git clone --depth 1 https://github.com/seanyboi/rugbydata data/rugbydata
[ -f .env ] || echo "ODDS_API_KEY=" > .env
echo "Done. Put your key in .env, then: source .venv/bin/activate && cd model && python3 verify_site.py"
