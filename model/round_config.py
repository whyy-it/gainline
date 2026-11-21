"""Single source of truth for this round's metadata - shared by produce.py (Matches/Tries tabs) and bets.py
(Bets tab), so there's no second hardcoded copy to drift out of sync (the bug found 21 Nov: duplicated round
metadata went stale in the template; the same risk existed between produce.py and bets.py too).
Update ALL of these together each week, alongside FIX/MKT in produce.py."""
COMPETITION = 'Nations Championship'
ROUND_LABEL = 'Round 6'
ROUND_DATE = 'Sat 21 Nov'   # see produce.py's note: assumes the whole round is one day
MARKET_DATE = '18 Nov'
