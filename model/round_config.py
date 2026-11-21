"""Single source of truth for this round's metadata AND fixtures/market board - shared by produce.py
(Matches/Tries tabs) and bets.py (Bets tab), so there's no second hardcoded copy to drift out of sync.
Found twice now (21 Nov): the round label/date duplicated separately in the template, then MKT duplicated
separately in bets.py - same bug class, different files, every time it's two copies instead of one.
Update ALL of these together each week - when a game in FIX finishes, remove it here (and its MKT entry);
a "projection" for an already-decided game, using ratings that already absorbed its own result, is
nonsensical, not just stale (see model/README.md, 21 Nov live-results entry)."""
COMPETITION = 'Nations Championship'
ROUND_LABEL = 'Round 6'
ROUND_DATE = 'Sat 21 Nov'   # see produce.py's note: assumes the whole round is one day
MARKET_DATE = '18 Nov'

# England v NZ and Scotland v Japan removed 21 Nov (both finished - England 37-12, Scotland 67-5 - merged
# into games.parquet via espn_catchup.py). Remaining 4: Ireland-SA/Italy-Fiji were live, France-Argentina/
# Wales-Australia not yet started, at the time of this edit.
FIX = [('Ireland', 'South Africa', 'Aviva Stadium', '16:40 GMT'), ('Italy', 'Fiji', 'Udine', '16:40 GMT'),
       ('France', 'Argentina', 'Stade de France', '20:10 GMT'), ('Wales', 'Australia', 'Principality Stadium', '20:10 GMT')]
# market board (decimal): 1, X, 2, home handicap, home hcp odds, away hcp odds  -- Sportingbet ZA, 17 Nov
MKT = {'Ireland': (4.2, 1e9, 1.2083, 9.5, 1.8929, 1.9259),
       'Italy': (1.4, 1e9, 2.85, -5.5, 1.9434, 1.8621),
       'France': (1.0286, 1e9, 10.8, -19.5, 1.9259, 1.8929),
       'Wales': (2.75, 1e9, 1.4255, 5.5, 1.8929, 1.9259)}
