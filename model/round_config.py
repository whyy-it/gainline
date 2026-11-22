"""Optional market-odds override, plus the ESPN league ids auto_rounds.py / the live-score poller need.
Until 22 Nov this also hand-maintained the round's fixture list (COMPETITION/ROUND_LABEL/ROUND_DATE/FIX) -
that's gone now, replaced by auto_rounds.py detecting fixtures straight from ESPN every run (see
model/README.md, "we need this to update automatically"). All that's left to maintain by hand is MKT, and
only when Wyatt actually has a sportsbook board to compare against - there's still no free live odds source
for rugby (checked again 22 Nov). No board this week = MKT stays empty, the Market column shows "-", and
bets.py produces no bets for that match - not an error, just nothing to compare.

To add a board: nest it by competition, then home team, same 6-tuple as before (decimal 1, X, 2, home
handicap, home hcp odds, away hcp odds):
  MKT = {'French Top 14': {'Toulouse': (1.2, 1e9, 4.5, -12.5, 1.9, 1.9), ...}, ...}
and set MARKET_DATE to when the board was taken (e.g. '27 Nov')."""
MARKET_DATE = None
MKT = {}

# ESPN league ids this round's live-score/live-winprob poller needs (site/mobile_template.html fetches this
# directly, client-side - CORS is wide open on site.api.espn.com, confirmed 22 Nov) AND auto_rounds.py uses
# to detect each competition's current fixtures. Keyed by our own competition-label convention.
ESPN_LEAGUE_IDS = {'Nations Championship': '17567', 'Six Nations': '180659', 'French Top 14': '270559',
                   'Gallagher Premiership': '267979', 'United Rugby Championship': '270557'}
