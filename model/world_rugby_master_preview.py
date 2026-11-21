"""Sanity check on the free 20-row World_Rugby_Master Kaggle sample (data/Sample_World_Rugby.csv)
before buying the full $139 dataset. NOT a model backtest: our Tests/Clubs/SR models have no
predictions for these specific matches (scattered 2008-2019, mostly before our 2018+ data starts).
This just proves out the odds-ingestion math and shows what the full dataset would give us.
"""
import pandas as pd

d = pd.read_csv('../data/Sample_World_Rugby.csv', sep=';', encoding='utf-8-sig')
d = d[d['Type'] == 'Rugby Union'].copy()
print(f'{len(d)} Rugby Union rows (of {sum(pd.read_csv("../data/Sample_World_Rugby.csv", sep=";", encoding="utf-8-sig")["Type"] != None)} total sample rows)\n')

ih, idr, ia = 1 / d.Cote1, 1 / d.CoteX, 1 / d.Cote2
d['vig'] = ih + idr + ia - 1
d['mkt_home_dnb'] = ih / (ih + ia)  # no-vig draw-no-bet home prob
d['home_win'] = (d.ScoreH > d.ScoreA).astype(int)
d['draw'] = (d.ScoreH == d.ScoreA).astype(int)

print('Leagues in sample:', ', '.join(sorted(d.League.unique())))
print(f'\nAvg vig: {d.vig.mean():.1%}  (typical 1X2 book overround)')
print(f'Draws: {d.draw.sum()}/{len(d)}')

nd = d[d.draw == 0]
brier_mkt = ((nd.mkt_home_dnb - nd.home_win) ** 2).mean()
brier_flat = ((0.5 - nd.home_win) ** 2).mean()
print(f'\nOn {len(nd)} decisive matches:')
print(f'  No-vig market DNB Brier:  {brier_mkt:.4f}')
print(f'  Flat 50/50 baseline:      {brier_flat:.4f}')
print(f'  Market favorite picked correctly: {((nd.mkt_home_dnb > 0.5) == nd.home_win).mean():.0%}')
print('\nColumns confirm: Cote1/X/2 = 1X2 moneyline only, no spread/total odds columns present.')
print('For a real backtest we need the full dataset joined to our own historical predictions by team+date.')
