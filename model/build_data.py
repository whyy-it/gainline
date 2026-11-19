"""Assemble clean match table: rugbydata base (2018-Sep 2026) + web-sourced 2026 gap fill."""
import pandas as pd

INTL_COMPS = ['International Test Match', 'Six Nations', 'Rugby World Cup', 'The Rugby Championship',
              'Autumn Nations Series', 'Pacific Nations Cup', 'Rugby Europe Championship']
CLUB_COMPS = ['United Rugby Championship', 'Gallagher Premiership', 'French Top 14',
              'European Rugby Champions Cup', 'European Rugby Challenge Cup']

NORM = {
    # clubs
    'Bordeaux': 'Bordeaux-Begles', 'Bordeaux Begles': 'Bordeaux-Begles', 'Castres Olympique': 'Castres',
    'Clermont': 'Clermont Auvergne', 'Montpellier Herault': 'Montpellier', 'Stade Francais Paris': 'Stade Francais',
    'Stade Toulousain': 'Toulouse', 'Us Montauban': 'US Montauban', 'Bath Rugby': 'Bath', 'Bristol': 'Bristol Bears',
    'Bristol Rugby': 'Bristol Bears', 'Exeter Chiefs': 'Exeter', 'Gloucester Rugby': 'Gloucester',
    'Leicester': 'Leicester Tigers', 'Newcastle': 'Newcastle', 'Newcastle Falcons': 'Newcastle',
    'Newcastle Red Bulls': 'Newcastle', 'Northampton Saints': 'Northampton', 'Sale': 'Sale Sharks',
    'Benetton Treviso': 'Benetton', 'Benetton Rugby': 'Benetton', 'Cardiff Blues': 'Cardiff', 'Cardiff Rugby': 'Cardiff',
    'Glasgow': 'Glasgow Warriors', 'Gwent Dragons': 'Dragons', 'Zebre Parma': 'Zebre',
    # nations
    'Usa': 'United States of America', 'USA': 'United States of America', 'Czechia': 'Czech Republic',
    'British Irish Lions': 'British and Irish Lions',
}

def norm(t):
    return NORM.get(t, t)

def load_base():
    m = pd.read_parquet('../matches.parquet')
    m = m[m.completed.fillna(False) & m.home_score.notna()].copy()
    m['date'] = pd.to_datetime(m.date)
    m['home'] = m.home_team.map(norm); m['away'] = m.away_team.map(norm)
    m['kind'] = None
    m.loc[m.competition.isin(INTL_COMPS), 'kind'] = 'intl'
    m.loc[m.competition.isin(CLUB_COMPS), 'kind'] = 'club'
    m = m[m.kind.notna()]
    # audit fix: drop results recorded for clubs after they folded (Wasps, Worcester: autumn 2022)
    folded = (m.home.isin(['Wasps', 'Worcester Warriors']) | m.away.isin(['Wasps', 'Worcester Warriors'])) & (m.date >= '2022-10-15')
    m = m[~folded]
    m = m[~m.home.str.contains('XV|Waratahs|Reds|Brumbies|Force|Cambridge', regex=True) | (m.kind == 'club')]
    # dedupe: same unordered pair within 1 day -> prefer non-'International Test Match' row
    m['pair'] = m.apply(lambda r: '|'.join(sorted([r.home, r.away])), axis=1)
    m['pref'] = (m.competition == 'International Test Match').astype(int)
    m = m.sort_values(['pair', 'date', 'pref'])
    keep, last = [], {}
    for i, r in m.iterrows():
        k = r.pair
        if k in last and abs((r.date - last[k]).days) <= 1:
            continue
        last[k] = r.date; keep.append(i)
    m = m.loc[keep]
    return m[['date', 'kind', 'competition', 'city_played', 'home', 'away', 'home_score', 'away_score']].rename(
        columns={'home_score': 'hs', 'away_score': 'as_', 'city_played': 'city'})

def add_rows():
    rows = []
    def A(d, h, hs, as_, a, comp, kind='intl', neutral=False):
        rows.append(dict(date=pd.Timestamp(d), kind=kind, competition=comp, city=None, home=norm(h), away=norm(a),
                         hs=hs, as_=as_, neutral=neutral))
    NC = 'Nations Championship'
    # July Nations Championship (neutral where the "home" side wasn't at home)
    for d, h, hs, as_, a, n in [
        ('2026-07-04','New Zealand',34,32,'France',0),('2026-07-04','Japan',27,10,'Italy',0),
        ('2026-07-04','Australia',31,33,'Ireland',0),('2026-07-04','Wales',39,24,'Fiji',0),  # Cardiff -> Wales home
        ('2026-07-04','South Africa',45,21,'England',0),('2026-07-04','Argentina',38,47,'Scotland',0),
        ('2026-07-11','New Zealand',47,17,'Italy',0),('2026-07-11','Australia',26,42,'France',0),
        ('2026-07-11','Japan',20,36,'Ireland',1),('2026-07-11','England',73,8,'Fiji',0),  # Liverpool -> England home
        ('2026-07-11','South Africa',42,28,'Scotland',0),('2026-07-11','Argentina',35,21,'Wales',0),
        ('2026-07-18','New Zealand',40,21,'Ireland',0),('2026-07-18','Japan',15,42,'France',0),
        ('2026-07-18','Australia',57,10,'Italy',0),('2026-07-18','Scotland',33,17,'Fiji',0),  # Murrayfield
        ('2026-07-18','South Africa',43,0,'Wales',0),('2026-07-18','Argentina',24,31,'England',0),
        # Aug/Sep tests not in base
        ('2026-09-05','South Africa',29,24,'New Zealand',0),('2026-09-05','Argentina',28,28,'Australia',0),
        ('2026-09-12','South Africa',43,28,'New Zealand',1),('2026-09-27','Australia',42,38,'South Africa',0),
        # November Nations Championship R4-R5
        ('2026-11-06','Ireland',42,12,'Argentina',0),('2026-11-07','Italy',7,66,'South Africa',0),
        ('2026-11-07','Scotland',21,43,'New Zealand',0),('2026-11-07','Wales',28,21,'Japan',0),
        ('2026-11-07','France',54,17,'Fiji',0),('2026-11-08','England',45,22,'Australia',0),
        ('2026-11-13','France',21,19,'South Africa',0),('2026-11-14','Italy',14,19,'Argentina',0),
        ('2026-11-14','Wales',35,28,'New Zealand',0),('2026-11-14','England',78,15,'Japan',0),
        ('2026-11-14','Ireland',43,15,'Fiji',0),('2026-11-15','Scotland',15,12,'Australia',0)]:
        A(d, h, hs, as_, a, NC, neutral=bool(n))
    # 2026-27 club seasons (Sky Sports results pages)
    top14 = """2026-11-08|Vannes|23|14|Bordeaux-Begles
2026-11-08|La Rochelle|22|18|Bayonne
2026-11-07|Toulouse|28|18|Clermont Auvergne
2026-11-07|Castres|28|30|Racing 92
2026-11-07|Montpellier|21|19|Lyon
2026-11-07|Pau|63|26|Perpignan
2026-11-07|Toulon|43|23|Stade Francais
2026-11-01|Montpellier|69|10|Castres
2026-11-01|Pau|34|14|Toulouse
2026-10-31|Clermont Auvergne|37|30|Racing 92
2026-10-31|Bordeaux-Begles|19|14|Bayonne
2026-10-31|Lyon|48|25|Vannes
2026-10-31|Stade Francais|43|24|La Rochelle
2026-10-31|Perpignan|24|25|Toulon
2026-10-25|La Rochelle|19|25|Bordeaux-Begles
2026-10-24|Vannes|27|26|Clermont Auvergne
2026-10-24|Bayonne|27|31|Lyon
2026-10-24|Castres|16|8|Stade Francais
2026-10-24|Racing 92|17|20|Montpellier
2026-10-24|Toulouse|38|33|Perpignan
2026-10-24|Toulon|41|20|Pau
2026-10-11|Toulouse|45|34|Bayonne
2026-10-10|Stade Francais|30|22|Montpellier
2026-10-10|Lyon|16|28|La Rochelle
2026-10-10|Pau|35|26|Castres
2026-10-10|Perpignan|21|12|Vannes
2026-10-10|Toulon|29|15|Racing 92
2026-10-10|Clermont Auvergne|22|23|Bordeaux-Begles
2026-10-04|Montpellier|28|0|Toulon
2026-10-03|Bordeaux-Begles|51|28|Lyon
2026-10-03|Bayonne|35|32|Stade Francais
2026-10-03|La Rochelle|38|13|Clermont Auvergne
2026-10-03|Racing 92|55|25|Perpignan
2026-10-03|Vannes|40|43|Pau
2026-10-03|Castres|7|8|Toulouse
2026-09-27|Toulouse|40|18|Montpellier
2026-09-26|Perpignan|38|25|Bordeaux-Begles
2026-09-26|Clermont Auvergne|20|6|Castres
2026-09-26|Racing 92|17|20|Bayonne
2026-09-26|Stade Francais|31|10|Lyon
2026-09-26|Toulon|28|10|Vannes
2026-09-26|Pau|39|17|La Rochelle
2026-09-20|Bordeaux-Begles|40|10|Stade Francais
2026-09-19|Castres|29|27|Toulon
2026-09-19|Bayonne|46|24|Clermont Auvergne
2026-09-19|La Rochelle|10|20|Racing 92
2026-09-19|Lyon|51|25|Pau
2026-09-19|Montpellier|50|13|Perpignan
2026-09-19|Vannes|23|29|Toulouse
2026-09-13|Toulouse|48|12|Bordeaux-Begles
2026-09-12|Clermont Auvergne|25|16|Stade Francais
2026-09-12|Pau|70|28|Bayonne
2026-09-12|Perpignan|43|29|Castres
2026-09-12|Racing 92|31|48|Lyon
2026-09-12|Vannes|12|26|Montpellier
2026-09-12|Toulon|20|21|La Rochelle
2026-09-06|La Rochelle|30|27|Toulouse
2026-09-05|Bayonne|27|26|Toulon
2026-09-05|Castres|29|20|Vannes
2026-09-05|Lyon|40|36|Clermont Auvergne
2026-09-05|Montpellier|19|27|Pau
2026-09-05|Stade Francais|28|26|Perpignan
2026-09-05|Bordeaux-Begles|64|5|Racing 92"""
    prem = """2026-10-31|Exeter|28|31|Harlequins
2026-10-31|Saracens|39|31|Newcastle Red Bulls
2026-10-31|Bath|36|28|Sale Sharks
2026-10-31|Northampton|28|14|Gloucester
2026-10-30|Bristol Bears|16|22|Leicester Tigers
2026-10-25|Harlequins|33|15|Saracens
2026-10-24|Leicester Tigers|28|7|Northampton
2026-10-24|Bristol Bears|20|7|Exeter
2026-10-23|Gloucester|23|5|Bath
2026-10-23|Newcastle Red Bulls|17|3|Sale Sharks
2026-10-11|Exeter|34|28|Newcastle Red Bulls
2026-10-11|Sale Sharks|28|24|Harlequins
2026-10-10|Northampton|71|48|Bath
2026-10-10|Saracens|30|7|Bristol Bears
2026-10-09|Leicester Tigers|38|31|Gloucester
2026-10-04|Saracens|17|21|Sale Sharks
2026-10-03|Bristol Bears|41|26|Northampton
2026-10-03|Gloucester|48|0|Harlequins
2026-10-03|Newcastle Red Bulls|19|24|Leicester Tigers
2026-10-02|Bath|40|34|Exeter
2026-09-27|Leicester Tigers|38|40|Saracens
2026-09-26|Exeter|24|45|Gloucester
2026-09-26|Sale Sharks|24|34|Bristol Bears
2026-09-25|Harlequins|17|33|Bath
2026-09-25|Northampton|59|26|Newcastle Red Bulls"""
    urc = """2026-10-31|Stormers|33|7|Ulster
2026-10-31|Benetton Rugby|31|12|Edinburgh
2026-10-31|Bulls|54|26|Scarlets
2026-10-31|Dragons|35|14|Zebre
2026-10-31|Munster|3|15|Sharks
2026-10-31|Ospreys|0|26|Cardiff Rugby
2026-10-30|Connacht|5|50|Leinster
2026-10-30|Glasgow Warriors|29|20|Lions
2026-10-24|Bulls|34|26|Ulster
2026-10-24|Benetton Rugby|20|24|Glasgow Warriors
2026-10-24|Stormers|52|40|Scarlets
2026-10-24|Leinster|24|18|Munster
2026-10-24|Ospreys|31|14|Dragons
2026-10-23|Cardiff Rugby|21|36|Sharks
2026-10-23|Connacht|48|27|Zebre
2026-10-23|Edinburgh|31|10|Lions
2026-10-10|Bulls|57|15|Lions
2026-10-10|Stormers|39|22|Sharks
2026-10-10|Zebre|12|29|Edinburgh
2026-10-10|Scarlets|27|26|Benetton Rugby
2026-10-10|Ulster|26|26|Munster
2026-10-10|Leinster|29|5|Cardiff Rugby
2026-10-09|Dragons|31|38|Ospreys
2026-10-09|Glasgow Warriors|36|19|Connacht
2026-10-03|Lions|35|19|Ospreys
2026-10-03|Dragons|38|26|Scarlets
2026-10-03|Glasgow Warriors|33|33|Ulster
2026-10-03|Sharks|26|20|Leinster
2026-10-03|Munster|26|22|Bulls
2026-10-02|Benetton Rugby|19|22|Connacht
2026-10-02|Cardiff Rugby|40|27|Zebre
2026-10-02|Edinburgh|17|13|Stormers
2026-09-26|Lions|27|26|Leinster
2026-09-26|Sharks|41|24|Ospreys
2026-09-26|Munster|20|26|Glasgow Warriors
2026-09-26|Zebre|13|47|Bulls
2026-09-26|Scarlets|15|21|Cardiff Rugby
2026-09-25|Benetton Rugby|19|19|Dragons
2026-09-25|Connacht|15|29|Stormers
2026-09-25|Ulster|29|34|Edinburgh"""
    for blob, comp in [(top14, 'French Top 14'), (prem, 'Gallagher Premiership'), (urc, 'United Rugby Championship')]:
        for line in blob.strip().splitlines():
            d, h, hs, as_, a = line.split('|')
            A(d, h, int(hs), int(as_), a, comp, kind='club')
    return pd.DataFrame(rows)

def neutral_flags(df):
    """Intl: home advantage by default; neutral for RWC2023 non-host games, French/London venues for visitors, known neutral cities."""
    FR = {'Saint-Denis','Marseille','Lyon','Lille','Nice','Paris','Toulouse','Saint-Etienne','Nantes','Bordeaux',
          "Villeneuve d'Ascq",'Bayonne','Chambery'}
    OWN = {'France': FR, 'England': {'London'}, 'Romania': {'Bucharest'}, 'Spain': {'Villajoyosa','Madrid','Malaga','Valladolid'}}
    ALWAYS = {'Chicago','Dubai','Amsterdam','Baltimore','Las Vegas'}
    def f(r):
        if r.kind == 'club':
            return False
        if isinstance(r.neutral, (bool,)) or r.neutral in (True, False, 0, 1):
            if r.city is None or (isinstance(r.city, float)):
                return bool(r.neutral)
        c = r.city
        if c is None or (isinstance(c, float)):
            return False
        if c in ALWAYS: return True
        if c in FR and r.home != 'France': return True
        if c == 'London' and r.home != 'England': return True
        if c == 'Bucharest' and r.home != 'Romania': return True
        if c == 'Villajoyosa' and r.home != 'Spain': return True
        return False
    df['neutral'] = df.apply(f, axis=1)
    return df

if __name__ == '__main__':
    base = load_base(); base['neutral'] = None
    add = add_rows()
    df = pd.concat([base, add], ignore_index=True).sort_values('date').reset_index(drop=True)
    df = neutral_flags(df)
    df['hs'] = df.hs.astype(float); df['as_'] = df.as_.astype(float)
    df.to_parquet('../model/games.parquet')
    print(df.groupby('kind').size(), df.date.max())
    print(df[df.kind == 'intl'].neutral.mean())
    print(df[(df.kind=='intl')&df.neutral].tail(15)[['date','home','away','city']])
