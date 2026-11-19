// ---------- team identity: colours (kit-based) and flags ----------
const TC = {
 'England':['#c8102e','#1d2a5c'], 'New Zealand':['#111111','#8a8d91','#e9ecef'], 'Scotland':['#1b2a5c','#6b8fd6'],
 'Japan':['#c8102e','#1b2a5c'], 'Ireland':['#169b62','#ff883e'], 'South Africa':['#e0a91b','#007a4d'],
 'Italy':['#1f6fd1','#0b2d5e'], 'Fiji':['#3a8fc4','#111111'], 'France':['#1b3a8c','#c8102e'], 'Argentina':['#74acdf','#1b3a8c'],
 'Wales':['#c8102e','#00a34a'], 'Australia':['#f2c300','#00843d'],
 'Toulouse':['#c8102e','#111'],'Bordeaux-Begles':['#7a1f3d','#fff'],'La Rochelle':['#f2c300','#111'],'Racing 92':['#6cb4e4','#fff'],
 'Stade Francais':['#e5007e','#1b2a5c'],'Toulon':['#c8102e','#111'],'Clermont Auvergne':['#f2c300','#1b2a5c'],'Montpellier':['#1b3a8c','#6cb4e4'],
 'Pau':['#00843d','#fff'],'Castres':['#1b3a8c','#fff'],'Bayonne':['#6cb4e4','#fff'],'Lyon':['#c8102e','#111'],'Perpignan':['#c8102e','#f2c300'],'Vannes':['#111','#2c6fbf'],
 'Bath':['#1b2a5c','#6cb4e4'],'Bristol Bears':['#1b2a5c','#c8102e'],'Exeter':['#111','#fff'],'Gloucester':['#a0162c','#fff'],'Harlequins':['#c2186b','#6cb4e4'],
 'Leicester Tigers':['#00754a','#c8102e'],'Newcastle':['#c8102e','#1b2a5c'],'Northampton':['#00513a','#c9a227'],'Sale Sharks':['#1b3a8c','#fff'],'Saracens':['#111','#c8102e'],
 'Leinster':['#1d4f91','#fff'],'Munster':['#c8102e','#fff'],'Ulster':['#ececec','#c8102e'],'Connacht':['#00843d','#111'],'Bulls':['#1d4f91','#fff'],
 'Stormers':['#1d4f91','#fff'],'Sharks':['#111','#fff'],'Lions':['#c8102e','#fff'],'Glasgow Warriors':['#1b2a5c','#f2a900'],'Edinburgh':['#7a1f3d','#111'],
 'Blues':['#1d4f91','#fff'],'Chiefs':['#c8102e','#f2c300'],'Crusaders':['#111','#c8102e'],'Highlanders':['#7a1f3d','#f2c300'],'Hurricanes':['#f2c300','#111'],'Moana Pasifika':['#00808a','#fff'],'Fijian Drua':['#0a9bd6','#fff'],'Brumbies':['#0b2d5e','#f2c300'],'Waratahs':['#6cb4e4','#1b2a5c'],'Reds':['#a6192e','#fff'],'Force':['#2b4ea2','#fff'],
 'Benetton':['#00843d','#fff'],'Zebre':['#111','#f2c300'],'Cardiff':['#1d4f91','#111'],'Ospreys':['#111','#fff'],'Scarlets':['#c8102e','#fff'],'Dragons':['#c8102e','#111']
};
const isDark = () => document.documentElement.dataset.theme==='dark' || (document.documentElement.dataset.theme!=='light' && matchMedia('(prefers-color-scheme: dark)').matches);
function tcol(t){const c=TC[t]||['#5d6b64','#999'];return (isDark()&&c[2])?c[2]:c[0];}
function rgb(h){h=h.replace('#','');if(h.length===3)h=h.split('').map(x=>x+x).join('');return [0,2,4].map(i=>parseInt(h.substr(i,2),16));}
function dist(a,b){const x=rgb(a),y=rgb(b);return Math.hypot(x[0]-y[0],x[1]-y[1],x[2]-y[2]);}
function pairCols(h,a){const ch=tcol(h);let ca=tcol(a);if(dist(ch,ca)<120){ca=(TC[a]||[])[1]||'#888';}return [ch,ca];}
function ink(bg){const [r,g,b]=rgb(bg);return (0.299*r+0.587*g+0.114*b)>150?'#111':'#fff';}
const UJ = '<g transform="scale(.25 .3334)"><rect width="60" height="30" fill="#012169"/><path d="M0,0 60,30M60,0 0,30" stroke="#fff" stroke-width="6"/><path d="M0,0 60,30M60,0 0,30" stroke="#C8102E" stroke-width="2"/><path d="M30,0v30M0,15h60" stroke="#fff" stroke-width="10"/><path d="M30,0v30M0,15h60" stroke="#C8102E" stroke-width="6"/></g>';
function star(cx,cy,r,fill,stroke){let p='';for(let i=0;i<10;i++){const a=Math.PI/5*i-Math.PI/2,rr=i%2?r*.42:r;p+=(i?'L':'M')+(cx+rr*Math.cos(a)).toFixed(2)+','+(cy+rr*Math.sin(a)).toFixed(2);}return `<path d="${p}Z" fill="${fill}"${stroke?` stroke="${stroke}" stroke-width=".5"`:''}/>`;}
const FLAG = {
 'England':'<rect width="30" height="20" fill="#fff"/><path d="M15,0v20M0,10h30" stroke="#CE1124" stroke-width="4"/>',
 'Scotland':'<rect width="30" height="20" fill="#005EB8"/><path d="M0,0 30,20M30,0 0,20" stroke="#fff" stroke-width="3.2"/>',
 'Wales':'<rect width="30" height="10" fill="#fff"/><rect y="10" width="30" height="10" fill="#00B140"/><path d="M7,14 C8,9 12,7 15,8.5 L17.5,5 L17.8,8.6 C20.5,8 23,9.5 23,12 L21,11.6 L20.5,14 L18.6,12.8 L17,15.5 L15.5,13.4 L13,15.8 L12,13.6 L9.5,15.6 Z" fill="#D30731"/>',
 'Ireland':'<rect width="10" height="20" fill="#169B62"/><rect x="10" width="10" height="20" fill="#fff"/><rect x="20" width="10" height="20" fill="#FF883E"/>',
 'France':'<rect width="10" height="20" fill="#002395"/><rect x="10" width="10" height="20" fill="#fff"/><rect x="20" width="10" height="20" fill="#ED2939"/>',
 'Italy':'<rect width="10" height="20" fill="#009246"/><rect x="10" width="10" height="20" fill="#fff"/><rect x="20" width="10" height="20" fill="#CE2B37"/>',
 'Japan':'<rect width="30" height="20" fill="#fff"/><circle cx="15" cy="10" r="6" fill="#BC002D"/>',
 'Argentina':'<rect width="30" height="20" fill="#74ACDF"/><rect y="6.67" width="30" height="6.67" fill="#fff"/><circle cx="15" cy="10" r="2.2" fill="#F6B40E" stroke="#85340A" stroke-width=".3"/>',
 'South Africa':'<rect width="30" height="10" fill="#E03C31"/><rect y="10" width="30" height="10" fill="#001489"/><path d="M0,0 12,10 0,20M12,10H30" stroke="#fff" stroke-width="6.6" fill="none"/><path d="M0,0 12,10 0,20M12,10H30" stroke="#007749" stroke-width="4" fill="none"/><path d="M0,2.6 9.2,10 0,17.4Z" fill="#FFB81C"/><path d="M0,4.4 7,10 0,15.6Z" fill="#000"/>',
 'New Zealand':'<rect width="30" height="20" fill="#012169"/>'+'UJ'+star(22,5.2,1.4,'#C8102E','#fff')+star(25,9.4,1.4,'#C8102E','#fff')+star(19.6,10.4,1.2,'#C8102E','#fff')+star(22,15.4,1.6,'#C8102E','#fff'),
 'Australia':'<rect width="30" height="20" fill="#012169"/>'+'UJ'+star(7.5,15,2.4,'#fff')+star(22.5,16.2,1.4,'#fff')+star(18.6,9,1.3,'#fff')+star(22.5,3.6,1.3,'#fff')+star(26.2,7.8,1.3,'#fff')+star(24.4,11,.7,'#fff'),
 'Fiji':'<rect width="30" height="20" fill="#68BFE5"/>'+'UJ'+'<path d="M18.5,5h7v6.5c0,2.6-3.5,4-3.5,4s-3.5-1.4-3.5-4Z" fill="#fff" stroke="#C8102E" stroke-width=".4"/><rect x="18.5" y="5" width="7" height="2.2" fill="#C8102E"/>'
};
function flag(t,h=16){if(!FLAG[t])return badge(t,h+4);return `<svg class="flag" viewBox="0 0 30 20" width="${h*1.5}" height="${h}" aria-hidden="true">${FLAG[t].replace('UJ',UJ)}</svg>`;}
const AB={'Blues':'BLU','Chiefs':'CHI','Crusaders':'CRU','Highlanders':'HIG','Hurricanes':'HUR','Moana Pasifika':'MOA','Fijian Drua':'DRU','Brumbies':'BRU','Waratahs':'WAR','Reds':'RED','Force':'FOR','La Rochelle':'LR','Racing 92':'R92','Bordeaux-Begles':'UBB','Stade Francais':'SF','Clermont Auvergne':'ASM','Leicester Tigers':'LEI','Bristol Bears':'BRI','Glasgow Warriors':'GLA','Sale Sharks':'SAL','Harlequins':'HQ','Northampton':'NOR','Saracens':'SAR','Gloucester':'GLO','Newcastle':'NEW','Exeter':'EXE','Bath':'BAT','Toulouse':'ST','Toulon':'RCT','Montpellier':'MHR','Castres':'CO','Bayonne':'AB','Lyon':'LOU','Perpignan':'USAP','Vannes':'RCV','Pau':'SP','Leinster':'LEI','Munster':'MUN','Ulster':'ULS','Connacht':'CON','Bulls':'BUL','Stormers':'STO','Sharks':'SHA','Lions':'LIO','Edinburgh':'EDI','Benetton':'BEN','Zebre':'ZEB','Cardiff':'CAR','Ospreys':'OSP','Scarlets':'SCA','Dragons':'DRA'};
function badge(t,s=22){const c=TC[t]||['#5d6b64','#fff'];const ab=AB[t]||t.replace(/-/g,' ').split(' ').filter(w=>!['de','des','la','le','rugby'].includes(w.toLowerCase())).map(w=>w[0]).join('').slice(0,3).toUpperCase();
  return `<svg class="badge" viewBox="0 0 24 24" width="${s}" height="${s}" aria-hidden="true"><rect x="1" y="1" width="22" height="22" rx="6" fill="${c[0]}" stroke="${c[1]}" stroke-width="2"/><text x="12" y="15.5" text-anchor="middle" font-family="Barlow Condensed,Arial Narrow,sans-serif" font-weight="700" font-size="${ab.length>3?6.8:ab.length>2?8.5:10}" fill="${ink(c[0])}">${ab}</text></svg>`;}

