import csv, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/q1_anchored_revision';FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
models=read(ROOT/'outputs/q1_final_rebuilt/q1_model_comparison.csv')
def growth(series,model):
    return float(next(r for r in models if r['series']==series and r['model']==model)['annual_growth'])

g={'N_M1':growth('Tourist_arrivals','M1'),'N_M2':growth('Tourist_arrivals','M2'),
   'I_M1':growth('Tourism_revenue','M1'),'I_M2':growth('Tourism_revenue','M2')}
actual={'N2024':2643.0,'I2024':221.0,'N2025':2691.0,'I2025':200.0}

validation=[]
for label,prefix,unit in [('旅游接待量','N','万人次'),('旅游综合收入','I','亿元')]:
    for scenario,model in [('正式锚定预测','M1'),('M2偏乐观敏感性','M2')]:
        rate=g[f'{prefix}_{model}']; pred=actual[f'{prefix}2024']*(1+rate); act=actual[f'{prefix}2025']
        validation.append([label,scenario,model,actual[f'{prefix}2024'],rate,pred,act,pred-act,abs(pred-act)/act,unit])

future=[]
for label,prefix,unit in [('旅游接待量','N','万人次'),('旅游综合收入','I','亿元')]:
    for scenario,model in [('正式锚定预测','M1'),('M2偏乐观敏感性','M2')]:
        rate=g[f'{prefix}_{model}']; base=actual[f'{prefix}2025']
        for h,yr in enumerate(range(2026,2031),1):
            future.append([label,scenario,model,yr,base,rate,base*(1+rate)**h,unit])

def write(name,header,data):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(data)
write('q1_anchored_2025_validation.csv',['indicator','scenario','growth_source','anchor_2024','annual_growth','prediction_2025','actual_2025_media','error','APE','unit'],validation)
write('q1_anchored_forecast_2026_2030.csv',['indicator','scenario','growth_source','year','anchor_2025','annual_growth','forecast','unit'],future)
write('q1_anchored_parameters.csv',['parameter','value','meaning'],[
    ['N_M1_growth',g['N_M1'],'正式游客量锚定增长率'],['I_M1_growth',g['I_M1'],'正式旅游收入锚定增长率'],
    ['N_M2_growth',g['N_M2'],'游客量偏乐观敏感性增长率'],['I_M2_growth',g['I_M2'],'旅游收入偏乐观敏感性增长率'],
    ['N_2024',actual['N2024'],'2025留出预测锚点'],['I_2024',actual['I2024'],'2025留出预测锚点'],
    ['N_2025',actual['N2025'],'2026—2030预测锚点；官方媒体补充'],['I_2025',actual['I2025'],'2026—2030预测锚点；官方媒体补充']])

def svg(path,title,hist_year,hist_val,rows,unit):
    W,H=900,520;L,R,T,B=85,35,55,70
    formal=[r for r in rows if r[1]=='正式锚定预测']; optimistic=[r for r in rows if r[1]=='M2偏乐观敏感性']
    years=[hist_year]+[int(r[3]) for r in formal]; f=[hist_val]+[float(r[6]) for r in formal]; o=[hist_val]+[float(r[6]) for r in optimistic]
    lo=min(f+o)*.90;hi=max(f+o)*1.08;sx=lambda x:L+(x-2025)/5*(W-L-R);sy=lambda y:T+(hi-y)/(hi-lo)*(H-T-B)
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:Microsoft YaHei,Arial;font-size:13px}}</style><text x="{W/2}" y="28" text-anchor="middle" font-size="19">{title}</text>']
    for j in range(6):
        v=lo+(hi-lo)*j/5;y=sy(v);p.append(f'<line x1="{L}" y1="{y}" x2="{W-R}" y2="{y}" stroke="#ddd"/><text x="{L-8}" y="{y+5}" text-anchor="end">{v:.0f}</text>')
    for x in years:p.append(f'<text x="{sx(x)}" y="{H-B+25}" text-anchor="middle">{x}</text>')
    for vals,color,label in [(f,'#1565C0','正式：M1增长率锚定'),(o,'#D84315','敏感性：M2增长率锚定')]:
        pts=' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(years,vals));p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x,y in zip(years,vals):p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
    p.append(f'<text transform="translate(20 {H/2}) rotate(-90)" text-anchor="middle">{unit}</text><text x="{L}" y="{H-15}" fill="#1565C0">正式：M1增长率锚定</text><text x="{L+210}" y="{H-15}" fill="#D84315">敏感性：M2增长率锚定</text></svg>')
    path.write_text(''.join(p),encoding='utf-8')

svg(FIG/'01_游客量锚定预测.svg','旅游接待量：以2025年为锚的2026—2030预测',2025,actual['N2025'],[r for r in future if r[0]=='旅游接待量'],'万人次')
svg(FIG/'02_旅游收入锚定预测.svg','旅游综合收入：以2025年为锚的2026—2030预测',2025,actual['I2025'],[r for r in future if r[0]=='旅游综合收入'],'亿元')
print(g)
