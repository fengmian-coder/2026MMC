import csv, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'data/processed/core_annual_clean_strict_2010_2025.csv'
OUT=ROOT/'outputs/q1_final_rebuilt'
FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)

with IN.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
def num(s):
    try:return float(s)
    except:return np.nan
for r in rows:
    for k in ['gdp_official','tertiary_official','tourist_arrivals_official','tourism_revenue_official','tourist_arrivals_non_strict_supplement','tourism_revenue_non_strict_supplement']:
        r[k]=num(r[k])
    r['year']=int(r['year'])

years=np.array([r['year'] for r in rows]); tau=years-2010
G=np.array([r['gdp_official'] for r in rows]); S=np.array([r['tertiary_official'] for r in rows])
N=np.array([r['tourist_arrivals_official'] for r in rows]); I=np.array([r['tourism_revenue_official'] for r in rows])
# Comparable-price bridge at the 2019 accounting-definition break.
qG=(G[8]*1.021)/G[9]; qS=(S[8]*1.044)/S[9]
Gadj=G.copy(); Sadj=S.copy(); Gadj[9:]*=qG; Sadj[9:]*=qS
# Per-capita tourism income (yuan/person), linearly interpolated between official 2020 and 2023 endpoints.
C=I*10000/N
C20=C[10]; C23=C[13]
Nmain=N.copy()
for idx in [11,12]:
    ci=C20+(C23-C20)*(years[idx]-2020)/3
    Nmain[idx]=I[idx]*10000/ci
# Natural-spline sensitivity on observed per-capita-income series.
obs=np.isfinite(C)
def natural_spline(x,y,xq):
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(x); h=np.diff(x)
    A=np.zeros((n,n)); b=np.zeros(n); A[0,0]=A[-1,-1]=1
    for i in range(1,n-1):
        A[i,i-1]=h[i-1]; A[i,i]=2*(h[i-1]+h[i]); A[i,i+1]=h[i]
        b[i]=6*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
    m=np.linalg.solve(A,b); j=np.searchsorted(x,xq)-1; j=max(0,min(j,n-2)); hj=x[j+1]-x[j]
    return m[j]*(x[j+1]-xq)**3/(6*hj)+m[j+1]*(xq-x[j])**3/(6*hj)+(y[j]-m[j]*hj**2/6)*(x[j+1]-xq)/hj+(y[j+1]-m[j+1]*hj**2/6)*(xq-x[j])/hj
Nspline=N.copy(); Nspline[11]=I[11]*10000/natural_spline(years[obs],C[obs],2021); Nspline[12]=I[12]*10000/natural_spline(years[obs],C[obs],2022)
# Revenue 2010: retain earlier model estimate only as transparent model-layer input.
Imain=I.copy(); Imain[0]=32.88
N2025=rows[-1]['tourist_arrivals_non_strict_supplement']; I2025=rows[-1]['tourism_revenue_non_strict_supplement']
Nmodel=Nmain.copy(); Nmodel[-1]=N2025
Imodel=Imain.copy(); Imodel[-1]=I2025

pair=np.isfinite(N)&np.isfinite(I)
def pearson(x,y):
    rr=float(np.corrcoef(x,y)[0,1]); z=abs(np.arctanh(rr))*math.sqrt(len(x)-3); return rr,math.erfc(z/math.sqrt(2))
r,p=pearson(N[pair],I[pair])
pair_sec=pair.copy(); pair_sec[0]=True
r_sec,p_sec=pearson(np.where(np.isfinite(N),N,np.nan)[pair_sec],Imain[pair_sec])

def ols(y, kind, train):
    t=tau[train]; dc=((years[train]>=2020)&(years[train]<=2022)).astype(float); dr=((years[train]>=2023)&(years[train]<=2024)).astype(float)
    cols=[np.ones(len(t)),t]
    if kind in ('M1','M2'): cols.append(dc)
    if kind=='M2': cols.append(dr)
    X=np.column_stack(cols); ly=np.log(y[train]); b=np.linalg.lstsq(X,ly,rcond=None)[0]
    e=ly-X@b; n=len(ly); k=len(b); sse=e@e; sig2=sse/(n-k); cov=sig2*np.linalg.inv(X.T@X); se=np.sqrt(np.diag(cov)); tv=b/se; pv=np.array([math.erfc(abs(v)/math.sqrt(2)) for v in tv])
    smear=float(np.mean(np.exp(e))); ll=-n/2*(math.log(2*math.pi)+1+math.log(sse/n)); aic=2*k-2*ll; aicc=aic+2*k*(k+1)/(n-k-1) if n>k+1 else np.inf; bic=k*math.log(n)-2*ll
    return dict(kind=kind,b=b,se=se,t=tv,p=pv,smear=smear,aicc=aicc,bic=bic,sigma=math.sqrt(sig2),df=n-k,cov=cov,resid=e,train=train)
def pred(m, yy):
    t=yy-2010; cols=[1,t]
    if m['kind'] in ('M1','M2'): cols.append(1.0 if 2020<=yy<=2022 else 0.0)
    if m['kind']=='M2': cols.append(1.0 if 2023<=yy<=2024 else 0.0)
    x=np.array(cols); mu=x@m['b']; se=math.sqrt(m['sigma']**2*(1+x@m['cov']@x)); crit=2.2 if m['df']<15 else 2.0
    return m['smear']*math.exp(mu), m['smear']*math.exp(mu-crit*se), m['smear']*math.exp(mu+crit*se)

series={'GDP_adjusted':Gadj,'Tertiary_adjusted':Sadj,'Tourist_arrivals':Nmodel,'Tourism_revenue':Imodel}
models=[]; fits={}; hold=[]
for name,y in series.items():
    m0=ols(y,'M0',(years<=2019)&np.isfinite(y))
    m1=ols(y,'M1',(years<=2024)&np.isfinite(y))
    m2=ols(y,'M2',(years<=2024)&np.isfinite(y))
    chosen=m2 if (m2['p'][-1]<.05 and m2['aicc']<m1['aicc']) else m1
    fits[name]=(m0,m1,m2,chosen)
    for m in [m0,m1,m2]:
        models.append([name,m['kind'],len(m['b']),m['aicc'],m['bic'],math.exp(m['b'][1])-1,m['smear'],m['p'][-1] if m['kind']=='M2' else np.nan,'chosen' if m is chosen else ''])
    pr,lo,hi=pred(chosen,2025); actual=y[-1]; hold.append([name,chosen['kind'],actual,pr,lo,hi,actual-pr,abs(actual-pr)/actual if np.isfinite(actual) else np.nan])

def write_csv(path,head,data):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(head);w.writerows(data)
processed=[]
for i,yr in enumerate(years): processed.append([yr,G[i],Gadj[i],S[i],Sadj[i],N[i],Nmain[i],Nspline[i],I[i],Imain[i],('official' if np.isfinite(N[i]) else ('per_capita_linear_imputed' if yr in (2021,2022) else 'media_holdout' if yr==2025 else 'missing')),('official' if np.isfinite(I[i]) else ('model_estimate' if yr==2010 else 'media_holdout' if yr==2025 else 'missing'))])
write_csv(OUT/'q1_model_input.csv',['year','GDP_raw','GDP_adjusted','Tertiary_raw','Tertiary_adjusted','Arrivals_official','Arrivals_main','Arrivals_natural_spline','Revenue_official','Revenue_model','Arrivals_status','Revenue_status'],processed)
write_csv(OUT/'q1_correlation_imputation.csv',['item','value','note'],[
 ['Pearson_r_strict_official',r,'12 official paired years, 2011-2020 and 2023-2024'],['Pearson_p_strict_official',p,'two-sided'],['Pearson_r_with_2010_secondary',r_sec,'adds 2010 revenue 35.5, not strict-government'],['N2021_per_capita_linear',Nmain[11],'main'],['N2022_per_capita_linear',Nmain[12],'main'],['N2021_natural_spline',Nspline[11],'sensitivity'],['N2022_natural_spline',Nspline[12],'sensitivity'],['GDP_2019_bridge_factor',qG,'post-2019 scaled to 2018 comparable-price growth'],['Tertiary_2019_bridge_factor',qS,'post-2019 scaled']])
write_csv(OUT/'q1_model_comparison.csv',['series','model','k','AICc','BIC','annual_growth','Duan_smearing','recovery_dummy_p','selected'],models)
write_csv(OUT/'q1_2025_holdout.csv',['series','selected_model','actual_2025','prediction','PI95_low','PI95_high','error','APE'],hold)

def svg_plot(path,title,x,series,ylabel):
    W,H=900,540; L,R,T,B=90,30,65,65; vals=np.concatenate([np.asarray(v,float)[np.isfinite(v)] for _,v,_ in series]); ymin,ymax=float(vals.min()),float(vals.max()); pad=(ymax-ymin)*.08 or 1; ymin-=pad;ymax+=pad
    sx=lambda v:L+(float(v)-min(x))/(max(x)-min(x))*(W-L-R); sy=lambda v:T+(ymax-float(v))/(ymax-ymin)*(H-T-B)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:Microsoft YaHei,Arial;font-size:14px}}</style><text x="{W/2}" y="32" text-anchor="middle" font-size="20">{title}</text>']
    for j in range(6):
        yy=ymin+(ymax-ymin)*j/5; py=sy(yy);parts.append(f'<line x1="{L}" y1="{py}" x2="{W-R}" y2="{py}" stroke="#ddd"/><text x="{L-8}" y="{py+5}" text-anchor="end">{yy:.0f}</text>')
    parts.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/><line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/><text transform="translate(20 {H/2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    for yr in x[::2]: parts.append(f'<text x="{sx(yr)}" y="{H-B+25}" text-anchor="middle">{int(yr)}</text>')
    for si,(label,v,color) in enumerate(series):
        pts=' '.join(f'{sx(xx):.1f},{sy(yy):.1f}' for xx,yy in zip(x,v) if np.isfinite(yy));parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for xx,yy in zip(x,v):
            if np.isfinite(yy):parts.append(f'<circle cx="{sx(xx)}" cy="{sy(yy)}" r="4" fill="{color}"/>')
        parts.append(f'<line x1="{L+si*230}" y1="{H-18}" x2="{L+si*230+28}" y2="{H-18}" stroke="{color}" stroke-width="3"/><text x="{L+si*230+36}" y="{H-13}">{label}</text>')
    parts.append('</svg>');path.write_text(''.join(parts),encoding='utf-8')

# Standalone SVG figures are vector, editable, and do not combine multiple panels.
ordr=np.argsort(I[pair]); z=np.polyfit(I[pair],N[pair],1); xx=np.linspace(min(I[pair]),max(I[pair]),100)
svg_plot(FIG/'01_pearson_scatter.svg',f'游客量与旅游收入：严格官方配对 r={r:.3f}',xx,[('线性拟合',np.polyval(z,xx),'#D84315')],'接待游客（万人次）')
svg_plot(FIG/'02_arrivals_imputation.svg','接待游客量：主插补与敏感性',years,[('主序列',Nmain,'#1565C0'),('自然样条',Nspline,'#2E7D32')],'万人次')
for idx,(name,y,title,unit) in enumerate([('GDP_adjusted',Gadj,'地区生产总值','亿元'),('Tertiary_adjusted',Sadj,'第三产业增加值','亿元'),('Tourist_arrivals',Nmodel,'接待游客量','万人次'),('Tourism_revenue',Imodel,'旅游综合收入','亿元')],3):
    chosen=fits[name][3]; predv=np.array([pred(chosen,int(yy))[0] for yy in years]);
    svg_plot(FIG/f'{idx:02d}_{name}_model.svg',f'{title}：阶段指数模型与2025留出预测',years,[('建模/留出数据',y,'#1565C0'),(f'{chosen["kind"]} 指数模型',predv,'#D84315')],unit)
summary={'pearson_strict':r,'pearson_strict_p':p,'pearson_with_2010_secondary':r_sec,'N2021_main':Nmain[11],'N2022_main':Nmain[12],'N2021_spline':Nspline[11],'N2022_spline':Nspline[12],'bridge_G':qG,'bridge_S':qS,'holdout':hold}
(OUT/'q1_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
