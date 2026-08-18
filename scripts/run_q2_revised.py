import csv, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'outputs/q1_final_rebuilt/q1_model_input.csv'
OUT=ROOT/'outputs/q2_revised'; FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)

with IN.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
def val(x):
    try:return float(x)
    except:return np.nan
year=np.array([int(r['year']) for r in rows])
N_main=np.array([val(r['Arrivals_main']) for r in rows]); N_spline=np.array([val(r['Arrivals_natural_spline']) for r in rows])
I=np.array([val(r['Revenue_model']) for r in rows])
# 2025 tourism values are source-tiered media supplements used only as quasi holdout / forecast origin.
N_main[-1]=2691.0; N_spline[-1]=2691.0; I[-1]=200.0

def drift_fit(y, end_year):
    z=np.log(y[year<=end_year]); d=np.diff(z); c=float(d.mean()); resid=d-c
    return {'c':c,'g':math.exp(c)-1,'resid':resid,'years':year[year<=end_year][1:]}

def holt_damped_fit(y,end_year):
    z=np.log(y[year<=end_year]); best=None
    for a in np.arange(.05,1.001,.05):
      for b in np.arange(.0,1.001,.05):
       for phi in np.arange(.70,1.001,.05):
        l=z[0]; trend=z[1]-z[0]; sse=0.0
        for t in range(1,len(z)):
            pred=l+phi*trend; e=z[t]-pred; sse+=e*e
            nl=a*z[t]+(1-a)*(l+phi*trend); nt=b*(nl-l)+(1-b)*phi*trend; l,trend=nl,nt
        if best is None or sse<best['sse']:best={'a':float(a),'b':float(b),'phi':float(phi),'level':float(l),'trend':float(trend),'sse':float(sse)}
    return best
def holt_forecast(m,h): return math.exp(m['level']+m['trend']*sum(m['phi']**j for j in range(1,h+1)))

def rolling_metrics(y, method, targets):
    e=[]
    for yy in targets:
        if method=='arima': pred=y[year==yy-1][0]*math.exp(drift_fit(y,yy-1)['c'])
        else: pred=holt_forecast(holt_damped_fit(y,yy-1),1)
        act=y[year==yy][0];e.append((act-pred,abs(act-pred)/act))
    return math.sqrt(sum(x*x for x,_ in e)/len(e)),100*sum(p for _,p in e)/len(e)

def bootstrap(y, scenario, nsim=200000):
    m=drift_fit(y,2025); resid=m['resid'].copy(); yrs=m['years']
    if scenario=='baseline': resid=resid[yrs!=2020]
    # Residual bootstrap requires zero-mean innovations. Removing the extreme
    # 2020 residual changes the sample mean, so re-center the filtered pool.
    mean_before=float(resid.mean())
    resid=resid-mean_before
    mean_after=float(resid.mean())
    rng=np.random.default_rng(20260818 if scenario=='baseline' else 20260819)
    shocks=rng.choice(resid,size=(nsim,5),replace=True); cs=np.cumsum(shocks,axis=1)
    h=np.arange(1,6); paths=np.exp(math.log(y[-1])+h*m['c']+cs)
    center=y[-1]*np.exp(h*m['c']); lo=np.quantile(paths,.025,axis=0); hi=np.quantile(paths,.975,axis=0)
    audit={'scenario':scenario,'removed_2020':scenario=='baseline','residual_count':len(resid),'residual_mean_before_centering':mean_before,'residual_mean_after_centering':mean_after}
    return m,center,lo,hi,audit

def covid_analysis(y):
    d=np.diff(np.log(y)); yrs=year[1:]; D=((yrs>=2020)&(yrs<=2022)).astype(float);X=np.column_stack([np.ones(len(d)),D]);b=np.linalg.lstsq(X,d,rcond=None)[0]
    c,beta=map(float,b); cf=[]; base=y[year==2019][0]
    for yy in [2020,2021,2022]:
        base*=math.exp(c); actual=y[year==yy][0];cf.append([yy,actual,base,1-actual/base])
    return c,beta,math.exp(beta)-1,cf

forecasts=[]; validation=[]; sensitivity=[]; covid_rows=[]; bootstrap_audit=[]; summaries={}
for key,y in [('游客接待量_主插补',N_main),('旅游综合收入',I)]:
    train=drift_fit(y,2024); p25=y[year==2024][0]*math.exp(train['c']);ape=abs(y[-1]-p25)/y[-1]
    hm=holt_damped_fit(y,2024);hp25=holt_forecast(hm,1);hape=abs(y[-1]-hp25)/y[-1]
    for meth in ['arima','holt']:
        nr=rolling_metrics(y,meth,[2017,2018,2019]); rr=rolling_metrics(y,meth,[2023,2024]); pp=p25 if meth=='arima' else hp25; pa=ape if meth=='arima' else hape
        validation.append([key,'ARIMA(0,1,0)+漂移' if meth=='arima' else 'Holt阻尼趋势',nr[0],nr[1],rr[0],rr[1],pp,y[-1],pa])
    ca,bet,impact,cf=covid_analysis(y)
    for rr in cf:covid_rows.append([key,bet,impact,*rr])
    summaries[key]={'holdout_prediction':p25,'holdout_actual':y[-1],'holdout_APE':ape,'drift_train_2010_2024':train['c'],'holt_2025':hp25,'holt_APE':hape,'covid_beta':bet,'covid_growth_effect':impact}
    for scen in ['baseline','stress']:
        full,center,lo,hi,audit=bootstrap(y,scen)
        bootstrap_audit.append([key,audit['scenario'],audit['removed_2020'],audit['residual_count'],audit['residual_mean_before_centering'],audit['residual_mean_after_centering']])
        for i,yy in enumerate(range(2026,2031)):forecasts.append([key,scen,yy,center[i],lo[i],hi[i],full['c'],full['g']])

for label,y in [('主插补',N_main),('自然样条',N_spline)]:
    full,center,lo,hi,audit=bootstrap(y,'baseline')
    for i,yy in enumerate(range(2026,2031)):sensitivity.append([label,yy,center[i],lo[i],hi[i],full['c'],full['g']])

def write(name,header,data):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:w=csv.writer(f);w.writerow(header);w.writerows(data)
write('q2_model_validation.csv',['indicator','model','normal_RMSE','normal_MAPE_pct','recovery_RMSE','recovery_MAPE_pct','forecast_2025','actual_2025_media','APE_2025'],validation)
write('q2_forecast_2026_2030.csv',['indicator','scenario','year','point_forecast','PI95_low','PI95_high','log_drift','annual_growth'],forecasts)
write('q2_arrivals_imputation_sensitivity.csv',['imputation','year','point_forecast','PI95_low','PI95_high','log_drift','annual_growth'],sensitivity)
write('q2_covid_counterfactual.csv',['indicator','covid_beta_on_dlog','growth_effect','year','actual_or_model_value','no_covid_counterfactual','level_gap_ratio'],covid_rows)
write('q2_bootstrap_residual_audit.csv',['indicator','scenario','removed_2020','residual_count','residual_mean_before_centering','residual_mean_after_centering'],bootstrap_audit)

def svg_forecast(path,title,hist,rowsf,unit):
    W,H=900,520;L,R,T,B=85,35,55,65; fy=np.array([r[2] for r in rowsf]);point=np.array([r[3] for r in rowsf]);lo=np.array([r[4] for r in rowsf]);hi=np.array([r[5] for r in rowsf]);x=np.r_[year,fy];vv=np.r_[hist,lo,hi];mn=max(0,float(vv.min())*.85);mx=float(vv.max())*1.08;sx=lambda v:L+(v-2010)/20*(W-L-R);sy=lambda v:T+(mx-v)/(mx-mn)*(H-T-B)
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:Microsoft YaHei,Arial;font-size:13px}}</style><text x="{W/2}" y="28" text-anchor="middle" font-size="19">{title}</text>']
    for j in range(6):v=mn+(mx-mn)*j/5;yy=sy(v);p.append(f'<line x1="{L}" y1="{yy}" x2="{W-R}" y2="{yy}" stroke="#ddd"/><text x="{L-8}" y="{yy+5}" text-anchor="end">{v:.0f}</text>')
    for yy in range(2010,2031,2):p.append(f'<text x="{sx(yy)}" y="{H-B+23}" text-anchor="middle">{yy}</text>')
    poly=' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(np.r_[2025,fy],np.r_[hist[-1],point])); band=' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(fy,lo))+' '+' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(fy[::-1],hi[::-1])); hp=' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(year,hist))
    p+= [f'<polygon points="{band}" fill="#90CAF9" opacity=".45"/><polyline points="{hp}" fill="none" stroke="#1565C0" stroke-width="3"/><polyline points="{poly}" fill="none" stroke="#D84315" stroke-width="3"/><line x1="{sx(2025.5)}" y1="{T}" x2="{sx(2025.5)}" y2="{H-B}" stroke="#777" stroke-dasharray="5 4"/><text transform="translate(20 {H/2}) rotate(-90)" text-anchor="middle">{unit}</text><text x="{L}" y="{H-12}" fill="#1565C0">历史/建模序列</text><text x="{L+150}" y="{H-12}" fill="#D84315">基准点预测</text><text x="{L+280}" y="{H-12}" fill="#4078A8">95%基准区间</text></svg>']
    path.write_text(''.join(p),encoding='utf-8')

for key,hist,file,title,unit in [('游客接待量_主插补',N_main,'01_游客量基准预测.svg','旅游接待量：2026—2030基准预测','万人次'),('旅游综合收入',I,'02_旅游收入基准预测.svg','旅游综合收入：2026—2030基准预测','亿元')]:
    rr=[r for r in forecasts if r[0]==key and r[1]=='baseline'];svg_forecast(FIG/file,title,hist,rr,unit)

(OUT/'q2_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summaries,ensure_ascii=False,indent=2))
