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

def adf_critical_values(regression,nobs):
    # MacKinnon (2010) response-surface coefficients, N=1.
    coeff={
      'c':{'1%':[-3.43035,-6.5393,-16.786,-79.433],'5%':[-2.86154,-2.8903,-4.234,-40.040],'10%':[-2.56677,-1.5384,-2.809,-3.480]},
      'ct':{'1%':[-3.95877,-9.0531,-28.428,-134.155],'5%':[-3.41049,-4.3904,-9.036,-45.374],'10%':[-3.12705,-2.5856,-3.925,-22.380]}}
    z=1.0/nobs
    return {k:float(sum(a*z**i for i,a in enumerate(v))) for k,v in coeff[regression].items()}

def adf_test(x,regression,maxlag=2):
    x=np.asarray(x,float); dy=np.diff(x); candidates=[]
    for lag in range(maxlag+1):
        yy=dy[lag:]; cols=[x[lag:-1],np.ones(len(yy))]
        if regression=='ct': cols.append(np.arange(lag+2,len(x)+1,dtype=float))
        for j in range(1,lag+1):cols.append(dy[lag-j:-j])
        X=np.column_stack(cols); beta=np.linalg.lstsq(X,yy,rcond=None)[0]; e=yy-X@beta
        n=len(yy); k=X.shape[1]; sse=float(e@e); bic=n*math.log(sse/n)+k*math.log(n)
        sigma2=sse/(n-k); se=math.sqrt(sigma2*np.linalg.inv(X.T@X)[0,0]); stat=float(beta[0]/se)
        candidates.append((bic,lag,n,stat))
    _,lag,nobs,stat=min(candidates,key=lambda z:z[0]); crit=adf_critical_values(regression,nobs)
    decision='拒绝单位根，序列平稳' if stat<crit['5%'] else '不能拒绝单位根，序列非平稳'
    return stat,lag,nobs,crit,decision

def kpss_test(x,regression,nlags=2):
    x=np.asarray(x,float); n=len(x)
    if regression=='ct':
        X=np.column_stack([np.ones(n),np.arange(1,n+1,dtype=float)]); resid=x-X@np.linalg.lstsq(X,x,rcond=None)[0]
        crit={'10%':0.119,'5%':0.146,'2.5%':0.176,'1%':0.216}
    else:
        resid=x-x.mean(); crit={'10%':0.347,'5%':0.463,'2.5%':0.574,'1%':0.739}
    eta=float(np.sum(np.cumsum(resid)**2)/(n*n)); s2=float(resid@resid/n)
    for lag in range(1,nlags+1):s2+=2*(1-lag/(nlags+1))*float(resid[lag:]@resid[:-lag])/n
    stat=eta/s2
    decision='拒绝平稳性，序列非平稳' if stat>crit['5%'] else '不能拒绝平稳性，序列平稳'
    return stat,nlags,n,crit,decision

def _nelder_mead(fun,x0,step=.18,maxiter=1400,tol=1e-10):
    n=len(x0); simplex=[np.array(x0,float)]
    for i in range(n):
        z=np.array(x0,float); z[i]+=step; simplex.append(z)
    vals=[fun(z) for z in simplex]
    for _ in range(maxiter):
        order=np.argsort(vals); simplex=[simplex[i] for i in order]; vals=[vals[i] for i in order]
        if np.std(vals)<tol:break
        c=np.mean(simplex[:-1],axis=0); xr=c+(c-simplex[-1]); fr=fun(xr)
        if vals[0]<=fr<vals[-2]:simplex[-1],vals[-1]=xr,fr;continue
        if fr<vals[0]:
            xe=c+2*(xr-c);fe=fun(xe)
            simplex[-1],vals[-1]=(xe,fe) if fe<fr else (xr,fr);continue
        xc=c+.5*(simplex[-1]-c);fc=fun(xc)
        if fc<vals[-1]:simplex[-1],vals[-1]=xc,fc;continue
        simplex=[simplex[0]]+[simplex[0]+.5*(z-simplex[0]) for z in simplex[1:]];vals=[fun(z) for z in simplex]
    i=int(np.argmin(vals));return simplex[i],vals[i]

def arma_css_fit(w,p,q):
    w=np.asarray(w,float); n=len(w)
    def unpack(u):return float(u[0]),.98*np.tanh(u[1:1+p]),.98*np.tanh(u[1+p:1+p+q])
    def calc(u):
        mu,phi,theta=unpack(u);e=np.zeros(n)
        for t in range(n):
            pred=mu
            for i in range(p):pred+=phi[i]*((w[t-i-1] if t-i-1>=0 else mu)-mu)
            for j in range(q):pred+=theta[j]*(e[t-j-1] if t-j-1>=0 else 0.0)
            e[t]=w[t]-pred
        return e
    def obj(u):
        e=calc(u);return float(e@e)
    base=np.r_[w.mean(),np.zeros(p+q)];starts=[base]
    rng=np.random.default_rng(20260818+p*10+q)
    for _ in range(7):starts.append(base+np.r_[rng.normal(0,.04),rng.normal(0,.35,p+q)])
    fits=[_nelder_mead(obj,s) for s in starts];u,sse=min(fits,key=lambda z:z[1]);mu,phi,theta=unpack(u);e=calc(u)
    sigma2=max(sse/n,1e-15);ll=-.5*n*(math.log(2*math.pi)+1+math.log(sigma2));k=p+q+2
    aic=-2*ll+2*k;bic=-2*ll+k*math.log(n);aicc=aic+2*k*(k+1)/(n-k-1) if n>k+1 else float('inf')
    return {'mu':mu,'phi':phi,'theta':theta,'resid':e,'ll':ll,'aic':aic,'aicc':aicc,'bic':bic,'k':k,'rmse_dlog':math.sqrt(sse/n)}

def arma_next(w,m):
    p=len(m['phi']);q=len(m['theta']);pred=m['mu'];e=m['resid'];n=len(w)
    for i in range(p):pred+=m['phi'][i]*(w[n-i-1]-m['mu'])
    for j in range(q):pred+=m['theta'][j]*e[n-j-1]
    return float(pred)

def arima_order_comparison(y,indicator):
    z=np.log(y);w=np.diff(z);out=[]
    for p in range(3):
      for q in range(3):
        m=arma_css_fit(w,p,q);apes=[]
        for yy in [2023,2024,2025]:
            idx=int(np.where(year==yy)[0][0]);wt=np.diff(z[:idx]);mt=arma_css_fit(wt,p,q);pred=math.exp(z[idx-1]+arma_next(wt,mt));apes.append(abs(y[idx]-pred)/y[idx])
        boundary=bool(any(abs(v)>=.95 for v in np.r_[m['phi'],m['theta']]))
        selected=(p==0 and q==0)
        if selected: note='主模型：结构最简，且滚动预测稳定'
        elif boundary: note='参数接近边界，存在小样本过拟合风险'
        else: note='候选模型，不作为主模型'
        out.append([indicator,p,1,q,m['k'],m['ll'],m['aic'],m['aicc'],m['bic'],m['rmse_dlog'],100*np.mean(apes),m['mu'],';'.join(f'{v:.6f}' for v in m['phi']),';'.join(f'{v:.6f}' for v in m['theta']),boundary,selected,note])
    return out

def acf_pacf_values(x,maxlag=5):
    x=np.asarray(x,float);x=x-x.mean();n=len(x);den=float(x@x)
    acf=[1.0]+[float(x[k:]@x[:-k]/den) for k in range(1,maxlag+1)]
    pacf=[1.0]
    for k in range(1,maxlag+1):
        R=np.array([[acf[abs(i-j)] for j in range(k)] for i in range(k)])
        pacf.append(float(np.linalg.solve(R,np.array(acf[1:k+1]))[-1]))
    return acf,pacf,1.96/math.sqrt(n)

def rs_hurst(x):
    x=np.asarray(x,float); rows=[]
    for scale in range(4,len(x)+1):
        vals=[]
        for start in range(0,len(x)-scale+1):
            seg=x[start:start+scale];dev=seg-seg.mean();s=float(seg.std(ddof=1))
            if s<=0:continue
            cs=np.cumsum(dev);vals.append(float((cs.max()-cs.min())/s))
        if vals:rows.append([scale,len(vals),float(np.mean(vals))])
    lx=np.log([r[0] for r in rows]);ly=np.log([r[2] for r in rows]);H,b=np.polyfit(lx,ly,1);fit=H*lx+b
    r2=1-float(np.sum((ly-fit)**2))/float(np.sum((ly-ly.mean())**2))
    return float(H),float(b),r2,rows

forecasts=[]; validation=[]; sensitivity=[]; covid_rows=[]; bootstrap_audit=[]; stationarity=[]; stationarity_data=[]; correlogram=[]; hurst_summary=[]; hurst_scales=[]; order_rows=[]; summaries={}
for key,y in [('游客接待量_主插补',N_main),('旅游综合收入',I)]:
    order_rows.extend(arima_order_comparison(y,key))
    z=np.log(y);dz=np.r_[np.nan,np.diff(z)]
    for yy,raw,lv,dv in zip(year,y,z,dz):stationarity_data.append([key,int(yy),float(raw),float(lv),'' if np.isnan(dv) else float(dv)])
    ac,pac,ci=acf_pacf_values(np.diff(z),5)
    for lag in range(6):correlogram.append([key,'对数一阶差分',lag,ac[lag],pac[lag],ci,-ci,len(z)-1])
    for transform,series in [('原对数',z),('对数一阶差分',np.diff(z))]:
        H,b,r2,hrs=rs_hurst(series);interpretation='持续性' if H>.55 else ('反持续性' if H<.45 else '接近随机游走')
        hurst_summary.append([key,transform,len(series),H,b,r2,interpretation,'探索性辅助，不用于直接决定ARIMA阶数或增长率'])
        for scale,windows,rs in hrs:hurst_scales.append([key,transform,scale,windows,rs,math.log(scale),math.log(rs),H*math.log(scale)+b])
    for transform,series,reg in [('原对数',np.log(y),'ct'),('对数一阶差分',np.diff(np.log(y)),'c')]:
        astat,alag,an,acrit,adec=adf_test(series,reg)
        kstat,klag,kn,kcrit,kdec=kpss_test(series,reg)
        adf_stationary=adec.startswith('拒绝单位根'); kpss_stationary=kdec.startswith('不能拒绝平稳性')
        joint='支持平稳' if (adf_stationary and kpss_stationary) else ('支持非平稳' if ((not adf_stationary) and (not kpss_stationary)) else '结论不完全一致，结合图形与差分结果判断')
        stationarity.append([key,transform,reg,astat,alag,an,acrit['1%'],acrit['5%'],acrit['10%'],adec,kstat,klag,kn,kcrit['1%'],kcrit['5%'],kcrit['10%'],kdec,joint])
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
write('q2_stationarity_tests.csv',['indicator','transform','regression','ADF_stat','ADF_lag_BIC','ADF_nobs','ADF_crit_1pct','ADF_crit_5pct','ADF_crit_10pct','ADF_conclusion_5pct','KPSS_stat','KPSS_lag','KPSS_nobs','KPSS_crit_1pct','KPSS_crit_5pct','KPSS_crit_10pct','KPSS_conclusion_5pct','joint_conclusion'],stationarity)
write('q2_arima_order_comparison.csv',['indicator','p','d','q','parameter_count_including_variance','conditional_loglik','AIC','AICc','BIC','in_sample_RMSE_dlog','rolling_MAPE_2023_2025_pct','drift_mean_dlog','AR_parameters','MA_parameters','parameter_boundary_flag','selected_main_model','selection_note'],order_rows)
write('q2_stationarity_input_data.csv',['indicator','year','original_value','log_value','dlog_value'],stationarity_data)
write('q2_acf_pacf_values.csv',['indicator','transform','lag','ACF','PACF','CI95_upper','CI95_lower','sample_size'],correlogram)
write('q2_hurst_summary.csv',['indicator','transform','sample_size','Hurst_H','intercept','R_squared','interpretation','usage_note'],hurst_summary)
write('q2_hurst_scale_values.csv',['indicator','transform','scale','window_count','mean_RS','log_scale','log_RS','fitted_log_RS'],hurst_scales)

def svg_corr(path,title,vals,ci,color):
    W,H=760,470;L,R,T,B=80,30,58,70;lags=np.arange(len(vals));mn=-1.05;mx=1.05;sx=lambda v:L+v/max(1,len(vals)-1)*(W-L-R);sy=lambda v:T+(mx-v)/(mx-mn)*(H-T-B)
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:Microsoft YaHei,Arial;font-size:14px}}</style><text x="{W/2}" y="30" text-anchor="middle" font-size="19">{title}</text>']
    for v in [-1,-.5,0,.5,1]:p.append(f'<line x1="{L}" y1="{sy(v)}" x2="{W-R}" y2="{sy(v)}" stroke="{("#555" if v==0 else "#ddd")}"/><text x="{L-10}" y="{sy(v)+5}" text-anchor="end">{v:.1f}</text>')
    for c in [ci,-ci]:p.append(f'<line x1="{L}" y1="{sy(c)}" x2="{W-R}" y2="{sy(c)}" stroke="#D84315" stroke-width="2" stroke-dasharray="7 5"/>')
    for lag,v in zip(lags,vals):p.append(f'<line x1="{sx(lag)}" y1="{sy(0)}" x2="{sx(lag)}" y2="{sy(v)}" stroke="{color}" stroke-width="7"/><circle cx="{sx(lag)}" cy="{sy(v)}" r="4" fill="{color}"/><text x="{sx(lag)}" y="{H-B+25}" text-anchor="middle">{lag}</text>')
    p.append(f'<text x="{W/2}" y="{H-18}" text-anchor="middle">滞后阶数</text><text transform="translate(22 {H/2}) rotate(-90)" text-anchor="middle">相关系数</text><text x="{W-R-5}" y="{sy(ci)-8}" text-anchor="end" fill="#D84315">95%置信界 ±{ci:.3f}</text></svg>');path.write_text("".join(p),encoding='utf-8')

for key,prefix,label in [('游客接待量_主插补','03_游客量','游客量对数一阶差分'),('旅游综合收入','04_旅游收入','旅游收入对数一阶差分')]:
    rr=[r for r in correlogram if r[0]==key];ci=rr[0][5]
    svg_corr(FIG/f'{prefix}_ACF.svg',f'{label} ACF',[r[3] for r in rr],ci,'#1565C0')
    svg_corr(FIG/f'{prefix}_PACF.svg',f'{label} PACF',[r[4] for r in rr],ci,'#2E7D32')

def svg_hurst(path,title,rows,H,b,r2,color):
    W,Ht=760,470;L,R,T,B=85,35,58,72;x=np.array([r[5] for r in rows]);y=np.array([r[6] for r in rows]);fit=np.array([r[7] for r in rows]);xmin,xmax=float(x.min()),float(x.max());ymin,ymax=float(min(y.min(),fit.min())-.08),float(max(y.max(),fit.max())+.08);sx=lambda v:L+(v-xmin)/(xmax-xmin)*(W-L-R);sy=lambda v:T+(ymax-v)/(ymax-ymin)*(Ht-T-B)
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Ht}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:Microsoft YaHei,Arial;font-size:14px}}</style><text x="{W/2}" y="30" text-anchor="middle" font-size="19">{title}</text>']
    for j in range(5):
        v=ymin+(ymax-ymin)*j/4;p.append(f'<line x1="{L}" y1="{sy(v)}" x2="{W-R}" y2="{sy(v)}" stroke="#ddd"/><text x="{L-8}" y="{sy(v)+5}" text-anchor="end">{v:.2f}</text>')
    p.append(f'<polyline points="'+ ' '.join(f'{sx(a):.1f},{sy(c):.1f}' for a,c in zip(x,fit)) +f'" fill="none" stroke="#D84315" stroke-width="3"/>')
    for a,c in zip(x,y):p.append(f'<circle cx="{sx(a)}" cy="{sy(c)}" r="5" fill="{color}"/>')
    p.append(f'<text x="{L+10}" y="{T+25}" fill="#333">H={H:.4f}，R²={r2:.4f}</text><text x="{W/2}" y="{Ht-18}" text-anchor="middle">ln(尺度)</text><text transform="translate(24 {Ht/2}) rotate(-90)" text-anchor="middle">ln(R/S)</text></svg>');path.write_text("".join(p),encoding='utf-8')

for key,prefix,label in [('游客接待量_主插补','05_游客量','游客量'),('旅游综合收入','06_旅游收入','旅游收入')]:
    for transform,suffix in [('原对数','原对数'),('对数一阶差分','差分')]:
        sr=next(r for r in hurst_summary if r[0]==key and r[1]==transform);rr=[r for r in hurst_scales if r[0]==key and r[1]==transform]
        svg_hurst(FIG/f'{prefix}_Hurst_{suffix}.svg',f'{label}{transform} R/S-Hurst',rr,sr[3],sr[4],sr[5],'#1565C0' if transform=='原对数' else '#2E7D32')

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
