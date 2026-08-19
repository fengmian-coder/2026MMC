from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "q3_results"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BETA_M = 2.4225
BETA_Q = 0.2646
BETA_F = None

PARAMS = {
    "baseline": {"M": 0.0377, "Q": [0.216] * 5, "F": 0.075, "U": 0.900},
    "optimistic": {"M": 0.0494, "Q": [0.283, 0.308, 0.308, 0.308, 0.308], "F": 0.10, "U": 1.131},
    "pessimistic": {"M": 0.0053, "Q": [0.032] * 5, "F": 0.0, "U": 0.669},
}


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict]):
    path = OUT / name
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), 3e-14) * (1 if d >= 0 else -1)
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 3e-14 else 3e-14)
        c = 1.0 + aa / c; c = c if abs(c) > 3e-14 else 3e-14
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 3e-14 else 3e-14)
        c = 1.0 + aa / c; c = c if abs(c) > 3e-14 else 3e-14
        delta = d * c; h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def regularized_beta(x, a, b):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
    if x < (a+1)/(a+b+2):
        return bt * _betacf(a,b,x) / a
    return 1 - bt * _betacf(b,a,1-x) / b


def student_t_two_sided_p(t, df):
    x = df / (df + t*t)
    return regularized_beta(x, df/2, 0.5)


def ols(y, columns, names, hac_lags=0):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in columns])
    n, k = X.shape
    inv = np.linalg.inv(X.T @ X)
    b = inv @ X.T @ y
    resid = y - X @ b
    rss = float(resid @ resid)
    df = n - k
    sigma2 = rss / df
    if hac_lags:
        meat = np.zeros((k, k))
        for t in range(n):
            meat += resid[t] ** 2 * np.outer(X[t], X[t])
        for lag in range(1, min(hac_lags, n - 1) + 1):
            weight = 1 - lag / (hac_lags + 1)
            gamma = np.zeros((k, k))
            for t in range(lag, n):
                gamma += resid[t] * resid[t-lag] * np.outer(X[t], X[t-lag])
            meat += weight * (gamma + gamma.T)
        cov = (n / df) * inv @ meat @ inv
    else:
        cov = sigma2 * inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    tval = b / se
    pval = np.asarray([student_t_two_sided_p(float(t), df) for t in tval])
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    aic = n * math.log(rss / n) + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else float("inf")
    bic = n * math.log(rss / n) + k * math.log(n)
    preds = []
    for i in range(n):
        mask = np.arange(n) != i
        bi = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
        preds.append(float(X[i] @ bi))
    loocv = math.sqrt(float(np.mean((y - np.asarray(preds)) ** 2)))
    return {
        "n": n,
        "df_resid": df,
        "names": ["intercept"] + names,
        "coef": b,
        "se": se,
        "p": pval,
        "r2": r2,
        "adj_r2": 1 - (1-r2)*(n-1)/df,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "loocv_rmse": loocv,
        "fitted": X @ b,
        "resid": resid,
    }


def f_test_p(rss_r, rss_u, q, df_u):
    f = max(0.0, ((rss_r-rss_u)/q)/(rss_u/df_u))
    # F(1,df) equals squared t(df); sufficient here because Granger lag=1.
    return f, student_t_two_sided_p(math.sqrt(f), df_u)


def svg_line(path: Path, title: str, ylabel: str, years, series, colors):
    W, H = 1100, 700
    L, R, T, B = 110, 45, 75, 100
    values = [v for vals in series.values() for v in vals]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.12 or 1
    lo, hi = max(0, lo - pad), hi + pad
    x = lambda i: L + i * (W - L - R) / (len(years) - 1)
    y = lambda v: T + (hi - v) * (H - T - B) / (hi - lo)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{W/2}" y="36" text-anchor="middle" font-size="25" font-family="Microsoft YaHei" font-weight="700">{title}</text>']
    for j in range(6):
        val = lo + j * (hi - lo) / 5
        yy = y(val)
        parts += [f'<line x1="{L}" y1="{yy:.1f}" x2="{W-R}" y2="{yy:.1f}" stroke="#e5e7eb"/>',
                  f'<text x="{L-12}" y="{yy+5:.1f}" text-anchor="end" font-size="16" font-family="Microsoft YaHei">{val:.1f}</text>']
    parts += [f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#374151"/>',
              f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#374151"/>',
              f'<text x="25" y="{H/2}" transform="rotate(-90 25 {H/2})" text-anchor="middle" font-size="18" font-family="Microsoft YaHei">{ylabel}</text>']
    for i, yr in enumerate(years):
        parts.append(f'<text x="{x(i):.1f}" y="{H-B+30}" text-anchor="middle" font-size="16" font-family="Microsoft YaHei">{yr}</text>')
    for (label, vals), color in zip(series.items(), colors):
        pts = ' '.join(f'{x(i):.1f},{y(v):.1f}' for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="4"/>')
        for i, v in enumerate(vals):
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="5" fill="{color}"/>')
    lx = L
    for (label, _), color in zip(series.items(), colors):
        parts += [f'<line x1="{lx}" y1="{H-40}" x2="{lx+38}" y2="{H-40}" stroke="{color}" stroke-width="5"/>',
                  f'<text x="{lx+48}" y="{H-34}" font-size="17" font-family="Microsoft YaHei">{label}</text>']
        lx += 220
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def svg_bar(path: Path, title: str, labels, values, ylabel: str, percent=True):
    W, H = 1000, 650
    L, R, T, B = 140, 50, 80, 120
    vmax = max(values) * 1.18
    bw = (W - L - R) / len(values) * 0.55
    gap = (W - L - R) / len(values)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">', '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{W/2}" y="38" text-anchor="middle" font-size="25" font-family="Microsoft YaHei" font-weight="700">{title}</text>']
    for j in range(6):
        v = vmax * j / 5; yy = H-B-v/vmax*(H-T-B)
        label=f'{v:.1%}' if percent else f'{v:.3f}'
        parts += [f'<line x1="{L}" y1="{yy:.1f}" x2="{W-R}" y2="{yy:.1f}" stroke="#e5e7eb"/>', f'<text x="{L-12}" y="{yy+5:.1f}" text-anchor="end" font-size="15" font-family="Microsoft YaHei">{label}</text>']
    colors = ['#2563eb','#10b981','#f59e0b','#8b5cf6']
    for i,(lab,v) in enumerate(zip(labels,values)):
        xx=L+gap*i+(gap-bw)/2; yy=H-B-v/vmax*(H-T-B)
        vlabel=f'{v:.2%}' if percent else f'{v:.4f}'
        parts += [f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{H-B-yy:.1f}" fill="{colors[i]}"/>',
                  f'<text x="{xx+bw/2:.1f}" y="{yy-10:.1f}" text-anchor="middle" font-size="17" font-family="Microsoft YaHei">{vlabel}</text>',
                  f'<text x="{xx+bw/2:.1f}" y="{H-B+32}" text-anchor="middle" font-size="17" font-family="Microsoft YaHei">{lab}</text>']
    parts += [f'<text x="28" y="{H/2}" transform="rotate(-90 28 {H/2})" text-anchor="middle" font-size="18" font-family="Microsoft YaHei">{ylabel}</text>', '</svg>']
    path.write_text('\n'.join(parts), encoding='utf-8')


def svg_single_with_gaps(path, title, ylabel, years, values, color="#2563eb"):
    W,H,L,R,T,B=1100,700,110,45,75,100
    valid=[v for v in values if v is not None]; lo,hi=min(valid),max(valid); pad=(hi-lo)*0.12 or 1; lo-=pad; hi+=pad
    x=lambda i:L+i*(W-L-R)/(len(years)-1); y=lambda v:T+(hi-v)*(H-T-B)/(hi-lo)
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="36" text-anchor="middle" font-size="25" font-family="Microsoft YaHei" font-weight="700">{title}</text>']
    for j in range(6):
        v=lo+j*(hi-lo)/5; yy=y(v); p += [f'<line x1="{L}" y1="{yy:.1f}" x2="{W-R}" y2="{yy:.1f}" stroke="#e5e7eb"/>',f'<text x="{L-12}" y="{yy+5:.1f}" text-anchor="end" font-size="15" font-family="Microsoft YaHei">{v:.3f}</text>']
    p += [f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#374151"/><line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#374151"/>',f'<text x="25" y="{H/2}" transform="rotate(-90 25 {H/2})" text-anchor="middle" font-size="18" font-family="Microsoft YaHei">{ylabel}</text>']
    for i,yr in enumerate(years): p.append(f'<text x="{x(i):.1f}" y="{H-B+30}" text-anchor="middle" font-size="14" font-family="Microsoft YaHei">{yr}</text>')
    for i,v in enumerate(values):
        if v is None: continue
        if i and values[i-1] is not None: p.append(f'<line x1="{x(i-1):.1f}" y1="{y(values[i-1]):.1f}" x2="{x(i):.1f}" y2="{y(v):.1f}" stroke="{color}" stroke-width="4"/>')
        p.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="6" fill="{color}"/>')
    p.append('</svg>'); path.write_text('\n'.join(p),encoding='utf-8')


def svg_tornado(path, rows):
    W,H,L,R,T,B=1100,650,230,80,80,70; maxabs=max(max(abs(r['plus_change_2030']),abs(r['minus_change_2030'])) for r in rows)*1.18
    cx=L+(W-L-R)/2; scale=(W-L-R)/2/maxabs; p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>','<text x="550" y="38" text-anchor="middle" font-size="25" font-family="Microsoft YaHei" font-weight="700">2030年收入±20% OAT龙卷风图</text>',f'<line x1="{cx}" y1="{T}" x2="{cx}" y2="{H-B}" stroke="#374151" stroke-width="2"/>']
    gap=(H-T-B)/len(rows)
    for i,r in enumerate(rows):
        yy=T+i*gap+gap*0.2; h=gap*0.55; neg=r['minus_change_2030']; pos=r['plus_change_2030']
        p += [f'<text x="{L-20}" y="{yy+h*0.7}" text-anchor="end" font-size="20" font-family="Microsoft YaHei">{r["factor"]}</text>',f'<rect x="{cx+neg*scale:.1f}" y="{yy:.1f}" width="{-neg*scale:.1f}" height="{h:.1f}" fill="#dc2626"/>',f'<rect x="{cx:.1f}" y="{yy:.1f}" width="{pos*scale:.1f}" height="{h:.1f}" fill="#16a34a"/>',f'<text x="{cx+neg*scale-8:.1f}" y="{yy+h*0.7:.1f}" text-anchor="end" font-size="16" font-family="Microsoft YaHei">{neg:.2%}</text>',f'<text x="{cx+pos*scale+8:.1f}" y="{yy+h*0.7:.1f}" font-size="16" font-family="Microsoft YaHei">+{pos:.2%}</text>']
    p.append('</svg>'); path.write_text('\n'.join(p),encoding='utf-8')


# Historical series: Q3 plan deliberately excludes 2021 and 2022 revenue from driver regressions.
revenue = {2010:35.5,2011:46,2012:52,2013:63.2,2014:79,2015:94,2016:109.6,2017:130,2018:142,2019:165,2020:103.4,2023:191.5,2024:221,2025:200}
arrivals = {2010:848,2011:1098.6,2012:1152,2013:1295,2014:1554,2015:1814.4,2016:2086.27,2017:2400,2018:2660,2019:2800,2020:1853.73,2023:2363,2024:2643,2025:2691}
market = {2016:100,2017:105.13,2018:111.45,2019:119.85,2020:107.94,2021:122.19,2022:117.73,2023:131.34,2024:136.37,2025:139.52}

# Fiscal model revision: F is the original culture/tourism/sports/media expenditure.
# 2010-2018 values are explicitly labelled stagewise/non-officially rechecked in the supplied PMU workbook.
fiscal = {2010:1738,2011:2231,2012:2899,2013:9457,2014:6245,2015:5593,2016:9312,2017:7986,2018:11694,2022:7445,2023:5677,2024:5110}
f_status = {t:("官方年鉴已核" if t>=2022 else "阶段性整理值，待官方逐年复核") for t in fiscal}
revenue_f = {2010:35.5,2011:46,2012:52,2013:63.2,2014:79,2015:94,2016:109.6,2017:130,2018:142,2019:165,2020:103.4,2021:110,2022:160,2023:191.5,2024:221,2025:200}
dln_f = {t:math.log(fiscal[t]/fiscal[t-1]) for t in fiscal if t-1 in fiscal}
dln_y_f = {t:math.log(revenue_f[t]/revenue_f[t-1]) for t in revenue_f if t-1 in revenue_f}

f_raw_rows=[]
for t in range(2010,2026):
    f_raw_rows.append({"year":t,"F_10k_cny":fiscal.get(t,""),"lnF":math.log(fiscal[t]) if t in fiscal else "","dlnF":dln_f.get(t,""),"status":f_status.get(t,"缺失，不插值")})
write_csv("q3_F_raw_transformed.csv",f_raw_rows)

# Fair 0/1/2 comparison requires all three fiscal lags on the same years.
f_common_years=[t for t in sorted(dln_y_f) if all(t-k in dln_f for k in (0,1,2))]
f_common_rows=[]; lag_models=[]
for t in f_common_years:
    f_common_rows.append({"year":t,"dlnY":dln_y_f[t],"dlnF_lag0":dln_f[t],"dlnF_lag1":dln_f[t-1],"dlnF_lag2":dln_f[t-2],"Shock2020":int(t==2020),"note":"Shock2020在共同样本恒为0，无法识别"})
write_csv("q3_F_lag_common_sample.csv",f_common_rows)
for lag in (0,1,2):
    fit=ols([dln_y_f[t] for t in f_common_years],[[dln_f[t-lag] for t in f_common_years]],[f"dlnF_lag{lag}"],hac_lags=1)
    lag_models.append((lag,fit))
best_lag=min(lag_models,key=lambda z:z[1]["aicc"])[0]
lag_rows=[]
for lag,fit in lag_models:
    lag_rows.append({"lag":lag,"years":';'.join(map(str,f_common_years)),"n":fit["n"],"Shock2020_identifiable":"no (all zero)","beta_F":fit["coef"][1],"HAC_SE":fit["se"][1],"p_value":fit["p"][1],"R2":fit["r2"],"adjusted_R2":fit["adj_r2"],"AIC":fit["aic"],"AICc":fit["aicc"],"BIC":fit["bic"],"LOOCV_RMSE":fit["loocv_rmse"],"selected_by_AICc":lag==best_lag})
write_csv("q3_F_lag_comparison.csv",lag_rows)

# Refit the selected lag on every valid observation; Shock2020 is retained only if it varies.
f_final_years=[t for t in sorted(dln_y_f) if t-best_lag in dln_f]
shock=[int(t==2020) for t in f_final_years]
f_columns=[[dln_f[t-best_lag] for t in f_final_years]]
f_names=[f"dlnF_lag{best_lag}"]
shock_identifiable=len(set(shock))>1
if shock_identifiable:
    f_columns.append(shock); f_names.append("Shock2020")
f_final=ols([dln_y_f[t] for t in f_final_years],f_columns,f_names,hac_lags=1)
BETA_F=float(f_final["coef"][1])
f_final_rows=[]
for name,coef,se,p in zip(f_final["names"],f_final["coef"],f_final["se"],f_final["p"]):
    # Small-sample t critical via bisection on the exact t CDF helper.
    lo,hi=0.0,20.0
    for _ in range(80):
        mid=(lo+hi)/2
        if student_t_two_sided_p(mid,f_final["df_resid"])>0.05: lo=mid
        else: hi=mid
    crit=(lo+hi)/2
    f_final_rows.append({"term":name,"coefficient":coef,"HAC_SE_maxlag1":se,"p_value":p,"CI95_low":coef-crit*se,"CI95_high":coef+crit*se,"n":f_final["n"],"df_resid":f_final["df_resid"],"years":';'.join(map(str,f_final_years)),"Shock2020_identifiable":shock_identifiable,"R2":f_final["r2"],"adjusted_R2":f_final["adj_r2"],"AIC":f_final["aic"],"AICc":f_final["aicc"],"BIC":f_final["bic"],"LOOCV_RMSE":f_final["loocv_rmse"]})
write_csv("q3_F_final_model.csv",f_final_rows)
write_csv("q3_F_actual_fitted.csv",[{"year":t,"actual_dlnY":dln_y_f[t],"fitted_dlnY":float(f_final["fitted"][i]),"residual":float(f_final["resid"][i])} for i,t in enumerate(f_final_years)])

# Bidirectional lag-1 Granger on the longest truly continuous block (2011-2018); no gaps are stitched.
continuous=list(range(2011,2019)); gy=np.array([dln_y_f[t] for t in continuous]); gf=np.array([dln_f[t] for t in continuous])
granger_rows=[]
def granger_direction(target,driver,label):
    yt=target[1:]; own=target[:-1]; drv=driver[:-1]
    unrestricted=ols(yt,[own,drv],["own_lag1","driver_lag1"],hac_lags=1)
    restricted=ols(yt,[own],["own_lag1"],hac_lags=1)
    rss_u=float(unrestricted["resid"]@unrestricted["resid"]); rss_r=float(restricted["resid"]@restricted["resid"])
    fstat,p=f_test_p(rss_r,rss_u,1,unrestricted["df_resid"])
    granger_rows.append({"test":label,"years":"2012-2018","n":len(yt),"F_stat":fstat,"p_value":p,"driver_coefficient":unrestricted["coef"][2],"driver_HAC_SE":unrestricted["se"][2],"driver_HAC_p":unrestricted["p"][2],"Shock2020_identifiable":"no (continuous block ends 2018)","interpretation":"auxiliary predictive-leading test only"})
granger_direction(gy,gf,"dlnF → dlnY")
granger_direction(gf,gy,"dlnY → dlnF")
write_csv("q3_F_granger_dynamic.csv",granger_rows)

m_years = [t for t in range(2017,2026) if t in revenue and t-1 in revenue]
m_fit = ols([math.log(revenue[t]/revenue[t-1]) for t in m_years],
            [[math.log(market[t]/market[t-1]) for t in m_years], [1 if t == 2020 else 0 for t in m_years]],
            ["dlnM", "Shock2020"])

# A transparent visitor calibration: relative historical log-growth volatility in normal adjacent years.
common = [t for t in range(2011,2026) if t in arrivals and t-1 in arrivals and t in revenue and t-1 in revenue and t != 2020]
arr_g = np.array([math.log(arrivals[t]/arrivals[t-1]) for t in common])
rev_g = np.array([math.log(revenue[t]/revenue[t-1]) for t in common])
visitor_scale = float(np.std(arr_g, ddof=1) / np.std(rev_g, ddof=1))

q2 = [r for r in read_csv(ROOT / "outputs/q2_revised/q2_forecast_2026_2030.csv") if r["scenario"] == "baseline"]
base_rev = {int(r["year"]): float(r["point_forecast"]) for r in q2 if r["indicator"] == "旅游综合收入"}
base_arr = {int(r["year"]): float(r["point_forecast"]) for r in q2 if r["indicator"] == "游客接待量_主插补"}
years = list(range(2026,2031))

def scenario_path(base, start2025, scenario, scale=1.0, fiscal=True):
    p = PARAMS[scenario]
    prev = start2025
    out = []
    for i,t in enumerate(years):
        base_prev = start2025 if i == 0 else base[t-1]
        gb = math.log(base[t]/base_prev)
        adj = BETA_M*(p["M"]-PARAMS["baseline"]["M"]) + BETA_Q*(p["Q"][i]-PARAMS["baseline"]["Q"][i])
        if fiscal and i >= best_lag:
            adj += BETA_F*(math.log1p(p["F"])-math.log1p(PARAMS["baseline"]["F"]))
        gs = gb + scale*adj
        prev = prev*math.exp(gs)
        out.append(prev)
    return out

paths_rev = {"宏观下行悲观": scenario_path(base_rev,200,"pessimistic"), "基准": [base_rev[y] for y in years], "乐观": scenario_path(base_rev,200,"optimistic")}
paths_arr = {"宏观下行悲观": scenario_path(base_arr,2691,"pessimistic",visitor_scale,False), "基准": [base_arr[y] for y in years], "乐观": scenario_path(base_arr,2691,"optimistic",visitor_scale,False)}

scenario_rows=[]
for i,t in enumerate(years):
    for s in ["宏观下行悲观","基准","乐观"]:
        scenario_rows.append({"year":t,"scenario":s,"tourist_arrivals_10k":round(paths_arr[s][i],4),"tourism_revenue_100m_cny":round(paths_rev[s][i],4)})
write_csv("q3_scenario_forecast_2026_2030.csv",scenario_rows)

# Risk stress paths are permanent one-off level shifts relative to the Q2 baseline, not probability forecasts.
covid = [r for r in read_csv(ROOT / "outputs/q2_revised/q2_covid_counterfactual.csv") if r["indicator"] == "旅游综合收入"][0]
d_covid = float(covid["covid_beta_on_dlog"])
stress_rows=[]
for t in years:
    stress_rows += [
        {"year":t,"stress":"一般重大事件（0.5×COVID对数冲击）","log_shock":0.5*d_covid,"revenue_100m_cny":base_rev[t]*math.exp(0.5*d_covid),"loss_vs_baseline":1-math.exp(0.5*d_covid)},
        {"year":t,"stress":"疫情级极端事件（完整COVID对数冲击）","log_shock":d_covid,"revenue_100m_cny":base_rev[t]*math.exp(d_covid),"loss_vs_baseline":1-math.exp(d_covid)},
    ]
write_csv("q3_event_stress_tests.csv",stress_rows)

# OAT sensitivity: apply equal relative perturbations to each baseline driver parameter.
sens_rows=[]
base_vals=np.array(paths_rev["基准"])
base_sum=float(base_vals.sum())
for rho in (0.10,0.20):
    for factor in ("M","Q","F"):
        def oat(sign):
            prev=200; vals=[]
            for i,t in enumerate(years):
                bp=200 if i==0 else base_rev[t-1]
                gb=math.log(base_rev[t]/bp)
                delta=0.0
                if factor=="M": delta=BETA_M*PARAMS["baseline"]["M"]*sign*rho
                if factor=="Q": delta=BETA_Q*PARAMS["baseline"]["Q"][i]*sign*rho
                if factor=="F" and i>=best_lag:
                    perturbed=PARAMS["baseline"]["F"]*(1+sign*rho)
                    delta=BETA_F*(math.log1p(perturbed)-math.log1p(PARAMS["baseline"]["F"]))
                prev*=math.exp(gb+delta); vals.append(prev)
            return np.array(vals)
        plus,minus=oat(1),oat(-1)
        sens_rows.append({"factor":factor,"perturbation":rho,"revenue_2030_plus":plus[-1],"revenue_2030_minus":minus[-1],"plus_change_2030":plus[-1]/base_vals[-1]-1,"minus_change_2030":minus[-1]/base_vals[-1]-1,"five_year_absolute_spread":float(np.abs(plus-minus).sum()),"normalized_sensitivity":float(np.abs(plus-minus).sum()/base_sum)})

# U-Q linkage uses the specified Q paths; U is a state label, not an estimated coefficient.
def q_only(qvals):
    prev=200; vals=[]
    for i,t in enumerate(years):
        bp=200 if i==0 else base_rev[t-1]; gb=math.log(base_rev[t]/bp)
        prev*=math.exp(gb+BETA_Q*(qvals[i]-0.216)); vals.append(prev)
    return np.array(vals)
u_low=q_only([0.032]*5); u_high=q_only([0.283,0.308,0.308,0.308,0.308])
u_norm=float(np.abs(u_high-u_low).sum()/base_sum)
sens_rows.append({"factor":"U-Q联动","perturbation":"低0.669/高1.131","revenue_2030_plus":u_high[-1],"revenue_2030_minus":u_low[-1],"plus_change_2030":u_high[-1]/base_vals[-1]-1,"minus_change_2030":u_low[-1]/base_vals[-1]-1,"five_year_absolute_spread":float(np.abs(u_high-u_low).sum()),"normalized_sensitivity":u_norm})
write_csv("q3_sensitivity_analysis.csv",sens_rows)

reg_rows=[]
for name,coef,se,p in zip(m_fit["names"],m_fit["coef"],m_fit["se"],m_fit["p"]):
    reg_rows.append({"model":"M单因素+Shock2020（本次复算）","term":name,"coefficient":coef,"std_error":se,"p_value":p,"n":m_fit["n"],"df_resid":m_fit["df_resid"],"r2":m_fit["r2"],"AICc":m_fit["aicc"],"BIC":m_fit["bic"],"LOOCV_RMSE":m_fit["loocv_rmse"],"status":"由仓库M和收入数据复算"})
for model,term,coef,p,note in [
    ("Q单因素+Shock2020（方案给定）","dQ",0.4345,0.038,"仓库缺少逐年Q原始构成，未能独立复算"),
    ("MQ联合模型（方案给定）","dlnM",2.4225,0.065,"仅作情景响应参数"),
    ("MQ联合模型（方案给定）","dQ",0.2646,0.042,"仅作情景响应参数")]:
    reg_rows.append({"model":model,"term":term,"coefficient":coef,"std_error":"","p_value":p,"n":"","df_resid":"","r2":"","AICc":"","BIC":"","LOOCV_RMSE":"","status":note})
reg_rows.append({"model":f"F最终模型（本次复算lag={best_lag}）","term":f"dlnF_lag{best_lag}","coefficient":BETA_F,"std_error":f_final["se"][1],"p_value":f_final["p"][1],"n":f_final["n"],"df_resid":f_final["df_resid"],"r2":f_final["r2"],"AICc":f_final["aicc"],"BIC":f_final["bic"],"LOOCV_RMSE":f_final["loocv_rmse"],"status":f"HAC(maxlags=1)；年份{';'.join(map(str,f_final_years))}；Shock2020可识别={shock_identifiable}"})
write_csv("q3_driver_model_results.csv",reg_rows)

param_rows=[]
for s,p in PARAMS.items():
    param_rows.append({"scenario":s,"M_growth":p["M"],"Q_2026":p["Q"][0],"Q_2027_2030":p["Q"][1],"F_growth":p["F"],"U_increment":p["U"]})
write_csv("q3_scenario_parameters.csv",param_rows)

svg_line(FIG/"01_旅游综合收入三情景.svg","2026—2030年旅游综合收入三情景","亿元",years,paths_rev,['#dc2626','#2563eb','#16a34a'])
svg_line(FIG/"02_游客接待量三情景.svg","2026—2030年游客接待量三情景","万人次",years,paths_arr,['#dc2626','#2563eb','#16a34a'])
stress_series={"基准":[base_rev[y] for y in years],"一般重大事件":[float(r["revenue_100m_cny"]) for r in stress_rows if r["stress"].startswith("一般")],"疫情级极端事件":[float(r["revenue_100m_cny"]) for r in stress_rows if r["stress"].startswith("疫情级")]}
svg_line(FIG/"03_突发事件压力测试.svg","旅游综合收入突发事件压力测试（非概率预测）","亿元",years,stress_series,['#2563eb','#f59e0b','#dc2626'])
rank20=sorted([r for r in sens_rows if r["perturbation"]==0.20],key=lambda r:r["normalized_sensitivity"],reverse=True)
svg_bar(FIG/"04_MQF归一化敏感度.svg","M、Q、F的±20%标准化OAT敏感度",[r["factor"] for r in rank20],[r["normalized_sensitivity"] for r in rank20],"五年归一化敏感度")
svg_single_with_gaps(FIG/"05_F财政支出原始序列.svg","文化旅游体育与传媒财政支出原始序列","万元",list(range(2010,2025)),[fiscal.get(t) for t in range(2010,2025)],'#2563eb')
svg_single_with_gaps(FIG/"06_F对数增长率.svg","财政投入对数增长率 ΔlnF","ΔlnF",list(range(2011,2025)),[dln_f.get(t) for t in range(2011,2025)],'#7c3aed')
svg_bar(FIG/"07_F滞后模型_LOOCV比较.svg","F的0/1/2年滞后模型LOOCV比较",[f"lag={r['lag']}" for r in lag_rows],[r['LOOCV_RMSE'] for r in lag_rows],"LOOCV RMSE（越低越好）",percent=False)
svg_line(FIG/"08_F最终模型实际与拟合.svg",f"F最终lag={best_lag}模型：实际与拟合ΔlnY","ΔlnY",f_final_years,{"实际":[dln_y_f[t] for t in f_final_years],"拟合":[float(v) for v in f_final['fitted']]},['#2563eb','#f59e0b'])
svg_bar(FIG/"09_F双向Granger检验.svg","F与旅游收入双向Granger辅助检验",[r['test'] for r in granger_rows],[r['p_value'] for r in granger_rows],"p值")
svg_tornado(FIG/"10_MQF_2030龙卷风图.svg",rank20)

summary={
    "visitor_scenario_scale":visitor_scale,
    "M_refit":{"years":m_years,"coefficient_dlnM":float(m_fit["coef"][1]),"p_dlnM":float(m_fit["p"][1]),"coefficient_shock2020":float(m_fit["coef"][2]),"p_shock2020":float(m_fit["p"][2]),"n":m_fit["n"],"r2":m_fit["r2"]},
    "covid_log_shock":d_covid,
    "F_revision":{"best_lag_by_AICc":best_lag,"beta_F":BETA_F,"HAC_SE":float(f_final['se'][1]),"p_value":float(f_final['p'][1]),"years":f_final_years,"shock2020_identifiable":shock_identifiable,"lag_comparison":lag_rows,"granger":granger_rows},
    "general_event_loss":1-math.exp(0.5*d_covid),
    "pandemic_event_loss":1-math.exp(d_covid),
    "sensitivity_rank_20pct":[r["factor"] for r in rank20],
    "uq_link_normalized_sensitivity":u_norm,
}
(OUT/"q3_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

md=f"""# 第三问情景预测、驱动识别与敏感性分析结果

## 1. 执行口径

- 基准路径严格继承第二问的2026—2030年ARIMA(0,1,0)漂移预测，不重新拟合另一套趋势模型。
- 收入三情景按修正版方案在年度对数增长率上修正，并从2025年实际值200亿元逐年递推。
- 原修正版方案要求M/Q驱动回归不使用2021、2022年旅游收入；仓库官方台账已有110亿元和160亿元。M模型仍按原方案排除；本次新版F建议改为“真实可用值即可进入”，因此F重估使用这两个官方值，但受F缺失影响它们不一定进入所有滞后模型。
- M单因素模型已由仓库数据重新计算；Q和MQ联合模型因仓库缺少逐年A级景区/重点村镇构成表，只能使用方案给定系数，已明确标为“未独立复算”。
- F已按新版建议重算：原始财政支出先转换为ΔlnF，0/1/2阶在同一共同样本比较，再用最佳阶数全部有效年份重估，统一报告HAC(maxlags=1)标准误。2010—2018仍是阶段性整理值，结论保持审慎。

## 2. 主要结果

| 年份 | 悲观收入（亿元） | 基准收入（亿元） | 乐观收入（亿元） | 悲观游客（万人次） | 基准游客（万人次） | 乐观游客（万人次） |
|---:|---:|---:|---:|---:|---:|---:|
"""
for i,t in enumerate(years):
    md+=f"| {t} | {paths_rev['宏观下行悲观'][i]:.2f} | {paths_rev['基准'][i]:.2f} | {paths_rev['乐观'][i]:.2f} | {paths_arr['宏观下行悲观'][i]:.2f} | {paths_arr['基准'][i]:.2f} | {paths_arr['乐观'][i]:.2f} |\n"
md+=f"""

### 驱动识别

- M模型复算：β_M={m_fit['coef'][1]:.4f}，p={m_fit['p'][1]:.4f}，n={m_fit['n']}，R²={m_fit['r2']:.4f}。系数可精确复现方案中的4.0029，但普通OLS双侧p值并非方案所写的0.010，而是0.1961；因此只能保留正向方向证据，不能写成“显著”。
- Q单因素与MQ联合结果暂沿用方案：β_Q=0.4345；联合响应参数β_M=2.4225、β_Q=0.2646。
- F共同样本为{','.join(map(str,f_common_years))}。按AICc选出的最佳阶数为lag={best_lag}；使用该阶全部有效年份重估后，β_F={BETA_F:.4f}，HAC SE={f_final['se'][1]:.4f}，p={f_final['p'][1]:.4f}，参与年份为{','.join(map(str,f_final_years))}。旧β_F=0.0787已完全删除。
- 共同样本与最长Granger连续段均不包含2020年，Shock2020列恒为0，无法估计疫情系数；文件明确记录这一识别限制，没有伪造控制结果。
- 游客情景修正幅度没有拍脑袋另设系数，而采用正常相邻年份中“游客对数增长波动/收入对数增长波动”的比值{visitor_scale:.4f}对收入修正幅度进行缩放。

### 压力测试

- 第二问COVID收入对数冲击D={d_covid:.4f}。
- 一般重大事件取0.5D，对应一次性水平损失{1-math.exp(0.5*d_covid):.2%}。
- 疫情级极端事件取完整D，对应一次性水平损失{1-math.exp(d_covid):.2%}。
- 两档压力路径均为压力测试，不代表发生概率。

### 敏感性排序

使用新β_F、把财政普通增长率转换为ln(1+g_F)，并按lag={best_lag}向后传导后，M、Q、F在基准参数±20%扰动下的五年归一化敏感度排序为：{' > '.join(r['factor'] for r in rank20)}。U不估计独立弹性，U–Q联动高低状态的归一化敏感度为{u_norm:.2%}。

## 3. 外部校准结论

蓟州区2026年政府工作部署中的“旅游综合收入增长8%”对应216亿元。第二问基准预测本身为224.43亿元（增长12.22%），已经高于8%；因此无法仅靠“收缩乐观外生系数”让乐观值降至8%而仍保持乐观路径高于基准。本文将8%作为官方目标参照线，不把它当作硬上限，并在论文中说明Q2基准与官方目标存在3.90%的水平差异。

## 4. 当前仍缺的关键数据

1. 逐年A级景区数量及等级、全国重点村镇数量：缺失导致Q、MQ模型和Q阶梯跳变留一法无法独立复算。
2. 2010—2018财政支出的逐年官方原表/URL，以及2019—2021、2025财政值：当前F证据链不足。
3. 国家等级民宿（甲/乙/丙）和新消费场景逐年明细：U状态可用于情景，但不能复算完整指数。
4. 第一问疫情反事实对数冲击的统一口径值：本次只报告Q2主压力，未伪造Q1更严格上界。

## 5. 可写入论文的定量建议

1. 优先扩大京津客源承接：以基准M年增3.77%为底线，乐观目标4.94%，并同步监测京津游客实际转化率。
2. 推进供给升级：基准ΔQ=0.216/年，乐观路径2026年0.283、2027—2030年0.308；补齐Q原始构成后再把目标转译为景区和重点村镇数量。
3. 财政投入按lag={best_lag}提前布局：情景中基准增速7.5%、乐观10%，进入模型时分别使用ln(1.075)、ln(1.10)，不能把项目总投资10亿元当财政支出。
4. 新业态与供给联动考核：U基准增量0.900，高速状态1.131，作为精品民宿、等级民宿与新消费场景的综合扩张状态。
5. 建立韧性预案：一般重大事件按约{1-math.exp(0.5*d_covid):.1%}收入损失、疫情级事件按约{1-math.exp(d_covid):.1%}收入损失配置应急方案。
"""
(OUT/"第三问建模结果说明.md").write_text(md,encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
