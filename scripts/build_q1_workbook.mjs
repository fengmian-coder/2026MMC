import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const ROOT = "D:/2026MMC";
const INPUT = path.join(ROOT, "data/processed/core_annual_2010_2025.csv");
const OUTDIR = path.join(ROOT, "outputs/q1_data_cleaning");
const OUT = path.join(OUTDIR, "q1_data_cleaning_and_visualization.xlsx");

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  return lines.slice(1).map(line => {
    const cells = line.split(",");
    return Object.fromEntries(header.map((h, i) => [h, cells[i] ?? ""]));
  });
}
const num = x => x === "" || x == null ? null : Number(x);
const round = (x, n=2) => x == null ? null : Number(x.toFixed(n));
function ols(xs, ys) {
  const n = xs.length, xb = xs.reduce((a,b)=>a+b,0)/n, yb = ys.reduce((a,b)=>a+b,0)/n;
  const slope = xs.reduce((s,x,i)=>s+(x-xb)*(ys[i]-yb),0) / xs.reduce((s,x)=>s+(x-xb)**2,0);
  const intercept = yb-slope*xb;
  const pred = xs.map(x=>intercept+slope*x);
  const sse = pred.reduce((s,p,i)=>s+(ys[i]-p)**2,0);
  const sst = ys.reduce((s,y)=>s+(y-yb)**2,0);
  return {intercept,slope,r2:1-sse/sst,rmse:Math.sqrt(sse/n),pred};
}
function metrics(actual, predicted) {
  const n=actual.length;
  const err=actual.map((v,i)=>predicted[i]-v);
  const mean=actual.reduce((a,b)=>a+b,0)/n;
  return {
    r2: 1-err.reduce((s,e)=>s+e*e,0)/actual.reduce((s,v)=>s+(v-mean)**2,0),
    rmse: Math.sqrt(err.reduce((s,e)=>s+e*e,0)/n),
    mae: err.reduce((s,e)=>s+Math.abs(e),0)/n,
    mape: err.reduce((s,e,i)=>s+Math.abs(e/actual[i]),0)/n*100,
  };
}

const raw = parseCsv(await fs.readFile(INPUT, "utf8")).map(r => ({
  year:Number(r.year), arrivals:num(r.tourist_arrivals), arrivalsStatus:r.tourist_arrivals_status,
  revenue:num(r.tourism_comprehensive_revenue), revenueStatus:r.tourism_comprehensive_revenue_status,
  gdp:num(r.gdp), gdpStatus:r.gdp_status, tertiary:num(r.tertiary_industry_value_added),
  tertiaryStatus:r.tertiary_industry_value_added_status, covid:Number(r.covid_dummy),
  break2019:Number(r.gdp_definition_break_2019)
}));

// Cross-indicator model uses only paired official observations.
const paired = raw.filter(r=>r.arrivals!=null && r.revenue!=null);
const cross = ols(paired.map(r=>Math.log(r.revenue)), paired.map(r=>Math.log(r.arrivals)));
const audit=[];
const clean = raw.map(r=>({...r, arrivalsClean:r.arrivals, revenueClean:r.revenue,
  arrivalsFlag:r.arrivals==null?"待处理":"官方实测", revenueFlag:r.revenue==null?"待处理":"官方实测"}));
const byYear = y => clean.find(r=>r.year===y);
function setVal(year, field, value, method, basis, low=null, high=null) {
  const r=byYear(year), isA=field==="arrivals";
  r[isA?"arrivalsClean":"revenueClean"] = value;
  r[isA?"arrivalsFlag":"revenueFlag"] = year===2025?"趋势预测":"建模插补";
  audit.push([year,isA?"游客接待量":"旅游综合收入",round(value),isA?"万人次":"亿元",year===2025?"趋势预测":"缺失插补",method,basis,round(low),round(high)]);
}

setVal(2012,"arrivals",Math.sqrt(byYear(2011).arrivals*byYear(2013).arrivals),"相邻年份对数线性插值","2011与2013官方实测值");
setVal(2012,"revenue",Math.sqrt(byYear(2011).revenue*byYear(2013).revenue),"相邻年份对数线性插值","2011与2013官方实测值");
// Estimate 2010 revenue using paired official observations only.
const revFromArr = ols(paired.map(r=>Math.log(r.arrivals)), paired.map(r=>Math.log(r.revenue)));
const rev2010=Math.exp(revFromArr.intercept+revFromArr.slope*Math.log(byYear(2010).arrivals));
setVal(2010,"revenue",rev2010,"游客量—收入对数回归","10组官方配对年度；仅用于填补",rev2010*Math.exp(-revFromArr.rmse),rev2010*Math.exp(revFromArr.rmse));
const rev2020=Math.sqrt(byYear(2019).revenue*byYear(2021).revenue);
setVal(2020,"revenue",rev2020,"相邻年份对数线性插值","2019与2021官方实测值",Math.min(byYear(2019).revenue,byYear(2021).revenue),Math.max(byYear(2019).revenue,byYear(2021).revenue));
for (const y of [2020,2021,2022]) {
  const rv=byYear(y).revenueClean, av=Math.exp(cross.intercept+cross.slope*Math.log(rv));
  setVal(y,"arrivals",av,"收入—游客量对数回归","10组官方配对年度；疫情期估计",av*Math.exp(-cross.rmse),av*Math.exp(cross.rmse));
}
const revGrowth=Math.sqrt(byYear(2024).revenue/byYear(2022).revenue);
setVal(2025,"revenue",byYear(2024).revenue*revGrowth,"恢复期几何平均增速外推","2022—2024官方收入，预测不视为实测");
const arrGrowth=byYear(2024).arrivals/byYear(2023).arrivals;
setVal(2025,"arrivals",byYear(2024).arrivals*arrGrowth,"最近年度增速外推","2023—2024官方游客量，预测不视为实测");

for (let i=0;i<clean.length;i++) {
  const r=clean[i], p=clean[i-1];
  r.arrivalsYoy=p?(r.arrivalsClean/p.arrivalsClean-1):null;
  r.revenueYoy=p?(r.revenueClean/p.revenueClean-1):null;
  r.gdpYoy=p?(r.gdp/p.gdp-1):null;
  r.tertiaryYoy=p?(r.tertiary/p.tertiary-1):null;
  r.spendPerTrip=r.revenueClean*10000/r.arrivalsClean;
  r.stage=r.year<=2019?"疫情前":r.year<=2022?"疫情冲击期":r.year<=2024?"恢复期":"预测期";
  r.anomaly=(r.year===2019?"GDP口径断点；不可直接比较增速":r.year>=2020&&r.year<=2022?"疫情冲击；不宜套用长期单趋势":r.year===2025?"预测值；需后续用实测更新":"");
}

function growthModel(field, years) {
  const rows=clean.filter(r=>years.includes(r.year));
  const fit=ols(rows.map(r=>r.year-years[0]),rows.map(r=>Math.log(r[field])));
  const predicted=rows.map(r=>Math.exp(fit.intercept+fit.slope*(r.year-years[0])));
  const m=metrics(rows.map(r=>r[field]),predicted);
  return {annualGrowth:(Math.exp(fit.slope)-1)*100,...m};
}
const preYears=Array.from({length:10},(_,i)=>2010+i);
const modelRows=[
  ["游客接待量","疫情前指数增长模型",...Object.values(growthModel("arrivalsClean",preYears)),"适合描述2010—2019长期扩张；不适合直接外推疫情冲击与恢复期"],
  ["旅游综合收入","疫情前指数增长模型",...Object.values(growthModel("revenueClean",preYears)),"2010为回归补值；适合基准趋势，不适合疫情期"],
  ["游客接待量","恢复期指数增长模型",...Object.values(growthModel("arrivalsClean",[2023,2024])),"仅2个实测点，拟合优度没有比较意义；2025预测不确定性高"],
  ["旅游综合收入","恢复期指数增长模型",...Object.values(growthModel("revenueClean",[2022,2023,2024])),"恢复期样本短，适合作情景外推而非长期预测"]
];

const wb=Workbook.create();
const data=wb.worksheets.add("清洗数据"), aud=wb.worksheets.add("补缺审计"), model=wb.worksheets.add("模型评价"), dash=wb.worksheets.add("图表"), notes=wb.worksheets.add("说明");
const headers=["年份","游客量官方值(万人次)","游客量建模值(万人次)","游客量性质","旅游收入官方值(亿元)","旅游收入建模值(亿元)","收入性质","GDP(亿元)","第三产业增加值(亿元)","人均次旅游消费(元/人次)","游客量同比","旅游收入同比","GDP同比","第三产业同比","阶段","异常/口径提示"];
data.getRange(`A1:P${clean.length+1}`).values=[headers,...clean.map(r=>[r.year,r.arrivals,round(r.arrivalsClean),r.arrivalsFlag,r.revenue,round(r.revenueClean),r.revenueFlag,round(r.gdp),round(r.tertiary),round(r.spendPerTrip),r.arrivalsYoy,r.revenueYoy,r.gdpYoy,r.tertiaryYoy,r.stage,r.anomaly])];
aud.getRange(`A1:J${audit.length+1}`).values=[["年份","指标","建模值","单位","性质","处理方法","数据依据","敏感性下限","敏感性上限","使用提醒"],...audit.map(r=>[...r,"严禁作为官方实测值引用；论文须注明处理方法"] )];
model.getRange("A1:H5").values=[["指标","模型","年均增长率(%)","R²","RMSE","MAE","MAPE(%)","适用性评价"],...modelRows.map(r=>[r[0],r[1],round(r[2]),round(r[3],4),round(r[4]),round(r[5]),round(r[6]),r[7]])];

dash.getRange("A1:E17").values=[["年份","游客量(万人次)","旅游收入(亿元)","GDP(亿元)","第三产业(亿元)"],...clean.map(r=>[r.year,round(r.arrivalsClean),round(r.revenueClean),round(r.gdp),round(r.tertiary)])];
dash.getRange("A20:D36").values=[["年份","游客量同比(%)","旅游收入同比(%)","人均次消费(元)"],...clean.map(r=>[r.year,r.arrivalsYoy==null?null:round(r.arrivalsYoy*100),r.revenueYoy==null?null:round(r.revenueYoy*100),round(r.spendPerTrip)])];
dash.getRange("A39:C55").values=[["年份","GDP(亿元)","第三产业(亿元)"],...clean.map(r=>[r.year,round(r.gdp),round(r.tertiary)])];
dash.getRange("E39:F55").values=[["年份","人均次旅游消费(元)"],...clean.map(r=>[r.year,round(r.spendPerTrip)])];
const c1=dash.charts.add("line",dash.getRange("A1:C17")); c1.setPosition("G2","P18"); c1.title="蓟州旅游规模趋势（含明确标记的建模值）"; c1.hasLegend=true;
const c2=dash.charts.add("line",dash.getRange("A39:C55")); c2.setPosition("Q2","Z18"); c2.title="经济与第三产业趋势"; c2.hasLegend=true;
const c3=dash.charts.add("line",dash.getRange("A20:C36")); c3.setPosition("G20","P36"); c3.title="游客量与旅游收入同比变化"; c3.hasLegend=true;
const c4=dash.charts.add("line",dash.getRange("E39:F55")); c4.setPosition("Q20","Z36"); c4.title="人均次旅游消费"; c4.hasLegend=false;

notes.getRange("A1:B10").values=[
  ["项目","第一问数据清洗与可视化说明"],
  ["原始数据","data/processed/core_annual_2010_2025.csv；其上游均来自仓库中的官方来源台账"],
  ["基本原则","官方原值不覆盖；补缺值、预测值分别标记；所有处理记录见“补缺审计”"],
  ["2012缺失","相邻官方年度做对数线性插值，适用于孤立缺失点"],
  ["2020收入","2019与2021官方值对数插值，仅作连续建模输入"],
  ["2020—2022游客量","用10组游客量与收入均为官方值的年份建立对数回归；交叉模型R²="+round(cross.r2,4)],
  ["2025","仅作恢复趋势外推，不作为官方实测；答题时建议同时报告情景敏感性"],
  ["GDP口径","2019存在统计口径断点，跨断点同比不作经济含义解释"],
  ["适用范围","当前年度表足以完成第一问的趋势、增速、结构关系和简单模型；因旅游缺失较多，结论需披露不确定性"],
  ["更新方式","获取新增官方实测后，替换原始核心表对应空值并重新运行本脚本"]
];

function style(sheet, range, color="#1F4E78") {
  const r=sheet.getRange(range); r.format.fill=color; r.format.font={bold:true,color:"#FFFFFF"}; r.format.horizontalAlignment="center";
}
for (const sh of [data,aud,model,dash,notes]) { sh.showGridLines=false; sh.freezePanes.freezeRows(1); }
style(data,"A1:P1"); style(aud,"A1:J1"); style(model,"A1:H1"); style(dash,"A1:E1"); style(dash,"A20:D20"); style(notes,"A1:B1");
data.getRange("K2:N17").format.numberFormat="0.0%"; dash.getRange("B21:C36").format.numberFormat="0.0";
for (const sh of [data,aud,model,dash,notes]) sh.getUsedRange().format.autofitColumns();
data.getRange("P1:P17").format.columnWidth=32; aud.getRange("F1:J9").format.columnWidth=28; model.getRange("H1:H5").format.columnWidth=48; notes.getRange("B1:B10").format.columnWidth=70;

await fs.mkdir(OUTDIR,{recursive:true});
const xlsx=await SpreadsheetFile.exportXlsx(wb); await xlsx.save(OUT);
const preview=await wb.render({sheetName:"图表",autoCrop:"all",scale:1,format:"png"});
await fs.writeFile(path.join(OUTDIR,"q1_dashboard_preview.png"),new Uint8Array(await preview.arrayBuffer()));
const check=await wb.inspect({kind:"sheet",include:"id,name",maxChars:3000});
console.log(JSON.stringify({output:OUT,auditCount:audit.length,crossR2:round(cross.r2,4),inspect:check.ndjson},null,2));
