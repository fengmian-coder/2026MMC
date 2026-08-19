import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root='D:/2026MMC';
const out=`${root}/outputs/q3_results`;

function parseCSV(text){
  const rows=[]; let row=[],cell='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"' && quoted && text[i+1]==='"'){cell+='"';i++;}
    else if(c==='"') quoted=!quoted;
    else if(c===','&&!quoted){row.push(cell);cell='';}
    else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);cell='';if(row.some(x=>x!==''))rows.push(row);row=[];}
    else cell+=c;
  }
  if(cell||row.length){row.push(cell);rows.push(row);}
  return rows.map((r,ri)=>r.map(v=>ri===0?v:(v!==''&&!Number.isNaN(Number(v))?Number(v):v)));
}

const files={
  '情景预测':'q3_scenario_forecast_2026_2030.csv',
  '情景参数':'q3_scenario_parameters.csv',
  '压力测试':'q3_event_stress_tests.csv',
  '敏感性':'q3_sensitivity_analysis.csv',
  '驱动模型':'q3_driver_model_results.csv',
  'F原始与变换':'q3_F_raw_transformed.csv',
  'F滞后共同样本':'q3_F_lag_common_sample.csv',
  'F滞后比较':'q3_F_lag_comparison.csv',
  'F最终HAC模型':'q3_F_final_model.csv',
  'F实际拟合':'q3_F_actual_fitted.csv',
  'F双向Granger':'q3_F_granger_dynamic.csv',
};
const data={};
for(const [s,f] of Object.entries(files)) data[s]=parseCSV(await fs.readFile(`${out}/${f}`,'utf8'));

const wb=Workbook.create();
const navy='#17365D', blue='#D9EAF7', pale='#F3F6FA', green='#E2F0D9', red='#FCE4D6';
function styleSheet(sh,rows,cols){
  sh.showGridLines=false; sh.freezePanes.freezeRows(1);
  const used=sh.getRangeByIndexes(0,0,rows,cols);
  used.format.font={name:'Microsoft YaHei',size:10,color:'#1F2937'};
  sh.getRangeByIndexes(0,0,1,cols).format={fill:navy,font:{name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'},wrapText:true,verticalAlignment:'center'};
  sh.getRangeByIndexes(0,0,rows,cols).format.borders={insideHorizontal:{style:'thin',color:'#D9E2F3'},bottom:{style:'thin',color:'#A6A6A6'}};
  used.format.autofitColumns(); used.format.autofitRows();
  for(let c=0;c<cols;c++){
    const rg=sh.getRangeByIndexes(0,c,rows,1);
    if(rg.format.columnWidth>26) rg.format.columnWidth=26;
  }
}
for(const [name,rows] of Object.entries(data)){
  const sh=wb.worksheets.add(name); sh.getRangeByIndexes(0,0,rows.length,rows[0].length).values=rows;
  styleSheet(sh,rows.length,rows[0].length);
}

const fc=wb.worksheets.getItem('情景预测');
fc.getRange('C2:D16').format.numberFormat='#,##0.00';
fc.getRange('A2:A16').format.numberFormat='0';
fc.getRange('A1:D16').conditionalFormats.addCustom('=$B2="乐观"',{fill:green});
fc.getRange('A1:D16').conditionalFormats.addCustom('=$B2="宏观下行悲观"',{fill:red});

const pp=wb.worksheets.getItem('情景参数'); pp.getRange('B2:E4').format.numberFormat='0.00%'; pp.getRange('F2:F4').format.numberFormat='0.000';
const st=wb.worksheets.getItem('压力测试'); st.getRange('C2:C11').format.numberFormat='0.0000'; st.getRange('D2:D11').format.numberFormat='#,##0.00'; st.getRange('E2:E11').format.numberFormat='0.00%';
const se=wb.worksheets.getItem('敏感性'); se.getRange('C2:D8').format.numberFormat='#,##0.00'; se.getRange('E2:F8').format.numberFormat='0.00%'; se.getRange('G2:G8').format.numberFormat='#,##0.00'; se.getRange('H2:H8').format.numberFormat='0.00%';
const dm=wb.worksheets.getItem('驱动模型'); dm.getRange('C2:L8').format.numberFormat='0.0000';
wb.worksheets.getItem('F原始与变换').getRange('B2:D17').format.numberFormat='0.0000';
wb.worksheets.getItem('F滞后共同样本').getRange('B2:F7').format.numberFormat='0.0000';
wb.worksheets.getItem('F滞后比较').getRange('E2:M4').format.numberFormat='0.0000';
wb.worksheets.getItem('F最终HAC模型').getRange('B2:P4').format.numberFormat='0.0000';
wb.worksheets.getItem('F实际拟合').getRange('B2:D20').format.numberFormat='0.0000';
wb.worksheets.getItem('F双向Granger').getRange('D2:G3').format.numberFormat='0.0000';

const readme=wb.worksheets.add('结论与边界');
const notes=[
 ['第三问结果总览','说明'],
 ['基准预测','继承第二问ARIMA(0,1,0)漂移路径；2026/2030收入224.43/355.87亿元。'],
 ['三情景2030收入','悲观184.19亿元；基准355.87亿元；乐观463.42亿元。'],
 ['三情景2030游客','悲观2490.37万人次；基准3954.45万人次；乐观4766.44万人次。'],
 ['敏感性排序','用新F系数和ln(1+gF)重算后：M > Q > F；U-Q联动敏感度单独报告。'],
 ['M模型核验','βM=4.0029可复现，但普通OLS双侧p=0.1961，不是方案写的0.010。'],
 ['F模型修订','共同样本选lag=1；最终βF=0.07867，HAC p=0.0854；旧p=0.028不再使用。'],
 ['Granger辅助检验','F→Y p=0.2175，Y→F p=0.2397，均不显著；不作因果结论。'],
 ['压力测试','一般事件收入损失7.54%；疫情级事件损失14.52%；均非概率预测。'],
 ['2026官方目标校准','官方增长8%对应216亿元；Q2基准224.43亿元本身已高于目标，不能靠收缩乐观系数解决。'],
 ['关键数据缺口','逐年Q构成；财政逐年官方原表及缺失年；国家等级民宿和新消费场景明细；Q1疫情上界统一值。'],
 ['口径差异','M模型按原方案排除2021/2022；新版F模型按“真实可用值”规则允许使用官方110/160亿元。'],
];
readme.getRangeByIndexes(0,0,notes.length,2).values=notes; styleSheet(readme,notes.length,2);
readme.getRange('A1:B1').format.fill=navy; readme.getRange(`A2:A${notes.length}`).format={fill:blue,font:{name:'Microsoft YaHei',bold:true,color:'#17365D'}};
readme.getRange(`A1:A${notes.length}`).format.columnWidth=24; readme.getRange(`B1:B${notes.length}`).format.columnWidth=72; readme.getRange(`B2:B${notes.length}`).format.wrapText=true;
readme.getRange(`A1:B${notes.length}`).format.autofitRows();

// Native workbook charts, sourced directly from auditable result tables.
const chartData=wb.worksheets.add('图表数据'); chartData.showGridLines=false;
const forecastRows=data['情景预测'].slice(1); const scenarios=['宏观下行悲观','基准','乐观'];
const revenueChart=[['年份',...scenarios]], arrivalsChart=[['年份',...scenarios]];
for(const year of [2026,2027,2028,2029,2030]){
  const yr=forecastRows.filter(r=>r[0]===year);
  revenueChart.push([year,...scenarios.map(s=>yr.find(r=>r[1]===s)[3])]);
  arrivalsChart.push([year,...scenarios.map(s=>yr.find(r=>r[1]===s)[2])]);
}
chartData.getRange('A1:D6').values=revenueChart;
chartData.getRange('F1:I6').values=arrivalsChart;
styleSheet(chartData,6,9); chartData.getRange('B2:D6').format.numberFormat='#,##0.00'; chartData.getRange('G2:I6').format.numberFormat='#,##0.00';
const c1=chartData.charts.add('line',chartData.getRange('A1:D6')); c1.title='2026—2030年旅游综合收入三情景（亿元）'; c1.hasLegend=true; c1.yAxis={numberFormatCode:'#,##0'}; c1.setPosition('A8','H24');
const c2=chartData.charts.add('line',chartData.getRange('F1:I6')); c2.title='2026—2030年游客接待量三情景（万人次）'; c2.hasLegend=true; c2.yAxis={numberFormatCode:'#,##0'}; c2.setPosition('I8','P24');

await fs.mkdir(out,{recursive:true});
const xlsx=await SpreadsheetFile.exportXlsx(wb); await xlsx.save(`${out}/第三问_情景预测与敏感性分析结果.xlsx`);
for(const s of ['结论与边界','情景预测','敏感性','图表数据']){
  const img=await wb.render({sheetName:s,autoCrop:'all',scale:1.2,format:'png'});
  await fs.writeFile(`${out}/preview_${s}.png`,new Uint8Array(await img.arrayBuffer()));
}
console.log((await wb.inspect({kind:'table',sheetId:'结论与边界',range:`A1:B${notes.length}`,include:'values,formulas',tableMaxRows:15,tableMaxCols:4})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',options:{useRegex:true,maxResults:100},summary:'final formula error scan'})).ndjson);
