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
const se=wb.worksheets.getItem('敏感性'); se.getRange('C2:H8').format.numberFormat='0.00%'; se.getRange('C2:D8').format.numberFormat='#,##0.00'; se.getRange('G2:G8').format.numberFormat='#,##0.00';
const dm=wb.worksheets.getItem('驱动模型'); dm.getRange('C2:L8').format.numberFormat='0.0000';

const readme=wb.worksheets.add('结论与边界');
const notes=[
 ['第三问结果总览','说明'],
 ['基准预测','继承第二问ARIMA(0,1,0)漂移路径；2026/2030收入224.43/355.87亿元。'],
 ['三情景2030收入','悲观184.03亿元；基准355.87亿元；乐观463.71亿元。'],
 ['三情景2030游客','悲观2490.37万人次；基准3954.45万人次；乐观4766.44万人次。'],
 ['敏感性排序','M > Q > F；U-Q联动敏感度单独报告，不视为U独立弹性。'],
 ['M模型核验','βM=4.0029可复现，但普通OLS双侧p=0.1961，不是方案写的0.010。'],
 ['压力测试','一般事件收入损失7.54%；疫情级事件损失14.52%；均非概率预测。'],
 ['2026官方目标校准','官方增长8%对应216亿元；Q2基准224.43亿元本身已高于目标，不能靠收缩乐观系数解决。'],
 ['关键数据缺口','逐年Q构成；财政逐年官方原表及缺失年；国家等级民宿和新消费场景明细；Q1疫情上界统一值。'],
 ['口径差异','仓库有2021/2022收入110/160亿元，但按修正版方案在第三问驱动回归中排除。'],
];
readme.getRangeByIndexes(0,0,notes.length,2).values=notes; styleSheet(readme,notes.length,2);
readme.getRange('A1:B1').format.fill=navy; readme.getRange('A2:A10').format={fill:blue,font:{name:'Microsoft YaHei',bold:true,color:'#17365D'}};
readme.getRange('A1:A10').format.columnWidth=24; readme.getRange('B1:B10').format.columnWidth=72; readme.getRange('B2:B10').format.wrapText=true;
readme.getRange('A1:B10').format.autofitRows();

// Native workbook charts, sourced directly from auditable result tables.
const chartData=wb.worksheets.add('图表数据'); chartData.showGridLines=false;
chartData.getRange('A1:D6').values=[['年份','宏观下行悲观','基准','乐观'],[2026,197.6295,224.4313,235.0131],[2027,194.1379,251.8471,278.5360],[2028,190.7079,282.6118,330.1192],[2029,187.3386,317.1347,391.2552],[2030,184.0287,355.8748,463.7132]];
chartData.getRange('F1:I6').values=[['年份','宏观下行悲观','基准','乐观'],[2026,2649.6215,2906.3521,3005.3735],[2027,2608.8792,3138.9380,3372.6583],[2028,2568.7634,3390.1371,3784.8289],[2029,2529.2645,3661.4389,4247.3705],[2030,2490.3729,3954.4520,4766.4390]];
styleSheet(chartData,6,9); chartData.getRange('B2:D6').format.numberFormat='#,##0.00'; chartData.getRange('G2:I6').format.numberFormat='#,##0.00';
const c1=chartData.charts.add('line',chartData.getRange('A1:D6')); c1.title='2026—2030年旅游综合收入三情景（亿元）'; c1.hasLegend=true; c1.yAxis={numberFormatCode:'#,##0'}; c1.setPosition('A8','H24');
const c2=chartData.charts.add('line',chartData.getRange('F1:I6')); c2.title='2026—2030年游客接待量三情景（万人次）'; c2.hasLegend=true; c2.yAxis={numberFormatCode:'#,##0'}; c2.setPosition('I8','P24');

await fs.mkdir(out,{recursive:true});
const xlsx=await SpreadsheetFile.exportXlsx(wb); await xlsx.save(`${out}/第三问_情景预测与敏感性分析结果.xlsx`);
for(const s of ['结论与边界','情景预测','敏感性','图表数据']){
  const img=await wb.render({sheetName:s,autoCrop:'all',scale:1.2,format:'png'});
  await fs.writeFile(`${out}/preview_${s}.png`,new Uint8Array(await img.arrayBuffer()));
}
console.log((await wb.inspect({kind:'table',sheetId:'结论与边界',range:'A1:B10',include:'values,formulas',tableMaxRows:12,tableMaxCols:4})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',options:{useRegex:true,maxResults:100},summary:'final formula error scan'})).ndjson);
