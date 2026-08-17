import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const root='D:/2026MMC', out=`${root}/outputs/q1_final_rebuilt`;
const parse=async(name)=>{const s=await fs.readFile(`${out}/${name}`,'utf8');return s.replace(/^\uFEFF/,'').trim().split(/\r?\n/).map(line=>{let a=[],v='',q=false;for(let i=0;i<line.length;i++){let c=line[i];if(c==='"'){if(q&&line[i+1]==='"'){v+='"';i++;}else q=!q;}else if(c===','&&!q){a.push(v);v='';}else v+=c;}a.push(v);return a.map((x,j)=>x===''||x.toLowerCase()==='nan'?'':(j===0?x:(Number.isFinite(Number(x))?Number(x):x)));});};
const wb=Workbook.create(); wb.comments.setSelf({displayName:'User'});
const theme={head:'#17365D',sub:'#D9EAF7',input:'#FFF2CC',note:'#F3F4F6'};
function addSheet(name,data,widths=[]){const s=wb.worksheets.add(name);s.getRangeByIndexes(0,0,data.length,data[0].length).values=data;s.getRangeByIndexes(0,0,1,data[0].length).format={fill:theme.head,font:{bold:true,color:'#FFFFFF'},wrapText:true};s.freezePanes.freezeRows(1);for(let i=0;i<data[0].length;i++)s.getRangeByIndexes(0,i,data.length,1).format.columnWidth=widths[i]||16;s.getUsedRange().format.rowHeight=20;return s;}
const input=await parse('q1_model_input.csv'), corr=await parse('q1_correlation_imputation.csv'), comp=await parse('q1_model_comparison.csv'), hold=await parse('q1_2025_holdout.csv');
const intro=[['第一问重建结果工作簿','内容'],['核心口径','官方原值不覆盖；插补值另列并带状态'],['游客量缺失','2021=1713.40、2022=2203.23 万人次（人均旅游收入线性插补）'],['敏感性','自然样条：2021=1824.61、2022=2237.85 万人次'],['相关性','严格官方配对 Pearson r=0.8992；加入2010二级来源后约0.9152'],['模型','M0疫情前指数趋势；M1统一疫情虚拟变量；M2增加恢复期虚拟变量'],['检验','2025完全留出；旅游2025值为官方媒体补充，证据层级低于政府统计公报'],['重要结论','旅游指标的阶段指数模型对2025严重高估，说明机械外推不适合恢复期转折，不能把高拟合度当预测能力。']];
const s0=addSheet('说明与结论',intro,[24,90]);s0.getRange('A1:B1').format.rowHeight=30;s0.getRange('A2:A8').format={fill:theme.sub,font:{bold:true}};s0.getRange('B2:B8').format.wrapText=true;s0.getRange('A1:B8').format.verticalAlignment='center';
const s1=addSheet('模型输入',input,[10,15,15,15,15,16,16,18,16,16,24,20]);s1.getRange('B2:J17').format.numberFormat='0.00';
const s2=addSheet('相关与插补',corr,[34,18,68]);s2.getRange('B2:B10').format.numberFormat='0.0000';
const s3=addSheet('模型比较',comp,[23,10,8,15,15,15,15,18,12]);s3.getRange('D2:H13').format.numberFormat='0.0000';
const s4=addSheet('2025留出检验',hold,[24,14,18,18,18,18,18,14]);s4.getRange('C2:G5').format.numberFormat='0.00';s4.getRange('H2:H5').format.numberFormat='0.0%';
// Native workbook charts, each on its own sheet.
for(const [nm,title,cols,unit] of [['图_游客量','接待游客量：主插补与自然样条',[0,6,7],'万人次'],['图_GDP','GDP：原值与断点衔接值',[0,1,2],'亿元'],['图_第三产业','第三产业：原值与断点衔接值',[0,3,4],'亿元']]){
 const s=wb.worksheets.add(nm);const data=input.map(r=>cols.map(c=>r[c]));s.getRangeByIndexes(0,0,data.length,data[0].length).values=data;s.getRange('A1:C1').format={fill:theme.head,font:{bold:true,color:'#fff'}};const ch=s.charts.add('line',s.getRange(`A1:C${data.length}`));ch.title=title;ch.hasLegend=true;ch.xAxis={axisType:'textAxis'};ch.yAxis={numberFormatCode:'#,##0'};ch.yAxis.title.text=unit;ch.setPosition('E2','N24');s.getRange('A:C').format.columnWidth=17;
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',options:{useRegex:true,maxResults:100},summary:'formula error scan'});console.log(errors.ndjson);
for(const name of ['说明与结论','模型输入','相关与插补','模型比较','2025留出检验','图_游客量','图_GDP','图_第三产业']){const b=await wb.render({sheetName:name,autoCrop:'all',scale:1,format:'png'});await fs.writeFile(`${out}/preview_${name}.png`,new Uint8Array(await b.arrayBuffer()));}
const x=await SpreadsheetFile.exportXlsx(wb);await x.save(`${out}/第一问_指数模型_重建结果.xlsx`);
console.log((await wb.inspect({kind:'table',range:'说明与结论!A1:B8',include:'values,formulas',tableMaxRows:10,tableMaxCols:3})).ndjson);
