import fs from 'node:fs/promises';
import sharp from 'sharp';
const dir='D:/2026MMC/outputs/q2_revised/figures';
for(const name of ['01_游客量基准预测','02_旅游收入基准预测','03_游客量_ACF','03_游客量_PACF','04_旅游收入_ACF','04_旅游收入_PACF','05_游客量_Hurst_原对数','05_游客量_Hurst_差分','06_旅游收入_Hurst_原对数','06_旅游收入_Hurst_差分']){
  await sharp(await fs.readFile(`${dir}/${name}.svg`)).png().toFile(`${dir}/${name}.png`);
}
