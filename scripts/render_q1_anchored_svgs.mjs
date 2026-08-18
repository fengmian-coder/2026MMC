import fs from 'node:fs/promises';
import sharp from 'sharp';
const dir='D:/2026MMC/outputs/q1_anchored_revision/figures';
for(const n of ['01_游客量锚定预测','02_旅游收入锚定预测'])await sharp(await fs.readFile(`${dir}/${n}.svg`)).png().toFile(`${dir}/${n}.png`);
