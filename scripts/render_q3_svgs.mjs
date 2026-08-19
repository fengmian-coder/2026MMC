import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

const dir = 'D:/2026MMC/outputs/q3_results/figures';
for (const name of await fs.readdir(dir)) {
  if (!name.endsWith('.svg')) continue;
  const input = path.join(dir, name);
  const output = path.join(dir, name.replace(/\.svg$/i, '.png'));
  await sharp(await fs.readFile(input), { density: 180 }).png().toFile(output);
  console.log(output);
}
