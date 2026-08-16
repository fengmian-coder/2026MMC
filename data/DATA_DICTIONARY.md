# 蓟州区旅游经济数据字典

这份文件用于让论文手、建模手和编程手统一理解数据。机器可读取的完整对照表见 `processed/data_dictionary.csv`。

## 一、四项核心指标

| 字段名 | 中文名称 | 单位 | 作用 | 重要说明 |
|---|---|---:|---|---|
| `tourist_arrivals` | 年度游客接待量 | 万人次 | 问题1趋势分析、问题2预测目标 | 统计的是人次，不是不同游客人数；缺失值不能当0 |
| `tourism_comprehensive_revenue` | 旅游综合收入 | 亿元 | 问题1趋势分析、问题2预测目标 | 与旅游直接收入不同，二者不能混用 |
| `gdp` | 地区生产总值 | 亿元 | 宏观经济解释变量 | 采用官方后续修订值；2019年前后存在口径变化 |
| `tertiary_industry_value_added` | 第三产业增加值 | 亿元 | 旅游相关产业总体发展指标 | 包含商业、金融、房地产等，不等于旅游业增加值 |

四项核心数据位于 `processed/core_annual_2010_2025.csv`。目前GDP和第三产业增加值覆盖2010—2025年，旅游数据仍有少量官方空缺。

## 二、模型标记与质量字段

| 字段名 | 含义 | 取值 |
|---|---|---|
| `covid_dummy` | 疫情冲击标记 | 2020—2022年为1，其余为0 |
| `gdp_definition_break_2019` | 2019年统计口径断点 | 2019年及以后为1，此前为0 |
| `core_missing_count` | 当年四项核心指标缺失个数 | 0—4，越大表示该年越不完整 |

以上三个字段是人为构造的建模或质检变量，不是政府公布的经济数据。

## 三、候选解释变量

| 字段名 | 中文名称 | 单位 | 建议用途 | 当前情况 |
|---|---|---:|---|---|
| `resident_disposable_income` | 居民人均可支配收入 | 元/人 | 消费能力 | 2016及2020—2024年 |
| `rural_disposable_income` | 农村居民人均可支配收入 | 元/人 | 乡村旅游发展、居民受益 | 2012—2016年 |
| `fixed_asset_investment` | 全社会固定资产投资 | 亿元 | 投资驱动 | 2012—2016年 |
| `fixed_asset_investment_non_agricultural` | 不含农户固定资产投资 | 亿元 | 投资驱动 | 2016—2017年；不能和上一指标直接拼接 |
| `tertiary_industry_investment` | 第三产业投资 | 亿元 | 产业投入 | 暂时只有2016年 |
| `road_mileage` | 公路通车里程 | 公里 | 交通可达性 | 2016及2021—2024年 |
| `road_passenger_volume` | 公路客运量 | 万人次 | 交通需求参考 | 暂时只有2016年；不等于游客量 |
| `boutique_homestays` | 精品民宿数量 | 户 | 问题3新业态情景 | 2023—2024年，不适合长期回归 |

这些字段已左连接到 `processed/modeling_features_annual_2010_2025.csv`。空白表示没有找到同口径官方值，不代表数值为0。

## 四、住宿餐饮和接待能力

数据来自官方统计年鉴中的“限额以上住宿和餐饮业经营情况”，覆盖2014—2024年。

| 字段名 | 中文名称 | 单位 |
|---|---|---:|
| `turnover_10k_cny` | 营业额 | 万元 |
| `room_revenue_10k_cny` | 客房收入 | 万元 |
| `meal_revenue_10k_cny` | 餐费收入 | 万元 |
| `goods_sales_10k_cny` | 商品销售额 | 万元 |
| `other_revenue_10k_cny` | 其他营业收入 | 万元 |
| `rooms` | 客房数 | 间 |
| `beds` | 床位数 | 张 |
| `dining_seats` | 餐位数 | 个 |

注意：这些数据仅覆盖“限额以上”住宿餐饮单位，并不代表蓟州区全部酒店、民宿和农家院。

## 五、数据状态代码

| 状态 | 含义 | 使用建议 |
|---|---|---|
| `official_actual` | 官方直接公布的实际值 | 可直接使用 |
| `official_preliminary` | 官方初步核算值 | 可以使用，但论文中注明“初步核算” |
| `official_revised` | 后续官方年鉴修订值 | 优先于早期初步值 |
| `official_final` | 官方最终核定值 | 优先使用 |
| `missing` | 尚未找到可信官方实际值 | 不得按0处理，需要补缺或模型估计 |

## 六、目标值和预计值

`raw/scenario_and_target_ledger.csv` 中的数值不能当作历史实际值。

| 类型 | 含义 | 用途 |
|---|---|---|
| `official_target` | 政府提出的未来目标 | 乐观情景或政策约束 |
| `official_estimate_lower_bound` | 官方预计值或下界 | 缺失值敏感性分析 |
| `official_alternative_report` | 同一年另一官方报道的不同精度口径 | 口径冲突检查 |

## 七、文件如何分工

| 文件 | 使用者 | 用途 |
|---|---|---|
| `processed/core_annual_2010_2025.csv` | 建模手 | 核心趋势、预测和异常分析 |
| `processed/modeling_features_annual_2010_2025.csv` | 建模手、编程手 | 核心指标与候选解释变量宽表 |
| `processed/data_dictionary.csv` | 全队 | 字段名中英文对照和机器读取 |
| `raw/official_indicator_ledger.csv` | 数据手、论文手 | 每条核心数据的来源、状态和备注 |
| `raw/auxiliary_indicator_ledger.csv` | 数据手、建模手 | 候选解释变量的来源底表 |
| `raw/accommodation_catering_2014_2024.csv` | 建模手 | 住宿餐饮及接待能力 |
| `raw/scenario_and_target_ledger.csv` | 问题3负责人 | 政策目标和情景参数 |

## 八、团队统一规则

1. 空值永远不等于0。
2. 游客量统一使用“万人次”，货币核心指标统一使用“亿元”。
3. 旅游综合收入与旅游直接收入不能混合成一条序列。
4. 初步值和修订值冲突时，优先采用后续官方年鉴修订值。
5. 目标值、预计值、节假日数据不能冒充全年实际值。
6. 论文引用数据时，应同时保存来源标题、链接和实际获取日期。
