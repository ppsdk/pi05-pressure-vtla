# 检索记录

日期：2026-10-29（Asia/Shanghai）

## 使用的公开查询

- `"vision-tactile-language-action" robot`
- arXiv API：`ti:"VTLA"`
- arXiv API：`ti:"Tactile-VLA"`
- arXiv API：`ti:"OmniVTLA"`
- arXiv API：`ti:"TacVLA"`
- arXiv API：`ti:"ForceVLA"`
- arXiv API：`all:"force" AND all:"vision-language-action"`
- arXiv API：`all:"pressure array" AND cat:cs.RO`
- arXiv API：`all:"tactile dataset" AND cat:cs.RO`

## 核验来源

- arXiv 原始摘要页/API：论文标题、作者、日期、摘要和稳定链接。
- 论文项目页：仅在 arXiv 摘要提供时作为补充入口。
- Google 搜索结果：只用于发现候选，不作为最终证据。

## 筛选说明

- 保留 2024–2026 年与 VTLA、触觉增强 VLA、力/压力阵列表征和 π₀/OpenVLA 基线直接相关的条目。
- 2025–2026 条目没有在本次工作中逐一核对正式会议录用状态，因此统一标为 arXiv 预印本。
- 没有把搜索摘要中未能回溯到原始论文页的条目放入重点表。
- 未纳入 MDPI 或无法确认来源质量的条目。

## 未解决的问题

- 压力阵列的行列数、采样频率、ADC/电阻单位、标定曲线和安装位置尚未确定。
- 以上论文多数使用视觉触觉图像、6 轴力或特定触觉硬件；电阻式压力阵列的跨设备迁移证据仍然有限。
- 本次没有下载和逐页阅读全部 PDF，也没有复现论文中的成功率。

## 给当前仓库的交接

- 先用真实压力 NPZ 完成端到端基线，再决定是否实施 Tactus 风格掩码预训练或 OmniVTLA 风格语义对齐。
- 记录 `frame_valid`、`taxel_valid`、`saturated`、`age`，并把校准/动态范围作为独立实验因素。
- 用 VT-Bridge 思路设置冻结主干对照，用 TacVLA 思路设置接触门控对照，用 Representation-Aligned Tactile Grounding 思路设置 ActionExpert 中间层辅助损失对照。
