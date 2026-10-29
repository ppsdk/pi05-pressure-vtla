# VTLA 与压力/力触觉 VLA 文献索引

日期：2026-10-29  
检索目的：为 `pi05-pressure-vtla` 的电阻式压力阵列接入、触觉编码器训练和实验设计提供可追踪的相关工作。  
口径：VTLA 指 Vision-Tactile-Language-Action；同时纳入直接相关的触觉增强 VLA、力反馈 VLA、压力阵列表征和 π₀/π₀.₅ 基线。  
来源政策：优先 arXiv 原始论文页、项目页和官方仓库；未核对正式录用状态的条目标为“arXiv 预印本”。本索引不把搜索摘要当作唯一证据，也不包含 MDPI 来源。

## 结论先行

- **最直接的 VTLA 先例**是 Zhang 等的 VTLA（2025）：视觉、触觉、语言和动作通过跨模态语言 grounding 融合，并使用 DPO 处理连续动作监督；任务是插入操作。
- **与你的电阻式压力阵列最贴近**的是 OmniVTLA（2025）和 Tactus（2026）。OmniVTLA 明确覆盖 force-based tactile，并提出语义对齐触觉 ViT；Tactus 专门处理低成本电阻压力阵列，强调传感器校准和掩码自编码预训练。
- **对当前 π₀.₅ 改造最有借鉴价值**的是 VT-Bridge（2026，预印本）：在 π₀、π₀.₅ 和 SmolVLA 上使用轻量残差适配器，主张用少量视觉-触觉示范降低改造成本；这支持先冻结主干、训练压力分支与动作专家的路线。
- **触觉是否要专门预训练**尚无单一答案：Tactile-VLA/VT-Bridge 支持少量示范或轻量适配；OmniVTLA、Tactus、N₀-VTLA 说明在跨任务/跨传感器泛化时，触觉预训练或对齐初始化可能有价值。当前仓库应先完成端到端监督训练，再以真实数据决定是否增加压力阵列自监督预训练。
- 当前仓库使用的是数值压力 token，不是图像触觉；因此视觉触觉论文只能作为融合/训练范式参考，不能直接当作传感器等价证据。

## 重点论文

| # | 论文 | 年份/状态 | 类型 | 与本项目的直接关系 |
|---|---|---|---|---|
| 1 | [VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation](https://arxiv.org/abs/2505.09577) | 2025，arXiv 预印本 | method + benchmark | 直接定义 VTLA；视觉/触觉/语言 grounding，DPO 连接离散策略学习与连续动作，插入任务。 |
| 2 | [Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization](https://arxiv.org/abs/2507.09160) | 2025，arXiv 预印本 | pure method | 强调利用已有 VLM 物理知识，通过少量示范接入触觉；包含混合位置-力控制和触觉反馈推理。 |
| 3 | [OmniVTLA: Vision-Tactile-Language-Action Models with Semantic-Aligned Tactile Sensing](https://arxiv.org/abs/2508.08706) | 2025，arXiv 预印本 | method + benchmark | 双路径触觉编码器（ViT + SA-ViT），覆盖视觉式和力式触觉；ObjTac 含文本、视觉、触觉三模态。对压力阵列 token 的语义对齐最有启发。 |
| 4 | [ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](https://arxiv.org/abs/2505.22159) | 2025，arXiv 预印本 | method + benchmark | 将 6 轴力反馈作为一等模态，用 force-aware MoE 做动态路由；ForceVLA-Data 提供视觉、本体状态和力矩同步数据。可作压力分支/门控的对照。 |
| 5 | [Tactile-VLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation](https://arxiv.org/abs/2603.12665) | 2026，arXiv 预印本 | pure method | 用接触感知 gate 只在接触时激活触觉 token；适合设计压力阵列的 `frame_valid`/接触门控消融。 |
| 6 | [$N_0$-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens](https://arxiv.org/abs/2607.23782) | 2026，arXiv 预印本 | method + benchmark | 视觉-触觉预训练、分阶段触觉接入和离线策略改进；代表大规模触觉预训练路线，计算和数据成本高。 |
| 7 | [VT-Bridge: Bridging Pretrained Foundation VLAs to VTLAs via Lightweight Residual Adaptation](https://arxiv.org/abs/2609.22606) | 2026，arXiv 预印本 | pure method | 在 π₀、π₀.₅、SmolVLA 上用轻量残差适配器接入触觉；直接支持“冻结主干 + 训练新增分支”的工程策略。 |
| 8 | [τ: Learning Touch-Augmented Vision-Language-Action Models from Future Visual Supervision](https://arxiv.org/abs/2607.24485) | 2026，arXiv 预印本 | method + benchmark | 用未来视觉监督学习动作条件的时空触觉表示；TacAura 提供同步视觉/本体/视觉触觉数据。可参考未来触觉预测辅助损失。 |
| 9 | [Representation-Aligned Tactile Grounding for Contact-Rich Robotic Manipulation](https://arxiv.org/abs/2607.14609) | 2026，arXiv 预印本 | pure method | 发现未来触觉更适合从中间 ActionExpert 表征预测，而非最终动作；可指导在 ActionExpert 中间层增加压力辅助头。 |
| 10 | [ReTouch: Empowering Contact-Rich Dexterous Manipulation with Online-Refined Tactile Prediction](https://arxiv.org/abs/2608.01824) | 2026，arXiv 预印本 | method + benchmark | 结构化 tactile patch encoder + 在线触觉预测/动作块修正；与本仓库的 patch MLP、历史帧和未来扩展方向相近。 |
| 11 | [Tactus: Open-Vocabulary Object Recognition from Low-Cost Pressure Arrays](https://arxiv.org/abs/2608.04043) | 2026，arXiv 预印本 | method + benchmark | 专门针对电阻式压力阵列；报告传感器自身 affine 校准、掩码自编码预训练和动态范围错误的影响。是当前项目最重要的压力阵列表征参考。 |
| 12 | [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) | 2024，arXiv 版本页 | pure method | 原始 Flow Matching VLA 基线；本项目保留其 ActionExpert、flow matching 目标和 Euler 采样结构。 |
| 13 | [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) | 2024，arXiv/开源项目 | method + benchmark | 开源 VLA 与参数高效微调基线；可用于比较触觉接入前后的冻结/LoRA/全量微调策略。 |
| 14 | [The Role of Tactile Sensing for Learning Reach and Grasp](https://arxiv.org/abs/2502.20367) | 2025，arXiv 预印本 | pure method | 比较不同复杂度的力式触觉输入对抓取学习的影响；提醒压力输入维度增加可能使训练更难，需要低维/patch/门控消融。 |

## 研究脉络与空缺

### 1. 直接 VTLA 与接触丰富操作

VTLA、Tactile-VLA、OmniVTLA 已覆盖“视觉 + 触觉 + 语言 + 动作”的基本范式，主要任务集中在插入、抓取和接触丰富操作。直接把“新增压力 token 接到 π₀.₅ 前缀，并保留原 ActionExpert/Flow Matching”作为唯一贡献，创新空间较弱；需要用压力阵列的低成本、异步、缺失/饱和建模和 CPU 可复现训练作为明确边界。

### 2. 触觉编码器与训练

文献路线分为三类：

1. **端到端轻量适配**：Tactile-VLA、VT-Bridge，适合少量真实示范和冻结大主干。
2. **语义/跨传感器预训练**：OmniVTLA、N₀-VTLA，适合希望跨传感器/跨任务迁移的设置，但需要更大数据和算力。
3. **触觉自监督或预测辅助**：Tactus 的 masked-autoencoder、τ 的未来视觉监督、ReTouch 的在线触觉预测，适合在动作监督之外稳定触觉表征。

对本项目的可证伪问题是：在固定 π₀.₅ 主干和同一压力数据下，`随机初始化端到端`、`压力自监督预训练后微调`、`冻结压力编码器` 三者在接触成功率、动作误差、压力扰动鲁棒性和数据效率上是否有差异。

### 3. 压力阵列特有问题

Tactus 直接指出传感器自身校准和动态范围处理可能比更换网络结构更重要。对当前电阻式阵列，必须单独报告 ADC/电阻到物理量的标定、零点漂移、饱和判据、坏 taxel、采样/到达时间和传感器编号；不能用视觉触觉论文中的图像归一化替代这些步骤。

## 面向 `pi05-pressure-vtla` 的建议基线

1. **主基线**：官方 π₀.₅ + 当前压力 patch MLP，冻结 PaliGemma，训练压力编码器和 ActionExpert。
2. **低成本对照**：只拼接单一 pooled pressure token；与当前局部 patch token 比较。
3. **门控对照**：无接触时屏蔽压力 token、接触时启用；对比始终启用。
4. **训练策略对照**：端到端随机初始化 vs. 压力 MAE/掩码重建预训练后微调。
5. **表征位置对照**：仅前缀融合 vs. 在 ActionExpert 中间层增加未来压力预测辅助头。
6. **压力特有鲁棒性**：随机缺帧、taxel 丢失、饱和、时间延迟、跨传感器归一化和动态范围缩放。

## 证据边界

本索引核对了 arXiv 原始摘要页/API 和项目页链接；除 π₀/OpenVLA 的开源项目信息外，未在本次检索中逐篇核对正式会议录用、完整实验表格、代码可用性或真实硬件复现。所有 2025–2026 条目均按当前可访问版本标为 arXiv 预印本；其摘要中的成功率仅作为论文自报结果，不是本项目结果。
