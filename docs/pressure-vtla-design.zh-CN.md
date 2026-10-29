# π0.5 → VTLA：电阻式压力阵列的输入融合改造规格

核查日期：2026-10-29。官方代码基线：`Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479`。

**交付状态：已基于官方 openpi 实现 PyTorch 压力阵列接入，CPU 93 项测试通过；未加载完整 π0.5 权重或运行完整模型训练。** 源码与运行说明见 [中文使用说明](../README_VTLA.zh-CN.md)。用户已明确触觉为电阻式压力阵列，非视触觉，本机仅 CPU。阵列数量、分辨率、读数格式、标定、机器人和频率仍待绑定。下文的 token 数与训练阶段是建议值，不是任务效果结论。缓存一致性使用调用真实模型方法的小型替身主干测试，未验证完整官方主干的数值一致性。

## 1. 改造边界与选型

推荐主线：原始阵列读数经标定/固定尺度归一化，进入轻量空间/时间触觉编码器，再将定长触觉 token 加入 PaliGemma 观测前缀，利用原前缀的双向注意力完成视觉、语言、状态和触觉的融合。视觉继续走原 SigLIP；压力阵列不走图像编码器。动作后缀继续由原 ActionExpert 处理。

这里“保留 ActionExpert”指保留结构、动作表示、时间条件、输出投影和求解流程；不等同于冻结其权重。允许微调原有专家。

官方源码中，π0.5 通过 `Pi0Config(pi05=True)` 启用：状态离散化后进入语言 token；时间经 MLP 进入 ActionExpert 的 adaRMSNorm。不能照搬 π0 的连续 state suffix token。[配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)、[tokenizer](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py)

两条路线应明确区分：

| 路线 | 用途 | 成本与边界 |
| --- | --- | --- |
| 阵列展平 → MLP → 每传感器少量 token | 最小工程基线 | 容易实现，但对固定 taxel 排列绑定较强，需实测是否丢失局部接触信息 |
| 规则阵列 → 局部 patch MLP → 定长空间 token → 原观测前缀 | 本版实现主线 | 保留局部空间结构，控制 token 预算；小 CNN 是可后续比较的替代方案 |
| 阵列 → 伪彩图 → 原视觉编码器 | 可选对照，不作为主线 | 引入颜色映射和图像尺度处理，不能假定视觉预训练自然适合电阻读数 |

不需要新增第三个语言专家或改动作生成头。两路前缀嵌入必须先对齐到 PaliGemma 的宽度；默认 `gemma_2b` 为 2048，而 `gemma_300m` ActionExpert 为 1024，实际数值从配置读取。[专家配置与注意力实现](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py)

## 2. 数据接口：先定义可部署的观测

建议新增独立触觉字段，保留原相机输入：

| 字段 | 批量形状 | 含义 |
| --- | --- | --- |
| `images` / `image_masks` | 沿用上游契约 | 外部/腕部视觉及有效性 |
| `state` | `[B, D_state]` | 机器人状态；沿原 π0.5 文本化链路 |
| `tactile_values` | `[B, S, K, R, C]` | 固定传感器顺序、从旧到新的阵列读数，明确 raw/标定后的含义 |
| `tactile_frame_valid` | `[B, S, K]`，bool | 帧是否存在、读取成功、时间上可用 |
| `tactile_taxel_valid` | `[B, S, K, R, C]`，bool | 每个感测单元是否正常，及阵列布局 padding |
| `tactile_saturated` | `[B, S, K, R, C]`，bool | 达到量程/ADC 上下限等测量状态，具体规则来自设备 |
| `tactile_age` | `[B, S, K]` | 观测时刻减采样时刻，单位秒 |
| `actions` | `[B, H, D_action]` | 仅训练标签，沿原动作契约 |

这是新增接口草案，不是上游现成字段。`R,C` 是阵列行列数，不是图像高宽的语义；若每 taxel 有多种实测量，可额外增加信号维度 `F`，得到 `[B,S,K,R,C,F]`。进入 PyTorch 小 CNN 时明确转为 `[B*S*K,F_in,R,C]`。不要将 ADC 整数交给原图像 uint8→[-1,1] 的转换分支。

数据读取应完成：

1. 在同一 episode 内以策略观测时刻对齐视觉、状态、触觉与动作标签。触觉只取当时已经到达且采样时间不晚于观测时刻的帧，不能用双向最近邻误取未来帧。
2. 首版取每片传感器最新一帧，即 `K=1`，`S` 按实际硬件确定。需要接触变化时再加入固定窗口历史；历史不能跨 episode，缺失历史保留占位并设 `frame_valid=False`。
3. 区分“无接触”和“缺失”：正常的零压力/基线附近读数是有效输入。传感器断流、帧损坏和超时才按缺失策略屏蔽。重复使用旧帧时记录真实 age，不伪装成新帧。
4. 保留原始 ADC/电压/电阻及标定信息。未完成读数到压力的标定时，字段称为“原始/归一化电阻式触觉信号”，不声称单位为 kPa 或 N。读数随压力增减的方向由设备或标定确定，不在通用读取器中臆定。
5. 明确零载基线、每 taxel 偏置、尺度、坏点和饱和判据。零点更新只能使用部署时已知无接触条件，不能用未来数据或整段 episode 的最小值来校正当前样本。漂移、迟滞、串扰是否显著需要实测，不能假定一个静态标定已经消除。
6. 归一化采用训练集或预先声明的标定集固定统计，排除无效读数，明确饱和读数的统计策略；不按每帧 min-max 或总压力归一化，以免消掉需要保留的幅值。饱和与缺失分开，坏点占位同时送入有效性信息。
7. 不复用相机的 resize/crop/rotate/ColorJitter；这些处理会改变 taxel 的位置或数值。不同阵列尺寸通过约定的布局 padding、mask、独立输入映射或物理坐标处理，不能直接当图片拉伸。
8. 数据集按 episode 或对象/场景划分训练与测试。记录传感器安装方向、taxel 物理间距、编号、标定、时间戳时钟域和重置边界。没有压力单位和 taxel 面积标定时，不从阵列求和宣称得到真实接触力。

机器人型号、动作是绝对量还是增量、坐标系、控制频率、动作 chunk 长度和执行步数必须沿真实平台配置，不根据模型默认值臆定。输入触觉不会自动把 chunk 策略变成高频触觉反馈控制器。

## 3. 触觉编码、token 拼接与身份

对规则阵列，建议将归一化读数、taxel 有效性和饱和指示组成输入通道，每帧经小 CNN 或局部 patch+MLP 编码为网格特征，再池化/组织为 `Nt` 个 token，投影到前缀宽度：

`Z_touch = Project(SpatialEncoder(values, taxel_valid, saturated)) + E_sensor + E_space + E_age`

`E_sensor` 区分传感器/安装位置，`E_space` 标记 taxel 或局部 patch 的位置，`E_age` 表示历史或测量年龄。规则阵列按固定网格展平；若硬件是非规则点阵，使用真实坐标的 taxel MLP/集合编码器，不强套相邻行列假设。空间位置与传感器安装方向共同定义。

先将 `B,S,K` 合并编码，再恢复 `[B,S,K,Nt,D_prefix]`，最终展平为 `[B,S*K*Nt,D_prefix]`。示例：若实际是两片阵列、每帧 16 个空间 token、`K=1`，则新增 32 个 token；这不是已确认的硬件配置。小阵列可逐 taxel 编码，大阵列可局部 patch 编码，token 预算需结合分辨率选择，不应一开始就强制池化成单个全局 token。

最小版使用随机初始化的小型压力编码器，不从 SigLIP 复制参数。`K=1` 只承载当前触觉，不能据此宣称获取了动态滑动信息；增加历史后先保留各帧 token 与 age，必要时再比较仅使用过去窗口的时间编码/压缩模块。

空间 token 的有效性由 `frame_valid` 与该 token 覆盖的 taxel 有效性共同决定：全坏 patch 屏蔽，部分坏 patch 保留 mask 信息；正常无接触 patch 仍有效。小 CNN 的坏点占位不能单靠“填零”解释，至少加入有效性通道；若使用 mask-aware 归约，避免坏点数量改变归约尺度。饱和通常是有信息的测量状态，保持独立标志，是否屏蔽由设备契约决定。

无效/非有限读数先用条件选择替换成固定占位，再进入卷积和池化，不能只在最终 token 上乘零：坏点可能通过局部卷积影响邻近有效 token，`NaN * 0` 也不会变成有效数值。训练时若使用跨样本统计的归一化层，还要验证缺失帧没有污染其他样本；最小编码器可避免这类跨样本统计依赖。

固定前缀顺序：

`prefix = concat(原视觉 tokens, 原语言+离散状态 tokens, 触觉 tokens)`

将触觉附在原前缀后，可以保留原视觉和文本 token 的位置编号；相较插入文本前，更少改变原有编号。文本 padding 仍按其有效性 mask 处理。新增触觉会改变有效前缀长度和注意力结果，不应宣称天然保持原模型输出。

所有触觉 token 进入同一个 PaliGemma 前缀流。触觉原始维度先投影到 `D_prefix`，不要直接拼到 1024 维动作专家流。这里的融合由原前缀注意力完成，不要求新增跨模态 Transformer。

## 4. Attention Mask：保留块语义

必须区分三个对象：token 是否有效的 `input_mask`、标记新注意力块的 `ar_mask`、最终注意力矩阵。

固定 π0.5 语义：

- 前缀视觉/语言/状态/触觉：`ar_mask=False`。
- 动作后缀：第一项 `True`，其余 `False`。
- 有效前缀彼此双向可见；前缀不能读动作后缀；动作可读全部有效前缀，动作 chunk 内彼此可见。
- 不要把动作后缀设成逐 token 因果 mask；Flow Matching 联合预测整个动作 chunk。

以查询行为行、键为列，忽略 padding 的块矩阵是：

| Query / Key | Prefix | Action |
| --- | --- | --- |
| Prefix | 允许 | 禁止 |
| Action | 允许 | 允许 |

源码 `make_attn_mask` 使用累计块编号实现这件事：[pi0.py:19](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L19)。可沿用公式：

```python
# 概念性 JAX 伪代码；不是已集成补丁。
prefix = concat([vision, language_state, tactile], axis=1)
prefix_valid = concat([vision_valid, language_valid, tactile_token_valid], axis=1)
prefix_ar = zeros(prefix.shape[1], dtype=bool)
action_ar = array([True] + [False] * (action_horizon - 1))

valid = concat([prefix_valid, action_valid], axis=1)
block_start = concat([prefix_ar, action_ar], axis=0)
mask = make_attn_mask(valid, block_start)
positions = cumsum(valid.astype(int32), axis=-1) - 1
```

根据 `frame_valid` 和局部 taxel 覆盖生成 `[B,S,K,Nt]` 的 token 有效性，再展平为 `[B,N_touch]`。缺失触觉必须 `valid=False`；把读数置零或把 token 乘零都不能代替屏蔽，因为偏置、位置和身份嵌入仍可能产生非零 token，且未屏蔽的键会影响 softmax。有效 taxel 的真零值不能被屏蔽。

底层 JAX 使用布尔允许矩阵；PyTorch 路径还会把它转换为四维 additive mask。沿各自接口转换，不能把 bool/0/1 直接当作同一种 mask。

## 5. 位置编码与训练/推理一致性

区分编码器内部空间位置、触觉传感器/采样年龄、Transformer 序列 RoPE、Flow Matching 的时间变量。它们承担不同作用。

- 保留视觉编码器内部位置编码。
- 压力阵列使用 taxel/patch 空间编码及 sensor/age 标签。
- Transformer 序列 positions 沿上游按有效 token 累加；不为每种模态重新从 0 开始，也不改 RoPE 实现。
- Flow Matching 的时间仍沿 π0.5 原 adaRMSNorm 路径注入，不能把触觉 age 当作生成时间。

训练时：`positions = cumsum(full_valid) - 1`。

缓存推理时：前缀 positions 为 `cumsum(prefix_valid)-1`；动作 positions 为：

`sum(prefix_valid, axis=-1)[:,None] + cumsum(suffix_valid, axis=-1) - 1`。

缓存的**物理序列长度**包含占位 token，而动作的**位置编号偏移**沿上游使用有效前缀长度。两者不能混用，尤其是批内缺失触觉数量不同时。

每次新观测重新编码包含触觉的完整前缀，建立 KV cache；同一个 action chunk 的各次去噪复用该 cache。观测或触觉窗口改变就重新建 cache。若要去噪中途更新触觉，需要重新设计一致性，超出首版范围。[训练和缓存推理](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L189)

## 6. 保持 Flow Matching 目标与动作结构

按此次核对的 openpi 代码约定：

- `t=1` 为噪声，`t=0` 为目标动作。
- `x_t = t * epsilon + (1-t) * actions`。
- 目标向量场 `u_t = epsilon - actions`。
- 预测 `v_theta(x_t,t | prefix)`，损失沿上游为 `(v_theta-u_t)^2` 的对应维度平均。
- 推理沿上游从 1 向 0 积分，保留原求解步数配置与动作输出投影。

只改变条件前缀。继续使用原噪声分布、时间采样、ActionExpert、adaRMSNorm 和反归一化；不要把其他论文或实现的相反时间约定拼进来。[compute_loss / sample_actions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L189)

如果数据存在 episode 尾部不完整动作 chunk，沿数据集明确的 padding/过滤契约处理；若引入动作标签有效性 mask，应在损失中同步使用，而非只修改注意力。不同机器人动作维度的填充也需要与既有训练策略一致。

## 7. 代码修改地图

本版已落地 dataclass 观测字段 `pressure`（含 values/frame_valid/taxel_valid/saturated/age），配置入口 `PressureConfig`，以及 manifest/NPZ 数据读取和固定压力统计。详细字段名以仓库运行说明和实现为准；前文 `tactile_*` 是概念拆分。下表的验收条件中，真实样本和完整模型运行仍待数据与环境绑定。

| 位置 | 必须完成的修改 | 验收条件 |
| --- | --- | --- |
| 数据转换脚本、平台 policy input transform | 读触觉、时间同步、传感器固定顺序、显式缺失标记；保留新增键 | 一条真实 episode 能从磁盘读到最终 batch |
| `training/data_loader.py` 及数据配置 | 加载字段、保留历史、collate、padding、worker 解码 | 批内缺失和短 episode 不改变静态结构 |
| `transforms.py` / 归一化统计 | 独立触觉预处理；保留原状态文本化；仅训练集统计 | 同一输入的训练/推理基础预处理一致 |
| `models/model.py` | `Observation`、`from_dict`、`to_dict`、`preprocess_observation` 全链路保留触觉字段 | 往返转换和预处理后触觉仍存在 |
| `models/pi0_config.py` | 新增触觉配置、`inputs_spec`/fake observation；扩展工厂 | shape 初始化包含所有新模块和字段 |
| 新阵列标定/归一化与编码/投影模块 | 固定 token 预算、taxel→token mask、sensor/space/age 编码 | 输出 `[B,N_touch,D_prefix]`，dtype 对齐，保留触觉幅值 |
| `models/pi0.py` 或对应 `models_pytorch/pi0_pytorch.py` | `embed_prefix` 接触觉；训练、采样共用；后缀结构沿用 | 梯度和输出均对有效触觉敏感 |
| `models_pytorch/preprocessing_pytorch.py` 与 `_preprocess_observation` | PyTorch 独立预处理保留触觉；扩展返回值及 forward/sample_actions 接口 | 触觉没有在三相机筛选或 tuple 解包中丢失 |
| `training/config.py`、训练脚本、权重加载 | 新参数初始化、冻结/优化器参数组、保存与恢复 | 只允许预期的新键缺失，原模型键全部校验 |
| policy/server/client 输入输出 | 传输触觉及有效性/age，使用相同配置与预处理 | 在线输入和训练样本具有同一 observation 契约 |

上游有一个已确认的漏点：`preprocess_observation()` 默认仅遍历 `IMAGE_KEYS` 中的三路相机，再重建 `Observation`；`from_dict()` 也只抽取已知字段。单纯新增字典键不会自动穿透这些步骤。[model.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/model.py)

压力阵列采用独立触觉字段时，要修改每个重建 `Observation` 的位置，经过专用数值预处理后独立编码。不要把阵列放入 `images` 并期待原图像转换器正确处理；即使做伪彩图对照，也要检查 image whitelist、输入 spec、mask 和增强。

历史读取另有已核对的边界：`data_loader.py` 的 `delta_timestamps` 当前为 `action_sequence_keys` 构造未来动作 chunk。触觉历史要走单独的非正时间索引，不能把触觉键加入未来动作列表。`compute_norm_stats.py` 当前只统计 `state/actions`，不会自动统计新的阵列字段。新增独立触觉统计 asset/过程，保持原状态和动作统计、`TokenizePrompt` 所读的原 `state` 不变；不要把大阵列塞进状态文本来绕过接口修改。[数据加载](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py#L140)、[归一化脚本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/compute_norm_stats.py#L102)、[PyTorch 预处理](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/preprocessing_pytorch.py)

## 8. 训练与权重适配

建议先只实现一种框架，避免同时维护两套未经验证的分支。目标是 JAX 还是 PyTorch 应由现有工程决定。

阶段 A：在 π0.5 权重基础上随机初始化新的小型压力编码器、投影与 sensor/space/age 参数，并训练这些新参数及原 ActionExpert；可冻结原视觉语言主干参数。不要冻结随机初始化的压力编码器。明确保存可训练参数清单和梯度统计。

阶段 B：在可测得触觉作用后，再比较更长历史、更细空间 token 或主干 LoRA。学习率、batch、步数和解冻范围由试跑确定，不把默认比例写成已验证配方。

冻结参数不等于关闭输入梯度：若触觉适配器的输出经过冻结的前缀 Transformer 才生成动作条件，整段 `no_grad()` / `stop_gradient()` 会切断适配器学习路径。冻结权重即可；是否可对单独的冻结图像编码器关闭梯度，要看其前面是否还有需要训练的模块。

checkpoint 分为两种：

1. **从官方权重初始化 VTLA**：核对原参数名称、形状、加载完整性；仅允许明确列出的新增触觉键初始化，不能整体忽略不匹配。
2. **恢复已训练 VTLA**：使用完整 VTLA 模型配置、所有新旧权重、优化器状态与归一化资源严格恢复，不能每次恢复时把触觉模块重新随机化。

配置中至少记录传感器顺序、窗口 K、每帧 token 数、预处理版本、时间对齐规则、动作契约和缺失策略。可加入触觉 modality dropout，但同一实验的视觉基线、训练数据、步数与模型选择规则需匹配。

已确认的框架差异不能省略：

- JAX `CheckpointWeightLoader` 默认允许缺失的正则为 `.*lora.*`。新增触觉模块需要扩展受控 allowlist，并保留这些新参数的初始化值。[weight_loaders.py:48](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/weight_loaders.py#L48)
- PyTorch 基座加载使用 `safetensors.torch.load_model`，没有现成的触觉缺失键处理。迁移时显式读取 missing/unexpected keys 并核对新模块白名单；VTLA resume 使用完整模型恢复。[train_pytorch.py:441](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train_pytorch.py#L441)
- JAX 训练按 `config.trainable_filter` 选择梯度参数；核对的 PyTorch 训练脚本直接将 `model.parameters()` 交给优化器，没有实现相同的 `freeze_filter` 语义。PyTorch 要显式设置 `requires_grad`，只把计划训练的参数放入参数组，并核查实际梯度。[JAX 训练](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train.py#L157)、[PyTorch 优化器](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train_pytorch.py#L457)

## 9. 分层验收与最小对照

先通过数据和计算正确性，再判断策略效果。

| 层次 | 必查行为 | 通过证据 |
| --- | --- | --- |
| 数据 | 无未来触觉、无跨 episode 历史、无接触与缺失区分 | 真实样本时间轴审计与可视化/数值记录 |
| 阵列 | 真零值、部分坏 taxel、全帧缺失、饱和、历史缺帧分别处理 | 五类样本的预处理/token/mask 检查 |
| mask/位置 | Prefix→Action 禁止；Action→Touch 允许；缺失 touch 不可见 | 小尺寸精确矩阵测试；批内不同有效长度测试 |
| 缺失不变性 | 修改被 mask 的触觉占位读数不应改变有效动作输出 | 相同噪声、同一模型、同一有效观测的数值比较 |
| 梯度 | 触觉投影得到有限梯度且参数更新 | 单 batch 前后参数/梯度检查 |
| 缓存 | 一次性全序列与缓存路径对相同 noisy action/time 的向量场一致 | 同权重、同输入、同噪声/时间、合理数值容差比较 |
| 恢复 | 保存/重载后结果一致；触觉权重没有重新初始化 | 固定观测与噪声的 round-trip 检查 |
| 端到端 | 从真实磁盘样本到 loss/backward/optimizer，再到动作采样 | 至少一个小规模真实数据 smoke run |

mask-only 测试或假数据 loss 下降不能替代完整训练验证。全部触觉屏蔽的模型与原 π0.5 等价，也只应在原权重、原有效位置、原预处理一致等前提下检验；训练后模型权重已经改变，不能要求天然还原原策略。

最低对照组：

- 视觉语言状态基线：相同平台、数据、训练预算下的 π0.5 微调。
- 阵列展平 MLP 前缀基线：检验空间编码相对于简单数值输入是否必要。
- 推荐方案：局部空间编码的阵列触觉前缀；与 MLP 对齐 token 数和尽量匹配参数量。
- 真实触觉、跨样本打乱触觉、时间错位触觉、屏蔽触觉；配对使用任务、初始条件和随机种子。仅去除触觉造成分布变化，不能单独作为充分的机制证据。

真实任务按平台选择，例如接触丰富的抓取/防滑、插接或受限力操作。固定任务成功定义，并报告失败类型、滑落、标定后可获取的接触指标、端到端延迟、内存和触觉缺失鲁棒性。未完成力标定时不报告“峰值力 N”，可如实报告传感器单位的峰值响应或另用标定测力设备。离线 action MSE 或 Flow Matching loss 不能直接等同于真实任务成功。

## 10. 实施状态与待绑定项

1. 已固定官方 commit、PyTorch 与 CPU 组件验证路径；尚未下载完整权重。
2. 已完成 NPZ/manifest 读取与因果对齐，用落盘的合成 episode 验证；真实设备数据未定。
3. 已实现 patch MLP encoder、prefix 拼接、mask 与位置编号接入。
4. 已实现受控基座加载、训练冻结/优化器筛选与 checkpoint/统计资产恢复，用小模型验证梯度和恢复契约。
5. 已实现压力进入 cached inference；小型替身主干下调用真实 PI0Pytorch 方法的 joint/cached 一致性通过。
6. 待硬件与数据绑定后运行真实样本、完整官方权重及任务评估。

CPU 环境：Python 3.11.1、torch 2.7.1+cpu、numpy 1.26.4，`cuda_available=False`；`pytest tests/vtla -q` 为 93 passed in 9.52s，21 个修改/新增 Python 文件 Ruff、格式与语法编译检查通过。AST 对比确认原动作 suffix、denoise_step、时间/噪声采样及 Flow Matching 目标表达式保持不变。未安装完整官方训练依赖；JAX 压力模型明确未实现。

尚缺：阵列数量与行列数、安装布局、raw ADC/电压/电阻或标定压力的样例字段；标定/坏点/量程契约；机器人与动作定义；采样/控制频率和可接受延迟；真实数据规模与完整主干运行资源。它们影响具体配置，不影响本文“数值触觉条件接入前缀、保留动作后缀”的总体设计。

当前没有训练收益、实时性或科研创新性的实验证据；本文没有进行触觉 VLA 领域的完整新颖性检索。
