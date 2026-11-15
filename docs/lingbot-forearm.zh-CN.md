# LingBot-Touch 小臂适配

`pi05_pressure_forearm` 使用动作去噪速度残差框架，可选择官方 LingBot-Touch
压阻 tokenizer 和 Transformer 权重作为触觉编码器。当前读取条件是机器人状态，
尚未加入视觉语言上下文查询。默认配置仍使用原 MLP；压力 JSON 中增加
`lingbot_touch` 才启用新编码器。

## 计算与训练

1. 沿用 `values/frame_valid/taxel_valid/saturated/age` 五字段和已有数据变换。
   编码器用固定训练统计还原原始信号，再按 `(raw - sensor_zero) / sensor_span`
   截断到 `[0,1]`，放入官方压阻 tokenizer 的 Z 通道，XY 为零。
   `sensor_span` 是有符号的满量程接触信号变化，支持负载越大电阻越小的传感器。
   原始电阻、ADC 和归一化读数的单位由设备标定决定。
2. 同一测点连续 `tubelet_size` 帧构成一个 token。Tiny 为 2，Small 为 4；
   `history` 必须能被 tubelet 大小整除。tubelet 内任一读数缺失、坏点、未来或陈旧，整段 token
   无效；不复制帧补齐。有效零值仍表示真实无接触。饱和有独立适配特征。
3. 用统一小臂坐标系内的 XYZ / `coordinate_scale_mm` 替代手指规范几何。
   原手指 RoPE 轴置零，新增皮肤片 embedding。时间 RoPE 使用真实平均 age /
   (`time_scale_seconds` × tubelet)。时间尺度是固定参考
   采样间隔；不重采样，也不替代实际时间戳。可选差分是 tubelet 内末帧减首帧；
   仅全部有效且未饱和时使用。旧 MLP 则输出逐帧相邻差分。
4. 每层注意力都应用逐样本动态有效性 mask，阻止无效数据污染其他接触位置。
   历史窗口内双向注意力合法，所有观测都须在当前决策时刻可得。
5. 用机器人状态生成 query，通过注意力读取时空 tokens，再接回原残差头。
   原零初始化、coverage 缩放、残差限幅与 Flow Matching 目标不变。
   残差在每个动作块开始计算一次，在块内各 Euler 去噪步复用；没有新增高频控制器。

`freeze_pretrained=true` 冻结所有预训练参数，训练皮肤身份/质量适配、输出映射、
状态条件读取及原残差头。冻结不使用 `no_grad()`，保留适配层的反向梯度。
设为 `false` 时，压阻 tokenizer、Transformer blocks、所选输出层归一化及可选
shared projector 可训练；其余传感模态和未使用层继续冻结。训练和恢复须使用
相同冻结配置。原始 JEPA 预训练过程没有在本项目实现。

## 使用

官方权重与匹配的 `config.json`、`encoder.pt` 放在
`checkpoints/lingbot-touch-tiny/`。本地已下载 Tiny，Git 忽略该目录，克隆仓库后需单独下载权重。
来源：[官方魔搭仓库](https://modelscope.cn/models/Robbyant/lingbot-touch-tiny)，
也可使用[官方 Hugging Face 仓库](https://huggingface.co/robbyant/lingbot-touch-tiny)。
源码固定在 `dec8025b212e7099b235cc2ef13a4921a7e59545`；七个原始文件及许可证
保存在 `src/openpi/models_pytorch/lingbot_touch/`，新增适配代码独立维护。
适配代码使用本项目的 torch/numpy 版本，无需安装 LingBot-Touch 的完整运行环境。

`examples/pressure/forearm_lingbot.example.json` 只供合成信号演示。
正式使用前，按实物修改片数、行列、坐标、单位、零载、满量程方向、时间尺度和过期阈值。
几何、标定、采样周期目前尚未由真实设备确认。不能根据张量形状自动推断这些参数。
raw 数据中的漂移/滞回修正如有需要，应在采集标定阶段完成。

完整训练环境中的调用方式（manifest、机器人维数与状态/动作资产仍按原项目配置）：

```bash
python scripts/train_pytorch.py pi05_pressure_forearm \
  --exp-name forearm_lingbot \
  --pressure-config-path /path/to/real_forearm.json \
  --lingbot-touch-bundle-path checkpoints/lingbot-touch-tiny \
  --pytorch-weight-path /path/to/pi05_pytorch_base \
  --data.manifest-path /path/to/train_manifest.json \
  --data.pressure-stats-path /path/to/pressure_stats.npz \
  --data.robot-action-dim REAL_ACTION_DIM
```

`pressure_stats.npz` 是现有流程用训练集计算的固定标准化资产；编码器会还原这一步，
不要再次把标准化结果当作原始设备数据。该路径不支持额外的标准化截断；如自行
增加输入变换，必须保证逆变换仍成立。真实训练还需要原项目要求的词表、图像、
语言、状态/动作归一化资产和已适配本体的基座。

初始化时严格检查 bundle 的完整键、形状、dtype、有限性及架构配置。不会静默退回
随机预训练参数。Small 或其他官方架构需在 JSON 的 `lingbot_touch.encoder_config`
中提供其完整 `config.json`，并使用匹配权重与历史长度。

训练 checkpoint 保存全部编码器权重、读取层、输入逆标准化 buffers 与源权重 SHA256；
恢复和部署不再读取初始化 bundle。压力几何、标定和冻结设置沿用已有 checkpoint
元数据检查。使用新结构需要重新训练触觉分支；旧 MLP 的 VTLA checkpoint 不能
直接严格加载为 LingBot checkpoint。原无触觉 π0.5 基座可通过原迁移入口加载。

从完整 VTLA checkpoint 开始新实验时，加载前也会核对几何、标定、输入时间尺度、
残差设置及归一化资产。允许改变冻结策略；改变传感器语义或统计会报错，即使张量
形状相同也不能直接沿用。完整 checkpoint 缺少 `pressure_config.json` 时同样拒绝加载。

## 两阶段微调

先确认 π0.5 已适配目标机器人的相机、状态和动作接口，并能完成基本抱夹。
若本体或动作表示尚未适配，应先完成基座训练。下面两阶段均冻结基座，
触觉分支负责在已有动作策略上学习修正。

### 数据与标定准备

每条示范轨迹需要图像、任务指令、机器人状态、连续触觉历史和示范动作。
数据应覆盖接近、收拢、抱持、搬运和释放。按完整轨迹或物体划分训练集与
验证集，避免相邻帧同时出现在两边。NPZ 和 manifest 格式见
[数据契约](../README_VTLA.zh-CN.md#数据契约)。

将实际皮肤片数、行列、测点坐标、信号单位、零载、满量程变化方向和参考
采样间隔填入配置。两阶段使用相同的标定、时间对齐和归一化资产。
`sensor_zero` 和 `sensor_span` 使用原始数据单位；训练集均值不能代替设备零载。
小臂坐标必须来自实际布局。Tiny 的历史长度需为 2 的倍数。

触觉统计只在训练划分上计算：

```bash
python scripts/compute_pressure_stats.py \
  --manifest /data/train.json \
  --output /data/pressure_stats.npz
```

状态与动作统计仍由 `scripts/compute_norm_stats.py` 计算。该脚本从训练注册配置
读取 manifest、动作维数和资产路径，不会继承另一条训练命令的 CLI 覆盖；先在
`src/openpi/training/config.py` 配好对应数据，再计算统计。完整依赖、词表、
Transformers 替换文件和基座权重按 [训练准备](../README_VTLA.zh-CN.md#绑定真实数据与完整模型训练)
配置。集群训练命令放入 sbatch 脚本，并按站点要求使用 srun。

### 阶段一：冻结 Tiny，训练小臂适配层

在完整小臂 JSON 的 `lingbot_touch` 对象内设置：

```json
"freeze_pretrained": true
```

上面只展示需要设置的字段。配置文件仍需包含阵列布局、几何、标定和其他
必需项。`pi05_pressure_forearm` 默认已启用残差路径并冻结基座。

| 模块 | 阶段一 | 阶段二 |
| --- | --- | --- |
| π0.5 视觉语言主干、动作专家和基础动作投影 | 冻结 | 冻结 |
| Tiny 压阻 tokenizer、Transformer blocks、所选输出层归一化、可选 shared projector | 冻结 | 更新 |
| Tiny 其他传感模态与未使用的最终归一化层 | 冻结 | 冻结 |
| 皮肤片 embedding、质量/时间适配层、输出映射 | 更新 | 更新 |
| 状态条件注意力读取、动作残差头 | 更新 | 更新 |

阶段一先让新增模块学会读取预训练特征。下面的路径和动作维数都是占位符；
学习率、5000 步训练和 200 步 warmup 仅作试跑起点，尚未由真实数据验证。
batch 大小沿用 preset 的 1，可在检查显存后调整。

```bash
python scripts/train_pytorch.py pi05_pressure_forearm \
  --exp-name forearm_adapter \
  --pressure-config-path /data/forearm_stage1.json \
  --lingbot-touch-bundle-path checkpoints/lingbot-touch-tiny \
  --pytorch-weight-path /models/pi05_robot_base \
  --data.manifest-path /data/train.json \
  --data.pressure-stats-path /data/pressure_stats.npz \
  --data.robot-action-dim YOUR_ACTION_DIM \
  --num-train-steps 5000 \
  --lr-schedule.peak-lr 1e-4 \
  --lr-schedule.decay-lr 1e-5 \
  --lr-schedule.warmup-steps 200 \
  --lr-schedule.decay-steps 5000
```

从验证表现中选定一个 checkpoint。默认保存位置为
`checkpoints/pi05_pressure_forearm/forearm_adapter/<step>/`。
该目录应包含完整模型、训练状态、压力配置及归一化资产。

### 阶段二：从阶段一权重解冻微调

复制阶段一的完整配置，仅将冻结字段改为：

```json
"freeze_pretrained": false
```

使用新的实验名和优化器，从选定的阶段一 checkpoint 加载所有模型权重。
这一步不传官方 Tiny bundle；训练入口会拒绝同时加载完整 VTLA checkpoint 和初始化 bundle，
以免覆盖已训练的 Tiny 参数。
下面以 2000 步、峰值学习率 `1e-5` 为试跑起点，需根据验证结果调整。

```bash
python scripts/train_pytorch.py pi05_pressure_forearm \
  --exp-name forearm_encoder_ft \
  --pressure-config-path /data/forearm_stage2.json \
  --pytorch-weight-path /models/selected_stage1_checkpoint \
  --data.manifest-path /data/train.json \
  --data.pressure-stats-path /data/pressure_stats.npz \
  --data.robot-action-dim YOUR_ACTION_DIM \
  --num-train-steps 2000 \
  --lr-schedule.peak-lr 1e-5 \
  --lr-schedule.decay-lr 1e-6 \
  --lr-schedule.warmup-steps 100 \
  --lr-schedule.decay-steps 2000
```

默认所有可训练参数仍使用同一学习率。若希望 Tiny 更新更小，可在阶段二命令中加入
`--lingbot-pretrained-lr-scale 0.1`：以上例峰值 `1e-5` 为基准，新增适配层和残差头
使用 `1e-5`，已解冻的 Tiny 参数使用 `1e-6`。这个比例在 warmup 和衰减期间保持不变，
冻结参数不会加入优化器。倍率必须为有限正数，默认值为 `1.0`。
这是可选起点，需要用验证集判断；当前没有自动两阶段调度或仅解冻最后几层的配置。

### 损失、恢复与模型选择

两阶段都使用示范动作监督，无需人工标注触觉特征或正确残差。训练沿用
`x_t = t * noise + (1-t) * actions`、`u_t = noise - actions`，优化
基础去噪速度与触觉修正之和对目标向量场的均方误差。梯度经残差头和读取层
传回新增适配层；阶段二还更新已解冻的 Tiny 参数。该过程没有新增 JEPA 损失。

同一阶段中断后，保持实验名、冻结设置、数据与模型配置，使用 `--resume`
继续训练。恢复会读取该实验最新 checkpoint 的优化器和步数。
新 checkpoint 还记录每个优化器参数组的参数名、顺序与学习率倍率；恢复时会核对
这些字段及 AdamW 的 betas、eps 和 weight decay。改变冻结集合、倍率或优化器设置
应新建实验。旧优化器文件没有参数名时，会提示警告并只核对组大小及已有设置。
两阶段切换时改变了优化器参数集合，应新建实验并通过
`--pytorch-weight-path` 加载权重，不能用 `--resume` 跨阶段恢复。

新训练若发现实验目录已存在，会要求更换实验名或明确使用 `--resume` / `--overwrite`；
不会静默混入旧 checkpoint。梯度非有限时会在优化器更新前报错。
当前恢复不保存数据迭代位置和所有随机数状态，不能保证与未中断训练逐步一致。

比较无触觉基座、冻结 Tiny 的适配版和解冻微调版，保持数据划分和评估条件
一致。验证集动作损失用于观察拟合，抱夹成功率、滑移和接触变化后的恢复
用于评估策略；有可靠标定时再报告物理压力或接触力。当前训练脚本没有自动
任务评测或早停流程，需要另行组织验证。若解冻只改善训练损失，应优先保留
验证表现更好的冻结版本。

## 验证与边界

组件测试：`python -m pytest tests/vtla -q`。

真实预训练权重、合成输入的 CPU 检查：

```bash
python scripts/check_lingbot_forearm.py \
  --bundle checkpoints/lingbot-touch-tiny \
  --output tmp/lingbot-touch-analysis/real-weight-smoke.json
```

该检查覆盖官方前向参考的一致性、冻结权重时适配层反向、解冻预训练层反向和完整
checkpoint 恢复；一致性对照暂时移除新增的皮肤 embedding，并使用匹配单指布局。
这项检查验证权重和计算复用；小臂零样本泛化仍需真实数据评估。
`tests/vtla` 的残差集成测试使用小型替身 VLA；`tests/pressure_integration` 则运行
真实模型模块的缩小配置、测试词表和合成数据。两者均未加载完整 π0.5 权重，
也未运行机器人数据训练、GPU 训练、实机任务或延迟测试。

2026-11-14 本机验证：Python 3.11、torch 2.7.1+cpu、numpy 1.26.4；
`tests/vtla` 共 166 项通过。真实 Tiny 权重的参考前向最大绝对误差为
`1.8477439880371094e-06`；冻结适配、解冻微调反向及独立 checkpoint 恢复通过。
源 `encoder.pt` SHA256 为
`ddb8b435963348fff5b8a83bb48f995978dc8841d82f87be007c775bf5a88055`。

2026-11-15 在独立 `.venv-integration` 环境补齐完整 CPU 集成依赖后，组件测试
185 项、集成测试 9 项通过。新增链路使用随机缩小 LingBot 和 π0.5：冻结编码器训练、
checkpoint 独立部署、阶段内恢复、加载阶段一并解冻，以及保留 0.1 学习率倍率的阶段二恢复
均通过检查。配置 CLI 也能解析新倍率。该检查建立的是训练链路正确性，未衡量任务效果。
复现命令与环境见 [验证记录](VALIDATION.md#2026-11-15-training-contracts-and-windows-integration)。
