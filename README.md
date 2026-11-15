# π0.5 Pressure VTLA

基于官方 [openpi](https://github.com/Physical-Intelligence/openpi) 的数值压力/触觉阵列扩展，采用 PyTorch。支持触觉前缀融合与触觉动作向量场残差，沿用原 ActionExpert 与 Flow Matching 动作生成结构。

CPU 组件测试及缩小模型的两阶段训练、恢复和推理链路已通过。LingBot-Touch Tiny 真实权重已完成合成输入的前向、反向和恢复检查；完整 π0.5 权重训练与真实机器人效果尚未验证。

## 文档入口

- [技术框图与项目总览（含小臂残差模式）](docs/technical-overview.zh-CN.md)
- [中文安装、数据格式与使用说明](README_VTLA.zh-CN.md)
- [LingBot-Touch 小臂适配与两阶段微调](docs/lingbot-forearm.zh-CN.md)
- [架构与改造规格](docs/pressure-vtla-design.zh-CN.md)
- [验证记录与证据边界](docs/VALIDATION.md)
- [上游来源与修改范围](UPSTREAM.md)
- [官方上游原始 README](docs/UPSTREAM_README.md)

## 架构

1. 原始电容/ADC/电阻或已标定信号使用固定统计缩放，保留有效性、坏点、饱和及实际采样年龄。
2. 编码器可选择局部 patch MLP，或使用 LingBot-Touch 预训练权重的小臂适配版本，输出宽度与 PaliGemma 一致。
3. 前缀融合模式的顺序为 `[视觉 tokens, 语言与离散状态 tokens, 压力 tokens]`，有效 token 双向融合。小臂默认残差模式只将触觉送入独立修正分支。
4. 沿用原 mask/位置编号和缓存推理；动作后缀与时间条件保留。残差模式在基础去噪速度上叠加触觉修正，使用相同的 Flow Matching 目标。

阵列数量、行列数、历史窗口和 patch 大小通过 `PressureConfig` 配置。仓库中的示例尺寸不代表实际设备。数据层不伪造 ADC/电阻到物理压力或接触力的标定。

小臂安装配置使用 `pi05_pressure_forearm`，当前默认冻结基座策略，触觉只进入动作向量场残差头。可通过 `--pressure-config-path examples/pressure/forearm.json` 配置布局、历史、信号单位，以及可选物理坐标和历史差分。示例几何不代表真实设备，详见中文说明与技术总览。

启用 LingBot-Touch 时，在完整压力 JSON 中配置小臂几何和信号标定，并加入
`lingbot_touch`。先冻结 Tiny 训练适配层，再按验证表现决定是否解冻微调。
阶段切换需要新建实验并加载上一阶段权重，具体命令见两阶段微调文档。
加载时会核对几何、标定和归一化资产；阶段二可用 `--lingbot-pretrained-lr-scale`
为已解冻的预训练编码器设置较小学习率。恢复会检查优化器参数组及倍率。

## CPU 快速验证

Python 3.11，在仓库根目录执行：

```powershell
python -m venv .venv-pressure
.\.venv-pressure\Scripts\python.exe -m pip install -r requirements-pressure-cpu.txt
.\.venv-pressure\Scripts\python.exe -m pytest tests/vtla -q
```

Linux/macOS 对应使用 `.venv-pressure/bin/python`。组件测试不下载模型权重，也不安装完整 openpi 训练栈。GitHub Actions 使用 CPU 依赖执行同一测试集。

其中接入链路测试调用真实 `PI0Pytorch` 方法，并使用小型替身注意力主干；它们不构成完整官方主干的数值一致性或任务效果证据。

## 主要文件

| 文件 | 功能 |
| --- | --- |
| [pressure_data.py](src/openpi/pressure_data.py) | 因果读取、固定归一化、输出动作裁维 |
| [pressure_dataset.py](src/openpi/training/pressure_dataset.py) | manifest/NPZ 数据集与训练压力统计 |
| [pressure_config.py](src/openpi/models/pressure_config.py) | 显式硬件/token 配置 |
| [pressure_pytorch.py](src/openpi/models_pytorch/pressure_pytorch.py) | 可训练压力 patch 编码器 |
| [lingbot_forearm.py](src/openpi/models_pytorch/lingbot_forearm.py) | 预训练触觉权重、小臂时空编码与状态条件读取 |
| [pi0_pytorch.py](src/openpi/models_pytorch/pi0_pytorch.py) | 前缀、训练 forward 与缓存推理接入 |
| [pressure_training.py](src/openpi/training/pressure_training.py) | 受控加载、冻结与 checkpoint 契约 |
| [pytorch_training.py](src/openpi/training/pytorch_training.py) | 分组学习率、优化器恢复和实验目录检查 |
| [tests/vtla](tests/vtla) | CPU 组件与接入测试 |

完整模型训练前需绑定真实阵列尺寸、原始数据和时间契约、标定资产、机器人动作定义与官方权重，详见中文使用说明。

## 来源

基线为 `Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479`。保留上游 Git 历史、许可证和源码声明。本仓库为二次开发，非 Physical Intelligence 官方发布。模型权重与实际数据未包含在内。

LingBot-Touch 源码固定于 `dec8025b212e7099b235cc2ef13a4921a7e59545`，保留其 Apache-2.0 许可证和第三方声明。预训练权重需从官方仓库另行下载。
