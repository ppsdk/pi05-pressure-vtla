# π0.5 Pressure VTLA

基于官方 [openpi](https://github.com/Physical-Intelligence/openpi) 的电阻式压力阵列触觉扩展，采用 **PyTorch**。压力 token 在输入前缀中与视觉、语言和机器人状态融合，保留原 ActionExpert 与 Flow Matching 动作生成结构。

这是工程实现首版。**CPU 组件测试已通过；尚未加载完整 π0.5 权重、运行完整模型训练或验证真实机器人效果。**

## 文档入口

- [中文安装、数据格式与使用说明](README_VTLA.zh-CN.md)
- [架构与改造规格](docs/pressure-vtla-design.zh-CN.md)
- [验证记录与证据边界](docs/VALIDATION.md)
- [上游来源与修改范围](UPSTREAM.md)
- [官方上游原始 README](docs/UPSTREAM_README.md)

## 架构

1. 原始 ADC/电阻或已标定信号使用固定统计缩放，保留有效性、坏点、饱和及实际采样年龄。
2. 局部 patch MLP 编码阵列，加入传感器、空间和时间标签，映射到 PaliGemma 前缀宽度。
3. 前缀顺序为 `[视觉 tokens, 语言与离散状态 tokens, 压力 tokens]`，有效 token 双向融合。
4. 沿用原 mask/位置编号和缓存推理；动作后缀、时间条件与 Flow Matching 目标保留。

阵列数量、行列数、历史窗口和 patch 大小通过 `PressureConfig` 配置。仓库中的示例尺寸不代表实际设备。数据层不伪造 ADC/电阻到物理压力或接触力的标定。

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
| [pi0_pytorch.py](src/openpi/models_pytorch/pi0_pytorch.py) | 前缀、训练 forward 与缓存推理接入 |
| [pressure_training.py](src/openpi/training/pressure_training.py) | 受控加载、冻结与 checkpoint 契约 |
| [tests/vtla](tests/vtla) | CPU 组件与接入测试 |

完整模型训练前需绑定真实阵列尺寸、原始数据和时间契约、标定资产、机器人动作定义与官方权重，详见中文使用说明。

## 来源

基线为 `Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479`。保留上游 Git 历史、许可证和源码声明。本仓库为二次开发，非 Physical Intelligence 官方发布。模型权重与实际数据未包含在内。
