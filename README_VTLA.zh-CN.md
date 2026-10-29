# π0.5 电阻式压力阵列 VTLA 接入

基于官方 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，基线提交 `215abfb217dbac7d5f1273282331b9b1866c0479`。新增实现采用 **PyTorch**；JAX 原模型保留，配置压力分支时会明确拒绝使用 JAX 模型工厂。

本机仅使用 CPU。本次验证包括数据、压力编码器、训练加载/冻结/恢复，以及调用真实 `PI0Pytorch` 方法的小型替身主干测试；**没有加载完整 π0.5 权重，没有运行完整主干训练，也没有真实机器人或传感器实验**。替身主干用于检查计算链路，不代表官方权重数值验证或任务效果。

2026-10-29 验证：Python 3.11.1、torch 2.7.1+cpu、numpy 1.26.4，`cuda_available=False`；CPU 测试 **93 passed in 9.52s**，21 个修改/新增 Python 文件 Ruff、格式与语法编译检查通过。源码保留上游历史，硬件参数与真实数据仍待绑定。

## 实现内容

- 数值压力阵列独立于 RGB 图像、机器人 `state` 和动作标签；不生成伪彩图，不经过 SigLIP 或图像增强。
- 每片阵列按局部 patch 经过共享 MLP 编码；加入 sensor、空间和实际采样 age 编码，再映射到 PaliGemma 的前缀宽度，默认 2048。
- token 顺序为 `[原视觉, 原语言+离散状态, 压力]`。前缀双向融合；动作后缀保留原块注意力、1024 维 ActionExpert、adaRMSNorm 时间条件、Flow Matching 目标与 Euler 采样。
- 压力有效性并入原 token mask 和 `cumsum(valid)-1` 位置编号；推理先建立包含压力的完整前缀 KV cache，同一次动作生成内复用。
- 无接触、缺帧、坏 taxel、饱和和陈旧度分别表达。无效/非有限信号在任何学习操作前清理；饱和仍作为受限测量输入。
- 基座迁移仅允许新增 `pressure_encoder.*` 参数缺失。正式恢复严格加载；checkpoint 保存压力配置、固定归一化资产和其 SHA256。
- 可冻结 PaliGemma 权重，动作专家与压力分支继续训练；不对整个冻结主干使用 `no_grad()`，保留压力输入的反向梯度。

## CPU 上立即运行验证

以下只安装组件测试依赖，不会安装完整 openpi 训练栈或下载模型权重。Python 3.11，在本仓库目录执行：

```powershell
python -m venv .venv-pressure
.\.venv-pressure\Scripts\python.exe -m pip install -r requirements-pressure-cpu.txt
.\.venv-pressure\Scripts\python.exe -m pytest tests/vtla -q
```

Linux/macOS 对应使用 `.venv-pressure/bin/python`。组件测试使用 CPU；GitHub Actions 执行相同测试集。

`pi05_pressure_debug` 现在使用随机初始化的小型视觉塔和 dummy Gemma 专家，可用于 CPU 链路调试；它不兼容官方 π0.5 权重，不能用于声称策略性能。

要额外运行含真实 openpi 数据转换、模型模块、训练脚本、checkpoint 恢复和策略推理的 CPU 集成测试，可在仓库根目录创建独立环境：

```powershell
python -m venv .venv-integration
.\.venv-integration\Scripts\python.exe -m pip install -r requirements-pressure-integration-cpu.txt
.\.venv-integration\Scripts\python.exe scripts/install_transformers_patch.py
.\.venv-integration\Scripts\python.exe -m pytest tests/pressure_integration -q
```

替换脚本只接受虚拟环境内的 Transformers 4.53.2，并安装仓库原有的 openpi 替换文件。集成测试使用合成 NPZ、测试用 SentencePiece 词表与随机初始化的缩小模型；不会下载官方模型权重。完整的官方模型、真实数据和真实词表需要按上游说明另行配置。

## 数据契约

硬件尚未确定，因此 `num_sensors/rows/cols/history/patch_rows/patch_cols` 是显式配置。仓库中的 `2 片 × 8×8、K=1、2×2 patch` 仅为可替换的配置示例，不是对用户设备的假定。

一个 manifest JSON 显式列出同一个数据划分的 episode：

```json
{
  "episodes": [
    {
      "observations": "episodes/robot_001.npz",
      "pressure": "episodes/pressure_001.npz",
      "prompt": "完成指定操作"
    }
  ]
}
```

路径相对 manifest 所在目录。一个条目为一个 episode，不允许历史或动作标签跨 episode。所有文件用 `allow_pickle=False` 读取。

机器人 NPZ：

| 键 | 形状/类型 | 说明 |
| --- | --- | --- |
| `timestamps` | `[N]`，秒 | 有限、严格递增，共同时钟域 |
| `state` | `[N,D_state]`，数值 | 真实状态，沿原 π0.5 归一化与状态文本化 |
| `actions` | `[N,D_robot]`，数值 | 真实动作标签，坐标系/绝对或增量语义由采集方声明 |
| `base_0_rgb` | `[N,H,W,3]` | uint8 或 float32 `[-1,1]`，必填 |
| `left_wrist_0_rgb` / `right_wrist_0_rgb` | 同上，可选 | 缺失时置图像 mask=false |
| `<相机键>_valid` | `[N]`，bool，可选 | 相机可用性 |

仅产生每 episode 的 `N-H+1` 个完整动作 chunk；短 episode 不产生样本，不完整尾部计入 `dropped_tail_samples`，不补造动作标签。dataset 构造时核对全部 episode 的阵列几何及 state/action 维数一致。

压力 NPZ：

| 键 | 形状/类型 | 说明 |
| --- | --- | --- |
| `timestamps` | `[T]`，秒 | 有限、严格递增 |
| `values` | `[T,S,R,C]`，数值 | ADC、电阻或已标定信号，明确单位 |
| `arrival_times` | `[T]`，可选 | 实际到达时刻，不能早于采样；未提供时明确假设零传输延迟 |
| `frame_valid` | `[T,S]`，bool，可选 | 默认存在帧有效 |
| `taxel_valid` | `[T,S,R,C]`，bool，可选 | 默认有限感测单元有效 |
| `saturated` | `[T,S,R,C]`，bool，可选 | 默认不饱和；实际设备需提供正确量程判据 |

读取器只选 `采样时刻 <= 决策时刻` 且 `到达时刻 <= 决策时刻` 的最近 K 个不同帧，按旧→新排列。历史不足在左侧补无效帧，记录真实 age，不重复旧帧伪装新观测。各传感器异步时，应在采集适配层保留真实时间/可用性，不人为声称同步。

模型中的独立观测字段为：

```python
pressure = {
    "values": ...,       # float32 [B,S,K,R,C]，已使用固定统计缩放
    "frame_valid": ...,  # bool [B,S,K]
    "taxel_valid": ...,  # bool [B,S,K,R,C]
    "saturated": ...,    # bool [B,S,K,R,C]
    "age": ...,          # float32 [B,S,K]，秒
}
```

在线 `Policy.infer()` 接收不带 B 的同一字典。`NormalizePressure` 由数据配置同时用于训练与在线推理；调用 `infer()` 前传入原标定/原信号值，不要再次手动归一化。底层直接调用模型时，则需自行完成这一预处理。

## 独立压力统计

下列命令只需要 NumPy；输入必须是明确的训练划分 manifest：

```powershell
python scripts/compute_pressure_stats.py --manifest D:/data/train.json --output D:/data/pressure_stats.npz
```

输出 `mean/scale/count[S,R,C]`。只统计有效、有限、未饱和读数；近零方差使用尺度下限，无有效样本 taxel 保留 `count=0`。禁止从测试集拟合或逐帧 min-max 归一化。

该脚本只缩放信号，**不提供设备的 ADC/电阻→Pa/N 校准曲线**。设备未定时不能伪造标定；基线漂移、迟滞、串扰及量程判据需在采集适配层验证。若零载基线需要在线更新，只能用当时可获得的参考条件。

## 绑定真实数据与完整模型训练

在 `src/openpi/training/config.py` 的 `pi05_pressure` 条目填写：

1. `PressureConfig(num_sensors=..., rows=..., cols=..., history=..., patch_rows=..., patch_cols=...)`；行列需整除 patch 尺寸。
2. `PressureNpzDataConfig(manifest_path=..., pressure_stats_path=..., robot_action_dim=...)`。动作真实维数必须显式给出，输出按该维数裁去模型补维。
3. 真实动作坐标系、动作时间步、action horizon 和执行步数；不要因数组形状正确就视为控制接口匹配。
4. `pytorch_weight_path` 指向已转换的官方 π0.5 PyTorch 权重目录。保留原骨干参数名；新压力编码器随机初始化。

完整训练仍使用官方依赖、固定的 Transformers 替换文件和官方安装流程。CPU 组件环境没有安装这些完整依赖；本次没有执行以下命令：

```bash
# 先将上述真实配置、压力统计、动作定义和官方权重准备好。
uv run scripts/compute_norm_stats.py --config-name pi05_pressure
uv run scripts/train_pytorch.py pi05_pressure --exp_name pressure_run --pytorch_weight_path /path/to/pi05_torch
```

原 `compute_norm_stats.py` 继续只统计 state/actions，压力统计单独保存。`pressure_freeze_backbone=True` 冻结旧 VLM 参数，优化器仅接收可训练参数。默认压力训练精度为 float32；CPU 部署压力模型时也选择 float32 并关闭编译。完整主干在本机的时间和内存可行性尚未测量。

checkpoint 包含 `model.safetensors`、原训练状态、`pressure_config.json`、`assets/pressure_stats.npz` 及原 state/action 统计。新 checkpoint 同时记录机器人动作维数、模型动作时域、状态文本化与归一化配置，并校验 state/action 统计内容；最后一次优化器更新也会保存。在线创建 policy 时使用 checkpoint 内统计，不需要保留训练 manifest 或训练资产目录。

旧版 schema 1 的 checkpoint 没有动作与 state/action 统计契约：加载时仍兼容，但会发出无法核验这些配置的警告。切换这些设置后不能把旧 checkpoint 视为自动验证一致。

## 文件地图与证据边界

| 文件 | 作用 |
| --- | --- |
| `src/openpi/pressure_data.py` | 因果读取、独立归一化、输出动作裁维 |
| `src/openpi/training/pressure_dataset.py` | manifest/NPZ 数据集、训练压力统计 |
| `src/openpi/models/pressure_config.py` | 显式硬件与 token 配置 |
| `src/openpi/models_pytorch/pressure_pytorch.py` | 可训练压力编码器 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 真实前缀、训练 forward、cached sampling 接入 |
| `src/openpi/training/pressure_training.py` | 受控基座加载、冻结、统计资产与恢复检查 |
| `tests/vtla/` | CPU 组件测试和小型替身主干接入测试 |

这些测试不能证明真实任务收益、完整官方主干缓存数值一致性或闭环实时性。硬件参数、原始数据格式和任务确定后，还需要真实样本端到端训练/采样 smoke、完整权重回归与实机评估。新增压力不会自动提升动作 chunk 内部的触觉反馈频率。
