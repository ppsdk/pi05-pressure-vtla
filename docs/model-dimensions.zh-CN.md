# π0.5 原生架构与压力 VTLA 模型维度流

本文先说明官方 π0.5 的原生 PyTorch 架构，再说明当前电阻式压力阵列 VTLA 的增量。对应实现为 `src/openpi/models_pytorch/pi0_pytorch.py`、`gemma_pytorch.py`、`pressure_pytorch.py` 和 `PressureConfig`。原生基线仍可在没有压力字段时运行；压力配置只在 PyTorch π0.5 路径启用。

符号：`B` batch；`S` 压力传感器槽位（可混合夹爪/小臂）；`K` 历史帧；`R,C` 阵列行列；`Pr,Pc` patch 行列；`Np=(R/Pr)*(C/Pc)` patch 数；`A=action_dim`；`H=action_horizon`；`L` 实际语言/离散状态 token 数；`Dv` PaliGemma 宽度；`De` ActionExpert 宽度；`Dh` 压力 MLP 隐藏宽度。

## 当前配置数值

| 配置 | `pi05_pressure` | `pi05_pressure_debug` |
|---|---:|---:|
| PaliGemma width `Dv` | 2048 | 64 |
| ActionExpert width `De` | 1024 | 64 |
| `action_dim A` | 32 | 8 |
| `action_horizon H` | 50 | 4 |
| `S,R,C,K,Pr,Pc` | 2,8,8,1,2,2 | 2,8,8,1,2,2 |
| 压力 token 数 `S*K*Np` | 32 | 32 |
| 压力 MLP `Dh` | 128 | 128 |

debug 配置是随机初始化 CPU 链路测试模型，不能代表官方 π0.5 权重。

## 0. π0.5 原生架构（无压力输入）

π0.5 不是单一宽度的普通语言模型，而是由 PaliGemma 视觉语言主干和一个较窄的 ActionExpert 组成的双专家模型。两者在每层注意力中使用相同的 `num_heads`、`num_kv_heads` 和 `head_dim`，但保持各自的 hidden width；各自投影得到 Q/K/V 后，在注意力序列维拼接，注意力输出再按原序列长度拆回两个专家。

### 0.1 原生配置和输入

官方 π0.5 默认配置为：

| 部分 | 变体 | width | layers | MLP width | query heads | KV heads | head dim |
|---|---|---:|---:|---:|---:|---:|---:|
| PaliGemma 语言模型 | `gemma_2b` | 2048 | 18 | 16384 | 8 | 1 | 256 |
| ActionExpert | `gemma_300m` | 1024 | 18 | 4096 | 8 | 1 | 256 |

π0.5 的原生观测包括三路图像、任务 prompt 和机器人 state。state 不作为连续 suffix token 输入，而是先按归一化范围 `[-1,1]` 离散到 256 个 bin，再与 prompt 拼成 PaliGemma 的语言输入。tokenizer 的逻辑可概括为：

```text
state                 [B,D_state]
离散 state             [B,D_state]，每个值 ∈ {0,...,255}
文本                   "Task: <prompt>, State: <discrete state>;\\nAction: "
token ids              [B,L]
token embeddings       [B,L,2048]
```

以三路 224×224 RGB、每路 256 个 SigLIP patch token 为例：

```text
base RGB               [B,3,224,224] -> [B,256,2048]
left wrist RGB         [B,3,224,224] -> [B,256,2048]
right wrist RGB        [B,3,224,224] -> [B,256,2048]
三路视觉 tokens                         [B,768,2048]
语言/离散状态 tokens                    [B,L,2048]
原生 prefix                             [B,P0,2048]
P0 = 768 + L
```

缺失相机通过 `image_mask=False` 保留槽位。它仍占据预处理产生的 token 槽位，但在 padding mask 中失效；实际有效 prefix 长度由有效 mask 决定。

### 0.2 原生 prefix 和 action suffix

原生 π0.5 的序列分成两个逻辑部分：

```text
prefix = [视觉 tokens, 语言/离散状态 tokens]
suffix = [动作 chunk tokens]
```

训练时给动作标签 `a` 加入 flow matching 噪声 `ε`：

```text
actions a             [B,H,32]
noise ε               [B,H,32]
t                     [B]，Beta(1.5, 1.0) 后缩放到约 (0,1)
x_t                   [B,H,32] = t*ε + (1-t)*a
u_t                   [B,H,32] = ε - a
```

动作 suffix 的输入和时间条件为：

```text
x_t                   [B,H,32]
action_in_proj        [B,H,32] -> [B,H,1024]
time sinusoid          [B,1024]
time MLP               [B,1024] -> [B,1024]
adaRMSNorm condition   [B,1024]
suffix embeddings      [B,H,1024]
```

π0.5 的 `H=50`、`action_dim=32` 是默认模型契约；实际机器人动作维度可以小于 32，由输入/输出适配器补零或裁剪。

### 0.3 原生 attention mask 和联合前向

原生 mask 不是把动作 chunk 当作逐 token 文本解码。用 `ar_mask` 表示注意力块边界时：

```text
prefix ar_mask       [0, 0, ..., 0]
suffix ar_mask       [1, 0, ..., 0]
```

由此得到以下可见性：

| 查询 token | 可以读取 |
|---|---|
| prefix | 所有有效 prefix token，不能读取 suffix |
| suffix 第一个 token | 所有有效 prefix 和整个 suffix chunk |
| suffix 后续 token | 所有有效 prefix 和整个 suffix chunk |

因此动作 chunk 是联合建模的，动作 token 之间可以互相注意；这与自回归文本生成的逐 token causal mask 不同。训练时，`prefix` 和 `suffix` 一次联合前向：

```text
prefix                 [B,P0,2048]
suffix                 [B,H,1024]
拼接后的逻辑长度       N0 = P0 + H
attention mask         [B,N0,N0]（再扩为 [B,1,N0,N0]）
position ids           [B,N0]
ActionExpert 输出      [B,H,1024]
action_out_proj        [B,H,1024] -> [B,H,32]
v_t                    [B,H,32]
loss                   mean((v_t-u_t)^2)
```

### 0.4 原生推理和 prefix KV cache

动作推理时，π0.5 不生成文本 token，而是先对当前观测的 prefix 做一次 prefill：

```python
PaliGemma(prefix, use_cache=True) -> past_key_values
```

每个 PaliGemma 层的缓存可抽象为：

```text
K_prefix[l], V_prefix[l]  [B,1,P0,256]
```

随后进行 `num_steps` 次 Euler flow matching 去噪。每次只重新构造动作 suffix：

```text
ActionExpert(suffix, past_key_values=prefix_cache, use_cache=False)
    -> v_t [B,H,32]
x_t <- x_t + dt*v_t
```

在注意力层内部，ActionExpert 当前动作 token 的 K/V 与缓存的 prefix K/V 按序列维拼接：

```text
K = concat(K_prefix, K_suffix)
V = concat(V_prefix, V_suffix)
Q = Q_suffix
attention(Q, K, V)
```

这里的 `past_key_values` 特指 PaliGemma **语言 Transformer** 的缓存。SigLIP 视觉塔先把 RGB 变为视觉 embedding；视觉塔本身不通过这份 `past_key_values` 做逐 patch 解码。视觉 embedding 和语言 embedding 一起作为语言 Transformer 的 prefix 输入，因此这些视觉 token 仍会在语言 Transformer 的 K/V 中留下对应位置。

所以原生 π0.5 的 cache 是观测 prefix 的 KV cache：PaliGemma 语言 Transformer 负责填充，ActionExpert 在每个去噪步读取。训练阶段 `use_cache=False`；新的图像、语言或 state 到来时，必须重新建立 prefix cache。

### 0.5 原生数据流数值例子

取 `B=2`、`L=120`、`H=50`、`A=32`：

```text
视觉 tokens             [2,768,2048]
语言/离散状态 tokens     [2,120,2048]
原生 prefix             [2,888,2048]
动作噪声 x_t             [2,50,32]
ActionExpert suffix      [2,50,1024]
联合逻辑长度             888 + 50 = 938
Flow 输出 v_t             [2,50,32]
最终动作 chunk            [2,50,32]
```

压力 VTLA 只在这个原生 prefix 后追加压力 tokens；它不改变 ActionExpert 的 `1024` 宽度、动作维度 `32`、动作 horizon `50` 或 Flow Matching 目标。

## 1. 数据输入

机器人数据：

```text
state   [N, D_robot_state]
actions [N, D_robot_action]
RGB     [N, H_img, W_img, 3]
```

进入模型后：

```text
state   [B,D_robot_state] -> PadStatesAndActions -> [B,A]
actions [B,H,D_robot_action] -> 补零 -> [B,H,A]
RGB     [B,H_img,W_img,3] -> resize/pad -> [B,3,224,224]
```

真实 `D_robot_action` 必须小于等于 `A`；输出适配器将 `[B,H,A]` 裁成 `[B,H,D_robot_action]`。

### 数值例子：夹爪机器人单步采样

下面是一组用于说明的假设值：双臂夹爪机器人，左右臂各 7 个关节和 1 个夹爪开合量；采集相机为 640×480；模型使用 `A=32`、`H=50`。这些数字是维度示例，不是对你的实际机器人接口的判断。

```text
N = 1200                         # 一个 episode 的同步时间步
D_robot_state = 16               # 左臂 8 + 右臂 8
D_robot_action = 16              # 左臂 8 + 右臂 8 的控制量
H_img = 480, W_img = 640

state   [1200,16]
actions [1200,16]
RGB     [1200,480,640,3]         # uint8 原始图像
```

取 batch 中第 `i=37` 个决策时刻，并构造长度为 50 的动作 chunk：

```text
单步 state                 [16]
单步 RGB                   [480,640,3]
动作标签 chunk             [50,16]

batch_size B=2 后：
state                      [2,16]
actions                    [2,50,16]
RGB                        [2,480,640,3]
```

经过模型数据变换：

```text
state  [2,16]       -> PadStatesAndActions -> [2,32]
actions[2,50,16]    -> 右侧补 16 个零维度 -> [2,50,32]
RGB    [2,480,640,3]-> resize/pad         -> [2,3,224,224]
```

模型推理输出 `[2,50,32]`，机器人适配器再裁掉补零维度：

```text
[2,50,32] -> [:,:,:16] -> [2,50,16]
```

这说明 `A=32` 是 π0.5 模型内部的统一动作宽度，`D_robot_action=16` 才是这个假设机器人实际执行的动作宽度。

### 数值例子：单臂夹爪 + 小臂压力阵列

再假设每个时间步有 2 个压力槽位：

```text
S=2                             # [left_gripper, forearm_left]
K=4                             # 最近 4 帧压力历史
R=C=8                           # 每个传感器 8×8
Pr=Pc=2                         # 2×2 patch
```

那么一个 batch 的压力字段为：

```text
values       [2,2,4,8,8]
frame_valid  [2,2,4]
taxel_valid  [2,2,4,8,8]
saturated    [2,2,4,8,8]
age          [2,2,4]
```

每个传感器每帧有 `Np=(8/2)*(8/2)=16` 个 patch，最终压力 token 为：

```text
[2,2,4,8,8] -> [2,128,2048]
```

如果某个时刻只有夹爪数据、小臂传感器掉线，仍然保持 `[2,128,2048]` 的形状，只把小臂对应的 64 个 token 标为无效；不会改变 batch 或序列宽度。

压力输入：

```text
values       [B,S,K,R,C]       float32
frame_valid  [B,S,K]           bool
taxel_valid  [B,S,K,R,C]       bool
saturated    [B,S,K,R,C]       bool
age          [B,S,K]           float32
```

`S=2` 可以是 `[left_gripper, forearm_left]`；`S=1` 可以只用小臂或只用夹爪。`sensor_names` 和 `sensor_locations` 是元数据，不增加张量维度。

## 2. 压力编码器

令 `Q=Pr*Pc`，`Np=(R/Pr)*(C/Pc)`：

```text
values                         [B,S,K,R,C]
patchify                       [B,S,K,Np,Q]
stack(value,valid,saturated)  [B,S,K,Np,3Q]
patch_in Linear(3Q -> Dh)      [B,S,K,Np,Dh]
SiLU + patch_out(Dh -> Dv)     [B,S,K,Np,Dv]
sensor embedding               [S,Dv] -> broadcast
spatial embedding              [Np,Dv] -> broadcast
age MLP                        [B,S,K,Dv] -> broadcast over Np
flatten                        [B,S*K*Np,Dv]
token_valid                    [B,S*K*Np]
```

当前示例：`Q=4`、`Np=16`，因此：

```text
[B,2,1,8,8] -> [B,2,1,16,4] -> [B,2,1,16,12]
             -> [B,2,1,16,128] -> [B,2,1,16,2048]
             -> [B,32,2048]
```

压力编码器输出最后一维必须等于 `Dv=2048`，不等于 ActionExpert 的 `De=1024`。

## 3. 视觉、语言和状态

三路图像槽位分别经过 PaliGemma 的 SigLIP 视觉塔。官方 224×224 输入、14×14 patch 产生：

```text
每路图像 [B,3,224,224] -> [B,256,Dv]
三路图像                 -> [B,768,Dv]
```

缺失相机保留槽位，但 `image_mask=False`。

π0.5 默认 `discrete_state_input=True`，prompt 与离散化 state 合并为：

```text
token ids      [B,L]
token mask     [B,L]
embeddings     [B,L,Dv]
```

所以当前 pi0.5 路径不会额外添加连续 state suffix token。压力也不进入 state tokenization。

## 4. Prefix 拼接

顺序固定为：

```text
[视觉 tokens] + [语言/离散状态 tokens] + [压力 tokens]
```

```text
image_embs    [B,768,Dv]
lang_embs     [B,L,Dv]
pressure_embs [B,S*K*Np,Dv]
prefix_embs   [B,P,Dv]
P = 768 + L + S*K*Np
```

当前示例为 `P=768+L+32=800+L`；如果 `L=200`，则 `prefix=[B,1000,2048]`。

所有 prefix token 使用双向 prefix attention；无效图像/压力 token 通过 padding mask 失效，位置编号为 `cumsum(valid)-1`。

## 5. ActionExpert 和 Flow Matching

训练时：

```text
actions, noise [B,H,A]
time            [B]
x_t             [B,H,A] = t*noise + (1-t)*actions
u_t             [B,H,A] = noise - actions
action_in_proj  [B,H,A] -> [B,H,De]
time sinusoid   [B,De]
time MLP        [B,De] -> [B,De]  (adaRMSNorm 条件)
suffix_embs     [B,H,De]
```

训练主干输入：

```text
prefix_embs [B,P,Dv]
suffix_embs [B,H,De]
logical length N=P+H
pad_masks   [B,N]
att_masks   [B,N]
att_2d      [B,N,N]
att_4d      [B,1,N,N]
position_ids[B,N]
```

ActionExpert 输出：

```text
expert hidden   [B,H,De]
action_out_proj [B,H,De] -> [B,H,A]
v_t             [B,H,A]
loss            mse(u_t,v_t) -> [B,H,A]，随后聚合
```

动作维度、动作 horizon 和 Flow Matching 结构没有因增加压力 token 而改变。

## 6. Cached inference

推理先对完整 prefix 建缓存：

```text
prefix_embs         [B,P,Dv]
prefix_pad_masks    [B,P]
prefix_position_ids [B,P]
prefix_att_2d       [B,P,P]
```

每个 PaliGemma 层的 KV cache 抽象序列形状为：

```text
K/V [B,num_kv_heads,P,head_dim]
```

官方 `gemma_2b` 配置为 `num_kv_heads=1`、`head_dim=256`。每个 Euler 步只重新计算：

```text
suffix_embs         [B,H,De]
position_ids        [B,H]
full attention mask [B,H,P+H]
v_t                 [B,H,A]
```

压力只改变 prefix 内容、有效性和缓存长度 `P`，不改变 `H`、`A`、`De`。

## 7. 混合夹爪+小臂示例

假设 `S=2,K=4,R=C=8,Pr=Pc=2,L=120,A=8,H=16`：

```text
pressure values  [B,2,4,8,8]
pressure tokens  [B,2*4*16,2048] = [B,128,2048]
image tokens     [B,768,2048]
lang/state       [B,120,2048]
prefix           [B,1016,2048]
suffix           [B,16,1024]
logical length   1032
flow output      [B,16,8]
robot output     [B,16,D_robot_action]
```

## 8. 核心结论

1. 新增序列长度只有 `ΔP=S*K*(R/Pr)*(C/Pc)`。
2. 压力 token 最后一维是 `Dv=2048`，必须映射到 PaliGemma 宽度。
3. ActionExpert/Flow Matching 仍输出 `[B,H,A]`。
4. `history` 或传感器槽位增加只增加 prefix 长度，不增加主干宽度。
5. 无效帧、坏 taxel、饱和不会改变 shape，只影响 token validity 和 attention mask。
6. 当前实现要求同一数据划分内的 `[S,R,C]` 几何一致；不同阵列需要先适配或重采样。

## 9. 证据边界

这些是代码配置与张量契约的维度，不是完整官方权重在本机已经训练得到的性能结果。当前 CPU 只验证小模型和组件链路；完整 π0.5 权重训练、真实夹爪/小臂数据和实机闭环尚未验证。
