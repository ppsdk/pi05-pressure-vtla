# π0.5 压力 VTLA 模型维度流

本文对应当前 PyTorch 实现 `src/openpi/models_pytorch/pi0_pytorch.py`、`pressure_pytorch.py` 和 `PressureConfig`。

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
