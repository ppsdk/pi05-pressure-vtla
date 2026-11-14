"""Build the editable technical overview using only Python's standard library."""

from html import escape
from pathlib import Path


OUTPUT = Path(__file__).with_name("pi05-pressure-vtla-overview.svg")
parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="600" viewBox="0 0 1600 600">',
    '<title>π0.5 Pressure VTLA technical overview</title>',
    '<desc>Forearm tactile residual path. The pi0.5 base is adapted as needed to the target embodiment. Base and tactile flow fields are added before action denoising.</desc>',
    '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8" fill="#475569"/></marker></defs>',
    '<rect width="1600" height="600" fill="white"/>',
    '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172b40">',
]


def text(x, y, content, size=20, bold=False, anchor="start", color="#172b40"):
    weight = ' font-weight="700"' if bold else ""
    parts.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"'
        f' fill="{color}"{weight}>{escape(content)}</text>'
    )


def box(x, y, w, h, lines, fill="#eef4fa", stroke="#829ab1"):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8"'
        f' fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    first_y = y + h / 2 - (len(lines) - 1) * 15 + 7
    for index, line in enumerate(lines):
        units = sum(1.0 if ord(char) > 127 else 0.62 for char in line)
        size = min(20, (w - 24) / units)
        text(x + w / 2, first_y + index * 30, line, size, index == 0, "middle")


def arrow(points, dashed=False, color="#475569"):
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"'
        f'{dash} marker-end="url(#arrow)"/>'
    )


text(40, 53, "π0.5 Pressure VTLA", 30, True)

parts.append('<rect x="250" y="90" width="810" height="270" rx="12" fill="#fbfcfe" stroke="#cbd5e1"/>')
text(270, 125, "π0.5 基座（按需适配本体）", 21, True)
box(40, 215, 180, 80, ["RGB / 语言 / 状态"])
box(270, 215, 220, 80, ["观测编码", "SigLIP / 状态离散化"])
box(550, 215, 200, 80, ["PaliGemma", "多模态观测融合"])
box(830, 215, 210, 80, ["ActionExpert", "基础去噪速度"])
box(1240, 215, 160, 80, ["动作去噪", "Euler 积分"])
box(1450, 215, 110, 80, ["动作"])

arrow("220,255 270,255")
arrow("490,255 550,255")
arrow("750,255 830,255")
text(790, 240, "KV", 17, anchor="middle", color="#52667a")
text(935, 167, "带噪动作 / 时间", 19, anchor="middle")
arrow("935,182 935,215")
arrow("1040,255 1141,255")
parts.append('<circle cx="1170" cy="255" r="29" fill="white" stroke="#475569" stroke-width="2"/>')
text(1170, 264, "+", 30, anchor="middle")
arrow("1199,255 1240,255")
arrow("1400,255 1450,255")

green = {"fill": "#e9f6f0", "stroke": "#4c9b8a"}
text(40, 424, "触觉分支（可训练）", 21, True, color="#007f73")
box(40, 465, 180, 80, ["小臂触觉", "阵列历史"], **green)
box(270, 465, 220, 80, ["信号预处理", "历史帧对齐 / 归一化"], **green)
box(550, 465, 200, 80, ["触觉编码器", "局部空间 / 时间信息"], **green)
box(830, 465, 210, 80, ["有效特征平均", "整体触觉特征"], **green)
box(1100, 465, 140, 80, ["残差头"], **green)
arrow("220,505 270,505")
arrow("490,505 550,505")
arrow("750,505 830,505")
arrow("1040,505 1100,505")
arrow("1170,465 1170,284")
text(1190, 386, "触觉修正", 18, color="#007f73")
parts.extend(["</g>", "</svg>"])
OUTPUT.write_text("\n".join(parts), encoding="utf-8")
print(OUTPUT)

# A separate dimension flow keeps the overview readable.
parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1900" height="940" viewBox="0 0 1900 940">',
    '<title>π0.5 Pressure VTLA tensor dimension flow</title>',
    '<desc>Default full-size model dimensions for observation prefix, action expert, and tactile flow residual.</desc>',
    '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8" fill="#475569"/></marker></defs>',
    '<rect width="1900" height="940" fill="white"/>',
    '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172b40">',
]
text(50, 52, "总体架构 · 维度流程", 30, True)
text(50, 87, "默认模型维度 · 小臂触觉残差 · 基座按需适配本体", 20, color="#52667a")

text(50, 125, "观测前缀", 22, True)
box(50, 145, 280, 80, ["RGB 图像（编码前）", "V 路 × [B,3,224,224]"])
box(390, 145, 250, 80, ["SigLIP + 视觉投影", "[B,256V,2048]"])
box(50, 270, 280, 80, ["语言 + 机器人状态", "state [B,D_s]"])
box(390, 270, 250, 80, ["离散化 / 文本嵌入", "[B,L] → [B,L,2048]"])
box(750, 205, 250, 80, ["拼接观测前缀", "[B,P,2048]"])
box(1100, 205, 300, 80, ["PaliGemma", "[B,P,2048]"])
box(1500, 205, 340, 80, ["每层 K / V 缓存", "各为 [B,1,P,256]"])
arrow("330,185 390,185")
arrow("330,310 390,310")
arrow("640,185 695,185 695,230 750,230")
arrow("640,310 695,310 695,260 750,260")
arrow("1000,245 1100,245")
arrow("1400,245 1500,245")

text(50, 413, "动作去噪", 22, True)
box(50, 455, 280, 80, ["带噪动作 x_t", "[B,H,32]"])
box(390, 455, 250, 80, ["动作输入投影", "[B,H,1024]"])
box(750, 455, 250, 80, ["ActionExpert", "[B,H,1024]"])
box(1100, 455, 300, 80, ["动作输出投影", "v_base [B,H,32]"])
box(1640, 445, 220, 100, ["Euler 积分 / 维度裁剪", "[B,H,32]", "→ [B,H,D_a]"])
arrow("330,495 390,495")
arrow("640,495 750,495")
arrow("1000,495 1100,495")
arrow("1400,495 1501,495")
parts.append('<circle cx="1530" cy="495" r="29" fill="white" stroke="#475569" stroke-width="2"/>')
text(1530, 504, "+", 30, anchor="middle")
arrow("1559,495 1640,495")
arrow("1670,285 1670,365 950,365 950,455")
text(1215, 350, "读取观测 KV", 18, anchor="middle", color="#52667a")
text(510, 413, "时间 t [B] → [B,1024]", 19)
arrow("740,410 800,410 800,455")

text(50, 638, "触觉残差", 22, True, color="#007f73")
box(50, 675, 280, 100, ["归一化触觉历史", "[B,S,K,R,C]"], **green)
box(390, 675, 250, 100, ["Patch MLP + 位置编码", "[B,N_t,2048]"], **green)
box(750, 675, 250, 100, ["特征平均 + 有效占比", "[B,2048] + [B,1]", "→ [B,2049]"], **green)
box(1100, 675, 300, 100, ["残差 MLP / reshape", "[B,2049] → [B,128]", "→ [B,H×32] → [B,H,32]"], **green)
arrow("330,725 390,725")
arrow("640,725 750,725")
arrow("1000,725 1100,725")
arrow("1400,725 1530,725 1530,524")
text(1550, 645, "Δv [B,H,32]", 19, color="#007f73")

text(50, 835, "P = 256V + L", 21)
text(390, 835, "N_t = S × K × (R / p_r) × (C / p_c)", 21)
text(50, 875, "B 批量 · V 相机输入路数 · L 文本长度 · H 动作步数（默认 50） · D_a 机器人动作维数 · D_s 状态维数", 20)
text(50, 915, "S 传感器数 · K 历史帧数 · R,C 阵列尺寸 · p_r,p_c 触觉 patch 尺寸。序列长度包含补齐位置。", 20)
parts.extend(["</g>", "</svg>"])
dimensions_output = OUTPUT.with_name("pi05-pressure-vtla-dimensions.svg")
dimensions_output.write_text("\n".join(parts), encoding="utf-8")
print(dimensions_output)
