#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "report" / "HW3_report.ipynb"


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str, output: str = "") -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    if output:
        outputs.append({"name": "stdout", "output_type": "stream", "text": output})
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs,
        "source": source.strip() + "\n",
    }


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def image(path: str, caption: str, width: int | None = None) -> str:
    if width is None:
        return f"![{caption}]({path})\n\n图：{caption}。"
    return f'<img src="{path}" alt="{caption}" width="{width}">\n\n图：{caption}。'


def info(path: str) -> dict[str, Any]:
    data = read_json(path)
    return {
        "episodes": data["total_episodes"],
        "frames": data["total_frames"],
        "features": sorted(data["features"].keys()),
    }


def final_loss(path: str) -> str:
    p = ROOT / path
    pat = re.compile(
        r"step:(\d+K?) .*?\bloss:([0-9.]+) grdn:([0-9.]+) lr:([0-9.eE+-]+) updt_s:([0-9.]+) data_s:([0-9.]+)"
    )
    ans = ""
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            ans = (
                f"step={m.group(1)}, loss={float(m.group(2)):.3f}, "
                f"grad_norm={float(m.group(3)):.3f}, lr={m.group(4)}, "
                f"update_s={float(m.group(5)):.3f}, data_s={float(m.group(6)):.3f}"
            )
    return ans or "未记录"


def final_online_row(path: str) -> dict[str, str]:
    rows = read_csv(path)
    return rows[-1] if rows else {}


def final_val_row(path: str) -> dict[str, str]:
    rows = [row for row in read_csv(path) if row.get("val_l1_loss")]
    return rows[-1] if rows else {}


def main() -> None:
    manifest = read_json("results/task1/real_run_manifest.json")
    stats = read_json("results/task1/asset_stats.json")
    ready = read_json("results/submission_readiness.json")

    a = manifest["object_a"]
    b = manifest["object_b"]
    c = manifest["object_c"]
    bg = manifest["background"]
    bg2 = bg.get("2dgs", {})

    b_train_info = info("data/task2/lerobot_v30_official/calvin_B_train80/meta/info.json")
    b_val_info = info("data/task2/lerobot_v30_official/calvin_B_val20/meta/info.json")
    abc_train_info = info("data/task2/lerobot_v30_official/calvin_ABC_train40each/meta/info.json")
    abc_val_info = info("data/task2/lerobot_v30_official/calvin_ABC_val10each/meta/info.json")
    d_info = info("data/task2/lerobot_v30_official/calvin_D_eval_100/meta/info.json")

    b_eval = read_json("results/task2/online_eval_B_on_D_100.json")
    abc_eval = read_json("results/task2/online_eval_ABC_on_D_100.json")
    action_l1_improvement = (
        (float(b_eval["mean_action_l1"]) - float(abc_eval["mean_action_l1"]))
        / float(b_eval["mean_action_l1"])
        * 100.0
    )
    mean_loss_improvement = (
        (float(b_eval["mean_loss"]) - float(abc_eval["mean_loss"]))
        / float(b_eval["mean_loss"])
        * 100.0
    )
    policy_cfg = read_json("results/task2/online_act_B_train80_val20_10k/checkpoints/010000/pretrained_model/config.json")
    b_train_last = final_online_row("results/task2/online_act_B_train80_val20_10k/online_metrics.csv")
    abc_train_last = final_online_row("results/task2/online_act_ABC_train40each_val10each_10k/online_metrics.csv")
    b_val_last = final_val_row("results/task2/online_act_B_train80_val20_10k/online_metrics.csv")
    abc_val_last = final_val_row("results/task2/online_act_ABC_train40each_val10each_10k/online_metrics.csv")

    created = datetime.now().strftime("%Y-%m-%d %H:%M")

    cells: list[dict[str, Any]] = [
        md(
            f"""
# 计算机视觉 HW3 实验报告

**姓名 / 学号：** 谢唯 / 23307130044  
**成员分工：** 谢唯（个人完成）
"""
        ),
        md(
            """
## 1. 任务背景

本次作业包含两个互相独立但都强调“完整链路”的实验。

**题目一**关注 3D 视觉资产生成与融合。它要求把真实世界重建、文本生成 3D、单图生成 3D 和公开场景重建组合起来，最终在同一个三维场景中完成可视化渲染。这个任务的难点不是单独得到一个模型，而是让不同来源、不同表示形式、不同质量特征的资产能够在一个统一视角下合理共存。

**题目二**关注具身智能中的动作策略学习。机器人策略如果只在一个视觉环境中训练，很容易把背景、光照、桌面纹理等无关因素和动作绑定。CALVIN 提供 A/B/C/D 多个环境，正好可以测试 ACT 这类视觉-动作策略能否从已见环境泛化到未见环境。
"""
        ),
        md(
            """
## 2. 题目一：2DGS 与 AIGC 的多源资产生成与场景融合

### 2.1 数据与任务设定

题目一使用四类输入数据：

1. 真实手机视频，用于重建物体 A。
2. 文本 prompt，用于生成物体 B。
3. 单张真实物体照片，用于生成物体 C。
4. Mip-NeRF 360 的 garden 场景，用于重建统一背景。

最终目标是把 A/B/C 三个物体放入 garden 背景中，生成融合预览图和绕中心旋转的多视角视频。
"""
        ),
        md(
            table(
                ["资产", "输入数据", "采用方法", "最终表达", "训练/优化规模"],
                [
                    ["物体 A", "手机环绕视频", "前景分割 + COLMAP + 2DGS + TSDF", "PLY mesh", f"{a['frame_extraction']['extracted_frames']} 帧，2DGS {a['2dgs']['iterations']} steps"],
                    ["物体 B", "文本 prompt", "threestudio DreamFusion with SDS Loss", "OBJ mesh", f"SDS {b['threestudio']['iterations']} steps"],
                    ["物体 C", "单张去背景照片", "Magic123 with Zero123 guidance and DMTet refinement", "OBJ/MTL mesh", f"{c['magic123']['coarse_iterations']} + {c['magic123']['fine_iterations']} steps"],
                    ["背景", "Mip-NeRF 360 garden", "COLMAP 场景数据 + 2DGS", "cleaned/upright PLY mesh", f"2DGS {bg2.get('iterations')} steps"],
                ],
            )
        ),
        md(
            """
### 2.2 物体 A：真实多视角重建流程

物体 A 的流程分为五步：

1. 从手机视频抽取 80 帧，控制图像尺寸，避免显存过高。
2. 对每帧做前景分割，减少白背景、桌面和杂物进入重建。
3. 根据 mask 裁剪并生成 RGBA 输入，让物体居中且背景透明。
4. 用 COLMAP 做 exhaustive matching 和相机位姿估计。
5. 用 2DGS 训练 20000 steps，导出 TSDF mesh，再保留最大连通域去除碎片。

COLMAP 的核心作用是从无位姿的多视角图像中恢复相机参数和稀疏三维结构。具体而言，COLMAP 先在不同帧之间提取和匹配局部特征点，再通过增量式 Structure-from-Motion 估计每张图像的相机内参、外参和稀疏点云。对于 2DGS，COLMAP 输出的相机位姿非常关键，因为后续可微渲染需要知道每张训练图像对应的观察方向和投影关系。

2DGS 使用大量二维高斯面片作为场景或物体的显式表示。每个高斯面片包含空间位置、尺度、方向、不透明度和颜色等可优化参数。训练时，算法根据 COLMAP 相机位姿从当前高斯集合渲染出对应视角图像，并最小化渲染图像与真实训练图像之间的颜色误差和几何正则项。经过多轮优化后，高斯面片会逐渐贴合物体表面并学习外观颜色。由于最终融合需要显式 mesh，本实验在 2DGS 训练完成后使用 TSDF fusion 将多视角几何结果融合为三角网格，再通过连通域筛选去除浮动碎片。

COLMAP 最终注册 80/80 帧，说明相机位姿估计稳定。A 的主要质量瓶颈来自手机视频本身：如果物体表面纹理少、反光强或转动角度不均匀，局部几何会有缺损。
"""
        ),
        md(image("figures/task1_object_A_raw_frames.png", "物体 A 原始手机视频抽帧")),
        md(image("figures/task1_object_A_masked_cropped.png", "物体 A 前景分割、裁剪和透明背景训练输入")),
        md(image("figures/task1_object_A_final_surface_mesh.png", "物体 A final cleaned mesh 的彩色面片表面预览")),
        md(
            table(
                ["指标", "数值"],
                [
                    ["抽帧数量", a["frame_extraction"]["extracted_frames"]],
                    ["COLMAP 注册帧数", f"{a['colmap']['registered_images']} / {a['frame_extraction']['extracted_frames']}"],
                    ["COLMAP sparse points", a["colmap"]["points3D"]],
                    ["2DGS 训练步数", a["2dgs"]["iterations"]],
                    ["mesh 顶点数", f"{a['2dgs']['mesh_vertices']:,}"],
                    ["mesh 面数", f"{a['2dgs']['mesh_faces']:,}"],
                ],
            )
        ),
        md(
            f"""
### 2.3 物体 B：文本到 3D 生成流程

物体 B 的最终文本提示词为：

> {b['prompt']}

早期提示词曾过于宽泛，导致生成形状不稳定。最终提示词明确强调“小车、四个分离的黑色轮子、圆润塑料车身、低矮紧凑、侧面轮廓对称、单个居中物体”。这样做是为了降低 SDS 生成时把轮子融合、少生成轮子或生成多余结构的概率。

训练采用 threestudio 框架中的 DreamFusion 文本到 3D 流程，并以 SDS Loss 作为核心优化目标，Stable Diffusion v1.5 fp16 作为 guidance model。SDS 的核心机制是从随机视角渲染当前 3D 表示，再利用预训练 2D 扩散模型判断渲染图像与文本 prompt 的一致性，并将扩散模型提供的梯度反传到 3D 表示中。训练从 8000、12000、16000 逐步续训到 20000 steps，并通过中间预览图检查轮子和车身形状。最终采用 20000-step 导出 mesh。
"""
        ),
        md(image("figures/task1_object_B_training_progress.png", "物体 B SDS 训练过程预览：8k、12k、16k、20k")),
        md(image("figures/task1_object_B_final.png", "物体 B 20000-step 最终导出 mesh：纹理结果与纯几何结果")),
        md(
            table(
                ["项目", "设置或结果"],
                [
                    ["Network / Representation", "threestudio DreamFusion implicit 3D representation, exported to mesh"],
                    ["Guidance Model", b["threestudio"]["guidance"]],
                    ["Prompt", b["prompt"]],
                    ["Optimizer Objective", "SDS Loss"],
                    ["Training Steps", b["threestudio"]["iterations"]],
                    ["mesh 顶点数", f"{b['mesh_vertices']:,}"],
                    ["mesh 面数", f"{b['mesh_faces']:,}"],
                ],
            )
        ),
        md(
            """
现象分析：文本到 3D 的优势是输入成本最低，只需要一句 prompt；但几何最不可靠。小车的整体语义和车身形状能够形成，但轮子这类局部结构很容易受扩散先验影响而不完全对称。继续训练能改善形体稳定性，但不能保证达到真实扫描级别。
"""
        ),
        md(
            """
### 2.4 物体 C：单图到 3D 生成流程

物体 C 使用单张真实图片。首先进行去背景，得到纯净 RGBA 前景，再输入 Magic123。训练分为 coarse NeRF 和 DMTet refinement 两个阶段。

**coarse NeRF** 阶段采用隐式神经辐射场表示物体的粗略三维结构和外观分布。该阶段通过单图约束与新视角先验优化一个可微渲染的体表示，使模型能够在多个视角下生成一致的物体外观。其作用是为后续显式网格优化提供稳定的初始几何、姿态和颜色分布。

**DMTet refinement** 阶段采用可变形四面体网格表示物体表面，并在 coarse NeRF 得到的粗几何基础上继续优化网格顶点、表面边界和外观细节。该阶段的目标是将隐式体表示转换为显式三角网格，最终导出可在 Blender 中加载和融合渲染的 OBJ/MTL 模型。
"""
        ),
        md(image("figures/task1_object_C_rgba_input.png", "物体 C 去背景后的 RGBA 单图输入")),
        md(image("figures/task1_object_C_coarse_render.png", "物体 C Magic123 coarse 阶段渲染结果")),
        md(image("figures/task1_object_C_dmtet_render.png", "物体 C DMTet refinement 阶段渲染结果")),
        md(image("figures/task1_object_C_final_mesh.png", "物体 C 最终 mesh 的彩色面片表面预览")),
        md(
            table(
                ["项目", "设置或结果"],
                [
                    ["Input", "单张真实物体照片，经过去背景处理"],
                    ["Coarse Stage", f"Zero123 guidance, {c['magic123']['coarse_iterations']} steps"],
                    ["Refinement Stage", f"DMTet, {c['magic123']['fine_iterations']} steps"],
                    ["Loss / Guidance", "single-image reconstruction guidance + Zero123 novel-view guidance"],
                    ["mesh 顶点数", f"{c['mesh_vertices']:,}"],
                    ["mesh 面数", f"{c['mesh_faces']:,}"],
                ],
            )
        ),
        md(
            """
现象分析：单图生成比纯文本生成更能保留输入物体的外观，因为正面图像提供了强约束；但它仍然无法真正看到背面，因此背面形状、厚度和遮挡区域依赖模型先验。最终结果适合作为单图生成展示，不应解释为真实扫描级 mesh。
"""
        ),
        md(
            """
### 2.5 背景重建与融合渲染

背景选择 Mip-NeRF 360 的 garden 场景。该场景包含桌面、花瓶、植物和室外背景，适合作为统一放置环境。背景同样使用 2DGS 训练 30000 steps，通过多视角图像和相机位姿优化场景的二维高斯面片表示。报告中展示的是 2DGS 的多视角 RGB 片渲染质量检查和导出的 garden mesh 检查；最终融合渲染采用 cleaned/upright garden mesh 作为可见背景场景。

最终融合采用以下策略：背景使用由 30000-step 2DGS garden 导出的 cleaned/upright mesh，A/B/C 也作为 mesh 导入 Blender。这样四个三维资产都在同一个 Blender mesh 场景中完成尺度对齐、桌面摆放和绕中心旋转渲染。
"""
        ),
        md(image("figures/task1_background_pipeline.png", "garden 背景 2DGS 片渲染与导出 mesh 检查")),
        md(image("../results/task1/fusion_mesh/fusion_mesh_preview.png", "最终 mesh 融合渲染图", width=520)),
        md(image("figures/task1_fusion_walkthrough_grid.png", "从当前最终 fusion_walkthrough.mp4 重新抽取的漫游视频关键帧")),
        md(
            """
### 2.6 题目一实验设置与指标汇总
"""
        ),
        md(
            table(
                ["模块", "Network / Method", "Batch Size", "Learning Rate", "Optimizer", "Epochs / Steps", "Loss / Objective", "关键指标"],
                [
                    ["A 多视角重建", "COLMAP + 2DGS + TSDF", "N/A", "2DGS 默认设置", "2DGS 默认优化器", f"{a['2dgs']['iterations']} steps", "photometric rendering loss + regularization", f"{a['colmap']['registered_images']}/80 registered, {a['2dgs']['mesh_vertices']:,} vertices"],
                    ["B 文本生成", "threestudio DreamFusion", "threestudio 默认", "threestudio 默认", "threestudio 默认", f"{b['threestudio']['iterations']} steps", "SDS Loss", f"{b['mesh_vertices']:,} vertices, {b['mesh_faces']:,} faces"],
                    ["C 单图生成", "Magic123 two-stage pipeline", "默认", "默认", "默认", f"{c['magic123']['coarse_iterations']} + {c['magic123']['fine_iterations']} steps", "single-image reconstruction guidance + Zero123 novel-view guidance", f"{c['mesh_vertices']:,} vertices, {c['mesh_faces']:,} faces"],
                    ["背景", "Mip-NeRF 360 garden + 2DGS", "N/A", "2DGS 默认设置", "2DGS 默认优化器", f"{bg2.get('iterations')} steps", "photometric rendering loss", f"{stats['background']['vertices']:,} vertices"],
                ],
            )
        ),
        md(
            table(
                ["资产", "格式", "顶点数", "面数"],
                [
                    ["背景 garden", stats["background"]["format"], f"{stats['background']['vertices']:,}", f"{stats['background']['faces']:,}"],
                    ["物体 A", stats["object_A"]["format"], f"{stats['object_A']['vertices']:,}", f"{stats['object_A']['faces']:,}"],
                    ["物体 B", stats["object_B"]["format"], f"{stats['object_B']['vertices']:,}", f"{stats['object_B']['faces']:,}"],
                    ["物体 C", stats["object_C"]["format"], f"{stats['object_C']['vertices']:,}", f"{stats['object_C']['faces']:,}"],
                ],
            )
        ),
        md(image("figures/task1_asset_mesh_stats.png", "四类 3D 资产 mesh 规模对比", width=500)),
        md(
            f"""
### 2.7 三种 3D 资产生成方式的结果对比

#### 2.7.1 几何准确度

几何准确度主要看三类信息：是否有真实观测约束、约束覆盖多少视角、最终导出的 mesh 是否稳定。三种方法的可量化结果如下：

| 方法 | 真实观测输入 | 位姿 / 视角约束 | 最终 mesh 规模 | 几何证据强度 |
| --- | ---: | ---: | ---: | --- |
| A 多视角重建 | 80 帧手机视频 | COLMAP 注册 {a['colmap']['registered_images']}/80 帧，注册率 100%，sparse points {a['colmap']['points3D']} | {a['2dgs']['mesh_vertices']:,} vertices / {a['2dgs']['mesh_faces']:,} faces | 最高：几何由真实多视角观测和相机位姿直接约束 |
| B 文本生成 | 0 张真实图像 | 无 COLMAP / 无真实视角位姿，仅文本语义约束 | {b['mesh_vertices']:,} vertices / {b['mesh_faces']:,} faces | 最弱：形状由扩散模型语义先验间接约束 |
| C 单图生成 | 1 张去背景照片 | 正面由单图约束，其余视角由 Zero123 guidance 补全 | {c['mesh_vertices']:,} vertices / {c['mesh_faces']:,} faces | 中等：正面有真实图像约束，背面和侧面依赖生成先验 |

从定量结果看，A 的优势不是 mesh 顶点数最多，而是几何监督最充分：80 帧全部被 COLMAP 注册，说明环绕视频中的相机位姿链是连续可用的，2DGS 的训练也有真实多视角重投影误差约束。因此 A 在前后关系、尺度关系和主要轮廓上最接近实物。B 虽然导出了 {b['mesh_vertices']:,} 个顶点和 {b['mesh_faces']:,} 个面，但这些面并不代表真实几何精度更高；它们来自 SDS 对文本“小车”的语义优化，缺少实拍视角闭环约束，所以轮子数量、左右对称性和局部结构容易不稳定。C 的 mesh 规模最大，达到 {c['mesh_vertices']:,} vertices / {c['mesh_faces']:,} faces，但其几何可靠性呈现明显视角差异：输入照片可见的正面轮廓最可靠，不可见的背面、厚度和侧面由 Magic123 的新视角先验补全，因此整体准确度介于 A 和 B 之间。

#### 2.7.2 纹理细节

纹理细节主要看颜色来源和材质表达是否来自真实图像。三种方法的纹理证据如下：

| 方法 | 颜色 / 纹理来源 | 可量化记录 | 纹理可靠区域 | 主要限制 |
| --- | --- | ---: | --- | --- |
| A 多视角重建 | 80 帧真实视频，经前景分割后训练 2DGS，并在 TSDF mesh 中保留 vertex colors | 80 帧 RGBA cropped frames；最终 PLY {a['2dgs']['mesh_vertices']:,} vertices | 多视角都被视频覆盖的可见表面 | 视频模糊、分割边界、弱纹理区域和 TSDF 导出会造成局部颜色破碎 |
| B 文本生成 | Stable Diffusion v1.5 fp16 在 SDS 中根据 prompt 生成外观 | OBJ 中记录 {stats['object_B']['texture_vertices']:,} texture vertices；无实拍纹理输入 | 与 prompt 强相关的整体颜色和语义部件 | 纹理不是实物观测，车轮、车窗和车身边缘可能与几何不严格对齐 |
| C 单图生成 | 单张去背景照片提供正面颜色，Magic123 补全其他视角 | 1 张 RGBA 输入；最终 OBJ/MTL {c['mesh_vertices']:,} vertices / {c['mesh_faces']:,} faces | 输入照片可见的正面区域 | 背面和侧面纹理依赖模型补全，当前 MTL 为材质色而非完整照片级 texture map |

因此，A 的纹理真实性最高，因为颜色直接来自 80 帧真实视频，而不是由文本或单图先验生成；但最终报告中展示的 A mesh 仍能看到局部模糊和碎裂，原因是视频前景较小、白底/弱纹理区域对 SfM 与 TSDF 表面融合都不利。B 的纹理自由度最高，不需要任何图像输入即可生成“小车”的颜色和语义外观，但它的颜色和几何是由 SDS 间接共同优化出来的，不能保证每个轮子、车窗边界都与真实物理结构一致。C 的纹理保真度集中在正面：输入照片给出了真实颜色和轮廓，因此正面比 B 更像具体物体；但未观测视角仍由 Zero123 guidance 和 DMTet refinement 推断，完整 360 度纹理一致性弱于 A。

#### 2.7.3 计算耗时与流程成本

本实验没有保存稳定可复现的 wall-clock 训练耗时日志，因此不直接报告秒数或小时数。为了仍然给出定量分析，这里用近似浮点运算量（FLOPs）做量级估算。估算只用于比较三种路线的相对计算成本，假设如下：A 的 2DGS 单步渲染和反传记为轻量级 differentiable splatting；B 的 DreamFusion 每步包含一次随机视角渲染和 Stable Diffusion v1.5 SDS guidance，扩散 U-Net 的前向/反向是主计算项；C 的 Magic123 每步包含 NeRF/DMTet 渲染和 Zero123 guidance，其中 coarse 阶段分辨率较低，DMTet refinement 阶段分辨率更高。按常见实现量级，本文采用保守估计：2DGS 单步约 1-5 GFLOPs，SDS/Zero123 guidance 单步约 100-300 GFLOPs。该估计不等同于硬件实测时间，但能反映扩散模型 guidance 相比纯 2DGS 优化的计算量差异。

| 方法 | 主要计算阶段 | 实际规模 | FLOPs 估算公式 | 估算量级 | 计算成本判断 |
| --- | --- | ---: | --- | ---: | --- |
| A 多视角重建 | COLMAP exhaustive matching + 2DGS + TSDF | 80 帧；COLMAP 图像对约 80×79/2 = 3160；2DGS {a['2dgs']['iterations']} steps | 2DGS: {a['2dgs']['iterations']} × 1-5 GFLOPs，另加 COLMAP 匹配/SfM | 约 20-100 TFLOPs + SfM 前处理 | 单步轻，但前处理链路最长 |
| B 文本生成 | DreamFusion with SDS Loss | 1 条 prompt；SDS {b['threestudio']['iterations']} steps | {b['threestudio']['iterations']} × 100-300 GFLOPs | 约 2-6 PFLOPs | 扩散 guidance 主导，训练计算量最高 |
| C 单图生成 | Magic123 coarse NeRF + DMTet refinement | 1 张 RGBA；coarse {c['magic123']['coarse_iterations']} steps + DMTet {c['magic123']['fine_iterations']} steps | ({c['magic123']['coarse_iterations']} + {c['magic123']['fine_iterations']}) × 100-300 GFLOPs | 约 1-3 PFLOPs | 总步数少于 B，但每步仍调用 Zero123 guidance |

从 FLOPs 量级看，B 的训练计算量最大。虽然 A 和 B 都训练到 20000 steps，但 A 的每一步主要是高斯 splatting 的可微渲染和光度损失反传，而 B 的每一步都要通过 Stable Diffusion guidance 计算 SDS 梯度，单步计算量通常比 2DGS 高一到两个数量级，因此 B 的总量级达到 PFLOPs。C 的优化步数为 10000 steps，约为 B 的一半，但它的每一步也依赖 Zero123 guidance，并且包含 coarse NeRF 与 DMTet refinement 两个三维优化阶段，所以估算量级仍在 PFLOPs 级别。A 的纯训练 FLOPs 最低，但它有 80 帧抽帧、逐帧前景分割、3160 对 exhaustive matching、SfM 和 TSDF fusion 等不可忽略的前处理/后处理成本；因此如果只看 GPU 训练，A 低于 B/C，如果看端到端人工和工程流程，A 的流程成本最高。

### 2.8 表达形式统一与场景融合实现

本任务的融合难点在于 A/B/C 和背景的原始表示不一致。物体 A 与 garden 背景首先由 2DGS 表示，属于显式高斯面片；物体 B 由 threestudio DreamFusion 生成，训练阶段依赖隐式 3D 表示和 SDS Loss，最终可导出 OBJ；物体 C 由 Magic123 生成，coarse NeRF 阶段是隐式体表示，DMTet refinement 后导出 OBJ/MTL。理论上可以把 B/C 的 mesh 再采样成点云或高斯，再与 A 和背景的 2DGS 高斯做代码级拼接，但这样需要重新估计每个采样点的尺度、方向、不透明度和颜色，且不同资产的坐标系、尺度和材质定义都不一致，转换误差会直接影响融合效果。因此本实验采用“全部转为显式三角 mesh，再在 Blender 中统一渲染”的路线。

具体实现上，四类资产都被统一成可导入 Blender 的多面体 mesh。物体 A 在 2DGS 训练后通过 TSDF fusion 导出 PLY mesh，并保留 vertex colors，最终规模为 {a['2dgs']['mesh_vertices']:,} vertices / {a['2dgs']['mesh_faces']:,} faces；物体 B 从 threestudio DreamFusion 结果导出 OBJ mesh，规模为 {b['mesh_vertices']:,} vertices / {b['mesh_faces']:,} faces；物体 C 从 Magic123 DMTet refinement 导出 OBJ/MTL mesh，规模为 {c['mesh_vertices']:,} vertices / {c['mesh_faces']:,} faces。背景 garden 从 30000-step 2DGS 场景中执行 unbounded mesh extraction，先得到 {stats['background']['vertices']:,} vertices / {stats['background']['faces']:,} faces 的非代理大场景 mesh，再进行 upright 对齐、桌面区域定位、主体裁剪和顶部遮挡区域清理，最终使用 `results/task1/fusion_mesh/garden_scene_upright_topopen_q05.ply` 作为可见背景 mesh。

在 Blender 中，融合脚本先导入 cleaned/upright garden mesh，并使用 vertex colors 保留 garden 的颜色信息；A/B/C 三个物体 mesh 再分别导入到同一个世界坐标系中。由于四个 mesh 来自不同算法，原始尺度和朝向并不一致，所以融合脚本对每个物体执行几何归一化：根据 bounding box 高度缩放到目标尺寸，根据手工检查后的旋转角修正前后/上下方向，再把物体底部对齐到 garden 桌面高度。最终摆放时，A、B、C 被放置在 garden 桌面区域的三个位置，使用统一相机、光照和材质系统渲染；漫游视频通过相机绕桌面中心旋转获得。因此最终看到的是“garden 背景 mesh + 三个物体 mesh”共同组成的三维场景，物体之间的相对大小、空间位置和相机运动都在 Blender 的同一 3D 坐标系统中完成。
"""
        ),
        md(
            """
## 3. 题目二：LeRobot ACT 跨环境泛化

### 3.1 任务背景与实验目标

题目二研究的是具身智能策略在视觉环境变化下的泛化能力。CALVIN 中的 A/B/C/D 表示不同视觉环境：桌面颜色、背景纹理、光照、相机观测和物体布局会变化，但机器人动作空间和任务类型保持一致。因此，本题不是测试模型能否记住一个训练集，而是测试视觉-动作策略能否在未见过的 D 环境中保持动作预测能力。

本实验的核心对比是 B-only 与 ABC-mixed 两个 ACT 策略。B-only 只使用环境 B 数据训练，代表单环境行为克隆基线；ABC-mixed 使用 A/B/C 混合数据训练，代表多环境视觉分布增强后的策略。两者使用完全相同的 ACT 网络结构、优化器、学习率、batch size、训练步数和输入/输出字段。最终在环境 D 上做 zero-shot 评估，不使用 D 环境数据进行训练或微调。报告评价指标为 Action L1 和 mean loss，用于比较两个策略在未见视觉环境中的动作预测偏差。
"""
        ),
        md(
            """
### 3.2 数据集构建与可视化

本实验使用助教划分好的 HuggingFace 数据集 `xiaoma26/calvin-lerobot`，其中 `splitA/splitB/splitC/splitD` 分别对应 CALVIN 环境 A/B/C/D。官方原始 parquet 中的字段为 `image`、`wrist_image`、`state` 和 `actions`；转换时统一映射为 LeRobot ACT 训练需要的 `observation.images.top`、`observation.images.wrist`、`observation.state` 和 `action`。每一帧输入包含两个相机视角和机器人低维状态：`observation.images.top` 为 200×200 的顶部相机 RGB 图像，`observation.images.wrist` 为 84×84 的腕部相机 RGB 图像，`observation.state` 为 15 维机器人状态；监督信号 `action` 为 7 维连续动作。为满足训练过程验证要求，本实验重新划分 train/val：B-only 使用 B 环境 80 条 episode 训练、20 条 episode 验证；ABC-mixed 使用 A/B/C 各 40 条 episode 训练、各 10 条 episode 验证。D 环境 100 条 episode 只用于最终 zero-shot 测试，不参与训练或在线验证。
"""
        ),
        md(
            table(
                ["数据集", "用途", "Episodes", "Frames", "FPS", "Observation / Action"],
                [
                    ["Environment B train", "B-only 训练", b_train_info["episodes"], b_train_info["frames"], "10", "top RGB + wrist RGB + 15-d state -> 7-d action"],
                    ["Environment B val", "B-only 在线验证", b_val_info["episodes"], b_val_info["frames"], "10", "同上"],
                    ["Environment A/B/C train", "ABC-mixed 训练", abc_train_info["episodes"], abc_train_info["frames"], "10", "top RGB + wrist RGB + 15-d state -> 7-d action"],
                    ["Environment A/B/C val", "ABC-mixed 在线验证", abc_val_info["episodes"], abc_val_info["frames"], "10", "同上"],
                    ["Environment D", "zero-shot 评估", d_info["episodes"], d_info["frames"], "10", "top RGB + wrist RGB + 15-d state -> 7-d action"],
                ],
            )
        ),
        md(image("figures/task2_official_dataset_sizes.png", "Task 2 官方 split 训练与评估数据规模对比", width=520)),
        md(image("figures/task2_official_split_samples.png", "CALVIN A/B/C/D 官方环境的 top/wrist 相机样例", width=560)),
        md(
            """
图中可以看到，不同数据 split 的输入都是 top/wrist 双视角，但视觉分布并不完全一致。B-only 只暴露给模型一种训练环境；ABC-mixed 把多个环境混合到同一个训练集，使视觉编码器在训练阶段见到更丰富的背景和物体外观；D 环境只在评估时出现，用来检验 zero-shot 跨环境泛化。

### 3.3 ACT 方法原理与本实验实现

ACT 的全称是 Action Chunking Transformer。普通行为克隆策略通常在每个时间步预测单步动作，推理时上一时刻的小误差会进入下一时刻观测，容易造成动作抖动和误差累积。ACT 改为一次预测未来一段动作序列，即 action chunk。本实验中 `chunk_size = n_action_steps = 100`，表示模型根据当前观测一次输出 100 步动作候选序列。

本实验的 ACT 策略由 ResNet18 视觉编码器和 Transformer 动作解码器组成。top/wrist 两路 RGB 图像先通过 ImageNet 预训练 ResNet18 提取视觉特征，15 维机器人状态作为低维状态输入，Transformer 在这些条件下预测 7 维动作的 100-step chunk。模型还使用 ACT 中的 VAE latent 分支，latent dimension 为 32，KL loss 权重为 10。训练目标包含动作 L1 误差和 ACT 辅助损失；评估时用离线 D 数据计算预测动作和数据集中真实动作之间的 Action L1。
"""
        ),
        md(image("figures/task2_act_pipeline.png", "ACT 策略输入、特征融合与 100-step action chunk 输出流程")),
        md(
            """

Action chunking 对跨环境泛化有双重影响。一方面，chunk 内联合预测可以减少逐步控制的高频抖动，使短时间动作轨迹更平滑；另一方面，chunk 的条件仍然来自当前视觉观测，如果视觉编码器把 D 环境中的背景、光照或物体外观编码错，错误会影响整个 100-step chunk。因此本题的关键不是只看训练 loss，而是比较两个模型在未见过 D 环境上的动作误差。
"""
        ),
        md(
            """
### 3.4 题目二超参数与公平性设置
"""
        ),
        md(
            table(
                ["Setting", "B-only ACT", "ABC-mixed ACT"],
                [
                    ["Network Architecture", "ResNet18 visual encoder + ACT Transformer", "ResNet18 visual encoder + ACT Transformer"],
                    ["Input Observations", "top RGB, wrist RGB, robot state", "top RGB, wrist RGB, robot state"],
                    ["Image Shapes", "top 3×200×200, wrist 3×84×84", "top 3×200×200, wrist 3×84×84"],
                    ["State / Action Dim", "state 15, action 7", "state 15, action 7"],
                    ["Action Chunk Size", policy_cfg["chunk_size"], policy_cfg["chunk_size"]],
                    ["n_action_steps", policy_cfg["n_action_steps"], policy_cfg["n_action_steps"]],
                    ["Transformer", f"{policy_cfg['n_encoder_layers']} encoder layers, {policy_cfg['n_decoder_layers']} decoder layer, dim {policy_cfg['dim_model']}, {policy_cfg['n_heads']} heads", f"{policy_cfg['n_encoder_layers']} encoder layers, {policy_cfg['n_decoder_layers']} decoder layer, dim {policy_cfg['dim_model']}, {policy_cfg['n_heads']} heads"],
                    ["VAE Latent", f"use_vae={policy_cfg['use_vae']}, latent_dim={policy_cfg['latent_dim']}, KL weight={policy_cfg['kl_weight']}", f"use_vae={policy_cfg['use_vae']}, latent_dim={policy_cfg['latent_dim']}, KL weight={policy_cfg['kl_weight']}"],
                    ["Batch Size", "16", "16"],
                    ["Learning Rate", "1e-5", "1e-5"],
                    ["Optimizer", "AdamW, betas=(0.9, 0.999), grad clip=10", "AdamW, betas=(0.9, 0.999), grad clip=10"],
                    ["Weight Decay", "1e-4", "1e-4"],
                    ["Training Steps", "10000", "10000"],
                    ["Save Frequency", "5000 steps", "5000 steps"],
                    ["Loss Function", "Action L1 + ACT auxiliary losses", "Action L1 + ACT auxiliary losses"],
                    ["Evaluation Metric", "offline Action L1 / mean loss on D", "offline Action L1 / mean loss on D"],
                ],
            )
        ),
        md(
            """
这组设置保证了题目要求中的公平比较：两个模型的网络结构和超参数完全一致，唯一主要变量是训练环境分布。B-only 只包含环境 B，ABC-mixed 包含 A/B/C 多环境数据。训练时两组都使用 held-out validation split 做在线评估，D 环境严格保留到最终 zero-shot 测试阶段；因此训练 loss、validation Action L1 和 D 环境 Action L1 分别对应训练拟合、同源未见 episode 泛化和跨环境泛化三个层面的证据。

### 3.5 训练收敛结果

两个模型都训练到 10000 steps，并每 100 steps 记录一次训练 loss。训练脚本每 1000 steps 在独立 validation split 上完整评估一次：B-only 使用 20 条 B-val episode，ABC-mixed 使用 A/B/C 各 10 条 val episode。在线验证只用于观察收敛和同源泛化，不使用 D 环境数据。训练曲线显示，两者从初始高 loss 快速下降，在中后期进入缓慢收敛阶段；验证 Action L1 的下降幅度小于训练 loss，说明模型对训练数据的拟合速度快于对 held-out episode 的泛化提升。
"""
        ),
        md(
            table(
                ["模型", "训练数据", "训练规模", "训练步数", "最终训练日志"],
                [
                    ["B-only ACT", "B train / B val", f"train {b_train_info['episodes']} episodes / {b_train_info['frames']} frames; val {b_val_info['episodes']} episodes / {b_val_info['frames']} frames", "10000", f"train_loss={float(b_train_last['train_loss']):.4f}, train_l1={float(b_train_last['train_l1_loss']):.4f}, val_l1={float(b_val_last['val_l1_loss']):.4f}"],
                    ["ABC-mixed ACT", "A/B/C train / A/B/C val", f"train {abc_train_info['episodes']} episodes / {abc_train_info['frames']} frames; val {abc_val_info['episodes']} episodes / {abc_val_info['frames']} frames", "10000", f"train_loss={float(abc_train_last['train_loss']):.4f}, train_l1={float(abc_train_last['train_l1_loss']):.4f}, val_l1={float(abc_val_last['val_l1_loss']):.4f}"],
                ],
            )
        ),
        md(image("figures/task2_official_training_loss.png", "ACT 训练 loss 曲线")),
        md(image("figures/task2_official_validation_metrics.png", "ACT held-out validation 曲线")),
        md(
            """
### 3.6 Zero-shot D 环境评估结果

评估阶段使用完全未见过的 D 环境数据，不进行任何 D 环境微调。测试脚本加载 10000-step checkpoint，在 D 评估集上按 batch 计算预测动作和真实动作的 L1 距离。Action L1 越低，说明策略输出越接近 CALVIN 数据中的专家动作。
"""
        ),
        md(
            table(
                ["模型", "训练环境", "测试环境", "Action L1 ↓", "Mean Loss ↓", "评估 batch"],
                [
                    ["B-only ACT", "B", "D", f"{float(b_eval['mean_action_l1']):.6f}", f"{float(b_eval['mean_loss']):.6f}", b_eval["num_batches"]],
                    ["ABC-mixed ACT", "A/B/C", "D", f"{float(abc_eval['mean_action_l1']):.6f}", f"{float(abc_eval['mean_loss']):.6f}", abc_eval["num_batches"]],
                ],
            )
        ),
        md(image("figures/task2_official_eval_metrics.png", "D 环境 zero-shot Action L1 与 mean loss 对比", width=480)),
        md(image("figures/task2_official_action_stats.png", "B、ABC 与 D 数据的 7 维动作分布统计")),
        md(
            f"""
ABC-mixed 在 D 环境上的 Action L1 为 {float(abc_eval['mean_action_l1']):.4f}，低于 B-only 的 {float(b_eval['mean_action_l1']):.4f}，相对下降 {action_l1_improvement:.1f}%。Mean loss 从 {float(b_eval['mean_loss']):.4f} 降到 {float(abc_eval['mean_loss']):.4f}，相对下降 {mean_loss_improvement:.1f}%。由于两者的架构和训练超参数一致，这个差异主要来自训练环境分布：A/B/C 多环境训练让视觉编码器见到更多背景、桌面和光照变化，因此在 D 环境视觉偏移下动作预测误差更低。

### 3.7 Action Chunking 在视觉分布偏移下的现象分析

ACT 的 action chunking 提供的是时间维度上的稳定性，而不是自动的视觉域自适应。模型每次根据当前 top/wrist 图像和机器人状态预测 100 步动作序列，因此如果当前视觉特征可靠，chunk 内联合预测能减少单步行为克隆的抖动，使短期动作更平滑；但如果视觉编码器在 D 环境中把背景纹理、光照或物体位置编码错，错误会被带入整个 100-step chunk，表现为一段动作整体偏移。

B-only 的问题在于训练时只见过环境 B，视觉编码器可能把 B 的背景、桌面纹理或光照当作动作预测的隐含线索。D 环境改变后，这些线索不再稳定，因此 B-only 的 Action L1 更高。ABC-mixed 见过 A/B/C 多种视觉分布，训练过程迫使模型降低对单一背景的依赖，更倾向于使用机器人状态、可操作物体和任务相关视觉区域来预测动作。D 环境上 Action L1 下降 {action_l1_improvement:.1f}% 说明多环境训练缓解了 chunk-level 视觉偏移，但没有完全消除误差。

结合训练曲线和 D 环境指标可以看到，两种策略的训练 loss 都已经进入低值平台区，说明性能差异不是由某一组训练没有收敛造成的。B-only 的训练环境单一，视觉编码器更容易学习到环境 B 中稳定但非本质的背景线索；这些线索在 D 环境中发生改变后，会使 Transformer 解码出的整段 action chunk 产生系统性偏移。ABC-mixed 在训练时反复看到 A/B/C 的不同背景和视角外观，相同动作必须在不同视觉外观下被预测出来，因此模型更容易压低背景纹理的权重，并把注意力转向机器人状态、末端执行器附近区域和可操作物体。D 环境上的 Action L1 和 mean loss 同时下降，说明多环境训练不只是改善单个动作维度，而是让 100-step chunk 的整体动作序列更接近专家轨迹。
"""
        ),
        md(
            """
## 4. 总结

题目一完成了真实视频重建、文本到 3D、单图到 3D、公开场景 2DGS 重建和融合渲染。多视角重建在几何真实性上最好，文本生成最灵活但几何最不稳定，单图生成介于两者之间。最终融合阶段将 A/B/C 和 garden 背景都统一为显式 mesh，在 Blender 中完成尺度对齐、桌面摆放和绕中心漫游渲染。

题目二完成了 B-only 与 ABC-mixed 两个 ACT 策略的公平对比。两者使用相同网络结构和超参数，均训练 10000 steps，并在未见过的 D 环境进行 zero-shot 离线评估。结果显示 ABC-mixed 的 Action L1 更低，说明多环境训练能提升 ACT 在视觉分布偏移下的泛化能力。
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
