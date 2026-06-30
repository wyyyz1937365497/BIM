"""RANSAC 可行性验证：query_semantics('wall') 的高斯能否拟合出合理的墙平面。

临时验证脚本（不改代码库）。验证通过后会规划正式墙拟合器模块。

流程:
  1. from_npy 加载场景 (含 feat.pt)
  2. query_semantics('wall', dominant) -> 墙高斯子集
  3. 检测重力轴 (floor centroid 最低的轴 = up)
  4. 迭代 RANSAC 提取多个平面
  5. 对每个平面分析: 法向、水平度、长度、厚度、高度
  6. 提取墙线段端点 (footprint 主成分)
  7. 输出结论

Run:
  python scripts/ransac_wall_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import open3d as o3d

from bim_recon.gs_scene import GSScene


# ---------------------------------------------------------------------------
# 迭代 RANSAC：从点云中逐个提取平面
# ---------------------------------------------------------------------------

def extract_planes(
    points: np.ndarray,
    max_planes: int = 8,
    min_inliers: int = 1000,
    distance_threshold: float = 0.08,
    num_iterations: int = 2000,
):
    """迭代 RANSAC：每次提取最大平面，移除 inliers，重复。

    Returns (planes, remaining):
      planes: list of dict(normal, d, num_inliers, inlier_pts)
      remaining: 未被任何平面收录的点 (N_remain, 3)
    """
    planes = []
    remaining = np.asarray(points, dtype=np.float64).copy()

    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining)
        model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=num_iterations,
        )
        if len(inliers) < min_inliers:
            break

        inlier_pts = remaining[inliers]
        # 移除 inliers
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]

        a, b, c, d = model
        norm = float(np.linalg.norm([a, b, c]))
        normal = np.array([a, b, c]) / norm

        planes.append({
            "normal": normal,
            "d": d / norm,
            "num_inliers": len(inliers),
            "inlier_pts": inlier_pts,
        })

    return planes, remaining


# ---------------------------------------------------------------------------
# 单个墙平面分析
# ---------------------------------------------------------------------------

def analyze_plane(plane: dict, up_axis: int) -> dict:
    """分析单个平面的几何属性。

    up_axis: 哪个轴是竖直方向 (0=x, 1=y, 2=z)。
    """
    pts = plane["inlier_pts"]
    normal = plane["normal"]

    # 法向的水平性：墙垂直 -> 法向水平 -> up 分量接近 0
    up_component = abs(normal[up_axis])
    horizontalness = 1.0 - up_component  # 越接近 1 越像墙

    # footprint = 投影到水平面（去掉 up 轴）
    h_axes = [i for i in range(3) if i != up_axis]
    footprint = pts[:, h_axes]

    # 主成分分析：footprint 的主方向 = 墙的走向
    footprint_centered = footprint - footprint.mean(axis=0)
    cov = np.cov(footprint_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 升序
    main_axis = eigvecs[:, -1]   # 最大特征值 -> 墙走向
    perp_axis = eigvecs[:, 0]    # 最小特征值 -> 墙厚度方向

    proj_main = footprint_centered @ main_axis
    proj_perp = footprint_centered @ perp_axis

    length = float(proj_main.max() - proj_main.min())
    thickness = float(proj_perp.max() - proj_perp.min())

    up_coords = pts[:, up_axis]
    height = float(up_coords.max() - up_coords.min())

    return {
        "normal": normal,
        "up_component": float(up_component),
        "horizontalness": float(horizontalness),
        "length": length,
        "thickness": thickness,
        "height": height,
        "y_min": float(up_coords.min()),
        "y_max": float(up_coords.max()),
        "main_axis_xy": main_axis.tolist(),
        "centroid_xy": footprint.mean(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("RANSAC 可行性验证：query_semantics('wall') 的墙平面拟合")
    print("=" * 72)

    # --- 1. 加载场景 ---
    scene = GSScene.from_npy(
        "data", device="cpu",
        feat_path="output/data_feat.pt",
        text_emb_path="data/bim_text_emb.pt",
        class_names_path="data/bim_class_names.json",
    )
    print(f"\n[1] 加载场景: {scene.num_gaussians} Gaussians")

    # --- 2. 拿到墙高斯 ---
    result = scene.query_semantics("wall", mode="dominant")
    wall_indices = result["indices"]
    wall_means = scene.means[
        torch.as_tensor(wall_indices, dtype=torch.long)
    ].cpu().numpy().astype(np.float64)
    print(f"[2] query_semantics('wall', dominant): {len(wall_means)} 墙高斯")
    print(f"    AABB: {wall_means.min(axis=0).round(2)} -> {wall_means.max(axis=0).round(2)}")

    # --- 3. 检测重力轴 (floor centroid 最低的轴 = up) ---
    floor_result = scene.query_semantics("floor", mode="dominant")
    floor_indices = floor_result["indices"]
    floor_means = scene.means[
        torch.as_tensor(floor_indices, dtype=torch.long)
    ].cpu().numpy().astype(np.float64)
    floor_centroid = floor_means.mean(axis=0)
    up_axis = int(np.argmin(floor_centroid))
    axis_name = "xyz"[up_axis]
    print(f"[3] 重力轴检测: floor centroid = {floor_centroid.round(2)}, "
          f"up_axis = {up_axis} ('{axis_name}')")

    # --- 4. 迭代 RANSAC ---
    print(f"\n[4] 迭代 RANSAC 平面拟合 (distance_threshold=0.08m, min_inliers=1000)...")
    planes, remaining = extract_planes(
        wall_means, max_planes=8, min_inliers=1000, distance_threshold=0.08,
    )
    total = len(wall_means)
    print(f"    提取 {len(planes)} 个平面, 剩余 {len(remaining)} 未拟合点 ({len(remaining)/total:.1%})")

    # --- 5. 分析每个平面 ---
    print(f"\n[5] 平面分析:")
    header = (f"    {'#':>2} {'inliers':>8} {'占比':>6} {'法向 [x,y,z]':>26} "
              f"{'水平度':>6} {'长度':>7} {'厚度':>6} {'高度':>6}")
    print(header)
    wall_candidates = []
    for i, p in enumerate(planes):
        info = analyze_plane(p, up_axis=up_axis)
        ratio = p["num_inliers"] / total
        is_wall = info["horizontalness"] > 0.85 and info["length"] > 0.5
        if is_wall:
            wall_candidates.append((i, p, info))
        marker = " <- 墙" if is_wall else ""
        n = info["normal"]
        n_str = f"[{n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f}]"
        print(f"    {i:>2} {p['num_inliers']:>8} {ratio:>5.1%} {n_str:>26} "
              f"{info['horizontalness']:>5.2f} {info['length']:>5.2f}m "
              f"{info['thickness']:>5.2f}m {info['height']:>5.2f}m{marker}")

    # --- 6. 墙线段提取 ---
    if wall_candidates:
        print(f"\n[6] 墙线段提取（{len(wall_candidates)} 个墙候选）:")
        print(f"    {'墙':>2} {'p0':>20} {'p1':>20} {'长度':>6} {'高度':>6} {'y范围':>16}")
        for idx, (_, p, info) in enumerate(wall_candidates):
            main_dir = np.array(info["main_axis_xy"])
            center = np.array(info["centroid_xy"])
            half_len = info["length"] / 2
            p0 = center - main_dir * half_len
            p1 = center + main_dir * half_len
            y_range = f"({info['y_min']:.2f}->{info['y_max']:.2f})"
            print(f"    {idx:>2} ({p0[0]:>7.2f},{p0[1]:>7.2f}) "
                  f"({p1[0]:>7.2f},{p1[1]:>7.2f}) "
                  f"{info['length']:>5.2f}m {info['height']:>5.2f}m {y_range:>16}")

    # --- 7. 结论 ---
    print(f"\n[7] 结论:")
    if len(wall_candidates) >= 2:
        total_wall_inliers = sum(p["num_inliers"] for _, p, _ in wall_candidates)
        print(f"    ✅ 验证通过: 识别 {len(wall_candidates)} 面墙, "
              f"占墙高斯的 {total_wall_inliers/total:.1%}")
        print(f"    → SceneSplat wall 高斯确实空间聚集在墙面")
        print(f"    → RANSAC 可提取清晰墙线段")
        print(f"    → 建议进入正式墙拟合器模块规划")
    elif len(wall_candidates) >= 1:
        print(f"    ⚠️ 部分通过: 仅识别 {len(wall_candidates)} 面墙")
        print(f"    → 可能需要调参 (distance_threshold/min_inliers) 或分类质量不足")
    else:
        print(f"    ❌ 验证失败: 未识别出任何墙平面")
        print(f"    → SceneSplat wall 高斯可能空间散布, 不适合直接 RANSAC")

    return 0 if len(wall_candidates) >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
