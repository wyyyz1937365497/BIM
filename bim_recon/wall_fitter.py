"""Wall fitter: extract wall line segments from 3DGS semantic point clouds.

Takes wall-classified Gaussian means (np.ndarray) -> iterative RANSAC ->
WallFit list (p0, p1, height, normal, thickness). Designed to be decoupled
from GSScene/SceneSplat -- only needs numpy + open3d.

Pipeline (completed across T1-T3):
  fit() = iterative RANSAC -> _plane_to_wall -> _gravity_align -> _merge_walls
          -> _refine_endpoints -> _compute_heights

T1 implements: fit() (RANSAC loop) + _plane_to_wall (PCA extraction).
T2 adds: _merge_walls (occlusion bridging) + _gravity_align.
T3 adds: _refine_endpoints (wall-wall intersection) + _compute_heights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from bim_recon.floorplan import FloorPlan, WallSegment


# ---------------------------------------------------------------------------
# WallFit: the output data structure
# ---------------------------------------------------------------------------

@dataclass
class WallFit:
    """A fitted wall segment in 3D, meters.

    ``p0`` and ``p1`` are the wall endpoints at floor level (up_axis
    coordinate = floor). ``height`` is the wall height. ``normal`` is the
    RANSAC plane normal (not yet gravity-aligned in T1; aligned in T2).
    """

    p0: np.ndarray          # (3,) start point at floor level
    p1: np.ndarray          # (3,) end point at floor level
    height: float           # wall height (m)
    normal: np.ndarray      # (3,) wall normal
    thickness: float        # wall thickness (m)
    num_inliers: int        # RANSAC inlier count
    confidence: float       # inlier ratio (inliers / total points)
    inlier_pts: Optional[np.ndarray] = None  # (N, 3) raw inliers, not serialized

    @property
    def length(self) -> float:
        """Wall length (m) -- distance between p0 and p1."""
        return float(np.linalg.norm(self.p1 - self.p0))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "p0": self.p0.tolist(),
            "p1": self.p1.tolist(),
            "height": self.height,
            "normal": self.normal.tolist(),
            "thickness": self.thickness,
            "num_inliers": self.num_inliers,
            "confidence": self.confidence,
            "length": self.length,
        }


# ---------------------------------------------------------------------------
# WallFitter: the fitting engine
# ---------------------------------------------------------------------------

class WallFitter:
    """Fit wall segments from point clouds via iterative RANSAC.

    Decoupled from GSScene -- takes numpy point arrays only.

    Usage::

        fitter = WallFitter()
        walls = fitter.fit(wall_points, up_axis=2)
        for w in walls:
            print(w.p0, w.p1, w.height, w.length)
    """

    def __init__(
        self,
        distance_threshold: float = 0.08,
        min_inliers: int = 500,
        max_planes: int = 20,
        num_iterations: int = 2000,
        max_thickness: float = 1.0,
    ):
        self.distance_threshold = distance_threshold
        self.min_inliers = min_inliers
        self.max_planes = max_planes
        self.num_iterations = num_iterations
        self.max_thickness = max_thickness

    def fit(
        self,
        points: np.ndarray,
        up_axis: int = 2,
        floor_z: Optional[float] = None,
        ceiling_z: Optional[float] = None,
    ) -> List[WallFit]:
        """Extract wall segments from a point cloud.

        Args:
            points: (N, 3) array of wall-classified Gaussian means (meters).
            up_axis: which axis is vertical (0=x, 1=y, 2=z).
            floor_z: floor level (up_axis coordinate). If given with
                ``ceiling_z``, overrides inlier-based heights.
            ceiling_z: ceiling level. If given with ``floor_z``, wall
                height = ``ceiling_z - floor_z``.

        Returns:
            List of WallFit with refined endpoints (wall-wall intersections
            at corners) and heights (from floor→ceiling if provided).
        """
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] == 0:
            return []

        total = points.shape[0]
        walls: List[WallFit] = []
        remaining = points.copy()

        for _ in range(self.max_planes):
            if remaining.shape[0] < self.min_inliers:
                break

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(remaining)
            model, inliers = pcd.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=3,
                num_iterations=self.num_iterations,
            )
            if len(inliers) < self.min_inliers:
                break

            inlier_pts = remaining[inliers]
            # Remove inliers from remaining
            mask = np.ones(remaining.shape[0], dtype=bool)
            mask[inliers] = False
            remaining = remaining[mask]

            a, b, c, d = model
            norm = float(np.linalg.norm([a, b, c]))
            normal = np.array([a, b, c]) / norm

            wall = self._plane_to_wall(
                inlier_pts, normal, up_axis, len(inliers), total,
            )
            # Filter out non-wall planes (too thick = scatter/slab, not a wall)
            if wall.thickness > self.max_thickness:
                continue
            walls.append(wall)

        # T2: gravity-align each wall, then merge coplanar fragments
        # (handles occlusion gaps and doors splitting one wall into segments)
        aligned = [self._gravity_align(w, up_axis) for w in walls]
        merged = self._merge_walls(aligned, up_axis)

        # T3: refine endpoints (wall-wall intersection at corners)
        refined = self._refine_endpoints(merged, up_axis)

        # T3: compute heights from floor→ceiling if provided
        final = self._compute_heights(refined, floor_z, ceiling_z, up_axis)
        return final

    # ------------------------------------------------------------------
    # Plane -> WallFit conversion (PCA-based)
    # ------------------------------------------------------------------

    def _plane_to_wall(
        self,
        pts: np.ndarray,
        normal: np.ndarray,
        up_axis: int,
        num_inliers: int,
        total_points: int,
    ) -> WallFit:
        """Convert a RANSAC plane's inliers to a WallFit.

        Uses PCA on the horizontal footprint (projected onto the plane
        perpendicular to up_axis) to extract wall direction, length, and
        thickness. Height from the up_axis range of inliers.
        """
        h_axes = [i for i in range(3) if i != up_axis]
        footprint = pts[:, h_axes]  # (N, 2)

        # PCA on footprint to find wall direction
        footprint_centered = footprint - footprint.mean(axis=0)
        cov = np.cov(footprint_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
        main_axis_2d = eigvecs[:, -1]  # largest eigenvalue -> wall direction
        perp_axis_2d = eigvecs[:, 0]   # smallest eigenvalue -> thickness

        proj_main = footprint_centered @ main_axis_2d
        proj_perp = footprint_centered @ perp_axis_2d

        length = float(proj_main.max() - proj_main.min())
        thickness = float(proj_perp.max() - proj_perp.min())

        # Height from up_axis range
        up_coords = pts[:, up_axis]
        floor_z = float(up_coords.min())
        height = float(up_coords.max() - up_coords.min())

        # Endpoints at floor level, spanning the full PCA length
        center_2d = footprint.mean(axis=0)
        half_len = length / 2.0
        p0_2d = center_2d - main_axis_2d * half_len
        p1_2d = center_2d + main_axis_2d * half_len

        # Reconstruct 3D endpoints (up_axis coordinate = floor level)
        p0 = np.zeros(3)
        p1 = np.zeros(3)
        for i, ax in enumerate(h_axes):
            p0[ax] = p0_2d[i]
            p1[ax] = p1_2d[i]
        p0[up_axis] = floor_z
        p1[up_axis] = floor_z

        return WallFit(
            p0=p0,
            p1=p1,
            height=height,
            normal=normal,
            thickness=thickness,
            num_inliers=num_inliers,
            confidence=num_inliers / total_points if total_points > 0 else 0.0,
            inlier_pts=pts,
        )

    # ------------------------------------------------------------------
    # T2: Gravity alignment + wall merge (occlusion bridging)
    # ------------------------------------------------------------------

    def _gravity_align(self, wall: WallFit, up_axis: int) -> WallFit:
        """Align wall normal to horizontal (remove up-axis component).

        After alignment, the normal lies in the horizontal plane, making
        coplanar checks between walls more accurate.
        """
        normal = wall.normal.copy()
        normal[up_axis] = 0.0
        norm = float(np.linalg.norm(normal))
        if norm > 1e-6:
            normal /= norm
        else:
            normal = wall.normal  # degenerate (horizontal normal), keep as-is

        return WallFit(
            p0=wall.p0,
            p1=wall.p1,
            height=wall.height,
            normal=normal,
            thickness=wall.thickness,
            num_inliers=wall.num_inliers,
            confidence=wall.confidence,
            inlier_pts=wall.inlier_pts,
        )

    def _merge_walls(
        self,
        walls: List[WallFit],
        up_axis: int,
        angle_thresh: float = 10.0,
        coplanar_thresh: float = 0.15,
    ) -> List[WallFit]:
        """Merge coplanar, near-parallel wall fragments into single walls.

        **Occlusion bridging**: 3DGS only reconstructs visible surfaces.
        Furniture behind a wall or door openings create spatial gaps in
        the wall Gaussian inliers. Two fragments of the same wall (same
        plane, near-parallel normal, even with a 1-2m spatial gap) are
        merged into one WallFit whose endpoints span the full extent of
        combined inlier projections -- the output wall line is continuous.

        Merge criteria:
          - Normal angle < ``angle_thresh`` (after gravity alignment)
          - Coplanar distance < ``coplanar_thresh`` (|d1-d2| on shared normal)
        """
        if len(walls) <= 1:
            return list(walls)

        merged_flags = [False] * len(walls)
        result: List[WallFit] = []

        for i in range(len(walls)):
            if merged_flags[i]:
                continue
            group_indices = [i]
            merged_flags[i] = True
            for j in range(i + 1, len(walls)):
                if merged_flags[j]:
                    continue
                if self._should_merge(walls[i], walls[j], angle_thresh, coplanar_thresh):
                    group_indices.append(j)
                    merged_flags[j] = True

            if len(group_indices) == 1:
                result.append(walls[i])
            else:
                group = [walls[k] for k in group_indices]
                result.append(self._merge_group(group, up_axis))

        return result

    def _should_merge(
        self,
        w1: WallFit,
        w2: WallFit,
        angle_thresh: float,
        coplanar_thresh: float,
    ) -> bool:
        """Check if two walls are coplanar and near-parallel (should merge).

        Handles antiparallel normals (same plane, opposite-facing normals)
        by using ``abs(dot)`` for the angle check.
        """
        dot = float(np.dot(w1.normal, w2.normal))
        angle = float(np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0))))
        if angle > angle_thresh:
            return False

        # Align normal direction (flip if antiparallel)
        n1 = w1.normal if dot >= 0 else -w1.normal
        n_avg = n1 + w2.normal
        n_norm = float(np.linalg.norm(n_avg))
        if n_norm < 1e-6:
            return True  # degenerate, treat as mergeable
        n_avg /= n_norm

        centroid1 = (w1.p0 + w1.p1) / 2.0
        centroid2 = (w2.p0 + w2.p1) / 2.0
        d1 = float(np.dot(n_avg, centroid1))
        d2 = float(np.dot(n_avg, centroid2))
        coplanar_dist = abs(d1 - d2)

        return coplanar_dist <= coplanar_thresh

    def _merge_group(self, group: List[WallFit], up_axis: int) -> WallFit:
        """Merge a group of coplanar walls into one WallFit.

        Combines inlier points from all fragments, recomputes PCA to get
        endpoints that span the full extent (bridging occlusion gaps).
        """
        # Collect all inlier points
        pts_list = [w.inlier_pts for w in group if w.inlier_pts is not None]
        if pts_list:
            all_pts = np.concatenate(pts_list)
        else:
            # Fallback: use wall endpoints as pseudo-points
            all_pts = np.array([w.p0 for w in group] + [w.p1 for w in group])

        base = max(group, key=lambda w: w.num_inliers)

        # Recompute PCA on combined footprint
        h_axes = [i for i in range(3) if i != up_axis]
        footprint = all_pts[:, h_axes]
        footprint_centered = footprint - footprint.mean(axis=0)
        cov = np.cov(footprint_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_axis_2d = eigvecs[:, -1]

        proj_main = footprint_centered @ main_axis_2d
        length = float(proj_main.max() - proj_main.min())

        center_2d = footprint.mean(axis=0)
        half_len = length / 2.0
        p0_2d = center_2d - main_axis_2d * half_len
        p1_2d = center_2d + main_axis_2d * half_len

        floor_z = min(float(w.p0[up_axis]) for w in group)
        height = max(w.height for w in group)

        p0 = np.zeros(3)
        p1 = np.zeros(3)
        for i, ax in enumerate(h_axes):
            p0[ax] = p0_2d[i]
            p1[ax] = p1_2d[i]
        p0[up_axis] = floor_z
        p1[up_axis] = floor_z

        total_inliers = sum(w.num_inliers for w in group)
        total_confidence = sum(w.confidence for w in group)

        return WallFit(
            p0=p0,
            p1=p1,
            height=height,
            normal=base.normal,
            thickness=base.thickness,
            num_inliers=total_inliers,
            confidence=total_confidence,
            inlier_pts=all_pts,
        )

    # ------------------------------------------------------------------
    # T3: Endpoint refinement (wall-wall intersection) + height extraction
    # ------------------------------------------------------------------

    def _refine_endpoints(
        self,
        walls: List[WallFit],
        up_axis: int,
        max_corner_dist: float = 1.5,
    ) -> List[WallFit]:
        """Refine wall endpoints using wall-wall intersections.

        For each wall, at each end (p0, p1), find the nearest neighboring
        wall and compute the intersection of the two wall lines in the
        horizontal plane. If the intersection is within ``max_corner_dist``
        of the current endpoint, snap to it. This ensures walls meet at
        corners, extending walls that were too short due to occlusion
        or PCA shrinkage.

        Isolated walls (no neighbor) keep their PCA endpoints.
        """
        if len(walls) <= 1:
            return list(walls)

        h_axes = [i for i in range(3) if i != up_axis]
        refined: List[WallFit] = []

        for i, wall in enumerate(walls):
            floor_z = float(wall.p0[up_axis])
            p0_2d = wall.p0[h_axes]
            p1_2d = wall.p1[h_axes]

            new_p0_2d = self._find_corner(
                p0_2d, p1_2d, walls, i, h_axes, max_corner_dist,
            )
            new_p1_2d = self._find_corner(
                p1_2d, p0_2d, walls, i, h_axes, max_corner_dist,
            )

            new_p0 = np.zeros(3)
            new_p1 = np.zeros(3)
            for k, ax in enumerate(h_axes):
                new_p0[ax] = new_p0_2d[k]
                new_p1[ax] = new_p1_2d[k]
            new_p0[up_axis] = floor_z
            new_p1[up_axis] = floor_z

            refined.append(WallFit(
                p0=new_p0,
                p1=new_p1,
                height=wall.height,
                normal=wall.normal,
                thickness=wall.thickness,
                num_inliers=wall.num_inliers,
                confidence=wall.confidence,
                inlier_pts=wall.inlier_pts,
            ))

        return refined

    def _find_corner(
        self,
        endpoint_2d: np.ndarray,
        other_end_2d: np.ndarray,
        walls: List[WallFit],
        wall_idx: int,
        h_axes: List[int],
        max_dist: float,
    ) -> np.ndarray:
        """Find corner point for ``endpoint_2d`` via wall-wall intersection.

        Searches all other walls for the nearest line-line intersection
        within ``max_dist``. Returns the original endpoint if no corner
        is found (isolated wall).
        """
        wall_dir = other_end_2d - endpoint_2d
        dir_norm = float(np.linalg.norm(wall_dir))
        if dir_norm < 1e-6:
            return endpoint_2d
        wall_dir = wall_dir / dir_norm

        best_pt = endpoint_2d
        best_dist = float("inf")

        for j, other in enumerate(walls):
            if j == wall_idx:
                continue
            other_p0_2d = other.p0[h_axes]
            other_p1_2d = other.p1[h_axes]
            other_dir = other_p1_2d - other_p0_2d
            other_dir_norm = float(np.linalg.norm(other_dir))
            if other_dir_norm < 1e-6:
                continue
            other_dir = other_dir / other_dir_norm

            intersection = self._line_intersection_2d(
                endpoint_2d, wall_dir, other_p0_2d, other_dir,
            )
            if intersection is None:
                continue

            dist = float(np.linalg.norm(intersection - endpoint_2d))
            if dist < best_dist and dist < max_dist:
                best_dist = dist
                best_pt = intersection

        return best_pt

    def _line_intersection_2d(
        self,
        p1: np.ndarray,
        d1: np.ndarray,
        p2: np.ndarray,
        d2: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Intersection of two 2D lines (infinite, not segments).

        Args:
            p1: (2,) point on line 1.
            d1: (2,) direction of line 1.
            p2: (2,) point on line 2.
            d2: (2,) direction of line 2.

        Returns:
            (2,) intersection point, or None if lines are parallel.
        """
        det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
        if abs(det) < 1e-8:
            return None
        diff = p2 - p1
        t = (diff[0] * (-d2[1]) - (-d2[0]) * diff[1]) / det
        return p1 + t * d1

    def _compute_heights(
        self,
        walls: List[WallFit],
        floor_z: Optional[float],
        ceiling_z: Optional[float],
        up_axis: int,
    ) -> List[WallFit]:
        """Update wall heights and floor level from floor→ceiling.

        If both ``floor_z`` and ``ceiling_z`` are provided, wall height is
        set to ``ceiling_z - floor_z`` and endpoints are moved to ``floor_z``.
        Otherwise, inlier-based heights are kept.
        """
        if floor_z is None and ceiling_z is None:
            return walls

        target_height = (ceiling_z - floor_z) if (floor_z is not None and ceiling_z is not None) else None
        refined: List[WallFit] = []

        for wall in walls:
            h = wall.height if target_height is None else target_height
            p0 = wall.p0.copy()
            p1 = wall.p1.copy()
            if floor_z is not None:
                p0[up_axis] = floor_z
                p1[up_axis] = floor_z

            refined.append(WallFit(
                p0=p0,
                p1=p1,
                height=h,
                normal=wall.normal,
                thickness=wall.thickness,
                num_inliers=wall.num_inliers,
                confidence=wall.confidence,
                inlier_pts=wall.inlier_pts,
            ))

        return refined


# ---------------------------------------------------------------------------
# Revit / FloorPlan conversion functions (T5)
# ---------------------------------------------------------------------------

def wallfit_to_line_based_element(wall: WallFit) -> Dict[str, Any]:
    """Convert WallFit to ``revit_create_line_based_element`` params.

    Units: meters -> millimeters.
    """
    return {
        "category": "OST_Walls",
        "locationLine": {
            "p0": {
                "x": float(wall.p0[0] * 1000),
                "y": float(wall.p0[1] * 1000),
                "z": float(wall.p0[2] * 1000),
            },
            "p1": {
                "x": float(wall.p1[0] * 1000),
                "y": float(wall.p1[1] * 1000),
                "z": float(wall.p1[2] * 1000),
            },
        },
        "thickness": float(wall.thickness * 1000),
        "height": float(wall.height * 1000),
        "baseLevel": 0,
        "baseOffset": 0,
    }


def wallfit_to_wall_segment(wall: WallFit, up_axis: int = 2):
    """Convert WallFit to FloorPlan.WallSegment (project to horizontal, 3D->2D).

    Drops the up_axis coordinate, keeping only the two horizontal axes.
    Units stay in meters (WallSegment convention).
    """
    from bim_recon.floorplan import WallSegment

    h_axes = [i for i in range(3) if i != up_axis]
    return WallSegment(
        x1=float(wall.p0[h_axes[0]]),
        y1=float(wall.p0[h_axes[1]]),
        x2=float(wall.p1[h_axes[0]]),
        y2=float(wall.p1[h_axes[1]]),
        thickness=float(wall.thickness),
    )


# ---------------------------------------------------------------------------
# FloorPlanGuidedFitter: floorplan-constrained wall fitting
# ---------------------------------------------------------------------------


def _point_to_segment_distance_2d(
    pts: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Return 2D point-to-segment distances for (N, 2) points and segment a->b."""
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    ap = pts - a
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(pts - closest, axis=1)


class FloorPlanGuidedFitter:
    """Fit walls using a 2D floorplan as a spatial prior.

    For each floorplan wall segment, only wall Gaussians within a horizontal
    corridor around that line are kept. A single RANSAC plane is fit inside
    the corridor, then checked against the expected wall normal direction.
    This eliminates furniture/classification noise that would otherwise create
    scattered phantom walls in unconstrained fitting.
    """

    def __init__(
        self,
        corridor_width: float = 0.5,
        normal_angle_thresh: float = 20.0,
        distance_threshold: float = 0.08,
        num_iterations: int = 2000,
        min_inliers: int = 100,
        max_thickness: float = 1.0,
    ):
        self.corridor_width = corridor_width
        self.normal_angle_thresh = normal_angle_thresh
        self.distance_threshold = distance_threshold
        self.num_iterations = num_iterations
        self.min_inliers = min_inliers
        self.max_thickness = max_thickness

    def fit_guided(
        self,
        points: np.ndarray,
        floorplan: "FloorPlan",
        up_axis: int = 2,
        floor_z: Optional[float] = None,
        ceiling_z: Optional[float] = None,
    ) -> List[WallFit]:
        """Fit walls constrained by ``floorplan``.

        Args:
            points: (N, 3) wall-classified Gaussian means (meters).
            floorplan: A FloorPlan already registered to the 3DGS horizontal plane.
            up_axis: Vertical axis index (0=x, 1=y, 2=z).
            floor_z: Optional floor level for height override.
            ceiling_z: Optional ceiling level for height override.

        Returns:
            List of WallFit, one per successfully fitted floorplan wall.
        """
        from bim_recon.floorplan import WallSegment

        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] == 0 or len(floorplan.walls) == 0:
            return []

        h_axes = [i for i in range(3) if i != up_axis]
        footprint = points[:, h_axes]  # (N, 2)
        total = points.shape[0]

        walls: List[WallFit] = []
        for segment in floorplan.walls:
            wall_fit = self._fit_one_segment(
                points=points,
                footprint=footprint,
                segment=segment,
                h_axes=h_axes,
                up_axis=up_axis,
                total=total,
                floor_z=floor_z,
            )
            if wall_fit is not None:
                walls.append(wall_fit)

        # Reuse WallFitter's gravity alignment, endpoint refinement, height logic.
        fitter = WallFitter()
        aligned = [fitter._gravity_align(w, up_axis) for w in walls]
        refined = fitter._refine_endpoints(aligned, up_axis)
        final = fitter._compute_heights(refined, floor_z, ceiling_z, up_axis)
        return final

    def _fit_one_segment(
        self,
        points: np.ndarray,
        footprint: np.ndarray,
        segment: "WallSegment",
        h_axes: List[int],
        up_axis: int,
        total: int,
        floor_z: Optional[float] = None,
    ) -> Optional[WallFit]:
        """Fit a single wall inside the corridor around one floorplan segment.

        Instead of unconstrained RANSAC (which picks a random plane among the
        corridor points), we use the floorplan wall normal as a hard constraint
        and solve for the single free parameter: the plane offset along that
        normal. This is deterministic, avoids phantom diagonal walls from
        furniture noise, and respects the floorplan geometry exactly.

        Algorithm:
          1. Corridor filter (adaptive width).
          2. Project corridor points onto the floorplan wall normal.
          3. Find the dominant projection value via histogram peak.
          4. Wall plane = fixed normal + dominant offset.
          5. Endpoints = floorplan segment endpoints projected onto the plane.
        """
        a = np.array([segment.x1, segment.y1], dtype=np.float64)
        b = np.array([segment.x2, segment.y2], dtype=np.float64)
        seg_dir = b - a
        seg_len = float(np.linalg.norm(seg_dir))
        if seg_len < 1e-6:
            return None

        # Floorplan wall normal in 2D and 3D (vertical plane, normal horizontal).
        seg_normal_2d = np.array([-seg_dir[1], seg_dir[0]], dtype=np.float64)
        seg_normal_2d /= float(np.linalg.norm(seg_normal_2d))
        normal_3d = np.zeros(3, dtype=np.float64)
        normal_3d[h_axes[0]] = seg_normal_2d[0]
        normal_3d[h_axes[1]] = seg_normal_2d[1]
        # up_axis component stays 0 -> vertical plane

        z_floor = floor_z if floor_z is not None else float(points[:, up_axis].min())

        best: Optional[tuple[np.ndarray, np.ndarray, float, int]] = None
        best_inliers = 0

        # Adaptive corridor widths.
        widths = [self.corridor_width * f for f in [0.5, 1.0, 1.5, 2.0, 3.0]]
        for width in widths:
            distances = _point_to_segment_distance_2d(footprint, a, b)
            mask = distances <= width
            corridor_points = points[mask]
            if corridor_points.shape[0] < self.min_inliers:
                continue

            # Project corridor points onto the fixed wall normal.
            proj = corridor_points @ normal_3d  # (N,)

            # Histogram peak finding: bin width = distance_threshold.
            # The wall Gaussian slab should form a narrow peak in projection space.
            bin_width = max(self.distance_threshold, 0.02)
            bins = np.arange(proj.min(), proj.max() + bin_width, bin_width)
            if len(bins) < 2:
                continue
            counts, edges = np.histogram(proj, bins=bins)
            peak_idx = int(np.argmax(counts))
            peak_count = int(counts[peak_idx])
            if peak_count < self.min_inliers:
                continue

            # Inliers = points in the peak bin.
            lo, hi = edges[peak_idx], edges[peak_idx + 1]
            inlier_mask = (proj >= lo) & (proj < hi)
            inlier_pts = corridor_points[inlier_mask]

            # Plane offset: median of inlier projections.
            offset = float(np.median(proj[inlier_mask]))

            if peak_count > best_inliers:
                best_inliers = peak_count
                best = (inlier_pts, normal_3d.copy(), offset, peak_count)

        if best is None:
            return None

        inlier_pts, normal, offset, num_inliers = best

        # Constrain endpoints to the floorplan segment projected onto the plane.
        # Plane equation: normal · x = offset.
        def project_point_to_plane(pt_2d: np.ndarray) -> np.ndarray:
            pt_3d = np.zeros(3, dtype=np.float64)
            for i, ax in enumerate(h_axes):
                pt_3d[ax] = pt_2d[i]
            pt_3d[up_axis] = z_floor
            dist_to_plane = float(np.dot(pt_3d, normal) - offset)
            return pt_3d - dist_to_plane * normal

        p0 = project_point_to_plane(a)
        p1 = project_point_to_plane(b)

        # Thickness from inlier spread along the wall normal.
        proj_thickness = inlier_pts @ normal
        thickness = float(proj_thickness.max() - proj_thickness.min())
        if thickness > self.max_thickness:
            return None

        # Height from overall point cloud up-axis range.
        up_coords = points[:, up_axis]
        height = float(up_coords.max() - up_coords.min())

        return WallFit(
            p0=p0,
            p1=p1,
            height=height,
            normal=normal,
            thickness=thickness,
            num_inliers=num_inliers,
            confidence=num_inliers / total if total > 0 else 0.0,
            inlier_pts=inlier_pts,
        )
