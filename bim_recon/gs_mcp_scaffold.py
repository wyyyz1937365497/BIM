"""3DGS 侧 MCP 工具边界脚手架。

这里先固定工具契约，不绑定具体 GPU 训练产物。后续接入 gsplat、SAGA 或
Gaussian Grouping 时，只需要实现 GsSceneBackend 协议。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class GsSceneBackend(Protocol):
    """3DGS 场景后端需要提供的最小能力。"""

    def render_from_pose(self, pose: list[list[float]], width: int, height: int) -> dict:
        """从给定位姿渲染 RGB 图。"""

    def get_depth(self, pose: list[list[float]], width: int, height: int) -> dict:
        """从给定位姿渲染深度图或期望深度。"""

    def select_cluster(self, mask_2d: dict) -> dict:
        """把 2D mask 反投影为 3D 高斯簇。"""


class BackendNotConfiguredError(RuntimeError):
    """调用 MCP 工具前未注入真实 3DGS 后端。"""


@dataclass(frozen=True)
class McpToolSpec:
    name: str
    purpose: str
    backend: str


TOOL_SPECS = [
    McpToolSpec(
        name="render_from_pose",
        purpose="VLM 从任意视角观察 3DGS 场景",
        backend='gsplat rasterization(render_mode="RGB")',
    ),
    McpToolSpec(
        name="get_depth",
        purpose="返回同视角深度，供几何拟合和遮挡判断使用",
        backend='gsplat rasterization(render_mode="ED" 或 "RGB+ED")',
    ),
    McpToolSpec(
        name="select_cluster",
        purpose="把 VLM/SAM 的 2D mask 关联到 3D 高斯簇",
        backend="SAGA 或 Gaussian Grouping",
    ),
    McpToolSpec(
        name="report_diff",
        purpose="报告 3DGS/VLM 检出墙线与 FloorPlan 的差异",
        backend="bim_recon.diff_report",
    ),
]


class GsMcpToolFacade:
    """MCP server 可直接包一层的 facade。

    facade 本身不加载模型，避免在没有 GPU 或训练产物的机器上 import 即失败。
    """

    def __init__(self, backend: GsSceneBackend | None = None) -> None:
        self._backend = backend

    def list_tool_specs(self) -> list[McpToolSpec]:
        return list(TOOL_SPECS)

    def render_from_pose(self, pose: list[list[float]], width: int = 1280, height: int = 720) -> dict:
        return self._require_backend().render_from_pose(pose, width, height)

    def get_depth(self, pose: list[list[float]], width: int = 1280, height: int = 720) -> dict:
        return self._require_backend().get_depth(pose, width, height)

    def select_cluster(self, mask_2d: dict) -> dict:
        return self._require_backend().select_cluster(mask_2d)

    def _require_backend(self) -> GsSceneBackend:
        if self._backend is None:
            raise BackendNotConfiguredError(
                "3DGS backend is not configured. Inject a gsplat/SAGA backend before serving MCP tools."
            )
        return self._backend
