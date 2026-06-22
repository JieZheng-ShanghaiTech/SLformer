from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


BLUE = "#2166AC"
GREY = "#7A8190"
PURPLE = "#5E3C99"
ORANGE = "#E69F00"
DARK_GREY = "#3A3A3A"
SL_COLORSCALE = [[0.0, PURPLE], [0.50, GREY], [1.0, BLUE]]


@dataclass(frozen=True)
class LocalTraceGeometry:
    neighbor_indices: np.ndarray
    neighbor_coords: np.ndarray
    neighbor_scores: np.ndarray
    curve_indices: np.ndarray
    curve_coords: np.ndarray
    curve_scores: np.ndarray
    target_coord: np.ndarray
    target_curve_row: int
    pca2_variance_sum: float
    mean_tangent_norm: float
    mean_residual_norm: float
    median_tangent_fraction: float


def local_trace_geometry(
    Xn: np.ndarray,
    y: np.ndarray,
    *,
    target_idx: int,
    direction: np.ndarray,
    neighbor_count: int,
    metric: str,
    core_radius_quantile: float,
    seed: int,
) -> LocalTraceGeometry:
    local_direction = direction.astype(np.float32)
    local_direction = local_direction / (np.linalg.norm(local_direction) + 1e-12)
    local_x0 = Xn[int(target_idx)]

    knn = NearestNeighbors(n_neighbors=int(neighbor_count) + 1, metric=str(metric))
    knn.fit(Xn)
    neighbor_indices = knn.kneighbors(local_x0[None, :], return_distance=False)[0]
    neighbor_indices = neighbor_indices[neighbor_indices != int(target_idx)][: int(neighbor_count)]

    plot_indices = np.r_[neighbor_indices, int(target_idx)]
    centered = Xn[plot_indices] - local_x0[None, :]
    tangent_t = centered @ local_direction
    residual = centered - tangent_t[:, None] * local_direction[None, :]
    pca2 = PCA(n_components=2, random_state=int(seed))
    orthogonal_pc = pca2.fit_transform(residual)
    coords = np.column_stack([tangent_t, orthogonal_pc])

    neighbor_coords = coords[:-1]
    target_coord = coords[-1]
    neighbor_radius = np.linalg.norm(neighbor_coords[:, 1:3], axis=1)
    core_mask = neighbor_radius <= np.quantile(neighbor_radius, float(core_radius_quantile))

    curve_indices_unsorted = np.r_[neighbor_indices[core_mask], int(target_idx)]
    curve_coords_unsorted = np.vstack([neighbor_coords[core_mask], target_coord[None, :]])
    curve_order = np.lexsort((np.linalg.norm(curve_coords_unsorted[:, 1:3], axis=1), curve_coords_unsorted[:, 0]))
    curve_indices = curve_indices_unsorted[curve_order]
    curve_coords = curve_coords_unsorted[curve_order]
    target_curve_row = int(np.flatnonzero(curve_indices == int(target_idx))[0])

    centered_neighbors = Xn[neighbor_indices] - local_x0[None, :]
    tangent_component = (centered_neighbors @ local_direction)[:, None] * local_direction[None, :]
    residual_component = centered_neighbors - tangent_component
    tangent_norm = np.linalg.norm(tangent_component, axis=1)
    residual_norm = np.linalg.norm(residual_component, axis=1)
    total_norm_sq = np.linalg.norm(centered_neighbors, axis=1) ** 2

    return LocalTraceGeometry(
        neighbor_indices=neighbor_indices.astype(int),
        neighbor_coords=neighbor_coords.astype(float),
        neighbor_scores=y[neighbor_indices].astype(float),
        curve_indices=curve_indices.astype(int),
        curve_coords=curve_coords.astype(float),
        curve_scores=y[curve_indices].astype(float),
        target_coord=curve_coords[target_curve_row].astype(float),
        target_curve_row=target_curve_row,
        pca2_variance_sum=float(pca2.explained_variance_ratio_.sum()),
        mean_tangent_norm=float(tangent_norm.mean()),
        mean_residual_norm=float(residual_norm.mean()),
        median_tangent_fraction=float(np.median(tangent_norm**2 / (total_norm_sq + 1e-12))),
    )


def feature_anchor_table(
    model,
    Xn: np.ndarray,
    geometry: LocalTraceGeometry,
    feature_rank: pd.DataFrame,
    llm_summary: pd.DataFrame,
    *,
    atom_count: int,
    device: str,
) -> pd.DataFrame:
    atom_table = feature_rank.merge(
        llm_summary[["feature", "hypothesis", "confidence", "rationale"]],
        on="feature",
        how="left",
    )
    atom_table = atom_table.sort_values("joint_rank_score", ascending=False).head(int(atom_count)).reset_index(drop=True)
    atom_table["explained"] = atom_table["feature"].isin(llm_summary["feature"].astype(int))
    atom_table["hypothesis"] = atom_table["hypothesis"].fillna("not selected for LLM explanation")
    atom_table["confidence"] = atom_table["confidence"].fillna("not explained")

    features = atom_table["feature"].astype(int).to_numpy()
    with torch.no_grad():
        curve_z = model.encode(torch.from_numpy(Xn[geometry.curve_indices].astype(np.float32)).to(device)).detach().cpu().numpy()
    curve_z = np.clip(curve_z[:, features], 0.0, None)
    activation_mass = curve_z.sum(axis=0)

    anchor_coords = []
    anchor_rank = []
    anchor_defined = []
    curve_rows = np.arange(len(geometry.curve_coords), dtype=float)
    for feature_offset, mass in enumerate(activation_mass):
        if mass > 0:
            weights = curve_z[:, feature_offset] / mass
            anchor_coords.append(weights @ geometry.curve_coords)
            anchor_rank.append(float(weights @ curve_rows))
            anchor_defined.append(True)
        else:
            anchor_coords.append([np.nan, np.nan, np.nan])
            anchor_rank.append(np.nan)
            anchor_defined.append(False)

    anchor_coords = np.asarray(anchor_coords, dtype=float)
    atom_table["anchor_t"] = anchor_coords[:, 0]
    atom_table["anchor_pc1"] = anchor_coords[:, 1]
    atom_table["anchor_pc2"] = anchor_coords[:, 2]
    atom_table["anchor_curve_rank"] = anchor_rank
    atom_table["anchor_defined"] = anchor_defined
    return atom_table


def plot_local_trace(
    geometry: LocalTraceGeometry,
    atom_table: pd.DataFrame,
    *,
    title: str,
) -> None:
    defined_atoms = atom_table[atom_table["anchor_defined"]].copy()
    defined_atoms["marker_size"] = 8.0 + 10.0 * defined_atoms["joint_rank_score"].rank(pct=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=geometry.neighbor_coords[:, 0],
            y=geometry.neighbor_coords[:, 1],
            z=geometry.neighbor_coords[:, 2],
            mode="markers",
            marker=dict(
                size=3.4,
                color=geometry.neighbor_scores,
                colorscale=SL_COLORSCALE,
                cmin=float(geometry.neighbor_scores.min()),
                cmax=float(geometry.neighbor_scores.max()),
                opacity=0.32,
                line=dict(color="rgba(31,42,95,0.18)", width=0.25),
                colorbar=dict(title="SL score", thickness=10, len=0.46, outlinewidth=0),
            ),
            name="local neighbours",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=geometry.curve_coords[:, 0],
            y=geometry.curve_coords[:, 1],
            z=geometry.curve_coords[:, 2],
            mode="markers+lines",
            marker=dict(
                size=4.2,
                color=geometry.curve_scores,
                colorscale=SL_COLORSCALE,
                cmin=float(geometry.neighbor_scores.min()),
                cmax=float(geometry.neighbor_scores.max()),
                opacity=0.78,
                line=dict(color="white", width=0.45),
                showscale=False,
            ),
            line=dict(color="rgba(94,60,153,0.58)", width=2.2),
            name="core trace",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(geometry.target_coord[0])],
            y=[float(geometry.target_coord[1])],
            z=[float(geometry.target_coord[2])],
            mode="markers+text",
            marker=dict(size=13, color="rgba(125,129,144,0.96)", line=dict(color=ORANGE, width=3.0)),
            text=["target"],
            textposition="top center",
            name="target pair",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=defined_atoms["anchor_t"],
            y=defined_atoms["anchor_pc1"],
            z=defined_atoms["anchor_pc2"],
            mode="markers+text",
            marker=dict(
                size=defined_atoms["marker_size"],
                color=defined_atoms["joint_rank_score"],
                colorscale=[[0.0, BLUE], [0.55, GREY], [1.0, PURPLE]],
                cmin=0.0,
                cmax=1.0,
                opacity=0.96,
                line=dict(color=ORANGE, width=1.8),
                showscale=False,
            ),
            text=[f"f{int(feature)}" for feature in defined_atoms["feature"]],
            textposition="top center",
            name="SAE atoms",
            hovertext=defined_atoms["hypothesis"],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.98, font=dict(size=15, color=DARK_GREY)),
        width=980,
        height=620,
        template="simple_white",
        margin=dict(l=0, r=0, t=56, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.72)"),
        scene=dict(
            xaxis=dict(title="score tangent $t$", showbackground=True, backgroundcolor="rgba(249,251,253,1)"),
            yaxis=dict(title="residual PC1", showbackground=True, backgroundcolor="rgba(249,251,253,1)"),
            zaxis=dict(title="residual PC2", showbackground=True, backgroundcolor="rgba(249,251,253,1)"),
            aspectmode="manual",
            aspectratio=dict(x=1.28, y=1.0, z=0.86),
            camera=dict(eye=dict(x=1.38, y=-1.55, z=1.05), center=dict(x=0.02, y=0.0, z=-0.03)),
        ),
    )
    fig.show()


def plot_trace_snapshots(
    geometries: Sequence[LocalTraceGeometry],
    *,
    neighbor_counts: Sequence[int],
    title: str,
) -> pd.DataFrame:
    fig = make_subplots(
        rows=1,
        cols=len(geometries),
        specs=[[{"type": "scene"} for _ in geometries]],
        subplot_titles=[f"K={int(count)}" for count in neighbor_counts],
        horizontal_spacing=0.006,
    )

    summary_rows = []
    for panel_index, geometry in enumerate(geometries):
        col = panel_index + 1
        line_rows = np.arange(0, len(geometry.curve_coords), max(1, len(geometry.curve_coords) // 16))
        if line_rows[-1] != len(geometry.curve_coords) - 1:
            line_rows = np.r_[line_rows, len(geometry.curve_coords) - 1]

        fig.add_trace(
            go.Scatter3d(
                x=geometry.neighbor_coords[:, 0],
                y=geometry.neighbor_coords[:, 1],
                z=geometry.neighbor_coords[:, 2],
                mode="markers",
                marker=dict(size=2.2, color=geometry.neighbor_scores, colorscale=SL_COLORSCALE, opacity=0.24, showscale=False),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=geometry.curve_coords[line_rows, 0],
                y=geometry.curve_coords[line_rows, 1],
                z=geometry.curve_coords[line_rows, 2],
                mode="lines",
                line=dict(color="rgba(94,60,153,0.64)", width=2.0),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=geometry.curve_coords[:, 0],
                y=geometry.curve_coords[:, 1],
                z=geometry.curve_coords[:, 2],
                mode="markers",
                marker=dict(size=2.5, color=geometry.curve_scores, colorscale=SL_COLORSCALE, opacity=0.68, line=dict(color="white", width=0.20), showscale=False),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[float(geometry.target_coord[0])],
                y=[float(geometry.target_coord[1])],
                z=[float(geometry.target_coord[2])],
                mode="markers",
                marker=dict(size=6.5, color="rgba(125,129,144,0.96)", line=dict(color=ORANGE, width=1.9)),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        summary_rows.append(
            {
                "neighbor_count": int(neighbor_counts[panel_index]),
                "core_curve_points": int(len(geometry.curve_coords)),
                "target_curve_rank": int(geometry.target_curve_row + 1),
                "mean_tangent_norm": geometry.mean_tangent_norm,
                "mean_residual_norm": geometry.mean_residual_norm,
                "median_tangent_fraction": geometry.median_tangent_fraction,
                "orthogonal_pca2_variance_sum": geometry.pca2_variance_sum,
            }
        )

    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgba(252,252,255,1)",
        gridcolor="rgba(198,203,230,0.22)",
        zeroline=False,
        showticklabels=False,
        title="",
    )
    for scene_name in ["scene"] + [f"scene{i}" for i in range(2, len(geometries) + 1)]:
        fig.update_layout(
            {
                scene_name: dict(
                    xaxis=axis_style,
                    yaxis=axis_style,
                    zaxis=axis_style,
                    aspectmode="manual",
                    aspectratio=dict(x=1.18, y=1.0, z=0.82),
                    camera=dict(eye=dict(x=1.36, y=-1.52, z=1.04), center=dict(x=0.02, y=0.0, z=-0.03)),
                )
            }
        )

    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.99, font=dict(size=15, color=DARK_GREY)),
        width=1180,
        height=360,
        template="simple_white",
        paper_bgcolor="white",
        margin=dict(l=4, r=4, t=56, b=4),
    )
    fig.show()
    return pd.DataFrame(summary_rows)
