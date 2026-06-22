from __future__ import annotations

from itertools import combinations, product
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from tqdm import tqdm


def parse_ks_config(value: Sequence[int] | dict[str, int], *, fit_k: int) -> tuple[int, ...]:
    if isinstance(value, dict):
        ks = tuple(range(int(value["start"]), int(value["stop"]), int(value["step"])))
    else:
        ks = tuple(int(k) for k in value)
    return tuple(k for k in ks if k <= int(fit_k))


class TangentSpace:
    def __init__(self, basis: np.ndarray, mean: np.ndarray):
        self.basis = basis
        self.mean = mean


def build_knn_index(X: np.ndarray, *, n_neighbors: int = 50, metric: str = "cosine") -> NearestNeighbors:
    nn = NearestNeighbors(n_neighbors=min(int(n_neighbors), int(X.shape[0])), metric=metric)
    nn.fit(X)
    return nn


def estimate_tangent_space(
    X_neighbors: np.ndarray,
    *,
    k: int = 32,
) -> TangentSpace:
    """Estimate local tangent via PCA on centered neighbor cloud."""
    mu = X_neighbors.mean(axis=0)
    Xc = X_neighbors - mu
    pca = PCA(n_components=int(k))
    pca.fit(Xc)
    basis = pca.components_.T  # (d, k)
    # Ensure orthonormal-ish
    # (PCA components are orthonormal in feature space)
    return TangentSpace(basis=basis.astype(np.float32), mean=mu.astype(np.float32))


def project_to_tangent(delta: np.ndarray, tangent: TangentSpace) -> Tuple[np.ndarray, np.ndarray]:
    """Return (tangent_component, normal_component)."""
    B = tangent.basis
    t = B @ (B.T @ delta)
    n = delta - t
    return t, n


def estimate_local_score_gradient(
    x0: np.ndarray,
    X_neighbors: np.ndarray,
    y_neighbors: np.ndarray,
    *,
    tangent_dim: int = 32,
    ridge_alpha: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Estimate a local gradient direction for score y in the tangent coordinates.

    We approximate y(x) locally with a linear model in tangent coordinates:
      y \approx a + g^T (B^T (x - mu))
    where B are PCA tangent basis vectors.

    Returns dict with:
    - grad_ambient: (d,) gradient in ambient space
    - grad_tangent: (k,) gradient in tangent coordinates
    - basis: (d, k)
    - mu: (d,)
    """
    tangent = estimate_tangent_space(X_neighbors, k=int(tangent_dim))
    B, mu = tangent.basis, tangent.mean

    Z = (X_neighbors - mu) @ B  # (n, k)
    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    model.fit(Z, y_neighbors)
    g_tan = model.coef_.astype(np.float32)  # (k,)
    g_amb = (B @ g_tan).astype(np.float32)  # (d,)

    return {"grad_ambient": g_amb, "grad_tangent": g_tan, "basis": B, "mu": mu}


def estimate_point_score_direction(
    X: np.ndarray,
    y: np.ndarray,
    *,
    point_index: int,
    n_neighbors: int = 80,
    metric: str = "cosine",
    tangent_dim: int = 32,
    ridge_alpha: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Estimate local score-increasing direction for one point in ambient space.

    Returns a dictionary containing x0/y0, neighbor indices, and both raw/unit
    ambient gradient directions estimated from local tangent regression.
    """

    i0 = int(point_index)
    # Query one extra neighbor because sklearn usually returns the point itself.
    nn = build_knn_index(X, n_neighbors=int(n_neighbors) + 1, metric=metric)
    idx = nn.kneighbors(X[i0 : i0 + 1], return_distance=False)[0]
    # Drop self if present as first entry.
    idx = idx[idx != i0][: int(n_neighbors)]

    Xn = X[idx]
    yn = y[idx]
    res = estimate_local_score_gradient(
        x0=X[i0],
        X_neighbors=Xn,
        y_neighbors=yn,
        tangent_dim=int(tangent_dim),
        ridge_alpha=float(ridge_alpha),
    )
    g = res["grad_ambient"].astype(np.float32)
    g_unit = g / (np.linalg.norm(g) + 1e-12)
    return {
        "point_index": np.asarray(i0, dtype=np.int64),
        "x0": X[i0].astype(np.float32),
        "y0": np.asarray(y[i0], dtype=np.float32),
        "neighbor_indices": idx.astype(np.int64),
        "grad_ambient": g,
        "grad_ambient_unit": g_unit.astype(np.float32),
        "grad_tangent": res["grad_tangent"].astype(np.float32),
        "basis": res["basis"].astype(np.float32),
        "mu": res["mu"].astype(np.float32),
    }


def ordered_spearman(values: np.ndarray) -> float:
    ranks = np.arange(len(values), 0, -1)
    if np.unique(values).size < 2:
        return np.nan
    return float(spearmanr(ranks, values).statistic)


def overlapping_interval_cover(values: np.ndarray, *, n_covers: int, overlap_fraction: float) -> list[tuple[float, float]]:
    values = np.asarray(values, dtype=float)
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        return [(lo - 0.5, hi + 0.5)]

    n = int(n_covers)
    overlap = float(overlap_fraction)
    width = (hi - lo) / (n - (n - 1) * overlap)
    step = width * (1.0 - overlap)
    intervals = []
    for j in range(n):
        left = lo + j * step
        right = left + width
        if j == n - 1:
            right = hi
        intervals.append((left, right))
    return intervals


def all_intersecting_interval_cover(
    values: np.ndarray,
    *,
    n_covers: int,
    core_fraction: float = 0.10,
    core_center: float | None = None,
) -> list[tuple[float, float]]:
    values = np.asarray(values, dtype=float)
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        return [(lo - 0.5, hi + 0.5)]

    n = int(n_covers)
    mid = 0.5 * (lo + hi) if core_center is None else float(core_center)
    core_half_width = 0.5 * float(core_fraction) * (hi - lo)
    centers = np.linspace(lo, hi, n)
    intervals = []
    for center in centers:
        half_width = abs(float(center) - mid) + core_half_width
        intervals.append((max(lo, float(center) - half_width), min(hi, float(center) + half_width)))
    intervals[0] = (lo, intervals[0][1])
    intervals[-1] = (intervals[-1][0], hi)
    return intervals


def make_interval_cover(
    values: np.ndarray,
    *,
    n_covers: int,
    overlap_fraction: float,
    cover_mode: str = "standard",
    cover_core_fraction: float = 0.10,
    cover_core_center: float | None = None,
) -> list[tuple[float, float]]:
    if str(cover_mode) == "standard":
        return overlapping_interval_cover(
            values,
            n_covers=int(n_covers),
            overlap_fraction=float(overlap_fraction),
        )
    if str(cover_mode) == "all-intersecting":
        return all_intersecting_interval_cover(
            values,
            n_covers=int(n_covers),
            core_fraction=float(cover_core_fraction),
            core_center=cover_core_center,
        )
    raise ValueError(f"unknown cover mode: {cover_mode}")


def build_mapper_nerve(
    coords: np.ndarray,
    *,
    lens_dims: Sequence[int] = (0, 1),
    n_covers: int = 6,
    overlap_fraction: float = 0.35,
    cover_mode: str = "standard",
    cover_core_fraction: float = 0.10,
    cover_anchor_position: int | None = None,
    cluster_mode: str = "dbscan",
    dbscan_eps_quantile: float = 0.65,
    dbscan_min_samples: int = 3,
) -> Dict:
    """Build a Mapper nerve graph from local chart coordinates.

    Returns a dictionary with the full nerve structure suitable for both
    distance computation and topological visualization:

    - ``node_members``: list of point-index arrays per node
    - ``node_centers``: (n_nodes, n_dims) barycentre array
    - ``point_node_memberships``: which nodes each point belongs to
    - ``graph``: (n_nodes, n_nodes) adjacency (inf = no edge)
    - ``cover_intervals``: list of interval-tuple lists per lens dimension
    - ``cover_boxes``: list of (dim0_interval_idx, dim1_interval_idx) per
      non-empty cover element
    - ``box_first_node``: index of first node created for each non-empty box
    """
    coords = np.asarray(coords, dtype=float)
    lens = coords[:, list(lens_dims)]
    cover_intervals = [
        make_interval_cover(
            lens[:, dim],
            n_covers=int(n_covers),
            overlap_fraction=float(overlap_fraction),
            cover_mode=str(cover_mode),
            cover_core_fraction=float(cover_core_fraction),
            cover_core_center=(
                float(lens[int(cover_anchor_position), dim])
                if cover_anchor_position is not None
                else None
            ),
        )
        for dim in range(lens.shape[1])
    ]

    node_members: list[np.ndarray] = []
    point_node_memberships: list[list[int]] = [[] for _ in range(coords.shape[0])]
    node_centers: list[np.ndarray] = []
    cover_boxes: list[tuple[int, int]] = []
    box_first_node: list[int] = []

    for box in product(*cover_intervals):
        in_box = np.ones(coords.shape[0], dtype=bool)
        for dim, (left, right) in enumerate(box):
            if right == cover_intervals[dim][-1][1]:
                in_box &= (lens[:, dim] >= left) & (lens[:, dim] <= right)
            else:
                in_box &= (lens[:, dim] >= left) & (lens[:, dim] < right)
        box_indices = np.flatnonzero(in_box)
        if len(box_indices) == 0:
            continue

        box_dim_indices: list[int] = []
        for dim in range(len(cover_intervals)):
            target_left = box[dim][0]
            idx = next(i for i, (l, r) in enumerate(cover_intervals[dim])
                       if abs(l - target_left) < 1e-14)
            box_dim_indices.append(idx)
        cover_boxes.append(tuple(box_dim_indices))
        box_first_node.append(len(node_members))

        if str(cluster_mode) == "cover":
            labels = np.zeros(len(box_indices), dtype=int)
        elif len(box_indices) == 1:
            labels = np.array([0], dtype=int)
        elif str(cluster_mode) == "dbscan":
            nn = NearestNeighbors(n_neighbors=min(2, len(box_indices))).fit(coords[box_indices])
            distances = nn.kneighbors(coords[box_indices], return_distance=True)[0]
            eps = float(np.quantile(distances[:, -1], float(dbscan_eps_quantile)))
            eps = max(eps, np.finfo(float).eps)
            labels = DBSCAN(eps=eps, min_samples=int(dbscan_min_samples)).fit_predict(coords[box_indices])
        else:
            raise ValueError(f"unknown Mapper cluster mode: {cluster_mode}")

        for cluster_label in sorted(label for label in np.unique(labels) if label >= 0):
            members = box_indices[labels == cluster_label]
            node_members.append(members)
            node_index = len(node_members) - 1
            for point_index in members:
                point_node_memberships[int(point_index)].append(node_index)
            node_centers.append(coords[members].mean(axis=0))

        for noise_position in np.flatnonzero(labels < 0):
            members = box_indices[noise_position : noise_position + 1]
            node_members.append(members)
            node_index = len(node_members) - 1
            point_node_memberships[int(members[0])].append(node_index)
            node_centers.append(coords[members[0]])

    node_centers_arr = np.vstack(node_centers)
    graph = np.full((len(node_members), len(node_members)), np.inf, dtype=float)
    np.fill_diagonal(graph, 0.0)
    for memberships in point_node_memberships:
        for i, j in combinations(memberships, 2):
            weight = float(np.linalg.norm(node_centers_arr[i] - node_centers_arr[j]))
            graph[i, j] = min(graph[i, j], weight)
            graph[j, i] = graph[i, j]

    return {
        "node_members": node_members,
        "node_centers": node_centers_arr,
        "point_node_memberships": point_node_memberships,
        "graph": graph,
        "cover_intervals": cover_intervals,
        "cover_boxes": cover_boxes,
        "box_first_node": box_first_node,
        "lens_dims": list(lens_dims),
        "cover_mode": str(cover_mode),
        "cover_core_fraction": float(cover_core_fraction),
        "cover_anchor_position": None if cover_anchor_position is None else int(cover_anchor_position),
        "cluster_mode": str(cluster_mode),
    }


def build_cover_grid_nerve(
    coords: np.ndarray,
    *,
    lens_dims: Sequence[int] = (0, 1),
    n_covers: int = 5,
    overlap_fraction: float = 0.35,
    cover_mode: str = "standard",
    cover_core_fraction: float = 0.10,
    cover_anchor_position: int | None = None,
) -> Dict:
    """Build a coarse Mapper-style graph with one node per non-empty cover box."""
    coords = np.asarray(coords, dtype=float)
    lens = coords[:, list(lens_dims)]
    cover_intervals = [
        make_interval_cover(
            lens[:, dim],
            n_covers=int(n_covers),
            overlap_fraction=float(overlap_fraction),
            cover_mode=str(cover_mode),
            cover_core_fraction=float(cover_core_fraction),
            cover_core_center=(
                float(lens[int(cover_anchor_position), dim])
                if cover_anchor_position is not None
                else None
            ),
        )
        for dim in range(lens.shape[1])
    ]

    node_members: list[np.ndarray] = []
    point_node_memberships: list[list[int]] = [[] for _ in range(coords.shape[0])]
    node_centers: list[np.ndarray] = []
    cover_boxes: list[tuple[int, int]] = []
    box_first_node: list[int] = []

    for box in product(*cover_intervals):
        in_box = np.ones(coords.shape[0], dtype=bool)
        for dim, (left, right) in enumerate(box):
            if right == cover_intervals[dim][-1][1]:
                in_box &= (lens[:, dim] >= left) & (lens[:, dim] <= right)
            else:
                in_box &= (lens[:, dim] >= left) & (lens[:, dim] < right)
        members = np.flatnonzero(in_box)
        if len(members) == 0:
            continue

        box_dim_indices: list[int] = []
        for dim in range(len(cover_intervals)):
            target_left = box[dim][0]
            idx = next(
                i for i, (l, r) in enumerate(cover_intervals[dim])
                if abs(l - target_left) < 1e-14
            )
            box_dim_indices.append(idx)
        cover_boxes.append(tuple(box_dim_indices))
        box_first_node.append(len(node_members))

        node_members.append(members)
        node_index = len(node_members) - 1
        for point_index in members:
            point_node_memberships[int(point_index)].append(node_index)
        node_centers.append(coords[members].mean(axis=0))

    node_centers_arr = np.vstack(node_centers)
    graph = np.full((len(node_members), len(node_members)), np.inf, dtype=float)
    np.fill_diagonal(graph, 0.0)
    for memberships in point_node_memberships:
        for i, j in combinations(memberships, 2):
            weight = float(np.linalg.norm(node_centers_arr[i] - node_centers_arr[j]))
            graph[i, j] = min(graph[i, j], weight)
            graph[j, i] = graph[i, j]

    return {
        "node_members": node_members,
        "node_centers": node_centers_arr,
        "point_node_memberships": point_node_memberships,
        "graph": graph,
        "cover_intervals": cover_intervals,
        "cover_boxes": cover_boxes,
        "box_first_node": box_first_node,
        "lens_dims": list(lens_dims),
        "cover_mode": str(cover_mode),
        "cover_core_fraction": float(cover_core_fraction),
        "cover_anchor_position": None if cover_anchor_position is None else int(cover_anchor_position),
    }


def mapper_nerve_distance(
    coords: np.ndarray,
    *,
    anchor_position: int = 0,
    lens_dims: Sequence[int] = (0, 1),
    n_covers: int = 6,
    overlap_fraction: float = 0.35,
    cover_mode: str = "standard",
    cover_core_fraction: float = 0.10,
    cluster_mode: str = "dbscan",
    dbscan_eps_quantile: float = 0.65,
    dbscan_min_samples: int = 3,
    fallback_unreachable: bool = True,
) -> Dict[str, np.ndarray]:
    """Approximate local geodesic distances with a Mapper nerve graph.

    The input ``coords`` is a local low-dimensional chart. Mapper covers a
    two-dimensional lens, clusters points inside each overlapping cover with
    DBSCAN, connects clusters that share samples, and runs Dijkstra over
    Euclidean distances between node centers.
    """
    nerve = build_mapper_nerve(
        coords,
        lens_dims=lens_dims,
        n_covers=n_covers,
        overlap_fraction=overlap_fraction,
        cover_mode=cover_mode,
        cover_core_fraction=cover_core_fraction,
        cover_anchor_position=int(anchor_position) if str(cover_mode) == "all-intersecting" else None,
        cluster_mode=cluster_mode,
        dbscan_eps_quantile=dbscan_eps_quantile,
        dbscan_min_samples=dbscan_min_samples,
    )
    node_centers_arr = nerve["node_centers"]
    point_node_memberships = nerve["point_node_memberships"]
    graph = nerve["graph"]

    point_node = np.empty(coords.shape[0], dtype=int)
    for point_index in range(coords.shape[0]):
        containing_nodes = np.array(point_node_memberships[point_index], dtype=int)
        candidate_nodes = containing_nodes if len(containing_nodes) > 0 else np.arange(len(node_centers_arr))
        center_distances = np.linalg.norm(node_centers_arr[candidate_nodes] - coords[point_index], axis=1)
        point_node[point_index] = int(candidate_nodes[np.argmin(center_distances)])

    anchor_node = int(point_node[int(anchor_position)])
    node_geodesic = dijkstra(graph, directed=False, indices=anchor_node)
    unreachable_nodes = ~np.isfinite(node_geodesic)
    node_reachable = np.isfinite(node_geodesic)
    n_components = connected_components(graph, directed=False, return_labels=False)
    if bool(fallback_unreachable) and unreachable_nodes.any():
        node_geodesic[unreachable_nodes] = np.linalg.norm(
            node_centers_arr[unreachable_nodes] - node_centers_arr[anchor_node],
            axis=1,
        )
    point_offsets = np.linalg.norm(coords - node_centers_arr[point_node], axis=1)
    anchor_offset = float(point_offsets[int(anchor_position)])
    point_distance = node_geodesic[point_node] + anchor_offset + point_offsets
    point_distance[int(anchor_position)] = 0.0

    return {
        "distance": point_distance.astype(float),
        "point_node": point_node.astype(int),
        "node_centers": node_centers_arr.astype(float),
        "node_geodesic": node_geodesic.astype(float),
        "graph": graph.astype(float),
        "node_reachable": node_reachable.astype(bool),
        "point_reachable": node_reachable[point_node].astype(bool),
        "n_connected_components": np.asarray(int(n_components), dtype=int),
        "fallback_unreachable": np.asarray(bool(fallback_unreachable), dtype=bool),
    }


def local_manifold_projection(
    X: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    anchor_index: int,
    fit_indices: np.ndarray,
    display_k: int,
    tangent_dim: int,
    ridge_rho: float,
    distance_method: str = "mapper-nerve",
    nerve_lens_dims: Sequence[int] = (0, 1),
    nerve_covers: int = 6,
    nerve_overlap: float = 0.35,
    nerve_cover_mode: str = "standard",
    nerve_cover_core_fraction: float = 0.10,
    nerve_cluster_mode: str = "dbscan",
    nerve_dbscan_eps_quantile: float = 0.65,
    nerve_dbscan_min_samples: int = 3,
    nerve_fallback_unreachable: bool = True,
) -> pd.DataFrame:
    fit_indices = np.asarray(fit_indices, dtype=int)
    display_indices = fit_indices[: int(display_k)]

    X_fit = X[fit_indices]
    local_mean = X_fit.mean(axis=0, keepdims=True)
    centered_fit = X_fit - local_mean
    _, _, vt = np.linalg.svd(centered_fit, full_matrices=False)
    q = min(int(tangent_dim), vt.shape[0])
    tangent_basis = vt[:q].T

    tangent_coords = centered_fit @ tangent_basis
    tangent_coords = tangent_coords - tangent_coords.mean(axis=0, keepdims=True)
    centered_scores = scores[fit_indices] - scores[fit_indices].mean()
    beta = np.linalg.solve(
        tangent_coords.T @ tangent_coords + float(ridge_rho) * np.eye(q),
        tangent_coords.T @ centered_scores,
    )
    score_direction = tangent_basis @ beta
    score_direction = score_direction / np.linalg.norm(score_direction)

    chart_indices = np.concatenate([[int(anchor_index)], display_indices])
    chart_displacements = X[chart_indices] - X[int(anchor_index)]
    chart_score_axis = chart_displacements @ score_direction
    chart_residual_displacements = chart_displacements - np.outer(chart_score_axis, score_direction)
    _, _, residual_vt = np.linalg.svd(chart_residual_displacements, full_matrices=False)
    residual_basis = residual_vt[:2].T
    chart_residual_coords = chart_residual_displacements @ residual_basis
    chart_coords = np.column_stack([chart_score_axis, chart_residual_coords])

    coords = chart_coords[1:]
    score_axis = coords[:, 0]
    residual_coords = coords[:, 1:]

    if distance_method == "mapper-nerve":
        nerve = mapper_nerve_distance(
            chart_coords,
            anchor_position=0,
            lens_dims=nerve_lens_dims,
            n_covers=int(nerve_covers),
            overlap_fraction=float(nerve_overlap),
            cover_mode=str(nerve_cover_mode),
            cover_core_fraction=float(nerve_cover_core_fraction),
            cluster_mode=str(nerve_cluster_mode),
            dbscan_eps_quantile=float(nerve_dbscan_eps_quantile),
            dbscan_min_samples=int(nerve_dbscan_min_samples),
            fallback_unreachable=bool(nerve_fallback_unreachable),
        )
        manifold_distance_by_index = dict(zip(chart_indices.tolist(), nerve["distance"].tolist()))
        manifold_distance = np.array([manifold_distance_by_index[int(i)] for i in display_indices], dtype=float)
        mapper_reachable_by_index = dict(zip(chart_indices.tolist(), nerve["point_reachable"].tolist()))
        mapper_node_by_index = dict(zip(chart_indices.tolist(), nerve["point_node"].tolist()))
        mapper_reachable = np.array([mapper_reachable_by_index[int(i)] for i in display_indices], dtype=bool)
        mapper_node = np.array([mapper_node_by_index[int(i)] for i in display_indices], dtype=int)
        mapper_n_components = int(nerve["n_connected_components"])
        mapper_reachable_node_fraction = float(nerve["node_reachable"].mean())
        mapper_fallback_node_count = int((~nerve["node_reachable"]).sum())
    elif distance_method == "chart-euclidean":
        manifold_distance = np.linalg.norm(residual_coords, axis=1)
        mapper_reachable = np.ones(len(display_indices), dtype=bool)
        mapper_node = np.full(len(display_indices), -1, dtype=int)
        mapper_n_components = 1
        mapper_reachable_node_fraction = 1.0
        mapper_fallback_node_count = 0
    else:
        raise ValueError(f"unknown manifold distance method: {distance_method}")

    return pd.DataFrame(
        {
            "embedding_rank": np.arange(1, len(display_indices) + 1),
            "index": display_indices.astype(int),
            "score": scores[display_indices].astype(float),
            "label": labels[display_indices].astype(int),
            "score_axis": score_axis.astype(float),
            "residual_axis_1": residual_coords[:, 0].astype(float),
            "residual_axis_2": residual_coords[:, 1].astype(float),
            "chart_distance": np.linalg.norm(coords, axis=1),
            "manifold_distance": manifold_distance,
            "mapper_reachable": mapper_reachable.astype(bool),
            "mapper_node": mapper_node.astype(int),
            "mapper_n_components": mapper_n_components,
            "mapper_reachable_node_fraction": mapper_reachable_node_fraction,
            "mapper_fallback_node_count": mapper_fallback_node_count,
        }
    )


def multi_anchor_local_manifold_projection(
    X: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    anchor_indices: np.ndarray,
    fit_indices: np.ndarray,
    tangent_dim: int,
    ridge_rho: float,
) -> pd.DataFrame:
    anchor_indices = np.asarray(anchor_indices, dtype=int)
    fit_indices = np.asarray(fit_indices, dtype=int)
    display_indices = np.unique(np.concatenate([anchor_indices, fit_indices]))

    X_fit = X[fit_indices]
    local_mean = X_fit.mean(axis=0, keepdims=True)
    centered_fit = X_fit - local_mean
    _, _, vt = np.linalg.svd(centered_fit, full_matrices=False)
    q = min(int(tangent_dim), vt.shape[0])
    tangent_basis = vt[:q].T

    tangent_coords = centered_fit @ tangent_basis
    tangent_coords = tangent_coords - tangent_coords.mean(axis=0, keepdims=True)
    centered_scores = scores[fit_indices] - scores[fit_indices].mean()
    beta = np.linalg.solve(
        tangent_coords.T @ tangent_coords + float(ridge_rho) * np.eye(q),
        tangent_coords.T @ centered_scores,
    )
    score_direction = tangent_basis @ beta
    score_direction = score_direction / np.linalg.norm(score_direction)

    chart_displacements = X[display_indices] - local_mean
    chart_score_axis = chart_displacements @ score_direction
    chart_residual_displacements = chart_displacements - np.outer(chart_score_axis, score_direction)
    _, _, residual_vt = np.linalg.svd(chart_residual_displacements, full_matrices=False)
    residual_basis = residual_vt[:2].T
    chart_residual_coords = chart_residual_displacements @ residual_basis

    anchor_label_by_index = {
        int(anchor_index): f"A{rank + 1}"
        for rank, anchor_index in enumerate(anchor_indices)
    }
    is_anchor = np.isin(display_indices, anchor_indices)

    return pd.DataFrame(
        {
            "embedding_rank": np.arange(len(display_indices), dtype=int),
            "index": display_indices.astype(int),
            "score": scores[display_indices].astype(float),
            "label": labels[display_indices].astype(int),
            "score_axis": chart_score_axis.astype(float),
            "residual_axis_1": chart_residual_coords[:, 0].astype(float),
            "residual_axis_2": chart_residual_coords[:, 1].astype(float),
            "chart_distance": np.linalg.norm(
                np.column_stack([chart_score_axis, chart_residual_coords]),
                axis=1,
            ),
            "is_anchor": is_anchor.astype(bool),
            "anchor_label": [
                anchor_label_by_index[int(index)] if int(index) in anchor_label_by_index else ""
                for index in display_indices
            ],
        }
    )


def local_manifold_chart(
    X: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    *,
    model_name: str,
    anchor_index: int,
    fit_k: int,
    tangent_dim: int,
    scenario: str,
) -> pd.DataFrame:
    if scenario == "mix-cancer":
        nn = NearestNeighbors(n_neighbors=int(fit_k) + 1, metric="cosine").fit(X)
        neighbors = nn.kneighbors(X[int(anchor_index) : int(anchor_index) + 1], return_distance=False)[0]
        fit_indices = neighbors[neighbors != int(anchor_index)][: int(fit_k)]
    elif scenario == "cancer-specific":
        anchor_cancer = str(meta.loc[int(anchor_index), "cancer"])
        cancer_indices = np.array(list(meta.groupby("cancer").groups[anchor_cancer]), dtype=int)
        nn = NearestNeighbors(n_neighbors=min(int(fit_k) + 1, len(cancer_indices)), metric="cosine").fit(X[cancer_indices])
        neighbors = cancer_indices[nn.kneighbors(X[int(anchor_index) : int(anchor_index) + 1], return_distance=False)[0]]
        fit_indices = neighbors[neighbors != int(anchor_index)][: int(fit_k)]
    else:
        raise ValueError(f"unknown manifold chart scenario: {scenario}")

    X_fit = X[fit_indices]
    local_mean = X_fit.mean(axis=0, keepdims=True)
    centered_fit = X_fit - local_mean
    _, _, vt = np.linalg.svd(centered_fit, full_matrices=False)
    q = min(int(tangent_dim), vt.shape[0])
    chart_basis = vt[: min(2, q)].T
    if chart_basis.shape[1] != 2:
        raise ValueError(f"contour chart requires two local tangent axes, got {chart_basis.shape[1]}")

    display_indices = np.concatenate([[int(anchor_index)], fit_indices])
    chart_coords = (X[display_indices] - X[int(anchor_index)]) @ chart_basis
    embedding_rank = np.concatenate([[0], np.arange(1, len(fit_indices) + 1)])

    return pd.DataFrame(
        {
            "model": model_name,
            "scenario": scenario,
            "anchor_index": int(anchor_index),
            "embedding_rank": embedding_rank.astype(int),
            "index": display_indices.astype(int),
            "m1": chart_coords[:, 0].astype(float),
            "m2": chart_coords[:, 1].astype(float),
            "score": scores[display_indices].astype(float),
            "label": labels[display_indices].astype(int),
            "cancer": meta.cancer.iloc[display_indices].astype(str).to_numpy(),
            "is_anchor": display_indices == int(anchor_index),
        }
    )


def multi_anchor_manifold_chart(
    X: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    *,
    model_name: str,
    anchor_indices: np.ndarray,
    fit_k: int,
    tangent_dim: int,
    scenario: str,
) -> pd.DataFrame:
    anchor_indices = np.asarray(anchor_indices, dtype=int)
    fit_parts: list[np.ndarray] = []

    if scenario == "mix-cancer":
        nn = NearestNeighbors(n_neighbors=int(fit_k) + 1, metric="cosine").fit(X)
        for anchor_index in anchor_indices:
            neighbors = nn.kneighbors(X[int(anchor_index) : int(anchor_index) + 1], return_distance=False)[0]
            fit_parts.append(neighbors[neighbors != int(anchor_index)][: int(fit_k)])
    elif scenario == "cancer-specific":
        for anchor_index in anchor_indices:
            anchor_cancer = str(meta.loc[int(anchor_index), "cancer"])
            cancer_indices = np.array(list(meta.groupby("cancer").groups[anchor_cancer]), dtype=int)
            nn = NearestNeighbors(n_neighbors=min(int(fit_k) + 1, len(cancer_indices)), metric="cosine").fit(
                X[cancer_indices]
            )
            neighbors = cancer_indices[
                nn.kneighbors(X[int(anchor_index) : int(anchor_index) + 1], return_distance=False)[0]
            ]
            fit_parts.append(neighbors[neighbors != int(anchor_index)][: int(fit_k)])
    else:
        raise ValueError(f"unknown manifold chart scenario: {scenario}")

    fit_indices = np.unique(np.concatenate(fit_parts))
    display_indices = np.unique(np.concatenate([anchor_indices, fit_indices]))
    local_mean = X[fit_indices].mean(axis=0, keepdims=True)
    centered_fit = X[fit_indices] - local_mean
    _, _, vt = np.linalg.svd(centered_fit, full_matrices=False)
    q = min(int(tangent_dim), vt.shape[0])
    chart_basis = vt[: min(2, q)].T
    if chart_basis.shape[1] != 2:
        raise ValueError(f"contour chart requires two local tangent axes, got {chart_basis.shape[1]}")

    chart_coords = (X[display_indices] - local_mean) @ chart_basis
    anchor_position = {int(anchor_index): i + 1 for i, anchor_index in enumerate(anchor_indices)}
    is_anchor = np.isin(display_indices, anchor_indices)
    anchor_label = np.array(
        [f"A{anchor_position[int(index)]}" if int(index) in anchor_position else "" for index in display_indices],
        dtype=object,
    )
    embedding_rank = np.arange(len(display_indices), dtype=int)

    return pd.DataFrame(
        {
            "model": model_name,
            "scenario": scenario,
            "anchor_index": -1,
            "embedding_rank": embedding_rank.astype(int),
            "index": display_indices.astype(int),
            "m1": chart_coords[:, 0].astype(float),
            "m2": chart_coords[:, 1].astype(float),
            "score": scores[display_indices].astype(float),
            "label": labels[display_indices].astype(int),
            "cancer": meta.cancer.iloc[display_indices].astype(str).to_numpy(),
            "is_anchor": is_anchor.astype(bool),
            "anchor_label": anchor_label,
        }
    )


def select_high_sl_anchors(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    high_score_pool: int,
    top_positive: int,
    top_score: int,
    max_anchors: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(labels == 1)
    high_score_indices = np.argsort(scores)[-int(high_score_pool) :]
    anchor_indices = np.unique(
        np.concatenate(
            [
                positive_indices[np.argsort(scores[positive_indices])[-int(top_positive) :]],
                high_score_indices[-int(top_score) :],
            ]
        )
    )
    if len(anchor_indices) > int(max_anchors):
        anchor_indices = rng.choice(anchor_indices, size=int(max_anchors), replace=False)
    return np.sort(anchor_indices)


def manifold_order_rows(
    X: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    *,
    model_name: str,
    scenario: str,
    anchor_index: int,
    anchor_cancer: str,
    fit_indices: np.ndarray,
    tangent_dim: int,
    ridge_rho: float,
    ks: Sequence[int],
    distance_method: str,
    nerve_lens_dims: Sequence[int],
    nerve_covers: int,
    nerve_overlap: float,
    nerve_cover_mode: str,
    nerve_cover_core_fraction: float,
    nerve_cluster_mode: str,
    nerve_dbscan_eps_quantile: float,
    nerve_dbscan_min_samples: int,
    nerve_fallback_unreachable: bool,
) -> list[dict]:
    manifold = local_manifold_projection(
        X,
        scores,
        labels,
        anchor_index=anchor_index,
        fit_indices=fit_indices,
        display_k=max(ks),
        tangent_dim=tangent_dim,
        ridge_rho=ridge_rho,
        distance_method=distance_method,
        nerve_lens_dims=nerve_lens_dims,
        nerve_covers=nerve_covers,
        nerve_overlap=nerve_overlap,
        nerve_cover_mode=nerve_cover_mode,
        nerve_cover_core_fraction=nerve_cover_core_fraction,
        nerve_cluster_mode=nerve_cluster_mode,
        nerve_dbscan_eps_quantile=nerve_dbscan_eps_quantile,
        nerve_dbscan_min_samples=nerve_dbscan_min_samples,
        nerve_fallback_unreachable=nerve_fallback_unreachable,
    )
    ordered_indices = manifold.sort_values("manifold_distance")["index"].to_numpy()
    rows = []
    for k in ks:
        neighbor_indices = ordered_indices[: int(k)]
        rows.append(
            {
                "model": model_name,
                "scenario": scenario,
                "anchor_index": int(anchor_index),
                "cancer": anchor_cancer,
                "k": int(k),
                "rho_manifold_score": ordered_spearman(scores[neighbor_indices]),
                "rho_manifold_label": ordered_spearman(labels[neighbor_indices]),
                "mean_score": float(scores[neighbor_indices].mean()),
                "median_score": float(np.median(scores[neighbor_indices])),
                "positive_fraction": float(labels[neighbor_indices].mean()),
                "mapper_reachable_fraction": float(manifold["mapper_reachable"].mean()),
                "mapper_n_components": int(manifold["mapper_n_components"].iloc[0]),
                "mapper_fallback_node_count": int(manifold["mapper_fallback_node_count"].iloc[0]),
                "same_cancer_fraction": float(
                    (meta.cancer.iloc[neighbor_indices].astype(str).to_numpy() == str(anchor_cancer)).mean()
                ),
            }
        )
    return rows


def multi_anchor_manifold_ordering(
    X: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    *,
    model_name: str,
    anchor_indices: np.ndarray,
    fit_k: int,
    tangent_dim: int,
    ridge_rho: float,
    ks: Sequence[int],
    distance_method: str = "mapper-nerve",
    nerve_lens_dims: Sequence[int] = (0, 1),
    nerve_covers: int = 6,
    nerve_overlap: float = 0.35,
    nerve_cover_mode: str = "standard",
    nerve_cover_core_fraction: float = 0.10,
    nerve_cluster_mode: str = "dbscan",
    nerve_dbscan_eps_quantile: float = 0.65,
    nerve_dbscan_min_samples: int = 3,
    nerve_fallback_unreachable: bool = True,
) -> pd.DataFrame:
    rows = []
    nn_global = NearestNeighbors(n_neighbors=int(fit_k) + 1, metric="cosine").fit(X)
    global_neighbors = nn_global.kneighbors(X[anchor_indices], return_distance=False)

    for row_index, anchor_index in tqdm(enumerate(anchor_indices), total=len(anchor_indices), desc="multi-anchor ordering"):
        anchor_cancer = str(meta.loc[int(anchor_index), "cancer"])
        neighbors = global_neighbors[row_index]
        fit_indices = neighbors[neighbors != anchor_index][: int(fit_k)]
        rows.extend(
            manifold_order_rows(
                X,
                scores,
                labels,
                meta,
                model_name=model_name,
                scenario="mix-cancer",
                anchor_index=int(anchor_index),
                anchor_cancer=anchor_cancer,
                fit_indices=fit_indices,
                tangent_dim=tangent_dim,
                ridge_rho=ridge_rho,
                ks=ks,
                distance_method=distance_method,
                nerve_lens_dims=nerve_lens_dims,
                nerve_covers=nerve_covers,
                nerve_overlap=nerve_overlap,
                nerve_cover_mode=nerve_cover_mode,
                nerve_cover_core_fraction=nerve_cover_core_fraction,
                nerve_cluster_mode=nerve_cluster_mode,
                nerve_dbscan_eps_quantile=nerve_dbscan_eps_quantile,
                nerve_dbscan_min_samples=nerve_dbscan_min_samples,
                nerve_fallback_unreachable=nerve_fallback_unreachable,
            )
        )
    anchor_items = pd.Series(np.arange(len(anchor_indices))).groupby(
        meta.cancer.iloc[anchor_indices].to_numpy()
    ).groups.items()
    for anchor_cancer, grouped_anchor_positions in tqdm(anchor_items, total=len(anchor_items), desc="cancer-specific ordering"):
        cancer_anchor_indices = anchor_indices[np.array(list(grouped_anchor_positions), dtype=int)]
        cancer_indices = np.array(list(meta.groupby("cancer").groups[anchor_cancer]), dtype=int)
        nn_cancer = NearestNeighbors(
            n_neighbors=min(int(fit_k) + 1, len(cancer_indices)),
            metric="cosine",
        ).fit(X[cancer_indices])
        cancer_neighbors = nn_cancer.kneighbors(X[cancer_anchor_indices], return_distance=False)
        for row_index, anchor_index in enumerate(cancer_anchor_indices):
            neighbors = cancer_indices[cancer_neighbors[row_index]]
            fit_indices = neighbors[neighbors != anchor_index][: int(fit_k)]
            scope_rows = manifold_order_rows(
                X,
                scores,
                labels,
                meta,
                model_name=model_name,
                scenario="cancer-specific",
                anchor_index=int(anchor_index),
                anchor_cancer=str(anchor_cancer),
                fit_indices=fit_indices,
                tangent_dim=tangent_dim,
                ridge_rho=ridge_rho,
                ks=ks,
                distance_method=distance_method,
                nerve_lens_dims=nerve_lens_dims,
                nerve_covers=nerve_covers,
                nerve_overlap=nerve_overlap,
                nerve_cover_mode=nerve_cover_mode,
                nerve_cover_core_fraction=nerve_cover_core_fraction,
                nerve_cluster_mode=nerve_cluster_mode,
                nerve_dbscan_eps_quantile=nerve_dbscan_eps_quantile,
                nerve_dbscan_min_samples=nerve_dbscan_min_samples,
                nerve_fallback_unreachable=nerve_fallback_unreachable,
            )
            for row in scope_rows:
                row["same_cancer_fraction"] = 1.0
            rows.extend(scope_rows)
    return pd.DataFrame(rows)


def summarize_multi_anchor_ordering(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["model", "scenario", "k"]).agg(
        n=("rho_manifold_score", "count"),
        label_valid=("rho_manifold_label", "count"),
        mean_rho_manifold_score=("rho_manifold_score", "mean"),
        median_rho_manifold_score=("rho_manifold_score", "median"),
        mean_rho_manifold_label=("rho_manifold_label", "mean"),
        median_rho_manifold_label=("rho_manifold_label", "median"),
        mean_score=("mean_score", "mean"),
        median_score=("median_score", "median"),
        mean_positive_fraction=("positive_fraction", "mean"),
        median_positive_fraction=("positive_fraction", "median"),
        mean_mapper_reachable_fraction=("mapper_reachable_fraction", "mean"),
        median_mapper_n_components=("mapper_n_components", "median"),
        median_mapper_fallback_node_count=("mapper_fallback_node_count", "median"),
        mean_same_cancer_fraction=("same_cancer_fraction", "mean"),
    ).reset_index()
