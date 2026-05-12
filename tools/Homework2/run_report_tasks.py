"""Generate Homework 2 report figures and tables for Q9, Q13, Q14, and Q16."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.sparse import diags
from scipy.sparse.linalg import cg, gmres, spsolve

TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_ROOT.parents[1]
ASSET_ROOT = PROJECT_ROOT / "report_assets" / "Homework2"
PINN_ROOT = PROJECT_ROOT / "experiments" / "PINN"
for path in (TOOL_ROOT / "src", PROJECT_ROOT, PINN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dl_optimizers import optimize
from io_utils import write_csv, write_json
from problems import get_assignment_problem
from experiments.PINN.lib_pinns.burgers.networks import BurgersNet
from experiments.PINN.lib_pinns.burgers.run_test import run_test


RAW = ASSET_ROOT / "raw"
TABLES = ASSET_ROOT / "tables"
FIGURES = ASSET_ROOT / "figures"

METHOD_COLORS = {
    "standard": "#1f77b4",
    "pcgrad": "#ff7f0e",
    "config": "#2ca02c",
    "mconfig": "#d62728",
}
EPOCH_MARKERS = {0: "o", 15000: "^", 30000: "s"}


def flatten_state_dict(state_dict: dict[str, torch.Tensor]) -> np.ndarray:
    chunks = []
    for value in state_dict.values():
        if not torch.is_floating_point(value):
            continue
        chunks.append(value.detach().cpu().reshape(-1).numpy())
    return np.concatenate(chunks, axis=0)


def load_state_points(run_dir: Path) -> list[dict]:
    network = torch.load(run_dir / "network_structure.pt", map_location="cpu", weights_only=False)
    checkpoints_dir = run_dir / "checkpoints"
    points = [
        {"epoch": 0, "state_dict": network.state_dict()},
        {
            "epoch": 15000,
            "state_dict": torch.load(checkpoints_dir / "checkpoint_15000.pt", map_location="cpu", weights_only=False)["network"],
        },
        {
            "epoch": 30000,
            "state_dict": torch.load(checkpoints_dir / "checkpoint_30000.pt", map_location="cpu", weights_only=False)["network"],
        },
    ]
    return points


def load_final_network(run_dir: Path, device: str) -> BurgersNet:
    network = BurgersNet().to(device)
    state_dict = torch.load(run_dir / "trained_network_weights.pt", map_location=device, weights_only=False)
    network.load_state_dict(state_dict)
    network.eval()
    return network


def run_q13() -> dict:
    problem_keys = ["q1", "q6", "q7"]
    x0_map = {
        "q1": np.array([0.0, 0.0], dtype=float),
        "q6": np.array([0.0, 0.0], dtype=float),
        "q7": np.array([0.0, 0.0], dtype=float),
    }
    config_map = {
        "q1": {
            "SGD": {"lr": 0.22},
            "Momentum": {"lr": 0.16, "momentum": 0.9},
            "RMSprop": {"lr": 0.08, "rho": 0.9},
            "Adam": {"lr": 0.14, "momentum": 0.9, "beta2": 0.999},
        },
        "q6": {
            "SGD": {"lr": 0.18},
            "Momentum": {"lr": 0.12, "momentum": 0.9},
            "RMSprop": {"lr": 0.06, "rho": 0.9},
            "Adam": {"lr": 0.14, "momentum": 0.9, "beta2": 0.999},
        },
        "q7": {
            "SGD": {"lr": 0.25},
            "Momentum": {"lr": 0.18, "momentum": 0.9},
            "RMSprop": {"lr": 0.08, "rho": 0.9},
            "Adam": {"lr": 0.18, "momentum": 0.9, "beta2": 0.999},
        },
    }

    all_results = []
    summary_rows = []
    for key in problem_keys:
        problem = get_assignment_problem(key)
        x0 = x0_map[key]
        for method, kwargs in config_map[key].items():
            result = optimize(problem.f, problem.grad, x0, method=method, steps=80, **kwargs)
            result["problem"] = problem.name
            all_results.append(result)
            final_x = np.asarray(result["final_x"], dtype=float)
            summary_rows.append(
                {
                    "problem": problem.name,
                    "optimizer": method,
                    "iterations": result["iterations"],
                    "final_x": "(" + ", ".join(f"{value:.6g}" for value in final_x) + ")",
                    "final_f": f"{float(result['final_f']):.10g}",
                    "value_error": f"{abs(float(result['final_f']) - float(problem.exact_value)):.10g}",
                    "final_grad_norm": f"{float(result['final_grad_norm']):.10g}",
                    "runtime_sec": f"{float(result['runtime_sec']):.6f}",
                }
            )

    write_json(RAW / "q13_dl_optimizers.json", all_results)
    write_csv(
        TABLES / "q13_dl_optimizers_summary.csv",
        summary_rows,
        ["problem", "optimizer", "iterations", "final_x", "final_f", "value_error", "final_grad_norm", "runtime_sec"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8), dpi=180)
    for ax, key in zip(axes, problem_keys):
        problem = get_assignment_problem(key)
        for result in [item for item in all_results if item["problem"] == problem.name]:
            f_values = np.array([entry["f"] for entry in result["history"]], dtype=float)
            gap = np.maximum(f_values - float(problem.exact_value), 1e-12)
            ax.semilogy(gap, label=result["name"], linewidth=1.6)
        ax.set_title(problem.name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective gap")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q13_objective_gaps.png", bbox_inches="tight")
    plt.close(fig)

    q6 = get_assignment_problem("q6")
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=180)
    xs = np.linspace(-0.5, 3.5, 240)
    ys = np.linspace(-0.5, 2.5, 240)
    xx, yy = np.meshgrid(xs, ys)
    zz = xx**2 + 4.0 * yy**2 - 4.0 * xx - 8.0 * yy
    ax.contour(xx, yy, zz, levels=16, cmap="Greys", alpha=0.65)
    for result in [item for item in all_results if item["problem"] == q6.name]:
        path = np.array([entry["x"] for entry in result["history"]], dtype=float)
        ax.plot(path[:, 0], path[:, 1], marker="o", markersize=2.0, linewidth=1.3, label=result["name"])
    ax.scatter([2.0], [1.0], color="black", marker="*", s=90, label="Optimal point")
    ax.set_title("Q13 trajectories on Q6")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "q13_q6_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    return {"results": all_results, "summary_rows": summary_rows}


def run_q9_q14(device: str = "cpu") -> dict:
    run_root = PROJECT_ROOT / "PINN_trained" / "Burgers" / "Report"
    methods = ["standard", "pcgrad", "config", "mconfig"]
    embeddings = []
    vectors = []
    pair_rows = []

    simulation_data = np.load(PINN_ROOT / "data" / "burgers" / "simulation_data.npy")

    for method in methods:
        method_root = run_root / f"{method}_30000_full"
        runs = sorted([path for path in method_root.iterdir() if path.is_dir()])
        for run_index, run_dir in enumerate(runs, start=1):
            for point in load_state_points(run_dir):
                vector = flatten_state_dict(point["state_dict"])
                vectors.append(vector)
                embeddings.append(
                    {
                        "method": method,
                        "run_name": run_dir.name,
                        "run_index": run_index,
                        "epoch": point["epoch"],
                        "vector": vector,
                    }
                )

    matrix = np.stack([item["vector"] for item in embeddings], axis=0)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    variance_ratio = variance / variance.sum()
    scores = centered @ vt[:3].T
    for item, score in zip(embeddings, scores):
        item["pc1"] = float(score[0])
        item["pc2"] = float(score[1])
        item["pc3"] = float(score[2])
        del item["vector"]

    write_json(
        RAW / "q14_parameter_trajectory_pca.json",
        {
            "explained_variance_ratio": variance_ratio[:8].tolist(),
            "samples": embeddings,
        },
    )
    write_csv(
        TABLES / "q14_parameter_trajectory_pca.csv",
        embeddings,
        ["method", "run_name", "run_index", "epoch", "pc1", "pc2", "pc3"],
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=180)
    method_labeled = set()
    for method in methods:
        method_rows = [item for item in embeddings if item["method"] == method]
        for run_name in sorted({item["run_name"] for item in method_rows}):
            path_rows = sorted([item for item in method_rows if item["run_name"] == run_name], key=lambda item: item["epoch"])
            xs = [item["pc1"] for item in path_rows]
            ys = [item["pc2"] for item in path_rows]
            ax.plot(xs, ys, color=METHOD_COLORS[method], linewidth=1.1, alpha=0.7)
            for item in path_rows:
                label = method if method not in method_labeled and item["epoch"] == 0 else None
                ax.scatter(item["pc1"], item["pc2"], color=METHOD_COLORS[method], marker=EPOCH_MARKERS[item["epoch"]], s=36, label=label)
                if label is not None:
                    method_labeled.add(method)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Q14 parameter trajectories on Burgers PINN")
    ax.grid(alpha=0.25)
    method_handles, method_labels = ax.get_legend_handles_labels()
    epoch_handles = [
        Line2D([0], [0], marker=EPOCH_MARKERS[epoch], color="black", linestyle="None", markersize=6, label=f"epoch={epoch}")
        for epoch in (0, 15000, 30000)
    ]
    legend_methods = ax.legend(method_handles, method_labels, frameon=False, fontsize=8, loc="upper left")
    ax.add_artist(legend_methods)
    ax.legend(handles=epoch_handles, frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "q14_parameter_trajectory_pca.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=180)
    top_k = min(6, variance_ratio.size)
    ax.bar(np.arange(1, top_k + 1), variance_ratio[:top_k], color="#4c72b0")
    ax.plot(np.arange(1, top_k + 1), np.cumsum(variance_ratio[:top_k]), color="#dd8452", marker="o", linewidth=1.5)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Q14 explained variance")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "q14_explained_variance.png", bbox_inches="tight")
    plt.close(fig)

    standard_runs = sorted([path for path in (run_root / "standard_30000_full").iterdir() if path.is_dir()])
    run_payload = []
    for run_dir in standard_runs:
        network = load_final_network(run_dir, device)
        mse_value, _mse_grid, prediction = run_test(network, simulation_data, device=device)
        state_dict = torch.load(run_dir / "trained_network_weights.pt", map_location="cpu", weights_only=False)
        run_payload.append(
            {
                "run_name": run_dir.name,
                "vector": flatten_state_dict(state_dict),
                "prediction": prediction,
                "test_mse": float(mse_value),
            }
        )

    for left, right in combinations(run_payload, 2):
        vector_left = left["vector"]
        vector_right = right["vector"]
        prediction_gap = float(np.mean((left["prediction"] - right["prediction"]) ** 2))
        cosine = float(np.dot(vector_left, vector_right) / (np.linalg.norm(vector_left) * np.linalg.norm(vector_right)))
        pair_rows.append(
            {
                "run_a": left["run_name"],
                "run_b": right["run_name"],
                "parameter_l2_distance": f"{float(np.linalg.norm(vector_left - vector_right)):.10g}",
                "parameter_cosine_similarity": f"{cosine:.10g}",
                "prediction_mse": f"{prediction_gap:.10g}",
                "test_mse_run_a": f"{left['test_mse']:.10g}",
                "test_mse_run_b": f"{right['test_mse']:.10g}",
            }
        )

    write_csv(
        TABLES / "q9_random_initialization_pairs.csv",
        pair_rows,
        [
            "run_a",
            "run_b",
            "parameter_l2_distance",
            "parameter_cosine_similarity",
            "prediction_mse",
            "test_mse_run_a",
            "test_mse_run_b",
        ],
    )
    write_json(RAW / "q9_random_initialization_pairs.json", pair_rows)

    return {
        "explained_variance_ratio": variance_ratio[:8].tolist(),
        "pair_rows": pair_rows,
    }


def run_q16() -> dict:
    n = 80
    b = np.ones(n)

    spd_matrix = diags([-1.0, 2.0, -1.0], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    nonsym_matrix = diags([-1.0, 2.0, -0.5], offsets=[-1, 0, 1], shape=(n, n), format="csr")

    cg_residuals = []
    gmres_spd_residuals = []
    gmres_nonsym_residuals = []

    def cg_callback(xk):
        residual = b - spd_matrix @ xk
        cg_residuals.append(float(np.linalg.norm(residual)))

    def gmres_spd_callback(residual_norm):
        gmres_spd_residuals.append(float(residual_norm))

    def gmres_nonsym_callback(residual_norm):
        gmres_nonsym_residuals.append(float(residual_norm))

    x_cg, info_cg = cg(spd_matrix, b, rtol=1e-10, atol=0.0, maxiter=500, callback=cg_callback)
    x_gmres_spd, info_gmres_spd = gmres(
        spd_matrix,
        b,
        rtol=1e-10,
        atol=0.0,
        restart=20,
        maxiter=500,
        callback=gmres_spd_callback,
        callback_type="pr_norm",
    )
    x_gmres_nonsym, info_gmres_nonsym = gmres(
        nonsym_matrix,
        b,
        rtol=1e-10,
        atol=0.0,
        restart=20,
        maxiter=500,
        callback=gmres_nonsym_callback,
        callback_type="pr_norm",
    )

    x_spd_exact = spsolve(spd_matrix, b)
    x_nonsym_exact = spsolve(nonsym_matrix, b)

    summary_rows = [
        {
            "matrix": "SPD tridiagonal",
            "method": "CG",
            "info": info_cg,
            "iterations": len(cg_residuals),
            "final_residual": f"{cg_residuals[-1]:.10g}",
            "solution_error": f"{float(np.linalg.norm(x_cg - x_spd_exact)):.10g}",
        },
        {
            "matrix": "SPD tridiagonal",
            "method": "GMRES",
            "info": info_gmres_spd,
            "iterations": len(gmres_spd_residuals),
            "final_residual": f"{gmres_spd_residuals[-1]:.10g}",
            "solution_error": f"{float(np.linalg.norm(x_gmres_spd - x_spd_exact)):.10g}",
        },
        {
            "matrix": "Nonsymmetric tridiagonal",
            "method": "GMRES",
            "info": info_gmres_nonsym,
            "iterations": len(gmres_nonsym_residuals),
            "final_residual": f"{gmres_nonsym_residuals[-1]:.10g}",
            "solution_error": f"{float(np.linalg.norm(x_gmres_nonsym - x_nonsym_exact)):.10g}",
        },
    ]

    write_csv(
        TABLES / "q16_gmres_summary.csv",
        summary_rows,
        ["matrix", "method", "info", "iterations", "final_residual", "solution_error"],
    )
    write_json(
        RAW / "q16_gmres_summary.json",
        {
            "summary_rows": summary_rows,
            "cg_residuals": cg_residuals,
            "gmres_spd_residuals": gmres_spd_residuals,
            "gmres_nonsym_residuals": gmres_nonsym_residuals,
        },
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=180)
    ax.semilogy(range(1, len(cg_residuals) + 1), cg_residuals, label="CG on SPD", linewidth=1.6)
    ax.semilogy(range(1, len(gmres_spd_residuals) + 1), gmres_spd_residuals, label="GMRES on SPD", linewidth=1.6)
    ax.semilogy(range(1, len(gmres_nonsym_residuals) + 1), gmres_nonsym_residuals, label="GMRES on nonsymmetric", linewidth=1.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual norm")
    ax.set_title("Q16 Krylov residual curves")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "q16_krylov_residuals.png", bbox_inches="tight")
    plt.close(fig)

    return {"summary_rows": summary_rows}


def main() -> None:
    for path in (RAW, TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)

    q13_payload = run_q13()
    q9_q14_payload = run_q9_q14(device="cpu")
    q16_payload = run_q16()

    print("Homework 2 report tasks completed.")
    print(f"Q13 rows: {len(q13_payload['summary_rows'])}")
    print(f"Q9 pairs: {len(q9_q14_payload['pair_rows'])}")
    print(f"Q16 rows: {len(q16_payload['summary_rows'])}")
    print(f"Artifacts written to: {ASSET_ROOT}")


if __name__ == "__main__":
    main()