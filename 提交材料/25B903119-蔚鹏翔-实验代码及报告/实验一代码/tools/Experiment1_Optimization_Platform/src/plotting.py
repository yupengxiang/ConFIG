"""Plotting utilities for experiment 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_functions import rosenbrock


def _history_array(result: dict) -> np.ndarray:
    return np.array([row["x"] for row in result["history"]], dtype=float)


STYLE = {
    "Gradient Descent": {"color": "#4477AA", "marker": "o"},
    "Armijo GD": {"color": "#66AA55", "marker": "s"},
    "Adam": {"color": "#CC6677", "marker": "^"},
    "Newton": {"color": "#AA3377", "marker": "D"},
    "Damped Newton": {"color": "#EE7733", "marker": "P"},
    "BFGS": {"color": "#228833", "marker": "X"},
    "PSO": {"color": "#BBBB44", "marker": "v"},
}


def _plot_contour_background(ax) -> None:
    x = np.linspace(-2.0, 2.0, 420)
    y = np.linspace(-1.0, 3.0, 420)
    xx, yy = np.meshgrid(x, y)
    zz = (1.0 - xx) ** 2 + 100.0 * (yy - xx**2) ** 2
    levels = np.logspace(-1, 3.5, 32)
    norm = matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1])
    ax.contour(xx, yy, zz, levels=levels, norm=norm, cmap="viridis", linewidths=0.8, alpha=0.82)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.0, 3.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_result_path(ax, result: dict) -> None:
    pts = _history_array(result)
    if len(pts) > 500:
        idx = np.unique(np.linspace(0, len(pts) - 1, 500).astype(int))
        pts = pts[idx]
    style = STYLE.get(result["name"], {"color": "#333333", "marker": "o"})
    markevery = max(1, len(pts) // 18)
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=style["color"],
        linewidth=1.45,
        marker=style["marker"],
        markersize=4.0,
        markevery=markevery,
        alpha=0.92,
        label=result["name"],
        zorder=3,
    )


def plot_rosenbrock_trajectories(results: list[dict], baseline: dict, path: Path) -> None:
    bx, by = baseline["x"]
    groups = [
        ("First-order methods", {"Gradient Descent", "Armijo GD", "Adam"}),
        ("Second-order / quasi-Newton methods", {"Newton", "Damped Newton", "BFGS"}),
        ("Derivative-free method", {"PSO"}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4), sharex=True, sharey=True)
    for ax, (title, names) in zip(axes, groups):
        _plot_contour_background(ax)
        for result in results:
            if result["name"] in names:
                _plot_result_path(ax, result)
        ax.scatter(
            [bx],
            [by],
            marker="*",
            s=360,
            color="red",
            edgecolor="black",
            linewidth=1.0,
            label="JuMP/Ipopt optimum",
            zorder=20,
        )
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.93)
    fig.suptitle("Rosenbrock optimization trajectories by algorithm family", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rosenbrock_convergence(results: list[dict], f_star: float, path: Path) -> None:
    max_len = max(len(result["history"]) for result in results)
    plt.figure(figsize=(9.2, 5.8))
    for result in results:
        vals = np.array([row["f"] for row in result["history"]], dtype=float)
        gap = np.maximum(vals - f_star, 1e-16)
        if len(gap) < max_len:
            gap = np.pad(gap, (0, max_len - len(gap)), mode="edge")
        style = STYLE.get(result["name"], {"color": "#333333", "marker": "o"})
        plt.semilogy(
            np.arange(max_len),
            gap,
            label=result["name"],
            linewidth=1.7,
            color=style["color"],
        )
    plt.xlabel("Iteration")
    plt.ylabel("f(x_k) - f*")
    plt.title("Rosenbrock convergence on a shared iteration axis")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()


def plot_algorithm_bar(summary_rows: list[dict], path: Path) -> None:
    names = [row["algorithm"] for row in summary_rows]
    final_f = np.array([float(row["final_f"]) for row in summary_rows])
    iterations = np.array([float(row["iterations"]) for row in summary_rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].bar(names, np.maximum(final_f, 1e-16), color="#4477AA")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Final objective")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Final objective")
    axes[1].bar(names, np.maximum(iterations, 1.0), color="#66AA55")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Iterations")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Iteration count (log scale)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_admm(admm_result: dict, residual_path: Path, recovery_path: Path) -> None:
    history = admm_result["history"]
    obj = np.array([h["objective"] for h in history], dtype=float)
    primal = np.array([h["primal_residual"] for h in history], dtype=float)
    dual = np.array([h["dual_residual"] for h in history], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].semilogy(np.maximum(obj - np.min(obj) + 1e-12, 1e-12), color="#4477AA")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective gap")
    axes[0].set_title("ADMM-Lasso objective gap")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.35)
    axes[1].semilogy(np.maximum(primal, 1e-16), label="Primal residual")
    axes[1].semilogy(np.maximum(dual, 1e-16), label="Dual residual")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("ADMM residuals")
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    residual_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(residual_path, dpi=220)
    plt.close(fig)

    x_true = np.array(admm_result["x_true"], dtype=float)
    x_hat = np.array(admm_result["final_x"], dtype=float)
    idx = np.arange(len(x_true))
    plt.figure(figsize=(11.2, 4.9))
    markerline, stemlines, _ = plt.stem(idx, x_true, linefmt="#4477AA", markerfmt="o", basefmt=" ")
    plt.setp(stemlines, linewidth=1.7, alpha=0.85)
    plt.setp(markerline, markersize=5.5, label="True sparse signal")
    markerline2, stemlines2, _ = plt.stem(idx + 0.25, x_hat, linefmt="#CC6677", markerfmt="s", basefmt=" ")
    plt.setp(stemlines2, linewidth=1.2, alpha=0.75)
    plt.setp(markerline2, markersize=4.6, label="ADMM estimate")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    plt.xlabel("Coefficient index")
    plt.ylabel("Value")
    plt.title("Sparse coefficient recovery with zero reference line")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(recovery_path, dpi=220)
    plt.close()
