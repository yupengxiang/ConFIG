"""Analyze saved Burgers PINN experiments and generate report figures.

Examples:
    python3 tools/Experiment2_ADMM_ASGO/analyze_burgers_results.py
    python3 tools/Experiment2_ADMM_ASGO/analyze_burgers_results.py --run-root PINN_trained/Burgers/Experiment2_ADMM_ASGO --methods standard_30000_exp2_full config_30000_exp2_full mconfig_30000_exp2_full asgo_30000_exp2_full adam_l1_30000_exp2_full asgo_l1_30000_exp2_full admm_asgo_30000_exp2_full
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[2]
PINN_ROOT = ROOT / "experiments" / "PINN"
for path in (ROOT, PINN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conflictfree.utils import get_gradient_vector  # noqa: E402
from experiments.PINN.lib_pinns.burgers.data_sampler import BurgersSampler  # noqa: E402
from experiments.PINN.lib_pinns.burgers.networks import BurgersNet  # noqa: E402
from experiments.PINN.lib_pinns.burgers.physical_residual import physical_residual  # noqa: E402
from experiments.PINN.lib_pinns.burgers.run_test import run_test  # noqa: E402
from experiments.PINN.lib_pinns.sparse_trainers import sparsity_metrics_from_state_dict  # noqa: E402


DEFAULT_METHODS = ["standard_1000", "pcgrad_1000", "config_1000"]
DISPLAY_NAME = {
    "standard": "Standard Adam",
    "standard_adam": "Standard Adam",
    "pcgrad": "PCGrad",
    "config": "ConFIG",
    "mconfig": "M-ConFIG",
    "adam_l1": "Adam + L1",
    "asgo": "ASGO",
    "asgo_l1": "ASGO + L1",
    "admm_asgo": "ADMM-ASGO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Burgers PINN experiment folders.")
    parser.add_argument("--run-root", default="PINN_trained/Burgers/Experiment2_ADMM_ASGO", help="Experiment root.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Run folder names.")
    parser.add_argument("--out-dir", default="report_assets/Experiment2_ADMM_ASGO", help="Output directory for figures/JSON.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for model evaluation.")
    parser.add_argument(
        "--gradient-seed",
        type=int,
        default=21339,
        help="Seed for the diagnostic gradient-conflict sample.",
    )
    return parser.parse_args()


def method_dir_with_fallback(method_dir: Path) -> Path:
    if not method_dir.exists() and method_dir.name.startswith("standard_"):
        fallback = method_dir.with_name(method_dir.name.replace("standard_", "standard_adam_", 1))
        if fallback.exists():
            return fallback
    return method_dir


def completed_runs(method_dir: Path) -> list[Path]:
    method_dir = method_dir_with_fallback(method_dir)
    if not method_dir.exists():
        raise FileNotFoundError(f"Method directory not found: {method_dir}")
    candidates = []
    for path in method_dir.iterdir():
        if not path.is_dir():
            continue
        log_path = path / "training_event.log"
        weights_path = path / "trained_network_weights.pt"
        if not log_path.exists() or not weights_path.exists():
            continue
        log_text = log_path.read_text(encoding="utf-8")
        if "Training finished!" in log_text:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No completed timestamped run folder found under {method_dir}")
    return sorted(candidates, key=lambda p: p.name)


def display_name(folder_name: str) -> str:
    stem = re.sub(r"_\d+.*$", "", folder_name)
    return DISPLAY_NAME.get(stem, stem)


def read_scalar_series(project_path: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    events = sorted((project_path / "records").glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No tensorboard events found in {project_path / 'records'}")
    ea = event_accumulator.EventAccumulator(str(events[0]))
    ea.Reload()
    values = ea.Scalars(tag)
    return np.asarray([v.step for v in values]), np.asarray([float(v.value) for v in values])


def grab_float(log_text: str, pattern: str) -> float | None:
    match = re.search(pattern, log_text)
    return float(match.group(1)) if match else None


def load_network(project_path: Path, device: str) -> BurgersNet:
    network = BurgersNet().to(device)
    weights = torch.load(project_path / "trained_network_weights.pt", map_location=device)
    network.load_state_dict(weights)
    network.eval()
    return network


def weight_sparsity(project_path: Path) -> dict[str, object]:
    weights = torch.load(project_path / "trained_network_weights.pt", map_location="cpu")
    return sparsity_metrics_from_state_dict(weights)


def gradient_conflict_metrics(network: BurgersNet, device: str, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    network.train()
    sampler = BurgersSampler(
        n_internal=10000,
        n_initial=250,
        n_boundary=250,
        device=device,
        update_data=True,
        seed=seed,
        data_sampler="latin_hypercube",
        x_start=-1.0,
        x_end=1.0,
        simulation_time=1.0,
    )

    x_internal, t_internal = sampler.sample_internal()
    internal_loss = torch.mean(physical_residual(network(x_internal, t_internal), x_internal, t_internal) ** 2)
    network.zero_grad()
    internal_loss.backward()
    internal_grad = get_gradient_vector(network).detach()

    x_bi, t_bi, value_bi = sampler.sample_initial_boundary()
    boundary_initial_loss = torch.mean((network(x_bi, t_bi) - value_bi) ** 2)
    network.zero_grad()
    boundary_initial_loss.backward()
    boundary_initial_grad = get_gradient_vector(network).detach()

    cosine = torch.nn.functional.cosine_similarity(internal_grad, boundary_initial_grad, dim=0)
    dot = torch.dot(internal_grad, boundary_initial_grad)
    return {
        "final_sample_internal_loss": float(internal_loss.detach().cpu()),
        "final_sample_boundary_initial_loss": float(boundary_initial_loss.detach().cpu()),
        "final_sample_grad_cosine_internal_vs_bi": float(cosine.detach().cpu()),
        "final_sample_grad_dot_internal_vs_bi": float(dot.detach().cpu()),
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    simulation_data = np.load(PINN_ROOT / "data" / "burgers" / "simulation_data.npy")
    method_runs = []
    for method in args.methods:
        runs = completed_runs(run_root / method)
        method_runs.append((display_name(method), runs))

    summary = {}

    plt.figure(figsize=(7.2, 4.2), dpi=160)
    representative_projects = []
    for label, runs in method_runs:
        run_items = []
        series = []
        for project_path in runs:
            steps, losses = read_scalar_series(project_path, "Loss/validation")
            series.append((steps, losses))
            log_text = (project_path / "training_event.log").read_text(encoding="utf-8")
            network = load_network(project_path, args.device)
            mse, _mse_grid, _prediction = run_test(network, simulation_data, device=args.device)
            sparsity = weight_sparsity(project_path)
            item = {
                "project_path": str(project_path),
                "final_validation_loss": grab_float(log_text, r"Final validation loss: ([0-9.eE+-]+)"),
                "best_validation_loss": grab_float(log_text, r"Best validation loss: ([0-9.eE+-]+)"),
                "final_training_loss": grab_float(log_text, r"Final training loss: ([0-9.eE+-]+)"),
                "training_speed_s_per_iter": grab_float(log_text, r"Training speed: ([0-9.eE+-]+) s/iteration"),
                "run_test_mse": float(mse),
                "global_sparsity": sparsity["sparsity"],
                "matrix_weight_total": sparsity["total"],
                "matrix_weight_zeros": sparsity["zeros"],
                "matrix_weight_nonzeros": sparsity["nonzeros"],
                "layer_sparsity": sparsity["layers"],
                "admm_primal_residual": grab_float(log_text, r"ADMM final primal residual: ([0-9.eE+-]+)"),
                "admm_dual_residual": grab_float(log_text, r"ADMM final dual residual: ([0-9.eE+-]+)"),
            }
            run_items.append(item)

        first_steps = series[0][0]
        same_grid = all(len(steps) == len(first_steps) and np.array_equal(steps, first_steps) for steps, _ in series)
        if same_grid:
            loss_matrix = np.stack([losses for _steps, losses in series], axis=0)
            mean_losses = loss_matrix.mean(axis=0)
            std_losses = loss_matrix.std(axis=0)
            plt.plot(first_steps, mean_losses, label=f"{label} mean", linewidth=1.7)
            if len(series) > 1:
                plt.fill_between(first_steps, mean_losses - std_losses, mean_losses + std_losses, alpha=0.16)
        else:
            for idx, (steps, losses) in enumerate(series, start=1):
                plt.plot(steps, losses, label=f"{label} run {idx}", linewidth=1.1, alpha=0.8)

        best_item = min(run_items, key=lambda item: item["run_test_mse"])
        representative_project = Path(best_item["project_path"])
        representative_network = load_network(representative_project, args.device)
        conflict_metrics = gradient_conflict_metrics(representative_network, args.device, args.gradient_seed)
        representative_projects.append((label, representative_project))

        def aggregate(key: str) -> dict[str, float | None]:
            values = np.asarray([item[key] for item in run_items if item[key] is not None], dtype=float)
            if values.size == 0:
                return {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                }
            return {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

        summary[label] = {
            "num_runs": len(run_items),
            "runs": run_items,
            "aggregate": {
                "final_validation_loss": aggregate("final_validation_loss"),
                "best_validation_loss": aggregate("best_validation_loss"),
                "final_training_loss": aggregate("final_training_loss"),
                "training_speed_s_per_iter": aggregate("training_speed_s_per_iter"),
                "run_test_mse": aggregate("run_test_mse"),
                "global_sparsity": aggregate("global_sparsity"),
                "matrix_weight_nonzeros": aggregate("matrix_weight_nonzeros"),
                "admm_primal_residual": aggregate("admm_primal_residual"),
                "admm_dual_residual": aggregate("admm_dual_residual"),
            },
            "representative_best_run": best_item,
            "representative_gradient_conflict": conflict_metrics,
        }

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE (log scale)")
    plt.title("Burgers PINN validation loss")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    loss_path = out_dir / "burgers_validation_loss.png"
    plt.savefig(loss_path)
    plt.close()

    labels = list(summary)
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(idx % 10) for idx, label in enumerate(labels)}
    markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
    mean_mse = [summary[label]["aggregate"]["run_test_mse"]["mean"] for label in labels]
    mean_sparsity = [summary[label]["aggregate"]["global_sparsity"]["mean"] for label in labels]
    plt.figure(figsize=(7.0, 4.2), dpi=160)
    for idx, (label, x_value, y_value) in enumerate(zip(labels, mean_sparsity, mean_mse)):
        plt.scatter(
            x_value,
            y_value,
            s=78,
            marker=markers[idx % len(markers)],
            color=colors[label],
            edgecolor="black",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
    plt.yscale("log")
    plt.xlabel("Global matrix-weight sparsity")
    plt.ylabel("Run-test MSE (log scale)")
    plt.title("Accuracy-sparsity trade-off")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8, frameon=True, ncol=2)
    plt.tight_layout()
    tradeoff_path = out_dir / "burgers_accuracy_sparsity.png"
    plt.savefig(tradeoff_path)
    plt.close()

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7.0, 1.0 * len(labels)), 4.0), dpi=160)
    bars = ax.bar(x, mean_sparsity, color=[colors[label] for label in labels], edgecolor="black", linewidth=0.5)
    for idx, value in enumerate(mean_sparsity):
        if value <= 0:
            ax.scatter(
                idx,
                0.012,
                marker="v",
                s=58,
                facecolors="white",
                edgecolors=colors[labels[idx]],
                linewidth=1.5,
                zorder=4,
            )
            ax.text(idx, 0.035, "0", ha="center", va="bottom", fontsize=8, color=colors[labels[idx]])
        else:
            ax.text(idx, value + 0.025, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    plt.xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, max(0.35, max(mean_sparsity) * 1.25))
    ax.set_ylabel("Global matrix-weight sparsity")
    ax.set_title("Final Burgers PINN weight sparsity")
    legend_handles = [Patch(facecolor=colors[label], edgecolor="black", label=label) for label in labels]
    legend_handles.append(Line2D([0], [0], marker="v", color="black", markerfacecolor="white", linestyle="None", label="zero sparsity marker"))
    ax.legend(handles=legend_handles, fontsize=7.5, frameon=True, ncol=2)
    fig.tight_layout()
    sparsity_path = out_dir / "burgers_weight_sparsity.png"
    fig.savefig(sparsity_path)
    plt.close(fig)

    representative_layers = {
        label: weight_sparsity(project_path)["layers"]
        for label, project_path in representative_projects
    }
    all_layer_names = sorted({layer for layers in representative_layers.values() for layer in layers})
    layer_sparsity_path = None
    if all_layer_names:
        width = 0.8 / max(1, len(labels))
        x_layers = np.arange(len(all_layer_names))
        fig, ax = plt.subplots(figsize=(max(7.2, 1.2 * len(all_layer_names)), 4.4), dpi=160)
        for idx, label in enumerate(labels):
            values = [
                representative_layers.get(label, {}).get(layer, {}).get("sparsity", 0.0)
                for layer in all_layer_names
            ]
            offsets = x_layers + (idx - (len(labels) - 1) / 2) * width
            ax.bar(offsets, values, width=width, label=label, color=colors[label], edgecolor="black", linewidth=0.3)
            zero_offsets = [offset for offset, value in zip(offsets, values) if value <= 0]
            if zero_offsets:
                ax.scatter(
                    zero_offsets,
                    [0.012] * len(zero_offsets),
                    marker="v",
                    s=26,
                    facecolors="white",
                    edgecolors=colors[label],
                    linewidth=1.0,
                    zorder=4,
                )
        ax.set_xticks(x_layers)
        ax.set_xticklabels(all_layer_names, rotation=25, ha="right")
        ax.set_ylim(0, max(0.35, max(max(layer.get("sparsity", 0.0) for layer in layers.values()) for layers in representative_layers.values()) * 1.25))
        ax.set_ylabel("Layer sparsity")
        ax.set_title("Representative run layer-wise sparsity")
        legend_handles = [Patch(facecolor=colors[label], edgecolor="black", label=label) for label in labels]
        legend_handles.append(Line2D([0], [0], marker="v", color="black", markerfacecolor="white", linestyle="None", label="zero sparsity marker"))
        ax.legend(handles=legend_handles, fontsize=7.5, frameon=True, ncol=2)
        fig.tight_layout()
        layer_sparsity_path = out_dir / "burgers_layer_sparsity.png"
        fig.savefig(layer_sparsity_path)
        plt.close(fig)

    n_rows = int(np.ceil(len(representative_projects) / 2))
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.2, 3.0 * n_rows), dpi=160, constrained_layout=True)
    axes = np.asarray(axes).reshape(n_rows, 4)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, (label, project_path) in enumerate(representative_projects):
        row = idx // 2
        col = (idx % 2) * 2
        network = load_network(project_path, args.device)
        _mse, _mse_grid, prediction = run_test(network, simulation_data, device=args.device)
        pred_ax = axes[row, col]
        err_ax = axes[row, col + 1]
        pred_ax.axis("on")
        err_ax.axis("on")
        pred_im = pred_ax.imshow(prediction, aspect="auto", origin="lower", extent=[-1, 1, 0, 1], cmap="RdBu_r")
        pred_ax.set_title(f"{label}: prediction", fontsize=10)
        pred_ax.set_xlabel("x")
        pred_ax.set_ylabel("t")
        fig.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.04)

        error = np.abs(simulation_data - prediction)
        err_im = err_ax.imshow(error, aspect="auto", origin="lower", extent=[-1, 1, 0, 1], cmap="magma")
        err_ax.set_title(f"{label}: absolute error", fontsize=10)
        err_ax.set_xlabel("x")
        err_ax.set_ylabel("t")
        fig.colorbar(err_im, ax=err_ax, fraction=0.046, pad=0.04)
    heatmap_path = out_dir / "burgers_predictions_errors.png"
    fig.savefig(heatmap_path)
    plt.close(fig)

    plt.figure(figsize=(5.2, 3.8), dpi=160)
    plt.imshow(simulation_data, aspect="auto", origin="lower", extent=[-1, 1, 0, 1], cmap="RdBu_r")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Burgers reference solution")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.tight_layout()
    reference_path = out_dir / "burgers_reference_solution.png"
    plt.savefig(reference_path)
    plt.close()

    summary_path = out_dir / "burgers_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {loss_path}")
    print(f"Saved: {tradeoff_path}")
    print(f"Saved: {sparsity_path}")
    if layer_sparsity_path is not None:
        print(f"Saved: {layer_sparsity_path}")
    print(f"Saved: {heatmap_path}")
    print(f"Saved: {reference_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
