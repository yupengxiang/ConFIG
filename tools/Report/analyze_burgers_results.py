"""Analyze saved Burgers PINN experiments and generate report figures.

Examples:
    python3 tools/Report/analyze_burgers_results.py
    python3 tools/Report/analyze_burgers_results.py --run-root PINN_trained/Burgers/Report --methods standard_30000_full pcgrad_30000_full config_30000_full mconfig_30000_full
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


DEFAULT_METHODS = ["standard_1000", "pcgrad_1000", "config_1000"]
DISPLAY_NAME = {
    "standard": "Standard Adam",
    "standard_adam": "Standard Adam",
    "pcgrad": "PCGrad",
    "config": "ConFIG",
    "mconfig": "M-ConFIG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Burgers PINN experiment folders.")
    parser.add_argument("--run-root", default="PINN_trained/Burgers/Report", help="Experiment root.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Run folder names.")
    parser.add_argument("--out-dir", default="report_assets/Report", help="Output directory for figures/JSON.")
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
            item = {
                "project_path": str(project_path),
                "final_validation_loss": grab_float(log_text, r"Final validation loss: ([0-9.eE+-]+)"),
                "best_validation_loss": grab_float(log_text, r"Best validation loss: ([0-9.eE+-]+)"),
                "final_training_loss": grab_float(log_text, r"Final training loss: ([0-9.eE+-]+)"),
                "training_speed_s_per_iter": grab_float(log_text, r"Training speed: ([0-9.eE+-]+) s/iteration"),
                "run_test_mse": float(mse),
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
    print(f"Saved: {heatmap_path}")
    print(f"Saved: {reference_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
