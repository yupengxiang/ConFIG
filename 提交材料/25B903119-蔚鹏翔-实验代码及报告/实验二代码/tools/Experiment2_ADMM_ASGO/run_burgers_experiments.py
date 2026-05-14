"""Run reproducible Burgers PINN experiments for the ConFIG report.

Examples:
    python3 tools/Experiment2_ADMM_ASGO/run_burgers_experiments.py --epochs 1000 --methods standard adam_l1 asgo admm_asgo
    python3 tools/Experiment2_ADMM_ASGO/run_burgers_experiments.py --epochs 30000 --num-run 3 --methods standard config mconfig asgo adam_l1 asgo_l1 admm_asgo
    python3 tools/Experiment2_ADMM_ASGO/run_burgers_experiments.py --epochs 30000 --num-run 3 --methods standard config mconfig asgo adam_l1 asgo_l1 admm_asgo --parallel-methods 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PINN_ROOT = ROOT / "experiments" / "PINN"
for path in (ROOT, PINN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conflictfree.grad_operator import ConFIGOperator, PCGradOperator
from experiments.PINN.lib_pinns.burgers.trainer import (  # noqa: E402
    BurgersTrainerBasis,
    StandardTrainer,
    run_burgers,
)
from experiments.PINN.lib_pinns.sparse_trainers import (  # noqa: E402
    ADMMASGOTrainerBasis,
    ASGOL1TrainerBasis,
    ASGOTrainerBasis,
    L1PenaltyTrainerBasis,
)
from experiments.PINN.lib_pinns.trainer_basis import (  # noqa: E402
    get_gradvec_trainer,
    get_momentum_trainer,
)


class AdamL1Trainer(BurgersTrainerBasis, L1PenaltyTrainerBasis):
    pass


class ASGOTrainer(BurgersTrainerBasis, ASGOTrainerBasis):
    pass


class ASGOL1Trainer(BurgersTrainerBasis, ASGOL1TrainerBasis):
    pass


class ADMMASGOTrainer(BurgersTrainerBasis, ADMMASGOTrainerBasis):
    pass


METHODS = {
    "standard": lambda: StandardTrainer(),
    "pcgrad": lambda: get_gradvec_trainer(BurgersTrainerBasis, PCGradOperator()),
    "config": lambda: get_gradvec_trainer(BurgersTrainerBasis, ConFIGOperator()),
    "mconfig": lambda: get_momentum_trainer(BurgersTrainerBasis, ConFIGOperator()),
    "adam_l1": lambda: AdamL1Trainer(),
    "asgo": lambda: ASGOTrainer(),
    "asgo_l1": lambda: ASGOL1Trainer(),
    "admm_asgo": lambda: ADMMASGOTrainer(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Burgers PINN optimizer comparisons.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["standard", "pcgrad", "config"],
        choices=sorted(METHODS),
        help="Methods to run.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs.")
    parser.add_argument("--num-run", type=int, default=1, help="Number of random seeds.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--n-losses", type=int, default=2, choices=[2, 3], help="Loss split count.")
    parser.add_argument(
        "--save-path",
        default="./PINN_trained/Burgers/Experiment2_ADMM_ASGO/",
        help="Directory where experiment folders will be created.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Optional suffix appended to each run name, e.g. _30000.",
    )
    parser.add_argument(
        "--save-epoch",
        type=int,
        default=None,
        help="Checkpoint save frequency. Defaults to max(1, epochs // 2).",
    )
    parser.add_argument(
        "--final-record-epoch",
        type=int,
        default=100,
        help="Number of final epochs used for final_losses.txt statistics.",
    )
    parser.add_argument(
        "--update-training-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to resample training points during training.",
    )
    parser.add_argument(
        "--parallel-methods",
        type=int,
        default=1,
        help="Run up to this many methods concurrently. Each method still runs its num-run seeds sequentially.",
    )
    parser.add_argument("--l1-lambda", type=float, default=1e-6, help="L1 penalty used by adam_l1/asgo_l1.")
    parser.add_argument("--admm-lambda", type=float, default=1e-5, help="L1 penalty weight used by admm_asgo.")
    parser.add_argument("--admm-rho", type=float, default=1e-2, help="ADMM quadratic penalty used by admm_asgo.")
    parser.add_argument("--asgo-momentum", type=float, default=0.9, help="ASGO momentum.")
    parser.add_argument("--asgo-weight-decay", type=float, default=0.0, help="ASGO weight decay.")
    parser.add_argument("--asgo-beta2", type=float, default=0.8, help="ASGO second-moment beta.")
    parser.add_argument("--asgo-eps", type=float, default=1e-10, help="ASGO preconditioner damping.")
    parser.add_argument("--asgo-matalg-steps", type=int, default=10, help="ASGO matrix inverse-root iterations.")
    return parser.parse_args()


def build_child_command(args: argparse.Namespace, method: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--epochs",
        str(args.epochs),
        "--num-run",
        str(args.num_run),
        "--methods",
        method,
        "--device",
        args.device,
        "--n-losses",
        str(args.n_losses),
        "--save-path",
        args.save_path,
        "--name-suffix",
        args.name_suffix,
        "--final-record-epoch",
        str(args.final_record_epoch),
        "--parallel-methods",
        "1",
        "--l1-lambda",
        str(args.l1_lambda),
        "--admm-lambda",
        str(args.admm_lambda),
        "--admm-rho",
        str(args.admm_rho),
        "--asgo-momentum",
        str(args.asgo_momentum),
        "--asgo-weight-decay",
        str(args.asgo_weight_decay),
        "--asgo-beta2",
        str(args.asgo_beta2),
        "--asgo-eps",
        str(args.asgo_eps),
        "--asgo-matalg-steps",
        str(args.asgo_matalg_steps),
    ]
    if args.save_epoch is not None:
        command.extend(["--save-epoch", str(args.save_epoch)])
    if not args.update_training_data:
        command.append("--no-update-training-data")
    return command


def run_methods_in_parallel(args: argparse.Namespace) -> None:
    max_parallel = max(1, args.parallel_methods)
    running: list[tuple[str, subprocess.Popen]] = []
    pending = list(args.methods)
    failures: list[tuple[str, int]] = []

    while pending or running:
        while pending and len(running) < max_parallel:
            method = pending.pop(0)
            command = build_child_command(args, method)
            print(f"\n=== Launching {method} in parallel ===", flush=True)
            running.append((method, subprocess.Popen(command, cwd=ROOT)))

        still_running: list[tuple[str, subprocess.Popen]] = []
        for method, process in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((method, process))
            elif return_code != 0:
                failures.append((method, return_code))
        running = still_running

        if running:
            time.sleep(5)

    if failures:
        failure_text = ", ".join(f"{method}: exit {code}" for method, code in failures)
        raise SystemExit(f"Parallel experiment failed: {failure_text}")


def main() -> None:
    args = parse_args()
    if args.parallel_methods > 1 and len(args.methods) > 1:
        run_methods_in_parallel(args)
        return

    save_epoch = args.save_epoch if args.save_epoch is not None else max(1, args.epochs // 2)
    final_record_epoch = min(args.final_record_epoch, args.epochs)

    common = {
        "epochs": args.epochs,
        "num_run": args.num_run,
        "save_path": args.save_path,
        "device": args.device,
        "save_epoch": save_epoch,
        "final_record_epoch": final_record_epoch,
        "n_losses": args.n_losses,
        "update_training_data": args.update_training_data,
    }
    asgo_common = {
        "asgo_momentum": args.asgo_momentum,
        "asgo_weight_decay": args.asgo_weight_decay,
        "asgo_beta2": args.asgo_beta2,
        "asgo_eps": args.asgo_eps,
        "asgo_matalg_steps": args.asgo_matalg_steps,
    }
    method_extra = {
        "adam_l1": {"l1_lambda": args.l1_lambda},
        "asgo": asgo_common,
        "asgo_l1": {**asgo_common, "l1_lambda": args.l1_lambda},
        "admm_asgo": {**asgo_common, "admm_lambda": args.admm_lambda, "admm_rho": args.admm_rho},
    }

    for method in args.methods:
        run_name = f"{method}_{args.epochs}{args.name_suffix}"
        print(f"\n=== Running {run_name} ===", flush=True)
        run_burgers(name=run_name, trainer=METHODS[method](), **common, **method_extra.get(method, {}))


if __name__ == "__main__":
    main()
