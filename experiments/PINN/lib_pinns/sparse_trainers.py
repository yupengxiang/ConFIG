"""Sparse PINN trainers used by the experiment-2 ADMM-ASGO study."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import torch

from .trainer_basis import StandardTrainerBasis


ROOT = Path(__file__).resolve().parents[3]
ASGO_ROOT = ROOT / "external" / "ASGO"
if ASGO_ROOT.exists() and str(ASGO_ROOT) not in sys.path:
    sys.path.insert(0, str(ASGO_ROOT))

try:  # Keep the experiment usable even before the ASGO fork is cloned.
    from optimizers.utils.matrix_decom import _matrix_inverse_root_PolarExpress
except Exception:  # pragma: no cover - only used when external/ASGO is missing.
    _matrix_inverse_root_PolarExpress = None


def soft_threshold(values: torch.Tensor, threshold: float) -> torch.Tensor:
    """Elementwise proximal operator for threshold * ||x||_1."""

    return torch.sign(values) * torch.clamp(torch.abs(values) - threshold, min=0.0)


def sparse_named_parameters(network: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Return trainable matrix weights managed by L1/ADMM sparsity."""

    return [(name, param) for name, param in network.named_parameters() if param.requires_grad and param.ndim >= 2]


def l1_penalty(network: torch.nn.Module) -> torch.Tensor:
    params = sparse_named_parameters(network)
    if not params:
        return torch.zeros((), device=next(network.parameters()).device)
    return torch.stack([param.abs().sum() for _name, param in params]).sum()


def sparsity_metrics_from_state_dict(state_dict: dict[str, torch.Tensor], atol: float = 0.0) -> dict[str, object]:
    layer_metrics = {}
    total = 0
    zeros = 0
    nonzeros = 0
    for name, value in state_dict.items():
        if value.ndim < 2:
            continue
        tensor = value.detach()
        zero_mask = torch.abs(tensor) <= atol
        layer_total = tensor.numel()
        layer_zeros = int(zero_mask.sum().item())
        layer_nonzeros = layer_total - layer_zeros
        total += layer_total
        zeros += layer_zeros
        nonzeros += layer_nonzeros
        layer_metrics[name] = {
            "total": layer_total,
            "zeros": layer_zeros,
            "nonzeros": layer_nonzeros,
            "sparsity": layer_zeros / layer_total if layer_total else 0.0,
        }
    return {
        "total": total,
        "zeros": zeros,
        "nonzeros": nonzeros,
        "sparsity": zeros / total if total else 0.0,
        "layers": layer_metrics,
    }


class NullLRScheduler:
    """Scheduler shim for wrapped optimizers whose inner groups own the LR."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        return None


class LocalASGO(torch.optim.Optimizer):
    """Small adapter around the ASGO paper implementation for 2-D tensors."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        beta2: float = 0.8,
        eps: float = 1e-10,
        matalg_steps: int = 10,
    ) -> None:
        defaults = {
            "lr": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "beta2": beta2,
            "eps": eps,
        }
        self.matalg_steps = matalg_steps
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.ndim < 2:
                    raise ValueError("LocalASGO only supports matrix parameters; use Adam for bias/1-D tensors.")
                if grad.ndim != 2:
                    raise ValueError("Burgers PINN ASGO adapter currently expects 2-D Linear weights.")

                state = self.state[param]
                if not state:
                    dim = min(grad.size(0), grad.size(1))
                    state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    state["precond"] = torch.zeros(dim, dim, device=grad.device, dtype=grad.dtype)
                    state["step"] = 0

                state["momentum_buffer"].lerp_(grad, 1 - momentum)
                if grad.size(0) < grad.size(1):
                    precond_tmp = torch.mm(grad, grad.t())
                else:
                    precond_tmp = torch.mm(grad.t(), grad)
                state["precond"].lerp_(precond_tmp, 1 - beta2)

                inverse_precond = self._matrix_inverse_root(state["precond"], eps)
                if torch.isnan(inverse_precond).any() or torch.isinf(inverse_precond).any():
                    inverse_precond = torch.eye(
                        state["precond"].size(0),
                        device=state["precond"].device,
                        dtype=state["precond"].dtype,
                    )

                if grad.size(0) < grad.size(1):
                    update = inverse_precond @ state["momentum_buffer"]
                else:
                    update = state["momentum_buffer"] @ inverse_precond
                norm = torch.linalg.matrix_norm(update, ord="fro").item()
                if norm > 0:
                    scale = 0.2 * math.sqrt(grad.size(0) * grad.size(1)) / norm
                    update.mul_(scale)

                if weight_decay > 0:
                    param.mul_(1 - lr * weight_decay)
                param.add_(update, alpha=-lr)
                state["step"] += 1

    def _matrix_inverse_root(self, matrix: torch.Tensor, eps: float) -> torch.Tensor:
        if _matrix_inverse_root_PolarExpress is not None:
            return _matrix_inverse_root_PolarExpress(matrix, eps=eps, N=self.matalg_steps).to(matrix.dtype)
        identity = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        return torch.linalg.inv(torch.linalg.cholesky(matrix + eps * identity)).t()


class HybridOptimizer:
    """Wrap ASGO for matrix weights and Adam for bias/1-D parameters."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        self.param_groups = []
        for optimizer in optimizers:
            self.param_groups.extend(optimizer.param_groups)

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, object]:
        return {f"optimizer_{idx}": optimizer.state_dict() for idx, optimizer in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        for idx, optimizer in enumerate(self.optimizers):
            key = f"optimizer_{idx}"
            if key in state_dict:
                optimizer.load_state_dict(state_dict[key])


class ASGOTrainerBasis(StandardTrainerBasis):
    """PINN trainer using ASGO on matrix weights and Adam on biases."""

    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("asgo_momentum", default_value=0.9, value_type=float)
        self.configs_handler.add_config_item("asgo_weight_decay", default_value=0.0, value_type=float)
        self.configs_handler.add_config_item("asgo_beta2", default_value=0.8, value_type=float)
        self.configs_handler.add_config_item("asgo_eps", default_value=1e-10, value_type=float)
        self.configs_handler.add_config_item("asgo_matalg_steps", default_value=10, value_type=int)

    def get_optimizer(self, network):
        matrix_params = [param for _name, param in sparse_named_parameters(network)]
        adam_params = [
            param
            for _name, param in network.named_parameters()
            if param.requires_grad and param.ndim < 2
        ]
        optimizers: list[torch.optim.Optimizer] = []
        if matrix_params:
            optimizers.append(
                LocalASGO(
                    matrix_params,
                    learning_rate=self.configs.lr,
                    momentum=self.configs.asgo_momentum,
                    weight_decay=self.configs.asgo_weight_decay,
                    beta2=self.configs.asgo_beta2,
                    eps=self.configs.asgo_eps,
                    matalg_steps=self.configs.asgo_matalg_steps,
                )
            )
        if adam_params:
            optimizers.append(torch.optim.Adam(adam_params, lr=self.configs.lr))
        return HybridOptimizer(optimizers)

    def get_lr_scheduler(self, optimizer):
        return NullLRScheduler(optimizer)


class L1PenaltyTrainerBasis(StandardTrainerBasis):
    """Adam trainer with explicit L1 penalty on matrix weights."""

    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("l1_lambda", default_value=1e-6, value_type=float)

    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        total_loss = super().train_step(network, batched_data, idx_batch, num_batches, idx_epoch, num_epoch)
        penalty = self.configs.l1_lambda * l1_penalty(network)
        self.recorder.add_scalar("sparsity/l1_penalty", penalty.item(), idx_epoch)
        return total_loss + penalty


class ASGOL1TrainerBasis(ASGOTrainerBasis):
    """ASGO trainer with a standard differentiable L1 penalty baseline."""

    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("l1_lambda", default_value=1e-6, value_type=float)

    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        total_loss = super().train_step(network, batched_data, idx_batch, num_batches, idx_epoch, num_epoch)
        penalty = self.configs.l1_lambda * l1_penalty(network)
        self.recorder.add_scalar("sparsity/l1_penalty", penalty.item(), idx_epoch)
        return total_loss + penalty


class ADMMASGOTrainerBasis(ASGOTrainerBasis):
    """ADMM wrapper around the ASGO theta-step for sparse PINN weights."""

    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("admm_lambda", default_value=1e-5, value_type=float)
        self.configs_handler.add_config_item("admm_rho", default_value=1e-2, value_type=float)

    def event_before_training(self, network):
        super().event_before_training(network)
        self._admm_params = sparse_named_parameters(network)
        self._admm_z = {name: param.detach().clone() for name, param in self._admm_params}
        self._admm_u = {name: torch.zeros_like(param) for name, param in self._admm_params}
        self._last_admm_primal_residual = 0.0
        self._last_admm_dual_residual = 0.0

    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        total_loss = super().train_step(network, batched_data, idx_batch, num_batches, idx_epoch, num_epoch)
        penalty_terms = []
        for name, param in self._admm_params:
            z = self._admm_z[name].to(device=param.device, dtype=param.dtype)
            u = self._admm_u[name].to(device=param.device, dtype=param.dtype)
            penalty_terms.append(torch.sum((param - z + u) ** 2))
        if penalty_terms:
            admm_penalty = 0.5 * self.configs.admm_rho * torch.stack(penalty_terms).sum()
        else:
            admm_penalty = torch.zeros((), device=total_loss.device)
        self.recorder.add_scalar("admm/theta_penalty", admm_penalty.item(), idx_epoch)
        return total_loss + admm_penalty

    @torch.no_grad()
    def event_after_training_iteration(self, network, idx_epoch, idx_batch):
        primal = []
        dual = []
        zeros = 0
        total = 0
        threshold = self.configs.admm_lambda / self.configs.admm_rho
        for name, param in self._admm_params:
            z_prev = self._admm_z[name].to(device=param.device, dtype=param.dtype)
            u_prev = self._admm_u[name].to(device=param.device, dtype=param.dtype)
            z_next = soft_threshold(param + u_prev, threshold)
            u_next = u_prev + param - z_next
            primal.append(torch.sum((param - z_next) ** 2))
            dual.append(torch.sum((self.configs.admm_rho * (z_next - z_prev)) ** 2))
            zeros += int((z_next == 0).sum().item())
            total += z_next.numel()
            self._admm_z[name] = z_next.detach().clone()
            self._admm_u[name] = u_next.detach().clone()
        self._last_admm_primal_residual = float(torch.sqrt(torch.stack(primal).sum()).detach().cpu()) if primal else 0.0
        self._last_admm_dual_residual = float(torch.sqrt(torch.stack(dual).sum()).detach().cpu()) if dual else 0.0
        sparsity = zeros / total if total else 0.0
        self.recorder.add_scalar("admm/primal_residual", self._last_admm_primal_residual, idx_epoch)
        self.recorder.add_scalar("admm/dual_residual", self._last_admm_dual_residual, idx_epoch)
        self.recorder.add_scalar("sparsity/global_weight_sparsity", sparsity, idx_epoch)

    @torch.no_grad()
    def event_after_training(self, network):
        for name, param in self._admm_params:
            param.copy_(self._admm_z[name].to(device=param.device, dtype=param.dtype))
        metrics = sparsity_metrics_from_state_dict(network.state_dict())
        self.logger.info("ADMM final primal residual: {}".format(self._last_admm_primal_residual))
        self.logger.info("ADMM final dual residual: {}".format(self._last_admm_dual_residual))
        self.logger.info("Final global weight sparsity: {}".format(metrics["sparsity"]))
        self.logger.info("Final nonzero matrix weights: {}".format(metrics["nonzeros"]))
