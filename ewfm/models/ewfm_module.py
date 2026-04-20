from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch.distributions import Distribution
from torchmetrics import MeanMetric

from ewfm.energies.base_energy_function import BaseEnergyFunction
from ewfm.energies.gmm_energy import GMMEnergy
from ewfm.models.components.ema import EMAWrapper
from ewfm.models.components.metrics import wasserstein
from ewfm.utils.logging_utils import fig_to_image

__all__ = ["EWFMModule"]


class EWFMModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        energy_function: BaseEnergyFunction,
        baseline_until_epoch: int,
        algorithm: str,
        bagging_buffer_size: int,
        num_samples_per_batch: int,
        use_train_data: bool,
        validation_uniform_samples: int,
        likelihood_plot_spacing: float,
        contour_plot_levels: int,
        flow_num_particles: int,
        eval_batch_size: int,
        test_batch_size: int,
        val_plot_batch_size: int,
        step_size: float,
        integration_method: str,
        p_0_prior: Distribution,
        atol: float,
        rtol: float,
        data_n_train_batches_per_epoch: int,
        annealing_epochs_per_temperature: int,
        total_annealing_epochs: int,
        use_exact_divergence: bool = False,
        vector_field_max_norm: float = None,
        enable_detailed_train_logging: bool = True,
        seed: int = None,
        q_1_prior: Optional[Distribution] = None,
        scheduler: Any = None,
        lr_scheduler_update_frequency: int = 1,
        enable_annealing: bool = False,
        initial_temperature: float = 1.0,
        final_temperature: float = 1.0,
        temperature_schedule: str = "geometric",
        temperature_values: Optional[List[float]] = None,
        clipping_method: Optional[str] = None,
        clipping_percentile: float = 95.0,
        use_ema: bool = False,
        ema_beta: float = 0.999,
        beta_warmup_denominator: float = 10,
        metric_batch_size: Optional[int] = None,
        batched_sampling: bool = False,
        sample_batch_size: int = 1000,
    ) -> None:
        super().__init__()

        self.optimizer_ctor = optimizer
        self.scheduler_ctor = scheduler
        self.lr_scheduler_update_frequency_val = lr_scheduler_update_frequency

        # Algorithm
        _samplers = {
            "baseline": self._sample_batch_baseline,  # Per-step draw from simple distribution
            "model": self._sample_batch_model,  # Once-per-epoch draw from model distribution
            "bagging": self._sample_batch_bagging,  # Buffer model samples at epoch start for random sampling during the epoch
        }

        if algorithm not in _samplers:
            raise ValueError(f"Unknown algorithm '{algorithm}'.")

        self.algorithm = algorithm
        self.bagging_buffer_size = bagging_buffer_size

        self._batch_sampler = _samplers[self.algorithm]

        if self.algorithm in ["model", "bagging"]:
            self._epoch_buffer_samples: Optional[torch.Tensor] = None
            self._epoch_buffer_likelihood: Optional[torch.Tensor] = None
            self._epoch_ptr: int = 0

        # Temperature annealing parameters
        self.enable_annealing = enable_annealing
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_schedule = temperature_schedule

        # Calculate dependent parameters by scaling the base values from config
        if data_n_train_batches_per_epoch <= 0:
            raise ValueError(
                "data_n_train_batches_per_epoch must be positive for these calculations."
            )

        scaling_factor_numerator = 100
        # Apply scaling: base_value * 100 // num_batches
        self.baseline_until_epoch = (
            baseline_until_epoch * scaling_factor_numerator // data_n_train_batches_per_epoch
        )
        self.annealing_epochs_per_temperature = (
            annealing_epochs_per_temperature
            * scaling_factor_numerator
            // data_n_train_batches_per_epoch
        )
        self.total_annealing_epochs = (
            total_annealing_epochs * scaling_factor_numerator // data_n_train_batches_per_epoch
        )

        # Validate annealing parameters (uses the scaled values)
        self._validate_annealing_parameters()

        # Initialize temperature schedule and state
        if self.enable_annealing:
            self.temperature_values = self._compute_temperature_schedule(
                self.temperature_schedule,
                self.initial_temperature,
                self.final_temperature,
                self.annealing_epochs_per_temperature,
                self.total_annealing_epochs,
                temperature_values,
            )
            self.current_temperature = self.initial_temperature
            self.current_temperature_index = 0
            self.temperature_transition_epoch = 0
        else:
            self.temperature_values = [self.final_temperature]
            self.current_temperature = self.final_temperature
            self.current_temperature_index = 0
            self.temperature_transition_epoch = 0

        # Sample and weight clipping parameters
        self.clipping_method = clipping_method
        self.clipping_percentile = clipping_percentile

        # Validate clipping parameters
        if self.clipping_method is not None:
            valid_methods = ["importance_weight", "energy_value", "modified_energy"]
            if self.clipping_method not in valid_methods:
                raise ValueError(
                    f"clipping_method must be one of {valid_methods} or None, got {self.clipping_method}"
                )
            if not (0.0 < self.clipping_percentile < 100.0):
                raise ValueError(
                    f"clipping_percentile must be between 0 and 100, got {self.clipping_percentile}"
                )

        self.energy_function = energy_function

        self.num_samples_per_batch = num_samples_per_batch
        self.use_train_data = use_train_data
        self.validation_uniform_samples = validation_uniform_samples
        self.likelihood_plot_spacing = likelihood_plot_spacing
        self.flow_num_particles = flow_num_particles
        self.contour_plot_levels = contour_plot_levels
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.val_plot_batch_size = val_plot_batch_size
        self.step_size = step_size
        self.integration_method = integration_method
        self.atol = atol
        self.rtol = rtol
        self.use_exact_divergence = use_exact_divergence
        self.vector_field_max_norm = vector_field_max_norm
        self.enable_detailed_train_logging = enable_detailed_train_logging
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.net = net(energy_function=energy_function)

        # EMA setup
        self.use_ema = use_ema
        if self.use_ema:
            self.net = EMAWrapper(
                self.net,
                decay=ema_beta,
                warmup_denominator=beta_warmup_denominator,
            )

        # Create a torch.device object from the input 'device' string for initial setup
        _init_torch_device = torch.device(device)

        # Distribution used to sample noise, which will be transformed to match the target distribution
        self.p_0 = p_0_prior
        self.q_1_prior = q_1_prior

        # flow matching components
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.solver = ODESolver(velocity_model=ModelWrapper(self.net))

        # general metrics
        self.train_loss = MeanMetric().to(_init_torch_device)
        self.val_loss = MeanMetric().to(_init_torch_device)
        self.val_uniform_loss = MeanMetric().to(_init_torch_device)
        self.test_ess = MeanMetric().to(_init_torch_device)

        # ess metric
        self.val_ess = MeanMetric().to(_init_torch_device)

        # nll metrics
        self.val_nll = MeanMetric().to(_init_torch_device)
        self.test_nll = MeanMetric().to(_init_torch_device)

        # time-stratified metrics for training
        self.time_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        self.train_time_stratified_loss = {
            f"{start:.1f}-{end:.1f}": MeanMetric().to(_init_torch_device)
            for start, end in self.time_bins
        }

        # time-stratified metrics for validation uniform loss
        self.val_uniform_time_stratified_loss = {
            f"{start:.1f}-{end:.1f}": MeanMetric().to(_init_torch_device)
            for start, end in self.time_bins
        }

        # energy-stratified metrics for training
        self.energy_bins = [
            (float("-inf"), 5),
            (5, 15),
            (15, 25),
            (25, 35),
            (35, float("inf")),
        ]
        self.train_energy_stratified_loss = {
            f"{start if start != float('-inf') else '-inf'}-{end if end != float('inf') else 'inf'}": MeanMetric().to(
                _init_torch_device
            )
            for start, end in self.energy_bins
        }

        # energy-stratified metrics for validation
        self.val_energy_stratified_loss = {
            f"{start if start != float('-inf') else '-inf'}-{end if end != float('inf') else 'inf'}": MeanMetric().to(
                _init_torch_device
            )
            for start, end in self.energy_bins
        }

        # bin fraction metrics
        self.train_energy_bin_fractions = {
            f"{start if start != float('-inf') else '-inf'}-{end if end != float('inf') else 'inf'}": MeanMetric().to(
                _init_torch_device
            )
            for start, end in self.energy_bins
        }

        self.val_energy_bin_fractions = {
            f"{start if start != float('-inf') else '-inf'}-{end if end != float('inf') else 'inf'}": MeanMetric().to(
                _init_torch_device
            )
            for start, end in self.energy_bins
        }

        # Batch size used when computing expensive metrics (ESS, NLL). If None, use full set.
        self.metric_batch_size = metric_batch_size

        # Batched sampling configuration
        self.batched_sampling = batched_sampling
        self.sample_batch_size = sample_batch_size

    def _get_sample_batch_size(self) -> Optional[int]:
        """Get the batch size for sample generation, or None for unbatched generation."""
        return self.sample_batch_size if self.batched_sampling else None

    def _create_time_grid(
        self,
        start_time: float,
        end_time: float,
        step_size: float,
        device: torch.device,
        integration_method: str,
    ) -> torch.Tensor:
        if integration_method == "dopri5":
            return torch.tensor([start_time, end_time], device=device)

        num_steps = math.ceil(abs(end_time - start_time) / step_size)
        if num_steps == 0:
            # Ensure at least two points for linspace if start and end are too close for the step_size
            return torch.tensor([start_time, end_time], device=device)
        return torch.linspace(start_time, end_time, num_steps + 1, device=device)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(t, x)

    def _calculate_flow_matching_loss(
        self, x_1: torch.Tensor, weights: torch.Tensor, return_components: bool = False
    ) -> torch.Tensor:
        """
        Calculate flow matching loss for the given target samples and weights.

        Args:
            x_1: Target samples (batch_size, dim)
            weights: Sample weights (batch_size,)
            return_components: If True, return loss components separately

        Returns:
            The weighted loss value, or (loss, loss_vec, v_pred, dx_t, t) if return_components=True
        """
        # Sample normal and uniform time
        x_0 = self.p_0.sample(x_1.shape[0]).to(self.device)

        t = torch.rand(x_1.shape[0], device=self.device)

        # Sample path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t, dx_t = path_sample.x_t, path_sample.dx_t

        # Apply vector field clipping if max_norm is specified
        if self.vector_field_max_norm is not None:
            dx_t_norm = torch.norm(dx_t, dim=1, keepdim=True)
            scaling_factor = torch.minimum(
                torch.ones_like(dx_t_norm), self.vector_field_max_norm / (dx_t_norm + 1e-8)
            )
            dx_t = dx_t * scaling_factor

        # Model prediction and loss
        v_pred = self.net(t, x_t)
        loss_vec = (v_pred - dx_t).pow(2).sum(dim=1)
        loss = (weights * loss_vec).sum()

        if return_components:
            return loss, loss_vec, v_pred, dx_t, t
        return loss

    def _apply_temperature_scaling(self, energies: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to energies: E'(x) = (T_final/T_current) * E(x)"""
        if not self.enable_annealing or self._is_annealing_complete():
            return energies

        scaling_factor = self.final_temperature / self.current_temperature
        return energies * scaling_factor

    def _apply_clipping(
        self, energies: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sample and weight clipping based on the configured method.

        Args:
            energies: Energy values (raw, modified, or temperature-scaled depending on method)
            weights: Importance weights (only used for importance_weight clipping method)

        Returns:
            Tuple of (clipped_energies, clipped_weights)
        """
        if self.clipping_method in ["energy_value", "modified_energy"]:
            # Clip energy values from below at the specified percentile
            # We want to clip the lowest x% of energy values (which have highest weights)
            threshold = torch.quantile(energies, (100.0 - self.clipping_percentile) / 100.0)
            clipped_energies = torch.clamp(energies, min=threshold)
            clipped_weights = torch.softmax(-clipped_energies, dim=0)
            return clipped_energies, clipped_weights

        elif self.clipping_method == "importance_weight":
            # First compute weights if not provided
            if weights is None:
                weights = torch.softmax(-energies, dim=0)
            # Clip importance weights from above at the specified percentile, then renormalize
            threshold = torch.quantile(weights, self.clipping_percentile / 100.0)
            clipped_weights = torch.clamp(weights, max=threshold)
            # Renormalize to sum to 1
            clipped_weights = clipped_weights / clipped_weights.sum()
            return energies, clipped_weights

        else:
            raise ValueError(f"Unknown clipping method: {self.clipping_method}")

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_1, energies, w = self._batch_sampler()

        if self.enable_detailed_train_logging:
            # Log the highest 3 weights into the log (not a metric)
            top_3_weights_values = torch.topk(w, min(3, w.size(0))).values
            weights_dict_to_log = {
                f"train/top_weight_{i}": top_3_weights_values[i]
                for i in range(top_3_weights_values.shape[0])
            }
            self.log_dict(weights_dict_to_log, on_step=False, on_epoch=True)

        # Calculate loss with components to log top losses and stratified metrics
        loss, loss_vec, _, _, t = self._calculate_flow_matching_loss(
            x_1, w, return_components=True
        )

        # Add temperature-specific logging
        if self.enable_annealing:
            self.log("annealing/current_temperature", self.current_temperature, on_epoch=True)
            self.log(
                "annealing/temperature_progress",
                self.current_temperature_index / max(1, len(self.temperature_values) - 1),
                on_epoch=True,
            )
            if self.enable_detailed_train_logging:
                self.log(
                    f"annealing/loss_T_{self.current_temperature:.3f}",
                    loss.detach(),
                    on_epoch=True,
                )

        if self.enable_detailed_train_logging:
            # Log the highest 3 loss values
            top_3_losses_values = torch.topk(loss_vec, min(3, loss_vec.size(0))).values
            losses_dict_to_log = {
                f"train/top_loss_{i}": top_3_losses_values[i]
                for i in range(top_3_losses_values.shape[0])
            }
            self.log_dict(losses_dict_to_log, on_step=False, on_epoch=True)

            # Calculate and log time-stratified losses
            self._update_time_stratified_losses(t, loss_vec, w, "train")

            # Calculate and log energy-stratified losses
            self._update_energy_stratified_losses(energies, loss_vec, w, "train")

        self.train_loss.update(loss.detach())
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step to handle EMA updates."""
        optimizer.step(closure=optimizer_closure)

        # Update EMA if enabled
        if self.use_ema:
            self.net.update_ema()

    def on_train_epoch_start(self) -> None:
        # Handle temperature transitions
        if self.enable_annealing:
            self._update_temperature_state()

        if (
            self.algorithm not in ["model", "bagging"]
            or self.trainer.current_epoch < self.baseline_until_epoch
        ):
            return

        dim = self.energy_function.dimensionality
        if self.algorithm == "model":
            # Draw all samples for the next epoch from the current model distribution
            num_batches = self.trainer.num_training_batches
            num_samples = num_batches * self.num_samples_per_batch

            batch_size = self._get_sample_batch_size()
            self._epoch_buffer_samples = self.generate_samples(num_samples, batch_size).detach()
            self._epoch_ptr = 0

        elif self.algorithm == "bagging":
            batch_size = self._get_sample_batch_size()
            self._epoch_buffer_samples = self.generate_samples(
                self.bagging_buffer_size, batch_size
            ).detach()

            t_grid = self._create_time_grid(
                1.0, 0.0, self.step_size, self.device, self.integration_method
            )

            # Compute likelihood in batches if batched sampling is enabled
            if self.batched_sampling:
                log_q_parts = []
                for start in range(0, self.bagging_buffer_size, self.sample_batch_size):
                    end = min(start + self.sample_batch_size, self.bagging_buffer_size)
                    batch_samples = self._epoch_buffer_samples[start:end]

                    _, log_q_batch = self.solver.compute_likelihood(
                        x_1=batch_samples,
                        time_grid=t_grid,
                        log_p0=self.p_0.log_prob,
                        method=self.integration_method,
                        step_size=(
                            self.step_size if self.integration_method != "dopri5" else None
                        ),
                        atol=self.atol,
                        rtol=self.rtol,
                    )
                    log_q_parts.append(log_q_batch)

                log_q = torch.cat(log_q_parts, dim=0)
            else:
                # Compute all at once (original behavior)
                _, log_q = self.solver.compute_likelihood(
                    x_1=self._epoch_buffer_samples,
                    time_grid=t_grid,
                    log_p0=self.p_0.log_prob,
                    method=self.integration_method,
                    step_size=(self.step_size if self.integration_method != "dopri5" else None),
                    atol=self.atol,
                    rtol=self.rtol,
                )

            self._epoch_buffer_likelihood = log_q

    def _sample_batch_baseline(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_train_data:
            # Use the energy function's sample_train_set method to get training samples
            x_1 = self.energy_function.sample_train_set(self.num_samples_per_batch).to(self.device)
            energies = self.energy_function(x_1).to(self.device)

            # Use uniform weights for standard flow matching when training on the training set directly
            w = torch.full_like(energies, 1.0 / energies.shape[0])
            return x_1, energies, w

        if self.q_1_prior is None:
            raise ValueError(
                "q_1_prior must be provided to __init__ when using 'custom' train_sampling_strategy."
            )
        x_1 = self.q_1_prior.sample(self.num_samples_per_batch).to(self.device)
        energies = self.energy_function(x_1).to(self.device)

        # Apply energy_value clipping before temperature scaling if specified
        if self.clipping_method == "energy_value":
            energies, _ = self._apply_clipping(energies)

        energies = self._apply_temperature_scaling(energies)
        log_prob_q1 = self.q_1_prior.log_prob(x_1).to(self.device)
        energies = energies + log_prob_q1

        # Apply final clipping if needed
        if self.clipping_method in ["modified_energy", "importance_weight"]:
            energies, w = self._apply_clipping(energies)
        else:
            w = torch.softmax(-energies, dim=0)

        return x_1, energies, w

    def _sample_batch_model(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use baseline sampling during initial epochs
        if self.trainer.current_epoch < self.baseline_until_epoch:
            return self._sample_batch_baseline()

        start = self._epoch_ptr
        end = start + self.num_samples_per_batch
        self._epoch_ptr = end

        # 1) Generate samples from the model
        x_1 = self._epoch_buffer_samples[start:end]

        # 2) compute log q_theta(x0)
        t_grid = self._create_time_grid(
            1.0, 0.0, self.step_size, x_1.device, self.integration_method
        )
        _, log_q = self.solver.compute_likelihood(
            x_1=x_1,
            time_grid=t_grid,
            log_p0=self.p_0.log_prob,
            method=self.integration_method,
            step_size=(self.step_size if self.integration_method != "dopri5" else None),
            atol=self.atol,
            rtol=self.rtol,
            exact_divergence=self.use_exact_divergence,
        )

        # Log the distribution of log_q values
        if self.enable_detailed_train_logging:
            self._log_log_q_distribution_plot(
                log_q_values=log_q.detach(),
                wandb_log_key="train/log_q_distribution",
                plot_title="Distribution of log_q in Training",
            )

        # 3) compute modified energy E'(x0)
        energies = self.energy_function(x_1).to(self.device)

        # Apply energy_value clipping before temperature scaling if specified
        if self.clipping_method == "energy_value":
            energies, _ = self._apply_clipping(energies)

        energies = self._apply_temperature_scaling(energies)
        energies = energies + log_q

        # Apply final clipping if needed
        if self.clipping_method in ["modified_energy", "importance_weight"]:
            energies, w = self._apply_clipping(energies)
        else:
            w = torch.softmax(-energies, dim=0)

        return x_1, energies, w

    def _sample_batch_bagging(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use baseline sampling during initial epochs
        if self.trainer.current_epoch < self.baseline_until_epoch:
            return self._sample_batch_baseline()  #

        # Draw random samples from buffer
        indices = torch.randint(0, self.bagging_buffer_size, (self.num_samples_per_batch,))
        x_1 = self._epoch_buffer_samples[indices]
        log_q = self._epoch_buffer_likelihood[indices]

        # Log the distribution of log_q values
        if self.enable_detailed_train_logging:
            self._log_log_q_distribution_plot(
                log_q_values=log_q.detach(),
                wandb_log_key="train/log_q_distribution",
                plot_title="Distribution of log_q in Training",
            )

        # Compute modified energy E'(x0)
        energies = self.energy_function(x_1).to(self.device)

        # Apply energy_value clipping before temperature scaling if specified
        if self.clipping_method == "energy_value":
            energies, _ = self._apply_clipping(energies)

        energies = self._apply_temperature_scaling(energies)
        energies = energies + log_q

        # Apply final clipping if needed
        if self.clipping_method in ["modified_energy", "importance_weight"]:
            energies, w = self._apply_clipping(energies)
        else:
            w = torch.softmax(-energies, dim=0)

        return x_1, energies, w

    def _update_time_stratified_losses(
        self, t: torch.Tensor, loss_vec: torch.Tensor, weights: torch.Tensor, prefix: str
    ) -> None:
        """Update time-stratified loss metrics."""
        metric_dict = (
            self.train_time_stratified_loss
            if prefix == "train"
            else self.val_uniform_time_stratified_loss
        )

        for (t_min, t_max), metric_key in zip(self.time_bins, metric_dict.keys()):
            # Create mask for samples in this time bin
            mask = (t >= t_min) & (t < t_max)
            if mask.sum() > 0:
                # Calculate weighted loss for samples in this bin
                bin_loss = (weights[mask] * loss_vec[mask]).sum() / weights[mask].sum()
                metric_dict[metric_key].update(bin_loss.detach())
                self.log(
                    f"{prefix}/time_loss/{metric_key}",
                    bin_loss.detach(),
                    on_step=False,
                    on_epoch=True,
                )

    def _update_energy_stratified_losses(
        self,
        energies: torch.Tensor,
        loss_vec: torch.Tensor,
        weights: torch.Tensor,
        prefix: str = "train",
    ) -> None:
        """Update energy-stratified loss metrics based on energy values.

        Args:
            energies: Energy values for each sample
            loss_vec: Loss values for each sample
            weights: Sample weights for weighted loss calculation
            prefix: Prefix for logging ("train" or "val")
        """
        metric_dict = (
            self.train_energy_stratified_loss
            if prefix == "train"
            else self.val_energy_stratified_loss
        )

        fraction_dict = (
            self.train_energy_bin_fractions if prefix == "train" else self.val_energy_bin_fractions
        )

        total_samples = energies.shape[0]

        for (e_min, e_max), metric_key in zip(self.energy_bins, metric_dict.keys()):
            # Create mask for samples with energies in this bin
            mask = (energies >= e_min) & (energies < e_max)
            bin_count = mask.sum().item()

            # Log fraction of samples in this bin
            bin_fraction = bin_count / total_samples if total_samples > 0 else 0.0
            bin_fraction_tensor = torch.tensor(bin_fraction, device=self.device)
            fraction_dict[metric_key].update(bin_fraction_tensor)
            self.log(
                f"{prefix}/energy_bin_fraction/{metric_key}",
                bin_fraction,
                on_step=False,
                on_epoch=True,
            )

            if bin_count > 0:
                # Calculate weighted loss for samples in this bin
                if weights is not None:
                    bin_loss = (weights[mask] * loss_vec[mask]).sum() / weights[mask].sum()

                metric_dict[metric_key].update(bin_loss.detach())
                self.log(
                    f"{prefix}/energy_loss/{metric_key}",
                    bin_loss.detach(),
                    on_step=False,
                    on_epoch=True,
                )

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # Generate samples for visualization
        generated_samples_for_plot = self.generate_samples(self.val_plot_batch_size)
        self.energy_function.log_samples(
            generated_samples_for_plot,
            self._get_wandb_logger(),
            should_unnormalize=True,
            name="val/samples",
        )

        generated_samples_for_metrics = self.generate_samples(self.eval_batch_size)

        true_samples_for_metrics = self.energy_function.sample_val_set(self.eval_batch_size)
        true_samples_for_metrics = true_samples_for_metrics.to(self.device)
        true_samples_for_metrics = self.energy_function.normalize(true_samples_for_metrics)

        self.compute_and_log_metric_ess(true_samples_for_metrics, "val")

        self.compute_and_log_metric_wasserstein_distance(
            generated_samples_for_metrics, true_samples_for_metrics, "val"
        )

        # Compute and log NLL
        self.compute_and_log_nll(true_samples_for_metrics, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # First plot a scatter plot of the generated samples
        generated_samples = self.generate_samples(self.test_batch_size)
        self.energy_function.log_samples(
            generated_samples,
            self._get_wandb_logger(),
            should_unnormalize=True,
            name="val/gmm_overview_plot",
        )

        # Then compute the ESS of the test samples
        samples = self.energy_function.sample_test_set(self.test_batch_size)
        samples = samples.to(self.device)
        samples = self.energy_function.normalize(samples)

        self.compute_and_log_metric_ess(samples, "test")

        self.compute_and_log_metric_wasserstein_distance(generated_samples, samples, "test")

        # Compute and log NLL
        self.compute_and_log_nll(samples, "test")

    def compute_and_log_metric_wasserstein_distance(
        self, generated_samples: torch.Tensor, samples: torch.Tensor, prefix: str
    ) -> None:
        pred_2d = self.energy_function.unnormalize(generated_samples)
        true_2d = self.energy_function.unnormalize(samples)

        wasserstein_value = wasserstein(pred_2d, true_2d, power=2)

        self.log(
            f"{prefix}/wasserstein_2_distance", wasserstein_value, on_epoch=True, prog_bar=True
        )

    def compute_and_log_metric_ess(self, samples, prefix) -> None:
        # Determine batch size for likelihood computation
        batch_size = (
            self.metric_batch_size
            if self.metric_batch_size is not None and self.metric_batch_size > 0
            else samples.shape[0]
        )

        logq_parts = []
        # Compute logq in smaller chunks if requested
        for start in range(0, samples.shape[0], batch_size):
            end = start + batch_size
            batch_samples = samples[start:end]
            t_eval_likelihood = self._create_time_grid(
                1.0, 0.0, self.step_size, batch_samples.device, self.integration_method
            )
            _, logq_batch = self.solver.compute_likelihood(
                x_1=batch_samples,
                method=self.integration_method,
                step_size=self.step_size if self.integration_method != "dopri5" else None,
                time_grid=t_eval_likelihood,
                log_p0=self.p_0.log_prob,
                atol=self.atol,
                rtol=self.rtol,
                exact_divergence=self.use_exact_divergence,
            )
            logq_parts.append(logq_batch)

        logq = torch.cat(logq_parts, dim=0)

        logp = self.energy_function(samples)

        # Plot the samples used for ESS calculation
        self.energy_function.log_samples(
            samples,
            self._get_wandb_logger(),
            should_unnormalize=True,
            name=f"{prefix}/ess_samples",
        )

        ess = (torch.nn.functional.softmax(-logp - logq, dim=0) ** 2).sum().pow(-1)

        ess = ess / samples.shape[0]
        ess_metric = getattr(self, f"{prefix}_ess")
        ess_metric.update(ess)
        self.log(
            f"{prefix}/ess",
            ess_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def generate_samples(self, num_samples: int, batch_size: int = None) -> torch.Tensor:
        """Generate samples from the model.

        Args:
            num_samples: Total number of samples to generate
            batch_size: If provided, generate samples in batches of this size to manage memory

        Returns:
            Generated samples tensor of shape (num_samples, ...)
        """
        if batch_size is None or batch_size >= num_samples:
            # Generate all samples at once (original behavior)
            x_0 = self.p_0.sample(num_samples).to(self.device)

            t_eval_sample = self._create_time_grid(
                0.0, 1.0, self.step_size, x_0.device, self.integration_method
            )
            x_1 = self.solver.sample(
                x_init=x_0,
                method=self.integration_method,
                step_size=self.step_size if self.integration_method != "dopri5" else None,
                time_grid=t_eval_sample,
                atol=self.atol,
                rtol=self.rtol,
            )

            return x_1
        else:
            # Generate samples in batches
            all_samples = []

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                current_batch_size = end_idx - start_idx

                x_0_batch = self.p_0.sample(current_batch_size).to(self.device)

                t_eval_sample = self._create_time_grid(
                    0.0, 1.0, self.step_size, x_0_batch.device, self.integration_method
                )
                x_1_batch = self.solver.sample(
                    x_init=x_0_batch,
                    method=self.integration_method,
                    step_size=self.step_size if self.integration_method != "dopri5" else None,
                    time_grid=t_eval_sample,
                    atol=self.atol,
                    rtol=self.rtol,
                )

                all_samples.append(x_1_batch)

            # Concatenate all batches
            return torch.cat(all_samples, dim=0)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer_ctor(params=self.trainer.model.parameters())
        if self.scheduler_ctor is not None:
            scheduler = self.scheduler_ctor(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # NOTE: This is different from the behaviour in the iDEM paper. There they use the val/loss.
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": self.lr_scheduler_update_frequency_val,
                },
            }
        return {"optimizer": optimizer}

    def _get_wandb_logger(self) -> Optional[WandbLogger]:
        if self.trainer is None:
            return None
        for lg in self.trainer.loggers:
            if isinstance(lg, WandbLogger):
                return lg
        return None

    def _log_log_q_distribution_plot(
        self, log_q_values: torch.Tensor, wandb_log_key: str, plot_title: str
    ) -> None:
        """Logs a histogram of the given log_q values."""
        wandb_logger = self._get_wandb_logger()
        if not wandb_logger:
            return

        fig, ax = plt.subplots()
        ax.hist(log_q_values.cpu().numpy(), bins="auto", density=True)
        ax.set_title(plot_title)
        ax.set_xlabel("Log Likelihood (log_q)")
        ax.set_ylabel("Density")

        img = fig_to_image(fig)
        wandb_logger.log_image(wandb_log_key, [img])
        plt.close(fig)

    def compute_and_log_nll(self, samples_x1: torch.Tensor, prefix: str) -> None:
        """
        Computes and logs the Negative Log-Likelihood (NLL).

        Args:
            samples_x1: Samples from the data distribution (x_1).
            prefix: Prefix for logging (e.g., "val" or "test").
        """
        nll_metric = getattr(self, f"{prefix}_nll")

        # Reset metric at the beginning of computation for an epoch
        nll_metric.reset()

        total_samples = samples_x1.shape[0]
        if total_samples == 0:
            self.log(
                f"{prefix}/nll",
                torch.tensor(float("nan")),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return

        batch_size = (
            self.metric_batch_size
            if self.metric_batch_size is not None and self.metric_batch_size > 0
            else total_samples
        )

        nll_sum = 0.0  # Accumulate as python float to avoid tensor/device issues

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_samples = samples_x1[start:end]

            t_eval_likelihood = self._create_time_grid(
                1.0, 0.0, self.step_size, batch_samples.device, self.integration_method
            )

            # compute_likelihood returns: x_0_reconstructed, log_q_values
            _, log_q_for_samples = self.solver.compute_likelihood(
                x_1=batch_samples,
                method=self.integration_method,
                step_size=self.step_size if self.integration_method != "dopri5" else None,
                time_grid=t_eval_likelihood,
                log_p0=self.p_0.log_prob,
                atol=self.atol,
                rtol=self.rtol,
                exact_divergence=self.use_exact_divergence,
            )

            nll_for_samples = -log_q_for_samples  # NLL = -log p_model(x_1)

            if isinstance(self.energy_function, GMMEnergy) and hasattr(
                self.energy_function, "data_normalization_factor"
            ):
                norm_factor = torch.tensor(
                    self.energy_function.data_normalization_factor,
                    device=batch_samples.device,
                    dtype=batch_samples.dtype,
                )
                nll_for_samples = nll_for_samples + 2 * torch.log(norm_factor)

            nll_sum += nll_for_samples.sum().item()

        # Compute mean NLL across all samples
        mean_nll = nll_sum / float(total_samples)

        nll_metric.update(mean_nll)

        self.log(
            f"{prefix}/nll",
            nll_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _validate_annealing_parameters(self) -> None:
        """Validate annealing configuration parameters."""
        if not self.enable_annealing:
            return

        if self.initial_temperature <= 0 or self.final_temperature <= 0:
            raise ValueError("Temperatures must be positive")

        if self.initial_temperature < self.final_temperature:
            raise ValueError("Initial temperature should be >= final temperature for annealing")

        if self.annealing_epochs_per_temperature <= 0:
            raise ValueError("annealing_epochs_per_temperature must be positive")

        if self.total_annealing_epochs <= 0:
            raise ValueError("total_annealing_epochs must be positive")

        if self.temperature_schedule not in ["geometric", "linear"]:
            raise ValueError("temperature_schedule must be 'geometric' or 'linear'")

    def _compute_temperature_schedule(
        self,
        schedule: str,
        initial_temperature: float,
        final_temperature: float,
        annealing_epochs_per_temperature: int,
        total_annealing_epochs: int,
        temperature_values: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute temperature schedule based on parameters."""
        if temperature_values is not None:
            return temperature_values

        n_temperatures = total_annealing_epochs // annealing_epochs_per_temperature

        if schedule == "geometric":
            ratio = (final_temperature / initial_temperature) ** (1 / (n_temperatures - 1))
            return [initial_temperature * (ratio**i) for i in range(n_temperatures)]
        elif schedule == "linear":
            return np.linspace(initial_temperature, final_temperature, n_temperatures).tolist()
        else:
            raise ValueError(f"Unknown temperature schedule: {schedule}")

    def _update_temperature_state(self) -> None:
        """Update current temperature based on training progress."""
        if not self.enable_annealing:
            return

        epoch = self.trainer.current_epoch
        epochs_at_current_temp = epoch - self.temperature_transition_epoch

        # Check if we need to transition to next temperature
        if (
            epochs_at_current_temp >= self.annealing_epochs_per_temperature
            and self.current_temperature_index < len(self.temperature_values) - 1
        ):
            # Transition to next temperature
            self.current_temperature_index += 1
            self.current_temperature = self.temperature_values[self.current_temperature_index]
            self.temperature_transition_epoch = epoch

            # Log temperature transition
            self.log("annealing/current_temperature", self.current_temperature, on_epoch=True)
            self.log("annealing/temperature_index", self.current_temperature_index, on_epoch=True)

            if self.enable_detailed_train_logging:
                print(
                    f"Epoch {epoch}: Transitioning to temperature {self.current_temperature:.4f}"
                )

    def _is_annealing_complete(self) -> bool:
        """Check if annealing process is complete."""
        return (
            not self.enable_annealing
            or self.current_temperature_index >= len(self.temperature_values) - 1
        )
