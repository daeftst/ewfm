from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from fab.target_distributions import gmm
from fab.utils.plotting import plot_contours, plot_marginal_pair
from lightning.pytorch.loggers import WandbLogger

from ewfm.energies.base_energy_function import BaseEnergyFunction
from ewfm.utils.logging_utils import fig_to_image


class GMMEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        device="cuda",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=50,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
    ):
        self.device = torch.device(device)

        use_gpu = self.device.type != "cpu"

        # seed for reproducibility in GMM sampling
        torch.manual_seed(0)

        # initialize GMM distribution and move to correct device
        self.gmm = gmm.GMM(
            dim=dimensionality,
            n_mixes=n_mixes,
            loc_scaling=loc_scaling,
            log_var_scaling=log_var_scaling,
            use_gpu=use_gpu,
            true_expectation_estimation_n_samples=true_expectation_estimation_n_samples,
        )

        self.gmm.to(self.device)

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.data_path_train = data_path_train

        self.name = "gmm"

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        return self.gmm.sample((self.test_set_size,))

    def setup_train_set(self):
        if self.data_path_train is None:
            train_samples = self.normalize(self.gmm.sample((self.train_set_size,)))

        else:
            # Assume the samples we are loading from disk are already normalized.
            # This breaks if they are not.

            if self.data_path_train.endswith(".pt"):
                data = torch.load(self.data_path_train).cpu().numpy()
            else:
                data = np.load(self.data_path_train, allow_pickle=True)
            train_samples = torch.tensor(data, device=self.device)
        return train_samples

    def setup_val_set(self):
        return self.gmm.sample((self.val_set_size,))

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # Ensure samples are on the correct device
        samples = samples.to(self.device)

        self.gmm.to(self.device)
        self.gmm.locs = self.gmm.locs.to(self.device)
        self.gmm.scale_trils = self.gmm.scale_trils.to(self.device)
        self.gmm.cat_probs = self.gmm.cat_probs.to(self.device)
        self.gmm.distribution.mixture_distribution.probs = (
            self.gmm.distribution.mixture_distribution.probs.to(self.device)
        )
        self.gmm.distribution.component_distribution.loc = (
            self.gmm.distribution.component_distribution.loc.to(self.device)
        )
        self.gmm.distribution.component_distribution.scale_tril = (
            self.gmm.distribution.component_distribution.scale_tril.to(self.device)
        )

        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        return -self.gmm.log_prob(samples)

    @property
    def dimensionality(self):
        return 2

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        prefix: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if self.should_unnormalize and latest_samples is not None:
                latest_samples = self.unnormalize(latest_samples)

            if latest_samples is not None:
                fig, ax = plt.subplots()
                ax.scatter(*latest_samples.detach().cpu().T)

                wandb_logger.log_image(f"{prefix}generated_samples_scatter", [fig_to_image(fig)])
                img = self.get_single_dataset_fig(latest_samples, "ewfm_generated_samples")
                wandb_logger.log_image(f"{prefix}generated_samples", [img])

            plt.close()

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        if wandb_logger is None:
            return

        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_single_dataset_fig(self, samples, name, plotting_bounds=(-1.4 * 40, 1.4 * 40)):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # scatter plot
        # temporarily move distribution to CPU for plotting
        self.gmm.to("cpu")
        plot_contours(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)

        # move back to original device
        self.gmm.to(self.device)

        # energy histogram
        test_samples = self.sample_test_set(1000)

        energy_samples = self(samples).detach().detach().cpu()
        energy_test_samples = self(test_samples).detach().detach().cpu()

        min_energy = 0
        max_energy = 100000

        axs[1].hist(
            energy_test_samples.cpu(),
            bins=25,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=25,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        return fig_to_image(fig)
