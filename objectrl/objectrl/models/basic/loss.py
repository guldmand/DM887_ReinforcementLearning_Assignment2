# -----------------------------------------------------------------------------------
# ObjectRL: An Object-Oriented Reinforcement Learning Codebase
# Copyright (C) 2025 ADIN Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------------

import typing
from abc import ABC, abstractmethod

import torch
from torch.nn.modules.loss import _Loss

from objectrl.utils.utils import dim_check

if typing.TYPE_CHECKING:
    from objectrl.config.model_configs.dsac import DSACConfig
    from objectrl.config.model_configs.pbac import PBACConfig


class ProbabilisticLoss(_Loss, ABC):
    """
    Base class for probabilistic loss functions.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. Default: 'mean'.
    Attributes:
        reduction (str): Reduction method for the loss.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.reduction = reduction

    @abstractmethod
    def forward(self, mu_lvar_dict: dict, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute loss (to be implemented in subclasses).

        Args:
            mu_lvar_dict (dict): Predicted mean and log_variance tensors
            y (Tensor): Target tensor.
        Returns:
            Tensor: Computed loss.
        """
        pass

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified reduction to the loss tensor.

        Args:
            loss: Tensor of loss values.
        Returns:
            Tensor: Reduced loss tensor based on the specified reduction method.
        Raises:
            ValueError: If an unknown reduction method is specified.
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# class PACBayesLoss(ProbabilisticLoss):
#     """
#     PAC-Bayesian loss combining empirical risk and complexity term.

#     Args:
#         config: Configuration object with loss parameters:
#         - lossparams.reduction (str): Reduction method.
#         - lossparams.logvar_lower_clamp (float): Lower clamp for log variance.
#         - lossparams.logvar_upper_clamp (float): Upper clamp for log variance.
#         - lossparams.complexity_coef (float): Coefficient for complexity term.
#     Attributes:
#         logvar_lower_clamp (float): Lower clamp for log variance.
#         logvar_upper_clamp (float): Upper clamp for log variance.
#         complexity_coef (float): Coefficient for complexity term.
#     """

#     def __init__(self, config: "PBACConfig"):
#         """
#         Initialize PACBayesLoss with configuration parameters.
#         """
#         super().__init__(reduction=config.lossparams.reduction)
#         self.logvar_lower_clamp = config.lossparams.logvar_lower_clamp
#         self.logvar_upper_clamp = config.lossparams.logvar_upper_clamp
#         self.complexity_coef = config.lossparams.complexity_coef

#     def forward(self, mu_lvar_dict: dict, y: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the PAC-Bayes loss.

#         Args:
#             mu_lvar_dict (dict): Dictionary with keys "mu" (mean) and "lvar" (log variance) tensors.
#             y (Tensor): Target tensor with shape [..., 2], where last dimension holds
#             true mean and true variance (mu_t, sig2_t).
#         Returns:
#             Tensor: Computed PAC-Bayes loss.
#         """
#         mu_t = y[:, :, 0]
#         sig2_t = y[:, :, 1]
#         mu, logvar = mu_lvar_dict["mu"], mu_lvar_dict["lvar"]
#         sig2 = logvar.exp().clamp(self.logvar_lower_clamp, self.logvar_upper_clamp)

#         # KL divergence term between predicted and true distributions
#         sig_ratio = sig2 / sig2_t
#         kl_vals = 0.5 * (sig_ratio - sig_ratio.log() + (mu - mu_t) ** 2 / sig2_t - 1)

#         # Empirical risk (expected squared error plus predicted variance)
#         empirical_risk = ((mu - mu_t) ** 2 + sig2).mean(-1)

#         # Complexity regularization scaled by coefficient
#         complexity = kl_vals.mean(-1) * self.complexity_coef

#         q_loss = empirical_risk + complexity
#         return q_loss.sum()


class PACBayesLoss(ProbabilisticLoss):
    """
    Implements PAC-Bayesian loss for critic training using uncertainty-aware estimates.
    Computes a PAC-Bayes bound-based Q-learning loss that penalizes uncertainty
    and uses bootstrapping for improved generalization.
    """

    def __init__(self, config: "PBACConfig"):
        """
        Args:
            config (PBACConfig): Configuration object containing model settings.
        """
        super().__init__(reduction=config.lossparams.reduction)
        self.prior_variance = config.lossparams.prior_variance
        self.bootstrap_rate = config.lossparams.bootstrap_rate
        self.gamma = config.lossparams.gamma
        self.sig2_lower_clamp = config.lossparams.sig2_lower_clamp

    def forward(
        self, q: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Computes the PAC-Bayes loss between predicted Q-values and targets.

        Args:
            q (Tensor): Predicted Q-values (ensemble shape: [ensemble, batch]).
            y (Tensor): Target Q-values (shape: [ensemble, batch]).
            weights (Tensor, optional): Sample weights (unused here).
        Returns:
            Tensor: Loss scalar.
        """
        mu_0 = y.mean(dim=0)
        sig2_0 = self.prior_variance

        bootstrap_mask = (torch.rand_like(q) >= self.bootstrap_rate) * 1.0
        sig2 = (q * bootstrap_mask).var(dim=0).clamp(self.sig2_lower_clamp, None)
        logsig2 = sig2.log()

        err_0 = (q - mu_0) * bootstrap_mask
        term1 = -0.5 * logsig2
        term2 = 0.5 * (err_0.pow(2)).mean(dim=0) / sig2_0
        kl_term = term1 + term2

        var_offset = -self.gamma**2 * logsig2
        emp_loss = ((q - y) * bootstrap_mask).pow(2)
        q_loss = emp_loss + kl_term + var_offset

        return self._apply_reduction(q_loss)


class DSACLoss(_Loss):
    """
    Distributional Soft Actor-Critic (DSAC) Loss Function.

    Args:
        config: Configuration object with loss parameters
    Attributes:
        _kappa (float): Huber loss threshold.
    """

    def __init__(self, config: "DSACConfig"):
        super().__init__()
        self._kappa = config.lossparams.kappa

    @torch.compile
    def vec_asymmetric_huber_loss_weighted(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tau: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized Asymmetric Huber Loss with weights.

        Args:
            pred (Tensor): Predicted quantiles: [n_member x n_batch x n_quantiles].
            target (Tensor): Target quantiles: [n_member x n_batch x n_quantiles].
            tau (Tensor): Quantile levels: [n_quantiles] or [n_batch x n_quantiles].
            weight (Tensor): Weights for each quantile: [n_quantiles] or [n_batch x n_quantiles].
        Returns:
            Tensor: Computed loss tensor.
        """
        dim_check(pred, target)
        pred = pred.unsqueeze(3)  # [... n_quantiles x 1]
        target = target.unsqueeze(2)  # [... 1 x n_quantiles]

        if len(tau.shape) == 1:
            assert tau.shape[0] == pred.shape[2]
            tau = tau.view(1, 1, tau.shape[0], 1)  # [1 x 1 x n_quantiles x 1]
        elif len(tau.shape) == 2:  # In this case we have samples for every quantile
            assert tau.shape[0] == pred.shape[1]

            tau = tau.view(1, tau.shape[0], tau.shape[1], 1)

        if len(weight.shape) == 1:
            assert weight.shape[0] == pred.shape[2]
            weight = weight.view(1, 1, 1, weight.shape[0])
        elif len(weight.shape) == 2:  # In this case we have samples for every quantile
            assert weight.shape[0] == pred.shape[1]
            weight = weight.view(
                1, weight.shape[0], 1, weight.shape[1]
            )  # [1 x n_batch x 1 x n_quantiles]

        u = target - pred  # [n_member x n_batch x n_quantiles x n_quantiles]
        abs_u = torch.abs(u)
        huber = torch.where(
            abs_u <= self._kappa,
            0.5 * u.pow(2),
            self._kappa * (abs_u - 0.5 * self._kappa),
        )
        loss = (
            (1.0 / u.shape[-1])
            * (torch.abs(tau - (u < 0).float()) * huber / self._kappa)
            * weight
        )

        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tau: torch.Tensor,
        target_tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the DSAC loss.

        Args:
            pred (Tensor): Predicted quantiles: [n_member x n_batch x n_quantiles].
            target (Tensor): Target quantiles: [n_member x n_batch x n_quantiles].
            tau (Tensor): Quantile levels: [n_quantiles] or [n_batch x n_quantiles].
            target_tau (Tensor): Target quantile levels: [n_quantiles] or [n_batch x n_quantiles].
        Returns:
            Tensor: Computed loss tensor.
        """
        return self.vec_asymmetric_huber_loss_weighted(pred, target, tau, target_tau)
