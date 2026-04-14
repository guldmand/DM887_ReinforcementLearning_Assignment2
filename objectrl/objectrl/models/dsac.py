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

import math
import typing

import torch

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.models.sac import SACActor, SACCritic

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class DSACActor(SACActor):
    """
    Distributional Soft Actor-Critic (DSAC) Actor.

    Args:
        config (MainConfig): Configuration object containing model specifications.
        dim_state (int): Dimension of the state space.
        dim_act (int): Dimension of the action space.
    Attributes:
        learnable_alpha (bool): Indicates if the temperature parameter alpha is learnable.
    """

    def __init__(self, config, dim_state, dim_act):

        super().__init__(config, dim_state, dim_act)
        self.learnable_alpha = config.model.learnable_alpha

        if not self.learnable_alpha:
            # Fix log_alpha as a non-learnable tensor (constant)
            self.log_alpha = torch.tensor(
                math.log(config.model.alpha),
                requires_grad=False,
                device=self.device,
            )
            # No optimizer for alpha
            del self.optim_alpha
            self.optim_alpha = None

    def update_alpha(self, act_dict: dict) -> None:
        """
        Updates alpha only if learnable_alpha=True.

        Args:
            act_dict (dict): Dictionary containing action information.
        Returns:
            None
        """
        if self.learnable_alpha:
            super().update_alpha(act_dict)

    def loss(
        self, state: torch.Tensor, critics: CriticEnsemble
    ) -> tuple[torch.Tensor, dict]:
        """
        Computes the actor loss for DSAC.

        Args:
            state (Tensor): Batch of states.
            critics (CriticEnsemble): Critic networks for Q-value estimation.
        Returns:
            tuple: Actor loss and action dictionary containing action and log probability.
        """
        batch_size = state.shape[0]
        act_dict = self.act(state)
        action, log_prob = act_dict["action"], act_dict["action_logprob"]

        tau, tau_hat, presum_tau = critics.get_tau(batch_size=batch_size)
        z_values = critics.Q(state, action, tau_hat)
        q_values = torch.sum(z_values * presum_tau, dim=-1, keepdim=True)
        q = torch.min(q_values, dim=0).values
        loss = (-q + self.log_alpha.exp() * log_prob).mean()
        return loss, act_dict


class DSACCritic(SACCritic):
    """
    Distributional Soft Actor-Critic (DSAC) Critic.

    Args:
        config (MainConfig): Configuration object containing model specifications.
        dim_state (int): Dimension of the state space.
        dim_act (int): Dimension of the action space.
    Attributes:
        num_quantiles (int): Number of quantile atoms.
        tau_type (str): Type of tau generation ('fix' or 'iqn').
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        assert (
            config.model.critic.n_quantiles > 1
        ), "Number of quantiles must be greater than one"
        super().__init__(config, dim_state, dim_act)

        self.num_quantiles = config.model.critic.n_quantiles
        self.tau_type = config.model.critic.tau_type

    def get_tau(self, batch_size):
        """
        Generates tau values based on the specified tau_type.

        Args:
            batch_size (int): The batch size for which tau values are generated.
        Returns:
            tuple: A tuple containing tau, tau_hat, and presum_tau tensors.
        """
        if self.tau_type == "fix":
            presum_tau = (
                torch.zeros(batch_size, self.num_quantiles, device=self.device)
                + 1.0 / self.num_quantiles
            )
        elif self.tau_type == "iqn":  # add 0.1 to prevent tau getting too close
            presum_tau = (
                torch.rand(batch_size, self.num_quantiles, device=self.device) + 0.1
            )
            presum_tau /= presum_tau.sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError(f"tau_type {self.tau_type} not implemented")
        tau = torch.cumsum(
            presum_tau, dim=1
        )  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.0
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.0
        return tau, tau_hat, presum_tau

    def Q(
        self, state: torch.Tensor, action: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Q-values for given state, action, and tau values.

        Args:
            state (torch.Tensor): The state tensor.
            action (torch.Tensor): The action tensor.
            tau (torch.Tensor): The tau tensor representing quantile fractions.
        Returns:
            torch.Tensor: The computed Q-values.
        """
        sa = torch.cat((state, action), -1)  # [n_batch x dim_state + dim_act]
        tau = tau  # [n_batch x n_quantiles]
        # Vectorize over the sample dimension
        output = torch.vmap(
            lambda a, b: self.model_ensemble((a, b)), in_dims=(None, 1), out_dims=2
        )(
            sa, tau.unsqueeze(1)
        )  # [n_member x n_batch x 1 x n_quantiles ]
        return output.squeeze(2)  # [n_member x n_batch x n_quantiles]

    def Q_t(
        self, state: torch.Tensor, action: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the target Q-values for given state, action, and tau values.

        Args:
            state (torch.Tensor): The state tensor.
            action (torch.Tensor): The action tensor.
            tau (torch.Tensor): The tau tensor representing quantile fractions.
        Returns:
            torch.Tensor: The computed target Q-values.
        """
        sa = torch.cat((state, action), -1)  # [n_batch x dim_state + dim_act]
        # Vectorize over the sample dimension
        output = torch.vmap(
            lambda a, b: self.target_ensemble((a, b)), in_dims=(None, 1), out_dims=2
        )(
            sa, tau.unsqueeze(1)
        )  # [n_member x n_batch x 1 x n_quantiles ]
        return output.squeeze(2)  # [n_member x n_batch x n_quantiles]

    @torch.no_grad()
    def get_bellman_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        actor: DSACActor,
    ) -> torch.Tensor:
        """
        Computes the Bellman target for the given reward, next state, done flag, and actor.

        Args:
            reward (torch.Tensor): The reward tensor.
            next_state (torch.Tensor): The next state tensor.
            done (torch.Tensor): The done flag tensor.
            actor (DSACActor): The actor instance.
        Returns:
            Tensor: Bellman target values.
        """
        batch_size = reward.shape[0]
        alpha = actor.log_alpha.exp().detach()
        # Get actions from target actor
        act_dict = actor.act_target(next_state)

        next_action = act_dict["action"]
        action_logprob = act_dict["action_logprob"]

        next_tau, next_tau_hat, next_presum_tau = self.get_tau(batch_size=batch_size)
        target_z_values = self.Q_t(next_state, next_action, next_tau_hat)

        min_clip_z_values = torch.min(target_z_values, dim=0).values
        z_next_values = min_clip_z_values - alpha * action_logprob

        z_target = (
            reward.unsqueeze(-1)
            + self._gamma * (1 - done.unsqueeze(-1)) * z_next_values
        )

        return z_target, next_presum_tau

    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        y: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Updates the critic network using the given state, action, and target values.

        Args:
            state (torch.Tensor): The state tensor.
            action (torch.Tensor): The action tensor.
            y (tuple[torch.Tensor, torch.Tensor]): The target values and target tau.
        Returns:
            None
        """
        self.optim.zero_grad()
        batch_size = state.shape[0]
        tau, tau_hat, presum_tau = self.get_tau(batch_size=batch_size)
        pred_quantiles = self.Q(
            state, action, tau_hat
        )  # [n_ensemble x n_batch x n_quantiles]

        y, target_tau = y  # Unpack target values and target tau
        loss = self.loss(
            pred_quantiles, self.model_ensemble.expand(y), tau_hat, target_tau
        )  # [n_ensemble x n_batch x n_quantiles x n_quantiles]

        loss = (
            loss.sum(-1).mean(axis=(1, 2)).sum(0)
            if self.n_members > 1
            else loss.sum(-1).mean()
        )
        loss.backward()

        self.optim.step()
        self.iter += 1


class DistributionalSoftActorCritic(ActorCritic):
    """
    Distributional Soft Actor-Critic agent combining DSACActor and DSACCritic.
    Ma et al. (2025): DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning
    """

    _agent_name = "DSAC"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = DSACCritic,
        actor_type: type = DSACActor,
    ) -> None:
        """
        Initializes DSAC agent.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type): Critic class type.
            actor_type (type): Actor class type.
        Returns:
            None
        """
        # Add postfix to name based on tau_type
        post_name_tag = "_Q" if config.model.critic.tau_type == "fix" else "_IQ"
        config.model.name += post_name_tag
        super().__init__(config, critic_type, actor_type)
