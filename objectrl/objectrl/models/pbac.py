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

import torch

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.models.sac import SACActor

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class PBACActor(SACActor):
    """
    Actor class for PBAC with posterior sampling-based ensemble head selection.

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): Observation space dimensions.
        dim_act (int): Action space dimensions.

    Samples a head from an ensemble of actor policies every N steps or at episode boundaries
    to simulate posterior sampling. At evaluation time, it averages actions.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)
        self.interaction_iter = 0
        self.sampling_rate = config.model.posterior_sampling_rate
        self.idx_active_critic = 0
        self.is_episode_end = False

    def act(self, state: torch.Tensor, is_training: bool = True) -> dict:
        """
        Selects an action, potentially sampling from different actor heads during training.

        Args:
            state (Tensor): Current observation.
            is_training (bool): Whether in training mode.
        Returns:
            dict: Action dictionary with 'action' and 'action_logprob'.
        """
        action_dict = super().act(state)
        action = action_dict["action"]
        e = action_dict["action_logprob"]
        if is_training:
            if self.is_episode_end or (self.interaction_iter % self.sampling_rate == 0):
                self.idx_active_critic = torch.randint(0, action.size(1), (1,)).item()
                action = action[:, self.idx_active_critic, :]
                e = e[:, self.idx_active_critic]
            if action.shape[0] == 1:
                action = action.squeeze()
                e = e.squeeze()
        else:
            action = action.mean(dim=1).squeeze()

        action_dict["action"], action_dict["action_logprob"] = action, (
            e if e is not None else torch.zeros_like(action)
        )
        return action_dict

    def set_episode_status(self, is_end: bool) -> None:
        """
        Sets whether the current episode has ended.

        Args:
            is_end (bool): Episode termination flag.
        Returns:
            None
        """
        self.is_episode_end = is_end


class PBACCritic(CriticEnsemble):
    """
    PBAC critic ensemble using PAC-Bayesian loss.

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): State space dimensions.
        dim_act (int): Action space dimensions.

    Implements target computation and weight updates using the PAC-Bayesian loss.
    """

    def __init__(
        self,
        config: "MainConfig",
        dim_state: int,
        dim_act: int,
    ) -> None:
        super().__init__(config, dim_state, dim_act)

    @torch.no_grad()
    def get_bellman_target(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        actor: PBACActor,
    ) -> torch.Tensor:
        """
        Computes target Q-values using entropy-regularized Bellman backup.

        Args:
            reward (Tensor): Rewards.
            next_state (Tensor): Next states.
            done (Tensor): Done flags.
            actor (PBACActor): Actor network for next state action selection.
        Returns:
            Tensor: Bellman targets.
        """
        alpha = actor.log_alpha.exp().detach() if hasattr(actor, "log_alpha") else 0
        action_dict = actor.act(next_state)
        next_action, ep = action_dict["action"], action_dict["action_logprob"]
        qp_ = self.Q_t(next_state, next_action)
        qp_t = qp_ - alpha * (ep if ep is not None else 0)
        y = reward.unsqueeze(-1) + (self._gamma * qp_t * (1 - done.unsqueeze(-1)))
        return y

    @torch.no_grad()
    def Q_t(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Computes target Q-values for state-action pairs.

        Args:
            s (Tensor): States.
            a (Tensor): Actions.
        Returns:
            Tensor: Target Q-values from the critic ensemble.
        """
        if len(a.shape) == 1:
            a = a.view(-1, 1)
        SA = torch.cat((s, a), -1)
        return self.target_ensemble(SA)

    def update(self, s: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> None:
        """
        Performs a critic update step.
        Args:
            s (Tensor): States.
            a (Tensor): Actions.
            y (Tensor): Target Q-values.
        """
        self.optim.zero_grad()
        self.loss(self.Q(s, a), y).backward()
        self.optim.step()
        self.iter += 1


class PACBayesianAC(ActorCritic):
    """
    PBAC agent class implementing PAC-Bayesian Actor-Critic logic.
    Combines the PBACActor and PBACCritic, manages training and interaction.
    Tasdighi et al. (2025): Deep Exploration with PAC-Bayes
    """

    _agent_name = "PBAC"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = PBACCritic,
        actor_type: type = PBACActor,
    ) -> None:
        """
        Initializes the PBAC agent.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type): Critic class type.
            actor_type (type): Actor class type.
        Returns:
            None
        """
        super().__init__(config, critic_type, actor_type)

    def store_transition(self, transition: dict) -> None:
        """
        Stores a transition and updates actor's episode status.

        Args:
            transition (dict): Transition containing state, action, reward, etc.
        Returns:
            None
        """
        super().store_transition(transition)
        self.actor.set_episode_status(
            transition["terminated"] or transition["truncated"]
        )
