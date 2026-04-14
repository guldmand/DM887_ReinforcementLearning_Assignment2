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


from dataclasses import dataclass, field
from typing import Literal

from objectrl.models.dsac import DSACActor, DSACCritic
from objectrl.nets.actor_nets import ActorNetProbabilistic
from objectrl.nets.critic_nets import QuantileCriticNet


# [start-config]
@dataclass
class CriticLossConfig:
    """
    Configuration for the DSAC critic loss.

    Attributes:
        kappa (float): Huber loss threshold.
    """

    kappa: float = 1.0


@dataclass
class DSACActorConfig:
    """
    Configuration for the DSAC Actor network.

    Attributes:
        arch (type): Architecture class for the actor network.
        actor_type (type): Actor class type.
        has_target (bool): Whether to use a target network.
    """

    arch: type = ActorNetProbabilistic
    actor_type: type = DSACActor
    has_target: bool = True


@dataclass
class DSACCriticConfig:
    """
    Configuration for the DSAC Critic network.

    Attributes:
        arch (type): Architecture class for the critic network.
        critic_type (type): Critic class type.
        n_quantiles (int): Number of atoms for quantile regression.
        has_target (bool): Whether to use a target network.
        n_members (int): Number of critic members.
        tau_type (Literal["fix", "iqn"]): Type of quantile regression.
    """

    arch: type = QuantileCriticNet
    critic_type: type = DSACCritic
    norm: bool = True
    n_quantiles: int = 8
    has_target: bool = True
    n_members: int = 2
    tau_type: Literal["fix", "iqn"] = "iqn"

    def __post_init__(self):
        self.dim_out = self.n_quantiles


@dataclass
class DSACConfig:
    """
    Main DSAC algorithm configuration.

    Attributes:
        name (str): Name of the algorithm.
        loss (str): Loss function used.
        policy_delay (int): Delay for policy updates.
        tau (float): Soft update coefficient.
        target_entropy (float | None): Target entropy for the policy.
        learnable_alpha (bool): Whether the temperature parameter alpha is learnable.
        alpha (float): Initial value of the temperature parameter alpha.
        actor (DSACActorConfig): Configuration for the actor network.
        critic (DSACCriticConfig): Configuration for the critic network.
    """

    name: str = "dsac"
    lossparams: CriticLossConfig = field(default_factory=CriticLossConfig)
    loss: str = "DSACLoss"
    policy_delay: int = 1
    tau: float = 0.005
    target_entropy: float | None = None
    learnable_alpha: bool = True
    alpha: float = 1.0

    actor: DSACActorConfig = field(default_factory=DSACActorConfig)
    critic: DSACCriticConfig = field(default_factory=DSACCriticConfig)

    def __post_init__(self):
        if isinstance(self.lossparams, dict):
            self.lossparams = CriticLossConfig(**self.lossparams)


# [end-config]
