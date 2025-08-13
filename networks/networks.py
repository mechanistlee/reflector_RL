"""
Neural network architectures for SAC
"""

import torch as th
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from config import NETWORK_CONFIG


class MLPExtractor(BaseFeaturesExtractor):
    """
    Multi-Layer Perceptron feature extractor for SAC
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        hidden_sizes: List[int] = None,
        activation_fn: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        super().__init__(observation_space, features_dim)
        
        if hidden_sizes is None:
            hidden_sizes = NETWORK_CONFIG["hidden_sizes"]
        
        input_dim = int(np.prod(observation_space.shape))
        
        # Create MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_dim, features_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)


class CustomActor(BasePolicy):
    """
    Custom Actor network for SAC with enhanced stability
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int = 256,
        hidden_sizes: List[int] = None,
        activation_fn: nn.Module = nn.ReLU,
        log_std_init: float = -3,
        log_std_min: float = -20,
        log_std_max: float = 2,
        epsilon: float = 1e-6,
        use_sde: bool = False,
        **kwargs
    ):
        super().__init__(observation_space, action_space, features_extractor, **kwargs)
        
        if hidden_sizes is None:
            hidden_sizes = NETWORK_CONFIG["hidden_sizes"]
        
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.log_std_init = log_std_init
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        self.use_sde = use_sde
        
        action_dim = int(np.prod(action_space.shape))
        
        # Policy network
        self.latent_pi = create_mlp(
            features_dim, -1, hidden_sizes, activation_fn
        )
        
        # Mean and log_std layers
        last_layer_dim = hidden_sizes[-1] if hidden_sizes else features_dim
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Linear(last_layer_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Action distribution
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
        
        # Initialize log_std
        nn.init.constant_(self.log_std.weight, 0.0)
        nn.init.constant_(self.log_std.bias, self.log_std_init)
    
    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Get action distribution parameters"""
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        
        mean = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        
        # Clip log_std for stability
        log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Forward pass through the network"""
        mean, log_std = self.get_action_dist_params(obs)
        
        # Update action distribution
        self.action_dist.proba_distribution(mean, log_std)
        
        # Sample action
        if deterministic:
            action = self.action_dist.mode()
        else:
            action = self.action_dist.sample()
        
        return action
    
    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Get action and log probability"""
        mean, log_std = self.get_action_dist_params(obs)
        
        # Update action distribution
        self.action_dist.proba_distribution(mean, log_std)
        
        # Sample action and get log prob
        action = self.action_dist.sample()
        log_prob = self.action_dist.log_prob(action)
        
        return action, log_prob
    
    def log_prob_from_params(self, mean: th.Tensor, log_std: th.Tensor) -> th.Tensor:
        """Get log probability from distribution parameters"""
        self.action_dist.proba_distribution(mean, log_std)
        action = self.action_dist.sample()
        return self.action_dist.log_prob(action)


class CustomCritic(nn.Module):
    """
    Custom Critic network for SAC with enhanced stability
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int = 256,
        hidden_sizes: List[int] = None,
        activation_fn: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False,
        **kwargs
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = NETWORK_CONFIG["hidden_sizes"]
        
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        
        action_dim = int(np.prod(action_space.shape))
        
        # Q-network
        self.q_net = create_mlp(
            features_dim + action_dim, 
            1, 
            hidden_sizes, 
            activation_fn
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        """Forward pass through the critic network"""
        features = self.features_extractor(obs)
        
        # Concatenate observation features and action
        q_input = th.cat([features, action], dim=1)
        
        return self.q_net(q_input)


class DoubleCritic(nn.Module):
    """
    Double Q-network for SAC
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        features_extractor_class: type = MLPExtractor,
        features_dim: int = 256,
        hidden_sizes: List[int] = None,
        activation_fn: nn.Module = nn.ReLU,
        **kwargs
    ):
        super().__init__()
        
        # Create separate feature extractors for each Q-network
        self.features_extractor_1 = features_extractor_class(
            observation_space, features_dim, hidden_sizes, activation_fn, **kwargs
        )
        self.features_extractor_2 = features_extractor_class(
            observation_space, features_dim, hidden_sizes, activation_fn, **kwargs
        )
        
        # Create two Q-networks
        self.q_net_1 = CustomCritic(
            observation_space, action_space, self.features_extractor_1, 
            features_dim, hidden_sizes, activation_fn, **kwargs
        )
        self.q_net_2 = CustomCritic(
            observation_space, action_space, self.features_extractor_2, 
            features_dim, hidden_sizes, activation_fn, **kwargs
        )
    
    def forward(self, obs: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass through both Q-networks"""
        q1 = self.q_net_1(obs, action)
        q2 = self.q_net_2(obs, action)
        return q1, q2
    
    def q1_forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        """Forward pass through first Q-network only"""
        return self.q_net_1(obs, action)


def create_networks(
    observation_space: spaces.Box,
    action_space: spaces.Box,
    config: Dict[str, Any] = None
) -> Tuple[CustomActor, DoubleCritic]:
    """
    Create actor and critic networks
    
    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        config: Network configuration
    
    Returns:
        Tuple of (actor, critic) networks
    """
    if config is None:
        config = NETWORK_CONFIG
    
    # Get activation function
    activation_name = config.get("activation", "relu")
    activation_fn = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU
    }.get(activation_name, nn.ReLU)
    
    features_dim = config.get("features_dim", 256)
    hidden_sizes = config.get("hidden_sizes", [256, 256])
    
    # Create feature extractor for actor
    actor_features_extractor = MLPExtractor(
        observation_space, features_dim, hidden_sizes, activation_fn
    )
    
    # Create actor
    actor = CustomActor(
        observation_space=observation_space,
        action_space=action_space,
        features_extractor=actor_features_extractor,
        features_dim=features_dim,
        hidden_sizes=hidden_sizes,
        activation_fn=activation_fn,
        log_std_init=config.get("log_std_init", -3),
        log_std_min=config.get("log_std_min", -20),
        log_std_max=config.get("log_std_max", 2)
    )
    
    # Create critic
    critic = DoubleCritic(
        observation_space=observation_space,
        action_space=action_space,
        features_extractor_class=MLPExtractor,
        features_dim=features_dim,
        hidden_sizes=hidden_sizes,
        activation_fn=activation_fn
    )
    
    return actor, critic


def get_network_info(network: nn.Module) -> Dict[str, Any]:
    """Get information about a network"""
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "network_structure": str(network),
        "device": next(network.parameters()).device
    }


def init_weights(module: nn.Module, gain: float = 1.0):
    """Initialize network weights"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
