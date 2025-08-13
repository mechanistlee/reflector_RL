"""
Utility functions for SAC training and evaluation
"""

import os
import json
import pickle
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from config import PATHS_CONFIG, LOGGING_CONFIG, TEST_CONFIG, VIS_CONFIG


def setup_logging(log_path: str, verbose: int = 1) -> Logger:
    """Setup logging configuration"""
    os.makedirs(log_path, exist_ok=True)
    
    # Create logger
    logger = Logger(
        folder=log_path,
        output_formats=['csv', 'tensorboard', 'log']
    )
    
    return logger


def create_environment(
    env_name: str,
    render_mode: str = "rgb_array",
    seed: Optional[int] = None,
    monitor_wrapper: bool = True,
    log_dir: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    apply_time_limit: bool = True,
    is_test: bool = False
) -> gym.Env:
    """Create and configure environment"""
    from gymnasium.wrappers import TimeLimit
    from config import ENV_CONFIG, TEST_CONFIG
    
    # Create environment
    env = gym.make(env_name, render_mode=render_mode)
    
    # Apply time limit wrapper if requested
    if apply_time_limit:
        if max_episode_steps is None:
            if is_test:
                max_episode_steps = TEST_CONFIG.get("max_episode_steps", 200)
            else:
                max_episode_steps = ENV_CONFIG.get("max_episode_steps", 200)
        
        # TimeLimit wrapper will handle both env termination and step limit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    # Set seed
    if seed is not None:
        env.reset(seed=seed)
    
    # Add monitor wrapper for logging
    if monitor_wrapper:
        if log_dir is None:
            log_dir = PATHS_CONFIG["logs_dir"]
        os.makedirs(log_dir, exist_ok=True)
        
        monitor_path = os.path.join(log_dir, f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        env = Monitor(env, monitor_path)
    
    return env


def create_vec_env(
    env_name: str,
    n_envs: int = 1,
    render_mode: str = "rgb_array",
    seed: Optional[int] = None,
    monitor_wrapper: bool = True,
    log_dir: Optional[str] = None
) -> VecEnv:
    """Create vectorized environment"""
    
    def make_env(rank: int):
        def _init():
            env = create_environment(
                env_name=env_name,
                render_mode=render_mode,
                seed=seed + rank if seed is not None else None,
                monitor_wrapper=monitor_wrapper,
                log_dir=log_dir,
                apply_time_limit=True  # 벡터화 환경에서도 시간 제한 적용
            )
            return env
        return _init
    
    env_fns = [make_env(i) for i in range(n_envs)]
    return DummyVecEnv(env_fns)


def save_model_and_stats(
    model,
    model_path: str,
    training_stats: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """Save model and training statistics"""
    
    # Create directories
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    model.save(model_path)
    
    # Save training stats
    stats_path = model_path.replace('.zip', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2, default=str)
    
    # Save metadata
    if metadata:
        metadata_path = model_path.replace('.zip', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    print(f"Model saved to: {model_path}")
    print(f"Stats saved to: {stats_path}")


def load_model_and_stats(model_path: str) -> Tuple[Any, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load model and training statistics"""
    
    # Load model
    from networks.sac import SAC
    model = SAC.load(model_path)
    
    # Load training stats
    stats_path = model_path.replace('.zip', '_stats.json')
    training_stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            training_stats = json.load(f)
    
    # Load metadata
    metadata_path = model_path.replace('.zip', '_metadata.json')
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, training_stats, metadata


def evaluate_model(
    model,
    env: gym.Env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    max_episode_length: int = 1000
) -> Dict[str, Any]:
    """Evaluate model performance"""
    
    # Setup video recording if requested
    if record_video:
        if video_dir is None:
            video_dir = PATHS_CONFIG["videos_dir"]
        os.makedirs(video_dir, exist_ok=True)
        
        video_path = os.path.join(video_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        env = RecordVideo(env, video_path, episode_trigger=lambda x: True)
    
    # Evaluate policy
    episode_rewards, episode_lengths = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, 
        deterministic=deterministic, return_episode_rewards=True
    )
    
    # Calculate statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "min_episode_length": np.min(episode_lengths),
        "max_episode_length": np.max(episode_lengths),
        "success_rate": np.mean(np.array(episode_rewards) > 200),  # LunarLander success threshold
        "episode_rewards": episode_rewards.tolist(),
        "episode_lengths": episode_lengths.tolist(),
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic
    }
    
    return stats


def plot_training_curves(
    training_stats: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = False
):
    """Plot training curves"""
    
    print(f"Debug: plot_training_curves called with show_plot={show_plot}")
    
    if not training_stats:
        print("No training statistics available for plotting")
        return
    
    # Extract data
    timesteps = training_stats.get("timesteps", [])
    rewards = training_stats.get("episode_rewards", [])
    episode_lengths = training_stats.get("episode_lengths", [])
    losses = training_stats.get("losses", {})
    
    print(f"Debug: rewards length = {len(rewards)}, episode_lengths length = {len(episode_lengths)}")
    
    if not rewards and not episode_lengths:
        print("No episode data available for plotting")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Progress", fontsize=16)
    
    # Plot episode rewards
    if rewards:
        axes[0, 0].plot(rewards, alpha=0.7)
        if len(rewards) > 10:
            # Moving average
            window = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    if episode_lengths:
        axes[0, 1].plot(episode_lengths, alpha=0.7)
        if len(episode_lengths) > 10:
            window = min(50, len(episode_lengths) // 10)
            moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(episode_lengths)), moving_avg, 'r-', linewidth=2)
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot losses
    if losses:
        if "actor_loss" in losses:
            axes[1, 0].plot(losses["actor_loss"], label="Actor Loss", alpha=0.7)
        if "critic_loss" in losses:
            axes[1, 0].plot(losses["critic_loss"], label="Critic Loss", alpha=0.7)
        axes[1, 0].set_title("Training Losses")
        axes[1, 0].set_xlabel("Update Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot entropy coefficient
    if "entropy_coef" in training_stats:
        entropy_coef = training_stats["entropy_coef"]
        if entropy_coef:
            axes[1, 1].plot(entropy_coef, alpha=0.7)
            axes[1, 1].set_title("Entropy Coefficient")
            axes[1, 1].set_xlabel("Update Step")
            axes[1, 1].set_ylabel("Entropy Coef")
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    # Show plot
    if show_plot:
        print("Debug: Attempting to show plot...")
        plt.show()
        print("Debug: plt.show() completed")
    else:
        print("Debug: show_plot is False, not showing plot")
        plt.close()
    
    print("Debug: plot_training_curves function completed")


def plot_evaluation_results(
    eval_stats: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = False
):
    """Plot evaluation results"""
    
    episode_rewards = eval_stats.get("episode_rewards", [])
    episode_lengths = eval_stats.get("episode_lengths", [])
    
    if not episode_rewards:
        print("No evaluation data available for plotting")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Evaluation Results", fontsize=16)
    
    # Plot reward distribution
    axes[0].hist(episode_rewards, bins=10, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0].set_title("Reward Distribution")
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    if episode_lengths:
        axes[1].hist(episode_lengths, bins=10, alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(episode_lengths), color='red', linestyle='--',
                       label=f'Mean: {np.mean(episode_lengths):.2f}')
        axes[1].set_title("Episode Length Distribution")
        axes[1].set_xlabel("Episode Length")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"], bbox_inches='tight')
        print(f"Evaluation results saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def check_nan_inf(tensor: th.Tensor, name: str = "tensor") -> bool:
    """Check for NaN or Inf values in tensor"""
    if th.isnan(tensor).any():
        print(f"Warning: NaN detected in {name}")
        return True
    if th.isinf(tensor).any():
        print(f"Warning: Inf detected in {name}")
        return True
    return False


def clip_gradients(model, max_grad_norm: float = 1.0):
    """Clip gradients for stability"""
    if hasattr(model, 'policy'):
        th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_grad_norm)
    if hasattr(model, 'critic'):
        th.nn.utils.clip_grad_norm_(model.critic.parameters(), max_grad_norm)


def safe_mean(arr: List[float]) -> float:
    """Calculate mean safely handling None values"""
    if not arr:
        return 0.0
    
    # Filter out None values
    valid_values = [x for x in arr if x is not None and not np.isnan(x)]
    
    if not valid_values:
        return 0.0
    
    return np.mean(valid_values)


def safe_std(arr: List[float]) -> float:
    """Calculate standard deviation safely handling None values"""
    if not arr:
        return 0.0
    
    # Filter out None values
    valid_values = [x for x in arr if x is not None and not np.isnan(x)]
    
    if len(valid_values) < 2:
        return 0.0
    
    return np.std(valid_values)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_device() -> th.device:
    """Get optimal device for training"""
    if th.cuda.is_available():
        return th.device("cuda")
    elif th.backends.mps.is_available():
        return th.device("mps")
    else:
        return th.device("cpu")


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)


def print_system_info():
    """Print system information"""
    print("System Information:")
    print(f"  PyTorch version: {th.__version__}")
    print(f"  CUDA available: {th.cuda.is_available()}")
    if th.cuda.is_available():
        print(f"  CUDA version: {th.version.cuda}")
        print(f"  GPU: {th.cuda.get_device_name(0)}")
    print(f"  Device: {get_device()}")
    print()


def create_summary_report(
    model_path: str,
    training_stats: Dict[str, Any],
    eval_stats: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Create a summary report of training and evaluation"""
    
    report = []
    report.append("="*60)
    report.append("SAC TRAINING SUMMARY REPORT")
    report.append("="*60)
    
    # Model information
    report.append(f"Model: {os.path.basename(model_path)}")
    report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Training statistics
    if training_stats:
        report.append("TRAINING STATISTICS:")
        report.append(f"  Total Episodes: {len(training_stats.get('episode_rewards', []))}")
        report.append(f"  Total Timesteps: {training_stats.get('total_timesteps', 'N/A')}")
        
        rewards = training_stats.get('episode_rewards', [])
        if rewards:
            report.append(f"  Mean Training Reward: {safe_mean(rewards):.2f}")
            report.append(f"  Best Training Reward: {max(rewards):.2f}")
        
        report.append("")
    
    # Evaluation statistics
    if eval_stats:
        report.append("EVALUATION STATISTICS:")
        report.append(f"  Evaluation Episodes: {eval_stats.get('n_eval_episodes', 'N/A')}")
        report.append(f"  Mean Reward: {eval_stats.get('mean_reward', 0):.2f} ± {eval_stats.get('std_reward', 0):.2f}")
        report.append(f"  Best Reward: {eval_stats.get('max_reward', 0):.2f}")
        report.append(f"  Success Rate: {eval_stats.get('success_rate', 0)*100:.1f}%")
        report.append(f"  Mean Episode Length: {eval_stats.get('mean_episode_length', 0):.1f}")
        report.append("")
    
    # Metadata
    if metadata:
        report.append("CONFIGURATION:")
        for key, value in metadata.items():
            report.append(f"  {key}: {value}")
        report.append("")
    
    report.append("="*60)
    
    return "\n".join(report)


def save_summary_report(
    model_path: str,
    training_stats: Dict[str, Any],
    eval_stats: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """Save summary report to file"""
    
    report = create_summary_report(model_path, training_stats, eval_stats, metadata)
    
    # Save report
    report_path = model_path.replace('.zip', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to: {report_path}")
    print("\n" + report)
# =============================================================================
# SPEOS Environment Creation Functions
# =============================================================================

def create_speos_environment(
    config_dict: Optional[Dict[str, Any]] = None,
    sample_data: Optional[Dict[str, Any]] = None,
    monitor_wrapper: bool = True,
    log_dir: Optional[str] = None,
    **kwargs
) -> gym.Env:
    """Create SPEOS environment using physics simulator"""
    from env.env_speos_v1 import SpeosEnv, SpeosConfig, SimulationType, PhysicsSimulatorFactory
    
    # Create SPEOS configuration
    config = SpeosConfig()
    
    # Update config with provided parameters
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Update with keyword arguments
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create environment
    env = PhysicsSimulatorFactory.create_environment(
        SimulationType.SPEOS, 
        config,
        sample_data
    )
    
    # Apply monitor wrapper if requested
    if monitor_wrapper and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir, allow_early_resets=True)
    
    return env


def create_speos_vectorized_environment(
    config_dict: Optional[Dict[str, Any]] = None,
    sample_data_list: Optional[List[Dict[str, Any]]] = None,
    n_envs: int = 1,
    monitor_wrapper: bool = True,
    log_dir: Optional[str] = None
) -> VecEnv:
    """Create vectorized SPEOS environments"""
    from env.env_speos_v1 import SpeosConfig, SimulationType, PhysicsSimulatorFactory
    
    # Create SPEOS configuration
    config = SpeosConfig()
    
    # Update config with provided parameters
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create vectorized environments
    return PhysicsSimulatorFactory.create_vectorized_environments(
        SimulationType.SPEOS,
        config,
        sample_data_list or [None] * n_envs,
        n_envs
    )


def validate_speos_environment(env: gym.Env) -> bool:
    """Validate SPEOS environment setup"""
    try:
        # Check if environment is properly initialized
        obs, info = env.reset()
        
        # Check observation space
        if not env.observation_space.contains(obs):
            print(f"Warning: Initial observation {obs} not in observation space")
            return False
        
        # Check action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check reward
        if not isinstance(reward, (int, float)):
            print(f"Warning: Reward {reward} is not a number")
            return False
        
        # Check info dictionary
        if not isinstance(info, dict):
            print(f"Warning: Info {info} is not a dictionary")
            return False
        
        print("✅ SPEOS environment validation passed")
        return True
        
    except Exception as e:
        print(f"❌ SPEOS environment validation failed: {e}")
        return False


def speos_environment_info(env: gym.Env) -> Dict[str, Any]:
    """Get information about SPEOS environment"""
    from env.env_speos_v1 import SpeosEnv
    
    info = {
        "environment_type": "SPEOS Optical Simulation",
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
    }
    
    # Add SPEOS-specific information if available
    if hasattr(env, 'config'):
        config = env.config
        info.update({
            "grid_size": f"{config.grid_rows}x{config.grid_cols}",
            "ray_count": config.ray_count,
            "wavelength_range": f"{config.wavelength_range[0]}-{config.wavelength_range[1]}nm",
            "reflection_model": config.reflection_model,
            "max_steps": config.max_steps
        })
    print(f"Environment info: {info}")
    return info
