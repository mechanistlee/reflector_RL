"""
Advanced visualization module for RL training analysis
Unified 3x3 grid visualization for SPEOS optical simulation training results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import os
from datetime import datetime
import json

# Configure matplotlib for proper font rendering (English only)
import matplotlib
# Use Arial as primary font (better Windows compatibility)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Calibri', 'Tahoma', 'sans-serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False
# Completely disable all font warnings and fallback
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Glyph*missing from font*")
warnings.filterwarnings("ignore", message="findfont*")
warnings.filterwarnings("ignore", message=".*font.*")
# Force matplotlib to use only ASCII characters
matplotlib.rcParams['axes.formatter.use_mathtext'] = True
# Clear font cache to ensure fresh start
try:
    matplotlib.font_manager._rebuild()
except AttributeError:
    # Alternative method for newer matplotlib versions
    matplotlib.font_manager.fontManager.__init__()

# Function to filter out non-ASCII characters
def filter_ascii_only(text: str) -> str:
    """Remove non-ASCII characters from text to prevent font errors"""
    if not isinstance(text, str):
        return str(text)
    # Keep only ASCII characters and basic symbols
    filtered = ''.join(char for char in text if ord(char) < 128)
    return filtered if filtered else "Training Results"

from config import VISUALIZATION_CONFIG


class TrainingVisualizer:
    """
    SPEOS RL Training Results 3x3 Advanced Visualization System
    Generates unified report with 8 graphs + 1 training summary
    """
    
    def __init__(self):
        self.config = VISUALIZATION_CONFIG
        self.metrics = defaultdict(list)
        self.processed_stats = {}
        
        # matplotlib style settings
        try:
            plt.style.use('seaborn-v0_8')
        except Exception:
            try:
                plt.style.use('seaborn')
            except Exception:
                plt.style.use('default')
        
        # English font settings only - use Arial for better compatibility
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Calibri', 'Tahoma', 'sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
        # Additional font warning suppression
        plt.rcParams['axes.formatter.use_mathtext'] = True
        
    def process_training_stats(self, training_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Process and preprocess training statistics data"""
        
        processed = {}
        
        print(f"Processing training stats - Input keys: {list(training_stats.keys())}")
        
        # Basic metrics
        if 'episode_rewards' in training_stats:
            processed['episode_rewards'] = np.array(training_stats['episode_rewards'])
            processed['episode_lengths'] = np.array(training_stats.get('episode_lengths', []))
            processed['success_rate'] = self._calculate_success_rate(training_stats['episode_rewards'])
            
        # Total training timesteps
        processed['total_timesteps'] = training_stats.get('total_timesteps', 200000)
        
        # Advanced metrics
        processed['q_values'] = training_stats.get('q_values', [])
        processed['actor_losses'] = training_stats.get('actor_losses', [])
        processed['critic_losses'] = training_stats.get('critic_losses', [])
        processed['entropy_values'] = training_stats.get('entropy_values', [])
        
        print(f"Advanced metrics collection complete:")
        print(f"  Total timesteps: {processed['total_timesteps']}")
        print(f"  Q-values: {len(processed['q_values'])} samples")
        print(f"  Actor losses: {len(processed['actor_losses'])} samples")
        print(f"  Critic losses: {len(processed['critic_losses'])} samples")
        print(f"  Entropy values: {len(processed['entropy_values'])} samples")
            
        self.processed_stats = processed
        return processed
    
    def _calculate_success_rate(self, rewards: List[float], window_size: int = 100) -> List[float]:
        """Calculate success rate (reward > 200)"""
        success_threshold = 200
        successes = [1 if r > success_threshold else 0 for r in rewards]
        
        success_rates = []
        for i in range(len(successes)):
            start_idx = max(0, i - window_size + 1)
            window_successes = successes[start_idx:i + 1]
            success_rates.append(np.mean(window_successes))
            
        return success_rates
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window_size:
            return data
        
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean().tolist()
    
    def create_unified_output(self, processed_stats: Dict[str, Any], output_png: str, 
                            output_json: str, title: str = "SPEOS RL Training Results") -> Dict[str, str]:
        """Create unified 3x3 visualization and JSON report"""
        
        created_files = {}
        
        print(f"Creating unified visualization: {output_png}")
        
        # Create 3x3 unified visualization
        self._create_advanced_3x3_visualization(output_png, title)
        created_files['visualization'] = output_png
        
        # Save JSON report
        self._save_json_report(output_json)
        created_files['report'] = output_json
        
        return created_files
    
    def _create_advanced_3x3_visualization(self, save_path: str, title: str) -> None:
        """Create 3x3 advanced visualization (8 graphs + 1 training summary)"""
        
        # Filter title to ASCII only
        title = filter_ascii_only(title)
        
        print("=" * 60)
        print("Creating 3x3 advanced visualization...")
        print(f"Save path: {save_path}")
        print(f"Title: {title}")
        print("=" * 60)
        
        # Reset matplotlib font settings (ensure English only)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Calibri', 'Tahoma', 'sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
        # Force English-only rendering
        plt.rcParams['axes.formatter.use_mathtext'] = True
        
        # Suppress all warnings during visualization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
        # Create figure and subplots (3x3 grid)
        fig = plt.figure(figsize=(20, 16))
        print("Figure created (20x16 size)")
        
        # Extract training data
        rewards = self.processed_stats.get('episode_rewards', [])
        episode_lengths = self.processed_stats.get('episode_lengths', [])
        success_rate = self.processed_stats.get('success_rate', [])
        q_values = self.processed_stats.get('q_values', [])
        actor_losses = self.processed_stats.get('actor_losses', [])
        critic_losses = self.processed_stats.get('critic_losses', [])
        entropy_values = self.processed_stats.get('entropy_values', [])
        total_timesteps = self.processed_stats.get('total_timesteps', 200000)
        
        print(f"Data check:")
        print(f"  - Rewards: {len(rewards)} items")
        print(f"  - Episode lengths: {len(episode_lengths)} items")
        print(f"  - Success rates: {len(success_rate)} items")
        print(f"  - Q-values: {len(q_values)} items")
        print(f"  - Actor losses: {len(actor_losses)} items")
        print(f"  - Critic losses: {len(critic_losses)} items")
        print(f"  - Entropy values: {len(entropy_values)} items")
        
        # Graph 1: Episode Rewards (top-left)
        print("Graph 1/9: Creating episode rewards...")
        ax1 = plt.subplot(3, 3, 1)
        if len(rewards) > 0:
            episodes = range(len(rewards))
            plt.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=0.8)
            if len(rewards) > 50:
                smoothed = self._moving_average(rewards, min(50, len(rewards)//10))
                plt.plot(episodes[:len(smoothed)], smoothed, color='red', linewidth=2, label='Moving Avg')
            plt.title('Episode Rewards', fontsize=12, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            if len(rewards) > 50:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No reward data available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Episode Rewards', fontsize=12, fontweight='bold')
        
        # Graph 2: Episode Steps (top-center)
        print("Graph 2/9: Creating episode steps...")
        ax2 = plt.subplot(3, 3, 2)
        if len(episode_lengths) > 0:
            episodes = range(len(episode_lengths))
            plt.plot(episodes, episode_lengths, alpha=0.5, color='green', linewidth=0.8)
            if len(episode_lengths) > 50:
                smoothed = self._moving_average(episode_lengths, min(50, len(episode_lengths)//10))
                plt.plot(episodes[:len(smoothed)], smoothed, color='darkgreen', linewidth=2, label='Moving Avg')
            plt.title('Episode Steps', fontsize=12, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True, alpha=0.3)
            if len(episode_lengths) > 50:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No episode length data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Episode Steps', fontsize=12, fontweight='bold')
        
        # Graph 3: Success Rate (top-right)
        print("Graph 3/9: Creating success rate...")
        ax3 = plt.subplot(3, 3, 3)
        if len(success_rate) > 0:
            episodes = range(len(success_rate))
            plt.plot(episodes, success_rate, color='orange', linewidth=2)
            plt.title('Success Rate', fontsize=12, fontweight='bold')
            plt.xlabel('Episode')
            plt.ylabel('Success Rate')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, 'No success rate data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Success Rate', fontsize=12, fontweight='bold')
        
        # Graph 4: Q-Values (middle-left)
        print("Graph 4/9: Creating Q-values...")
        ax4 = plt.subplot(3, 3, 4)
        if len(q_values) > 0:
            collection_interval = 100
            total_samples = len(q_values)
            max_steps = min(total_samples * collection_interval, total_timesteps)
            steps = np.linspace(0, max_steps, total_samples)
            
            plt.plot(steps, q_values, color='purple', alpha=0.7, linewidth=0.8)
            if len(q_values) > 50:
                smoothed = self._moving_average(q_values, min(50, len(q_values)//10))
                plt.plot(steps[:len(smoothed)], smoothed, color='darkviolet', linewidth=2, label='Moving Avg')
            plt.title('Q-Values', fontsize=12, fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Q-Value')
            plt.grid(True, alpha=0.3)
            plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            if len(q_values) > 50:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Q-value data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Q-Values', fontsize=12, fontweight='bold')
        
        # Graph 5: Actor Loss (middle-center)
        print("Graph 5/9: Creating actor loss...")
        ax5 = plt.subplot(3, 3, 5)
        if len(actor_losses) > 0:
            collection_interval = 100
            total_samples = len(actor_losses)
            max_steps = min(total_samples * collection_interval, total_timesteps)
            steps = np.linspace(0, max_steps, total_samples)
            
            plt.plot(steps, actor_losses, color='red', alpha=0.7, linewidth=0.8)
            if len(actor_losses) > 50:
                smoothed = self._moving_average(actor_losses, min(50, len(actor_losses)//10))
                plt.plot(steps[:len(smoothed)], smoothed, color='darkred', linewidth=2, label='Moving Avg')
            plt.title('Actor Loss', fontsize=12, fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            if len(actor_losses) > 50:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No actor loss data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Actor Loss', fontsize=12, fontweight='bold')
        
        # Graph 6: Critic Loss (middle-right)
        print("Graph 6/9: Creating critic loss...")
        ax6 = plt.subplot(3, 3, 6)
        if len(critic_losses) > 0:
            collection_interval = 100
            total_samples = len(critic_losses)
            max_steps = min(total_samples * collection_interval, total_timesteps)
            steps = np.linspace(0, max_steps, total_samples)
            
            plt.plot(steps, critic_losses, color='blue', alpha=0.7, linewidth=0.8)
            if len(critic_losses) > 50:
                smoothed = self._moving_average(critic_losses, min(50, len(critic_losses)//10))
                plt.plot(steps[:len(smoothed)], smoothed, color='darkblue', linewidth=2, label='Moving Avg')
            plt.title('Critic Loss', fontsize=12, fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            if len(critic_losses) > 50:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No critic loss data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Critic Loss', fontsize=12, fontweight='bold')
        
        # Graph 7: Entropy (bottom-left)
        print("Graph 7/9: Creating entropy...")
        ax7 = plt.subplot(3, 3, 7)
        if len(entropy_values) > 0:
            collection_interval = 100
            total_samples = len(entropy_values)
            max_steps = min(total_samples * collection_interval, total_timesteps)
            steps = np.linspace(0, max_steps, total_samples)
            
            plt.plot(steps, entropy_values, color='green', alpha=0.7, linewidth=0.8)
            if len(entropy_values) > 50:
                smoothed = self._moving_average(entropy_values, min(50, len(entropy_values)//10))
                plt.plot(steps[:len(smoothed)], smoothed, color='darkgreen', linewidth=2, label='Moving Avg')
            plt.title('Entropy', fontsize=12, fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Entropy')
            plt.grid(True, alpha=0.3)
            plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            if len(entropy_values) > 50:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'No entropy data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Entropy', fontsize=12, fontweight='bold')
        
        # Graph 8: Reward Distribution (bottom-center)
        print("Graph 8/9: Creating reward distribution...")
        ax8 = plt.subplot(3, 3, 8)
        if len(rewards) > 0:
            plt.hist(rewards, bins=50, alpha=0.7, color='cyan', edgecolor='black')
            plt.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(rewards):.2f}')
            plt.axvline(200, color='green', linestyle='--', linewidth=2, label='Success: 200')
            plt.title('Reward Distribution', fontsize=12, fontweight='bold')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No reward data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Reward Distribution', fontsize=12, fontweight='bold')
        
        # Graph 9: Training Summary (bottom-right)
        print("Graph 9/9: Creating training summary...")
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')  # Remove axes for text display
        
        # Calculate summary statistics
        summary_text = "Training Summary\n" + "="*35 + "\n\n"
        
        if len(rewards) > 0:
            total_episodes = len(rewards)
            total_steps = sum(episode_lengths) if len(episode_lengths) > 0 else 0
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)
            
            # Last 100 episodes statistics
            last_100_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
            last_100_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
            last_100_success = success_rate[-100:] if len(success_rate) >= 100 else success_rate
            
            last_100_reward_mean = np.mean(last_100_rewards)
            last_100_success_rate = np.mean(last_100_success) if len(last_100_success) > 0 else 0
            last_100_steps_mean = np.mean(last_100_lengths) if len(last_100_lengths) > 0 else 0
            
            summary_text += f"Total Steps: {total_steps:,}\n"
            summary_text += f"Total Episodes: {total_episodes:,}\n\n"
            summary_text += f"Reward Mean: {reward_mean:.2f}\n"
            summary_text += f"Reward Std: {reward_std:.2f}\n\n"
            summary_text += f"Last 100 Episodes:\n"
            summary_text += f"  • Reward Mean: {last_100_reward_mean:.2f}\n"
            summary_text += f"  • Success Rate: {last_100_success_rate:.2%}\n"
            summary_text += f"  • Mean Steps: {last_100_steps_mean:.1f}\n\n"
            
            # Training completion status
            if last_100_reward_mean > 200:
                summary_text += "Status: ✓ CONVERGED\n"
            else:
                summary_text += "Status: ○ Training\n"
        else:
            summary_text += "No training data available"
        
        # Add summary text to the plot
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.9))
        
        # Overall title
        print("Setting overall title...")
        # Ensure title is ASCII only
        title = filter_ascii_only(title)
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout and save
        print("Adjusting layout...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
            plt.subplots_adjust(top=0.94)
        
        print(f"Saving file: {save_path}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ File saved successfully: {save_path}")
        except Exception as save_error:
            print(f"✗ File save failed: {save_error}")
            raise
        finally:
            plt.close()
            print("Figure cleanup complete")
        
        print("=" * 60)
        print("3x3 advanced visualization creation complete!")
        print("=" * 60)
    
    def _save_json_report(self, save_path: str) -> None:
        """Save detailed training report in JSON format"""
        
        rewards = self.processed_stats.get('episode_rewards', [])
        episode_lengths = self.processed_stats.get('episode_lengths', [])
        success_rate = self.processed_stats.get('success_rate', [])
        q_values = self.processed_stats.get('q_values', [])
        actor_losses = self.processed_stats.get('actor_losses', [])
        critic_losses = self.processed_stats.get('critic_losses', [])
        entropy_values = self.processed_stats.get('entropy_values', [])
        total_timesteps = self.processed_stats.get('total_timesteps', 200000)
        
        # Report structure
        report = {
            "metadata": {
                "generated_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "analysis_type": "SPEOS Optical Simulation RL Training Analysis",
                "environment": "SPEOS Optical Simulation",
                "algorithm": "SAC (Soft Actor-Critic)"
            },
            "training_configuration": {
                "total_timesteps": total_timesteps,
                "episode_limit": len(rewards) if len(rewards) > 0 else 0,
                "data_collection_interval": 100
            },
            "training_summary": {
                "total_episodes": len(rewards) if len(rewards) > 0 else 0,
                "total_steps": int(sum(episode_lengths)) if len(episode_lengths) > 0 else 0,
                "training_completed": True if len(rewards) > 0 else False,
                "convergence_achieved": float(np.mean(rewards[-100:])) > 200 if len(rewards) >= 100 else False
            },
            "reward_statistics": {
                "overall": {
                    "mean": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
                    "std": float(np.std(rewards)) if len(rewards) > 0 else 0.0,
                    "min": float(np.min(rewards)) if len(rewards) > 0 else 0.0,
                    "max": float(np.max(rewards)) if len(rewards) > 0 else 0.0,
                    "median": float(np.median(rewards)) if len(rewards) > 0 else 0.0
                },
                "last_100_episodes": {
                    "reward_mean": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
                    "success_rate": float(np.mean(success_rate[-100:])) if len(success_rate) >= 100 else float(np.mean(success_rate)) if len(success_rate) > 0 else 0.0,
                    "steps_mean": float(np.mean(episode_lengths[-100:])) if len(episode_lengths) >= 100 else float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0.0
                }
            },
            "performance_metrics": {
                "final_performance": {
                    "final_episode_reward": float(rewards[-1]) if len(rewards) > 0 else 0.0,
                    "final_success_rate": float(success_rate[-1]) if len(success_rate) > 0 else 0.0,
                    "final_episode_length": int(episode_lengths[-1]) if len(episode_lengths) > 0 else 0
                },
                "learning_metrics": {
                    "final_q_value": float(q_values[-1]) if len(q_values) > 0 else 0.0,
                    "final_actor_loss": float(actor_losses[-1]) if len(actor_losses) > 0 else 0.0,
                    "final_critic_loss": float(critic_losses[-1]) if len(critic_losses) > 0 else 0.0,
                    "final_entropy": float(entropy_values[-1]) if len(entropy_values) > 0 else 0.0
                }
            },
            "data_availability": {
                "episode_rewards": len(rewards),
                "episode_lengths": len(episode_lengths), 
                "success_rates": len(success_rate),
                "q_values": len(q_values),
                "actor_losses": len(actor_losses),
                "critic_losses": len(critic_losses),
                "entropy_values": len(entropy_values)
            }
        }
        
        # Save JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"JSON report saved successfully: {save_path}")


# Factory function
def create_training_visualizer() -> TrainingVisualizer:
    """Create TrainingVisualizer instance"""
    return TrainingVisualizer()


# Main analysis function (unified)
def create_comprehensive_report(training_stats: Dict[str, Any], 
                              save_dir: str = "./plots", 
                              model_name: str = "model",
                              title: str = "SPEOS RL Training Results") -> Dict[str, str]:
    """
    Create comprehensive SPEOS RL training results report
    
    Args:
        training_stats: Training statistics dictionary
        save_dir: Save directory
        model_name: Model name (used in filename)
        title: Visualization title
        
    Returns:
        Dictionary containing created file paths
    """
    
    # Filter title to ASCII only
    title = filter_ascii_only(title)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate file paths
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(save_dir, f"{model_name}_{timestamp}_training_summary.png")
    json_path = os.path.join(save_dir, f"{model_name}_{timestamp}_training_stats.json")
    
    # Create visualizer and process data
    visualizer = create_training_visualizer()
    processed_stats = visualizer.process_training_stats(training_stats)
    
    # Generate unified output
    created_files = visualizer.create_unified_output(
        processed_stats=processed_stats,
        output_png=png_path,
        output_json=json_path,
        title=title
    )
    
    print(f"Comprehensive report creation complete:")
    for file_type, file_path in created_files.items():
        print(f"  {file_type}: {file_path}")
    
    return created_files


# Legacy compatibility functions (all create unified reports)
def advanced_analysis(training_stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Advanced analysis (legacy compatibility)"""
    if save_path:
        save_dir = os.path.dirname(save_path) or "./plots"
        model_name = os.path.splitext(os.path.basename(save_path))[0] or "model"
        create_comprehensive_report(training_stats, save_dir, model_name)


def quick_analysis(training_stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Quick analysis (legacy compatibility)"""
    advanced_analysis(training_stats, save_path)


def expert_analysis(training_stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Expert analysis (legacy compatibility)"""
    advanced_analysis(training_stats, save_path)


# More specific legacy compatibility functions
def plot_training_progress(training_stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Training progress plots (legacy compatibility)"""
    advanced_analysis(training_stats, save_path)


def plot_rewards(training_stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Reward graphs (legacy compatibility)"""
    advanced_analysis(training_stats, save_path)


def plot_losses(training_stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Loss graphs (legacy compatibility)"""
    advanced_analysis(training_stats, save_path)


def save_plots(training_stats: Dict[str, Any], save_path: str) -> None:
    """Save plots (legacy compatibility)"""
    advanced_analysis(training_stats, save_path)


def create_dashboard(training_stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Create dashboard (legacy compatibility)"""
    advanced_analysis(training_stats, save_path)


# Export all functions
__all__ = [
    'TrainingVisualizer',
    'create_training_visualizer',
    'create_comprehensive_report',
    'advanced_analysis',
    'quick_analysis', 
    'expert_analysis',
    'plot_training_progress',
    'plot_rewards',
    'plot_losses',
    'save_plots',
    'create_dashboard'
]
