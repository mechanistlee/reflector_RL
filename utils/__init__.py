"""
Utils Package# Reward Calculation Functions
from env.calculate_reward import (
    calculate_speos_reward,
    SpeosRewardCalculator
)

# 하위 호환성을 위한 별칭
RewardCalculator = SpeosRewardCalculator==========

유틸리티 함수들을 모아놓은 패키지입니다.
"""

# SPEOS Integration Utilities (Class-Based Structure)
from env.temporary_speos_utility import (
    SpeosUtility,
    create_speos_config,
    pointcloud_to_stl,
    # Deprecated functions (하위 호환성)
    xmp_to_txt,
    wait_for_file_update,
    generate_origin_pointcloud
)

# Reward Calculation Functions
from env.calculate_reward import (
    calculate_speos_reward,
    SpeosRewardCalculator
)

# 하위 호환성을 위한 별칭
RewardCalculator = SpeosRewardCalculator

# 하위 호환성을 위한 별칭
SpeosIntegration = SpeosUtility

# Data Visualization utilities
from .data_visualization import (
    TrainingVisualizer,
    create_training_visualizer,
    quick_analysis,
    advanced_analysis,
    expert_analysis,
    # 하위 호환성 함수들
    plot_training_progress,
    plot_rewards,
    plot_losses,
    save_plots,
    create_dashboard
)

# CAD Visualization utilities
from .cad_visualization import (
    CADVisualizer,
    visualize_pointcloud,
    show_pointcloud_viewer,
    visualize_stl,
    show_viewer
)

# General utilities
from .utils import (
    setup_logging,
    create_environment,
    save_model_and_stats,
    load_model_and_stats,
    evaluate_model,
    plot_training_curves,
    check_nan_inf,
    clip_gradients,
    safe_mean,
    safe_std,
    get_device,
    set_seed,
    format_time,
    create_summary_report,
    print_system_info,
    save_summary_report
)

# Safe evaluation utilities
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from safe_evaluation import (
        safe_episode_evaluation,
        safe_training_evaluation,
        safe_final_evaluation,
        create_step_limit_wrapper
    )
except ImportError:
    # Fallback for when safe_evaluation is not available
    safe_episode_evaluation = None
    safe_training_evaluation = None
    safe_final_evaluation = None
    create_step_limit_wrapper = None

__all__ = [
    # SPEOS 통합 관련
    'SpeosUtility',
    'SpeosIntegration',  # 하위 호환성 별칭
    'create_speos_config',
    'pointcloud_to_stl',  # 통합된 포인트클라우드→STL 변환 함수
    # Deprecated functions (하위 호환성)
    'xmp_to_txt',
    'wait_for_file_update',
    'generate_origin_pointcloud',
    
    # 리워드 계산 관련
    'calculate_speos_reward',
    'SpeosRewardCalculator',
    'RewardCalculator',  # 하위 호환성 별칭
    
    # 데이터 시각화 관련
    'TrainingVisualizer',
    'create_training_visualizer', 
    'quick_analysis',
    'advanced_analysis',
    'expert_analysis',
    'plot_training_progress',
    'plot_rewards',
    'plot_losses',
    'save_plots',
    'create_dashboard',
    
    # CAD 시각화 관련
    'CADVisualizer',
    'visualize_pointcloud',
    'show_pointcloud_viewer',
    'visualize_stl',
    'show_viewer',
    
    # 일반 유틸리티
    'setup_logging',
    'create_environment',
    'save_model_and_stats',
    'load_model_and_stats',
    'evaluate_model',
    'plot_training_curves',
    'check_nan_inf',
    'clip_gradients',
    'safe_mean',
    'safe_std',
    'get_device',
    'set_seed',
    'format_time',
    'create_summary_report',
    'print_system_info',
    'save_summary_report',
    
    # 안전한 평가 유틸리티
    'safe_episode_evaluation',
    'safe_training_evaluation',
    'safe_final_evaluation',
    'create_step_limit_wrapper'
]
