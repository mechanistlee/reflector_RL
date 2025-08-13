"""
Configuration file for SAC training with optical simulation

멀티 리플렉터 설정:
=============================
- num_reflectors: 동시에 처리할 리플렉터 개수 (기본값: 100)
- reflector_spacing_x: 리플렉터 간 X축 간격 (기본값: 200mm)
- reflector_spacing_y: 리플렉터 간 Y축 간격 (기본값: 0mm)
- reflector_spacing_z: 리플렉터 간 Z축 간격 (기본값: 0mm)

리플렉터 배치:
- reflector1: (grid_origin_x, grid_origin_y, grid_origin_z)
- reflector2: (grid_origin_x + 200mm, grid_origin_y, grid_origin_z)
- reflector3: (grid_origin_x + 400mm, grid_origin_y, grid_origin_z)
- ...
- reflector100: (grid_origin_x + 19800mm, grid_origin_y, grid_origin_z)

출력 파일:
- Direct.1.Intensity.1.xmp (reflector1 결과)
- Direct.1.Intensity.2.xmp (reflector2 결과)
- ...
- Direct.1.Intensity.{num_reflectors}.xmp (마지막 리플렉터 결과)
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

# Environment settings
ENV_NAME = "SpeosEnv-v1"
RENDER_MODE = "rgb_array"  # for training, "human" for visualization

# Environment configuration
ENV_CONFIG = {
    # 기본 환경 설정
    "max_episode_steps": 50,
    "apply_time_limit": True,
    "respect_env_termination": True,  # 환경의 자연스러운 종료 신호도 처리
    
    # 그리드 설정
    "grid_rows": 5,
    "grid_cols": 5,
    "grid_cell_size_x": 10,  # 🔧 그리드 1칸의 X축 크기 (mm) - 여기서 설정!
    "grid_cell_size_y": 10,  # 🔧 그리드 1칸의 Y축 크기 (mm) - 여기서 설정!
    "grid_origin_x": 0.0,     # X축 시작점 (mm)
    "grid_origin_y": 0.0,     # Y축 시작점 (mm)
    "grid_origin_z": 0.0,     # Z축 초기값 (mm)
    
    # 🎯 멀티 리플렉터 설정
    "num_reflectors": 100,    # 🔧 동시에 처리할 리플렉터 개수 
    "reflector_spacing_x": 200.0,  # 🔧 리플렉터 간 X축 간격 (mm) - 여기서 설정!
    "reflector_spacing_y": 0.0,    # Y축 간격 (mm) - 필요시 사용
    "reflector_spacing_z": 0.0,    # Z축 간격 (mm) - 필요시 사용
    "initial_shape_diversity": True,  # 🔥 각 리플렉터 초기 형상에 다양성 추가
    
    # 시뮬레이션 설정
    "ray_count": 20000000,
    "wavelength_range": (400.0, 700.0),
    "reflection_model": "lambertian",
    "target_intensity_threshold": 0.8,
    "xmp_update_timeout": 120,

    
    # 환경 설정
    "max_steps": 50,
    "action_min": -2.5,
    "action_max": 2.5,
    "z_min": -25.0,
    "z_max": 25.0,
    "enable_visualization": True,    # 🎯 시각화 활성화 (메쉬 시각화 위해 필요)
    "visualize_interval": 1,
    "enable_mesh_visualization": True,  # 🎯 실시간 메쉬 시각화 활성화
    
    # 파일 경로 설정
    "xmp_file_path": "cad\\SPEOS output files\\speos\\Direct.1.Intensity.1.xmp",
    "control_file_path": "env\\SpeosControl.txt",
    "txt_output_path": "cad\\intensity_output.txt",
    "mesh_save_path": "cad\\Reflector.stl",

    # 변환 설정
    "flip_updown": False,     # 상하 반전 여부
    "flip_leftright": False,   # 좌우 반전 여부
    
    # LED 출력 설정
    "led_output": 100,  # LED 출력 (초기값: 100)
    
    # 연동 설정
    "use_real_simulator": True,  # True로 설정하면 실제 시뮬레이터 연동
    "xmp_update_timeout": 150,  # 🔧 1번 XMP 파일 업데이트 대기 타임아웃 (초) 
    "xmp_secondary_timeout": 2,  # 🔧 2~100번 XMP 파일 업데이트 대기 타임아웃 (초) - 5→2초로 최적화
    "use_gpu_batch_processing": True,  # 🚀 GPU 배치 처리로 XMP 변환 가속화 (4개 이상 리플렉터시)
    
    # 🔥 개별 리플렉터 액션 생성 설정
    "use_agent_individual_actions": True,  # True: 에이전트가 각 리플렉터 상태별 개별 액션 생성, False: 기본 액션 변조 방식
    "enable_state_adaptive_actions": True,  # 리플렉터 상태에 따른 적응적 액션 활성화
    "use_position_weighting": True,  # 리플렉터 위치에 따른 가중치 적용
    "use_performance_adaptation": True,  # 이전 성능에 따른 액션 적응
    "smoothing_threshold": 8.0,  # 표면 평활화 임계값 (mm)
    "boundary_protection": 2.0,  # Z값 경계 보호 범위 (mm)
    
    # 병렬 처리 설정
    "use_parallel_xmp_processing": True,   # 병렬 XMP 처리 활성화
    "max_parallel_xmp_workers": 4,         # 최대 병렬 워커 수 (COM 객체 개수)
    
    # 💡 사용법: 느린 시스템에서는 60~120초로 증가, 빠른 시스템에서는 10~20초로 감소
}

# Network architecture
NETWORK_CONFIG = {
    "net_arch": [512, 512, 256],     # Network architecture for policy and value networks
    "activation_fn": "relu",    # Activation function
    "log_std_init": -3,         # Initial log standard deviation
    "log_std_min": -20,         # Minimum log standard deviation
    "log_std_max": 2            # Maximum log standard deviation
}

# Training hyperparameters (optimized for individual reflector learning)
TRAINING_CONFIG = {
    "total_timesteps": 300,  # Default timesteps for SPEOS training
    "learning_rate": 3e-4,      # Slightly lower for stable individual reflector learning
    "buffer_size": 100000,      # Normal buffer size (1 step = 100 experiences)
    "batch_size": 256,           # Reduced batch size for memory efficiency
    "tau": 0.02,
    "gamma": 0.995,              # Slightly lower gamma for episodic tasks
    "train_freq": 1,            # Train every step for better sample efficiency
    "gradient_steps": 8,
    "learning_starts": 5,    # Reduced for faster startup with more experience generation / step 기준
    "ent_coef": "auto",
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "policy_kwargs": {
        "net_arch": NETWORK_CONFIG["net_arch"],           # Reference to NETWORK_CONFIG
        "activation_fn": NETWORK_CONFIG["activation_fn"]  # Reference to NETWORK_CONFIG
    }
}

# Test/evaluation settings
TEST_CONFIG = {
    "n_eval_episodes": 3,      # 최종 테스트 에피소드 수 (1 → 3으로 증가)
    "render_episodes": 2,      # 녹화할 에피소드 수
    "deterministic": True,
    "record_video": True,
    "video_length": 100,
    "save_stats": True,
    "max_episode_steps": 25,   # 🚨 에피소드당 최대 스텝 제한 (5 → 50으로 증가)
    "timeout_penalty": -5.0    # 🚨 타임아웃 시 추가 페널티
}
# Default configurations for different training modes
QUICK_TEST_CONFIG = {
    "total_timesteps": 100,
    "learning_starts": 10,
    "log_interval": 5,
    "eval_freq": 20,
    "save_freq": 50
}

FULL_TRAINING_CONFIG = {
    "total_timesteps": 2000,
    "learning_starts": 100,
    "log_interval": 10,
    "eval_freq": 100,
    "save_freq": 200
}

EXTENDED_TRAINING_CONFIG = {
    "total_timesteps": 10000,
    "learning_starts": 200,
    "log_interval": 20,
    "eval_freq": 200,
    "save_freq": 500
}

# Training stability settings
STABILITY_CONFIG = {
    "clip_rewards": True,
    "reward_clip_range": (-10.0, 10.0),
    "gradient_clip_value": 0.5,
    "max_grad_norm": 1.0,
    "entropy_coef_min": 0.01,
    "entropy_coef_max": 0.2,
    "learning_rate_min": 1e-6,
    "learning_rate_max": 5e-3,
    "nan_check_interval": 1000,
    "save_interval": 10000
}

# Logging and evaluation
LOGGING_CONFIG = {
    "log_interval": 10,
    "eval_freq": 100,        # 평가 주기를 10,000 → 50,000으로 증가 (평가 횟수 대폭 감소)
    "eval_episodes": 5,        # 중간 평가 에피소드 수를 10 → 5로 감소
    "max_eval_episode_steps": 200,  # 🚨 중간 평가 시 에피소드당 최대 스텝 제한
    "save_freq": 100,
    "verbose": 1,
    "tensorboard_log": "./sac_speos_tensorboard/",
    "log_path": "./speos_logs/"
}

# File paths
PATHS_CONFIG = {
    "models_dir": "./models/",
    "meshes_dir": "./cad/meshes/",
    "logs_dir": "./logs/",
    "tensorboard_dir": "./tensorboard_logs/",
    "results_dir": "./results/",
    "videos_dir": "./videos/",
    "plots_dir": "./plots/"
}


# Visualization settings
VIS_CONFIG = {
    "plot_training_curves": True,
    "plot_reward_distribution": True,
    "plot_episode_length": True,
    "save_plots": True,           # 플롯 저장 활성화
    "show_plots": False,          # 화면 표시 비활성화 (저장만)
    "plot_format": "png",         # 저장 포맷 (png, jpg, pdf, svg)
    "dpi": 300                    # 고해상도 저장
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "visualization_level": 1,  # 1: Basic, 2: Advanced, 3: Expert
    "save_plots": True,
    "show_interactive": True,  # 화면에 그래프 표시
    "create_dashboard": False,
    "detailed_logging": {
        "log_every_n_steps": 100,
        "log_every_n_episodes": 10,
        "save_raw_data": False,
        "moving_window_size": 1000,
        "sample_rate": 0.1  # 메모리 절약을 위한 샘플링 비율
    },
    "plot_settings": {
        "figure_size": (15, 10),
        "dpi": 300,
        "style": "seaborn-v0_8",
        "save_format": "png",
        "interactive_backend": "plotly"
    }
}



# =============================================================================
# Environment Configuration Classes
# =============================================================================

from enum import Enum
from dataclasses import dataclass, field

class SimulationType(Enum):
    """시뮬레이션 타입 열거형"""
    REAL = "real"               # 실제 시뮬레이터 연동
    MOCK = "mock"               # 모의 시뮬레이션
    PLACEHOLDER = "placeholder"  # 플레이스홀더 시뮬레이션

@dataclass
class TrainingConfig:
    """
    훈련 전용 설정 클래스 (고급 사용자용)
    
    ⚠️  주의: 일반적인 사용에서는 ENV_CONFIG를 사용하세요!
    이 클래스는 고급 설정이나 특별한 요구사항이 있을 때만 사용됩니다.
    
    기본 runtime 설정: ENV_CONFIG (config.py의 딕셔너리)
    고급/특별 설정: TrainingConfig (이 데이터클래스)
    """
    
    # 기본 그리드 설정 (ENV_CONFIG와 동일한 기본값)
    grid_rows: int = 10
    grid_cols: int = 10
    grid_cell_size_x: float = 1.0  # 그리드 1칸의 X축 크기 (mm)
    grid_cell_size_y: float = 1.0  # 그리드 1칸의 Y축 크기 (mm)
    grid_origin_x: float = 0.0     # X축 시작점 (mm)
    grid_origin_y: float = 0.0     # Y축 시작점 (mm)
    grid_origin_z: float = 0.0     # Z축 초기값 (mm)
    
    # 멀티 리플렉터 설정 (ENV_CONFIG와 동일한 기본값)
    num_reflectors: int = 100          # 동시에 처리할 리플렉터 개수
    reflector_spacing_x: float = 200.0 # 리플렉터 간 X축 간격 (mm)
    reflector_spacing_y: float = 0.0   # 리플렉터 간 Y축 간격 (mm)
    reflector_spacing_z: float = 0.0   # 리플렉터 간 Z축 간격 (mm)
    
    # 그리드 전체 물리적 범위 설정 (mm 단위) - 선택사항
    # None이면 grid_cell_size와 grid_rows/cols로 자동 계산
    physical_range_x: Optional[float] = None  # 전체 X축 범위 (mm)
    physical_range_y: Optional[float] = None  # 전체 Y축 범위 (mm)
    
    # 시뮬레이션 설정 (ENV_CONFIG와 동일한 기본값)
    simulation_type: SimulationType = SimulationType.MOCK
    use_real_simulator: bool = True
    xmp_update_timeout: int = 30  # XMP 파일 업데이트 대기 타임아웃 (초) - 시스템 성능에 따라 조절
    xmp_secondary_timeout: int = 2  # 2~100번 XMP 파일 업데이트 대기 타임아웃 (초) - 5→2초로 최적화
    use_gpu_batch_processing: bool = True  # 🚀 GPU 배치 처리로 XMP 변환 가속화
    ray_count: int = 100000
    wavelength_range: tuple = (400.0, 700.0)
    reflection_model: str = "lambertian"
    target_intensity_threshold: float = 0.8
    led_output: float = 100.0
    
    # 환경 설정 (ENV_CONFIG와 동일한 기본값)
    max_steps: int = 100
    action_min: float = -1.0
    action_max: float = 1.0
    z_min: float = -10.0
    z_max: float = 10.0
    enable_visualization: bool = False
    visualize_interval: int = 10
    
    # 파일 경로 설정 (프로젝트 루트 기준 상대 경로)
    project_root: str = "."
    xmp_file_path: str = "cad/SPEOS output files/speos/Direct.1.Intensity.1.xmp"
    control_file_path: str = "env/SpeosControl.txt"
    txt_output_path: str = "cad/intensity_output.txt"
    stl_output_path: str = "cad/reflector_output.stl"
    ply_output_path: str = "cad/reflector_output.ply"
    
    # 변환 설정 (ENV_CONFIG와 동일한 기본값)
    flip_updown: bool = False
    flip_leftright: bool = False
    
    def __post_init__(self):
        """초기화 후 검증"""
        # LED 출력이 양수인지 확인
        if self.led_output <= 0:
            raise ValueError(f"LED 출력은 양수여야 합니다: {self.led_output}")
        
        # 그리드 크기가 양수인지 확인
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError(f"그리드 크기는 양수여야 합니다: {self.grid_rows}x{self.grid_cols}")
        
        # 그리드 셀 크기가 양수인지 확인
        if self.grid_cell_size_x <= 0 or self.grid_cell_size_y <= 0:
            raise ValueError(f"그리드 셀 크기는 양수여야 합니다: {self.grid_cell_size_x}x{self.grid_cell_size_y}mm")
        
        # 멀티 리플렉터 설정 검증
        if self.num_reflectors <= 0:
            raise ValueError(f"리플렉터 개수는 양수여야 합니다: {self.num_reflectors}")
        
        if self.reflector_spacing_x < 0:
            raise ValueError(f"리플렉터 X축 간격은 음수일 수 없습니다: {self.reflector_spacing_x}mm")
        
        # physical_range가 None이면 자동 계산
        if self.physical_range_x is None:
            self.physical_range_x = (self.grid_cols - 1) * self.grid_cell_size_x
        if self.physical_range_y is None:
            self.physical_range_y = (self.grid_rows - 1) * self.grid_cell_size_y
        
        # 실제 시뮬레이터 연동 시 파일 경로가 설정되어 있는지 확인
        if self.use_real_simulator or self.simulation_type == SimulationType.REAL:
            required_paths = [self.xmp_file_path, self.control_file_path, self.txt_output_path]
            for path in required_paths:
                if not path or path.strip() == "":
                    raise ValueError(f"실제 시뮬레이터 연동 시 파일 경로가 필요합니다: {path}")
    
    def get_grid_spacing_x(self) -> float:
        """그리드 X축 간격(mm) 반환"""
        return self.grid_cell_size_x
    
    def get_grid_spacing_y(self) -> float:
        """그리드 Y축 간격(mm) 반환"""
        return self.grid_cell_size_y
    
    def get_grid_x_coords(self) -> np.ndarray:
        """그리드 X좌표 배열 반환 (grid_origin_x를 중심으로) (mm)"""
        # grid_origin_x를 중심으로 하는 좌표 생성
        start_x = self.grid_origin_x - self.physical_range_x / 2
        end_x = self.grid_origin_x + self.physical_range_x / 2
        return np.linspace(start_x, end_x, self.grid_cols)
    
    def get_grid_y_coords(self) -> np.ndarray:
        """그리드 Y좌표 배열 반환 (grid_origin_y를 중심으로) (mm)"""
        # grid_origin_y를 중심으로 하는 좌표 생성
        start_y = self.grid_origin_y - self.physical_range_y / 2
        end_y = self.grid_origin_y + self.physical_range_y / 2
        return np.linspace(start_y, end_y, self.grid_rows)
    
    def get_total_physical_area(self) -> float:
        """전체 그리드의 물리적 면적 반환 (mm²)"""
        return self.physical_range_x * self.physical_range_y
    
    def get_reflector_positions(self) -> List[Tuple[float, float, float]]:
        """모든 리플렉터의 중심 좌표 반환 (mm)"""
        positions = []
        for i in range(self.num_reflectors):
            x = self.grid_origin_x + i * self.reflector_spacing_x
            y = self.grid_origin_y + i * self.reflector_spacing_y
            z = self.grid_origin_z + i * self.reflector_spacing_z
            positions.append((x, y, z))
        return positions
    
    def get_reflector_position(self, reflector_id: int) -> Tuple[float, float, float]:
        """특정 리플렉터의 중심 좌표 반환 (mm)"""
        if reflector_id < 0 or reflector_id >= self.num_reflectors:
            raise ValueError(f"리플렉터 ID가 범위를 벗어났습니다: {reflector_id} (0-{self.num_reflectors-1})")
        
        x = self.grid_origin_x + reflector_id * self.reflector_spacing_x
        y = self.grid_origin_y + reflector_id * self.reflector_spacing_y
        z = self.grid_origin_z + reflector_id * self.reflector_spacing_z
        return (x, y, z)
    
    def get_reflector_name(self, reflector_id: int) -> str:
        """리플렉터 이름 반환 (reflector1, reflector2, ...)"""
        if reflector_id < 0 or reflector_id >= self.num_reflectors:
            raise ValueError(f"리플렉터 ID가 범위를 벗어났습니다: {reflector_id} (0-{self.num_reflectors-1})")
        return f"reflector{reflector_id + 1}"
    
    def get_total_workspace_size(self) -> Tuple[float, float, float]:
        """전체 작업공간 크기 반환 (모든 리플렉터 포함) (mm)"""
        if self.num_reflectors == 1:
            return (self.physical_range_x, self.physical_range_y, 0.0)
        
        # 마지막 리플렉터까지의 전체 X축 범위 계산
        last_reflector_x = (self.num_reflectors - 1) * self.reflector_spacing_x
        total_x = last_reflector_x + self.physical_range_x
        
        # Y, Z축도 동일하게 계산
        last_reflector_y = (self.num_reflectors - 1) * self.reflector_spacing_y
        total_y = max(self.physical_range_y, last_reflector_y + self.physical_range_y)
        
        last_reflector_z = (self.num_reflectors - 1) * self.reflector_spacing_z
        total_z = last_reflector_z
        
        return (total_x, total_y, total_z)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, SimulationType):
                result[key] = value.value
            elif isinstance(value, tuple):
                result[key] = value
            else:
                result[key] = value
        return result
    

def get_config(config_name: str) -> Dict[str, Any]:
    """Get configuration by name"""
    configs = {
        "training": TRAINING_CONFIG,  # Updated to use unified config
        "network": NETWORK_CONFIG,
        "stability": STABILITY_CONFIG,
        "logging": LOGGING_CONFIG,
        "paths": PATHS_CONFIG,
        "test": TEST_CONFIG,
        "visualization": VIS_CONFIG
    }
    return configs.get(config_name, {})

def get_model_path(model_name: str, models_dir: Optional[str] = None) -> str:
    """Get full path for model file"""
    if models_dir is None:
        models_dir = PATHS_CONFIG["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, f"{model_name}.zip")

def get_log_path(log_name: str, logs_dir: Optional[str] = None) -> str:
    """Get full path for log file"""
    if logs_dir is None:
        logs_dir = PATHS_CONFIG["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, log_name)

def create_directories():
    """Create all necessary directories"""
    for path in PATHS_CONFIG.values():
        os.makedirs(path, exist_ok=True)

def update_config(config_name: str, updates: Dict[str, Any]) -> None:
    """Update configuration values"""
    configs = {
        "training": TRAINING_CONFIG,  # Updated to use unified config
        "network": NETWORK_CONFIG,
        "stability": STABILITY_CONFIG,
        "logging": LOGGING_CONFIG,
        "paths": PATHS_CONFIG,
        "test": TEST_CONFIG,
        "visualization": VIS_CONFIG
    }
    
    if config_name in configs:
        configs[config_name].update(updates)
    else:
        raise ValueError(f"Unknown config name: {config_name}")

def get_default_model_name() -> str:
    """Get default model name with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sac_speos_{timestamp}"


def get_training_mode_config(mode: str) -> Dict[str, Any]:
    """Get configuration for different training modes"""
    modes = {
        "default": TRAINING_CONFIG,  # Use TRAINING_CONFIG for default mode
        "quick": QUICK_TEST_CONFIG,
        "full": FULL_TRAINING_CONFIG,
        "extended": EXTENDED_TRAINING_CONFIG
    }
    return modes.get(mode, TRAINING_CONFIG)  # Updated to use unified config as default


def create_training_config(project_root: str = ".", 
                          use_real_simulator: bool = False,
                          **kwargs) -> TrainingConfig:
    """훈련 설정 생성 편의 함수"""
    config_dict = {
        "project_root": project_root,
        "use_real_simulator": use_real_simulator,
        "simulation_type": SimulationType.REAL if use_real_simulator else SimulationType.MOCK
    }
    config_dict.update(kwargs)
    return TrainingConfig(**config_dict)

def get_default_config() -> TrainingConfig:
    """기본 설정 반환"""
    return TrainingConfig()

def validate_config(config: TrainingConfig) -> bool:
    """설정 검증"""
    try:
        # led_output 검증
        if config.led_output is None or config.led_output <= 0:
            print(f"❌ LED 출력값이 잘못되었습니다: {config.led_output}")
            return False
        
        # 그리드 크기 검증
        if config.grid_rows <= 0 or config.grid_cols <= 0:
            print(f"❌ 그리드 크기가 잘못되었습니다: {config.grid_rows}x{config.grid_cols}")
            return False
        
        # 멀티 리플렉터 검증
        if config.num_reflectors <= 0:
            print(f"❌ 리플렉터 개수가 잘못되었습니다: {config.num_reflectors}")
            return False
        
        print(f"✅ 설정 검증 성공:")
        print(f"   - LED 출력: {config.led_output}")
        print(f"   - 그리드 크기: {config.grid_rows}x{config.grid_cols}")
        print(f"   - 리플렉터 개수: {config.num_reflectors}")
        print(f"   - 리플렉터 간격: {config.reflector_spacing_x}mm")
        print(f"   - 전체 작업공간: {config.get_total_workspace_size()}")
        print(f"   - 실제 시뮬레이터 사용: {config.use_real_simulator}")
        print(f"   - 시뮬레이션 타입: {config.simulation_type.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 설정 검증 실패: {e}")
        return False
