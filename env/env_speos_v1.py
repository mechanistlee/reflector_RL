"""
SPEOS 강화학습용 물리 시뮬레이터
===============================

SPEOS 광선 추적을 사용한 강화학습용 광학 시뮬레이션 환경입니다.

이 환경은 Gymnasium 표준 인터페이스를 따르며 
리플렉터 최적화를 위한 포인트 클라우드에서 STL로의 변환을 통합합니다.
"""
import os
import time
import datetime
import numpy as np
import warnings
import gymnasium as gym

# NumPy 경고 필터링 (0으로 나누기 경고 등)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from enum import Enum
from dataclasses import dataclass, field
import threading
import queue
import open3d as o3d
import subprocess
import win32com.client
import pythoncom
import concurrent.futures
import random
import uuid

# temporary_speos_utility에서 유틸리티 임포트
from .temporary_speos_utility import SpeosUtility, xmp_to_txt, pointcloud_to_stl

# 리워드 계산 함수 임포트
from .calculate_reward import calculate_speos_reward

# 설정 클래스 임포트
try:
    from config import SpeosTrainingConfig, SimulationType, validate_speos_config
except ImportError:
    # 설정 모듈을 사용할 수 없을 때의 대체 방안
    SpeosTrainingConfig = None
    SimulationType = None
    validate_speos_config = None



# =============================================================================
# 설정 클래스들
# =============================================================================

class SimulationType(Enum):
    """지원되는 시뮬레이션 타입들"""
    SPEOS = "speos"


class BaseSimConfig:
    """모든 시뮬레이션 환경을 위한 기본 설정 클래스"""
    
    def __init__(self):
        # 공통 매개변수들
        self.max_steps: int = 100
        self.grid_rows: int = 10
        self.grid_cols: int = 10
        self.action_min: float = -1.0
        self.action_max: float = 1.0
        self.z_min: float = -10.0
        self.z_max: float = 10.0
        
        # 시각화
        self.enable_visualization: bool = False
        self.visualize_interval: int = 10
        
        # 로깅
        self.log_level: str = "INFO"
        self.log_simulation_data: bool = True


class SpeosConfig(BaseSimConfig):
    """SPEOS 광학 시뮬레이션용 설정"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # SPEOS specific parameters
        self.wavelength_range: Tuple[float, float] = kwargs.get('wavelength_range', (400.0, 700.0))  # nm
        self.ray_count: int = kwargs.get('ray_count', 100000)
        self.reflection_model: str = kwargs.get('reflection_model', "lambertian")
        self.material_properties: Dict = kwargs.get('material_properties', {
            "reflectance": 0.8,
            "roughness": 0.1
        })
        self.target_intensity_threshold: float = kwargs.get('target_intensity_threshold', 0.8)
        
        # Environment grid settings (추가)
        self.grid_rows: int = kwargs.get('grid_rows', 10)
        self.grid_cols: int = kwargs.get('grid_cols', 10)
        self.max_steps: int = kwargs.get('max_steps', 100)
        # max_episode_steps 별칭 추가 (호환성 위해)
        self.max_episode_steps: int = self.max_steps
        
        # 🎯 Grid cell size 설정 (mm 단위) - 물리적 치수의 핵심!
        self.grid_cell_size_x: float = kwargs.get('grid_cell_size_x', 1.0)  # 그리드 1칸의 X축 크기 (mm)
        self.grid_cell_size_y: float = kwargs.get('grid_cell_size_y', 1.0)  # 그리드 1칸의 Y축 크기 (mm)
        
        # Grid origin 설정 (mm 단위)
        self.grid_origin_x: float = kwargs.get('grid_origin_x', 0.0)  # X축 시작점 (mm)
        self.grid_origin_y: float = kwargs.get('grid_origin_y', 0.0)  # Y축 시작점 (mm)
        self.grid_origin_z: float = kwargs.get('grid_origin_z', 0.0)  # Z축 초기값 (mm)
        
        # 🎯 멀티 리플렉터 설정 (새로 추가)
        self.num_reflectors: int = kwargs.get('num_reflectors', 100)          # 동시에 처리할 리플렉터 개수
        self.reflector_spacing_x: float = kwargs.get('reflector_spacing_x', 200.0)  # 리플렉터 간 X축 간격 (mm)
        self.reflector_spacing_y: float = kwargs.get('reflector_spacing_y', 0.0)    # Y축 간격 (mm)
        self.reflector_spacing_z: float = kwargs.get('reflector_spacing_z', 0.0)    # Z축 간격 (mm)
        
        # Action space settings (추가)
        self.action_min: float = kwargs.get('action_min', -1.0)
        self.action_max: float = kwargs.get('action_max', 1.0)
        self.z_min: float = kwargs.get('z_min', -10.0)
        self.z_max: float = kwargs.get('z_max', 10.0)
        
        # 프로젝트 루트 디렉토리 찾기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = kwargs.get('project_root', os.path.dirname(current_dir))  # env 폴더의 상위 폴더
        
        # File paths - 프로젝트 루트 기준으로 절대 경로 설정
        self.speos_script_path: str = os.path.join(project_root, "speos_script")
        self.stl_output_path: str = os.path.join(project_root, "cad", "Reflector.stl")
        
        # SPEOS 파일 경로 설정 - 프로젝트 루트 기준
        default_xmp_path = os.path.join(project_root, "cad", "SPEOS output files", "speos", "Direct.1.Intensity.1.xmp")
        default_control_path = os.path.join(project_root, "env", "SpeosControl.txt")
        default_txt_path = os.path.join(project_root, "data", "simulation_result", "intensity_output.txt")
        
        self.xmp_file_path: str = kwargs.get('xmp_file_path', default_xmp_path)
        self.control_file_path: str = kwargs.get('control_file_path', default_control_path)
        self.txt_output_path: str = kwargs.get('txt_output_path', default_txt_path)
        
        # Target 파일 저장 경로 추가
        default_target_path = os.path.join(project_root, "data", "target", "target_intensity.txt")
        self.target_output_path: str = kwargs.get('target_output_path', default_target_path)
        
        # SPEOS 변환 설정 추가
        self.flip_updown: bool = kwargs.get('flip_updown', False)      # 상하 반전 여부
        self.flip_leftright: bool = kwargs.get('flip_leftright', False)   # 좌우 반전 여부
        
        # SPEOS LED 출력 설정 추가
        self.led_output: float = kwargs.get('led_output', 100.0)      # LED 출력 (초기값: 100)
        
        # SPEOS 연동 설정
        self.use_real_speos: bool = kwargs.get('use_real_speos', True)   # True로 설정하면 실제 SPEOS 연동, False면 플레이스홀더
        
        # SPEOS 타임아웃 설정 추가
        self.xmp_update_timeout: int = kwargs.get('xmp_update_timeout', 30)  # XMP 파일 업데이트 대기 타임아웃 (초)
        
        # Visualization settings (추가)
        self.enable_visualization: bool = kwargs.get('enable_visualization', False)
        self.visualize_interval: int = kwargs.get('visualize_interval', 10)
    
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
            physical_range_x = (self.grid_cols - 1) * self.grid_cell_size_x
            physical_range_y = (self.grid_rows - 1) * self.grid_cell_size_y
            return (physical_range_x, physical_range_y, 0.0)
        
        # 마지막 리플렉터까지의 전체 X축 범위 계산
        last_reflector_x = (self.num_reflectors - 1) * self.reflector_spacing_x
        physical_range_x = (self.grid_cols - 1) * self.grid_cell_size_x
        total_x = last_reflector_x + physical_range_x
        
        # Y, Z축도 동일하게 계산
        last_reflector_y = (self.num_reflectors - 1) * self.reflector_spacing_y
        physical_range_y = (self.grid_rows - 1) * self.grid_cell_size_y
        total_y = max(physical_range_y, last_reflector_y + physical_range_y)
        
        last_reflector_z = (self.num_reflectors - 1) * self.reflector_spacing_z
        total_z = last_reflector_z
        
        return (total_x, total_y, total_z)

# =============================================================================
# 추상 기본 환경
# =============================================================================

class SpeosEnv(gym.Env):
    """
    SPEOS 광학 시뮬레이션을 위한 강화학습 환경.
    단일 리플렉터 학습 + 100개 병렬 경험 생성 구조.
    """
    
    def __init__(self, config: BaseSimConfig, sample_data: Optional[Dict] = None):
        super().__init__()
        
        self.config = config
        self.sample_data = sample_data or {}
        
        # 환경 인스턴스 ID 생성 (디버깅용)
        import uuid
        self.instance_id = str(uuid.uuid4())[:8]
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.instance_id}]")
        
        # 환경 생성 로그
        self.logger.info(f"🔧 새로운 SPEOS 환경 인스턴스 생성 (ID: {self.instance_id})")
        
        # Common attributes
        self.grid_size = config.grid_rows * config.grid_cols
        self.episode_step = 0
        self.episode_reward = 0.0
        self.current_step = 0
        self.simulation_history = []
        
        # 학습 시작 시간 기록
        self.training_start_time = time.time()
        
        # 🎯 경험 버퍼 관리 속성 추가
        self._experiences_buffer = []
        self._buffer_file_index = 1  # 현재 저장 중인 파일 인덱스
        self._total_experiences_saved = 0  # 지금까지 저장된 총 경험 수
        self._model_name = getattr(config, 'model_name', 'default_model')  # 모델 이름 (외부에서 설정 가능)
        
        # config에 action_size 속성 추가 (누락된 속성)
        if not hasattr(config, 'action_size'):
            config.action_size = self.grid_size
        
        # 🎯 시각화 매니저 설정
        self.enable_mesh_visualization = config.enable_visualization
        self.visualization_manager = None
        self.mesh_visualizer = None
        self._vis_window = None  # STL 시각화 윈도우
        if self.enable_mesh_visualization:
            try:
                from utils.cad_visualization import CADVisualizer
                self.mesh_visualizer = CADVisualizer()
                self.visualization_update_interval = getattr(config, 'visualize_interval', 10)
                self.logger.info(f"🎯 실시간 메쉬 시각화 활성화 (간격: {self.visualization_update_interval} 스텝)")
            except ImportError as e:
                self.logger.warning(f"❌ CAD 시각화 모듈 임포트 실패: {e}")
                self.enable_mesh_visualization = False
        
        # Define spaces (to be overridden by subclasses)
        self._setup_spaces()
        
        # Initialize simulation state
        self._initialize_simulation()


    # =============================================================================

    def _setup_spaces(self):
        """Setup observation and action spaces for SPEOS environment"""
        # 🎯 Observation space: Combined observations from all reflectors
        # Single reflector: Z-values + intensity map + 2 scalars (efficiency, total_flux)
        single_reflector_z_size = self.grid_size  # 단일 리플렉터 Z-values (10×10 = 100)
        optical_data_size = (self.config.grid_rows * self.config.grid_cols +  # intensity map (10×10 = 100)
                           2)  # efficiency + total_flux (2개 스칼라)
        
        single_reflector_observation_size = single_reflector_z_size + optical_data_size  # 100 + 100 + 2 = 202
        
        # 🔥 All reflectors combined observation size
        total_observation_size = single_reflector_observation_size * self.config.num_reflectors  # 202 × num_reflectors
        
        if self.config.num_reflectors > 1:
            self.logger.info(f"🔧 Multi-reflector observation space setup:")
            self.logger.info(f"   - Single reflector Z-values: {single_reflector_z_size}")
            self.logger.info(f"   - Single reflector optical data: {optical_data_size}")
            self.logger.info(f"   - Single reflector observation size: {single_reflector_observation_size}")
            self.logger.info(f"   - Combined {self.config.num_reflectors} reflectors observation: {total_observation_size}")
        else:
            self.logger.info(f"🔧 Single reflector observation space: {total_observation_size}")
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_observation_size,),
            dtype=np.float32
        )
        
        # Action space: 모든 리플렉터의 총 액션 공간 (리플렉터 수 × 그리드 크기)
        total_action_size = self.grid_size * self.config.num_reflectors
        self.action_space = spaces.Box(
            low=self.config.action_min, high=self.config.action_max,
            shape=(total_action_size,),  # Total action for all reflectors (125 dimensions for 5 reflectors)
            dtype=np.float32
        )
        
        if self.config.num_reflectors > 1:
            self.logger.info(f"🔧 Multi-reflector action space: {self.action_space.shape}")
            self.logger.info(f"   - Base action adapted individually for {self.config.num_reflectors} reflectors")
            self.logger.info(f"   - Each reflector gets state-specific action modification")
            self.logger.info(f"   - Each reflector generates individual learning experience")
            self.logger.info(f"   - 1 step = {self.config.num_reflectors} learning samples in buffer")
        else:
            self.logger.info(f"🔧 Single reflector action space: {self.action_space.shape}")
            
        self.logger.info(f"🔧 Action space: {self.action_space.shape}")
        self.logger.info(f"🔧 Observation space: {self.observation_space.shape}")
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Debug: 리셋 호출 이유 파악
        import traceback
        caller_info = traceback.extract_stack()[-2]
        self.logger.debug(f"🔄 환경 리셋 호출됨 - {caller_info.filename}:{caller_info.lineno} in {caller_info.name}")
        
        # Reset global counters
        self.episode_step = 0
        self.episode_reward = 0.0
        self.current_step = 0
        self.simulation_history.clear()
        
        return self._reflector_reset(seed, options)
    
    def _reflector_reset(self, seed: Optional[int] = None, 
                               options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset with new ReflectorClass architecture"""
        # Log episode start with caller info for debugging
        elapsed_time = self._format_elapsed_time(self.training_start_time)
        import traceback
        caller_info = traceback.extract_stack()[-2]
        #self.logger.info(f"🚀 새로운 에피소드 시작 (ID: {self.instance_id}) - 최대 스텝: {self.config.max_steps}, {self.config.num_reflectors} reflectors [경과: {elapsed_time}]")
        #self.logger.debug(f"   호출 위치: {caller_info.filename}:{caller_info.lineno} in {caller_info.name}")
        
        # Re-initialize simulation
        self._initialize_simulation()
        
        # Reset all reflectors to initial state
        for reflector in self.reflectors:
            reflector._update_state()  # Reset s0/s1 states
            reflector._initialize_Reflector()  # Reset to initial configuration
        
        # Get combined observation from all reflectors
        all_observations = []
        for reflector in self.reflectors:
            observation = reflector._get_observation()
            all_observations.extend(observation)
        
        combined_observation = np.array(all_observations, dtype=np.float32)
        
        info = {
            "episode_step": self.episode_step,
            "num_reflectors": len(self.reflectors),
            "observation_size_per_reflector": len(all_observations) // len(self.reflectors) if self.reflectors else 0
        }
        
        #self.logger.info(f"✅ 환경 초기화 완료 - {len(self.reflectors)} reflectors, Combined observation shape: {combined_observation.shape}")

        return combined_observation, info

    def _initialize_simulation(self):
        """Initialize SPEOS simulation state with individual reflector objects"""
        
        # 🔥 Initialize individual reflector objects
        self.reflectors = []  # 리플렉터 객체들을 저장할 리스트
        self.completed_episodes = 0  # 완료된 에피소드 개수
        
        if self.config.num_reflectors > 1:
            #self.logger.info(f"🔧 Initializing {self.config.num_reflectors} individual reflector objects...")
            
            # Create individual reflector objects with spacing-based positions
            for i in range(self.config.num_reflectors):
                reflector = ReflectorClass(i, self.config)
                reflector._initialize_Reflector()
                
                # Set reflector position using config spacing
                reflector_pos = self.config.get_reflector_position(i)
                reflector.center_position = reflector_pos
                
                self.reflectors.append(reflector)
                
                self.logger.debug(f"   Reflector {i+1}: initialized at position {reflector.center_position}")
                
            #self.logger.info(f"✅ {self.config.num_reflectors} individual reflector objects initialized")
            
        else:
            # Single reflector (backward compatibility)
            reflector = ReflectorClass(0, self.config)
            reflector._initialize_Reflector()
            reflector.center_position = self.config.get_reflector_position(0)
            self.reflectors.append(reflector)
        
        # Initialize simulation result cache to None
        self._last_simulation_result = None
        
        #self.logger.info("SPEOS simulation with reflector objects initialized")
    
    def _save_experience_buffer(self):
        """경험 버퍼를 파일로 저장 (buffer_size 단위로 분할)"""
        from config import TRAINING_CONFIG
        
        buffer_size_limit = TRAINING_CONFIG.get('buffer_size', 100000)
        
        if len(self._experiences_buffer) >= buffer_size_limit:
            # 저장할 데이터 준비
            buffer_to_save = self._experiences_buffer[:buffer_size_limit]
            self._experiences_buffer = self._experiences_buffer[buffer_size_limit:]
            
            # 파일 경로 생성
            os.makedirs("data/experience_buffer", exist_ok=True)
            file_path = f"data/experience_buffer/experience_buffer_{self._model_name}_{self._buffer_file_index}.h5"
            
            try:
                import h5py
                
                with h5py.File(file_path, 'w') as f:
                    # 경험 데이터를 HDF5 형식으로 저장
                    observations = []
                    actions = []
                    rewards = []
                    reflector_ids = []
                    step_numbers = []
                    
                    for exp in buffer_to_save:
                        observations.append(exp['observation'])
                        actions.append(exp['action'])
                        rewards.append(exp['reward'])
                        reflector_ids.append(exp['reflector_id'])
                        step_numbers.append(exp['step_number'])
                    
                    # NumPy 배열로 변환하여 저장
                    observations = np.array(observations)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    reflector_ids = np.array(reflector_ids)
                    step_numbers = np.array(step_numbers)
                    
                    f.create_dataset('observations', data=observations)
                    f.create_dataset('actions', data=actions)
                    f.create_dataset('rewards', data=rewards)
                    f.create_dataset('reflector_ids', data=reflector_ids)
                    f.create_dataset('step_numbers', data=step_numbers)
                    f.create_dataset('buffer_index', data=self._buffer_file_index)
                    f.create_dataset('total_experiences', data=len(buffer_to_save))
                
                self._total_experiences_saved += len(buffer_to_save)
                self.logger.info(f"📁 경험 버퍼 파일 저장 완료: {file_path} ({len(buffer_to_save)}개 경험)")
                self.logger.info(f"   - 파일 인덱스: {self._buffer_file_index}, 총 저장된 경험: {self._total_experiences_saved}")
                
                # 다음 파일 인덱스로 증가
                self._buffer_file_index += 1
                
            except ImportError:
                self.logger.warning("❌ h5py가 설치되지 않았습니다. 경험 버퍼를 pickle로 저장합니다.")
                # h5py가 없으면 pickle로 대체 저장
                import pickle
                pickle_path = f"data/experience_buffer/experience_buffer_{self._model_name}_{self._buffer_file_index}.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump(buffer_to_save, f)
                self._total_experiences_saved += len(buffer_to_save)
                self.logger.info(f"📁 경험 버퍼 pickle 파일 저장 완료: {pickle_path} ({len(buffer_to_save)}개 경험)")
                self._buffer_file_index += 1
                
            except Exception as e:
                self.logger.error(f"❌ 경험 버퍼 저장 실패: {e}")
                # 저장 실패 시 다시 버퍼에 추가
                self._experiences_buffer = buffer_to_save + self._experiences_buffer
    
    def set_model_name(self, model_name: str):
        """모델 이름 설정 (경험 버퍼 파일명에 사용)"""
        self._model_name = model_name
        self.logger.info(f"🏷️  모델 이름 설정: {model_name}")
    
    def get_experience_buffer_stats(self) -> Dict:
        """경험 버퍼 통계 반환"""
        return {
            'current_buffer_size': len(self._experiences_buffer),
            'buffer_file_index': self._buffer_file_index,
            'total_experiences_saved': self._total_experiences_saved,
            'model_name': self._model_name
        }
    
    def _run_simulation(self, action: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run SPEOS optical simulation with individual reflector actions"""
        import time
        start_time = time.time()

        # 🔥 현재 액션 저장 (개별 리플렉터 경험 데이터 생성용)
        self._last_action = action.copy()
        
        # 🔥 Apply individual actions to each reflector
        if self.config.num_reflectors > 1:
            self._apply_individual_reflector_actions(action)
        else:
            # Single reflector (backward compatibility)
            z_values = self.current_pointcloud[:, 2] + action
            z_values = np.clip(z_values, self.config.z_min, self.config.z_max)
            self.current_pointcloud[:, 2] = z_values
            # Update individual pointcloud as well
            if 0 in self.reflector_pointclouds:
                self.reflector_pointclouds[0][:, 2] = z_values


        # 1. Export updated pointcloud to STL/mesh format
        try:
            if self.config.num_reflectors > 1:
                # Multi-reflector: create separated STL with independent mesh components
                combined_pointcloud = self._create_combined_reflector_pointcloud_from_individual()
                
                # Create separated STL with independent mesh components (포인트클라우드 파일 저장 생략)
                self.logger.info(f"🔧 Creating separated STL for {self.config.num_reflectors} reflectors...")
                stl_success = self._create_separated_multi_reflector_stl_from_individual(self.config.stl_output_path)
                
                if stl_success:
                    self.logger.info(f"✅ Separated {self.config.num_reflectors} reflectors STL created")
                else:
                    self.logger.warning("❌ Separated STL creation failed, trying fallback method")
                    # Fallback: use original combined method
                    stl_success = pointcloud_to_stl(
                        combined_pointcloud,
                        self.config.stl_output_path,
                        poisson_depth=8
                    )
                    if stl_success:
                        self.logger.info(f"✅ Fallback combined STL created: {self.config.stl_output_path}")
            else:
                # Single reflector (backward compatibility)
                stl_success = pointcloud_to_stl(
                    self.current_pointcloud, 
                    self.config.stl_output_path,
                    poisson_depth=8  # 빠른 처리를 위해 깊이 조정
                )
            if not stl_success:
                self.logger.warning("STL 변환 실패, 시뮬레이션 계속 진행")
        except Exception as e:
            self.logger.error(f"STL 변환 중 오류: {e}")


        # 2. Call SPEOS simulation engine
        simulation_results = self._simulate_optical_raytracing()
        
        # Extract results
        intensity_map = simulation_results.get("intensity_map", np.zeros((self.config.grid_rows, self.config.grid_cols)))
        illuminance_map = simulation_results.get("illuminance_map", intensity_map.copy())
        efficiency = simulation_results.get("efficiency", 0.0)
        total_flux = simulation_results.get("total_flux", np.sum(intensity_map))
        
        # Calculate performance metrics
        uniformity_ratio = self._calculate_uniformity(intensity_map)
        target_match_score = self._calculate_target_match(intensity_map)
        
        # Prepare simulation result
        simulation_result = {
            "intensity_map": intensity_map,
            "illuminance_map": illuminance_map,
            "efficiency": efficiency,
            "total_flux": total_flux,
            "uniformity_ratio": uniformity_ratio,
            "target_match_score": target_match_score,
            "peak_intensity": np.max(intensity_map)
        }
        
        # Prepare metadata
        computation_time = time.time() - start_time
        metadata = {
            # Simulation settings
            "total_rays": self.config.ray_count,
            "wavelength_range": self.config.wavelength_range,
            "reflection_model": self.config.reflection_model,
            "material_properties": self.config.material_properties,
            
            # Computation info
            "computation_time": computation_time,
            "convergence_status": "converged",
            "simulation_quality": 0.95,
            
            # Technical info
            "mesh_quality": 0.9,
            "ray_intersection_count": self.config.ray_count,
            "simulation_engine": "SPEOS_v1.0",
            
            # Status
            "warnings": [],
            "errors": [],
            "status_code": 0
        }
        
        self._last_simulation_result = simulation_result
        
        # 🎯 실시간 메쉬 시각화 업데이트 (1번 리플렉터)
        if self.enable_mesh_visualization and self.visualization_manager and 0 in self.reflector_pointclouds:
            try:
                # 1번 리플렉터 포인트클라우드 가져오기 (위치 오프셋 적용)
                reflector_0_pointcloud = self.reflector_pointclouds[0].copy()
                reflector_pos = self.reflector_positions[0]
                
                # 실제 공간 좌표로 변환
                reflector_0_pointcloud[:, 0] += reflector_pos[0]  # X offset
                reflector_0_pointcloud[:, 1] += reflector_pos[1]  # Y offset
                reflector_0_pointcloud[:, 2] += reflector_pos[2]  # Z offset
                
                # 시각화 업데이트 (비동기)
                self.visualization_manager.update_mesh(reflector_0_pointcloud, self.current_step)
                
                self.logger.debug(f"🎯 Updated mesh visualization for step {self.current_step}")
                
            except Exception as e:
                self.logger.warning(f"Mesh visualization update failed: {e}")
        
        # 스텝 카운터 증가
        self.current_step += 1
        

        return simulation_result, metadata
    
    def _start_speos_simulation(self) -> float:
        """SPEOS 시뮬레이션 실행"""
        
        # 1. control_file_path의 txt 파일을 열어서 값을 1로 변경해서 스페오스에서 해석이 시작
        with open(self.config.control_file_path, 'w') as f:
            f.write("1")
        
        # Control 신호 시간 기록
        control_time = time.time()
        
        #self.logger.info("SPEOS 시뮬레이션 시작 신호 전송 완료")
        
        # 2. SPEOS 시뮬레이션 대기 (시간 최적화)
        time.sleep(6.0)  

        # 3. control_file_path의 txt 파일을 다시 열어서 값을 0으로 변경해서 해석 명령을 정지
        with open(self.config.control_file_path, 'w') as f:
            f.write("0")

        return control_time
    
    def _wait_for_xmp_update(self, xmp_file_path: str, control_time: float, timeout: int = 30) -> bool:
        """
        Control 신호 이후 마지막 XMP 파일이 생성되거나 업데이트되었는지 확인 (파일이 없었다가 새로 생기는 경우도 포함)
        Args:
            xmp_file_path: 마지막 XMP 파일 경로 (예: Direct.1.Intensity.100.xmp)
            control_time: Control 신호를 보낸 시간 (time.time())
            timeout: 타임아웃 (초)
        Returns:
            bool: 업데이트 감지 여부
        """
        import datetime
        
        # 🎯 상세한 디버깅 정보 로그
        #control_time_str = datetime.datetime.fromtimestamp(control_time).strftime('%H:%M:%S.%f')[:-3]
        #self.logger.info(f"🔍 XMP 파일 업데이트 감지 시작:")
        #self.logger.info(f"   - 대상 파일: {xmp_file_path}")
        #self.logger.info(f"   - Control 신호 시간: {control_time_str}")
        #self.logger.info(f"   - 타임아웃: {timeout}초")
        
        start_time = time.time()
        check_count = 0
        check_interval = 0.25 if timeout <= 5 else 0.5
        file_was_present = os.path.exists(xmp_file_path)
        file_mtime = os.path.getmtime(xmp_file_path) if file_was_present else None
        
        if file_was_present:
            initial_mtime_str = datetime.datetime.fromtimestamp(file_mtime).strftime('%H:%M:%S.%f')[:-3]
            #self.logger.info(f"   - 초기 파일 상태: 존재함 (수정시간: {initial_mtime_str})")
        else:
            #self.logger.info(f"   - 초기 파일 상태: 존재하지 않음")
            1

        while time.time() - start_time < timeout:
            try:
                exists_now = os.path.exists(xmp_file_path)
                if exists_now:
                    new_mtime = os.path.getmtime(xmp_file_path)
                    check_count += 1
                    
                    # 🎯 상세한 파일 상태 로그 (매 10번째 체크마다)
                    if check_count % 20 == 0:
                        new_mtime_str = datetime.datetime.fromtimestamp(new_mtime).strftime('%H:%M:%S.%f')[:-3]
                        elapsed = time.time() - start_time
                        time_diff = new_mtime - control_time
                        self.logger.debug(f"   체크 #{check_count}: 파일 수정시간={new_mtime_str}, Control대비={time_diff:.3f}초, 경과={elapsed:.1f}초")
                    
                    # 파일이 없었다가 새로 생긴 경우, 또는 control_time 이후에 생성/수정된 경우
                    update_detected = False
                    if not file_was_present and new_mtime > control_time:
                        update_detected = True
                        #self.logger.info(f"✅ 새 파일 생성 감지!")
                    elif file_was_present and new_mtime > control_time:
                        update_detected = True
                        #self.logger.info(f"✅ 기존 파일 업데이트 감지!")
                    
                    if update_detected:
                        elapsed = time.time() - start_time
                        new_mtime_str = datetime.datetime.fromtimestamp(new_mtime).strftime('%H:%M:%S.%f')[:-3]
                        #self.logger.info(f"✅ XMP 파일(마지막) 업데이트/생성 감지! {os.path.basename(xmp_file_path)}")
                        #self.logger.info(f"   - 파일 수정시간: {new_mtime_str}   - 감지 경과시간: {elapsed:.1f}초")
                        return True
                        
                time.sleep(check_interval)
            except OSError as e:
                self.logger.error(f"XMP 파일 확인 오류: {e}")
                return False
                
        # 타임아웃 발생 시 최종 상태 로그
        final_exists = os.path.exists(xmp_file_path)
        if final_exists:
            final_mtime = os.path.getmtime(xmp_file_path)
            final_mtime_str = datetime.datetime.fromtimestamp(final_mtime).strftime('%H:%M:%S.%f')[:-3]
            time_diff = final_mtime - control_time
            self.logger.warning(f"❌ XMP 파일 타임아웃! {os.path.basename(xmp_file_path)}")
            self.logger.warning(f"   - 최종 파일 수정시간: {final_mtime_str}")
            self.logger.warning(f"   - Control 대비 시간차: {time_diff:.3f}초")
            self.logger.warning(f"   - 총 체크 횟수: {check_count}")
        else:
            self.logger.warning(f"❌ XMP 파일이 생성되지 않음: {os.path.basename(xmp_file_path)} (타임아웃 {timeout}s)")
        return False
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute a single step in the environment (Gymnasium interface)"""
        return self._step(action)

    def _step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """새로운 ReflectorClass 아키텍처로 스텝 실행"""

        # 액션을 numpy 배열로 변환하고 차원 확인
        action = np.array(action, dtype=np.float32)
        if action.ndim == 0:  # scalar인 경우
            self.logger.error(f"❌ 액션이 scalar입니다: {action}")
            raise ValueError(f"액션이 scalar입니다. 배열이 필요합니다: {action}")
        
        if action.ndim > 1:  # 다차원 배열인 경우 평면화
            action = action.flatten()
        
        # 액션 차원 검증 (모든 리플렉터에 개별 액션 적용: 총 액션 크기 체크)
        num_reflectors = len(self.reflectors)
        action_size_per_reflector = self.config.grid_rows * self.config.grid_cols
        expected_total_actions = num_reflectors * action_size_per_reflector
        
        if len(action) != expected_total_actions:
            self.logger.error(f"❌ 액션 차원 불일치: 받은 값 {len(action)}, 예상 값 {expected_total_actions}")
            raise ValueError(f"액션 차원 불일치: 받은 값 {len(action)}, 예상 값 {expected_total_actions}")
        
        # 액션 범위 검증
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 스텝 카운터 업데이트
        self.current_step += 1
        self.episode_step += 1
        
        # 현재 스텝 로그
        elapsed_time = self._format_elapsed_time(self.training_start_time)
        #self.logger.info(f"🔄 스텝 {self.current_step} (에피소드 스텝 {self.episode_step}) 시작 - {len(self.reflectors)}개 리플렉터 [경과: {elapsed_time}]")
        
        # 🎯 1단계: 개별 리플렉터에 액션 분배 적용
        self.logger.debug(f"🎯 {num_reflectors}개 리플렉터에 액션 분배 적용 중...")
        all_observations = []  # 모든 리플렉터의 관찰값 수집용
        for i, reflector in enumerate(self.reflectors):
            # 이 리플렉터의 관찰값 가져오기
            reflector_observation = reflector._get_observation()
            all_observations.extend(reflector_observation)
            # 각 리플렉터에 해당하는 액션 슬라이스 적용
            start_idx = i * action_size_per_reflector
            end_idx = start_idx + action_size_per_reflector
            reflector_action = action[start_idx:end_idx]
            reflector._apply_actions(reflector_action)
        
        strat_time = time.time()

        # 🎯 2단계: 모든 리플렉터 메쉬 결합하여 STL 파일 저장 + 포인트클라우드 XYZ 저장
        self.logger.debug("🔧 모든 리플렉터 메쉬 결합 중...")
        # 2-1. 모든 리플렉터 포인트클라우드 결합 및 XYZ 저장
        try:
            all_pointclouds = []
            for reflector in self.reflectors:
                # 포인트클라우드가 None이 아니면 위치 오프셋 적용
                pc = reflector.pointcloud_s1
                if pc is not None:
                    # 리플렉터 중심 위치 오프셋 적용
                    pc_offset = pc.copy()
                    cx, cy, cz = reflector.center_position
                    pc_offset[:, 0] += cx
                    pc_offset[:, 1] += cy
                    pc_offset[:, 2] += cz
                    all_pointclouds.append(pc_offset)
            if all_pointclouds:
                combined_pc = np.vstack(all_pointclouds)
                # 저장 경로: mesh_save_path와 동일, 확장자만 .xyz
                mesh_save_path = getattr(self.config, 'mesh_save_path', self.config.stl_output_path)
                xyz_path = os.path.splitext(mesh_save_path)[0] + '.xyz'
                np.savetxt(xyz_path, combined_pc, fmt='%.6f', delimiter=' ', header='X Y Z', comments='')
                #self.logger.info(f"✅ 결합된 포인트클라우드 XYZ 저장 완료: {xyz_path}")
            else:
                self.logger.warning("❌ 결합할 포인트클라우드가 없습니다")
        except Exception as e:
            self.logger.error(f"❌ 포인트클라우드 XYZ 저장 실패: {e}")
        # 2-2. 메쉬 결합 및 STL 저장
        combined_mesh_success = self._combine_meshes()
        stl_time = time.time() - strat_time
        
        # 🎯 3단계: 시뮬레이션 실행 (실패 시 2단계 재시도)
        max_simulation_retries = 3  # 최대 재시도 횟수
        simulation_retry_count = 0
        simulation_success = False
        sim_time = 0.0  # 시뮬레이션 시간 기본값 (오류 방지)
        
        while simulation_retry_count < max_simulation_retries and not simulation_success:
            simulation_retry_count += 1
            strat_time = time.time()
            
            if simulation_retry_count > 1:
                self.logger.warning(f"🔄 시뮬레이션 재시도 {simulation_retry_count}/{max_simulation_retries}")
                
                # 2단계로 돌아가서 메쉬 재결합 및 STL 재생성
                self.logger.info("🔧 메쉬 재결합 및 STL 재생성 중...")
                retry_stl_start = time.time()
                retry_mesh_success = self._combine_meshes()
                retry_stl_time = time.time() - retry_stl_start
                
                if not retry_mesh_success:
                    self.logger.error(f"❌ 재시도 {simulation_retry_count}: 메쉬 결합 실패")
                    continue
                
                self.logger.info(f"✅ 재시도 {simulation_retry_count}: 메쉬 재결합 완료 ({retry_stl_time:.1f}s)")
                stl_time += retry_stl_time  # 총 STL 생성 시간에 추가
            
            #self.logger.debug(f"⚙️ 시뮬레이션 실행 중... (시도 {simulation_retry_count})")
            
            try:
                # SPEOS 시뮬레이션 시작
                control_time = self._start_speos_simulation()

                # 마지막 리플렉터의 XMP 파일 경로 생성
                last_reflector_xmp_path = self._get_reflector_xmp_path(self.config.num_reflectors)
                
                # config에서 타임아웃 값 가져오기 (재시도시 더 긴 타임아웃)
                base_timeout = getattr(self.config, 'xmp_update_timeout', 30)
                xmp_update_timeout = base_timeout + (simulation_retry_count - 1) * 10  # 재시도마다 10초씩 추가
                
                #self.logger.debug(f"SPEOS 시뮬레이션 대기 중... (XMP 타임아웃: {xmp_update_timeout}s, 시도 {simulation_retry_count})")
                xmp_updated = self._wait_for_xmp_update(last_reflector_xmp_path, control_time, timeout=xmp_update_timeout)
                
                if xmp_updated:
                    sim_time = time.time() - strat_time
                    self.logger.info(f"✅ 시뮬레이션 성공적으로 완료 (시도 {simulation_retry_count})")
                    simulation_success = True
                else:
                    self.logger.warning(f"⚠️ 시뮬레이션 시도 {simulation_retry_count}: XMP 업데이트 확인 실패")
                    if simulation_retry_count < max_simulation_retries:
                        self.logger.info(f"🔄 {max_simulation_retries - simulation_retry_count}번의 재시도 기회가 남았습니다")
                        # 잠시 대기 후 재시도
                        time.sleep(2.0)
                    
            except Exception as e:
                self.logger.error(f"❌ 시뮬레이션 시도 {simulation_retry_count} 실행 실패: {e}")
                if simulation_retry_count < max_simulation_retries:
                    self.logger.info(f"🔄 오류로 인한 재시도 준비 중...")
                    time.sleep(2.0)
        
        # 모든 재시도 실패 시 처리
        if not simulation_success:
            self.logger.error(f"❌ 시뮬레이션 최종 실패 ({max_simulation_retries}번 시도)")
            sim_time = 0.0
        else:
            self.logger.debug(f"✅ 시뮬레이션 최종 성공 (총 {simulation_retry_count}번 시도)")
        
        # 🎯 4단계: 시뮬레이션 결과 저장 (멀티스레드 XMP → TXT 변환 후 객체에 분배)
        self.logger.debug("📊 시뮬레이션 결과 처리 중...")
        
        # 🚀 멀티스레드로 모든 XMP 파일을 TXT로 일괄 변환
        strat_time = time.time()

        xmp_txt_pairs = []
        for i in range(self.config.num_reflectors):
            xmp_path = self._get_reflector_xmp_path(i + 1)  # 1부터 시작
            txt_path = self._get_reflector_txt_path(i + 1)
            xmp_txt_pairs.append((xmp_path, txt_path, i + 1))
        
        # 멀티스레드 변환 실행 (config에서 워커 개수 가져오기)
        max_workers = getattr(self.config, 'max_parallel_xmp_workers', 4)
        conversion_results = self._convert_xmp_to_txt_batch(xmp_txt_pairs, max_workers=max_workers)
        
        # 변환 결과를 시뮬레이션 결과로 변환
        all_simulation_results = []
        for i, intensity_map in enumerate(conversion_results):
            if intensity_map is not None:
                # Efficiency 계산 (intensity_map에서 파생)
                total_flux = np.sum(intensity_map)
                max_intensity = np.max(intensity_map)
                efficiency = min(total_flux if max_intensity > 0 else 0.0, 1.0)
                
                simulation_result = {
                    "intensity_map": intensity_map,
                    "efficiency": efficiency,
                    "total_flux": total_flux,
                    "reflector_id": i,
                    "success": True
                }
            else:
                simulation_result = {
                    "intensity_map": np.zeros((self.config.grid_rows, self.config.grid_cols)),
                    "efficiency": 0.0,
                    "total_flux": 0.0,
                    "reflector_id": i,
                    "success": False
                }
                self.logger.warning(f"리플렉터 {i+1} XMP 변환 실패 - 기본값 사용")
            all_simulation_results.append(simulation_result)
        
        xmp_time = time.time() - strat_time

        # 변환된 결과를 각 리플렉터 객체에 저장
        for i, reflector in enumerate(self.reflectors):
            if i < len(all_simulation_results):
                reflector._save_simulation_result(all_simulation_results[i])
        
        # 🎯 5단계: 모든 리플렉터 객체에서 리워드 계산
        self.logger.debug("🏆 모든 리플렉터의 리워드 계산 중...")
        
        total_reward = 0.0
        individual_rewards = []
        reward_metadata_list = []  # 리워드 메타데이터 리스트 추가
        
        for reflector in self.reflectors:
            reward = reflector._calculate_reward()
            total_reward += reward
            individual_rewards.append(reward)
            # 각 리플렉터의 리워드 메타데이터 수집
            if hasattr(reflector, 'reward_metadata'):
                reward_metadata_list.append(reflector.reward_metadata)
            else:
                # 기본 메타데이터 생성
                reward_metadata_list.append({
                    "distribution_factor": 0.0,
                    "efficiency_factor": 0.0,
                    "shape_factor": 0.0,
                    "size_penalty": 0.0
                })
        
        # 🎯 6단계: 모든 리플렉터 객체에서 경험 생성하여 버퍼에 저장
        self.logger.debug("📝 모든 리플렉터의 경험 데이터 생성 중...")
        
        experiences = []
        for reflector in self.reflectors:
            experience = reflector._get_experiences()
            if experience:
                experiences.append(experience)
        
        # 경험 버퍼에 저장
        if hasattr(self, '_experiences_buffer'):
            self._experiences_buffer.extend(experiences)
        else:
            self._experiences_buffer = experiences
        
        # 🎯 경험 버퍼가 설정된 크기에 도달하면 파일로 저장
        self._save_experience_buffer()
        
        # 🎯 버퍼 상태 출력
        buffer_size = len(self._experiences_buffer) if hasattr(self, '_experiences_buffer') else 0
        #self.logger.info(f"📊 경험 버퍼 상태: {len(experiences)}개 새로운 경험 추가, 현재 버퍼: {buffer_size}개, 파일: {self._buffer_file_index-1}개 저장됨")
        
        # 🎯 7단계: 각 리플렉터 종료 조건 확인 및 처리
        self.logger.debug("🏁 리플렉터별 종료 조건 확인 중...")
        
        completed_reflectors = 0
        active_reflectors = []
        
        for reflector in self.reflectors:
            is_terminated = reflector._check_termination()
            
            if is_terminated:
                # 해당 리플렉터 객체만 초기화
                reflector._initialize_Reflector()
                completed_reflectors += 1
                self.completed_episodes += 1
                #self.logger.info(f"🎯 리플렉터 {reflector.reflector_id + 1} 에피소드 완료! (총 완료: {self.completed_episodes})")
            else:
                active_reflectors.append(reflector)
        
        # 🎯 8단계: 에피소드 상태 업데이트
        avg_reward = total_reward / len(self.reflectors) if self.reflectors else 0.0
        self.episode_reward += avg_reward
        
        # 모든 리플렉터가 종료되거나 최대 스텝에 도달하면 에피소드 종료
        all_terminated = completed_reflectors == len(self.reflectors)
        truncated = self.current_step >= self.config.max_steps
        terminated = all_terminated
        
        # 스텝 결과 로그 (리워드 구성 요소 포함)
        elapsed_time = self._format_elapsed_time(self.training_start_time)
        
        # 평균 리워드 구성 요소 계산
        if reward_metadata_list:
            avg_distribution = sum(meta.get("distribution_factor", 0.0) for meta in reward_metadata_list) / len(reward_metadata_list)
            avg_efficiency = sum(meta.get("efficiency_factor", 0.0) for meta in reward_metadata_list) / len(reward_metadata_list)
            avg_shape = sum(meta.get("shape_factor", 0.0) for meta in reward_metadata_list) / len(reward_metadata_list)
            avg_size_penalty = sum(meta.get("size_penalty", 0.0) for meta in reward_metadata_list) / len(reward_metadata_list)

            #self.logger.info(f"📊 경험 버퍼 상태: {len(experiences)}개 새로운 경험 추가, 현재 버퍼: {buffer_size}개, 파일: {self._buffer_file_index-1}개 저장됨")
            self.logger.info(f"스텝 {self.current_step} / 리워드 {avg_reward:.4f} / 에피소드 {self.episode_reward:.4f} / 분포 {avg_distribution:.3f} / 효율 {avg_efficiency:.3f} / 형상 {avg_shape:.3f} / 크기 {avg_size_penalty:.3f} / STL {stl_time:.1f}s / sim {sim_time:.1f}s / XMP {xmp_time:.1f}s / 총 시간 {elapsed_time} / 버퍼: {buffer_size}개")
            #self.logger.info(f"   📊 구성요소 - 분배: {avg_distribution:.3f}, 효율: {avg_efficiency:.3f}, 형상: {avg_shape:.3f}, 크기페널티: {avg_size_penalty:.3f} ⏱️ 시간 - [총 {elapsed_time} / STL생성 {stl_time:.1f}s / 시뮬레이션 {sim_time:.1f}s / XMP변환 {xmp_time:.1f}s]")
            #self.logger.info(f"   ⏱️ 시간 - [총 {elapsed_time} / STL생성 {stl_time:.1f}s / 시뮬레이션 {sim_time:.1f}s / XMP변환 {xmp_time:.1f}s]")
        else:
            self.logger.info(f"✅ 스텝 {self.current_step} 완료 - 리워드: {avg_reward:.4f}, 에피소드 리워드: {self.episode_reward:.4f} [총 시간 {elapsed_time} / STL 생성 {stl_time:.1f}s / Simulation {sim_time:.1f}s / XMP 변환 {xmp_time:.1f}s]")


        # 시뮬레이션 히스토리 저장
        self.simulation_history.append({
            "step": self.current_step,
            "action": action.copy(),
            "reward": avg_reward,
            "individual_rewards": individual_rewards,
            "completed_reflectors": completed_reflectors,
            "active_reflectors": len(active_reflectors)
        })
        
        # 종료 상태 로그
        if terminated:
            self.logger.info(f"🏁 에피소드 종료 - 스텝 {self.current_step} (모든 리플렉터 완료)")
        elif truncated:
            self.logger.info(f"⏰ 에피소드 중단 - 스텝 {self.current_step} (최대 스텝 도달: {self.config.max_steps})")
        
        # 결합된 관찰값
        combined_observation = np.array(all_observations, dtype=np.float32)
        
        # 정보 딕셔너리 준비
        info = {
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "individual_rewards": individual_rewards,
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "experiences_generated": len(experiences),
            "completed_reflectors": completed_reflectors,
            "active_reflectors": len(active_reflectors),
            "simulation_success": combined_mesh_success
        }
        
        if terminated or truncated:
            elapsed_time = self._format_elapsed_time(self.training_start_time)
            self.logger.info(f"📊 에피소드 요약 - 총 스텝: {self.episode_step}, 총 리워드: {self.episode_reward:.4f} [경과: {elapsed_time}]")
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_step
            }
        
        # 🎯 실시간 STL 메쉬 시각화 (1번 리플렉터)
        if (self.enable_mesh_visualization and 
            self.mesh_visualizer is not None and 
            self.current_step % self.visualization_update_interval == 0):
            try:
                # STL 파일 경로 확인 (결합된 메쉬 파일)
                stl_path = getattr(self.config, 'mesh_save_path', self.config.stl_output_path)
                
                if os.path.exists(stl_path):
                    if not hasattr(self, '_vis_window') or self._vis_window is None:
                        # 첫 번째 시각화 윈도우 생성
                        self._vis_window = self.mesh_visualizer.visualize_stl(
                            stl_path, 
                            window_name=f"리플렉터 실시간 STL (Step {self.current_step})",
                            non_blocking=True
                        )
                        self.logger.info(f"🎯 STL 시각화 윈도우 생성됨 (Step {self.current_step})")
                    else:
                        # 기존 윈도우 업데이트
                        self._vis_window = self.mesh_visualizer.visualize_stl(
                            stl_path, 
                            vis=self._vis_window,
                            non_blocking=True
                        )
                        self.logger.debug(f"🎯 STL 시각화 업데이트됨 (Step {self.current_step})")
                else:
                    self.logger.warning(f"⚠️ STL 파일을 찾을 수 없음: {stl_path}")
                    
            except Exception as e:
                self.logger.error(f"❌ 실시간 STL 시각화 실패: {e}")
        
        # 선택적 시각화 (기존 코드 유지)
        if (self.config.enable_visualization and 
            self.current_step % self.config.visualize_interval == 0):
            self._visualize(all_simulation_results[0] if all_simulation_results else None)
        
        return combined_observation, avg_reward, terminated, truncated, info

    def _get_agent_action_for_reflector(self, reflector_id: int, reflector_obs: np.ndarray, base_action: np.ndarray) -> np.ndarray:
        """
        개별 리플렉터에 대한 에이전트 액션 생성
        
        Args:
            reflector_id: 리플렉터 ID
            reflector_obs: 해당 리플렉터의 관찰값
            base_action: 기본 액션 (에이전트가 제공한 액션)
        
        Returns:
            np.ndarray: 해당 리플렉터에 최적화된 액션
        """
        # TODO: 실제 구현에서는 외부 에이전트를 호출하여 해당 리플렉터의 관찰에 기반한 개별 액션 생성
        # 현재는 기본 액션을 해당 리플렉터 상태에 맞게 적응
        
        if reflector_id in self.reflector_pointclouds:
            current_z_values = self.reflector_pointclouds[reflector_id][:, 2]
            adapted_action = self._generate_state_adaptive_action(reflector_id, reflector_obs, base_action)
        else:
            adapted_action = base_action.copy()
        
        self.logger.debug(f"리플렉터 {reflector_id + 1} 액션 생성: base_range=[{np.min(base_action):.3f}, {np.max(base_action):.3f}] → adapted_range=[{np.min(adapted_action):.3f}, {np.max(adapted_action):.3f}]")
        return adapted_action
    
    def _combine_meshes(self) -> bool:
        """
        모든 리플렉터 객체의 메쉬를 결합하여 STL 파일로 저장
        
        Returns:
            bool: 메쉬 결합 및 저장 성공 여부
        """
        try:
            import open3d as o3d
            
            self.logger.debug(f"🔧 {len(self.reflectors)}개 리플렉터에서 메쉬 수집 중...")
            
            # 각 리플렉터 객체에서 메쉬 데이터 수집
            meshes = []
            for reflector in self.reflectors:
                reflector_mesh = reflector._get_mesh()
                if reflector_mesh is not None:
                    meshes.append(reflector_mesh)
                    
                    # 🎯 메쉬 위치 디버깅: 메쉬의 bounding box 중심 좌표 확인
                    vertices = np.asarray(reflector_mesh.vertices)
                    if len(vertices) > 0:
                        bbox_center = vertices.mean(axis=0)
                        bbox_min = vertices.min(axis=0)
                        bbox_max = vertices.max(axis=0)
                        expected_pos = reflector.center_position
                        
                        #self.logger.info(f"   리플렉터 {reflector.reflector_id + 1}: 메쉬 수집 완료")
                        #self.logger.info(f"     - 예상 중심 위치: ({expected_pos[0]:.1f}, {expected_pos[1]:.1f}, {expected_pos[2]:.1f})mm")
                        #self.logger.info(f"     - 실제 메쉬 중심: ({bbox_center[0]:.1f}, {bbox_center[1]:.1f}, {bbox_center[2]:.1f})mm")
                        #self.logger.info(f"     - 메쉬 범위: X[{bbox_min[0]:.1f}, {bbox_max[0]:.1f}], Y[{bbox_min[1]:.1f}, {bbox_max[1]:.1f}], Z[{bbox_min[2]:.1f}, {bbox_max[2]:.1f}]")
                    else:
                        self.logger.warning(f"   리플렉터 {reflector.reflector_id + 1}: 메쉬에 정점이 없습니다")
                else:
                    self.logger.warning(f"   리플렉터 {reflector.reflector_id + 1}: 메쉬 수집 실패")
            
            if len(meshes) == 0:
                self.logger.error("❌ 수집된 메쉬가 없습니다")
                return False
            
            #self.logger.info(f"✅ {len(meshes)}개 메쉬 수집 완료")
            
            # 메쉬들을 하나로 결합
            if len(meshes) == 1:
                combined_mesh = meshes[0]
            else:
                # 여러 메쉬를 결합
                all_vertices = []
                all_faces = []
                vertex_offset = 0
                
                for mesh in meshes:
                    vertices = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.triangles)
                    
                    if len(vertices) > 0 and len(faces) > 0:
                        # 정점 추가
                        all_vertices.append(vertices)
                        
                        # 면 인덱스 조정 후 추가
                        adjusted_faces = faces + vertex_offset
                        all_faces.append(adjusted_faces)
                        
                        vertex_offset += len(vertices)
                
                if len(all_vertices) == 0:
                    self.logger.error("❌ 유효한 정점이 없습니다")
                    return False
                
                # 결합된 메쉬 생성
                combined_vertices = np.vstack(all_vertices)
                combined_faces = np.vstack(all_faces)
                
                combined_mesh = o3d.geometry.TriangleMesh()
                combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
                combined_mesh.triangles = o3d.utility.Vector3iVector(combined_faces)
                
                # 법선 벡터 계산
                combined_mesh.compute_vertex_normals()
            
            # config.py의 mesh_save_path 경로에 STL 파일 저장
            mesh_save_path = getattr(self.config, 'mesh_save_path', self.config.stl_output_path)
            
            # STL 파일로 저장
            success = o3d.io.write_triangle_mesh(mesh_save_path, combined_mesh)
            
            if success:
                # 🎯 결합된 메쉬의 전체 범위 확인
                combined_vertices = np.asarray(combined_mesh.vertices)
                if len(combined_vertices) > 0:
                    overall_min = combined_vertices.min(axis=0)
                    overall_max = combined_vertices.max(axis=0)
                    overall_center = combined_vertices.mean(axis=0)
                    
                    #self.logger.info(f"✅ 결합된 메쉬 STL 저장 완료: {mesh_save_path}")
                    #self.logger.info(f"🎯 결합된 메쉬 전체 범위:")
                    #self.logger.info(f"     - 중심: ({overall_center[0]:.1f}, {overall_center[1]:.1f}, {overall_center[2]:.1f})mm")
                    #self.logger.info(f"     - 범위: X[{overall_min[0]:.1f}, {overall_max[0]:.1f}], Y[{overall_min[1]:.1f}, {overall_max[1]:.1f}], Z[{overall_min[2]:.1f}, {overall_max[2]:.1f}]")
                    #self.logger.info(f"     - 크기: X={overall_max[0]-overall_min[0]:.1f}mm, Y={overall_max[1]-overall_min[1]:.1f}mm, Z={overall_max[2]-overall_min[2]:.1f}mm")
                #else:
                    #self.logger.info(f"✅ 결합된 메쉬 STL 저장 완료: {mesh_save_path}")
                
                # 🎯 추가: 스텝 번호가 붙은 STL 파일을 mesh_record_path에 저장
                try:
                    mesh_record_path = getattr(self.config, 'mesh_record_path', None)
                    if mesh_record_path:
                        # 디렉토리 생성 (존재하지 않으면)
                        record_dir = os.path.dirname(mesh_record_path)
                        os.makedirs(record_dir, exist_ok=True)
                        
                        # 파일명에 스텝 번호 추가 (3자리 패딩)
                        base_name = os.path.splitext(os.path.basename(mesh_record_path))[0]
                        extension = os.path.splitext(mesh_record_path)[1]
                        step_filename = f"{base_name}{self.current_step:03d}{extension}"
                        step_filepath = os.path.join(record_dir, step_filename)
                        
                        # 스텝별 STL 파일 저장
                        step_success = o3d.io.write_triangle_mesh(step_filepath, combined_mesh)
                        if step_success:
                            self.logger.debug(f"✅ 스텝별 메쉬 저장 완료: {step_filename}")
                        else:
                            self.logger.warning(f"❌ 스텝별 메쉬 저장 실패: {step_filename}")
                    else:
                        self.logger.debug("mesh_record_path가 설정되지 않아 스텝별 저장을 건너뜁니다")
                except Exception as step_save_error:
                    self.logger.warning(f"❌ 스텝별 메쉬 저장 중 오류: {step_save_error}")
                
                return True
            else:
                self.logger.error(f"❌ STL 파일 저장 실패: {mesh_save_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ 메쉬 결합 실패: {e}")
            return False
    
    def _convert_xmp_to_txt(self, xmp_path: str, txt_path: str, reflector_num: int, flip_updown: bool = False, flip_leftright: bool = False) -> Optional[np.ndarray]:
        """단일 XMP 파일을 TXT로 변환 (멀티스레드에서 사용)"""
        import win32com.client
        import pythoncom
        import datetime
        
        try:
            # 🎯 멀티스레드 환경에서 COM 초기화 (필수)
            pythoncom.CoInitialize()
            
            # 절대 경로로 변환
            xmp_path = os.path.abspath(xmp_path)
            txt_path = os.path.abspath(txt_path)
            tmp_raw_path = txt_path + ".raw"

            # XMP Viewer COM 객체 생성
            VPL = win32com.client.Dispatch("XmpViewer.Application")

            # XMP 파일 열기
            result = VPL.OpenFile(xmp_path)
            if result != 1:
                self.logger.error(f"XMP 파일 열기 실패: {os.path.basename(xmp_path)}")
                return None

            export_result = VPL.ExportTXT(tmp_raw_path)
            if export_result == 0:
                self.logger.error(f"TXT 내보내기 실패: {os.path.basename(xmp_path)}")
                return None

            # ▼ "x y value" 이후 데이터만 읽기
            with open(tmp_raw_path, "r", encoding="utf-8") as fin:
                lines = fin.readlines()

            start_idx = None
            for i, line in enumerate(lines):
                if "x" in line.lower() and "y" in line.lower() and "value" in line.lower():
                    start_idx = i + 1
                    break
            if start_idx is None:
                self.logger.error(f"'x y value' 구간을 찾을 수 없습니다: {os.path.basename(xmp_path)}")
                return None

            # ▼ 값 파싱
            data = []
            x_set = set()
            y_set = set()
            for line in lines[start_idx:]:
                parts = line.strip().split()
                if len(parts) == 3:
                    x, y, v = map(float, parts)
                    data.append((x, y, v))
                    x_set.add(x)
                    y_set.add(y)
            
            x_list = sorted(list(x_set))
            y_list = sorted(list(y_set))
            x_index = {x: i for i, x in enumerate(x_list)}
            y_index = {y: i for i, y in enumerate(y_list)}
            value_map = np.zeros((len(y_list), len(x_list)))
            for x, y, v in data:
                i = y_index[y]  # 행
                j = x_index[x]  # 열
                value_map[i, j] = v
            
            # flip 적용
            if flip_updown:
                value_map = np.flipud(value_map)
            if flip_leftright:
                value_map = np.fliplr(value_map)
            
            # 결과 저장
            np.savetxt(txt_path, value_map, fmt="%.6f", delimiter="\t")
            
            # 임시 파일 삭제
            os.remove(tmp_raw_path)
            
            self.logger.debug(f"✅ XMP→TXT 변환 완료: reflector{reflector_num}")
            return value_map
            
        except Exception as e:
            self.logger.error(f"❌ XMP→TXT 변환 실패 (reflector{reflector_num}): {e}")
            return None
        finally:
            # 🎯 COM 해제 (필수)
            try:
                pythoncom.CoUninitialize()
            except:
                pass
    
    def _convert_xmp_to_txt_batch(self, xmp_txt_pairs: List[Tuple[str, str, int]], max_workers: int = 4) -> List[Optional[np.ndarray]]:
        """멀티스레드로 여러 XMP 파일을 TXT로 일괄 변환"""
        import concurrent.futures
        import threading
        
        #self.logger.info(f"🔄 멀티스레드 XMP→TXT 변환 시작: {len(xmp_txt_pairs)}개 파일, {max_workers}개 워커")
        
        # 스레드 로컬 저장소 사용 (COM 객체는 스레드별로 독립적이어야 함)
        local_data = threading.local()
        
        def convert_single_file(args):
            """단일 파일 변환 (스레드 워커 함수)"""
            xmp_path, txt_path, reflector_num = args
            
            try:
                return self._convert_xmp_to_txt(xmp_path, txt_path, reflector_num, 
                                             flip_updown=self.config.flip_updown,
                                             flip_leftright=self.config.flip_leftright)
            except Exception as e:
                self.logger.error(f"❌ 스레드 워커에서 변환 실패 (reflector{reflector_num}): {e}")
                return None
        
        # 멀티스레드 실행
        results = [None] * len(xmp_txt_pairs)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 작업 제출
            future_to_index = {
                executor.submit(convert_single_file, pair): i 
                for i, pair in enumerate(xmp_txt_pairs)
            }
            
            # 결과 수집
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    completed_count += 1
                    if completed_count % 20 == 0 or completed_count == len(xmp_txt_pairs):
                        self.logger.debug(f"   진행률: {completed_count}/{len(xmp_txt_pairs)} 완료")
                except Exception as e:
                    self.logger.error(f"❌ 멀티스레드 결과 수집 실패 (index {index}): {e}")
                    results[index] = None
        
        success_count = sum(1 for r in results if r is not None)
        #self.logger.info(f"✅ 멀티스레드 XMP→TXT 변환 완료: {success_count}/{len(xmp_txt_pairs)} 성공")
        
        return results
        
    def _get_reflector_xmp_path(self, reflector_id: int) -> str:
        """리플렉터별 XMP 파일 경로 생성"""
        base_path = self.config.xmp_file_path
        # Direct.1.Intensity.1.xmp → Direct.1.Intensity.{reflector_id}.xmp
        return base_path.replace('.1.xmp', f'.{reflector_id}.xmp')
    
    def _get_reflector_txt_path(self, reflector_id: int) -> str:
        """리플렉터별 TXT 파일 경로 생성 (simulation_result 폴더 사용)"""
        base_path = self.config.txt_output_path
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # data/simulation_result 폴더 생성 (존재하지 않으면)
        os.makedirs(base_dir, exist_ok=True)
        
        # data/simulation_result 폴더에 저장
        return os.path.join(base_dir, f"{name_without_ext}_reflector{reflector_id}.txt")
    
    def _format_elapsed_time(self, start_time: float) -> str:
        """학습 시작부터 경과된 시간을 읽기 쉬운 형태로 포맷"""
        elapsed_seconds = time.time() - start_time
        
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _visualize(self, simulation_result: Optional[Dict] = None):
        """
        시뮬레이션 결과를 시각화합니다.
        
        Args:
            simulation_result: 시뮬레이션 결과 딕셔너리
        """
        try:
            if simulation_result is None:
                self.logger.debug("시뮬레이션 결과가 없어 시각화를 건너뜁니다.")
                return
            
            # 강도 맵 시각화 (matplotlib 사용)
            intensity_map = simulation_result.get("intensity_map")
            if intensity_map is not None:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 6))
                    plt.imshow(intensity_map, cmap='hot', interpolation='nearest')
                    plt.colorbar(label='Intensity')
                    plt.title(f'Intensity Map - Step {self.current_step}')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.show(block=False)  # 비블로킹으로 표시
                    plt.pause(0.1)  # 짧은 pause로 화면 업데이트
                    self.logger.debug(f"🎯 강도 맵 시각화 업데이트됨 (Step {self.current_step})")
                except ImportError:
                    self.logger.warning("matplotlib를 사용할 수 없어 강도 맵 시각화를 건너뜁니다.")
                except Exception as viz_e:
                    self.logger.error(f"강도 맵 시각화 실패: {viz_e}")
            
        except Exception as e:
            self.logger.error(f"시각화 실패: {e}")
    
    def close(self):
        """Clean up resources"""
        # 🎯 남은 경험 버퍼 저장
        if hasattr(self, '_experiences_buffer') and len(self._experiences_buffer) > 0:
            try:
                # 파일 경로 생성
                os.makedirs("data/experience_buffer", exist_ok=True)
                
                try:
                    import h5py
                    file_path = f"data/experience_buffer/experience_buffer_{self._model_name}_{self._buffer_file_index}.h5"
                    
                    with h5py.File(file_path, 'w') as f:
                        # 경험 데이터를 HDF5 형식으로 저장
                        observations = []
                        actions = []
                        rewards = []
                        reflector_ids = []
                        step_numbers = []
                        
                        for exp in self._experiences_buffer:
                            observations.append(exp['observation'])
                            actions.append(exp['action'])
                            rewards.append(exp['reward'])
                            reflector_ids.append(exp['reflector_id'])
                            step_numbers.append(exp['step_number'])
                        
                        # NumPy 배열로 변환하여 저장
                        observations = np.array(observations)
                        actions = np.array(actions)
                        rewards = np.array(rewards)
                        reflector_ids = np.array(reflector_ids)
                        step_numbers = np.array(step_numbers)
                        
                        f.create_dataset('observations', data=observations)
                        f.create_dataset('actions', data=actions)
                        f.create_dataset('rewards', data=rewards)
                        f.create_dataset('reflector_ids', data=reflector_ids)
                        f.create_dataset('step_numbers', data=step_numbers)
                        f.create_dataset('buffer_index', data=self._buffer_file_index)
                        f.create_dataset('total_experiences', data=len(self._experiences_buffer))
                        f.create_dataset('is_final_buffer', data=True)  # 마지막 버퍼임을 표시
                    
                    self.logger.info(f"📁 최종 경험 버퍼 HDF5 파일 저장 완료: {file_path}")
                    
                except ImportError:
                    # h5py가 없으면 pickle로 저장
                    import pickle
                    file_path = f"data/experience_buffer/experience_buffer_{self._model_name}_{self._buffer_file_index}.pkl"
                    with open(file_path, 'wb') as f:
                        pickle.dump(self._experiences_buffer, f)
                    self.logger.info(f"📁 최종 경험 버퍼 pickle 파일 저장 완료: {file_path}")
                
                self._total_experiences_saved += len(self._experiences_buffer)
                self.logger.info(f"   - 총 저장된 경험: {self._total_experiences_saved}, 총 파일 수: {self._buffer_file_index}")
                
                # 버퍼 초기화
                self._experiences_buffer.clear()
                
            except Exception as e:
                self.logger.error(f"❌ 최종 경험 버퍼 저장 실패: {e}")
        
        # STL 시각화 윈도우 정리
        if hasattr(self, '_vis_window') and self._vis_window is not None:
            try:
                self._vis_window.destroy_window()
                self.logger.info("STL 시각화 윈도우 정리됨")
            except:
                pass
        
        self.logger.info("Closing environment and cleaning up resources")

# =============================================================================
# 리플렉터 클래스
# =============================================================================

class ReflectorClass():
    """
    개별 리플렉터 상태와 액션을 관리하는 리플렉터 클래스.
    이 클래스는 개별 리플렉터의 상태와 액션 로직을 캡슐화합니다.
    """
    
    def __init__(self, reflector_id: int, config):
        # 기본 정보
        self.reflector_id = reflector_id
        self.config = config
        self.step_number = 0
        self.terminated = False
        
        # 로거 설정
        self.logger = logging.getLogger(f"Reflector_{reflector_id}")
        
        # 리플렉터 물리적 설정
        self.center_position = config.get_reflector_position(reflector_id)
        self.grid_size = config.grid_rows * config.grid_cols
        self.cell_size = (config.grid_cell_size_x, config.grid_cell_size_y)
        
        # Point cloud: 현재(s1)와 이전(s0) 상태 2세트 저장
        self.pointcloud_s0 = None  # 이전 상태 포인트클라우드
        self.pointcloud_s1 = None  # 현재 상태 포인트클라우드
        
        # Target
        self.target = None
        
        # Simulation result: 현재(s1)와 이전(s0) 상태 2세트 저장
        self.simulation_result_s0 = None  # 이전 상태 시뮬레이션 결과
        self.simulation_result_s1 = None  # 현재 상태 시뮬레이션 결과
        
        # Action: 현재(s1)와 이전(s0) 상태 2세트 저장
        self.action_s0 = np.zeros(self.grid_size, dtype=np.float32)  # 이전 액션
        self.action_s1 = np.zeros(self.grid_size, dtype=np.float32)  # 현재 액션
        
        # Reward: 현재(s1)와 이전(s0) 상태 2세트 저장
        self.reward_s0 = 0.0  # 이전 리워드
        self.reward_s1 = 0.0  # 현재 리워드
        self.reward_metadata = {  # 리워드 구성 요소 메타데이터
            "distribution_factor": 0.0,
            "efficiency_factor": 0.0,
            "shape_factor": 0.0,
            "size_penalty": 0.0
        }
        
        # Observation: SB3에서 다룰 수 있도록 numpy array로 정규화
        self.observation = None
    
    def _update_state(self):
        """새로운 데이터가 저장될 때 s1 데이터를 s0으로 이동"""
        self.pointcloud_s0 = self.pointcloud_s1.copy() if self.pointcloud_s1 is not None else None
        self.simulation_result_s0 = self.simulation_result_s1.copy() if self.simulation_result_s1 is not None else None
        self.action_s0 = self.action_s1.copy()
        self.reward_s0 = self.reward_s1
    
    def _initialize_Reflector(self):
        """리플렉터 중심 좌표, 그리드 크기, 셀 크기를 제외하고 모두 초기화"""
        self.step_number = 0
        self.terminated = False
        
        # 포인트클라우드 초기화
        self.pointcloud_s0 = None
        self.pointcloud_s1 = self._generate_default_pointcloud()
        
        # 타겟 생성
        self.target = self._generate_target()
        
        # 시뮬레이션 결과 초기화
        self.simulation_result_s0 = None
        self.simulation_result_s1 = None
        
        # 액션 초기화
        self.action_s0.fill(0.0)
        self.action_s1.fill(0.0)
        
        # 리워드 초기화
        self.reward_s0 = 0.0
        self.reward_s1 = 0.0
        
        # 관찰 초기화
        self.observation = None
    
    def _get_setting(self) -> Dict:
        """리플렉터 중심 좌표, 그리드 크기, 셀 크기 반환"""
        return {
            'center_position': self.center_position,
            'grid_size': (self.config.grid_rows, self.config.grid_cols),
            'cell_size': self.cell_size
        }
    
    def _set_reflector(self, center_position: Tuple[float, float, float], 
                      grid_rows: int, grid_cols: int, 
                      cell_size: Tuple[float, float]):
        """리플렉터 중심 좌표, 그리드 크기, 셀 크기 설정"""
        self.center_position = center_position
        self.config.grid_rows = grid_rows
        self.config.grid_cols = grid_cols
        self.grid_size = grid_rows * grid_cols
        self.cell_size = cell_size
    
    def reset(self):
        """Reset the reflector state"""
        self._initialize_Reflector()

    def _generate_default_pointcloud(self) -> np.ndarray:
        """Generate a default point cloud for testing using config-defined grid spacing"""
        # 🎯 SpeosConfig에서 grid cell size 정보 사용 (우선순위 1)
        if hasattr(self.config, 'grid_cell_size_x') and hasattr(self.config, 'grid_cell_size_y'):
            # SpeosConfig의 grid cell size를 사용   
            spacing_x = self.config.grid_cell_size_x
            spacing_y = self.config.grid_cell_size_y
            center_x = getattr(self.config, 'grid_origin_x', 0.0)  # 중심점 좌표
            center_y = getattr(self.config, 'grid_origin_y', 0.0)  # 중심점 좌표
            origin_z = getattr(self.config, 'grid_origin_z', 0.0)
            
            # 🎯 grid_origin을 중심으로 그리드 생성
            # 전체 그리드 범위 계산
            total_width_x = (self.config.grid_cols - 1) * spacing_x
            total_width_y = (self.config.grid_rows - 1) * spacing_y
            
            # 중심점 기준으로 시작점과 끝점 계산
            start_x = center_x - total_width_x / 2
            end_x = center_x + total_width_x / 2
            start_y = center_y - total_width_y / 2
            end_y = center_y + total_width_y / 2
            
            # 물리적 좌표 계산 (중심점 기준)
            x_coords = np.linspace(start_x, end_x, self.config.grid_cols)
            y_coords = np.linspace(start_y, end_y, self.config.grid_rows)
            
        elif hasattr(self.config, 'get_grid_x_coords'):
            # SpeosTrainingConfig 사용시 (우선순위 2)
            x_coords = self.config.get_grid_x_coords()
            y_coords = self.config.get_grid_y_coords()
            origin_z = getattr(self.config, 'grid_origin_z', 0.0)
            
        else:
            # 기존 방식 (하위 호환성, 우선순위 3) - 원점 중심
            x_coords = np.linspace(-5, 5, self.config.grid_cols)
            y_coords = np.linspace(-5, 5, self.config.grid_rows)
            origin_z = 0.0
        
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = np.full_like(X, origin_z)
        
        pointcloud = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        
        # 로그로 그리드 정보 출력
        cell_size_x = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0
        cell_size_y = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0
        center_x = (x_coords[0] + x_coords[-1]) / 2 if len(x_coords) > 0 else 0
        center_y = (y_coords[0] + y_coords[-1]) / 2 if len(y_coords) > 0 else 0
        
        #self.logger.info(f"포인트클라우드 생성: {pointcloud.shape}")
        #self.logger.info(f"  - 그리드 크기: {self.config.grid_rows}×{self.config.grid_cols}")
        #self.logger.info(f"  - 그리드 1칸: {cell_size_x:.3f}×{cell_size_y:.3f}mm")
        #self.logger.info(f"  - 중심점: ({center_x:.3f}, {center_y:.3f}, {origin_z:.3f})mm")
        #self.logger.info(f"  - X 범위: [{x_coords[0]:.3f}, {x_coords[-1]:.3f}]mm")
        #self.logger.info(f"  - Y 범위: [{y_coords[0]:.3f}, {y_coords[-1]:.3f}]mm")
        
        return pointcloud.astype(np.float32)
    
    def _generate_target(self) -> np.ndarray:
        """Generate a randomized target intensity map"""
        import random
        
        target = np.zeros((self.config.grid_rows, self.config.grid_cols))
        center_r, center_c = self.config.grid_rows // 2, self.config.grid_cols // 2
        
        # Randomly choose pattern type
        pattern_type = random.choice(['center_spot'])
        
        if pattern_type == 'circular':
            # Circular pattern with random intensity and radius
            max_radius = random.uniform(2, min(self.config.grid_rows, self.config.grid_cols) // 2)
            intensity_scale = random.uniform(0.5, 1.0)
            
            for r in range(self.config.grid_rows):
                for c in range(self.config.grid_cols):
                    dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                    if dist <= max_radius:
                        target[r, c] = intensity_scale * np.exp(-0.1 * dist**2)
        
        elif pattern_type == 'square':
            # Square pattern with random size
            max_half_size = max(1, min(self.config.grid_rows, self.config.grid_cols) // 3)
            min_half_size = min(1, max_half_size)
            half_size = random.randint(min_half_size, max(min_half_size, max_half_size))
            intensity = random.uniform(0.5, 1.0)
            
            r_start = max(0, center_r - half_size)
            r_end = min(self.config.grid_rows, center_r + half_size + 1)
            c_start = max(0, center_c - half_size)
            c_end = min(self.config.grid_cols, center_c + half_size + 1)
            
            target[r_start:r_end, c_start:c_end] = intensity
        
        elif pattern_type == 'cross':
            # Cross pattern
            max_thickness = max(1, min(self.config.grid_rows, self.config.grid_cols) // 5)
            thickness = random.randint(1, max(1, max_thickness))
            intensity = random.uniform(0.5, 1.0)
            
            # Vertical line
            c_start = max(0, center_c - thickness)
            c_end = min(self.config.grid_cols, center_c + thickness + 1)
            target[:, c_start:c_end] = intensity
            
            # Horizontal line
            r_start = max(0, center_r - thickness)
            r_end = min(self.config.grid_rows, center_r + thickness + 1)
            target[r_start:r_end, :] = intensity
        
        elif pattern_type == 'random_spots':
            # Random bright spots
            max_spots = max(2, min(8, self.config.grid_rows * self.config.grid_cols // 4))
            num_spots = random.randint(2, max_spots)
            for _ in range(num_spots):
                spot_r = random.randint(0, self.config.grid_rows - 1)
                spot_c = random.randint(0, self.config.grid_cols - 1)
                spot_intensity = random.uniform(0.3, 1.0)
                max_spot_size = max(1, min(self.config.grid_rows, self.config.grid_cols) // 4)
                spot_size = random.randint(0, max_spot_size)
                
                r_start = max(0, spot_r - spot_size)
                r_end = min(self.config.grid_rows, spot_r + spot_size + 1)
                c_start = max(0, spot_c - spot_size)
                c_end = min(self.config.grid_cols, spot_c + spot_size + 1)
                
                target[r_start:r_end, c_start:c_end] = spot_intensity
        
        elif pattern_type == 'center_spot':
            # Center spot pattern - 100% concentrated intensity only at the center pixel
            intensity = 1.0  # Maximum intensity (100%)
            
            # Set only center pixel to maximum intensity, all others remain 0
            target[center_r, center_c] = intensity
        
        # Add some random noise for variety
        #noise_level = random.uniform(0.0, 0.1)
        #noise = np.random.normal(0, noise_level, target.shape)
        #target = np.clip(target + noise, 0, 1)
        
        # Target을 data/target 폴더에 txt 파일로 저장
        self._save_target_as_txt(target, pattern_type)
        
        #self.logger.info(f"Generated {pattern_type} target pattern")
        return target.astype(np.float32)
    
    def _save_pointcloud_as_xyz(self, pointcloud: np.ndarray, filename: str):
        """포인트클라우드를 XYZ 파일로 저장 - 리플렉터 ID 포함"""
        try:
            # cad 폴더 경로 확인 및 생성
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cad_folder = os.path.join(project_root, "cad")
            
            if not os.path.exists(cad_folder):
                os.makedirs(cad_folder)
            
            # 파일명에 리플렉터 ID 추가
            name_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            filename_with_id = f"{name_without_ext}_reflector{self.reflector_id + 1}{ext}"
            
            # 전체 파일 경로
            file_path = os.path.join(cad_folder, filename_with_id)
            
            # XYZ 형식으로 저장 (X Y Z 좌표)
            np.savetxt(file_path, pointcloud, fmt='%.6f', delimiter=' ', 
                      header='X Y Z', comments='')
            
            self.logger.info(f"✅ 리플렉터 {self.reflector_id + 1} 포인트클라우드 XYZ 파일 저장: {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 리플렉터 {self.reflector_id + 1} 포인트클라우드 XYZ 저장 실패: {e}")

    def _save_target_as_txt(self, target: np.ndarray, pattern_type: str):
        """Target을 data/target 폴더에 txt 파일로 저장"""
        try:
            # data/target 폴더 경로 확인 및 생성
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            target_folder = os.path.join(project_root, "data", "target")
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            # 파일명에 리플렉터 ID와 패턴 타입 추가
            filename = f"target_reflector{self.reflector_id + 1}_{pattern_type}.txt"
            file_path = os.path.join(target_folder, filename)
            
            # TXT 형식으로 저장 (탭으로 구분)
            np.savetxt(file_path, target, fmt='%.6f', delimiter='\t')
            
            #self.logger.info(f"✅ 리플렉터 {self.reflector_id + 1} target 파일 저장: {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 리플렉터 {self.reflector_id + 1} target 저장 실패: {e}")

    def _apply_actions(self, action: np.ndarray):
        """액션을 받아서 포인트 클라우드에 적용"""
        
        # 액션 크기 검증
        expected_size = self.config.grid_rows * self.config.grid_cols
        if action.shape[0] != expected_size:
            raise ValueError(f"Action size mismatch: expected {expected_size}, got {action.shape[0]}")
        
        # 상태 업데이트 (s1 → s0)
        self._update_state()
        
        # 새로운 액션 저장
        self.action_s1 = action.copy()
        
        # 포인트 클라우드가 없는 경우에만 초기화 (한 번만)
        if self.pointcloud_s1 is None:
            self.pointcloud_s1 = self._generate_default_pointcloud()
        
        # 액션을 포인트 클라우드의 Z값에 적용 (누적 업데이트)
        current_z = self.pointcloud_s1[:, 2].copy()
        new_z = current_z + action
        
        # Z값을 설정된 범위로 제한
        new_z = np.clip(new_z, self.config.z_min, self.config.z_max)
        
        # 포인트 클라우드 업데이트 (이전 상태에서 계속 누적)
        self.pointcloud_s1[:, 2] = new_z
        
        self.step_number += 1

    def _save_simulation_result(self, simulation_result: Dict):
        """변환된 결과 데이터 객체에 저장"""
        self.simulation_result_s1 = simulation_result
    
    def _calculate_reward(self) -> float:
        """리워드 계산 - calculate_reward.py 파일 호출"""
        if self.simulation_result_s1 is None:
            self.reward_s1 = 0.0
            return self.reward_s1
            
        try:
            from .calculate_reward import calculate_speos_reward
            
            # metadata 딕셔너리 생성
            metadata = {
                "computation_time": 0.0,
                "status_code": 0,
                "warnings": [],
                "errors": []
            }
            
            reward_value, reward_metadata = calculate_speos_reward(
                simulation_result=self.simulation_result_s1,
                metadata=metadata,
                current_pointcloud=self.pointcloud_s1,  # 올바른 매개변수명 사용
                config=self.config
            )
            
            self.reward_s1 = reward_value
            self.reward_metadata = reward_metadata  # 메타데이터 저장
            return self.reward_s1
            
        except Exception as e:
            self.logger.error(f"Reward calculation failed for reflector {self.reflector_id}: {e}")
            self.reward_s1 = 0.0
            self.reward_metadata = {  # 기본 메타데이터 설정
                "distribution_factor": 0.0,
                "efficiency_factor": 0.0,
                "shape_factor": 0.0,
                "size_penalty": 0.0,
                "error": str(e)
            }
            return self.reward_s1
            return self.reward_s1
    
    def _get_observation(self) -> np.ndarray:
        """현재 리플렉터 객체 observation 반환"""
        try:
            # 포인트클라우드 Z값
            pointcloud_z = self.pointcloud_s1[:, 2] if self.pointcloud_s1 is not None else np.zeros(self.grid_size)
            
            # 시뮬레이션 결과
            if self.simulation_result_s1 is not None:
                intensity_map = self.simulation_result_s1.get('intensity_map', np.zeros((self.config.grid_rows, self.config.grid_cols)))
                efficiency = self.simulation_result_s1.get('efficiency', 0.0)
                total_flux = self.simulation_result_s1.get('total_flux', 0.0)
            else:
                intensity_map = np.zeros((self.config.grid_rows, self.config.grid_cols))
                efficiency = 0.0
                total_flux = 0.0
            
            # 관찰 벡터 구성
            observation = np.concatenate([
                pointcloud_z.flatten(),  # 포인트클라우드 Z값
                intensity_map.flatten(),  # intensity map
                [efficiency, total_flux]  # 스칼라 값들
            ])
            
            self.observation = observation.astype(np.float32)
            return self.observation
            
        except Exception as e:
            print(f"Observation generation failed for reflector {self.reflector_id}: {e}")
            # 기본 관찰값 반환
            default_size = self.grid_size + self.grid_size + 2
            self.observation = np.zeros(default_size, dtype=np.float32)
            return self.observation
    
    def _get_experiences(self) -> Dict:
        """현재 리플렉터의 경험 반환 (observation, action, reward, terminated) - 포인트클라우드 중심을 원점으로 이동"""
        try:
            observation = self._get_observation()
            
            # 포인트클라우드 중심을 원점으로 이동하여 학습 일관성 확보
            normalized_pointcloud = None
            center_x = 0.0
            center_y = 0.0
            
            if self.pointcloud_s1 is not None:
                # 원본 복사 (원본 데이터는 변경하지 않음)
                normalized_pointcloud = self.pointcloud_s1.copy()
                
                # 중심점 계산 (X, Y만 - Z는 그대로 유지)
                center_x = np.mean(normalized_pointcloud[:, 0])
                center_y = np.mean(normalized_pointcloud[:, 1])
                
                # 복사본의 중심을 원점으로 이동 (원본은 그대로 유지)
                normalized_pointcloud[:, 0] -= center_x
                normalized_pointcloud[:, 1] -= center_y
                
                self.logger.debug(f"리플렉터 {self.reflector_id + 1} 포인트클라우드 중심 이동: ({center_x:.3f}, {center_y:.3f}) → (0, 0) [버퍼용 복사본만]")
            
            return {
                'observation': observation,
                'action': self.action_s1.copy(),
                'reward': self.reward_s1,
                'terminated': self.terminated,
                'reflector_id': self.reflector_id,
                'step_number': self.step_number,
                'normalized_pointcloud': normalized_pointcloud,  # 정규화된 포인트클라우드 (버퍼 저장용)
                'original_center': (center_x, center_y)  # 원래 중심점 정보
            }
        except Exception as e:
            self.logger.error(f"경험 데이터 생성 실패: {e}")
            return {
                'observation': np.zeros(1),
                'action': self.action_s1.copy(),
                'reward': 0.0,
                'terminated': True,
                'reflector_id': self.reflector_id,
                'step_number': self.step_number,
                'error': str(e)
            }

    def _get_mesh(self):
        """현재 포인트클라우드에서 Open3D 메쉬 객체 생성 및 반환 (리플렉터 위치 오프셋 적용)"""
        try:
            import open3d as o3d
            
            if self.pointcloud_s1 is None:
                self.logger.warning(f"리플렉터 {self.reflector_id + 1}: 포인트클라우드가 없습니다")
                return None
            
            # 🎯 포인트클라우드에 리플렉터 중심 위치 오프셋 적용
            positioned_pointcloud = self.pointcloud_s1.copy()
            cx, cy, cz = self.center_position
            positioned_pointcloud[:, 0] += cx  # X offset
            positioned_pointcloud[:, 1] += cy  # Y offset
            positioned_pointcloud[:, 2] += cz  # Z offset
            
            self.logger.debug(f"리플렉터 {self.reflector_id + 1} 메쉬 생성: 중심위치 ({cx:.1f}, {cy:.1f}, {cz:.1f})mm 적용")
            
            # 포인트클라우드를 Open3D PointCloud 객체로 변환
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positioned_pointcloud)
            
            # 법선 벡터 추정
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Poisson 표면 재구성을 사용하여 메쉬 생성
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=3, width=0, scale=1.1, linear_fit=True
            )
            
            if len(mesh.vertices) == 0:
                self.logger.warning(f"리플렉터 {self.reflector_id + 1}: 메쉬 생성 실패")
                return None
            
            # 메쉬 정리 및 법선 벡터 계산
            mesh.compute_vertex_normals()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"리플렉터 {self.reflector_id + 1} 메쉬 생성 실패: {e}")
            return None
    
    @staticmethod
    def _pointcloud_to_stl(pointcloud, stl_output_path: str, 
                      ply_output_path: Optional[str] = None,
                      freecad_cmd_path: Optional[str] = None,
                      poisson_depth: int = 9) -> bool:
        """
        포인트클라우드를 STL 메쉬 파일로 변환하는 통합 함수
        
        Args:
            pointcloud: 입력 포인트클라우드 (numpy array 또는 file path 또는 open3d PointCloud)
            stl_output_path: 출력할 STL 파일 경로
            ply_output_path: 중간 PLY 파일 경로 (기본값: STL 경로에서 확장자만 변경)
            freecad_cmd_path: FreeCAD 실행 파일 경로 (기본값: 시스템 PATH에서 찾기)
            poisson_depth: Poisson 메쉬 생성 깊이 (기본값: 9)
        
        Returns:
            bool: 변환 성공 여부
        """
        try:
            
            # 1. 포인트클라우드 로드 및 변환
            if isinstance(pointcloud, str):
                # 파일 경로인 경우 파일에서 로드
                ext = os.path.splitext(pointcloud)[1].lower()
                if ext in ['.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts']:
                    pcd = o3d.io.read_point_cloud(pointcloud)
                elif ext in ['.txt', '.csv']:
                    data = np.loadtxt(pointcloud, delimiter=None)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
                else:
                    raise ValueError(f"지원하지 않는 파일 형식: {ext}")
            elif isinstance(pointcloud, np.ndarray):
                # numpy 배열인 경우 open3d PointCloud로 변환
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            elif hasattr(pointcloud, 'points'):
                # 이미 open3d PointCloud 객체인 경우
                pcd = pointcloud
            else:
                raise ValueError("포인트클라우드는 파일 경로, numpy 배열, 또는 open3d PointCloud 객체여야 합니다")
            
            # 2. 포인트 클라우드 검증
            if len(pcd.points) == 0:
                raise ValueError("포인트클라우드가 비어있습니다")
            
            # 3. 법선 벡터 추정 및 Poisson 메쉬 생성
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
            
            # 4. 메쉬 정리 (중복 제거, 퇴화 삼각형 제거 등)
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.remove_non_manifold_edges()
            
            # 메쉬 법선 벡터 계산 (STL 저장을 위해 필수)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            # 5. PLY 파일 경로 설정
            if ply_output_path is None:
                ply_output_path = os.path.splitext(stl_output_path)[0] + ".ply"
            
            # 6. 출력 디렉토리 생성
            os.makedirs(os.path.dirname(stl_output_path), exist_ok=True)
            os.makedirs(os.path.dirname(ply_output_path), exist_ok=True)
            
            # 7. PLY 파일로 메쉬 저장
            success = o3d.io.write_triangle_mesh(ply_output_path, mesh)
            if not success:
                raise RuntimeError(f"PLY 파일 저장 실패: {ply_output_path}")
            
            # 8. FreeCAD를 사용하여 STL로 변환
            convert_script_path = os.path.join(os.path.dirname(ply_output_path), "convert_ply_to_stl.py")
            converter_script = f"""# -*- coding: utf-8 -*-
import Mesh
import FreeCAD

try:
    FreeCAD.newDocument()
    mesh_obj = Mesh.Mesh(r\"{ply_output_path}\")
    doc = FreeCAD.ActiveDocument
    mesh_feature = doc.addObject(\"Mesh::Feature\", \"MeshObj\")
    mesh_feature.Mesh = mesh_obj
    mesh_obj.write(r\"{stl_output_path}\")
    print(\"STL export completed successfully.\")
except Exception as e:
    print(f\"Error during STL conversion: {{e}}\")
    exit(1)
"""
            
            with open(convert_script_path, "w", encoding='utf-8') as f:
                f.write(converter_script)
            
            # 9. FreeCAD 실행
            freecad_commands = [
                freecad_cmd_path if freecad_cmd_path else "freecad",
                "FreeCAD",
                "freecad-daily"
            ]
            
            conversion_success = False
            for cmd in freecad_commands:
                if cmd is None:
                    continue
                try:
                    result = subprocess.run(
                        [cmd, convert_script_path], 
                        check=True, 
                        capture_output=True, 
                        text=True,
                        timeout=60
                    )
                    if "STL export completed successfully" in result.stdout:
                        conversion_success = True
                        break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            # 10. 임시 파일 정리
            try:
                os.remove(convert_script_path)
            except:
                pass
            
            if not conversion_success:
                # FreeCAD를 사용할 수 없는 경우 Open3D로 직접 STL 저장 시도
                try:
                    # STL 저장 전에 메쉬 법선 계산 (필수)
                    mesh.compute_vertex_normals()
                    mesh.compute_triangle_normals()
                    
                    success = o3d.io.write_triangle_mesh(stl_output_path, mesh)
                    if success:
                        print(f"[INFO] Open3D를 사용하여 STL 파일 생성: {stl_output_path}")
                        return True
                    else:
                        raise RuntimeError("Open3D STL 저장 실패")
                except Exception as e:
                    print(f"[ERROR] STL 변환 실패 (FreeCAD 및 Open3D): {e}")
                    return False
            else:
                print(f"[INFO] FreeCAD를 사용하여 STL 파일 생성: {stl_output_path}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] pointcloud_to_stl 변환 중 오류 발생: {e}")
            return False

    def _check_termination(self) -> bool:
        """조기 종료 조건 확인 (100 스텝 초과, 분포 일치율 95% 이상)"""
        # 100 스텝 초과
        if self.step_number >= 100:
            self.terminated = True
            return True
        
        # 분포 일치율 95% 이상
        if self.simulation_result_s1 is not None and self.target is not None:
            try:
                intensity_map = self.simulation_result_s1.get('intensity_map')
                if intensity_map is not None:
                    # 간단한 일치율 계산
                    correlation = np.corrcoef(intensity_map.flatten(), self.target.flatten())[0, 1]
                    if not np.isnan(correlation) and correlation >= 0.95:
                        self.terminated = True
                        return True
            except Exception:
                pass
        
        return False
    
    def _get_pointcloud(self) -> np.ndarray:
        """현재 포인트 클라우드 반환"""
        return self.current_pointcloud.copy() if self.current_pointcloud is not None else np.zeros((0, 3), dtype=np.float32)
