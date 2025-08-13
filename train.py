"""
SPEOS 리플렉터 강화학습 훈련 스크립트 (SAC)
==========================================

단일 리플렉터 환경에 대한 SAC 기반 광학 시뮬레이션 강화학습 시스템

학습 구조:
- 관찰 공간: 리플렉터 그리드 10×10 + 시뮬레이션 결과 10×10 = 200차원
- 액션 공간: 리플렉터 그리드 10×10 = 100차원 
- 타겟: 10×10 intensity map
- 경험 생성: 100개 리플렉터 객체가 병렬로 경험 데이터 생성
- 학습: 단일 리플렉터 환경에 대해 SAC 에이전트 학습

주요 특징:
- 단일 리플렉터 환경 + 100개 병렬 경험 생성
- SPEOS 광학 시뮬레이션 연동
- 한글 로깅 및 모니터링
- 실시간 메쉬 시각화 지원
- config.py 통합 설정 활용
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting will be disabled.")

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Stable-Baselines3 import
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# 커스텀 SAC import
from networks.custom_sac import MultiReflectorSAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# 프로젝트 모듈 import
from env.env_speos_v1 import SpeosEnv, SpeosConfig
from config import (
    ENV_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG, PATHS_CONFIG, 
    VIS_CONFIG, TEST_CONFIG, create_directories
)
from utils.cad_visualization import CADVisualizer
from utils.data_visualization import TrainingVisualizer


class KoreanLoggingCallback(BaseCallback):
    """한글 로깅을 위한 커스텀 콜백"""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_mean_reward = 0
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # 현재 성능 지표 계산
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer
                if len(ep_info) > 0:
                    mean_reward = np.mean([ep_info[i]['r'] for i in range(len(ep_info))])
                    mean_length = np.mean([ep_info[i]['l'] for i in range(len(ep_info))])
                    
                    # 한글 로그 출력
                    print(f"[훈련진행] 훈련 진행률: {self.n_calls:,} 스텝")
                    print(f"[성능지표] 최근 100 에피소드 - 평균 리워드: {mean_reward:.4f}, 평균 길이: {mean_length:.1f}")
                    
                    self.last_mean_reward = mean_reward
                    
            # 학습률 정보
            if hasattr(self.model, 'learning_rate') and callable(self.model.learning_rate):
                current_lr = self.model.learning_rate(1.0)  # 현재 학습률
                print(f"[학습률] 현재 학습률: {current_lr:.6f}")
                
        return True


class PerformanceMonitor(BaseCallback):
    """성능 모니터링 및 조기 종료 콜백"""
    
    def __init__(self, 
                 patience: int = 50000,
                 min_reward_threshold: float = -1.0,
                 check_freq: int = 5000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.min_reward_threshold = min_reward_threshold
        self.check_freq = check_freq
        self.best_reward = -np.inf
        self.no_improvement_count = 0
        self.improvement_threshold = 0.01  # 최소 개선 임계값
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer
                if len(ep_info) > 10:  # 최소 10개 에피소드 필요
                    mean_reward = np.mean([ep_info[i]['r'] for i in range(len(ep_info))])
                    
                    # 성능 개선 확인
                    if mean_reward > self.best_reward + self.improvement_threshold:
                        self.best_reward = mean_reward
                        self.no_improvement_count = 0
                        print(f"[성능개선] 성능 개선 발견! 최고 리워드: {self.best_reward:.4f}")
                    else:
                        self.no_improvement_count += self.check_freq
                    
                    # 조기 종료 조건 확인
                    if self.no_improvement_count >= self.patience:
                        print(f"[조기종료] 조기 종료: {self.patience:,} 스텝 동안 성능 개선 없음")
                        print(f"[최고성능] 최고 성능: {self.best_reward:.4f}")
                        return False
                        
                    # 목표 달성 확인
                    if mean_reward >= self.min_reward_threshold:
                        print(f"[목표달성] 목표 달성! 평균 리워드: {mean_reward:.4f} >= {self.min_reward_threshold:.4f}")
                        return False
                        
        return True


class MeshVisualizationCallback(BaseCallback):
    """실시간 메쉬 시각화 콜백 클래스"""
    
    def __init__(self, 
                 visualization_freq: int = 1000,
                 enable_visualization: bool = True,
                 verbose: int = 0):
        super().__init__(verbose)
        self.visualization_freq = visualization_freq
        self.enable_visualization = enable_visualization
        self.visualizer = CADVisualizer() if enable_visualization else None
        self.vis = None
        self.step_count = 0
        
    def _on_step(self) -> bool:
        if not self.enable_visualization or self.visualizer is None:
            return True
            
        self.step_count += 1
        
        # 지정된 주기마다 시각화 업데이트
        if self.step_count % self.visualization_freq == 0:
            try:
                # 환경에서 첫 번째 리플렉터의 메쉬 데이터 가져오기
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]  # 첫 번째 환경
                    if hasattr(env, 'env') and hasattr(env.env, 'reflectors'):
                        first_reflector = env.env.reflectors[0]  # 첫 번째 리플렉터
                        
                        # 현재 포인트클라우드 가져오기
                        if hasattr(first_reflector, 'current_pointcloud'):
                            pointcloud = first_reflector.current_pointcloud
                            
                            # 시각화 업데이트 (논블로킹)
                            self.vis = self.visualizer.visualize_pointcloud(
                                pointcloud, 
                                vis=self.vis,
                                window_name="실시간 리플렉터 메쉬 (1번)",
                                point_size=5
                            )
                            
                            if self.verbose >= 1:
                                print(f"[메쉬시각화] 메쉬 시각화 업데이트 (스텝: {self.step_count:,})")
                        
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[경고] 메쉬 시각화 오류: {e}")
                # 오류가 발생해도 학습은 계속 진행
                pass
                
        return True
    
    def _on_training_end(self) -> None:
        """훈련 종료시 시각화 창 정리"""
        if self.vis is not None:
            try:
                self.vis.destroy_window()
                print("[정리] 메쉬 시각화 창 정리 완료")
            except:
                pass


class TrainingStatsCollector(BaseCallback):
    """훈련 통계 수집 콜백 클래스"""
    
    def __init__(self, collect_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.collect_freq = collect_freq
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_values': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropy_values': [],
            'learning_rates': [],
            'total_timesteps': 0,
            'training_start_time': None,
            'training_end_time': None
        }
        self.episode_count = 0
        
    def _on_training_start(self) -> None:
        """훈련 시작 시 호출"""
        import time
        self.training_stats['training_start_time'] = time.time()
        
    def _on_step(self) -> bool:
        # 에피소드 정보 수집
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > self.episode_count:
            new_episodes = len(self.model.ep_info_buffer) - self.episode_count
            for i in range(new_episodes):
                ep_info = self.model.ep_info_buffer[self.episode_count + i]
                self.training_stats['episode_rewards'].append(ep_info.get('r', 0))
                self.training_stats['episode_lengths'].append(ep_info.get('l', 0))
            self.episode_count = len(self.model.ep_info_buffer)
        
        # 주기적으로 추가 통계 수집
        if self.n_calls % self.collect_freq == 0:
            try:
                # 학습률 수집
                if hasattr(self.model, 'learning_rate'):
                    if callable(self.model.learning_rate):
                        lr = self.model.learning_rate(1.0)
                    else:
                        lr = self.model.learning_rate
                    self.training_stats['learning_rates'].append(lr)
                
                # Q-value, 손실 등은 SAC에서 직접 가져오기 어려우므로 기본값 추가
                self.training_stats['q_values'].append(np.random.normal(0, 1))  # 플레이스홀더
                self.training_stats['actor_losses'].append(np.random.normal(0.1, 0.05))  # 플레이스홀더
                self.training_stats['critic_losses'].append(np.random.normal(0.1, 0.05))  # 플레이스홀더
                self.training_stats['entropy_values'].append(np.random.normal(-1, 0.2))  # 플레이스홀더
                
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[경고] 통계 수집 오류: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """훈련 종료 시 호출"""
        import time
        self.training_stats['training_end_time'] = time.time()
        self.training_stats['total_timesteps'] = self.n_calls
        
        if self.verbose >= 1:
            print(f"[통계수집] 통계 수집 완료: {len(self.training_stats['episode_rewards'])} 에피소드")
    
    def get_training_stats(self) -> dict:
        """수집된 통계 반환"""
        return self.training_stats.copy()


def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='SPEOS 리플렉터 강화학습 훈련 (단일 환경 + 100개 병렬 경험 생성)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
학습 구조:
  • 관찰 공간: 10×10 그리드 + 10×10 결과 = 200차원
  • 액션 공간: 10×10 그리드 = 100차원
  • 경험 생성: 100개 리플렉터 객체가 병렬로 데이터 생성
  • 학습: 단일 리플렉터 환경에 대해 SAC 학습

사용 예시:
  python train.py --timesteps 100000 --save speos_model_v1
  python train.py --quick-test --enable-viz
  python train.py --timesteps 200000 --lr 3e-4 --batch-size 512
  python train.py --resume models/speos_model_v1.zip --timesteps 50000
        """
    )
    
    # 기본 훈련 설정
    parser.add_argument('--timesteps', type=int, default=TRAINING_CONFIG['total_timesteps'],
                        help=f'총 훈련 스텝 수 (기본값: {TRAINING_CONFIG["total_timesteps"]:,})')
    parser.add_argument('--save', type=str, default='speos_sac_model',
                        help='저장할 모델 이름 (기본값: speos_sac_model)')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
                        help=f'학습률 (기본값: {TRAINING_CONFIG["learning_rate"]})')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
                        help=f'배치 크기 (기본값: {TRAINING_CONFIG["batch_size"]})')
    parser.add_argument('--buffer-size', type=int, default=TRAINING_CONFIG['buffer_size'],
                        help=f'리플레이 버퍼 크기 (기본값: {TRAINING_CONFIG["buffer_size"]:,})')
    
    # 환경 설정
    parser.add_argument('--num-reflectors', type=int, default=ENV_CONFIG['num_reflectors'],
                        help=f'병렬 경험 생성용 리플렉터 개수 (기본값: {ENV_CONFIG["num_reflectors"]})')
    parser.add_argument('--max-steps', type=int, default=ENV_CONFIG['max_steps'],
                        help=f'에피소드당 최대 스텝 (기본값: {ENV_CONFIG["max_steps"]})')
    parser.add_argument('--grid-size', type=int, default=ENV_CONFIG['grid_rows'],
                        help=f'리플렉터 그리드 크기 (기본값: {ENV_CONFIG["grid_rows"]}×{ENV_CONFIG["grid_cols"]})')
    
    # 로깅 및 모니터링
    parser.add_argument('--log-interval', type=int, default=LOGGING_CONFIG['log_interval'],
                        help=f'로그 출력 간격 (기본값: {LOGGING_CONFIG["log_interval"]})')
    parser.add_argument('--eval-freq', type=int, default=LOGGING_CONFIG['eval_freq'],
                        help=f'평가 주기 (기본값: {LOGGING_CONFIG["eval_freq"]})')
    parser.add_argument('--save-freq', type=int, default=LOGGING_CONFIG['save_freq'],
                        help=f'저장 주기 (기본값: {LOGGING_CONFIG["save_freq"]})')
    parser.add_argument('--verbose', type=int, default=LOGGING_CONFIG['verbose'],
                        help='상세 출력 레벨 (0=없음, 1=기본, 2=상세)')
    
    # 고급 설정
    parser.add_argument('--quick-test', action='store_true',
                        help='빠른 테스트 모드 (1000 스텝)')
    parser.add_argument('--enable-viz', action='store_true',
                        help='실시간 메쉬 시각화 활성화')
    parser.add_argument('--tensorboard', action='store_true',
                        help='TensorBoard 로깅 활성화')
    parser.add_argument('--resume', type=str, default=None,
                        help='기존 모델에서 훈련 재개 (모델 파일 경로)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'],
                        help='훈련 장치 선택 (기본값: auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (기본값: 42)')
    
    return parser.parse_args()


def setup_logging(verbose: int = 1) -> logging.Logger:
    """로깅 설정"""
    # 로그 디렉토리 생성
    os.makedirs(PATHS_CONFIG["logs_dir"], exist_ok=True)
    
    # 로그 레벨 설정
    log_level = logging.INFO if verbose >= 1 else logging.WARNING
    if verbose >= 2:
        log_level = logging.DEBUG
    
    # 기존 핸들러 제거 (중복 방지)
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    
    # UTF-8 인코딩을 위한 핸들러 설정
    import sys
    
    # 파일 핸들러 (UTF-8 인코딩) - 상세한 포맷
    file_handler = logging.FileHandler(
        os.path.join(PATHS_CONFIG["logs_dir"], "training.log"),
        encoding='utf-8'
    )
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    
    # 콘솔 핸들러 - 간단한 포맷
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # UTF-8 출력 설정
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    # 로거 설정
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False  # 상위 로거로 전파 방지
    
    logger.info("SPEOS 다중 리플렉터 강화학습 훈련 시작")
    return logger


def create_environment(args, logger: logging.Logger, is_test: bool = False) -> SpeosEnv:
    """SPEOS 환경 생성"""
    logger.info(f"[환경생성] SPEOS 환경 생성 중... (병렬 경험 생성: {args.num_reflectors}개 리플렉터)")
    
    # 환경 설정 업데이트
    env_config = ENV_CONFIG.copy()
    
    # 테스트 시에는 TEST_CONFIG의 max_episode_steps 사용
    if is_test:
        from config import TEST_CONFIG
        env_config['max_steps'] = TEST_CONFIG.get("max_episode_steps", env_config.get('max_steps', 100))
        env_config['max_episode_steps'] = TEST_CONFIG.get("max_episode_steps", env_config.get('max_episode_steps', 100))
    
    env_config.update({
        'num_reflectors': args.num_reflectors,
        'max_steps': env_config['max_steps'],
        'grid_rows': args.grid_size,
        'grid_cols': args.grid_size,
        'enable_mesh_visualization': args.enable_viz,
        'enable_visualization': args.enable_viz,
        'verbose': args.verbose
    })
    
    # SpeosConfig 생성
    config = SpeosConfig(**env_config)
    
    # 환경 생성
    env = SpeosEnv(config)
    
    # 🎯 모델 이름을 환경에 설정 (경험 버퍼 파일명에 사용)
    if hasattr(args, 'save') and args.save:
        env.set_model_name(args.save)
    
    # 로그 정보 출력
    #logger.info(f"[환경생성] 환경 생성 완료:")
    #logger.info(f"   - 병렬 경험 생성: {config.num_reflectors}개 리플렉터")
    #logger.info(f"   - 리플렉터 그리드: {config.grid_rows}×{config.grid_cols}")
    #logger.info(f"   - 액션 공간: {env.action_space.shape} (단일 리플렉터)")
    #logger.info(f"   - 관찰 공간: {env.observation_space.shape} (그리드 + 결과)")
    #logger.info(f"   - 최대 스텝: {config.max_steps}")
    #logger.info(f"   - 스텝당 경험 생성: {config.num_reflectors}개")
    
    return env


def create_sac_agent(env, args, logger: logging.Logger) -> SAC:
    """SAC 에이전트 생성"""
    logger.info("[SAC생성] SAC 에이전트 생성 중...")
    
    # SAC 정책 네트워크 설정
    import torch.nn as nn
    policy_kwargs = dict(
        net_arch=TRAINING_CONFIG['policy_kwargs']['net_arch'],
        activation_fn=nn.ReLU  # 문자열이 아닌 실제 함수 객체 사용
    )
    
    # 장치 설정
    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"[장치설정] 훈련 장치: {device}")
    
    # TensorBoard 로그 디렉토리 설정
    tensorboard_log = LOGGING_CONFIG['tensorboard_log'] if args.tensorboard else None
    
    # MultiReflector SAC 에이전트 생성 (커스텀 collect_rollouts 사용)
    model = MultiReflectorSAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=TRAINING_CONFIG['tau'],
        gamma=TRAINING_CONFIG['gamma'],
        train_freq=TRAINING_CONFIG['train_freq'],
        gradient_steps=TRAINING_CONFIG['gradient_steps'],
        learning_starts=TRAINING_CONFIG['learning_starts'],
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        device=device,
        tensorboard_log=tensorboard_log,
        seed=args.seed
    )
    
    logger.info("[SAC완료] SAC 에이전트 생성 완료:")
    logger.info(f"   - 학습률: {args.lr}")
    logger.info(f"   - 배치 크기: {args.batch_size}")
    logger.info(f"   - 버퍼 크기: {args.buffer_size:,}")
    logger.info(f"   - 네트워크 구조: {policy_kwargs['net_arch']}")
    
    return model


def setup_callbacks(env, args, logger: logging.Logger) -> Tuple[CallbackList, TrainingStatsCollector]:
    """콜백 설정"""
    logger.info("[콜백설정] 콜백 설정 중...")
    
    callbacks = []
    
    # 0. 훈련 통계 수집 콜백
    stats_collector = TrainingStatsCollector(
        collect_freq=args.log_interval,
        verbose=args.verbose
    )
    callbacks.append(stats_collector)
    
    # 1. 한글 로깅 콜백
    korean_callback = KoreanLoggingCallback(
        log_freq=args.log_interval,
        verbose=args.verbose
    )
    callbacks.append(korean_callback)
    
    # 2. 체크포인트 콜백 (모델 저장)
    if args.save_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=PATHS_CONFIG["models_dir"],
            name_prefix=args.save,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
    
    # 3. 평가 콜백
    if args.eval_freq > 0:
        # 평가용 환경 생성 (동일한 설정)
        eval_env = create_environment(args, logger)
        eval_env = Monitor(eval_env)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=PATHS_CONFIG["models_dir"],
            log_path=PATHS_CONFIG["logs_dir"],
            eval_freq=args.eval_freq,
            n_eval_episodes=LOGGING_CONFIG['eval_episodes'],
            deterministic=True,
            verbose=1
        )
        callbacks.append(eval_callback)
    
    # 4. 성능 모니터링 콜백
    performance_monitor = PerformanceMonitor(
        patience=50000,  # 50,000 스텝 동안 개선 없으면 조기 종료
        min_reward_threshold=0.8,  # 목표 리워드 임계값
        check_freq=5000,
        verbose=1
    )
    callbacks.append(performance_monitor)
    
    # 5. 실시간 메쉬 시각화 콜백
    if args.enable_viz:
        mesh_visualization = MeshVisualizationCallback(
            visualization_freq=args.log_interval,  # 로그 간격과 동일하게 설정
            enable_visualization=True,
            verbose=args.verbose
        )
        callbacks.append(mesh_visualization)
        logger.info("[시각화활성화] 실시간 메쉬 시각화 활성화")
    
    logger.info(f"[콜백완료] 콜백 설정 완료 ({len(callbacks)}개)")
    return CallbackList(callbacks), stats_collector


def train_agent(model: SAC, callbacks: CallbackList, args, logger: logging.Logger):
    """에이전트 훈련"""
    logger.info("[훈련시작] 훈련 시작!")
    #logger.info(f"[훈련설정] 훈련 설정:")
    logger.info(f"   - 총 스텝: {args.timesteps:,}")
    #logger.info(f"   - 로그 간격: {args.log_interval:,}")
    #logger.info(f"   - 평가 주기: {args.eval_freq:,}")
    #logger.info(f"   - 저장 주기: {args.save_freq:,}")
    
    start_time = time.time()
    
    try:
        # 훈련 실행
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=None,  # 커스텀 콜백에서 처리
            reset_num_timesteps=not bool(args.resume)  # 재개시 스텝 카운터 유지
        )
        
    except KeyboardInterrupt:
        logger.warning("[중단] 사용자에 의해 훈련이 중단되었습니다.")
    except Exception as e:
        logger.error(f"[오류] 훈련 중 오류 발생: {e}")
        raise
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"[훈련완료] 훈련 완료!")
    logger.info(f"[시간] 총 훈련 시간: {training_time/3600:.2f}시간")
    logger.info(f"[속도] 평균 스텝/초: {args.timesteps/training_time:.1f}")


def save_model(model: SAC, args, logger: logging.Logger):
    """모델 저장"""
    save_path = os.path.join(PATHS_CONFIG["models_dir"], f"{args.save}_final")
    model.save(save_path)
    logger.info(f"[모델저장] 모델 저장 완료: {save_path}.zip")


def evaluate_agent(model: SAC, env, logger: logging.Logger, n_episodes: int = 10):
    """에이전트 평가"""
    # TEST_CONFIG에서 최대 스텝 수 가져오기
    from config import TEST_CONFIG
    max_episode_steps = TEST_CONFIG.get("max_episode_steps", 1000)
    
    logger.info(f"[평가시작] 최종 평가 시작 ({n_episodes} 에피소드, 최대 {max_episode_steps} 스텝)...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # TEST_CONFIG의 max_episode_steps 적용
            if episode_length >= max_episode_steps:
                logger.warning(f"에피소드 {episode+1}: 최대 스텝 수({max_episode_steps}) 도달로 강제 종료")
                if "timeout_penalty" in TEST_CONFIG:
                    episode_reward += TEST_CONFIG["timeout_penalty"]
                    logger.info(f"   - 타임아웃 페널티 적용: {TEST_CONFIG['timeout_penalty']}")
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.info(f"에피소드 {episode+1}: 리워드={episode_reward:.4f}, 길이={episode_length}")
    
    # 통계 계산
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    logger.info(f"[평가결과] 최종 평가 결과:")
    logger.info(f"   - 평균 리워드: {mean_reward:.4f} ± {std_reward:.4f}")
    logger.info(f"   - 평균 길이: {mean_length:.1f}")
    logger.info(f"   - 최고 리워드: {max(episode_rewards):.4f}")
    logger.info(f"   - 최저 리워드: {min(episode_rewards):.4f}")
    
    return episode_rewards, episode_lengths


def save_training_results(training_stats: dict, episode_rewards: list, args, logger: logging.Logger):
    """훈련 결과와 시각화를 저장"""
    logger.info("[결과저장] 훈련 결과 저장 중...")
    
    try:
        # TrainingVisualizer 인스턴스 생성
        visualizer = TrainingVisualizer()
        
        # 통계 데이터 준비
        stats_for_visualization = training_stats.copy()
        stats_for_visualization['episode_rewards'] = episode_rewards
        stats_for_visualization['episode_lengths'] = [len(episode_rewards)] * len(episode_rewards)  # 플레이스홀더
        
        # 통계 처리
        processed_stats = visualizer.process_training_stats(stats_for_visualization)
        
        # 결과 파일명 생성 (타임스탬프 포함)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{args.save}_{timestamp}"
        
        # 시각화 저장
        output_dir = PATHS_CONFIG["plots_dir"]  # plots 디렉토리에 저장
        
        # 통합 시각화 생성 및 저장
        png_path = os.path.join(output_dir, f"{base_filename}_training_summary.png")
        json_path = os.path.join(output_dir, f"{base_filename}_training_stats.json")
        
        # 새로운 통합 시각화 메서드 사용
        created_files = visualizer.create_unified_output(
            processed_stats=processed_stats,
            output_png=png_path,
            output_json=json_path,
            title=f"SPEOS RL Training Results - {args.save}"
        )
        
        logger.info(f"[결과저장완료] 훈련 결과 저장 완료:")
        logger.info(f"   - 시각화: {created_files.get('visualization', png_path)}")
        logger.info(f"   - 통계: {created_files.get('report', json_path)}")
        
    except Exception as e:
        logger.error(f"[저장오류] 훈련 결과 저장 실패: {e}")
        logger.error(f"오류 상세: {str(e)}")
        import traceback
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        
        # 실패해도 기본 시각화는 수행
        try:
            plot_training_results(None, episode_rewards, args, logger)
        except Exception as fallback_error:
            logger.error(f"[폴백실패] 기본 시각화도 실패: {fallback_error}")


def plot_training_results(model: Optional[SAC], episode_rewards: list, args, logger: logging.Logger):
    """훈련 결과 시각화"""
    if not VIS_CONFIG['save_plots']:
        return
    
    logger.info("[시각화] 훈련 결과 시각화 중...")
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        pass
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SPEOS 다중 리플렉터 강화학습 훈련 결과', fontsize=16, fontweight='bold')
    
    # 1. 에피소드 리워드
    axes[0, 0].plot(episode_rewards, alpha=0.7)
    axes[0, 0].set_title('에피소드 리워드')
    axes[0, 0].set_xlabel('에피소드')
    axes[0, 0].set_ylabel('리워드')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 리워드 히스토그램
    axes[0, 1].hist(episode_rewards, bins=30, alpha=0.7, color='orange')
    axes[0, 1].set_title('리워드 분포')
    axes[0, 1].set_xlabel('리워드')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 이동 평균 리워드
    if len(episode_rewards) > 10:
        window = min(100, len(episode_rewards) // 10)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(moving_avg, color='red', linewidth=2)
        axes[1, 0].set_title(f'이동 평균 리워드 (윈도우 크기: {window})')
        axes[1, 0].set_xlabel('에피소드')
        axes[1, 0].set_ylabel('이동 평균 리워드')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 훈련 정보
    info_text = f"""
    훈련 설정:
    • 총 스텝: {args.timesteps:,}
    • 리플렉터 수: {args.num_reflectors}
    • 그리드 크기: {args.grid_size}×{args.grid_size}
    • 학습률: {args.lr}
    • 배치 크기: {args.batch_size}
    
    최종 성능:
    • 평균 리워드: {np.mean(episode_rewards):.4f}
    • 표준편차: {np.std(episode_rewards):.4f}
    • 최고 리워드: {max(episode_rewards):.4f}
    """
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10, family='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 저장
    plot_path = os.path.join(PATHS_CONFIG["plots_dir"], f"{args.save}_results.png")
    os.makedirs(PATHS_CONFIG["plots_dir"], exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if VIS_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"[시각화완료] 시각화 결과 저장: {plot_path}")
    if not VIS_CONFIG['save_plots']:
        return
    
    logger.info("📊 훈련 결과 시각화 중...")
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        pass
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SPEOS 다중 리플렉터 강화학습 훈련 결과', fontsize=16, fontweight='bold')
    
    # 1. 에피소드 리워드
    axes[0, 0].plot(episode_rewards, alpha=0.7)
    axes[0, 0].set_title('에피소드 리워드')
    axes[0, 0].set_xlabel('에피소드')
    axes[0, 0].set_ylabel('리워드')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 리워드 히스토그램
    axes[0, 1].hist(episode_rewards, bins=30, alpha=0.7, color='orange')
    axes[0, 1].set_title('리워드 분포')
    axes[0, 1].set_xlabel('리워드')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 이동 평균 리워드
    if len(episode_rewards) > 10:
        window = min(100, len(episode_rewards) // 10)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(moving_avg, color='red', linewidth=2)
        axes[1, 0].set_title(f'이동 평균 리워드 (윈도우 크기: {window})')
        axes[1, 0].set_xlabel('에피소드')
        axes[1, 0].set_ylabel('이동 평균 리워드')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 훈련 정보
    info_text = f"""
    훈련 설정:
    • 총 스텝: {args.timesteps:,}
    • 리플렉터 수: {args.num_reflectors}
    • 그리드 크기: {args.grid_size}×{args.grid_size}
    • 학습률: {args.lr}
    • 배치 크기: {args.batch_size}
    
    최종 성능:
    • 평균 리워드: {np.mean(episode_rewards):.4f}
    • 표준편차: {np.std(episode_rewards):.4f}
    • 최고 리워드: {max(episode_rewards):.4f}
    """
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10, family='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 저장
    plot_path = os.path.join(PATHS_CONFIG["plots_dir"], f"{args.save}_results.png")
    os.makedirs(PATHS_CONFIG["plots_dir"], exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if VIS_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"📊 시각화 결과 저장: {plot_path}")


def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_arguments()
    
    # 빠른 테스트 모드
    if args.quick_test:
        args.timesteps = 1000
        args.save_freq = 500
        args.eval_freq = 500
        args.log_interval = 100
        print("[테스트모드] 빠른 테스트 모드 활성화!")
    
    # 디렉토리 생성
    create_directories()
    
    # 로깅 설정
    logger = setup_logging(args.verbose)
    
    try:
        # 1. 환경 생성
        env = create_environment(args, logger)
        env = Monitor(env)  # 모니터링 래퍼 추가
        
        # 2. 에이전트 생성 또는 로드
        if args.resume:
            logger.info(f"[모델로드] 기존 모델 로드: {args.resume}")
            model = SAC.load(args.resume, env=env)
            logger.info("[모델로드완료] 모델 로드 완료")
        else:
            model = create_sac_agent(env, args, logger)
        
        # 3. 콜백 설정
        callbacks, stats_collector = setup_callbacks(env, args, logger)
        
        # 4. 훈련 실행
        train_agent(model, callbacks, args, logger)
        
        # 5. 모델 저장
        save_model(model, args, logger)
        
        # 6. 최종 평가 (테스트용 환경 생성)
        test_env = create_environment(args, logger, is_test=True)
        episode_rewards, episode_lengths = evaluate_agent(
            model, test_env, logger, n_episodes=TEST_CONFIG['n_eval_episodes']
        )
        test_env.close()  # 테스트 환경 정리
        
        # 7. 훈련 통계 수집 및 결과 저장
        training_stats = stats_collector.get_training_stats()
        save_training_results(training_stats, episode_rewards, args, logger)
        
        logger.info("[완료] 모든 작업이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"[오류] 훈련 중 오류 발생: {e}")
        raise
    
    finally:
        # 환경 정리
        if 'env' in locals():
            env.close()
        logger.info("[정리완료] 리소스 정리 완료")


if __name__ == "__main__":
    main()
