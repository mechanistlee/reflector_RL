"""
커스텀 SAC 알고리즘 - 멀티 리플렉터 환경 지원
===============================================

이 모듈은 stable-baselines3의 SAC 알고리즘을 상속받아
멀티 리플렉터 환경에서 개별 액션 샘플링을 지원합니다.
"""

import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from typing import Any, Dict, Optional, Type, Union

import torch as th
from stable_baselines3.sac import SAC
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import ActionNoise

# 우리가 수정한 OffPolicyAlgorithm 임포트
from common.off_policy_algorithm import OffPolicyAlgorithm


class MultiReflectorSAC(SAC):
    """
    멀티 리플렉터 환경을 위한 커스텀 SAC 알고리즘
    
    기본 SAC 알고리즘을 상속받되, collect_rollouts 메서드는
    우리가 수정한 OffPolicyAlgorithm의 것을 사용합니다.
    """
    
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        print("🎯 MultiReflectorSAC 초기화 완료 - 멀티 리플렉터 환경 지원")
    
    def collect_rollouts(self, *args, **kwargs):
        """
        우리가 수정한 OffPolicyAlgorithm의 collect_rollouts 메서드 사용
        """
        return OffPolicyAlgorithm.collect_rollouts(self, *args, **kwargs)
    
    def _sample_action(self, *args, **kwargs):
        """
        우리가 수정한 OffPolicyAlgorithm의 _sample_action 메서드 사용
        """
        return OffPolicyAlgorithm._sample_action(self, *args, **kwargs)


# 편의를 위한 별칭
CustomSAC = MultiReflectorSAC
