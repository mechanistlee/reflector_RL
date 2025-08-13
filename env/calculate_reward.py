"""
SPEOS 광학 시뮬레이션 리워드 계산 모듈
========================================

이 모듈은 SPEOS 광학 시뮬레이션 환경을 위한 리워드 계산 함수들을 포함합니다.
주요 기능:
- SPEOS 광학 성능 기반 리워드 계산
- 광학 효율, 균일성, 타겟 매칭 평가
- 클래스 기반 리워드 계산기 제공
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging


def calculate_shape_smoothness_penalty(current_pointcloud: np.ndarray, config) -> float:
    """
    리플렉터의 형상 평활성을 기반으로 페널티를 계산합니다.
    인접한 포인트 간의 Z값 차이가 smoothing_threshold를 초과하면 페널티를 부과합니다.
    
    Args:
        current_pointcloud: 포인트클라우드 배열 (N, 3) - [x, y, z]
        config: 설정 객체 (smoothing_threshold, grid_rows, grid_cols 포함)
    
    Returns:
        float: 평활성 점수 (0.0 ~ 1.0, 높을수록 좋음)
    """
    try:
        # config에서 smoothing_threshold 가져오기 (기본값: 1.5mm)
        smoothing_threshold = getattr(config, 'smoothing_threshold', 1.5)
        grid_rows = getattr(config, 'grid_rows', 10)
        grid_cols = getattr(config, 'grid_cols', 10)
        
        # 포인트클라우드를 그리드 형태로 재구성
        if len(current_pointcloud) != grid_rows * grid_cols:
            # 포인트 수가 맞지 않으면 기본 페널티 반환
            return -0.5
        
        # Z값을 그리드 형태로 변환
        z_grid = current_pointcloud[:, 2].reshape(grid_rows, grid_cols)
        
        penalty_count = 0
        total_comparisons = 0
        
        # 각 포인트에 대해 인접한 포인트들과 Z값 차이 확인 (중복 없는 4방향만)
        for i in range(grid_rows):
            for j in range(grid_cols):
                current_z = z_grid[i, j]
                
                # 중복 없는 4방향 확인: 아래(1,0), 오른쪽(0,1), 오른쪽아래(1,1), 왼쪽아래(1,-1)
                # 이렇게 하면 각 포인트 쌍은 한 번만 계산됨
                directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    # 경계 체크: 인접 포인트가 그리드 범위 내에 있는지 확인
                    if 0 <= ni < grid_rows and 0 <= nj < grid_cols:
                        neighbor_z = z_grid[ni, nj]
                        z_diff = abs(current_z - neighbor_z)
                        total_comparisons += 1
                        
                        if z_diff > smoothing_threshold:
                            # 임계값 초과량만큼 페널티 누적 (연속적 가중 페널티)
                            penalty_count += max(0, z_diff - smoothing_threshold)
        
        # 평활성 점수 계산 (페널티 비율을 이용)
        if total_comparisons > 0:
            penalty_ratio = penalty_count / total_comparisons
            # 페널티 비율을 역으로 변환하여 평활성 점수로 사용
            # 0% 페널티 = 1.0 점수, 100% 페널티 = -1.0 점수
            smoothness_score = 1.0 - 2.0 * penalty_ratio
        else:
            smoothness_score = 0.0
        
        return smoothness_score
        
    except Exception as e:
        # 오류 발생 시 기본 페널티 반환
        logging.warning(f"Shape smoothness calculation error: {e}")
        return -0.5


def calculate_speos_reward(simulation_result: Dict, 
                          metadata: Dict,
                          current_pointcloud: np.ndarray,
                          config) -> Tuple[float, Dict]:
    """
    Calculate reward based on SPEOS optical performance
    
    Args:
        simulation_result: Dictionary containing simulation results (intensity_map, efficiency, etc.)
        metadata: Dictionary containing simulation metadata (computation_time, status_code, etc.)
        current_pointcloud: Current point cloud array with shape (N, 3)
        config: Configuration object with grid_rows, grid_cols, etc.
    
    Returns:
        Tuple of (reward_value, reward_metadata)
    """
    
    # Extract results from simulation_result dictionary
    intensity_map = simulation_result.get("intensity_map", np.zeros((config.grid_rows, config.grid_cols)))
    efficiency = simulation_result.get("efficiency", 0.0)
    
    # Calculate missing metrics from intensity_map
    # 1. Uniformity ratio calculation
    if intensity_map.size > 0:
        mean_intensity = np.mean(intensity_map)
        if mean_intensity > 0:
            uniformity_ratio = 1.0 - (np.std(intensity_map) / mean_intensity)  # Higher is better
            uniformity_ratio = max(0.0, min(1.0, uniformity_ratio))  # Clamp to [0,1]
        else:
            uniformity_ratio = 0.0
    else:
        uniformity_ratio = 0.0
    
    # 2. Target match score calculation (simplified target: center-focused distribution)
    if intensity_map.size > 0:
        # Create ideal target pattern (gaussian-like center focus)
        center_r, center_c = config.grid_rows // 2, config.grid_cols // 2
        target_pattern = np.zeros_like(intensity_map)
        
        for r in range(config.grid_rows):
            for c in range(config.grid_cols):
                dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                target_pattern[r, c] = np.exp(-0.1 * dist**2)  # Gaussian target
        
        # Normalize both patterns to [0,1] range for fair comparison
        intensity_normalized = intensity_map / (np.max(intensity_map) + 1e-8)
        target_normalized = target_pattern / (np.max(target_pattern) + 1e-8)
        
        # Calculate target match score: 1 / (1 + MSE)
        # MSE = mean squared error between normalized patterns
        mse = np.mean((intensity_normalized - target_normalized)**2)
        
        # Normalize MSE to [0,1] range (since both patterns are normalized to [0,1])
        # Maximum possible MSE is 2.0 (when one pattern is all 0s and other is all 1s)
        # But practically, we clamp to 1.0 for better score distribution
        mse_normalized = min(mse / 1.0, 1.0)  # Normalize and clamp to [0,1]
        
        target_match_score = ( 2.0 / (1.0 + mse_normalized) ) - 1  # Higher score for lower MSE
    else:
        target_match_score = 0.0
    
    # Calculate reward components
    # 1. Target pattern matching (40% weight) - calculated as 1/(1+normalized_MSE) between intensity and ideal pattern
    distribution_factor = target_match_score
    
    # 2. Optical efficiency (30% weight) - from simulation result
    efficiency_factor = efficiency 
    
    # 3. Uniformity (20% weight) - calculated from intensity map std/mean ratio
    uniformity_factor = uniformity_ratio
    
    # 4. Shape regularity (10% weight) - based on smoothness between adjacent points
    shape_factor = calculate_shape_smoothness_penalty(current_pointcloud, config) /40
    
    # Additional penalties
    # Size constraint penalty
    z_range = np.max(current_pointcloud[:, 2]) - np.min(current_pointcloud[:, 2])
    size_penalty = -max(0, z_range - 40.0) * 0.1
    
    # Simulation quality penalty
    quality_penalty = 0.0
    if metadata.get("status_code", 0) != 0:
        quality_penalty = 0.5  # Penalty for simulation errors
    
    # Combined reward with updated weights including shape factor
    # 분배: 타겟 매칭 50%, 효율성 30%, 형상 평활성 10%, 균일성 10%
    reward = (0.7 * efficiency_factor + 
              0.3 * shape_factor)
    
    reward_metadata = {
        "distribution_factor": float(distribution_factor),
        "efficiency_factor": float(efficiency_factor),
        "uniformity_factor": float(uniformity_factor),
        "shape_factor": float(shape_factor),
        "size_penalty": float(size_penalty),
        "quality_penalty": float(quality_penalty),
        "computation_time": metadata.get("computation_time", 0.0),
        "simulation_quality": metadata.get("simulation_quality", 0.0)
    }
    
    return float(reward), reward_metadata





class SpeosRewardCalculator:
    """
    SPEOS 광학 시뮬레이션 전용 리워드 계산기
    """
    
    def __init__(self):
        """
        SPEOS 리워드 계산기 초기화
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # SPEOS 리워드 가중치 설정 (형상 평활성 포함)
        self.weights = {
            "distribution_factor": 0.5,  # 타겟 패턴 매칭 (50%)
            "efficiency_factor": 0.25,   # 광학 효율 (25%)
            "shape_factor": 0.15,        # 형상 평활성 (15%) - 새로 추가된 중요 요소
            "uniformity_factor": 0.1     # 균일성 (10%)
        }
    
    def calculate_reward(self, simulation_result: Dict,
                        metadata: Dict,
                        current_pointcloud: np.ndarray,
                        config) -> Tuple[float, Dict]:
        """
        SPEOS 시뮬레이션 결과에 기반한 리워드 계산
        
        Args:
            simulation_result: 시뮬레이션 결과 딕셔너리
            metadata: 시뮬레이션 메타데이터 딕셔너리
            current_pointcloud: 현재 포인트클라우드 (N, 3)
            config: 설정 객체
        
        Returns:
            Tuple of (reward_value, reward_metadata)
        """
        try:
            reward, reward_metadata = calculate_speos_reward(
                simulation_result, metadata, current_pointcloud, config
            )
            
            self.logger.debug(f"SPEOS 리워드 계산 완료: {reward:.6f}")
            return reward, reward_metadata
            
        except Exception as e:
            self.logger.error(f"SPEOS 리워드 계산 오류: {e}")
            # 오류 시 기본 리워드 반환
            return 0.0, {"error": str(e)}
    
    def get_reward_weights(self) -> Dict[str, float]:
        """
        SPEOS 리워드 구성 요소 가중치 반환
        
        Returns:
            리워드 구성 요소 가중치 딕셔너리
        """
        return self.weights.copy()
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        리워드 구성 요소 가중치 업데이트
        
        Args:
            new_weights: 새로운 가중치 딕셔너리
        """
        # 가중치 검증
        if not all(key in self.weights for key in new_weights.keys()):
            raise ValueError("잘못된 가중치 키가 포함되어 있습니다")
        
        # 가중치 합이 1에 가까운지 확인
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"가중치 합이 1이 아닙니다: {total_weight}")
        
        self.weights.update(new_weights)
        self.logger.info(f"SPEOS 리워드 가중치 업데이트: {new_weights}")


# SPEOS 리워드 계산을 위한 편의 함수
def speos_reward_calculation(simulation_result: Dict, metadata: Dict, 
                           current_pointcloud: np.ndarray, config) -> Tuple[float, Dict]:
    """SPEOS 리워드 계산을 위한 편의 함수"""
    return calculate_speos_reward(simulation_result, metadata, current_pointcloud, config)


# SPEOS 리워드 계산 모듈 테스트
if __name__ == "__main__":
    print("SPEOS 리워드 계산 모듈 - 테스트 모드")
    
    # SPEOS 리워드 계산 테스트
    print("SPEOS 리워드 계산 테스트 중...")
    
    # 모의 데이터
    class MockConfig:
        def __init__(self):
            self.grid_rows = 10
            self.grid_cols = 10
    
    config = MockConfig()
    
    simulation_result = {
        "intensity_map": np.random.random((10, 10)),
        "efficiency": 0.75,
        "uniformity_ratio": 0.6,
        "target_match_score": 0.8
    }
    
    metadata = {
        "computation_time": 2.5,
        "status_code": 0,
        "simulation_quality": 0.95
    }
    
    current_pointcloud = np.random.random((100, 3))
    
    reward, reward_metadata = calculate_speos_reward(
        simulation_result, metadata, current_pointcloud, config
    )
    
    print(f"테스트 리워드: {reward:.6f}")
    print(f"리워드 구성 요소: {reward_metadata}")
    
    # SpeosRewardCalculator 클래스 테스트
    print("\nSpeosRewardCalculator 클래스 테스트 중...")
    calculator = SpeosRewardCalculator()
    reward2, metadata2 = calculator.calculate_reward(
        simulation_result, metadata, current_pointcloud, config
    )
    
    print(f"클래스 기반 리워드: {reward2:.6f}")
    print(f"가중치: {calculator.get_reward_weights()}")
    
    print("테스트 완료!")

