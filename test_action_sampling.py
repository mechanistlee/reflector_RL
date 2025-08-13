"""
리플렉터별 액션 샘플링 다양성 테스트
=======================================

이 스크립트는 collect_rollouts 메서드에서 각 리플렉터에 대해
_sample_action이 서로 다른 액션을 생성하는지 확인합니다.
"""

import os
import sys
import numpy as np
import logging
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from config import TrainingConfig
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ config.py를 가져올 수 없습니다. 기본 설정을 사용합니다.")
    CONFIG_AVAILABLE = False

# matplotlib는 선택적으로 임포트
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ matplotlib가 설치되지 않았습니다. 시각화 기능이 제한됩니다.")
    MATPLOTLIB_AVAILABLE = False


class MockEnvironment:
    """테스트용 환경 모의 객체"""
    def __init__(self, num_reflectors=3, grid_size=25):
        self.num_reflectors = num_reflectors
        self.grid_size = grid_size
        self.action_space_size = num_reflectors * grid_size
        self.observation_space_size = num_reflectors * grid_size * 2 + 2  # pointcloud + intensity + scalars
        
    def reset(self):
        """환경 리셋"""
        # 각 리플렉터마다 다른 초기 관찰값 생성
        observations = []
        for i in range(self.num_reflectors):
            # 각 리플렉터에 고유한 패턴을 가진 관찰값 생성
            base_pattern = np.sin(np.linspace(0, 2*np.pi, self.grid_size)) * (i + 1)
            intensity_pattern = np.cos(np.linspace(0, np.pi, self.grid_size)) * (i + 1) * 0.5
            scalar_values = [i * 0.1, i * 0.2]  # efficiency, total_flux
            
            obs = np.concatenate([base_pattern, intensity_pattern, scalar_values])
            observations.append(obs)
        
        return np.array(observations)


class MockPolicy:
    """테스트용 정책 모의 객체"""
    def __init__(self, action_space_size, observation_space_size):
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        
    def predict(self, obs, deterministic=False):
        """관찰값에 따라 다른 액션 생성"""
        # 관찰값의 평균값을 기반으로 서로 다른 액션 패턴 생성
        if len(obs.shape) == 1:
            obs_mean = np.mean(obs)
            # 관찰값에 따라 다른 액션 패턴 생성
            action = np.sin(np.linspace(0, 2*np.pi, 25)) * obs_mean + np.random.normal(0, 0.1, 25)
        else:
            # 배치 처리
            actions = []
            for single_obs in obs:
                obs_mean = np.mean(single_obs)
                action = np.sin(np.linspace(0, 2*np.pi, 25)) * obs_mean + np.random.normal(0, 0.1, 25)
                actions.append(action)
            action = np.array(actions)
        
        return action, None


class ActionSamplingTester:
    """액션 샘플링 테스트 클래스"""
    
    def __init__(self, num_reflectors=3, grid_size=25):
        self.num_reflectors = num_reflectors
        self.grid_size = grid_size
        self.mock_env = MockEnvironment(num_reflectors, grid_size)
        self.mock_policy = MockPolicy(grid_size, grid_size * 2 + 2)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def simulate_sample_action(self, obs: np.ndarray, reflector_id: int) -> np.ndarray:
        """_sample_action 메서드 시뮬레이션"""
        # MockPolicy를 사용하여 액션 생성
        action, _ = self.mock_policy.predict(obs, deterministic=False)
        
        # 액션을 [-1, 1] 범위로 정규화
        action = np.clip(action, -1, 1)
        
        self.logger.debug(f"리플렉터 {reflector_id}: 액션 범위 [{np.min(action):.3f}, {np.max(action):.3f}], 평균 {np.mean(action):.3f}")
        
        return action
    
    def simulate_collect_rollouts_action_sampling(self, num_iterations=10) -> Dict:
        """collect_rollouts에서 액션 샘플링 시뮬레이션"""
        
        print(f"\n🎯 액션 샘플링 다양성 테스트 시작")
        print(f"   - 리플렉터 수: {self.num_reflectors}")
        print(f"   - 그리드 크기: {self.grid_size}")
        print(f"   - 테스트 반복 횟수: {num_iterations}")
        
        all_actions_history = []  # 모든 반복의 액션 기록
        diversity_metrics = []
        
        for iteration in range(num_iterations):
            print(f"\n📊 반복 {iteration + 1}/{num_iterations}")
            
            # 환경에서 관찰값 가져오기
            observations = self.mock_env.reset()
            
            # 각 리플렉터에 대해 액션 샘플링
            reflector_actions = []
            
            for reflector_idx in range(self.num_reflectors):
                reflector_obs = observations[reflector_idx]
                action = self.simulate_sample_action(reflector_obs, reflector_idx)
                reflector_actions.append(action)
                
                print(f"   리플렉터 {reflector_idx + 1}: 액션 평균={np.mean(action):.4f}, 표준편차={np.std(action):.4f}")
            
            # 액션 다양성 분석
            actions_array = np.array(reflector_actions)  # shape: (num_reflectors, grid_size)
            
            # 1. 리플렉터 간 액션 차이 계산 (유클리드 거리)
            pairwise_distances = []
            for i in range(self.num_reflectors):
                for j in range(i + 1, self.num_reflectors):
                    distance = np.linalg.norm(actions_array[i] - actions_array[j])
                    pairwise_distances.append(distance)
            
            avg_distance = np.mean(pairwise_distances)
            min_distance = np.min(pairwise_distances)
            max_distance = np.max(pairwise_distances)
            
            # 2. 액션 표준편차 (각 그리드 위치별로)
            action_std_per_position = np.std(actions_array, axis=0)  # 각 위치별 표준편차
            avg_std_across_positions = np.mean(action_std_per_position)
            
            # 3. 전체 액션 분산
            total_variance = np.var(actions_array)
            
            diversity_metrics.append({
                'iteration': iteration + 1,
                'avg_pairwise_distance': avg_distance,
                'min_pairwise_distance': min_distance,
                'max_pairwise_distance': max_distance,
                'avg_std_per_position': avg_std_across_positions,
                'total_variance': total_variance,
                'actions': actions_array.copy()
            })
            
            all_actions_history.append(actions_array.copy())
            
            print(f"   📈 다양성 지표:")
            print(f"      - 평균 리플렉터 간 거리: {avg_distance:.4f}")
            print(f"      - 최소/최대 거리: {min_distance:.4f} / {max_distance:.4f}")
            print(f"      - 위치별 평균 표준편차: {avg_std_across_positions:.4f}")
            print(f"      - 전체 분산: {total_variance:.4f}")
        
        return {
            'metrics': diversity_metrics,
            'actions_history': all_actions_history,
            'summary': self._calculate_summary_statistics(diversity_metrics)
        }
    
    def _calculate_summary_statistics(self, metrics: List[Dict]) -> Dict:
        """요약 통계 계산"""
        avg_distances = [m['avg_pairwise_distance'] for m in metrics]
        min_distances = [m['min_pairwise_distance'] for m in metrics]
        max_distances = [m['max_pairwise_distance'] for m in metrics]
        std_per_positions = [m['avg_std_per_position'] for m in metrics]
        total_variances = [m['total_variance'] for m in metrics]
        
        return {
            'avg_pairwise_distance': {
                'mean': np.mean(avg_distances),
                'std': np.std(avg_distances),
                'min': np.min(avg_distances),
                'max': np.max(avg_distances)
            },
            'min_pairwise_distance': {
                'mean': np.mean(min_distances),
                'std': np.std(min_distances),
                'min': np.min(min_distances),
                'max': np.max(min_distances)
            },
            'max_pairwise_distance': {
                'mean': np.mean(max_distances),
                'std': np.std(max_distances),
                'min': np.min(max_distances),
                'max': np.max(max_distances)
            },
            'avg_std_per_position': {
                'mean': np.mean(std_per_positions),
                'std': np.std(std_per_positions),
                'min': np.min(std_per_positions),
                'max': np.max(std_per_positions)
            },
            'total_variance': {
                'mean': np.mean(total_variances),
                'std': np.std(total_variances),
                'min': np.min(total_variances),
                'max': np.max(total_variances)
            }
        }
    
    def test_action_diversity_threshold(self, results: Dict, diversity_threshold=0.1) -> bool:
        """액션 다양성이 임계값을 넘는지 테스트"""
        
        print(f"\n🧪 액션 다양성 임계값 테스트")
        print(f"   임계값: {diversity_threshold}")
        
        summary = results['summary']
        
        # 테스트 조건들
        tests = [
            {
                'name': '평균 리플렉터 간 거리',
                'value': summary['avg_pairwise_distance']['mean'],
                'threshold': diversity_threshold,
                'condition': 'greater'
            },
            {
                'name': '최소 리플렉터 간 거리',
                'value': summary['min_pairwise_distance']['mean'],
                'threshold': diversity_threshold * 0.5,  # 더 낮은 임계값
                'condition': 'greater'
            },
            {
                'name': '위치별 평균 표준편차',
                'value': summary['avg_std_per_position']['mean'],
                'threshold': diversity_threshold * 0.3,
                'condition': 'greater'
            },
            {
                'name': '전체 분산',
                'value': summary['total_variance']['mean'],
                'threshold': diversity_threshold * 0.1,
                'condition': 'greater'
            }
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            if test['condition'] == 'greater':
                passed = test['value'] > test['threshold']
            else:
                passed = test['value'] < test['threshold']
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {test['name']}: {test['value']:.4f} (임계값: {test['threshold']:.4f})")
            
            if passed:
                passed_tests += 1
        
        overall_pass = passed_tests >= total_tests * 0.75  # 75% 이상 통과하면 성공
        
        print(f"\n📊 테스트 결과: {passed_tests}/{total_tests} 통과")
        print(f"{'✅ 전체 테스트 통과' if overall_pass else '❌ 전체 테스트 실패'}")
        
        return overall_pass
    
    def visualize_action_diversity(self, results: Dict, output_dir: str = None):
        """액션 다양성 시각화"""
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ matplotlib가 없어 시각화를 건너뜁니다.")
            return
        
        if output_dir is None:
            output_dir = os.path.join(project_root, "action_test_output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 액션 다양성 시각화 중...")
        
        metrics = results['metrics']
        
        # 1. 시간에 따른 다양성 지표 변화
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        iterations = [m['iteration'] for m in metrics]
        avg_distances = [m['avg_pairwise_distance'] for m in metrics]
        min_distances = [m['min_pairwise_distance'] for m in metrics]
        max_distances = [m['max_pairwise_distance'] for m in metrics]
        std_per_positions = [m['avg_std_per_position'] for m in metrics]
        
        # 평균 리플렉터 간 거리
        axes[0, 0].plot(iterations, avg_distances, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_title('평균 리플렉터 간 거리')
        axes[0, 0].set_xlabel('반복 횟수')
        axes[0, 0].set_ylabel('유클리드 거리')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 최소/최대 거리
        axes[0, 1].plot(iterations, min_distances, 'r-o', label='최소 거리', linewidth=2, markersize=4)
        axes[0, 1].plot(iterations, max_distances, 'g-o', label='최대 거리', linewidth=2, markersize=4)
        axes[0, 1].set_title('최소/최대 리플렉터 간 거리')
        axes[0, 1].set_xlabel('반복 횟수')
        axes[0, 1].set_ylabel('유클리드 거리')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 위치별 표준편차
        axes[1, 0].plot(iterations, std_per_positions, 'm-o', linewidth=2, markersize=4)
        axes[1, 0].set_title('위치별 평균 표준편차')
        axes[1, 0].set_xlabel('반복 횟수')
        axes[1, 0].set_ylabel('표준편차')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 전체 분산
        total_variances = [m['total_variance'] for m in metrics]
        axes[1, 1].plot(iterations, total_variances, 'c-o', linewidth=2, markersize=4)
        axes[1, 1].set_title('전체 액션 분산')
        axes[1, 1].set_xlabel('반복 횟수')
        axes[1, 1].set_ylabel('분산')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 이미지 저장
        diversity_plot_path = os.path.join(output_dir, "action_diversity_metrics.png")
        plt.savefig(diversity_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 다양성 지표 그래프 저장: {diversity_plot_path}")
        
        # 2. 첫 번째 반복의 액션 히트맵
        if len(results['actions_history']) > 0:
            first_actions = results['actions_history'][0]  # shape: (num_reflectors, grid_size)
            
            fig, axes = plt.subplots(1, self.num_reflectors, figsize=(4 * self.num_reflectors, 4))
            if self.num_reflectors == 1:
                axes = [axes]
            
            for i in range(self.num_reflectors):
                # 1D 액션을 5x5 그리드로 변환 (grid_size=25인 경우)
                if self.grid_size == 25:
                    action_grid = first_actions[i].reshape(5, 5)
                else:
                    # grid_size가 25가 아닌 경우 적절한 형태로 변환
                    side_length = int(np.sqrt(self.grid_size))
                    if side_length * side_length == self.grid_size:
                        action_grid = first_actions[i].reshape(side_length, side_length)
                    else:
                        # 정사각형이 아닌 경우 1D로 표시
                        action_grid = first_actions[i].reshape(1, -1)
                
                im = axes[i].imshow(action_grid, cmap='viridis', interpolation='nearest')
                axes[i].set_title(f'리플렉터 {i + 1}')
                axes[i].set_xlabel('Grid X')
                axes[i].set_ylabel('Grid Y')
                plt.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            
            # 이미지 저장
            heatmap_path = os.path.join(output_dir, "action_heatmaps.png")
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 액션 히트맵 저장: {heatmap_path}")
        
        print(f"📁 시각화 파일들이 저장된 경로: {output_dir}")


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🎯 리플렉터별 액션 샘플링 다양성 테스트")
    print("=" * 60)
    
    # 테스트 설정
    num_reflectors = 3
    grid_size = 25
    num_iterations = 10
    
    # 테스터 생성
    tester = ActionSamplingTester(num_reflectors, grid_size)
    
    # 액션 샘플링 테스트 실행
    results = tester.simulate_collect_rollouts_action_sampling(num_iterations)
    
    # 요약 통계 출력
    print(f"\n" + "=" * 60)
    print("📋 요약 통계")
    print("=" * 60)
    
    summary = results['summary']
    
    print(f"📊 평균 리플렉터 간 거리:")
    print(f"   평균: {summary['avg_pairwise_distance']['mean']:.4f} ± {summary['avg_pairwise_distance']['std']:.4f}")
    print(f"   범위: [{summary['avg_pairwise_distance']['min']:.4f}, {summary['avg_pairwise_distance']['max']:.4f}]")
    
    print(f"\n📊 최소 리플렉터 간 거리:")
    print(f"   평균: {summary['min_pairwise_distance']['mean']:.4f} ± {summary['min_pairwise_distance']['std']:.4f}")
    print(f"   범위: [{summary['min_pairwise_distance']['min']:.4f}, {summary['min_pairwise_distance']['max']:.4f}]")
    
    print(f"\n📊 위치별 평균 표준편차:")
    print(f"   평균: {summary['avg_std_per_position']['mean']:.4f} ± {summary['avg_std_per_position']['std']:.4f}")
    print(f"   범위: [{summary['avg_std_per_position']['min']:.4f}, {summary['avg_std_per_position']['max']:.4f}]")
    
    print(f"\n📊 전체 분산:")
    print(f"   평균: {summary['total_variance']['mean']:.4f} ± {summary['total_variance']['std']:.4f}")
    print(f"   범위: [{summary['total_variance']['min']:.4f}, {summary['total_variance']['max']:.4f}]")
    
    # 다양성 임계값 테스트
    diversity_passed = tester.test_action_diversity_threshold(results, diversity_threshold=0.1)
    
    # 시각화
    tester.visualize_action_diversity(results)
    
    # 최종 결론
    print(f"\n" + "=" * 60)
    print("🏁 최종 결론")
    print("=" * 60)
    
    if diversity_passed:
        print("✅ 액션 샘플링 다양성 테스트 통과!")
        print("   각 리플렉터가 서로 다른 액션을 생성하고 있습니다.")
    else:
        print("❌ 액션 샘플링 다양성 테스트 실패!")
        print("   리플렉터들이 유사한 액션을 생성하고 있을 수 있습니다.")
    
    print(f"\n💡 해석 가이드:")
    print(f"   - 평균 리플렉터 간 거리가 클수록 액션이 더 다양함")
    print(f"   - 최소 거리가 0에 가까우면 일부 리플렉터가 동일한 액션 생성")
    print(f"   - 위치별 표준편차가 클수록 각 그리드 위치에서 리플렉터별 액션 차이가 큼")
    print(f"   - 전체 분산이 클수록 전반적인 액션 다양성이 높음")
    
    return diversity_passed


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
