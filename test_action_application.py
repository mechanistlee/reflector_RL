"""
액션 적용 테스트 스크립트
=======================

이 스크립트는 각 리플렉터에 서로 다른 액션이 제대로 적용되는지 확인합니다.
STL 파일 생성 전후의 포인트클라우드 변화를 분석합니다.
"""

import os
import sys
import numpy as np
import logging
from typing import List, Dict, Tuple

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from config import TrainingConfig
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ config.py를 가져올 수 없습니다. 기본 설정을 사용합니다.")
    CONFIG_AVAILABLE = False

try:
    from env.env_speos_v1 import SpeosEnv, ReflectorClass
    ENV_AVAILABLE = True
except ImportError:
    print("⚠️ env_speos_v1.py를 가져올 수 없습니다.")
    ENV_AVAILABLE = False


class ActionApplicationTester:
    """액션 적용 테스트 클래스"""
    
    def __init__(self, config=None):
        if config is None:
            if CONFIG_AVAILABLE:
                self.config = TrainingConfig()
            else:
                # 기본 설정
                class DefaultConfig:
                    def __init__(self):
                        self.num_reflectors = 3
                        self.grid_rows = 5
                        self.grid_cols = 5
                        self.action_size = 25
                        self.grid_cell_size_x = 1.0
                        self.grid_cell_size_y = 1.0
                        self.z_min = -5.0
                        self.z_max = 5.0
                    
                    def get_reflector_position(self, reflector_id):
                        positions = [
                            (0.0, 0.0, 0.0),
                            (20.0, 0.0, 0.0),
                            (0.0, 20.0, 0.0),
                        ]
                        return positions[reflector_id % len(positions)]
                
                self.config = DefaultConfig()
        else:
            self.config = config
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def create_test_reflectors(self) -> List[ReflectorClass]:
        """테스트용 리플렉터 객체들 생성"""
        reflectors = []
        
        for i in range(self.config.num_reflectors):
            reflector = ReflectorClass(reflector_id=i, config=self.config)
            reflector._initialize_Reflector()
            reflectors.append(reflector)
            
            print(f"🔧 리플렉터 {i+1} 생성:")
            print(f"   - 중심 위치: {reflector.center_position}")
            print(f"   - 그리드 크기: {self.config.grid_rows}×{self.config.grid_cols}")
            if reflector.pointcloud_s1 is not None:
                print(f"   - 포인트클라우드: {reflector.pointcloud_s1.shape}")
                print(f"   - Z 범위: [{np.min(reflector.pointcloud_s1[:, 2]):.3f}, {np.max(reflector.pointcloud_s1[:, 2]):.3f}]")
        
        return reflectors
    
    def generate_test_actions(self) -> np.ndarray:
        """각 리플렉터별로 서로 다른 테스트 액션 생성"""
        action_size_per_reflector = self.config.grid_rows * self.config.grid_cols
        total_action_size = self.config.num_reflectors * action_size_per_reflector
        
        all_actions = []
        
        for i in range(self.config.num_reflectors):
            # 각 리플렉터마다 다른 패턴의 액션 생성
            if i == 0:
                # 리플렉터 1: 중심이 높은 돔 형태
                action = np.ones(action_size_per_reflector) * 2.0
                center_idx = action_size_per_reflector // 2
                action[center_idx] = 4.0  # 중심을 더 높게
            elif i == 1:
                # 리플렉터 2: 사인 파형
                action = np.sin(np.linspace(0, 2*np.pi, action_size_per_reflector)) * 3.0
            elif i == 2:
                # 리플렉터 3: 계단 형태
                action = np.ones(action_size_per_reflector) * -1.0
                half = action_size_per_reflector // 2
                action[half:] = 1.0
            else:
                # 추가 리플렉터: 랜덤 패턴
                np.random.seed(i)
                action = np.random.uniform(-2.0, 2.0, action_size_per_reflector)
            
            # 액션 범위 제한
            action = np.clip(action, self.config.z_min, self.config.z_max)
            all_actions.extend(action)
            
            print(f"🎯 리플렉터 {i+1} 액션 생성:")
            print(f"   - 패턴: {'돔' if i == 0 else '사인파' if i == 1 else '계단' if i == 2 else '랜덤'}")
            print(f"   - 범위: [{np.min(action):.3f}, {np.max(action):.3f}]")
            print(f"   - 평균: {np.mean(action):.3f}")
        
        return np.array(all_actions)
    
    def apply_actions_to_reflectors(self, reflectors: List[ReflectorClass], actions: np.ndarray) -> Dict:
        """액션을 리플렉터들에 적용하고 변화 기록"""
        
        print(f"\n🔄 액션 적용 시작...")
        print(f"   - 총 액션 크기: {len(actions)}")
        print(f"   - 리플렉터 수: {len(reflectors)}")
        
        action_size_per_reflector = self.config.grid_rows * self.config.grid_cols
        before_states = []
        after_states = []
        
        # 적용 전 상태 기록
        for i, reflector in enumerate(reflectors):
            if reflector.pointcloud_s1 is not None:
                before_z = reflector.pointcloud_s1[:, 2].copy()
                before_states.append(before_z)
                print(f"   리플렉터 {i+1} 적용 전 Z 범위: [{np.min(before_z):.3f}, {np.max(before_z):.3f}]")
            else:
                before_states.append(None)
        
        # 액션 적용
        for i, reflector in enumerate(reflectors):
            start_idx = i * action_size_per_reflector
            end_idx = (i + 1) * action_size_per_reflector
            
            if end_idx <= len(actions):
                reflector_action = actions[start_idx:end_idx]
                reflector._apply_actions(reflector_action)
                
                print(f"✅ 리플렉터 {i+1}: 액션 [{start_idx}:{end_idx}] 적용")
                print(f"   - 액션 범위: [{np.min(reflector_action):.3f}, {np.max(reflector_action):.3f}]")
            else:
                print(f"❌ 리플렉터 {i+1}: 액션 크기 부족")
        
        # 적용 후 상태 기록
        for i, reflector in enumerate(reflectors):
            if reflector.pointcloud_s1 is not None:
                after_z = reflector.pointcloud_s1[:, 2].copy()
                after_states.append(after_z)
                print(f"   리플렉터 {i+1} 적용 후 Z 범위: [{np.min(after_z):.3f}, {np.max(after_z):.3f}]")
            else:
                after_states.append(None)
        
        return {
            'before_states': before_states,
            'after_states': after_states,
            'action_size_per_reflector': action_size_per_reflector
        }
    
    def analyze_action_effects(self, states: Dict) -> Dict:
        """액션 효과 분석"""
        
        print(f"\n📊 액션 효과 분석...")
        
        before_states = states['before_states']
        after_states = states['after_states']
        
        analysis = {
            'reflector_changes': [],
            'total_change': 0.0,
            'max_change': 0.0,
            'all_different': True
        }
        
        for i in range(len(before_states)):
            if before_states[i] is not None and after_states[i] is not None:
                before = before_states[i]
                after = after_states[i]
                
                # 변화량 계산
                change = after - before
                total_change = np.sum(np.abs(change))
                max_change = np.max(np.abs(change))
                mean_change = np.mean(change)
                std_change = np.std(change)
                
                reflector_analysis = {
                    'reflector_id': i + 1,
                    'total_absolute_change': total_change,
                    'max_absolute_change': max_change,
                    'mean_change': mean_change,
                    'std_change': std_change,
                    'changed': total_change > 1e-6
                }
                
                analysis['reflector_changes'].append(reflector_analysis)
                analysis['total_change'] += total_change
                analysis['max_change'] = max(analysis['max_change'], max_change)
                
                print(f"   리플렉터 {i+1}:")
                print(f"      - 총 변화량: {total_change:.6f}")
                print(f"      - 최대 변화: {max_change:.6f}")
                print(f"      - 평균 변화: {mean_change:.6f}")
                print(f"      - 변화 여부: {'예' if reflector_analysis['changed'] else '아니오'}")
        
        # 리플렉터 간 차이 분석
        differences = []
        for i in range(len(after_states)):
            for j in range(i + 1, len(after_states)):
                if after_states[i] is not None and after_states[j] is not None:
                    diff = np.linalg.norm(after_states[i] - after_states[j])
                    differences.append(diff)
                    print(f"   리플렉터 {i+1} vs {j+1} 차이: {diff:.6f}")
        
        if differences:
            analysis['min_difference'] = np.min(differences)
            analysis['max_difference'] = np.max(differences)
            analysis['mean_difference'] = np.mean(differences)
            analysis['all_different'] = analysis['min_difference'] > 1e-3
        
        return analysis
    
    def test_action_application(self):
        """전체 액션 적용 테스트"""
        
        print("=" * 60)
        print("🧪 액션 적용 테스트")
        print("=" * 60)
        
        if not ENV_AVAILABLE:
            print("❌ 환경 모듈을 가져올 수 없어 테스트를 중단합니다.")
            return False
        
        # 1. 테스트용 리플렉터 생성
        print(f"\n📋 1단계: 테스트 리플렉터 생성")
        reflectors = self.create_test_reflectors()
        
        # 2. 테스트 액션 생성
        print(f"\n📋 2단계: 테스트 액션 생성")
        test_actions = self.generate_test_actions()
        
        # 3. 액션 적용
        print(f"\n📋 3단계: 액션 적용")
        states = self.apply_actions_to_reflectors(reflectors, test_actions)
        
        # 4. 효과 분석
        print(f"\n📋 4단계: 효과 분석")
        analysis = self.analyze_action_effects(states)
        
        # 5. 결과 판정
        print(f"\n📋 5단계: 결과 판정")
        
        success_criteria = {
            'actions_applied': analysis['total_change'] > 0.1,
            'different_results': analysis.get('all_different', False),
            'no_failures': all(change['changed'] for change in analysis['reflector_changes'])
        }
        
        print(f"   ✅ 액션 적용됨: {'통과' if success_criteria['actions_applied'] else '실패'}")
        print(f"   ✅ 서로 다른 결과: {'통과' if success_criteria['different_results'] else '실패'}")
        print(f"   ✅ 모든 리플렉터 변화: {'통과' if success_criteria['no_failures'] else '실패'}")
        
        overall_success = all(success_criteria.values())
        
        print(f"\n🏁 최종 결과: {'✅ 성공' if overall_success else '❌ 실패'}")
        
        if overall_success:
            print("   각 리플렉터에 서로 다른 액션이 올바르게 적용되었습니다!")
        else:
            print("   액션 적용에 문제가 있습니다.")
            
        # 상세 정보 출력
        print(f"\n📊 상세 정보:")
        print(f"   - 총 변화량: {analysis['total_change']:.6f}")
        print(f"   - 최대 변화량: {analysis['max_change']:.6f}")
        if 'mean_difference' in analysis:
            print(f"   - 리플렉터 간 평균 차이: {analysis['mean_difference']:.6f}")
            print(f"   - 리플렉터 간 최소 차이: {analysis['min_difference']:.6f}")
        
        return overall_success


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🎯 액션 적용 확인 테스트")
    print("=" * 60)
    
    # 테스터 생성
    tester = ActionApplicationTester()
    
    # 테스트 실행
    success = tester.test_action_application()
    
    print(f"\n" + "=" * 60)
    print("🏁 테스트 완료")
    print("=" * 60)
    
    if success:
        print("✅ 모든 테스트 통과!")
        print("   액션이 각 리플렉터에 올바르게 적용되고 있습니다.")
    else:
        print("❌ 테스트 실패!")
        print("   액션 적용에 문제가 있을 수 있습니다.")
    
    return success


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
