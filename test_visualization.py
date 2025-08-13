#!/usr/bin/env python3
"""
데이터 시각화 시스템 테스트 스크립트
"""

import sys
import os
import numpy as np

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.data_visualization import TrainingVisualizer, create_comprehensive_report
    print("✓ data_visualization 모듈 임포트 성공")
except ImportError as e:
    print(f"✗ data_visualization 모듈 임포트 실패: {e}")
    sys.exit(1)

def create_test_data():
    """테스트용 가짜 훈련 데이터 생성"""
    print("테스트 데이터 생성 중...")
    
    # 에피소드 수
    num_episodes = 200
    
    # 가짜 리워드 데이터 (학습 진행에 따라 점진적 향상)
    base_reward = -400
    improvement = np.linspace(0, 600, num_episodes)
    noise = np.random.normal(0, 50, num_episodes)
    episode_rewards = base_reward + improvement + noise
    
    # 가짜 에피소드 길이 (점진적 개선)
    base_length = 200
    length_improvement = np.linspace(100, -50, num_episodes)
    length_noise = np.random.normal(0, 20, num_episodes)
    episode_lengths = np.maximum(50, base_length + length_improvement + length_noise).astype(int)
    
    # 고급 메트릭 (SAC 관련)
    num_training_steps = 100  # 100번의 훈련 스텝
    q_values = np.random.normal(-50, 20, num_training_steps) + np.linspace(0, 30, num_training_steps)
    actor_losses = np.random.exponential(0.5, num_training_steps)
    critic_losses = np.random.exponential(1.0, num_training_steps)
    entropy_values = np.random.normal(1.5, 0.3, num_training_steps)
    
    training_stats = {
        'episode_rewards': episode_rewards.tolist(),
        'episode_lengths': episode_lengths.tolist(),
        'total_timesteps': 20000,
        'q_values': q_values.tolist(),
        'actor_losses': actor_losses.tolist(),
        'critic_losses': critic_losses.tolist(),
        'entropy_values': entropy_values.tolist()
    }
    
    print(f"테스트 데이터 생성 완료:")
    print(f"  - 에피소드: {len(episode_rewards)}개")
    print(f"  - 훈련 메트릭: {len(q_values)}개")
    print(f"  - 리워드 범위: {min(episode_rewards):.2f} ~ {max(episode_rewards):.2f}")
    
    return training_stats

def test_visualizer():
    """TrainingVisualizer 테스트"""
    print("\n" + "="*60)
    print("TrainingVisualizer 테스트 시작")
    print("="*60)
    
    # 테스트 데이터 생성
    training_stats = create_test_data()
    
    # 시각화 객체 생성
    print("\nTrainingVisualizer 객체 생성 중...")
    visualizer = TrainingVisualizer()
    print("✓ TrainingVisualizer 생성 성공")
    
    # 통계 처리
    print("\n통계 데이터 처리 중...")
    processed_stats = visualizer.process_training_stats(training_stats)
    print("✓ 통계 처리 성공")
    
    # 출력 파일 경로 설정
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, "test_training_visualization.png")
    json_path = os.path.join(output_dir, "test_training_stats.json")
    
    # 통합 시각화 생성
    print(f"\n통합 시각화 생성 중...")
    print(f"PNG 경로: {png_path}")
    print(f"JSON 경로: {json_path}")
    
    try:
        created_files = visualizer.create_unified_output(
            processed_stats=processed_stats,
            output_png=png_path,
            output_json=json_path,
            title="SPEOS RL 테스트 시각화"
        )
        
        print("✓ 통합 시각화 생성 성공!")
        print(f"생성된 파일:")
        for file_type, file_path in created_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  {file_type}: {file_path} ({file_size:,} bytes)")
            else:
                print(f"  {file_type}: {file_path} (파일 없음)")
                
    except Exception as e:
        print(f"✗ 통합 시각화 생성 실패: {e}")
        import traceback
        print(f"오류 상세:\n{traceback.format_exc()}")
        return False
    
    return True

def test_comprehensive_report():
    """create_comprehensive_report 함수 테스트"""
    print("\n" + "="*60)
    print("create_comprehensive_report 함수 테스트")
    print("="*60)
    
    # 테스트 데이터 생성
    training_stats = create_test_data()
    
    # 출력 디렉토리 설정
    output_dir = "./test_output"
    
    print(f"\n종합 리포트 생성 중...")
    print(f"출력 디렉토리: {output_dir}")
    
    try:
        created_files = create_comprehensive_report(
            training_stats=training_stats,
            save_dir=output_dir,
            model_name="test_model",
            title="SPEOS RL 종합 테스트 리포트"
        )
        
        print("✓ 종합 리포트 생성 성공!")
        print(f"생성된 파일:")
        for file_type, file_path in created_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  {file_type}: {file_path} ({file_size:,} bytes)")
            else:
                print(f"  {file_type}: {file_path} (파일 없음)")
                
        return True
        
    except Exception as e:
        print(f"✗ 종합 리포트 생성 실패: {e}")
        import traceback
        print(f"오류 상세:\n{traceback.format_exc()}")
        return False

def main():
    """메인 테스트 함수"""
    print("SPEOS RL 데이터 시각화 시스템 테스트")
    print("="*60)
    
    # 테스트 결과 추적
    test_results = []
    
    # 개별 테스트 실행
    test_results.append(("TrainingVisualizer", test_visualizer()))
    test_results.append(("comprehensive_report", test_comprehensive_report()))
    
    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n총 테스트: {total}개")
    print(f"성공: {passed}개")
    print(f"실패: {total - passed}개")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과!")
        return True
    else:
        print(f"\n❌ {total - passed}개 테스트 실패")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
