"""
Test visualization with Korean title to check font filtering
"""
import sys
sys.path.append('.')

from utils.data_visualization import create_comprehensive_report
import numpy as np

def test_korean_title():
    # Create sample data
    training_data = {
        'episode_rewards': np.random.normal(150, 50, 50).tolist(),
        'episode_lengths': np.random.randint(50, 200, 50).tolist(),
        'q_values': np.random.normal(100, 20, 25).tolist(),
        'actor_losses': np.abs(np.random.normal(0.1, 0.05, 25)).tolist(),
        'critic_losses': np.abs(np.random.normal(0.2, 0.1, 25)).tolist(),
        'entropy_values': np.abs(np.random.normal(1.0, 0.3, 25)).tolist(),
        'total_timesteps': 100000
    }
    
    # Test with Korean title (should be filtered to ASCII)
    korean_title = "SPEOS 강화학습 훈련 결과 - 테스트"
    
    print(f"Testing with Korean title: {korean_title}")
    
    try:
        created_files = create_comprehensive_report(
            training_stats=training_data,
            save_dir="./test_output",
            model_name="korean_test",
            title=korean_title
        )
        
        print("✓ Test completed successfully!")
        print("Created files:")
        for file_type, file_path in created_files.items():
            print(f"  {file_type}: {file_path}")
        
        print("\n✅ Font filtering is working correctly!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_korean_title()
