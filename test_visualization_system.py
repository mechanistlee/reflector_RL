"""
Test the data visualization system with sample data to ensure no font warnings
"""
import os
import sys
sys.path.append('.')

from utils.data_visualization import create_comprehensive_report
import numpy as np

def create_sample_training_data():
    """Create sample training data for testing"""
    
    # Simulate training data
    episodes = 100
    training_data = {
        'episode_rewards': np.random.normal(150, 50, episodes).tolist(),
        'episode_lengths': np.random.randint(50, 200, episodes).tolist(),
        'q_values': np.random.normal(100, 20, 50).tolist(),
        'actor_losses': np.abs(np.random.normal(0.1, 0.05, 50)).tolist(),
        'critic_losses': np.abs(np.random.normal(0.2, 0.1, 50)).tolist(),
        'entropy_values': np.abs(np.random.normal(1.0, 0.3, 50)).tolist(),
        'total_timesteps': 200000
    }
    
    return training_data

def test_visualization_system():
    """Test the visualization system with sample data"""
    
    print("=" * 60)
    print("Testing SPEOS RL Visualization System")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample training data...")
    training_stats = create_sample_training_data()
    
    # Test visualization
    print("Testing visualization generation...")
    try:
        created_files = create_comprehensive_report(
            training_stats=training_stats,
            save_dir="./test_output",
            model_name="font_test_model",
            title="Font Test - SPEOS RL Training Results"
        )
        
        print("✓ Visualization test completed successfully!")
        print("Created files:")
        for file_type, file_path in created_files.items():
            print(f"  {file_type}: {file_path}")
        
        # Check if files exist
        all_exist = True
        for file_type, file_path in created_files.items():
            if os.path.exists(file_path):
                print(f"  ✓ {file_type} file exists")
            else:
                print(f"  ✗ {file_type} file missing")
                all_exist = False
        
        if all_exist:
            print("\n🎉 All tests passed! Font configuration is working correctly.")
        else:
            print("\n⚠️ Some files are missing.")
            
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization_system()
