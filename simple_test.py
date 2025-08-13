import sys
import os
sys.path.append('.')

try:
    from utils.data_visualization import TrainingVisualizer
    print("SUCCESS: TrainingVisualizer import successful")
    
    # 간단한 인스턴스 생성 테스트
    visualizer = TrainingVisualizer()
    print("SUCCESS: TrainingVisualizer instance created")
    
    # 메서드 존재 확인
    if hasattr(visualizer, 'create_unified_output'):
        print("SUCCESS: create_unified_output method exists")
    else:
        print("ERROR: create_unified_output method missing")
        
    if hasattr(visualizer, '_create_advanced_3x3_visualization'):
        print("SUCCESS: _create_advanced_3x3_visualization method exists")
    else:
        print("ERROR: _create_advanced_3x3_visualization method missing")
        
    print("All basic tests passed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
"이것도 바뀌나?"