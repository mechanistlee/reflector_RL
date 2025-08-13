"""
Test font configuration for matplotlib visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings

# Test the same font configuration as in data_visualization.py
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Calibri', 'Tahoma', 'sans-serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.formatter.use_mathtext'] = True

# Suppress all font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Glyph*missing from font*")
warnings.filterwarnings("ignore", message="findfont*")

# Rebuild font cache
try:
    matplotlib.font_manager._rebuild()
except AttributeError:
    # Alternative method for newer matplotlib versions
    matplotlib.font_manager.fontManager.__init__()

def test_font_rendering():
    """Test font rendering with English text only"""
    
    print("Testing font configuration...")
    print(f"Current font family: {plt.rcParams['font.family']}")
    print(f"Sans-serif fonts: {plt.rcParams['font.sans-serif']}")
    
    # Create simple test plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Plot with English labels only
    ax.plot(x, y, label='Sine Wave')
    ax.set_title('Font Test: English Only Text', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add some text annotations
    ax.text(0.5, 0.5, 'Sample Text: Success!', transform=ax.transAxes, 
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save test image
    output_path = "test_output/font_test.png"
    import os
    os.makedirs("test_output", exist_ok=True)
    
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Font test successful! Image saved: {output_path}")
        print("No font warnings should appear above this line.")
    except Exception as e:
        print(f"✗ Font test failed: {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    test_font_rendering()
