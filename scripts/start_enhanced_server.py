"""
Start the HSEF Web Application Server
Updated version with real feature extraction
"""

import subprocess
import sys

if __name__ == '__main__':
    print("="*70)
    print("HSEF Web Application - Starting Server")
    print("="*70)
    print("\n✨ NEW FEATURES:")
    print("  • Real URL feature extraction (80 features)")
    print("  • SHAP-based interpretability")
    print("  • Base model predictions display")
    print("  • Meta-layer fusion weights")
    print("  • Feature importance analysis")
    print("\n" + "="*70)
    print("Starting Flask server at http://127.0.0.1:5000")
    print("Press CTRL+C to stop")
    print("="*70 + "\n")
    
    # Run the app
    subprocess.run([sys.executable, 'app.py'])
