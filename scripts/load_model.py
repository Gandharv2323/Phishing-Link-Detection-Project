"""
Load existing HSEF model with better error handling
"""

import pickle
import json
from pathlib import Path

print("\n" + "="*80)
print(" "*25 + "LOADING EXISTING MODEL")
print("="*80)

# Check model info
info_path = Path('models/model_info.json')
with open(info_path, 'r') as f:
    model_info = json.load(f)

print("\nModel Information:")
print(f"  Type: {model_info['model_type']}")
print(f"  Base Learners: {', '.join(model_info['base_learners'])}")
print(f"  Meta Learner: {model_info['meta_learner']}")
print(f"  Test Accuracy: {model_info['test_accuracy']*100:.2f}%")
print(f"  GPU Enabled: {model_info['gpu_enabled']}")
print(f"  Trained: {model_info['timestamp']}")

# Try loading model with different protocol
model_path = Path('models/hsef_model.pkl')
scaler_path = Path('models/hsef_scaler.pkl')

print("\nAttempting to load model...")
print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

try:
    # Try with protocol 5 (Python 3.8+)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
    
except Exception as e1:
    print(f"⚠ Failed with default protocol: {e1}")
    
    try:
        # Try with encoding
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        print("✓ Model loaded with latin1 encoding!")
        
    except Exception as e2:
        print(f"⚠ Failed with latin1: {e2}")
        print("\n❌ Model file may be corrupted or incompatible")
        print("\nPossible reasons:")
        print("  1. Model was saved with a different Python version")
        print("  2. Model file is corrupted")
        print("  3. Incomplete save/write operation")
        print("\nSolution: Retrain the model using hsef_model.py")
        exit(1)

try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully!")
except Exception as e:
    print(f"⚠ Failed to load scaler: {e}")

# Test model
print("\nModel Details:")
print(f"  Type: {type(model).__name__}")

if hasattr(model, 'estimators_'):
    print(f"  Number of base estimators: {len(model.estimators_)}")
    
if hasattr(model, 'final_estimator_'):
    print(f"  Meta-learner: {type(model.final_estimator_).__name__}")

print("\n" + "="*80)
print("✓ MODEL LOADED SUCCESSFULLY")
print("="*80)
print(f"\nThis model has {model_info['test_accuracy']*100:.2f}% accuracy!")
print("\nYou can now:")
print("  1. Use it in your web app (app.py)")
print("  2. Test it with test_existing_model.py")
print("  3. Deploy it to production")
