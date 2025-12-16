"""
Quick GPU test for XGBoost
"""
import xgboost as xgb
import numpy as np

print("Testing XGBoost GPU support...")
print("-" * 50)

try:
    # Create small test dataset
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # Try GPU mode
    print("\nAttempting GPU mode...")
    model = xgb.XGBClassifier(
        n_estimators=10,
        tree_method='hist',
        device='cuda:0'
    )
    
    model.fit(X, y)
    score = model.score(X, y)
    
    print(f"✓ GPU MODE WORKING!")
    print(f"  Test accuracy: {score:.4f}")
    print(f"  Device: {model.get_params()['device']}")
    print(f"  Tree method: {model.get_params()['tree_method']}")
    
except Exception as e:
    print(f"✗ GPU mode failed: {e}")
    print("\nAttempting CPU mode...")
    
    try:
        model = xgb.XGBClassifier(
            n_estimators=10,
            tree_method='hist'
        )
        model.fit(X, y)
        score = model.score(X, y)
        print(f"✓ CPU mode working")
        print(f"  Test accuracy: {score:.4f}")
    except Exception as e2:
        print(f"✗ CPU mode also failed: {e2}")
