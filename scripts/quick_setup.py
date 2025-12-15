"""
Quick model creation for web deployment
Creates a lightweight model from the already trained results
"""

from hsef_model import HSEFModel
import pandas as pd
import joblib
import json
from pathlib import Path
import numpy as np

print("\n" + "="*70)
print("QUICK MODEL SETUP FOR WEB DEPLOYMENT")
print("="*70 + "\n")

# Initialize HSEF with fast mode
print("Initializing HSEF model in FAST mode...")
hsef = HSEFModel(fast_mode=True, use_gpu=True)

print("Loading dataset...")
df = pd.read_csv('All.csv')
print(f"✓ Dataset loaded: {len(df)} samples\n")

# Load and preprocess
print("Loading and preprocessing data...")
hsef.load_data('All.csv')
print("✓ Data loaded and preprocessed\n")

# Build base learners
print("Building base learners...")
hsef.build_base_learners()
print("✓ Base learners configured\n")

# Train on full dataset (fast mode)
print("Training models (FAST MODE - this will be quick)...")
print("  - Random Forest (10 trees)...")
hsef.rf_model.fit(hsef.X_train, hsef.y_train)
print("  ✓ Random Forest trained")

print("  - XGBoost (50 estimators with GPU)...")
hsef.xgb_model.fit(hsef.X_train, hsef.y_train)
print("  ✓ XGBoost trained")

print("  - SVM (LinearSVC)...")
hsef.svm_model.fit(hsef.X_train, hsef.y_train)
print("  ✓ SVM trained\n")

# Train stacking ensemble
print("Building stacking ensemble architecture...")
hsef.build_stacking_ensemble()
print("Training stacking ensemble...")
hsef.train_stacking_ensemble()
print("✓ Stacking ensemble trained\n")

# Evaluate quickly
print("Quick evaluation on test set...")
from sklearn.metrics import accuracy_score
y_pred = hsef.stacking_model.predict(hsef.X_test)
accuracy = accuracy_score(hsef.y_test, y_pred)
print(f"✓ Test Accuracy: {accuracy:.4f}\n")

# Save for deployment
print("Saving models for web deployment...")
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# Save stacking model
model_path = models_dir / 'hsef_model.pkl'
joblib.dump(hsef.stacking_model, model_path)
print(f"✓ Saved: {model_path}")

# Save scaler
scaler_path = models_dir / 'hsef_scaler.pkl'
joblib.dump(hsef.scaler, scaler_path)
print(f"✓ Saved: {scaler_path}")

# Save feature names
features_path = models_dir / 'feature_names.json'
features_data = {
    'features': hsef.feature_names,
    'classes': hsef.class_names.tolist()
}
with open(features_path, 'w') as f:
    json.dump(features_data, f, indent=2)
print(f"✓ Saved: {features_path}")

# Save model info
info_path = models_dir / 'model_info.json'
model_info = {
    'model_type': 'HSEF - Heterogeneous Stacking Ensemble Framework',
    'base_learners': ['Random Forest', 'XGBoost', 'SVM'],
    'meta_learner': 'Logistic Regression',
    'n_features': len(hsef.feature_names),
    'n_classes': len(hsef.class_names),
    'classes': hsef.class_names.tolist(),
    'gpu_enabled': hsef.gpu_available,
    'fast_mode': True,
    'test_accuracy': float(accuracy),
    'timestamp': pd.Timestamp.now().isoformat()
}
with open(info_path, 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"✓ Saved: {info_path}")

print("\n" + "="*70)
print("✅ MODEL READY FOR WEB DEPLOYMENT!")
print("="*70)
print(f"\nModel files saved to: {models_dir.absolute()}")
print(f"Test accuracy: {accuracy:.2%}")
print("\n🚀 Next step: Start the web application")
print("   python app.py")
print("\n   Then open: http://127.0.0.1:5000")
print("="*70 + "\n")
