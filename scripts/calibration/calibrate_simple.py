"""
ULTRA-SIMPLE Calibration - Single-threaded, minimal memory
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

print("\n" + "="*80)
print(" "*20 + "SIMPLE HSEF CALIBRATION")
print("="*80)
print("\nSingle-threaded to avoid memory issues\n")

# Create output dir
output_dir = Path('publication_outputs/models')
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv('All.csv')
X = df.drop('URL_Type_obf_Type', axis=1)
y = df['URL_Type_obf_Type']
print(f"✓ Loaded {len(df)} samples")

# Handle missing/inf
X = X.fillna(0).replace([np.inf, -np.inf], 0)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"✓ Split: {len(X_train)} train, {len(X_test)} test")

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✓ Data scaled")

# Build model (n_jobs=1 to avoid memory issues)
print("\nBuilding HSEF model (single-threaded)...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=1  # Single thread
)

if XGB_AVAILABLE:
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        tree_method='hist',
        device='cpu',
        random_state=42,
        n_jobs=1  # Single thread
    )
    print("  Using XGBoost (CPU)")
else:
    xgb_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=1
    )
    print("  XGBoost not available, using RF")

svm = SVC(
    kernel='rbf',
    probability=True,
    random_state=42
)

meta_lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=1  # Single thread
)

estimators = [
    ('rf', rf),
    ('xgb', xgb_model),
    ('svm', svm)
]

# Baseline model
print("\nTraining baseline model (may take 5-8 minutes)...")
baseline_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_lr,
    cv=3,  # Reduced from 5 to save time
    n_jobs=1  # Single thread - CRITICAL for stability
)

baseline_model.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))
print(f"✓ Baseline trained: {baseline_acc*100:.2f}% accuracy")

# Calibrated model
print("\nTraining calibrated model (may take 3-5 minutes)...")
calibrated_model = CalibratedClassifierCV(
    baseline_model,
    method='sigmoid',
    cv=3,  # Reduced from 5
    n_jobs=1  # Single thread - CRITICAL for stability
)

calibrated_model.fit(X_train, y_train)
calibrated_acc = accuracy_score(y_test, calibrated_model.predict(X_test))
print(f"✓ Calibrated trained: {calibrated_acc*100:.2f}% accuracy")

# Save models
print("\nSaving models...")
with open(output_dir / 'hsef_baseline_model.pkl', 'wb') as f:
    pickle.dump(baseline_model, f)
print("  ✓ Saved baseline model")

with open(output_dir / 'hsef_calibrated_model.pkl', 'wb') as f:
    pickle.dump(calibrated_model, f)
print("  ✓ Saved calibrated model")

with open(output_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  ✓ Saved scaler")

with open(output_dir / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("  ✓ Saved label encoder")

# Generate report
print("\nGenerating report...")
report_dir = Path('publication_outputs/reports')
report_dir.mkdir(parents=True, exist_ok=True)

y_pred = calibrated_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_)

with open(report_dir / 'performance_report_calibrated.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("HSEF CALIBRATED MODEL PERFORMANCE REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Baseline Accuracy: {baseline_acc*100:.2f}%\n")
    f.write(f"Calibrated Accuracy: {calibrated_acc*100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("  ✓ Saved performance report")

print("\n" + "="*80)
print("✓ CALIBRATION COMPLETE")
print("="*80)
print(f"\nCalibrated Model Accuracy: {calibrated_acc*100:.2f}%")
print(f"Output Location: {output_dir.absolute()}")
print("\n✅ Model is ready to deploy!")
