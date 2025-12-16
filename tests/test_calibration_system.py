"""
Quick Test Script for HSEF Calibration System
==============================================

Tests all components before running full pipeline.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """Test that all required packages are installed"""
    print("\n" + "="*70)
    print("TEST 1: Checking Imports")
    print("="*70)
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'shap': 'shap',
        'yaml': 'pyyaml',
        'tldextract': 'tldextract'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    # XGBoost is optional
    try:
        import xgboost
        print(f"  ✓ xgboost (GPU available)")
    except ImportError:
        print(f"  ⚠ xgboost - NOT FOUND (will use RF fallback)")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required packages installed")
    return True


def test_data_files():
    """Test that required data files exist"""
    print("\n" + "="*70)
    print("TEST 2: Checking Data Files")
    print("="*70)
    
    required_files = {
        'All.csv': 'Training data',
        'test_urls.csv': 'Test URLs (optional)',
        'url_feature_extractor.py': 'Feature extractor',
        'models/hsef_model.pkl': 'Original model (optional)',
    }
    
    all_exist = True
    for file, desc in required_files.items():
        path = Path(file)
        if path.exists():
            if file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(file)
                print(f"  ✓ {file} - {desc} ({len(df)} rows)")
            else:
                print(f"  ✓ {file} - {desc}")
        else:
            if file == 'test_urls.csv' or 'models' in file:
                print(f"  ⚠ {file} - {desc} (optional, will be created)")
            else:
                print(f"  ✗ {file} - {desc} (REQUIRED)")
                all_exist = False
    
    if not all_exist:
        print("\n❌ Missing required files")
        return False
    
    print("\n✓ All required files present")
    return True


def test_feature_extraction():
    """Test feature extraction"""
    print("\n" + "="*70)
    print("TEST 3: Testing Feature Extraction")
    print("="*70)
    
    try:
        from url_feature_extractor import URLFeatureExtractor
        
        extractor = URLFeatureExtractor()
        test_url = "https://www.example.com/test/page.html?id=123"
        
        print(f"  Testing URL: {test_url}")
        features = extractor.extract_features(test_url)
        
        print(f"  ✓ Extracted {len(features)} features")
        
        # Check key features
        key_features = ['urlLen', 'domainlength', 'Entropy_URL', 'NumberofDotsinURL']
        for feat in key_features:
            if feat in features:
                print(f"    {feat}: {features[feat]}")
        
        print("\n✓ Feature extraction working")
        return True
        
    except Exception as e:
        print(f"\n❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calibration_modules():
    """Test that calibration modules can be imported"""
    print("\n" + "="*70)
    print("TEST 4: Testing Calibration Modules")
    print("="*70)
    
    try:
        from hsef_calibration_system import HSEFCalibrationSystem
        print("  ✓ hsef_calibration_system.py")
        
        from hsef_calibration_phase2 import HSEFCalibratorPhase2
        print("  ✓ hsef_calibration_phase2.py")
        
        from hsef_helpers import CalibratedHSEFPredictor
        print("  ✓ hsef_helpers.py")
        
        print("\n✓ All calibration modules can be imported")
        return True
        
    except Exception as e:
        print(f"\n❌ Module import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_baseline():
    """Test quick baseline training (1 iteration)"""
    print("\n" + "="*70)
    print("TEST 5: Quick Baseline Training (Small Sample)")
    print("="*70)
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Load small sample
        print("  Loading data...")
        df = pd.read_csv('All.csv')
        sample = df.sample(n=min(1000, len(df)), random_state=42)
        
        X = sample.drop('URL_Type_obf_Type', axis=1)
        y = sample['URL_Type_obf_Type']
        
        # Prepare
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"  ✓ Prepared {len(X_train)} train, {len(X_test)} test samples")
        
        # Quick RF
        print("  Training quick Random Forest...")
        rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        score = rf.score(X_test_scaled, y_test)
        
        print(f"  ✓ RF trained - Test accuracy: {score*100:.1f}%")
        
        print("\n✓ Baseline training works")
        return True
        
    except Exception as e:
        print(f"\n❌ Baseline training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "█"*70)
    print("█" + " "*20 + "HSEF CALIBRATION SYSTEM" + " "*26 + "█")
    print("█" + " "*24 + "Pre-Flight Checks" + " "*28 + "█")
    print("█"*70)
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data_files),
        ("Feature Extraction", test_feature_extraction),
        ("Calibration Modules", test_calibration_modules),
        ("Quick Baseline", test_quick_baseline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED - SYSTEM READY")
        print("="*70)
        print("\nYou can now run:")
        print("  python run_full_calibration.py")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠ SOME TESTS FAILED - FIX ISSUES BEFORE RUNNING")
        print("="*70)
        print("\nPlease address the failed tests above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
