"""
HSEF Calibration System for Publication-Ready Model
=====================================================

Autonomous system to:
1. Detect false positives on benign URLs
2. Analyze misclassifications with SHAP
3. Apply feature corrections and domain whitelisting
4. Calibrate meta-layer with regularization
5. Retrain HSEF with augmented data
6. Generate publication-ready artifacts

Author: AI Assistant
Date: October 24, 2025
"""

import numpy as np
import pandas as pd
import joblib
import json
import yaml
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, accuracy_score,
                            precision_recall_fscore_support)

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")

# SHAP for interpretability
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
from url_feature_extractor import URLFeatureExtractor


class HSEFCalibrationSystem:
    """
    Comprehensive system for calibrating HSEF model to reduce false positives
    and prepare for academic publication
    """
    
    def __init__(self, data_path='All.csv', output_dir='publication_outputs'):
        """
        Initialize calibration system
        
        Args:
            data_path: Path to training data CSV
            output_dir: Directory for outputs (plots, models, reports)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
        
        # Components
        self.feature_extractor = URLFeatureExtractor()
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.base_models = {}
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None
        
        # Analysis results
        self.false_positives = []
        self.shap_values = None
        self.shap_explainer = None
        
        # Configuration for corrections
        self.config = {
            'trusted_domains': [
                'youtube.com', 'google.com', 'github.com', 'microsoft.com',
                'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com',
                'instagram.com', 'reddit.com', 'wikipedia.org', 'stackoverflow.com',
                'apple.com', 'netflix.com', 'spotify.com', 'zoom.us',
                'dropbox.com', 'adobe.com', 'salesforce.com', 'oracle.com',
                'w3.org', 'mozilla.org', 'python.org', 'nodejs.org',
                'npmjs.com', 'docker.com', 'kubernetes.io', 'cloudflare.com'
            ],
            'feature_corrections': {
                'entropy_normalization': True,
                'length_capping': {'max_url_length': 500},
                'outlier_handling': 'clip'
            },
            'meta_layer_calibration': {
                'method': 'sigmoid',  # or 'isotonic'
                'cv': 5
            },
            'regularization': {
                'C': 1.0,  # Will be tuned
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000
            }
        }
        
        # Logging
        self.training_log = {
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'metrics': {},
            'config': self.config
        }
        
        print(f"✓ HSEF Calibration System initialized")
        print(f"  Output directory: {self.output_dir.absolute()}")
    
    def load_data(self):
        """Load and prepare training data"""
        print("\n" + "="*70)
        print("STEP 1: Loading Training Data")
        print("="*70)
        
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df)} samples from {self.data_path}")
        
        # Separate features and target
        X = df.drop('URL_Type_obf_Type', axis=1)
        y = df['URL_Type_obf_Type']
        
        self.feature_names = list(X.columns)
        print(f"✓ Features: {len(self.feature_names)}")
        
        # Handle missing values
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        print(f"✓ Handled missing/infinite values")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = list(self.label_encoder.classes_)
        print(f"✓ Classes: {self.class_names}")
        
        # Class distribution
        class_counts = pd.Series(y).value_counts()
        print(f"\n  Class Distribution:")
        for cls, count in class_counts.items():
            print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")
        
        # Train-test split (stratified)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"\n✓ Split: {len(self.X_train)} train, {len(self.X_test)} test")
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print(f"✓ Features scaled (StandardScaler)")
        
        self.training_log['steps'].append({
            'step': 'load_data',
            'samples': len(df),
            'features': len(self.feature_names),
            'classes': len(self.class_names),
            'train_size': len(self.X_train),
            'test_size': len(self.X_test)
        })
        
        return self
    
    def train_baseline_model(self):
        """Train baseline HSEF model before calibration"""
        print("\n" + "="*70)
        print("STEP 2: Training Baseline HSEF Model")
        print("="*70)
        
        # Base learners
        print("\nConfiguring base learners...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        print("  ✓ Random Forest: n_estimators=100, max_depth=20")
        
        if XGB_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                tree_method='hist',
                device='cuda',
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )
            print("  ✓ XGBoost: n_estimators=100, device=cuda")
        else:
            xgb_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            print("  ⚠ XGBoost unavailable, using RF substitute")
        
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            cache_size=1000
        )
        print("  ✓ SVM: kernel=rbf, probability=True")
        
        # Meta-learner (not calibrated yet)
        meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        print("  ✓ Meta-learner: Logistic Regression (uncalibrated)")
        
        # Stacking ensemble
        self.model = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model),
                ('svm', svm)
            ],
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
            verbose=0
        )
        
        print("\n Training baseline model...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Baseline model trained")
        
        # Evaluate
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        
        print(f"\n  Baseline Performance:")
        print(f"    Train Accuracy: {train_score*100:.2f}%")
        print(f"    Test Accuracy:  {test_score*100:.2f}%")
        
        # Store base models
        self.base_models = {
            'Random Forest': self.model.estimators_[0],
            'XGBoost': self.model.estimators_[1],
            'SVM': self.model.estimators_[2]
        }
        
        self.training_log['steps'].append({
            'step': 'train_baseline',
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score)
        })
        
        return self
    
    def detect_false_positives(self, test_urls_path='test_urls.csv', 
                              additional_benign_urls=None):
        """
        Detect false positives on known benign URLs
        
        Args:
            test_urls_path: Path to CSV with url,label columns
            additional_benign_urls: List of additional benign URLs to test
        """
        print("\n" + "="*70)
        print("STEP 3: Detecting False Positives on Benign URLs")
        print("="*70)
        
        benign_urls = []
        
        # Load from test file if exists
        test_path = Path(test_urls_path)
        if test_path.exists():
            df = pd.read_csv(test_path)
            if 'url' in df.columns and 'label' in df.columns:
                benign_df = df[df['label'].str.lower() == 'benign']
                benign_urls.extend(benign_df['url'].tolist())
                print(f"✓ Loaded {len(benign_df)} benign URLs from {test_urls_path}")
        
        # Add trusted domains as full URLs
        for domain in self.config['trusted_domains']:
            benign_urls.append(f"https://{domain}/")
        
        # Add any additional URLs
        if additional_benign_urls:
            benign_urls.extend(additional_benign_urls)
        
        # Remove duplicates
        benign_urls = list(set(benign_urls))
        print(f"✓ Total benign URLs to test: {len(benign_urls)}")
        
        # Extract features and predict
        print("\nExtracting features and predicting...")
        results = []
        
        for i, url in enumerate(benign_urls):
            try:
                # Extract features
                features = self.feature_extractor.extract_features(url)
                feature_vector = self.feature_extractor.get_feature_vector(self.feature_names)
                
                # Prepare for prediction
                X = pd.DataFrame([feature_vector], columns=self.feature_names)
                X = X.fillna(0).replace([np.inf, -np.inf], 0)
                X_scaled = self.scaler.transform(X)
                
                # Predict
                pred = self.model.predict(X_scaled)[0]
                pred_proba = self.model.predict_proba(X_scaled)[0]
                pred_class = self.class_names[pred]
                confidence = float(pred_proba.max())
                
                # Get base model predictions
                base_preds = {}
                for name, base_model in self.base_models.items():
                    base_pred = base_model.predict(X_scaled)[0]
                    base_proba = base_model.predict_proba(X_scaled)[0]
                    base_preds[name] = {
                        'prediction': self.class_names[base_pred],
                        'confidence': float(base_proba.max())
                    }
                
                result = {
                    'url': url,
                    'true_label': 'benign',
                    'predicted_label': pred_class,
                    'confidence': confidence,
                    'is_false_positive': pred_class.lower() != 'benign',
                    'base_model_predictions': base_preds,
                    'features': features
                }
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(benign_urls)} URLs")
                
            except Exception as e:
                print(f"  ⚠ Error processing {url}: {e}")
                continue
        
        # Separate false positives
        self.false_positives = [r for r in results if r['is_false_positive']]
        true_negatives = [r for r in results if not r['is_false_positive']]
        
        print(f"\n✓ Analysis complete:")
        print(f"    True Negatives (correct): {len(true_negatives)} ({len(true_negatives)/len(results)*100:.1f}%)")
        print(f"    False Positives: {len(self.false_positives)} ({len(self.false_positives)/len(results)*100:.1f}%)")
        
        if self.false_positives:
            print(f"\n  False Positive Breakdown:")
            fp_classes = pd.Series([fp['predicted_label'] for fp in self.false_positives]).value_counts()
            for cls, count in fp_classes.items():
                print(f"    Predicted as {cls}: {count}")
        
        # Save results
        results_df = pd.DataFrame(results)
        fp_path = self.reports_dir / 'false_positives.csv'
        results_df.to_csv(fp_path, index=False)
        print(f"\n✓ Saved results to {fp_path}")
        
        self.training_log['steps'].append({
            'step': 'detect_false_positives',
            'total_tested': len(results),
            'true_negatives': len(true_negatives),
            'false_positives': len(self.false_positives),
            'fp_rate': len(self.false_positives) / len(results) if results else 0
        })
        
        return self
    
    def analyze_false_positives_with_shap(self, n_background=100, max_analyze=10):
        """
        Analyze false positives using SHAP to identify misclassification drivers
        
        Args:
            n_background: Number of background samples for SHAP
            max_analyze: Maximum number of FPs to analyze (SHAP is slow)
        """
        print("\n" + "="*70)
        print("STEP 4: SHAP Interpretability Analysis")
        print("="*70)
        
        if not self.false_positives:
            print("⚠ No false positives to analyze")
            return self
        
        # Limit analysis to avoid extremely long SHAP computation
        fps_to_analyze = self.false_positives[:max_analyze]
        if len(self.false_positives) > max_analyze:
            print(f"\n⚠ Limiting SHAP analysis to first {max_analyze} of {len(self.false_positives)} false positives")
            print(f"  (SHAP computation is very slow - ~1-2 min per URL)")
        
        # Create background dataset for SHAP
        print(f"\nInitializing SHAP explainer with {n_background} background samples...")
        background = shap.sample(self.X_train, n_background, random_state=42)
        
        # Create explainer
        self.shap_explainer = shap.KernelExplainer(
            self.model.predict_proba,
            background
        )
        print("✓ SHAP explainer initialized")
        
        # Analyze each false positive
        print(f"\nComputing SHAP values for {len(fps_to_analyze)} false positives...")
        
        shap_results = []
        
        for i, fp in enumerate(fps_to_analyze):
            try:
                # Prepare feature vector
                feature_vector = self.feature_extractor.get_feature_vector(self.feature_names)
                X = pd.DataFrame([feature_vector], columns=self.feature_names)
                X = X.fillna(0).replace([np.inf, -np.inf], 0)
                X_scaled = self.scaler.transform(X)
                
                # Compute SHAP values
                shap_vals = self.shap_explainer.shap_values(X_scaled)
                
                # Get predicted class index
                pred_idx = self.class_names.index(fp['predicted_label'])
                
                # Extract SHAP values for predicted class
                if isinstance(shap_vals, list):
                    class_shap = shap_vals[pred_idx][0]
                else:
                    class_shap = shap_vals[0, :, pred_idx]
                
                # Get top contributing features
                top_indices = np.argsort(np.abs(class_shap))[-10:][::-1]
                
                top_features = [
                    {
                        'feature': self.feature_names[idx],
                        'value': float(X.iloc[0, idx]),
                        'shap_value': float(class_shap[idx]),
                        'impact': 'increases' if class_shap[idx] > 0 else 'decreases'
                    }
                    for idx in top_indices
                ]
                
                shap_results.append({
                    'url': fp['url'],
                    'predicted_as': fp['predicted_label'],
                    'top_features': top_features
                })
                
                if (i + 1) % 5 == 0:
                    print(f"  Analyzed {i + 1}/{len(self.false_positives)} false positives")
                
            except Exception as e:
                print(f"  ⚠ SHAP analysis failed for {fp['url']}: {e}")
                continue
        
        print(f"\n✓ SHAP analysis complete for {len(shap_results)} URLs")
        
        # Save SHAP results
        shap_path = self.reports_dir / 'shap_analysis_false_positives.json'
        with open(shap_path, 'w') as f:
            json.dump(shap_results, f, indent=2)
        print(f"✓ Saved SHAP analysis to {shap_path}")
        
        # Identify most problematic features across all FPs
        feature_impact = {}
        for result in shap_results:
            for feat in result['top_features'][:5]:  # Top 5 per URL
                fname = feat['feature']
                if fname not in feature_impact:
                    feature_impact[fname] = {'count': 0, 'avg_impact': 0, 'values': []}
                feature_impact[fname]['count'] += 1
                feature_impact[fname]['values'].append(abs(feat['shap_value']))
        
        # Calculate average impact
        for fname in feature_impact:
            vals = feature_impact[fname]['values']
            feature_impact[fname]['avg_impact'] = float(np.mean(vals))
        
        # Sort by frequency
        sorted_features = sorted(
            feature_impact.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        print(f"\n  Top Features Driving Misclassification:")
        for feat, info in sorted_features[:10]:
            print(f"    {feat}: appears in {info['count']} FPs, avg impact={info['avg_impact']:.4f}")
        
        self.training_log['steps'].append({
            'step': 'shap_analysis',
            'analyzed': len(shap_results),
            'top_problematic_features': [f[0] for f in sorted_features[:10]]
        })
        
        return self
    
    def save_config(self):
        """Save configuration to YAML file"""
        config_path = self.output_dir / 'config_corrections.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"✓ Saved configuration to {config_path}")
    
    def save_training_log(self):
        """Save training log to JSON"""
        self.training_log['end_time'] = datetime.now().isoformat()
        log_path = self.output_dir / 'training_log_corrected.json'
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(f"✓ Saved training log to {log_path}")
    
    def generate_publication_plots(self):
        """Generate publication-ready plots"""
        print("\n" + "="*70)
        print("GENERATING PUBLICATION-READY PLOTS")
        print("="*70)
        
        # This will be implemented in subsequent steps
        print("⚠ Plot generation will be implemented after calibration")
        return self


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" " * 20 + "HSEF CALIBRATION SYSTEM")
    print(" " * 15 + "Publication-Ready Model Preparation")
    print("="*80)
    
    # Initialize system
    calibrator = HSEFCalibrationSystem(
        data_path='All.csv',
        output_dir='publication_outputs'
    )
    
    # Execute pipeline
    try:
        calibrator.load_data()
        calibrator.train_baseline_model()
        calibrator.detect_false_positives()
        calibrator.analyze_false_positives_with_shap()
        
        # Save artifacts
        calibrator.save_config()
        calibrator.save_training_log()
        
        print("\n" + "="*80)
        print("✓ CALIBRATION PHASE 1 COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review false_positives.csv and SHAP analysis")
        print("  2. Run calibration and retraining")
        print("  3. Generate publication plots")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
