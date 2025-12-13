"""
Heterogeneous Stacking Ensemble Framework (HSEF) for Multi-Class URL Classification

This module implements a sophisticated ensemble learning framework that combines:
- Random Forest (RF): Hierarchical interaction-based learning
- XGBoost: Gradient boosting with GPU acceleration
- SVM: High-dimensional decision boundaries

The framework uses dynamic logistic regression meta-classifier for intelligent fusion
with entropy-aware feature gating for robust obfuscation detection.

Author: HSEF Research Team
Date: October 24, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
from datetime import datetime
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    GridSearchCV,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

# XGBoost with GPU detection
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class HSEFModel:
    """
    Heterogeneous Stacking Ensemble Framework (HSEF)
    
    A sophisticated multi-class URL classifier combining RF, XGBoost, and SVM
    with dynamic logistic regression meta-classifier and entropy-aware feature gating.
    """
    
    def __init__(self, output_dir='hsef_results', use_gpu=True, fast_mode=False):
        """
        Initialize HSEF Model
        
        Parameters:
        -----------
        output_dir : str
            Directory to save all artifacts (plots, logs, models)
        use_gpu : bool
            Enable GPU acceleration for XGBoost if available
        fast_mode : bool
            Use LinearSVC instead of RBF SVM for faster training
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.use_gpu = use_gpu
        self.fast_mode = fast_mode
        
        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Base learners
        self.rf_model = None
        self.xgb_model = None
        self.svm_model = None
        
        # Stacking ensemble
        self.stacking_model = None
        
        # Feature information
        self.feature_names = None
        self.entropy_features = None
        self.class_names = None
        
        # Training history
        self.history = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'config': {
                'use_gpu': use_gpu,
                'fast_mode': fast_mode
            }
        }
        
        # GPU detection for XGBoost
        self.gpu_available = self._detect_gpu()
        
        print(f"HSEF Initialized:")
        print(f"  - Output Directory: {self.output_dir}")
        print(f"  - GPU Available: {self.gpu_available}")
        print(f"  - Fast Mode: {self.fast_mode}")
        print(f"  - XGBoost Available: {XGBOOST_AVAILABLE}")
        print(f"  - SHAP Available: {SHAP_AVAILABLE}")
    
    def _detect_gpu(self):
        """Detect GPU availability for XGBoost"""
        if not XGBOOST_AVAILABLE or not self.use_gpu:
            return False
        
        try:
            # Try to create a GPU-enabled XGBoost model
            test_model = xgb.XGBClassifier(
                tree_method='hist', 
                device='cuda:0',
                n_estimators=1
            )
            # Test with dummy data
            X_test = np.random.rand(10, 5)
            y_test = np.random.randint(0, 2, 10)
            test_model.fit(X_test, y_test)
            print("  ✓ GPU acceleration enabled for XGBoost (NVIDIA GeForce RTX 2050)")
            return True
        except Exception as e:
            print(f"  ✗ GPU not available, using CPU fallback: {str(e)}")
            print(f"  Attempting CUDA detection...")
            # Try alternative method
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("  ℹ NVIDIA GPU detected but XGBoost GPU support unavailable")
                    print("  ℹ Continuing with CPU mode")
            except:
                pass
            return False
    
    def _identify_entropy_features(self, feature_names):
        """Identify entropy-based features for dynamic gating"""
        entropy_keywords = ['entropy', 'Entropy']
        entropy_features = [
            i for i, name in enumerate(feature_names) 
            if any(keyword in name for keyword in entropy_keywords)
        ]
        print(f"\nIdentified {len(entropy_features)} entropy features for dynamic gating")
        return entropy_features
    
    def _apply_entropy_gating(self, X, threshold=0.7):
        """
        Apply entropy-aware feature gating
        
        Prioritize structural and entropy features for high-entropy URLs
        """
        if self.entropy_features is None or len(self.entropy_features) == 0:
            return X
        
        X_gated = X.copy()
        
        # Calculate mean entropy per sample
        entropy_values = X[:, self.entropy_features].mean(axis=1)
        
        # For high-entropy samples, boost entropy feature importance
        high_entropy_mask = entropy_values > threshold
        
        if np.any(high_entropy_mask):
            # Boost entropy features by 1.5x for high-entropy URLs
            X_gated[high_entropy_mask][:, self.entropy_features] *= 1.5
        
        return X_gated
    
    def load_data(self, csv_path, target_column='URL_Type_obf_Type', test_size=0.2, random_state=42):
        """
        Load and preprocess dataset
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file
        target_column : str
            Name of target column
        test_size : float
            Proportion of test set
        random_state : int
            Random seed for reproducibility
        """
        print(f"\n{'='*70}")
        print("LOADING AND PREPROCESSING DATA")
        print(f"{'='*70}")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"\nDataset Shape: {df.shape}")
        print(f"Target Column: {target_column}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Identify entropy features
        self.entropy_features = self._identify_entropy_features(self.feature_names)
        
        # Handle missing values
        print(f"\nMissing values before handling: {X.isnull().sum().sum()}")
        X = X.fillna(X.median())
        print(f"Missing values after handling: {X.isnull().sum().sum()}")
        
        # Handle infinity values
        print(f"\nInfinity values before handling: {np.isinf(X.values).sum()}")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        print(f"Infinity values after handling: {np.isinf(X.values).sum()}")
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        print(f"\nClass Distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(y_encoded == i)
            print(f"  - {class_name}: {count} ({count/len(y_encoded)*100:.2f}%)")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Store original data for later use
        self.X_train_raw = X_train.copy()
        self.X_test_raw = X_test.copy()
        
        # Feature scaling
        print("\nApplying StandardScaler to continuous features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply entropy gating
        print("Applying entropy-aware feature gating...")
        X_train_scaled = self._apply_entropy_gating(X_train_scaled)
        X_test_scaled = self._apply_entropy_gating(X_test_scaled)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\n{'='*70}")
        print("DATA LOADING COMPLETE")
        print(f"{'='*70}\n")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_base_learners(self):
        """
        Build and configure base learners:
        - Random Forest
        - XGBoost (with GPU if available)
        - SVM (RBF kernel or LinearSVC for fast mode)
        """
        print(f"\n{'='*70}")
        print("BUILDING BASE LEARNERS")
        print(f"{'='*70}\n")
        
        # 1. Random Forest
        print("Configuring Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        print("  ✓ Random Forest configured")
        
        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            print("Configuring XGBoost...")
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multi:softprob',
                'random_state': 42,
                'n_jobs': -1
            }
            
            if self.gpu_available:
                xgb_params['tree_method'] = 'hist'
                xgb_params['device'] = 'cuda:0'
                print("  ✓ XGBoost configured with GPU acceleration (RTX 2050)")
            else:
                xgb_params['tree_method'] = 'hist'
                print("  ✓ XGBoost configured with CPU")
            
            self.xgb_model = xgb.XGBClassifier(**xgb_params)
        else:
            print("  ✗ XGBoost not available, using additional Random Forest")
            self.xgb_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=25,
                random_state=43,
                n_jobs=-1
            )
        
        # 3. SVM
        print("Configuring SVM...")
        if self.fast_mode:
            from sklearn.svm import LinearSVC
            print("  ⚡ Fast mode enabled: Using LinearSVC")
            svm_base = LinearSVC(
                C=1.0,
                max_iter=1000,
                random_state=42,
                dual=False
            )
            # Calibrate for probability estimates
            self.svm_model = CalibratedClassifierCV(svm_base, cv=3)
        else:
            print("  Standard mode: Using RBF SVM")
            self.svm_model = SVC(
                C=10.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42,
                cache_size=1000
            )
        print("  ✓ SVM configured")
        
        print(f"\n{'='*70}")
        print("BASE LEARNERS CONFIGURED")
        print(f"{'='*70}\n")
    
    def train_base_learners_with_cv(self, n_folds=5):
        """
        Train base learners with stratified k-fold cross-validation
        
        Parameters:
        -----------
        n_folds : int
            Number of folds for cross-validation
        """
        print(f"\n{'='*70}")
        print(f"TRAINING BASE LEARNERS WITH {n_folds}-FOLD CROSS-VALIDATION")
        print(f"{'='*70}\n")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        models = {
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model,
            'SVM': self.svm_model
        }
        
        cv_results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=skf, scoring='accuracy', n_jobs=-1 if name != 'SVM' else 1
            )
            cv_results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores.tolist()
            }
            print(f"  ✓ {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        self.history['cv_results'] = cv_results
        
        # Train on full training set
        print("\nTraining on full training set...")
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(self.X_train, self.y_train)
            train_acc = model.score(self.X_train, self.y_train)
            print(f"    Training Accuracy: {train_acc:.4f}")
        
        print(f"\n{'='*70}")
        print("BASE LEARNERS TRAINING COMPLETE")
        print(f"{'='*70}\n")
    
    def build_stacking_ensemble(self):
        """
        Build stacking ensemble with dynamic logistic regression meta-classifier
        """
        print(f"\n{'='*70}")
        print("BUILDING STACKING ENSEMBLE")
        print(f"{'='*70}\n")
        
        # Define base estimators
        estimators = [
            ('rf', self.rf_model),
            ('xgb', self.xgb_model),
            ('svm', self.svm_model)
        ]
        
        # Dynamic logistic regression meta-classifier
        meta_classifier = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        # Build stacking classifier
        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
            verbose=0
        )
        
        print("Stacking Ensemble Architecture:")
        print("  Base Layer:")
        print("    - Random Forest (RF)")
        print("    - XGBoost (XGB)")
        print("    - Support Vector Machine (SVM)")
        print("  Meta Layer:")
        print("    - Dynamic Logistic Regression")
        print("  Fusion Method: Probabilistic Output Concatenation")
        
        print(f"\n{'='*70}")
        print("STACKING ENSEMBLE CONFIGURED")
        print(f"{'='*70}\n")
    
    def train_stacking_ensemble(self):
        """Train the complete stacking ensemble"""
        print(f"\n{'='*70}")
        print("TRAINING STACKING ENSEMBLE")
        print(f"{'='*70}\n")
        
        print("Training stacked model with meta-classifier fusion...")
        self.stacking_model.fit(self.X_train, self.y_train)
        
        train_acc = self.stacking_model.score(self.X_train, self.y_train)
        print(f"  ✓ Training Accuracy: {train_acc:.4f}")
        
        print(f"\n{'='*70}")
        print("STACKING ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*70}\n")
    
    def evaluate_models(self):
        """
        Comprehensive evaluation of base learners and stacking ensemble
        """
        print(f"\n{'='*70}")
        print("MODEL EVALUATION")
        print(f"{'='*70}\n")
        
        models = {
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model,
            'SVM': self.svm_model,
            'HSEF (Stacking)': self.stacking_model
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name}:")
            print("-" * 50)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC-AUC (multi-class)
            try:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = np.nan
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        self.history['test_results'] = {
            k: {metric: v[metric] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']}
            for k, v in results.items()
        }
        
        print(f"\n{'='*70}")
        print("MODEL EVALUATION COMPLETE")
        print(f"{'='*70}\n")
        
        return results
    
    def plot_confusion_matrices(self, results):
        """Generate confusion matrices for all models"""
        print("Generating confusion matrices...")
        
        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, (name, data) in enumerate(results.items()):
            cm = confusion_matrix(self.y_test, data['y_pred'])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[idx],
                cbar_kws={'label': 'Count'}
            )
            
            axes[idx].set_title(f'{name}\nAccuracy: {data["accuracy"]:.4f}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
            axes[idx].set_ylabel('True Label', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / 'confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def plot_roc_curves(self, results):
        """Generate ROC curves for all models"""
        print("Generating ROC curves...")
        
        n_classes = len(self.class_names)
        
        # Create subplot for each class
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(self.y_test, classes=range(n_classes))
        
        for class_idx in range(min(4, n_classes)):
            ax = axes[class_idx]
            
            for name, data in results.items():
                y_score = data['y_pred_proba'][:, class_idx]
                
                fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(f'ROC Curve: {self.class_names[class_idx]}', fontsize=12, fontweight='bold')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def plot_feature_importance(self):
        """Plot feature importance from Random Forest and XGBoost"""
        print("Generating feature importance plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Random Forest importance
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        axes[0].barh(range(len(rf_importance)), rf_importance['importance'], color='steelblue')
        axes[0].set_yticks(range(len(rf_importance)))
        axes[0].set_yticklabels(rf_importance['feature'], fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Importance', fontsize=10)
        axes[0].set_title('Random Forest: Top 20 Features', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # XGBoost importance
        if XGBOOST_AVAILABLE and isinstance(self.xgb_model, xgb.XGBClassifier):
            xgb_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            axes[1].barh(range(len(xgb_importance)), xgb_importance['importance'], color='coral')
            axes[1].set_yticks(range(len(xgb_importance)))
            axes[1].set_yticklabels(xgb_importance['feature'], fontsize=9)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Importance', fontsize=10)
            axes[1].set_title('XGBoost: Top 20 Features', fontsize=12, fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'XGBoost not available', 
                        ha='center', va='center', fontsize=14)
            axes[1].set_title('XGBoost Feature Importance', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / 'feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def plot_model_comparison(self, results):
        """Create comprehensive model comparison charts"""
        print("Generating model comparison chart...")
        
        # Extract metrics
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        data = {metric: [results[model][metric] for model in models] for metric in metrics}
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.15
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, (metric, values) in enumerate(data.items()):
            offset = width * (i - 2)
            bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), color=colors[i])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('HSEF: Comprehensive Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def generate_classification_reports(self, results):
        """Generate detailed classification reports for all models"""
        print("Generating classification reports...")
        
        report_path = self.output_dir / 'classification_reports.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HSEF: DETAILED CLASSIFICATION REPORTS\n")
            f.write("="*80 + "\n\n")
            
            for name, data in results.items():
                f.write(f"\n{name}\n")
                f.write("-"*80 + "\n")
                report = classification_report(
                    self.y_test, data['y_pred'],
                    target_names=self.class_names,
                    digits=4
                )
                f.write(report)
                f.write("\n")
        
        print(f"  ✓ Saved: {report_path}")
    
    def generate_architecture_diagram(self):
        """Generate HSEF architecture diagram"""
        print("Generating architecture diagram...")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Heterogeneous Stacking Ensemble Framework (HSEF)', 
                ha='center', fontsize=18, fontweight='bold')
        
        # Feature Layer
        feature_box = plt.Rectangle((0.1, 0.75), 0.8, 0.12, 
                                    fill=True, facecolor='lightblue', 
                                    edgecolor='black', linewidth=2)
        ax.add_patch(feature_box)
        ax.text(0.5, 0.81, 'Multi-Domain Feature Vector (80 features)', 
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(0.5, 0.78, 'Lexical • Structural • Entropy • Semantic', 
                ha='center', va='center', fontsize=9, style='italic')
        
        # Arrows to base learners
        ax.arrow(0.25, 0.75, 0, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
        ax.arrow(0.5, 0.75, 0, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
        ax.arrow(0.75, 0.75, 0, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
        
        # Base Learners
        rf_box = plt.Rectangle((0.05, 0.5), 0.25, 0.15, 
                               fill=True, facecolor='#3498db', 
                               edgecolor='black', linewidth=2)
        xgb_box = plt.Rectangle((0.375, 0.5), 0.25, 0.15, 
                                fill=True, facecolor='#e74c3c', 
                                edgecolor='black', linewidth=2)
        svm_box = plt.Rectangle((0.7, 0.5), 0.25, 0.15, 
                                fill=True, facecolor='#2ecc71', 
                                edgecolor='black', linewidth=2)
        
        ax.add_patch(rf_box)
        ax.add_patch(xgb_box)
        ax.add_patch(svm_box)
        
        ax.text(0.175, 0.6, 'Random Forest', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        ax.text(0.175, 0.55, 'Bagging\nGini Impurity', ha='center', va='center', 
                fontsize=8, color='white')
        
        ax.text(0.5, 0.6, 'XGBoost', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        ax.text(0.5, 0.55, 'Gradient Boosting\nGPU-Enabled', ha='center', va='center', 
                fontsize=8, color='white')
        
        ax.text(0.825, 0.6, 'SVM', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        ax.text(0.825, 0.55, 'RBF Kernel\nHigh-Dim Boundary', ha='center', va='center', 
                fontsize=8, color='white')
        
        # Probability outputs
        ax.text(0.175, 0.45, 'P_RF', ha='center', fontsize=9, style='italic')
        ax.text(0.5, 0.45, 'P_XGB', ha='center', fontsize=9, style='italic')
        ax.text(0.825, 0.45, 'P_SVM', ha='center', fontsize=9, style='italic')
        
        # Arrows to meta-classifier
        ax.arrow(0.175, 0.43, 0.15, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
        ax.arrow(0.5, 0.43, 0, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
        ax.arrow(0.825, 0.43, -0.15, -0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
        
        # Meta-Classifier
        meta_box = plt.Rectangle((0.25, 0.2), 0.5, 0.13, 
                                 fill=True, facecolor='#f39c12', 
                                 edgecolor='black', linewidth=2)
        ax.add_patch(meta_box)
        ax.text(0.5, 0.29, 'Dynamic Logistic Regression Meta-Classifier', 
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(0.5, 0.24, 'ŷ = σ(W·Z + b)  |  Adaptive Fusion Weights', 
                ha='center', va='center', fontsize=9, style='italic')
        
        # Output
        ax.arrow(0.5, 0.2, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        output_box = plt.Rectangle((0.3, 0.05), 0.4, 0.08, 
                                   fill=True, facecolor='lightgreen', 
                                   edgecolor='black', linewidth=2)
        ax.add_patch(output_box)
        ax.text(0.5, 0.09, 'Class Prediction + Probabilities + SHAP Explanations', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Side annotations
        ax.text(0.02, 0.81, 'Input\nLayer', ha='left', va='center', 
                fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.text(0.02, 0.575, 'Base\nLayer', ha='left', va='center', 
                fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.text(0.02, 0.265, 'Meta\nLayer', ha='left', va='center', 
                fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.text(0.02, 0.09, 'Output\nLayer', ha='left', va='center', 
                fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        save_path = self.output_dir / 'hsef_architecture.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    
    def save_training_log(self):
        """Save comprehensive training log"""
        print("Saving training log...")
        
        log_path = self.output_dir / 'training_log.json'
        
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"  ✓ Saved: {log_path}")
    
    def save_for_deployment(self):
        """Save model for web application deployment"""
        print("\n" + "="*70)
        print("SAVING MODEL FOR WEB DEPLOYMENT")
        print("="*70 + "\n")
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        try:
            # Save stacking model
            import joblib
            model_path = models_dir / 'hsef_model.pkl'
            joblib.dump(self.stacking_model, model_path)
            print(f"✓ Saved stacking model: {model_path}")
            
            # Save scaler
            scaler_path = models_dir / 'hsef_scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            print(f"✓ Saved scaler: {scaler_path}")
            
            # Save feature names and class names
            features_path = models_dir / 'feature_names.json'
            features_data = {
                'features': self.feature_names,
                'classes': self.class_names.tolist()
            }
            with open(features_path, 'w') as f:
                json.dump(features_data, f, indent=2)
            print(f"✓ Saved feature metadata: {features_path}")
            
            # Save model info
            info_path = models_dir / 'model_info.json'
            model_info = {
                'model_type': 'HSEF - Heterogeneous Stacking Ensemble Framework',
                'base_learners': ['Random Forest', 'XGBoost', 'SVM'],
                'meta_learner': 'Logistic Regression',
                'n_features': len(self.feature_names),
                'n_classes': len(self.class_names),
                'classes': self.class_names.tolist(),
                'gpu_enabled': self.gpu_available,
                'fast_mode': self.fast_mode,
                'training_metrics': self.history.get('test_results', {}),
                'timestamp': self.history.get('timestamp', '')
            }
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            print(f"✓ Saved model info: {info_path}")
            
            print("\n" + "="*70)
            print("WEB DEPLOYMENT FILES READY")
            print("="*70)
            print(f"\nModel files saved to: {models_dir.absolute()}")
            print("\n🚀 You can now run the web application:")
            print("   python app.py")
            print("\n   Then open: http://127.0.0.1:5000")
            
        except Exception as e:
            print(f"\n✗ Error saving deployment files: {str(e)}")
    
    def generate_shap_explanations(self, n_samples=100):
        """Generate SHAP explanations for model interpretability"""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping SHAP explanations...")
            return
        
        print(f"Generating SHAP explanations for {n_samples} samples...")
        
        try:
            # Use Random Forest for SHAP (faster than ensemble)
            explainer = shap.TreeExplainer(self.rf_model)
            
            # Select sample
            X_sample = self.X_test[:n_samples]
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            fig, ax = plt.subplots(figsize=(12, 8))
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0]  # Use first class for multi-class
            else:
                shap_values_plot = shap_values
            
            shap.summary_plot(
                shap_values_plot, X_sample, 
                feature_names=self.feature_names,
                show=False, max_display=20
            )
            
            plt.tight_layout()
            save_path = self.output_dir / 'shap_summary.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {save_path}")
            
        except Exception as e:
            print(f"  ✗ SHAP generation failed: {str(e)}")
    
    def run_complete_pipeline(self, csv_path, target_column='URL_Type_obf_Type'):
        """
        Execute complete HSEF pipeline
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV dataset
        target_column : str
            Name of target column
        """
        print("\n" + "="*70)
        print("HETEROGENEOUS STACKING ENSEMBLE FRAMEWORK (HSEF)")
        print("Multi-Class URL Classification Pipeline")
        print("="*70)
        
        # 1. Load and preprocess data
        self.load_data(csv_path, target_column)
        
        # 2. Build base learners
        self.build_base_learners()
        
        # 3. Train base learners with cross-validation
        self.train_base_learners_with_cv(n_folds=5)
        
        # 4. Build stacking ensemble
        self.build_stacking_ensemble()
        
        # 5. Train stacking ensemble
        self.train_stacking_ensemble()
        
        # 6. Evaluate models
        results = self.evaluate_models()
        
        # 7. Generate visualizations
        print(f"\n{'='*70}")
        print("GENERATING ARTIFACTS")
        print(f"{'='*70}\n")
        
        self.plot_confusion_matrices(results)
        self.plot_roc_curves(results)
        self.plot_feature_importance()
        self.plot_model_comparison(results)
        self.generate_classification_reports(results)
        self.generate_architecture_diagram()
        self.generate_shap_explanations(n_samples=100)
        self.save_training_log()
        
        print(f"\n{'='*70}")
        print("PIPELINE EXECUTION COMPLETE")
        print(f"{'='*70}\n")
        print(f"All artifacts saved to: {self.output_dir.absolute()}")
        
        # Final summary
        print("\nFinal Performance Summary:")
        print("-" * 70)
        for name, data in results.items():
            print(f"{name:20s} | Acc: {data['accuracy']:.4f} | F1: {data['f1_score']:.4f} | AUC: {data['roc_auc']:.4f}")
        
        # Save model for deployment
        self.save_for_deployment()
        
        return results


def main():
    """Main execution function"""
    
    # Initialize HSEF
    hsef = HSEFModel(
        output_dir='hsef_results',
        use_gpu=True,
        fast_mode=False  # Set to True for faster training with LinearSVC
    )
    
    # Run complete pipeline
    results = hsef.run_complete_pipeline(
        csv_path='All.csv',
        target_column='URL_Type_obf_Type'
    )
    
    print("\n" + "="*70)
    print("HSEF PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
