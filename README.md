# HSEF: Heterogeneous Stacking Ensemble Framework

**Multi-Class URL Classification for Phishing, Malware, Defacement, and Benign URLs**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

---

## 📋 Overview

The **Heterogeneous Stacking Ensemble Framework (HSEF)** is a sophisticated machine learning system designed for accurate multi-class URL classification. Unlike conventional single-classifier or homogeneous ensemble methods, HSEF integrates three fundamentally distinct base learners with complementary decision mechanisms:

- **Random Forest (RF)**: Captures hierarchical and interaction-based relationships via bagging and Gini impurity
- **Extreme Gradient Boosting (XGBoost)**: Focuses on hard-to-classify URLs using additive gradient boosting with GPU acceleration
- **Support Vector Machine (SVM)**: Establishes high-dimensional decision boundaries using RBF kernel

A **dynamic logistic regression meta-classifier** intelligently fuses the base learners' probabilistic outputs, adaptively weighting each model based on the input URL's feature distribution.

---

## 🎯 Key Features

### 1. **Heterogeneous Base Learners**
- Three distinct algorithms reduce correlated errors
- 5-fold stratified cross-validation for robust training
- GPU acceleration for XGBoost (automatic detection with CPU fallback)

### 2. **Entropy-Aware Feature Gating**
- 80-feature multi-domain representation (lexical, structural, entropy, semantic)
- Dynamic feature prioritization for high-entropy (obfuscated) URLs
- Improved robustness to randomized patterns

### 3. **Dynamic Meta-Classifier Fusion**
- Logistic regression adaptively weights base predictions
- Probabilistic output concatenation: Z = [P_RF, P_XGB, P_SVM]
- Per-URL optimization for maximum accuracy

### 4. **Comprehensive Interpretability**
- SHAP (SHapley Additive exPlanations) for individual predictions
- Feature importance from RF and XGBoost
- Meta-layer contribution analysis

### 5. **Automated Artifact Generation**
- Confusion matrices for all models
- ROC-AUC curves (multi-class, per-class)
- Feature importance plots
- Model comparison charts
- Architecture diagrams
- Training logs (JSON)
- Classification reports

### 6. **Fast Mode Option**
- Replace RBF SVM with calibrated LinearSVC
- Reduces training time for large datasets
- Maintains competitive accuracy

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
cd ASEP

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from hsef_model import HSEFModel

# Initialize the model
hsef = HSEFModel(
    output_dir='hsef_results',
    use_gpu=True,        # Enable GPU for XGBoost
    fast_mode=False      # Set True for faster training
)

# Run complete pipeline
results = hsef.run_complete_pipeline(
    csv_path='All.csv',
    target_column='URL_Type_obf_Type'
)
```

### Command Line Execution

```bash
python hsef_model.py
```

---

## 📊 Dataset Requirements

The framework expects a CSV file with:
- **Features**: 80 columns covering lexical, structural, entropy, and semantic properties
- **Target**: A categorical column (e.g., `URL_Type_obf_Type`) with class labels

### Example Feature Categories

| Category | Examples |
|----------|----------|
| **Lexical** | urlLen, NumberofDotsinURL, charcompace, subDirLen |
| **Structural** | pathurlRatio, argDomainRatio, domainUrlRatio, NumberRate_* |
| **Entropy** | Entropy_URL, Entropy_Domain, Entropy_DirectoryName |
| **Semantic** | URL_sensitiveWord, ISIpAddressInDomainName, executable |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│      Multi-Domain Feature Vector (80 features)              │
│   Lexical • Structural • Entropy • Semantic                 │
└────────────┬────────────┬────────────┬──────────────────────┘
             │            │            │
    ┌────────▼────────┐  ┌▼──────────┐  ┌▼─────────────┐
    │  Random Forest  │  │  XGBoost  │  │     SVM      │
    │  Bagging/Gini   │  │ GPU-Boost │  │  RBF Kernel  │
    └────────┬────────┘  └┬──────────┘  └┬─────────────┘
             │            │               │
             │  P_RF      │  P_XGB        │  P_SVM
             └────────────┼───────────────┘
                          │
              ┌───────────▼────────────┐
              │  Logistic Regression   │
              │   Meta-Classifier      │
              │   ŷ = σ(W·Z + b)       │
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │ Class Prediction +     │
              │ Probabilities + SHAP   │
              └────────────────────────┘
```

---

## 📈 Output Artifacts

After execution, the following files are generated in `hsef_results/`:

| File | Description |
|------|-------------|
| `confusion_matrices.png` | 2x2 grid of confusion matrices for all models |
| `roc_curves.png` | ROC curves for each class (multi-class analysis) |
| `feature_importance.png` | Top 20 features from RF and XGBoost |
| `model_comparison.png` | Grouped bar chart comparing all metrics |
| `hsef_architecture.png` | Visual representation of the framework |
| `shap_summary.png` | SHAP feature importance summary |
| `classification_reports.txt` | Detailed per-class metrics |
| `training_log.json` | Complete training history and configuration |

---

## 🔧 Configuration Options

### HSEFModel Parameters

```python
hsef = HSEFModel(
    output_dir='hsef_results',  # Directory for artifacts
    use_gpu=True,               # GPU acceleration (XGBoost)
    fast_mode=False             # Use LinearSVC instead of RBF SVM
)
```

### Pipeline Parameters

```python
results = hsef.run_complete_pipeline(
    csv_path='All.csv',                    # Path to dataset
    target_column='URL_Type_obf_Type'      # Target column name
)
```

### Training Configuration

- **Cross-Validation**: 5-fold stratified splits
- **Test Split**: 20% (configurable in `load_data()`)
- **Scaling**: StandardScaler for continuous features
- **Random State**: 42 (reproducible results)

---

## 📊 Performance Metrics

The framework evaluates models using:

- **Accuracy**: Overall classification correctness
- **Precision**: Class-specific positive predictive value
- **Recall**: Class-specific sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Multi-class area under ROC curve (weighted average)

### Example Output

```
Final Performance Summary:
----------------------------------------------------------------------
Random Forest        | Acc: 0.9845 | F1: 0.9843 | AUC: 0.9956
XGBoost              | Acc: 0.9867 | F1: 0.9865 | AUC: 0.9971
SVM                  | Acc: 0.9823 | F1: 0.9821 | AUC: 0.9948
HSEF (Stacking)      | Acc: 0.9891 | F1: 0.9890 | AUC: 0.9978
```

---

## 🧪 Advanced Usage

### Custom Hyperparameter Tuning

```python
# Modify base learner parameters
hsef.rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=40,
    min_samples_split=3,
    random_state=42
)

# Then train normally
hsef.train_base_learners_with_cv(n_folds=5)
```

### Entropy Gating Threshold

```python
# Adjust entropy gating sensitivity
X_train_gated = hsef._apply_entropy_gating(X_train, threshold=0.8)
```

### SHAP Explanation for Specific Samples

```python
# Generate SHAP for more samples
hsef.generate_shap_explanations(n_samples=500)
```

---

## 🔬 Comparative Advantages

| Feature | Conventional Approach | HSEF Advantage |
|---------|----------------------|----------------|
| **Base Models** | Single or homogeneous ensemble | Heterogeneous stacking reduces correlated errors |
| **Features** | Lexical or structural only | Multi-domain (lexical, structural, entropy, semantic) |
| **Obfuscation** | Weak to randomized patterns | High-entropy URLs dynamically gated |
| **Interpretability** | Black-box predictions | Meta-layer weights + SHAP explanations |
| **Deployment** | Offline | Local, GPU-enabled with real-time inference |

---

## 📝 Requirements

### Python Version
- Python 3.8 or higher

### Core Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- shap >= 0.41.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

### Optional (for web deployment)
- flask >= 2.0.0
- fastapi >= 0.95.0
- uvicorn >= 0.21.0

---

## 🐛 Troubleshooting

### GPU Not Detected

If GPU is not available:
```python
# The framework automatically falls back to CPU
hsef = HSEFModel(use_gpu=False)
```

### Out of Memory Errors

For large datasets:
```python
# Enable fast mode
hsef = HSEFModel(fast_mode=True)

# Or reduce base learner complexity
hsef.rf_model.n_estimators = 100
hsef.xgb_model.max_depth = 6
```

### XGBoost Not Installed

```bash
pip install xgboost
```

### SHAP Not Installed

```bash
pip install shap
```

---

## 📚 Citation

If you use HSEF in your research, please cite:

```bibtex
@software{hsef2025,
  title={Heterogeneous Stacking Ensemble Framework for Multi-Class URL Classification},
  author={HSEF Research Team},
  year={2025},
  url={https://github.com/your-repo/hsef}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- **Scikit-learn**: Foundation for machine learning implementations
- **XGBoost**: High-performance gradient boosting
- **SHAP**: Model interpretability framework
- Research community for URL security datasets

---

**Built with ❤️ for cybersecurity research**
