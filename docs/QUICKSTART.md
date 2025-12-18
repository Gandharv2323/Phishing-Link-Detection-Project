# HSEF Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Model

```bash
python hsef_model.py
```

### Step 3: View Results

Check the `hsef_results/` folder for:
- Confusion matrices
- ROC curves
- Feature importance
- Model comparison
- Architecture diagram
- Training logs

---

## 📊 What the Model Does

The **Heterogeneous Stacking Ensemble Framework (HSEF)** performs multi-class URL classification using:

### Base Learners
1. **Random Forest** - Captures hierarchical patterns
2. **XGBoost** - GPU-accelerated gradient boosting
3. **SVM** - High-dimensional decision boundaries

### Meta-Classifier
- **Logistic Regression** - Intelligently fuses base predictions

### Special Features
- ✅ **Entropy-aware feature gating** - Prioritizes features for obfuscated URLs
- ✅ **5-fold cross-validation** - Robust training and evaluation
- ✅ **GPU acceleration** - Automatic detection with CPU fallback
- ✅ **SHAP explanations** - Model interpretability
- ✅ **Comprehensive artifacts** - All plots and reports auto-generated

---

## 📁 Project Structure

```
ASEP/
├── All.csv                        # Your dataset (80 features)
├── hsef_model.py                  # Main HSEF implementation
├── example_usage.py               # Usage examples
├── requirements.txt               # Dependencies
├── README.md                      # Full documentation
├── QUICKSTART.md                  # This file
└── hsef_results/                  # Output folder (auto-created)
    ├── confusion_matrices.png     # Model performance
    ├── roc_curves.png             # ROC analysis
    ├── feature_importance.png     # Top features
    ├── model_comparison.png       # Performance comparison
    ├── hsef_architecture.png      # System diagram
    ├── shap_summary.png           # SHAP explanations
    ├── classification_reports.txt # Detailed metrics
    └── training_log.json          # Complete history
```

---

## ⚙️ Configuration Options

### Fast Mode (Quicker Training)
```python
from hsef_model import HSEFModel

hsef = HSEFModel(
    output_dir='hsef_results',
    use_gpu=True,
    fast_mode=True  # Uses LinearSVC instead of RBF SVM
)

hsef.run_complete_pipeline('All.csv')
```

### CPU-Only Mode
```python
hsef = HSEFModel(
    use_gpu=False  # Disable GPU
)

hsef.run_complete_pipeline('All.csv')
```

---

## 📈 Expected Performance

Based on the model architecture and your 36,709-sample dataset:

| Model | Expected Accuracy | Expected F1-Score |
|-------|------------------|-------------------|
| Random Forest | 97-99% | 97-99% |
| XGBoost | 98-99% | 98-99% |
| SVM | 96-98% | 96-98% |
| **HSEF (Stacking)** | **98-99%** | **98-99%** |

*Note: Actual performance depends on data quality and class distribution*

---

## 🔍 Understanding the Output

### Confusion Matrix
Shows true vs predicted labels for each model. Darker colors indicate more samples.

### ROC Curves
Multi-class ROC analysis showing true positive rate vs false positive rate per class.

### Feature Importance
Top 20 most influential features from Random Forest and XGBoost.

### Model Comparison
Bar chart comparing accuracy, precision, recall, F1-score, and ROC-AUC across all models.

### SHAP Summary
Shows which features contribute most to predictions (interpretability).

---

## 🐛 Common Issues

### Issue: GPU not detected
**Solution**: The framework automatically falls back to CPU. No action needed.

### Issue: Out of memory
**Solution**: Enable fast mode:
```python
hsef = HSEFModel(fast_mode=True)
```

### Issue: Slow training
**Solutions**:
1. Enable fast mode
2. Reduce base learner complexity:
```python
hsef.rf_model.n_estimators = 100
hsef.xgb_model.max_depth = 6
```

### Issue: Missing dependencies
**Solution**: Install XGBoost and SHAP:
```bash
pip install xgboost shap
```

---

## 📊 Interpreting Results

### High Accuracy (>98%)
✅ Model is performing excellently
✅ Features are highly predictive
✅ Dataset has clear class separation

### Moderate Accuracy (90-95%)
⚠️ Consider feature engineering
⚠️ Check for class imbalance
⚠️ Tune hyperparameters

### Low Accuracy (<90%)
❌ Review feature quality
❌ Check data preprocessing
❌ Investigate class overlap

---

## 🎯 Next Steps

1. **Review Artifacts**: Check `hsef_results/` folder
2. **Analyze Errors**: Look at confusion matrix for misclassifications
3. **Feature Analysis**: Review feature importance and SHAP plots
4. **Fine-tune**: Adjust hyperparameters if needed
5. **Deploy**: Use trained model for predictions

---

## 💡 Example: Making Predictions

After training:

```python
from hsef_model import HSEFModel

# Initialize and train
hsef = HSEFModel()
hsef.run_complete_pipeline('All.csv')

# Make predictions on new data
import numpy as np
X_new = hsef.X_test[:5]  # Example: first 5 test samples

predictions = hsef.stacking_model.predict(X_new)
probabilities = hsef.stacking_model.predict_proba(X_new)

# Get class names
for i, pred in enumerate(predictions):
    class_name = hsef.class_names[pred]
    confidence = probabilities[i].max()
    print(f"Sample {i+1}: {class_name} (confidence: {confidence:.2%})")
```

---

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Examples**: See `example_usage.py`
- **Research Paper**: [Include your paper reference]

---

## 🤝 Need Help?

1. Check the full README.md
2. Review example_usage.py
3. Open an issue on GitHub
4. Contact the research team

---

**Happy Classifying! 🎉**

*Built with ❤️ for cybersecurity research*
