"""
HSEF Model Debugger and Analyzer
Provides detailed analysis of URL classification decisions including:
- Individual base model predictions and probabilities
- Meta-layer fusion weights and contributions
- SHAP feature importance analysis
- Corrective insights for misclassifications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP not available. Install with: pip install shap")


class HSEFDebugger:
    """
    Debug and analyze HSEF model predictions with detailed breakdowns
    """
    
    def __init__(self, models_dir='models', output_dir='debug_results'):
        """
        Initialize the debugger
        
        Parameters:
        -----------
        models_dir : str
            Directory containing saved models
        output_dir : str
            Directory for saving debug outputs
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("HSEF MODEL DEBUGGER - INITIALIZATION")
        print("="*70 + "\n")
        
        # Load models and metadata
        self._load_models()
        
        # Initialize SHAP explainers if available
        if SHAP_AVAILABLE:
            self._initialize_shap()
        
        print("\n✓ Debugger ready\n")
    
    def _load_models(self):
        """Load all necessary models and metadata"""
        try:
            # Load stacking model
            model_path = self.models_dir / 'hsef_model.pkl'
            self.stacking_model = joblib.load(model_path)
            print(f"✓ Loaded stacking model")
            
            # Load scaler
            scaler_path = self.models_dir / 'hsef_scaler.pkl'
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Loaded scaler")
            
            # Load feature names and classes
            features_path = self.models_dir / 'feature_names.json'
            with open(features_path, 'r') as f:
                data = json.load(f)
                self.feature_names = data['features']
                self.class_names = data['classes']
            print(f"✓ Loaded metadata: {len(self.feature_names)} features, {len(self.class_names)} classes")
            
            # Extract base models from stacking ensemble
            self.base_models = {
                'Random Forest': self.stacking_model.estimators_[0],
                'XGBoost': self.stacking_model.estimators_[1],
                'SVM': self.stacking_model.estimators_[2]
            }
            
            # Meta-classifier (Logistic Regression)
            self.meta_classifier = self.stacking_model.final_estimator_
            print(f"✓ Extracted base models: {list(self.base_models.keys())}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")
    
    def _initialize_shap(self):
        """Initialize SHAP explainers for each model"""
        print("\nInitializing SHAP explainers...")
        self.shap_explainers = {}
        
        # Load a small background dataset for SHAP
        try:
            df = pd.read_csv('All.csv')
            X = df.drop('URL_Type_obf_Type', axis=1)[self.feature_names]
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Use a small sample as background (for speed)
            background = X.sample(min(100, len(X)), random_state=42)
            background_scaled = self.scaler.transform(background)
            
            # Create explainer for stacking model
            self.shap_explainers['ensemble'] = shap.KernelExplainer(
                self.stacking_model.predict_proba,
                background_scaled[:50]  # Use even smaller sample for kernel
            )
            print("✓ SHAP explainer initialized")
            
        except Exception as e:
            print(f"⚠ Could not initialize SHAP: {e}")
            self.shap_explainers = {}
    
    def analyze_url(self, url_features, url_name="Unknown URL", actual_class=None):
        """
        Analyze a single URL's classification in detail
        
        Parameters:
        -----------
        url_features : dict or pd.Series
            Dictionary or Series with feature values
        url_name : str
            Name/identifier for the URL
        actual_class : str, optional
            True class if known (for misclassification analysis)
            
        Returns:
        --------
        dict : Complete analysis results
        """
        print("\n" + "="*70)
        print(f"ANALYZING: {url_name}")
        print("="*70 + "\n")
        
        # Prepare features
        if isinstance(url_features, dict):
            X = pd.DataFrame([url_features], columns=self.feature_names)
        else:
            X = pd.DataFrame([url_features.values], columns=self.feature_names)
        
        # Handle missing values
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # 1. Get base model predictions
        print("1️⃣  BASE MODEL PREDICTIONS")
        print("-" * 70)
        base_predictions = self._analyze_base_models(X_scaled)
        
        # 2. Get meta-layer analysis
        print("\n2️⃣  META-LAYER FUSION ANALYSIS")
        print("-" * 70)
        meta_analysis = self._analyze_meta_layer(X_scaled, base_predictions)
        
        # 3. Final prediction
        final_prediction = self.stacking_model.predict(X_scaled)[0]
        final_probabilities = self.stacking_model.predict_proba(X_scaled)[0]
        
        print("\n3️⃣  FINAL ENSEMBLE PREDICTION")
        print("-" * 70)
        print(f"Predicted Class: {self.class_names[final_prediction]}")
        print(f"Confidence: {final_probabilities.max():.2%}")
        print("\nAll Class Probabilities:")
        for i, cls in enumerate(self.class_names):
            print(f"  {cls:15s}: {final_probabilities[i]:6.2%}")
        
        # 4. SHAP analysis
        shap_analysis = None
        if SHAP_AVAILABLE and 'ensemble' in self.shap_explainers:
            print("\n4️⃣  SHAP FEATURE IMPORTANCE ANALYSIS")
            print("-" * 70)
            shap_analysis = self._analyze_shap(X_scaled, X, final_prediction)
        
        # 5. Misclassification analysis
        if actual_class:
            print("\n5️⃣  MISCLASSIFICATION ANALYSIS")
            print("-" * 70)
            misclass_analysis = self._analyze_misclassification(
                actual_class, final_prediction, base_predictions, 
                final_probabilities, shap_analysis
            )
        else:
            misclass_analysis = None
        
        # Compile results
        results = {
            'url': url_name,
            'actual_class': actual_class,
            'predicted_class': self.class_names[final_prediction],
            'confidence': float(final_probabilities.max()),
            'all_probabilities': {
                self.class_names[i]: float(p) 
                for i, p in enumerate(final_probabilities)
            },
            'base_models': base_predictions,
            'meta_layer': meta_analysis,
            'shap_analysis': shap_analysis,
            'misclassification_analysis': misclass_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate visualizations
        self._generate_visualizations(results, url_name)
        
        # Save detailed report
        self._save_report(results, url_name)
        
        return results
    
    def _analyze_base_models(self, X_scaled):
        """Analyze predictions from each base model"""
        predictions = {}
        
        for name, model in self.base_models.items():
            # Get prediction and probabilities
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            predictions[name] = {
                'predicted_class': self.class_names[pred],
                'predicted_index': int(pred),
                'confidence': float(proba.max()),
                'probabilities': {
                    self.class_names[i]: float(p) 
                    for i, p in enumerate(proba)
                }
            }
            
            # Print summary
            print(f"\n{name}:")
            print(f"  Predicted: {self.class_names[pred]}")
            print(f"  Confidence: {proba.max():.2%}")
            print(f"  Top 3 classes:")
            top_3 = np.argsort(proba)[-3:][::-1]
            for idx in top_3:
                print(f"    {self.class_names[idx]:15s}: {proba[idx]:6.2%}")
        
        return predictions
    
    def _analyze_meta_layer(self, X_scaled, base_predictions):
        """Analyze meta-layer fusion weights and contributions"""
        
        # Get base model predictions (stacked features)
        base_preds = self.stacking_model.transform(X_scaled)
        
        # Get meta-classifier coefficients
        # For multiclass LogisticRegression, coef_ shape is (n_classes, n_features)
        meta_coefs = self.meta_classifier.coef_
        
        # Calculate contribution of each base model to each class
        contributions = {}
        
        print("\nMeta-Layer Fusion Weights:")
        print("(How much each base model contributes to final decision)\n")
        
        # For each class
        for class_idx, class_name in enumerate(self.class_names):
            class_coefs = meta_coefs[class_idx]
            
            # Each base model outputs probabilities for all classes
            # So we have 3 models * 5 classes = 15 features going into meta-layer
            model_contributions = {}
            
            start_idx = 0
            for model_name in ['Random Forest', 'XGBoost', 'SVM']:
                # Get coefficients for this model's predictions
                model_coef_slice = class_coefs[start_idx:start_idx + len(self.class_names)]
                # Get actual predictions from this model
                model_pred_slice = base_preds[0][start_idx:start_idx + len(self.class_names)]
                # Calculate weighted contribution
                contribution = np.sum(model_coef_slice * model_pred_slice)
                model_contributions[model_name] = float(contribution)
                start_idx += len(self.class_names)
            
            contributions[class_name] = model_contributions
        
        # Find which class was predicted
        final_pred = self.stacking_model.predict(X_scaled)[0]
        final_class = self.class_names[final_pred]
        
        print(f"Contributions to predicted class '{final_class}':")
        for model_name, contrib in contributions[final_class].items():
            print(f"  {model_name:15s}: {contrib:+.4f}")
        
        # Normalize to show relative importance
        total = sum(abs(c) for c in contributions[final_class].values())
        if total > 0:
            print(f"\nRelative importance:")
            for model_name, contrib in contributions[final_class].items():
                pct = abs(contrib) / total * 100
                print(f"  {model_name:15s}: {pct:5.1f}%")
        
        return {
            'all_contributions': contributions,
            'predicted_class_contributions': contributions[final_class],
            'base_predictions_stacked': base_preds.tolist()
        }
    
    def _analyze_shap(self, X_scaled, X_original, predicted_class):
        """Perform SHAP analysis on the prediction"""
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainers['ensemble'].shap_values(X_scaled)
            
            # For multiclass, shap_values is a list of arrays (one per class)
            # Get SHAP values for the predicted class
            if isinstance(shap_values, list):
                shap_for_pred = shap_values[predicted_class][0]
            else:
                shap_for_pred = shap_values[0]
            
            # Get top features
            top_n = 10
            feature_importance = np.abs(shap_for_pred)
            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
            
            print(f"\nTop {top_n} Most Important Features (SHAP):")
            print(f"{'Feature':<30} {'Value':<10} {'SHAP Impact':<12} {'Direction'}")
            print("-" * 70)
            
            top_features = []
            for idx in top_indices:
                feature_name = self.feature_names[idx]
                feature_value = X_original.iloc[0, idx]
                shap_value = shap_for_pred[idx]
                direction = "→ " + self.class_names[predicted_class] if shap_value > 0 else "← away"
                
                print(f"{feature_name:<30} {feature_value:<10.2f} {shap_value:+.6f}    {direction}")
                
                top_features.append({
                    'feature': feature_name,
                    'value': float(feature_value),
                    'shap_value': float(shap_value),
                    'direction': direction
                })
            
            # Categorize features by domain
            feature_categories = self._categorize_features(top_features)
            
            print(f"\nFeature Breakdown by Domain:")
            for domain, features in feature_categories.items():
                if features:
                    print(f"  {domain}: {len(features)} features")
                    avg_impact = np.mean([abs(f['shap_value']) for f in features])
                    print(f"    Average impact: {avg_impact:.6f}")
            
            return {
                'top_features': top_features,
                'feature_categories': feature_categories,
                'shap_values_full': shap_for_pred.tolist()
            }
            
        except Exception as e:
            print(f"⚠ SHAP analysis failed: {e}")
            return None
    
    def _categorize_features(self, features):
        """Categorize features by domain (lexical, structural, entropy, semantic)"""
        categories = {
            'Lexical': [],
            'Structural': [],
            'Entropy': [],
            'Semantic': [],
            'Other': []
        }
        
        for feat in features:
            name = feat['feature'].lower()
            
            if any(x in name for x in ['len', 'length', 'char', 'digit', 'letter']):
                categories['Lexical'].append(feat)
            elif any(x in name for x in ['entropy', 'std', 'variance']):
                categories['Entropy'].append(feat)
            elif any(x in name for x in ['url', 'domain', 'path', 'dot', 'dash', 'slash']):
                categories['Structural'].append(feat)
            elif any(x in name for x in ['tld', 'brand', 'suspicious']):
                categories['Semantic'].append(feat)
            else:
                categories['Other'].append(feat)
        
        return categories
    
    def _analyze_misclassification(self, actual, predicted_idx, base_preds, 
                                   final_probs, shap_analysis):
        """Analyze why a misclassification occurred"""
        predicted = self.class_names[predicted_idx]
        
        is_misclassified = (actual != predicted)
        
        if not is_misclassified:
            print(f"✓ Correct classification: {actual}")
            return {'is_misclassified': False}
        
        print(f"✗ MISCLASSIFICATION DETECTED")
        print(f"  Actual: {actual}")
        print(f"  Predicted: {predicted}")
        print(f"  Confidence: {final_probs[predicted_idx]:.2%}")
        
        # Check if actual class was in top predictions
        actual_idx = self.class_names.index(actual)
        actual_prob = final_probs[actual_idx]
        
        print(f"\n  Probability for actual class '{actual}': {actual_prob:.2%}")
        print(f"  Difference: {(final_probs[predicted_idx] - actual_prob):.2%}")
        
        # Analyze base model agreement
        base_agreements = []
        for model_name, pred_info in base_preds.items():
            if pred_info['predicted_class'] == predicted:
                base_agreements.append(model_name)
        
        print(f"\n  Base models agreeing with wrong prediction: {len(base_agreements)}/3")
        for model in base_agreements:
            print(f"    - {model}")
        
        # Corrective insights
        insights = []
        
        if len(base_agreements) == 3:
            insights.append("All base models agreed on wrong class - systematic bias likely")
        elif len(base_agreements) == 2:
            insights.append("Majority of base models wrong - check feature engineering")
        else:
            insights.append("Meta-layer weighted minority opinion - fusion weights may need adjustment")
        
        if final_probs[predicted_idx] - actual_prob < 0.1:
            insights.append("LOW CONFIDENCE misclassification - close decision boundary")
            insights.append("→ Consider: probability threshold adjustment or 'uncertain' class")
        else:
            insights.append("HIGH CONFIDENCE misclassification - strong systematic error")
            insights.append("→ Consider: rebalancing training data or feature selection")
        
        # Check SHAP features
        if shap_analysis:
            top_feat = shap_analysis['top_features'][0]
            insights.append(f"Top influencing feature: {top_feat['feature']} = {top_feat['value']:.2f}")
            insights.append(f"→ Investigate if this feature causes false positives for {actual} URLs")
        
        print(f"\n  Corrective Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"    {i}. {insight}")
        
        return {
            'is_misclassified': True,
            'actual_class': actual,
            'predicted_class': predicted,
            'confidence': float(final_probs[predicted_idx]),
            'actual_class_probability': float(actual_prob),
            'probability_gap': float(final_probs[predicted_idx] - actual_prob),
            'base_model_agreement': base_agreements,
            'insights': insights
        }
    
    def _generate_visualizations(self, results, url_name):
        """Generate debug visualization plots"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = url_name.replace('/', '_').replace(':', '_')[:50]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'HSEF Debug Analysis: {url_name}', fontsize=16, fontweight='bold')
        
        # 1. Base Model Predictions
        ax1 = axes[0, 0]
        models = list(results['base_models'].keys())
        predictions = [results['base_models'][m]['predicted_class'] for m in models]
        confidences = [results['base_models'][m]['confidence'] * 100 for m in models]
        
        colors = ['#2ecc71' if p == results['predicted_class'] else '#e74c3c' 
                  for p in predictions]
        bars = ax1.barh(models, confidences, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Confidence (%)', fontweight='bold')
        ax1.set_title('Base Model Predictions', fontweight='bold')
        ax1.set_xlim(0, 100)
        
        for i, (bar, pred) in enumerate(zip(bars, predictions)):
            ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                    pred, va='center', fontsize=10)
        
        # 2. Meta-Layer Contributions
        ax2 = axes[0, 1]
        if results['meta_layer']:
            contrib = results['meta_layer']['predicted_class_contributions']
            models = list(contrib.keys())
            values = list(contrib.values())
            
            colors = ['#3498db' if v > 0 else '#e67e22' for v in values]
            bars = ax2.barh(models, values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Contribution Score', fontweight='bold')
            ax2.set_title(f'Meta-Layer Contributions\n(for {results["predicted_class"]})', 
                         fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            
            for bar, val in zip(bars, values):
                x_pos = val + (0.05 if val > 0 else -0.05)
                ax2.text(x_pos, bar.get_y() + bar.get_height()/2, 
                        f'{val:+.3f}', va='center', ha='left' if val > 0 else 'right')
        
        # 3. Class Probabilities
        ax3 = axes[1, 0]
        classes = list(results['all_probabilities'].keys())
        probs = [results['all_probabilities'][c] * 100 for c in classes]
        
        colors = ['#2ecc71' if c == results['predicted_class'] else '#95a5a6' 
                  for c in classes]
        bars = ax3.bar(range(len(classes)), probs, color=colors, alpha=0.7, 
                       edgecolor='black')
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.set_ylabel('Probability (%)', fontweight='bold')
        ax3.set_title('Final Class Probabilities', fontweight='bold')
        ax3.set_ylim(0, 100)
        
        for bar, prob in zip(bars, probs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Top SHAP Features
        ax4 = axes[1, 1]
        if results['shap_analysis']:
            top_feats = results['shap_analysis']['top_features'][:8]
            feat_names = [f['feature'][:20] for f in top_feats]
            shap_vals = [f['shap_value'] for f in top_feats]
            
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_vals]
            bars = ax4.barh(feat_names, shap_vals, color=colors, alpha=0.7, 
                           edgecolor='black')
            ax4.set_xlabel('SHAP Value (Impact)', fontweight='bold')
            ax4.set_title('Top Feature Contributions (SHAP)', fontweight='bold')
            ax4.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            
            for bar, val in zip(bars, shap_vals):
                x_pos = val + (0.0001 if val > 0 else -0.0001)
                ax4.text(x_pos, bar.get_y() + bar.get_height()/2,
                        f'{val:+.4f}', va='center', ha='left' if val > 0 else 'right',
                        fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'SHAP analysis not available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f'debug_analysis_{safe_name}_{timestamp}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {output_path}")
        plt.close()
    
    def _save_report(self, results, url_name):
        """Save detailed debug report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = url_name.replace('/', '_').replace(':', '_')[:50]
        
        # Save JSON report
        json_path = self.output_dir / f'debug_report_{safe_name}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved JSON report: {json_path}")
        
        # Save CSV summary
        csv_data = {
            'URL': [url_name],
            'Actual_Class': [results.get('actual_class', 'Unknown')],
            'Predicted_Class': [results['predicted_class']],
            'Confidence': [results['confidence']],
            'RF_Prediction': [results['base_models']['Random Forest']['predicted_class']],
            'RF_Confidence': [results['base_models']['Random Forest']['confidence']],
            'XGB_Prediction': [results['base_models']['XGBoost']['predicted_class']],
            'XGB_Confidence': [results['base_models']['XGBoost']['confidence']],
            'SVM_Prediction': [results['base_models']['SVM']['predicted_class']],
            'SVM_Confidence': [results['base_models']['SVM']['confidence']],
        }
        
        # Add class probabilities
        for cls, prob in results['all_probabilities'].items():
            csv_data[f'Prob_{cls}'] = [prob]
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / f'debug_summary_{safe_name}_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV summary: {csv_path}")
    
    def analyze_csv(self, csv_path, output_summary=True):
        """
        Analyze multiple URLs from a CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file with features and optional actual classes
        output_summary : bool
            Whether to create aggregate summary report
            
        Returns:
        --------
        list : Analysis results for each URL
        """
        print("\n" + "="*70)
        print(f"BATCH ANALYSIS: {csv_path}")
        print("="*70 + "\n")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} URLs for analysis\n")
        
        # Check if actual classes are present
        has_labels = 'URL_Type_obf_Type' in df.columns
        
        results_list = []
        
        # Analyze each URL
        for idx, row in df.iterrows():
            # Get features
            features = row.drop('URL_Type_obf_Type') if has_labels else row
            actual_class = row['URL_Type_obf_Type'] if has_labels else None
            url_name = f"URL_{idx+1}"
            
            print(f"\n{'='*70}")
            print(f"Processing {idx+1}/{len(df)}: {url_name}")
            print(f"{'='*70}")
            
            # Analyze
            result = self.analyze_url(features, url_name, actual_class)
            results_list.append(result)
        
        # Generate aggregate summary
        if output_summary:
            self._generate_aggregate_summary(results_list)
        
        print(f"\n{'='*70}")
        print("BATCH ANALYSIS COMPLETE")
        print(f"{'='*70}\n")
        print(f"✓ Analyzed {len(results_list)} URLs")
        print(f"✓ Results saved to: {self.output_dir}")
        
        return results_list
    
    def _generate_aggregate_summary(self, results_list):
        """Generate aggregate summary of batch analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Compile statistics
        total = len(results_list)
        misclassified = sum(1 for r in results_list 
                          if r.get('misclassification_analysis', {}).get('is_misclassified', False))
        
        # Base model agreement
        rf_correct = sum(1 for r in results_list 
                        if r['base_models']['Random Forest']['predicted_class'] == r['predicted_class'])
        xgb_correct = sum(1 for r in results_list 
                         if r['base_models']['XGBoost']['predicted_class'] == r['predicted_class'])
        svm_correct = sum(1 for r in results_list 
                         if r['base_models']['SVM']['predicted_class'] == r['predicted_class'])
        
        print(f"\n{'='*70}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*70}\n")
        print(f"Total URLs analyzed: {total}")
        if misclassified > 0:
            print(f"Misclassifications: {misclassified} ({misclassified/total*100:.1f}%)")
            print(f"Accuracy: {(total-misclassified)/total*100:.1f}%")
        
        print(f"\nBase Model Agreement with Final Prediction:")
        print(f"  Random Forest: {rf_correct}/{total} ({rf_correct/total*100:.1f}%)")
        print(f"  XGBoost:       {xgb_correct}/{total} ({xgb_correct/total*100:.1f}%)")
        print(f"  SVM:           {svm_correct}/{total} ({svm_correct/total*100:.1f}%)")
        
        # Save aggregate CSV
        aggregate_data = []
        for r in results_list:
            row = {
                'URL': r['url'],
                'Actual': r.get('actual_class', 'Unknown'),
                'Predicted': r['predicted_class'],
                'Confidence': r['confidence'],
                'Misclassified': r.get('misclassification_analysis', {}).get('is_misclassified', False),
                'RF_Pred': r['base_models']['Random Forest']['predicted_class'],
                'XGB_Pred': r['base_models']['XGBoost']['predicted_class'],
                'SVM_Pred': r['base_models']['SVM']['predicted_class'],
            }
            aggregate_data.append(row)
        
        df_agg = pd.DataFrame(aggregate_data)
        agg_path = self.output_dir / f'aggregate_summary_{timestamp}.csv'
        df_agg.to_csv(agg_path, index=False)
        print(f"\n✓ Saved aggregate summary: {agg_path}")


def analyze_url(url_features, url_name="URL", actual_class=None):
    """
    Convenience function to analyze a single URL
    
    Parameters:
    -----------
    url_features : dict or pd.Series
        Feature values for the URL
    url_name : str
        Identifier for the URL
    actual_class : str, optional
        True class if known
        
    Returns:
    --------
    dict : Complete analysis results
    """
    debugger = HSEFDebugger()
    return debugger.analyze_url(url_features, url_name, actual_class)


def analyze_csv(csv_path):
    """
    Convenience function to analyze URLs from a CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with features
        
    Returns:
    --------
    list : Analysis results for each URL
    """
    debugger = HSEFDebugger()
    return debugger.analyze_csv(csv_path)


if __name__ == "__main__":
    print("""
    HSEF Model Debugger
    ===================
    
    Usage:
    
    1. Analyze single URL:
       from hsef_debugger import analyze_url
       import pandas as pd
       
       # Load data
       df = pd.read_csv('All.csv')
       row = df.iloc[0]
       features = row.drop('URL_Type_obf_Type')
       actual = row['URL_Type_obf_Type']
       
       # Analyze
       result = analyze_url(features, "example_url", actual)
    
    2. Analyze batch from CSV:
       from hsef_debugger import analyze_csv
       
       results = analyze_csv('test_batch.csv')
    
    3. Advanced usage:
       from hsef_debugger import HSEFDebugger
       
       debugger = HSEFDebugger()
       # Use debugger methods directly
    """)
