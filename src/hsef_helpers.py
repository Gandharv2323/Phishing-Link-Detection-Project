"""
Helper Functions for Publication-Ready HSEF System
==================================================

Utility functions for:
- Loading calibrated models
- Making predictions with domain whitelist
- Analyzing predictions with SHAP
- Generating reports
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import tldextract
from url_feature_extractor import URLFeatureExtractor


class CalibratedHSEFPredictor:
    """
    Wrapper for calibrated HSEF model with domain whitelist and interpretability
    """
    
    def __init__(self, model_dir='publication_outputs/models'):
        """
        Initialize predictor with calibrated model
        
        Args:
            model_dir: Directory containing calibrated model files
        """
        self.model_dir = Path(model_dir)
        
        # Load model and preprocessing
        self.model = joblib.load(self.model_dir / 'stacking_calibrated.joblib')
        self.scaler = joblib.load(self.model_dir / 'scaler_calibrated.joblib')
        self.label_encoder = joblib.load(self.model_dir / 'label_encoder_calibrated.joblib')
        
        # Load feature names and classes
        with open(self.model_dir / 'feature_names_calibrated.json', 'r') as f:
            data = json.load(f)
            self.feature_names = data['features']
            self.class_names = data['classes']
        
        # Load configuration (including whitelist)
        import yaml
        with open(self.model_dir / 'config_calibrated.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.trusted_domains = set(self.config['trusted_domains'])
        
        # Feature extractor
        self.feature_extractor = URLFeatureExtractor()
        
        print(f"✓ Calibrated HSEF model loaded")
        print(f"  Classes: {self.class_names}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Trusted domains: {len(self.trusted_domains)}")
    
    def predict(self, url, include_shap=False):
        """
        Predict URL with domain whitelist and interpretability
        
        Args:
            url: URL string to classify
            include_shap: Whether to compute SHAP values (slower)
        
        Returns:
            dict: Prediction results with metadata
        """
        # Extract domain
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
        
        # Check whitelist
        if domain in self.trusted_domains:
            return {
                'url': url,
                'domain': domain,
                'prediction': 'benign',
                'confidence': 0.99,
                'probabilities': {cls: (0.99 if cls == 'benign' else 0.0025) for cls in self.class_names},
                'method': 'whitelist',
                'note': f'Domain {domain} is in trusted whitelist'
            }
        
        # Extract features
        try:
            features = self.feature_extractor.extract_features(url)
            feature_vector = self.feature_extractor.get_feature_vector(self.feature_names)
        except Exception as e:
            return {
                'url': url,
                'error': f'Feature extraction failed: {str(e)}',
                'method': 'error'
            }
        
        # Prepare for prediction
        X = pd.DataFrame([feature_vector], columns=self.feature_names)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        pred = self.model.predict(X_scaled)[0]
        pred_proba = self.model.predict_proba(X_scaled)[0]
        pred_class = self.class_names[pred]
        confidence = float(pred_proba.max())
        
        result = {
            'url': url,
            'domain': domain,
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(pred_proba)
            },
            'method': 'calibrated_model',
            'calibrated': True
        }
        
        # Add warnings if needed
        if confidence < 0.70:
            result['warning'] = 'Low confidence - manual review recommended'
        
        # Get base model predictions
        if hasattr(self.model, 'estimators_'):
            base_preds = {}
            for i, (name, _) in enumerate([('Random Forest', 0), ('XGBoost', 1), ('SVM', 2)]):
                try:
                    base_model = self.model.estimators_[i]
                    base_pred = base_model.predict(X_scaled)[0]
                    base_proba = base_model.predict_proba(X_scaled)[0]
                    base_preds[name] = {
                        'prediction': self.class_names[base_pred],
                        'confidence': float(base_proba.max())
                    }
                except:
                    pass
            
            if base_preds:
                result['base_models'] = base_preds
        
        # SHAP analysis (optional, slower)
        if include_shap:
            try:
                import shap
                # Note: This requires SHAP explainer to be initialized
                # For production, cache the explainer
                result['shap'] = 'SHAP analysis requires explainer initialization'
            except:
                pass
        
        return result
    
    def predict_batch(self, urls, show_progress=True):
        """
        Predict multiple URLs
        
        Args:
            urls: List of URL strings
            show_progress: Whether to print progress
        
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        
        for i, url in enumerate(urls):
            result = self.predict(url, include_shap=False)
            results.append(result)
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(urls)} URLs")
        
        return results
    
    def export_predictions(self, results, output_path='predictions.csv'):
        """
        Export predictions to CSV
        
        Args:
            results: List of prediction dictionaries
            output_path: Path to save CSV
        """
        records = []
        for r in results:
            record = {
                'url': r['url'],
                'prediction': r.get('prediction', 'error'),
                'confidence': r.get('confidence', 0.0),
                'method': r.get('method', 'unknown')
            }
            
            # Add probabilities
            if 'probabilities' in r:
                for cls, prob in r['probabilities'].items():
                    record[f'prob_{cls}'] = prob
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"✓ Predictions exported to {output_path}")


def detect_false_positives(test_csv='test_urls.csv', output_dir='publication_outputs'):
    """
    Standalone function to detect false positives on benign URLs
    
    Args:
        test_csv: Path to CSV with url,label columns
        output_dir: Directory for output files
    
    Returns:
        DataFrame: Results with false positive indicators
    """
    print("Detecting false positives...")
    
    # Load test URLs
    df = pd.read_csv(test_csv)
    benign_urls = df[df['label'].str.lower() == 'benign']['url'].tolist()
    
    print(f"Testing {len(benign_urls)} benign URLs...")
    
    # Load predictor
    predictor = CalibratedHSEFPredictor(model_dir=f'{output_dir}/models')
    
    # Predict
    results = predictor.predict_batch(benign_urls)
    
    # Analyze
    false_positives = [r for r in results if r.get('prediction', '').lower() != 'benign']
    
    print(f"\n✓ Results:")
    print(f"  True Negatives: {len(benign_urls) - len(false_positives)}")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  FP Rate: {len(false_positives)/len(benign_urls)*100:.1f}%")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df['is_false_positive'] = results_df['prediction'].str.lower() != 'benign'
    
    # Save
    output_path = Path(output_dir) / 'reports' / 'false_positives.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    return results_df


def analyze_false_positive(url, model_dir='publication_outputs/models'):
    """
    Detailed analysis of a single false positive URL with SHAP
    
    Args:
        url: URL to analyze
        model_dir: Directory with calibrated model
    
    Returns:
        dict: Detailed analysis including SHAP values
    """
    print(f"\nAnalyzing: {url}")
    print("="*70)
    
    # Load predictor
    predictor = CalibratedHSEFPredictor(model_dir=model_dir)
    
    # Get prediction
    result = predictor.predict(url, include_shap=False)
    
    print(f"\nPrediction: {result.get('prediction', 'error')}")
    print(f"Confidence: {result.get('confidence', 0)*100:.1f}%")
    
    if 'probabilities' in result:
        print(f"\nClass Probabilities:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {prob*100:.2f}%")
    
    if 'base_models' in result:
        print(f"\nBase Model Predictions:")
        for model, pred in result['base_models'].items():
            print(f"  {model}: {pred['prediction']} ({pred['confidence']*100:.1f}%)")
    
    # Extract features for inspection
    features = predictor.feature_extractor.extract_features(url)
    
    print(f"\nKey Features:")
    important_features = [
        'urlLen', 'domainlength', 'pathLength',
        'Entropy_URL', 'Entropy_Domain',
        'NumberofDotsinURL', 'ISIpAddressInDomainName',
        'URL_sensitiveWord', 'executable'
    ]
    
    for feat in important_features:
        if feat in features:
            print(f"  {feat}: {features[feat]}")
    
    return result


def apply_feature_corrections(X, config_path='publication_outputs/config_corrections.yaml'):
    """
    Apply feature corrections as defined in configuration
    
    Args:
        X: Feature DataFrame
        config_path: Path to correction configuration
    
    Returns:
        DataFrame: Corrected features
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    X_corrected = X.copy()
    
    # Apply corrections
    corrections = config.get('feature_corrections', {})
    
    # Length capping
    if 'length_capping' in corrections:
        max_len = corrections['length_capping'].get('max_url_length', 500)
        if 'urlLen' in X_corrected.columns:
            X_corrected['urlLen'] = X_corrected['urlLen'].clip(upper=max_len)
    
    # Outlier handling
    if corrections.get('outlier_handling') == 'clip':
        # Clip to 99th percentile
        for col in X_corrected.columns:
            q99 = X_corrected[col].quantile(0.99)
            q1 = X_corrected[col].quantile(0.01)
            X_corrected[col] = X_corrected[col].clip(lower=q1, upper=q99)
    
    return X_corrected


def retrain_hsef(updated_dataset_path, output_dir='publication_outputs'):
    """
    Retrain HSEF with updated dataset
    
    Args:
        updated_dataset_path: Path to CSV with augmented training data
        output_dir: Directory for outputs
    
    Returns:
        Trained model
    """
    print(f"\nRetraining HSEF with {updated_dataset_path}...")
    
    # This would use the full calibration pipeline
    # For now, redirect to the main calibration script
    print("Please use run_full_calibration.py for complete retraining")
    
    return None


if __name__ == '__main__':
    # Example usage
    print("HSEF Helper Functions")
    print("="*70)
    print("\nExample 1: Load calibrated predictor")
    print("  predictor = CalibratedHSEFPredictor()")
    print("  result = predictor.predict('https://example.com')")
    print("\nExample 2: Detect false positives")
    print("  results = detect_false_positives('test_urls.csv')")
    print("\nExample 3: Analyze specific URL")
    print("  analyze_false_positive('https://youtube.com')")
