"""
HSEF Web Application
Flask-based web interface for URL classification using the trained HSEF model
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import json
from datetime import datetime
import shap
from url_feature_extractor import URLFeatureExtractor
import tldextract

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and scaler
model = None
scaler = None
feature_names = None
class_names = None
model_info = None
feature_extractor = None
shap_explainer = None
base_models = None

# Model directory - works from both root and app/ folder
MODEL_DIR = Path(__file__).parent.parent / 'models'
if not MODEL_DIR.exists():
    MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# Trusted domains whitelist (known benign sites)
TRUSTED_DOMAINS = {
    'youtube.com', 'google.com', 'github.com', 'microsoft.com',
    'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com',
    'instagram.com', 'reddit.com', 'wikipedia.org', 'stackoverflow.com',
    'apple.com', 'netflix.com', 'spotify.com', 'zoom.us',
    'dropbox.com', 'adobe.com', 'salesforce.com', 'oracle.com'
}


def load_model():
    """Load the trained HSEF model and preprocessing objects"""
    global model, scaler, feature_names, class_names, model_info, feature_extractor, shap_explainer, base_models
    
    try:
        model_path = MODEL_DIR / 'hsef_model.pkl'
        scaler_path = MODEL_DIR / 'hsef_scaler.pkl'
        features_path = MODEL_DIR / 'feature_names.json'
        info_path = MODEL_DIR / 'model_info.json'
        
        if not model_path.exists():
            return False, "Model file not found. Please train the model first."
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                data = json.load(f)
                feature_names = data['features']
                class_names = data['classes']
        
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        
        # Initialize feature extractor
        feature_extractor = URLFeatureExtractor()
        
        # Extract base models from the stacking classifier
        try:
            if hasattr(model, 'estimators_'):
                base_models = {
                    'Random Forest': model.estimators_[0],
                    'XGBoost': model.estimators_[1],
                    'SVM': model.estimators_[2]
                }
            else:
                base_models = None
        except Exception as e:
            print(f"Warning: Could not extract base models: {e}")
            base_models = None
        
        # Initialize SHAP explainer for interpretability
        try:
            # Load a small sample of training data for SHAP background
            df = pd.read_csv('All.csv')
            sample = df.drop('URL_Type_obf_Type', axis=1).sample(n=100, random_state=42)
            sample = sample[feature_names]
            sample = sample.fillna(0).replace([np.inf, -np.inf], 0)
            
            if scaler:
                sample_scaled = scaler.transform(sample)
            else:
                sample_scaled = sample.values
            
            # Create SHAP explainer
            shap_explainer = shap.KernelExplainer(model.predict_proba, sample_scaled)
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            shap_explainer = None
        
        return True, "Model loaded successfully"
        
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


def extract_url_features(url):
    """
    Extract features from a single URL
    For now, returns dummy features matching the training data structure
    In production, implement full feature extraction logic
    """
    # Create dummy features with proper structure
    # This should match your actual feature extraction pipeline
    
    if feature_names is None:
        raise ValueError("Feature names not loaded")
    
    # Initialize with zeros
    features = {name: 0.0 for name in feature_names}
    
    # Basic URL features that can be extracted
    try:
        features['urlLen'] = len(url) if 'urlLen' in features else 0
        features['NumberofDotsinURL'] = url.count('.') if 'NumberofDotsinURL' in features else 0
        features['NumDash'] = url.count('-') if 'NumDash' in features else 0
        features['AtSymbol'] = 1 if '@' in url else 0 if 'AtSymbol' in features else 0
        
        # Add more feature extraction logic here based on your actual features
        # For now, using zeros as placeholders for missing features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
    
    return features


@app.route('/')
def index():
    """Main page"""
    # Check if model is loaded
    is_loaded = model is not None
    return render_template('index.html', 
                         model_loaded=is_loaded,
                         model_info=model_info if is_loaded else None)


@app.route('/test')
def test_page():
    """Test page for API debugging"""
    with open('test_page.html', 'r') as f:
        return f.read()


@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """Load the model"""
    success, message = load_model()
    return jsonify({
        'success': success,
        'message': message,
        'model_info': model_info
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict single URL with real feature extraction and interpretability"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run quick_setup.py first.'}), 400
    
    if feature_extractor is None:
        return jsonify({'error': 'Feature extractor not initialized.'}), 400
    
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Check if domain is in whitelist
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
        
        is_whitelisted = domain in TRUSTED_DOMAINS
        
        if is_whitelisted:
            # Trusted domain - override with benign
            return jsonify({
                'url': url,
                'prediction': 'benign',
                'confidence': 0.99,
                'probabilities': {
                    'benign': 0.99,
                    'Defacement': 0.0025,
                    'malware': 0.0025,
                    'phishing': 0.0025,
                    'spam': 0.0025
                },
                'feature_summary': {
                    'domain': domain,
                    'whitelisted': True
                },
                'mode': 'whitelist_override',
                'note': f'Domain {domain} is in trusted whitelist',
                'timestamp': datetime.now().isoformat()
            })
        
        # Extract features from the URL
        try:
            features = feature_extractor.extract_features(url)
            feature_vector = feature_extractor.get_feature_vector(feature_names)
        except Exception as e:
            return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500
        
        # Convert to DataFrame for proper handling
        X = pd.DataFrame([feature_vector], columns=feature_names)
        
        # Handle missing values - fill with 0
        X = X.fillna(0)
        
        # Handle infinity values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Get predictions from base models (if available)
        base_predictions = {}
        if base_models:
            try:
                for name, base_model in base_models.items():
                    base_pred = base_model.predict(X_scaled)[0]
                    base_proba = base_model.predict_proba(X_scaled)[0]
                    base_predictions[name] = {
                        'prediction': class_names[base_pred],
                        'confidence': float(base_proba.max()),
                        'probabilities': {
                            class_names[i]: float(prob) 
                            for i, prob in enumerate(base_proba)
                        }
                    }
            except Exception as e:
                print(f"Warning: Could not get base model predictions: {e}")
        
        # Final ensemble prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(probabilities.max())
        
        # Add warning for low confidence or feature mismatch
        warnings = []
        if confidence < 0.80:
            warnings.append(f'Low confidence ({confidence*100:.1f}%) - manual review recommended')
        
        # Note about feature mismatch
        warnings.append('Note: Model trained on different feature extraction method. See FEATURE_MISMATCH_ISSUE.md')
        
        # Analyze meta-layer contributions (if available)
        meta_analysis = None
        try:
            if hasattr(model, 'final_estimator_'):
                # Get meta-features (base model predictions)
                meta_features = np.column_stack([
                    estimator.predict_proba(X_scaled) 
                    for estimator in model.estimators_
                ])
                
                # Get meta-layer coefficients
                if hasattr(model.final_estimator_, 'coef_'):
                    meta_coef = model.final_estimator_.coef_[prediction]
                    
                    # Calculate contribution of each base model
                    contributions = {}
                    total_contrib = 0
                    
                    for i, name in enumerate(['Random Forest', 'XGBoost', 'SVM']):
                        # Each base model contributes n_classes values
                        start_idx = i * len(class_names)
                        end_idx = start_idx + len(class_names)
                        base_contrib = np.sum(meta_coef[start_idx:end_idx] * meta_features[0, start_idx:end_idx])
                        contributions[name] = float(base_contrib)
                        total_contrib += abs(base_contrib)
                    
                    # Normalize to percentages
                    meta_analysis = {
                        name: {
                            'weight': contrib,
                            'percentage': (abs(contrib) / total_contrib * 100) if total_contrib > 0 else 0
                        }
                        for name, contrib in contributions.items()
                    }
        except Exception as e:
            print(f"Warning: Could not analyze meta-layer: {e}")
        
        # Get SHAP values for top contributing features
        shap_analysis = None
        if shap_explainer:
            try:
                shap_values = shap_explainer.shap_values(X_scaled)
                
                # Get SHAP values for the predicted class
                if isinstance(shap_values, list):
                    class_shap = shap_values[prediction][0]
                else:
                    class_shap = shap_values[0, :, prediction]
                
                # Get top 5 features by absolute SHAP value
                top_indices = np.argsort(np.abs(class_shap))[-5:][::-1]
                
                shap_analysis = {
                    'top_features': [
                        {
                            'feature': feature_names[idx],
                            'value': float(X.iloc[0, idx]),
                            'shap_value': float(class_shap[idx]),
                            'impact': 'increases' if class_shap[idx] > 0 else 'decreases'
                        }
                        for idx in top_indices
                    ]
                }
            except Exception as e:
                print(f"Warning: SHAP analysis failed: {e}")
        
        # Format results
        result = {
            'url': url,
            'prediction': class_names[prediction],
            'confidence': confidence,
            'probabilities': {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            'base_models': base_predictions if base_predictions else None,
            'meta_layer_analysis': meta_analysis,
            'shap_analysis': shap_analysis,
            'feature_summary': {
                'url_length': features['urlLen'],
                'domain_length': features['domainlength'],
                'path_length': features['pathLength'],
                'has_ip_address': features['ISIpAddressInDomainName'] == 1,
                'is_executable': features['executable'] == 1,
                'has_sensitive_word': features['URL_sensitiveWord'] == 1,
                'entropy_url': features['Entropy_URL'],
                'entropy_domain': features['Entropy_Domain'],
                'number_of_dots': features['NumberofDotsinURL']
            },
            'warnings': warnings,
            'mode': 'real_feature_extraction',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    """Predict batch of URLs from CSV file - supports both feature-rich and URL-only formats"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run quick_setup.py first.'}), 400
    
    if feature_extractor is None:
        return jsonify({'error': 'Feature extractor not initialized.'}), 400
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Determine CSV format
        has_features = all(feature in df.columns for feature in feature_names)
        has_url_column = 'url' in df.columns or 'URL' in df.columns
        has_target = 'URL_Type_obf_Type' in df.columns
        
        if has_target:
            # Format 1: Training data format (features + target)
            X = df.drop('URL_Type_obf_Type', axis=1)
            actual_classes = df['URL_Type_obf_Type'].values
            urls = None
        elif has_features:
            # Format 2: Features only
            X = df[feature_names]
            actual_classes = None
            urls = None
        elif has_url_column:
            # Format 3: URLs only - extract features
            url_col = 'url' if 'url' in df.columns else 'URL'
            urls = df[url_col].values
            actual_classes = df['label'].values if 'label' in df.columns else None
            
            # Extract features for all URLs
            print(f"Extracting features for {len(urls)} URLs...")
            feature_list = []
            
            for i, url in enumerate(urls):
                try:
                    features = feature_extractor.extract_features(url)
                    feature_vector = feature_extractor.get_feature_vector(feature_names)
                    feature_list.append(feature_vector)
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1}/{len(urls)} URLs")
                except Exception as e:
                    print(f"  Warning: Failed to extract features for URL {i}: {e}")
                    # Use zeros as fallback
                    feature_list.append([0] * len(feature_names))
            
            X = pd.DataFrame(feature_list, columns=feature_names)
            print(f"Feature extraction complete!")
        else:
            return jsonify({
                'error': 'CSV format not recognized. Supported formats:\n'
                        '1. Full features with URL_Type_obf_Type column (training data format)\n'
                        '2. Full features without target column\n'
                        '3. URLs only with columns: url (or URL), optional: label'
            }), 400
        
        # Ensure we have the right feature order
        X = X[feature_names]
        
        # Handle missing values - fill with 0
        X = X.fillna(0)
        
        # Handle infinity values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result_item = {
                'index': i,
                'prediction': class_names[pred],
                'confidence': float(probs.max()),
                'probabilities': {
                    class_names[j]: float(prob) 
                    for j, prob in enumerate(probs)
                }
            }
            
            if urls is not None:
                result_item['url'] = urls[i]
            
            if actual_classes is not None:
                result_item['actual_class'] = actual_classes[i]
                result_item['correct'] = (class_names[pred] == actual_classes[i])
            
            results.append(result_item)
        
        # Calculate accuracy if we have actual classes
        accuracy = None
        if actual_classes is not None:
            correct = sum(1 for r in results if r.get('correct', False))
            accuracy = (correct / len(results)) * 100
        
        return jsonify({
            'total': len(results),
            'results': results,
            'accuracy': accuracy,
            'mode': 'url_extraction' if urls is not None else 'feature_based',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def api_model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    return jsonify({
        'loaded': True,
        'info': model_info,
        'classes': class_names,
        'n_features': len(feature_names)
    })


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Try to load model on startup
    success, message = load_model()
    if success:
        print(f"✓ {message}")
    else:
        print(f"⚠ {message}")
        print("  You can train and save the model, then restart the app.")
    
    # Run the app
    print("\n" + "="*70)
    print("HSEF Web Application Starting...")
    print("="*70)
    print("\nAccess the web interface at: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
