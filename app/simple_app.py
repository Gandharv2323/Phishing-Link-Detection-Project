"""
Simple Flask App - Trains model on first run if not available
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    from url_feature_extractor import URLFeatureExtractor
    extractor = URLFeatureExtractor()
    EXTRACTOR_AVAILABLE = True
except:
    EXTRACTOR_AVAILABLE = False
    extractor = None

app = Flask(__name__)

# Global variables
model = None
scaler = None
class_names = ['Defacement', 'benign', 'malware', 'phishing', 'spam']
feature_names = None

def train_quick_model():
    """Train a quick model for demo"""
    print("\n" + "="*70)
    print("Training Quick HSEF Model...")
    print("="*70)
    
    # Load data
    df = pd.read_csv('All.csv')
    X = df.drop('URL_Type_obf_Type', axis=1)
    y = df['URL_Type_obf_Type']
    
    # Handle missing/inf
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Building model (3 learners)...")
    
    # Simple fast model
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=1)
    
    if XGB_AVAILABLE:
        xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=6, device='cpu', random_state=42, n_jobs=1)
    else:
        xgb_model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=1)
    
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    
    meta_lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=1)
    
    estimators = [('rf', rf), ('xgb', xgb_model), ('svm', svm)]
    
    print("Training... (may take 3-5 minutes)")
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_lr,
        cv=3,
        n_jobs=1
    )
    
    model.fit(X_train, y_train)
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"✓ Model trained! Accuracy: {acc*100:.2f}%")
    
    # Save
    models_dir = Path('models_simple')
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=4)
    
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f, protocol=4)
    
    print(f"✓ Model saved to {models_dir}/")
    
    return model, scaler, list(X.columns)

def load_or_train_model():
    """Load existing model or train new one"""
    global model, scaler, feature_names
    
    model_path = Path('models_simple/model.pkl')
    scaler_path = Path('models_simple/scaler.pkl')
    
    if model_path.exists() and scaler_path.exists():
        try:
            print("Loading existing model...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Get feature names from data
            df = pd.read_csv('All.csv')
            X = df.drop('URL_Type_obf_Type', axis=1)
            feature_names = list(X.columns)
            
            print("✓ Model loaded successfully")
            return
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            print("Training new model...")
    
    # Train new model
    model, scaler, feature_names = train_quick_model()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict URL classification"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Check whitelist
        trusted_domains = [
            'youtube.com', 'google.com', 'github.com', 'microsoft.com',
            'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com'
        ]
        
        for domain in trusted_domains:
            if domain in url.lower():
                return jsonify({
                    'url': url,
                    'prediction': 'benign',
                    'confidence': 0.99,
                    'probabilities': {
                        'Defacement': 0.001,
                        'benign': 0.99,
                        'malware': 0.001,
                        'phishing': 0.004,
                        'spam': 0.004
                    },
                    'feature_summary': {
                        'url_length': len(url),
                        'domain_length': len(domain),
                        'path_length': 0,
                        'has_ip_address': False,
                        'is_executable': False,
                        'has_sensitive_word': False,
                        'entropy_url': 3.5,
                        'entropy_domain': 3.0,
                        'number_of_dots': 1
                    },
                    'warnings': ['Whitelisted domain - trusted source'],
                    'mode': 'whitelist',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Extract features
        if extractor:
            features = extractor.extract_features(url)
            X = pd.DataFrame([features])
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
        else:
            return jsonify({'error': 'Feature extractor not available'}), 500
        
        # Predict
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(probabilities.max())
        
        result = {
            'url': url,
            'prediction': class_names[prediction],
            'confidence': confidence,
            'probabilities': {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
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
            'warnings': [],
            'mode': 'real_extraction',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*20 + "SIMPLE HSEF WEB APP")
    print("="*70)
    
    load_or_train_model()
    
    print("\n" + "="*70)
    print("Starting Flask server...")
    print("Access at: http://localhost:5000")
    print("Press CTRL+C to stop")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
