"""
DEMO App - Uses Random Forest only (no stacking, trains fast)
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    from url_feature_extractor import URLFeatureExtractor
    extractor = URLFeatureExtractor()
    EXTRACTOR_AVAILABLE = True
except:
    EXTRACTOR_AVAILABLE = False
    extractor = None

app = Flask(__name__)

# Global
model = None
scaler = None
class_names = ['Defacement', 'benign', 'malware', 'phishing', 'spam']

def train_demo_model():
    """Train simple Random Forest"""
    print("Training demo model (RF only)...")
    
    df = pd.read_csv('All.csv')
    X = df.drop('URL_Type_obf_Type', axis=1)
    y = df['URL_Type_obf_Type']
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Simple RF - trains in ~2 minutes
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training... (~2 min)")
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✓ Accuracy: {acc*100:.2f}%")
    
    # Save
    Path('models_demo').mkdir(exist_ok=True)
    with open('models_demo/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=4)
    with open('models_demo/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f, protocol=4)
    
    return model, scaler

def load_model():
    global model, scaler
    
    model_path = Path('models_demo/rf_model.pkl')
    scaler_path = Path('models_demo/scaler.pkl')
    
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Model loaded")
            return
        except:
            pass
    
    model, scaler = train_demo_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Whitelist
        trusted = ['youtube.com', 'google.com', 'github.com', 'microsoft.com']
        for domain in trusted:
            if domain in url.lower():
                return jsonify({
                    'url': url,
                    'prediction': 'benign',
                    'confidence': 0.99,
                    'probabilities': {'Defacement': 0.001, 'benign': 0.99, 'malware': 0.001, 'phishing': 0.004, 'spam': 0.004},
                    'feature_summary': {'url_length': len(url), 'domain_length': len(domain), 'path_length': 0, 'has_ip_address': False, 'is_executable': False, 'has_sensitive_word': False, 'entropy_url': 3.5, 'entropy_domain': 3.0, 'number_of_dots': 1},
                    'warnings': ['Trusted domain'],
                    'mode': 'whitelist',
                    'timestamp': datetime.now().isoformat()
                })
        
        if not extractor:
            return jsonify({'error': 'Feature extractor unavailable'}), 500
        
        features = extractor.extract_features(url)
        X = pd.DataFrame([features]).fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        confidence = float(probabilities.max())
        
        return jsonify({
            'url': url,
            'prediction': class_names[prediction],
            'confidence': confidence,
            'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(probabilities)},
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
            'mode': 'rf_model',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n"  + "="*60)
    print(" "*15 + "DEMO: Random Forest URL Classifier")
    print("="*60)
    
    load_model()
    
    print("\n✓ Starting server at http://localhost:5000")
    print("Press CTRL+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
