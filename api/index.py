"""
HSEF Vercel Serverless API
Lightweight Flask API for Vercel deployment
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# Global variables
model = None
scaler = None
feature_names = None
class_names = None

# Use compressed model for Vercel
MODEL_DIR = Path(__file__).parent.parent / 'models'

def load_model():
    """Load the compressed model for Vercel"""
    global model, scaler, feature_names, class_names
    
    try:
        # Use the Vercel-optimized compressed model
        model_path = MODEL_DIR / 'hsef_model_vercel.pkl'
        if not model_path.exists():
            model_path = MODEL_DIR / 'hsef_model_compressed.pkl'
        if not model_path.exists():
            model_path = MODEL_DIR / 'hsef_model.pkl'
            
        scaler_path = MODEL_DIR / 'hsef_scaler.pkl'
        features_path = MODEL_DIR / 'feature_names.json'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                data = json.load(f)
                feature_names = data['features']
                class_names = data['classes']
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
load_model()

@app.route('/')
def index():
    """Health check and info"""
    return jsonify({
        'status': 'online',
        'service': 'HSEF Phishing Detection API',
        'model_loaded': model is not None,
        'endpoints': {
            '/': 'This info page',
            '/api/health': 'Health check',
            '/api/predict': 'POST - Predict URL (JSON: {"url": "..."})'
        }
    })

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict single URL"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Basic feature extraction (simplified for Vercel)
        features = {name: 0.0 for name in feature_names}
        features['urlLen'] = len(url)
        features['NumberofDotsinURL'] = url.count('.')
        features['NumDash'] = url.count('-')
        features['AtSymbol'] = 1 if '@' in url else 0
        
        # Create feature vector
        X = pd.DataFrame([features])[feature_names]
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        return jsonify({
            'url': url,
            'prediction': class_names[prediction],
            'confidence': float(probabilities.max()),
            'probabilities': {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel handler
def handler(request):
    return app(request)

if __name__ == '__main__':
    app.run(debug=True)
