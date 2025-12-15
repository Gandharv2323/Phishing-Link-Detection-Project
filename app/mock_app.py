"""
MOCK Demo App - Works instantly without training
Shows demo predictions for testing the web interface
"""

from flask import Flask, render_template, request, jsonify
from datetime import datetime
import random

app = Flask(__name__)

class_names = ['Defacement', 'benign', 'malware', 'phishing', 'spam']

@app.route('/')
def index():
    return render_template('index.html', model_loaded=True)

@app.route('/api/status')
def status():
    """Return model status"""
    return jsonify({
        'model_loaded': True,
        'model_type': 'HSEF Demo (Rule-based)',
        'mode': 'demo',
        'accuracy': 98.53,
        'status': 'ready'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Whitelist check
        trusted_domains = [
            'youtube.com', 'google.com', 'github.com', 'microsoft.com',
            'amazon.com', 'facebook.com', 'twitter.com', 'linkedin.com',
            'apple.com', 'netflix.com', 'stackoverflow.com', 'wikipedia.org'
        ]
        
        # Check if trusted
        is_trusted = any(domain in url.lower() for domain in trusted_domains)
        
        if is_trusted:
            # Trusted domain - predict benign
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
                    'domain_length': 15,
                    'path_length': max(0, len(url) - 20),
                    'has_ip_address': False,
                    'is_executable': False,
                    'has_sensitive_word': False,
                    'entropy_url': 3.5,
                    'entropy_domain': 3.2,
                    'number_of_dots': url.count('.')
                },
                'warnings': ['✓ Trusted domain - Known safe source'],
                'mode': 'demo_whitelist',
                'timestamp': datetime.now().isoformat()
            })
        
        # Simple heuristics for demo
        url_lower = url.lower()
        
        # Check for suspicious patterns
        if any(word in url_lower for word in ['malware', 'virus', 'hack', 'exploit']):
            prediction = 'malware'
            confidence = 0.92
            probs = {'Defacement': 0.02, 'benign': 0.03, 'malware': 0.92, 'phishing': 0.02, 'spam': 0.01}
        elif any(word in url_lower for word in ['login', 'verify', 'account', 'secure', 'update']):
            prediction = 'phishing'
            confidence = 0.85
            probs = {'Defacement': 0.03, 'benign': 0.05, 'malware': 0.02, 'phishing': 0.85, 'spam': 0.05}
        elif any(word in url_lower for word in ['free', 'win', 'prize', 'click', 'offer']):
            prediction = 'spam'
            confidence = 0.88
            probs = {'Defacement': 0.02, 'benign': 0.04, 'malware': 0.01, 'phishing': 0.05, 'spam': 0.88}
        elif len(url) > 100 or url.count('-') > 5:
            prediction = 'Defacement'
            confidence = 0.78
            probs = {'Defacement': 0.78, 'benign': 0.10, 'malware': 0.05, 'phishing': 0.05, 'spam': 0.02}
        else:
            prediction = 'benign'
            confidence = 0.82
            probs = {'Defacement': 0.05, 'benign': 0.82, 'malware': 0.03, 'phishing': 0.06, 'spam': 0.04}
        
        # Generate mock feature summary
        result = {
            'url': url,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs,
            'feature_summary': {
                'url_length': len(url),
                'domain_length': len(url.split('/')[2]) if '/' in url else 10,
                'path_length': len(url.split('/', 3)[-1]) if url.count('/') > 2 else 0,
                'has_ip_address': any(char.isdigit() for char in url.split('/')[2] if '/' in url),
                'is_executable': url.endswith(('.exe', '.dll', '.bat', '.sh')),
                'has_sensitive_word': any(word in url_lower for word in ['login', 'password', 'admin']),
                'entropy_url': round(3.0 + random.random() * 2, 3),
                'entropy_domain': round(2.5 + random.random() * 1.5, 3),
                'number_of_dots': url.count('.')
            },
            'warnings': [
                '[INFO] DEMO MODE: Using rule-based predictions',
                'Note: This is a demonstration. Real model achieves 98.53% accuracy.'
            ],
            'mode': 'demo_heuristic',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "HSEF DEMO WEB APPLICATION")
    print("="*70)
    print("\n[OK] Running in DEMO MODE (rule-based predictions)")
    print("[OK] No model training required")
    print("[OK] Real trained model: 98.53% accuracy\n")
    print("-"*70)
    print("\nAccess the web interface at:")
    print("\n   >> http://localhost:5000")
    print("   >> http://127.0.0.1:5000")
    print("\n" + "-"*70)
    print("\nTry these test URLs:")
    print("  [OK] https://www.youtube.com (trusted)")
    print("  [OK] https://www.google.com (trusted)")
    print("  [!] http://suspicious-login-verify.com (phishing)")
    print("  [!] http://free-prize-click-here.com (spam)")
    print("\n" + "="*70)
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
