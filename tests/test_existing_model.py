"""
Load and test the existing trained HSEF model
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "="*80)
print(" "*25 + "LOADING EXISTING MODEL")
print("="*80)

# Load the model
model_path = Path('models/hsef_model.pkl')
scaler_path = Path('models/hsef_scaler.pkl')

print("\nLoading model files...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"✓ Loaded model from {model_path}")

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print(f"✓ Loaded scaler from {scaler_path}")

# Load data to test
print("\nLoading test data...")
df = pd.read_csv('All.csv')
X = df.drop('URL_Type_obf_Type', axis=1)
y = df['URL_Type_obf_Type']

# Handle missing/inf
X = X.fillna(0).replace([np.inf, -np.inf], 0)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split (same random state as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✓ Loaded {len(df)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Predict
print("\nTesting model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Test on benign URLs
print("\n" + "="*80)
print("TESTING ON BENIGN URLs")
print("="*80)

try:
    test_urls_df = pd.read_csv('test_urls.csv')
    print(f"\nFound {len(test_urls_df)} benign URLs in test_urls.csv")
    
    from url_feature_extractor import URLFeatureExtractor
    extractor = URLFeatureExtractor()
    
    benign_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.google.com",
        "https://www.github.com",
        "https://www.microsoft.com",
        "https://www.amazon.com"
    ]
    
    print("\nTesting 5 known benign URLs:")
    for url in benign_urls:
        features = extractor.extract_features(url)
        features_df = pd.DataFrame([features])
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        features_scaled = scaler.transform(features_df)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        predicted_class = le.classes_[prediction]
        confidence = probabilities[prediction] * 100
        
        print(f"\n  URL: {url}")
        print(f"  Predicted: {predicted_class} ({confidence:.2f}% confidence)")
        
except Exception as e:
    print(f"\n⚠ Could not test on benign URLs: {e}")

print("\n" + "="*80)
print("✓ MODEL LOADED AND TESTED SUCCESSFULLY")
print("="*80)
print(f"\nModel location: {model_path.absolute()}")
print(f"Scaler location: {scaler_path.absolute()}")
print(f"\nThis model is ready to use with {accuracy*100:.2f}% accuracy!")
