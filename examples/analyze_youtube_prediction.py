"""
Analyze why YouTube is classified as Defacement instead of benign
"""

import pandas as pd
import numpy as np
from url_feature_extractor import URLFeatureExtractor
import joblib
import json

# Load training data
print("Loading training data...")
df = pd.read_csv('All.csv')

# Load model and scaler
print("Loading model...")
model = joblib.load('models/hsef_model.pkl')
scaler = joblib.load('models/hsef_scaler.pkl')
with open('models/feature_names.json') as f:
    data = json.load(f)
    feature_names = data['features']
    class_names = data['classes']

# Extract YouTube features
print("\n" + "="*70)
print("ANALYZING: https://www.youtube.com/")
print("="*70)

extractor = URLFeatureExtractor()
youtube_features = extractor.extract_features("https://www.youtube.com/")
youtube_vector = extractor.get_feature_vector(feature_names)

print("\nExtracted Features:")
print(f"  urlLen: {youtube_features['urlLen']}")
print(f"  domainlength: {youtube_features['domainlength']}")
print(f"  pathLength: {youtube_features['pathLength']}")
print(f"  Entropy_URL: {youtube_features['Entropy_URL']:.4f}")
print(f"  domain_token_count: {youtube_features['domain_token_count']}")
print(f"  path_token_count: {youtube_features['path_token_count']}")

# Compare with training data
print("\n" + "="*70)
print("COMPARISON WITH TRAINING DATA")
print("="*70)

for class_name in class_names:
    class_data = df[df['URL_Type_obf_Type'] == class_name]
    print(f"\n{class_name.upper()} (n={len(class_data)}):")
    print(f"  urlLen: {class_data['urlLen'].mean():.1f} ± {class_data['urlLen'].std():.1f}")
    print(f"  domainlength: {class_data['domainlength'].mean():.1f} ± {class_data['domainlength'].std():.1f}")
    print(f"  pathLength: {class_data['pathLength'].mean():.1f} ± {class_data['pathLength'].std():.1f}")
    print(f"  Entropy_URL: {class_data['Entropy_URL'].mean():.3f} ± {class_data['Entropy_URL'].std():.3f}")

# Predict with model
print("\n" + "="*70)
print("MODEL PREDICTION")
print("="*70)

X = pd.DataFrame([youtube_vector], columns=feature_names)
X = X.fillna(0).replace([np.inf, -np.inf], 0)
X_scaled = scaler.transform(X)

prediction = model.predict(X_scaled)[0]
probabilities = model.predict_proba(X_scaled)[0]

print(f"\nPrediction: {class_names[prediction]}")
print(f"Confidence: {probabilities.max()*100:.2f}%")
print("\nAll probabilities:")
for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
    print(f"  {cls}: {prob*100:.2f}%")

# Find similar URLs in training data
print("\n" + "="*70)
print("FINDING SIMILAR URLs IN TRAINING DATA")
print("="*70)

# Find URLs with similar characteristics
similar = df[
    (df['urlLen'].between(20, 30)) &
    (df['domainlength'].between(10, 20)) &
    (df['pathLength'] <= 5)
]

print(f"\nFound {len(similar)} URLs with similar characteristics:")
print(similar['URL_Type_obf_Type'].value_counts())

# Check if there are benign URLs with these exact features
exact_match = df[
    (df['urlLen'] == 24) &
    (df['domainlength'] == 15) &
    (df['pathLength'] == 1) &
    (df['NumberofDotsinURL'] == 2)
]

if len(exact_match) > 0:
    print(f"\nFound {len(exact_match)} URLs with EXACT same basic features:")
    print(exact_match['URL_Type_obf_Type'].value_counts())
else:
    print("\nNo URLs in training data with these exact features")

# Recommendation
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("""
The model is classifying YouTube as 'Defacement' because:

1. The training data may not have many short, simple benign URLs
2. The extracted features might differ slightly from training data format
3. The model was trained on a specific dataset that may not generalize well

SOLUTIONS:
1. Retrain the model with more diverse benign URLs (including popular sites)
2. Adjust feature extraction to match training data format exactly
3. Add post-processing to whitelist known benign domains
4. Use a confidence threshold (e.g., only trust predictions >80%)
""")
