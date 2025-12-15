"""
Prepare trained HSEF model for web deployment
This script loads the trained model and saves it in the correct format for the web app
"""

from hsef_model import HSEFModel
import pandas as pd

print("\n" + "="*70)
print("PREPARING MODEL FOR WEB DEPLOYMENT")
print("="*70 + "\n")

print("Step 1: Loading and training HSEF model...")
print("-" * 70)

# Initialize and train the model
hsef = HSEFModel(fast_mode=False, use_gpu=True)

# Load the dataset
print("\nLoading dataset: All.csv")
df = pd.read_csv('All.csv')
print(f"Dataset loaded: {len(df)} samples")

# Run the complete pipeline
print("\nTraining HSEF model with GPU acceleration...")
print("This will:")
print("  1. Train all base learners (RF, XGBoost, SVM)")
print("  2. Train the stacking ensemble")
print("  3. Evaluate on test set")
print("  4. Automatically save models for deployment")
print()

hsef.run_complete_pipeline('All.csv')

print("\n" + "="*70)
print("DEPLOYMENT PREPARATION COMPLETE!")
print("="*70)
print("\nModels saved to: models/")
print("\nNext step: Start the web application")
print("  python app.py")
print("\nThen open: http://127.0.0.1:5000")
print("="*70 + "\n")
