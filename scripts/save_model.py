"""
Save trained HSEF model for web deployment
"""

import joblib
import json
from pathlib import Path
from hsef_model import HSEFModel

def save_model_for_deployment(hsef_model, output_dir='models'):
    """
    Save trained model and necessary components for deployment
    
    Parameters:
    -----------
    hsef_model : HSEFModel
        Trained HSEF model instance
    output_dir : str
        Directory to save model files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING MODEL FOR DEPLOYMENT")
    print("="*70 + "\n")
    
    try:
        # 1. Save the stacking model
        model_path = output_path / 'hsef_model.pkl'
        joblib.dump(hsef_model.stacking_model, model_path)
        print(f"✓ Saved stacking model: {model_path}")
        
        # 2. Save the scaler
        scaler_path = output_path / 'hsef_scaler.pkl'
        joblib.dump(hsef_model.scaler, scaler_path)
        print(f"✓ Saved scaler: {scaler_path}")
        
        # 3. Save feature names and class names
        features_path = output_path / 'feature_names.json'
        features_data = {
            'features': hsef_model.feature_names,
            'classes': hsef_model.class_names.tolist()
        }
        with open(features_path, 'w') as f:
            json.dump(features_data, f, indent=2)
        print(f"✓ Saved feature metadata: {features_path}")
        
        # 4. Save model information
        info_path = output_path / 'model_info.json'
        model_info = {
            'model_type': 'HSEF - Heterogeneous Stacking Ensemble Framework',
            'base_learners': ['Random Forest', 'XGBoost', 'SVM'],
            'meta_learner': 'Logistic Regression',
            'n_features': len(hsef_model.feature_names),
            'n_classes': len(hsef_model.class_names),
            'classes': hsef_model.class_names.tolist(),
            'gpu_enabled': hsef_model.gpu_available,
            'fast_mode': hsef_model.fast_mode,
            'training_metrics': hsef_model.history.get('test_results', {}),
            'timestamp': hsef_model.history.get('timestamp', '')
        }
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✓ Saved model info: {info_path}")
        
        # 5. Save individual base models (optional, for analysis)
        base_models_path = output_path / 'base_models'
        base_models_path.mkdir(exist_ok=True)
        
        joblib.dump(hsef_model.rf_model, base_models_path / 'random_forest.pkl')
        joblib.dump(hsef_model.xgb_model, base_models_path / 'xgboost.pkl')
        joblib.dump(hsef_model.svm_model, base_models_path / 'svm.pkl')
        print(f"✓ Saved base models: {base_models_path}")
        
        print("\n" + "="*70)
        print("MODEL SAVED SUCCESSFULLY FOR DEPLOYMENT")
        print("="*70)
        print(f"\nModel files saved to: {output_path.absolute()}")
        print("\nYou can now run the web application with:")
        print("  python app.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error saving model: {str(e)}")
        return False


if __name__ == "__main__":
    print("This script should be called after training the HSEF model.")
    print("\nExample usage:")
    print("""
from hsef_model import HSEFModel
from save_model import save_model_for_deployment

# Train the model
hsef = HSEFModel()
hsef.run_complete_pipeline('All.csv')

# Save for deployment
save_model_for_deployment(hsef)
""")
