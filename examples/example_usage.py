"""
HSEF - Example Usage Script

This script demonstrates various ways to use the Heterogeneous Stacking Ensemble Framework
for URL classification tasks.
"""

from hsef_model import HSEFModel
import os


def example_basic_usage():
    """Example 1: Basic usage with default settings"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Initialize model with default settings
    hsef = HSEFModel(
        output_dir='hsef_results_basic',
        use_gpu=True,
        fast_mode=False
    )
    
    # Run complete pipeline
    results = hsef.run_complete_pipeline(
        csv_path='All.csv',
        target_column='URL_Type_obf_Type'
    )
    
    print("\n✓ Basic example completed!")
    print(f"  Results saved to: {hsef.output_dir.absolute()}")


def example_fast_mode():
    """Example 2: Fast mode for quick training"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Fast Mode (Quick Training)")
    print("="*70)
    
    # Initialize with fast mode (uses LinearSVC instead of RBF SVM)
    hsef = HSEFModel(
        output_dir='hsef_results_fast',
        use_gpu=True,
        fast_mode=True  # Faster training
    )
    
    results = hsef.run_complete_pipeline(
        csv_path='All.csv',
        target_column='URL_Type_obf_Type'
    )
    
    print("\n✓ Fast mode example completed!")


def example_cpu_only():
    """Example 3: CPU-only training (no GPU)"""
    print("\n" + "="*70)
    print("EXAMPLE 3: CPU-Only Training")
    print("="*70)
    
    # Initialize without GPU
    hsef = HSEFModel(
        output_dir='hsef_results_cpu',
        use_gpu=False,
        fast_mode=False
    )
    
    results = hsef.run_complete_pipeline(
        csv_path='All.csv',
        target_column='URL_Type_obf_Type'
    )
    
    print("\n✓ CPU-only example completed!")


def example_custom_configuration():
    """Example 4: Custom model configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Configuration")
    print("="*70)
    
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    from sklearn.svm import SVC
    
    # Initialize model
    hsef = HSEFModel(
        output_dir='hsef_results_custom',
        use_gpu=True,
        fast_mode=False
    )
    
    # Load data
    hsef.load_data('All.csv', target_column='URL_Type_obf_Type')
    
    # Customize base learners
    print("\nCustomizing base learners...")
    
    # Custom Random Forest
    hsef.rf_model = RandomForestClassifier(
        n_estimators=300,      # More trees
        max_depth=40,          # Deeper trees
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    
    # Custom XGBoost
    hsef.xgb_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=10,
        learning_rate=0.05,    # Slower learning
        random_state=42,
        tree_method='gpu_hist' if hsef.gpu_available else 'hist'
    )
    
    # Custom SVM
    hsef.svm_model = SVC(
        C=15.0,                # Stronger regularization
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    # Train with custom configuration
    hsef.train_base_learners_with_cv(n_folds=5)
    hsef.build_stacking_ensemble()
    hsef.train_stacking_ensemble()
    
    # Evaluate
    results = hsef.evaluate_models()
    
    # Generate artifacts
    hsef.plot_confusion_matrices(results)
    hsef.plot_roc_curves(results)
    hsef.plot_feature_importance()
    hsef.plot_model_comparison(results)
    hsef.generate_classification_reports(results)
    hsef.generate_architecture_diagram()
    hsef.save_training_log()
    
    print("\n✓ Custom configuration example completed!")


def example_step_by_step():
    """Example 5: Step-by-step execution with inspection"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Step-by-Step Execution")
    print("="*70)
    
    # Initialize
    hsef = HSEFModel(
        output_dir='hsef_results_stepwise',
        use_gpu=True,
        fast_mode=False
    )
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    hsef.load_data('All.csv', target_column='URL_Type_obf_Type')
    print(f"  Features: {len(hsef.feature_names)}")
    print(f"  Classes: {hsef.class_names}")
    print(f"  Training samples: {hsef.X_train.shape[0]}")
    print(f"  Test samples: {hsef.X_test.shape[0]}")
    
    # Step 2: Build base learners
    print("\nStep 2: Building base learners...")
    hsef.build_base_learners()
    
    # Step 3: Train with cross-validation
    print("\nStep 3: Training base learners with 5-fold CV...")
    hsef.train_base_learners_with_cv(n_folds=5)
    
    # Inspect individual model performance
    print("\nInspecting individual base learner predictions:")
    for name, model in [('RF', hsef.rf_model), ('XGB', hsef.xgb_model), ('SVM', hsef.svm_model)]:
        y_pred = model.predict(hsef.X_test)
        accuracy = (y_pred == hsef.y_test).mean()
        print(f"  {name} Test Accuracy: {accuracy:.4f}")
    
    # Step 4: Build ensemble
    print("\nStep 4: Building stacking ensemble...")
    hsef.build_stacking_ensemble()
    
    # Step 5: Train ensemble
    print("\nStep 5: Training ensemble...")
    hsef.train_stacking_ensemble()
    
    # Step 6: Evaluate
    print("\nStep 6: Evaluating all models...")
    results = hsef.evaluate_models()
    
    # Step 7: Generate artifacts
    print("\nStep 7: Generating visualizations...")
    hsef.plot_confusion_matrices(results)
    hsef.plot_roc_curves(results)
    hsef.plot_feature_importance()
    hsef.plot_model_comparison(results)
    hsef.generate_classification_reports(results)
    hsef.generate_architecture_diagram()
    hsef.generate_shap_explanations(n_samples=100)
    hsef.save_training_log()
    
    print("\n✓ Step-by-step example completed!")


def example_predict_new_urls():
    """Example 6: Training model and making predictions on new data"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Training and Prediction")
    print("="*70)
    
    import numpy as np
    
    # Train model
    hsef = HSEFModel(
        output_dir='hsef_results_prediction',
        use_gpu=True,
        fast_mode=False
    )
    
    results = hsef.run_complete_pipeline(
        csv_path='All.csv',
        target_column='URL_Type_obf_Type'
    )
    
    # Make predictions on test set (first 10 samples)
    print("\nMaking predictions on sample URLs:")
    print("-" * 70)
    
    n_samples = 10
    X_sample = hsef.X_test[:n_samples]
    y_true = hsef.y_test[:n_samples]
    
    # Get predictions from stacking model
    y_pred = hsef.stacking_model.predict(X_sample)
    y_pred_proba = hsef.stacking_model.predict_proba(X_sample)
    
    # Display results
    for i in range(n_samples):
        true_class = hsef.class_names[y_true[i]]
        pred_class = hsef.class_names[y_pred[i]]
        confidence = y_pred_proba[i].max()
        
        match = "✓" if y_true[i] == y_pred[i] else "✗"
        
        print(f"\nSample {i+1}:")
        print(f"  True:       {true_class}")
        print(f"  Predicted:  {pred_class}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Status:     {match}")
        
        # Show probability distribution
        print(f"  Probabilities:")
        for j, class_name in enumerate(hsef.class_names):
            print(f"    {class_name:15s}: {y_pred_proba[i][j]:.4f}")
    
    print("\n✓ Prediction example completed!")


def main():
    """Main function to run examples"""
    
    print("\n" + "="*70)
    print("HSEF - EXAMPLE USAGE DEMONSTRATIONS")
    print("="*70)
    
    print("\nAvailable examples:")
    print("  1. Basic usage with default settings")
    print("  2. Fast mode for quick training")
    print("  3. CPU-only training")
    print("  4. Custom model configuration")
    print("  5. Step-by-step execution with inspection")
    print("  6. Training and making predictions")
    
    # Check if data file exists
    if not os.path.exists('All.csv'):
        print("\n❌ Error: 'All.csv' not found in current directory!")
        print("   Please ensure the dataset is available before running examples.")
        return
    
    # Run Example 1 by default
    print("\n" + "="*70)
    print("Running Example 1: Basic Usage")
    print("="*70)
    
    try:
        example_basic_usage()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        print("\nTo run other examples, call them individually:")
        print("  - example_fast_mode()")
        print("  - example_cpu_only()")
        print("  - example_custom_configuration()")
        print("  - example_step_by_step()")
        print("  - example_predict_new_urls()")
        
    except Exception as e:
        print(f"\n❌ Error running example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
