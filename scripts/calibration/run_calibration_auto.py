"""
Automatic HSEF Calibration Pipeline - No Prompts
=================================================

Runs the full calibration without requiring user input.
SHAP analysis is limited to 5 samples for speed.
"""

from hsef_calibration_system import HSEFCalibrationSystem
from hsef_calibration_phase2 import HSEFCalibratorPhase2
from datetime import datetime
import sys


def run_auto_calibration():
    """Execute complete calibration pipeline automatically"""
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print(" " * 25 + "HSEF AUTOMATIC CALIBRATION")
    print(" " * 20 + "Publication-Ready Model Preparation")
    print("="*80)
    print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRunning automatic calibration (no prompts)...")
    print("SHAP analysis limited to 5 samples for speed.")
    print("\n" + "="*80)
    
    try:
        # ==========================================
        # PHASE 1: Detection and Analysis
        # ==========================================
        print("\n\n")
        print("█" * 80)
        print("█" + " " * 33 + "PHASE 1" + " " * 39 + "█")
        print("█" + " " * 25 + "Detection & Analysis" + " " * 33 + "█")
        print("█" * 80)
        
        calibrator_p1 = HSEFCalibrationSystem(
            data_path='All.csv',
            output_dir='publication_outputs'
        )
        
        calibrator_p1.load_data()
        calibrator_p1.train_baseline_model()
        calibrator_p1.detect_false_positives()
        
        # SHAP analysis (limited to 5 for speed)
        print("\n⚠ SHAP analysis limited to 5 false positives (takes ~5-10 min)")
        calibrator_p1.analyze_false_positives_with_shap(n_background=50, max_analyze=5)
        
        # Save Phase 1 artifacts
        calibrator_p1.save_config()
        calibrator_p1.save_training_log()
        
        print("\n" + "="*80)
        print("✓ PHASE 1 COMPLETE")
        print("="*80)
        
        # ==========================================
        # PHASE 2: Calibration and Artifacts
        # ==========================================
        print("\n\n")
        print("█" * 80)
        print("█" + " " * 33 + "PHASE 2" + " " * 39 + "█")
        print("█" + " " * 20 + "Calibration & Artifact Generation" + " " * 24 + "█")
        print("█" * 80)
        
        calibrator_p2 = HSEFCalibratorPhase2(output_dir='publication_outputs')
        
        calibrator_p2.load_data_and_baseline()
        calibrator_p2.calibrate_meta_layer()
        calibrator_p2.retrain_with_augmented_data()
        calibrator_p2.generate_publication_artifacts()
        calibrator_p2.generate_performance_report()
        calibrator_p2.save_calibrated_model()
        calibrator_p2.finalize()
        
        print("\n" + "="*80)
        print("✓ PHASE 2 COMPLETE")
        print("="*80)
        
        # ==========================================
        # SUMMARY
        # ==========================================
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n\n")
        print("█" * 80)
        print("█" + " " * 25 + "CALIBRATION COMPLETE" + " " * 33 + "█")
        print("█" * 80)
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  Duration: {duration}")
        print(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📁 Output Directory: publication_outputs/")
        print(f"\n  📊 Plots (publication_outputs/plots/):")
        print(f"     - confusion_matrix_calibrated.png")
        print(f"     - roc_curves_calibrated.png")
        print(f"     - feature_importance_top20.png")
        print(f"     - shap_summary_calibrated.png (if generated)")
        print(f"     - meta_weights_calibrated.png")
        
        print(f"\n  🤖 Models (publication_outputs/models/):")
        print(f"     - stacking_calibrated.joblib")
        print(f"     - scaler_calibrated.joblib")
        print(f"     - label_encoder_calibrated.joblib")
        print(f"     - config_calibrated.yaml")
        
        print(f"\n  📄 Reports (publication_outputs/reports/):")
        print(f"     - false_positives.csv")
        print(f"     - shap_analysis_false_positives.json")
        print(f"     - performance_report_calibrated.txt")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Review generated plots in publication_outputs/plots/")
        print("2. Read performance_report_calibrated.txt for metrics")
        print("3. Examine false_positives.csv for remaining issues")
        print("4. Update web app to use calibrated model")
        print("5. Include artifacts in research paper")
        
        print("\n" + "="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = run_auto_calibration()
    sys.exit(exit_code)
