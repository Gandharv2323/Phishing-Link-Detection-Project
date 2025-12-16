"""
Unified HSEF Calibration Pipeline
==================================

Single entry point for complete calibration and publication preparation.

Usage:
    python run_full_calibration.py

This will:
1. Detect false positives on benign URLs
2. Analyze with SHAP
3. Calibrate meta-layer
4. Retrain model
5. Generate publication artifacts
"""

import sys
from pathlib import Path
from datetime import datetime

# Import phases
from hsef_calibration_system import HSEFCalibrationSystem
from hsef_calibration_phase2 import HSEFCalibratorPhase2


def run_complete_pipeline():
    """Execute complete calibration pipeline"""
    
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print(" " * 25 + "HSEF CALIBRATION PIPELINE")
    print(" " * 20 + "Publication-Ready Model Preparation")
    print("="*80)
    print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis pipeline will:")
    print("  ✓ Load and analyze training data")
    print("  ✓ Train baseline HSEF model")
    print("  ✓ Detect false positives on known benign URLs")
    print("  ✓ Perform SHAP interpretability analysis")
    print("  ✓ Calibrate meta-layer with regularization")
    print("  ✓ Retrain with optimized configuration")
    print("  ✓ Generate publication-ready artifacts")
    print("\n" + "="*80)
    
    # Confirm execution
    response = input("\nProceed with full calibration? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Calibration cancelled.")
        return
    
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
        
        # SHAP analysis (optional, can be slow)
        print("\n" + "="*70)
        print("SHAP analysis can be very slow (~1-2 min per URL).")
        print("You can skip it and still get all other artifacts.")
        skip_shap = input("Skip SHAP analysis? (yes/no, default=no): ").strip().lower()
        
        if skip_shap not in ['yes', 'y']:
            calibrator_p1.analyze_false_positives_with_shap(n_background=50, max_analyze=5)
        else:
            print("⚠ Skipping SHAP analysis")
        
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
        print(f"     - shap_summary_calibrated.png")
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
        
        print(f"\n  📝 Logs:")
        print(f"     - publication_outputs/training_log_corrected.json")
        print(f"     - publication_outputs/config_corrections.yaml")
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Review generated plots in publication_outputs/plots/")
        print("2. Read performance_report_calibrated.txt for metrics")
        print("3. Examine false_positives.csv for remaining issues")
        print("4. Update web app to use calibrated model")
        print("5. Include artifacts in research paper")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_phase1_only():
    """Run only Phase 1 (detection and analysis)"""
    print("\n" + "="*80)
    print("PHASE 1 ONLY: False Positive Detection & SHAP Analysis")
    print("="*80)
    
    calibrator = HSEFCalibrationSystem(
        data_path='All.csv',
        output_dir='publication_outputs'
    )
    
    calibrator.load_data()
    calibrator.train_baseline_model()
    calibrator.detect_false_positives()
    calibrator.analyze_false_positives_with_shap()
    calibrator.save_config()
    calibrator.save_training_log()
    
    print("\n✓ Phase 1 complete. Run Phase 2 separately if needed.")


def run_phase2_only():
    """Run only Phase 2 (calibration and artifacts)"""
    print("\n" + "="*80)
    print("PHASE 2 ONLY: Calibration & Publication Artifacts")
    print("="*80)
    
    # Check if Phase 1 was completed
    phase1_log = Path('publication_outputs/training_log_corrected.json')
    if not phase1_log.exists():
        print("❌ Error: Phase 1 must be completed first!")
        print("   Run: python run_full_calibration.py --phase1")
        return
    
    calibrator = HSEFCalibratorPhase2(output_dir='publication_outputs')
    
    calibrator.load_data_and_baseline()
    calibrator.calibrate_meta_layer()
    calibrator.retrain_with_augmented_data()
    calibrator.generate_publication_artifacts()
    calibrator.generate_performance_report()
    calibrator.save_calibrated_model()
    calibrator.finalize()
    
    print("\n✓ Phase 2 complete. Check publication_outputs/ for artifacts.")


def main():
    """Main entry point with command-line argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HSEF Calibration Pipeline for Publication-Ready Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_calibration.py              # Run complete pipeline
  python run_full_calibration.py --phase1     # Run Phase 1 only
  python run_full_calibration.py --phase2     # Run Phase 2 only
        """
    )
    
    parser.add_argument(
        '--phase1',
        action='store_true',
        help='Run Phase 1 only (detection and analysis)'
    )
    
    parser.add_argument(
        '--phase2',
        action='store_true',
        help='Run Phase 2 only (calibration and artifacts)'
    )
    
    args = parser.parse_args()
    
    if args.phase1:
        run_phase1_only()
    elif args.phase2:
        run_phase2_only()
    else:
        run_complete_pipeline()


if __name__ == '__main__':
    main()
