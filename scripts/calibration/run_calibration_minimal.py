"""
MINIMAL Calibration - Just calibrate and save model (no plots)
Completes in ~8 minutes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from hsef_calibration_phase2 import HSEFCalibratorPhase2
import warnings
warnings.filterwarnings('ignore')

def run_minimal_calibration():
    """Calibrate and save model only - skip all plots"""
    
    print("\n" + "="*80)
    print(" "*15 + "MINIMAL HSEF CALIBRATION (MODEL ONLY)")
    print("="*80)
    
    from datetime import datetime
    start_time = datetime.now()
    print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nCalibrating model and saving - skipping all plots for speed")
    print("="*80 + "\n")
    
    try:
        # Initialize
        calibrator = HSEFCalibratorPhase2()
        
        # Load data
        print("Loading data...")
        calibrator.load_data_and_baseline()
        
        # Calibrate meta-layer
        print("\nCalibrating meta-layer...")
        calibrator.calibrate_meta_layer()
        
        # Save model ONLY
        print("\nSaving calibrated model...")
        calibrator.save_calibrated_model()
        
        # Generate minimal report
        print("\nGenerating performance report...")
        calibrator.generate_performance_report()
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print("✓ CALIBRATION COMPLETE")
        print("="*80)
        print(f"\nTotal time: {duration:.1f} minutes")
        print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📁 Output Location:")
        print(f"   {calibrator.output_dir.absolute()}")
        
        print("\n📊 Generated Files:")
        print("   Models:")
        print("     ✓ hsef_calibrated_model.pkl")
        print("     ✓ hsef_baseline_model.pkl")
        print("   Reports:")
        print("     ✓ performance_report_calibrated.txt")
        
        print("\n✅ Your calibrated model is ready to deploy!")
        print(f"   Accuracy: {calibrator.test_acc_calibrated*100:.2f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Calibration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = run_minimal_calibration()
    sys.exit(exit_code)
