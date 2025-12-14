"""
Gradual Git Commit Script
Commits 5-6 files per day until all files are committed.
Automatically stops when complete.
"""

import os
import subprocess
import json
from datetime import datetime

# Configuration
FILES_PER_DAY = 5
PROJECT_DIR = r"c:\Users\shind\Downloads\To Move Files\ASEP\ASEP 2 Pishing_link"
STATE_FILE = os.path.join(PROJECT_DIR, ".commit_state.json")
REMOTE_URL = "https://github.com/Gandharv2323/Phishing-Link-Detection-Project.git"

# Files to commit (in order)
ALL_FILES = [
    # Day 1: Core files
    "README.md",
    "requirements.txt",
    ".gitignore",
    "STREAK_LOG.md",
    "START_SERVER.bat",
    "START_WEB_APP.bat",
    
    # Day 2: Source code
    "src/hsef_model.py",
    "src/hsef_helpers.py",
    "src/hsef_debugger.py",
    "src/url_feature_extractor.py",
    "src/hsef_calibration_system.py",
    
    # Day 3: More source + automation
    "DAILY_COMMIT.bat",
    "gradual_commit.py",
    "src/hsef_calibration_phase2.py",
    "app/app.py",
    "app/simple_app.py",
    "app/demo_app.py",
    "app/mock_app.py",
    
    # Day 4: Templates & scripts
    "app/templates/index.html",
    "scripts/start_server.py",
    "scripts/start_enhanced_server.py",
    "scripts/prepare_deployment.py",
    "scripts/quick_setup.py",
    
    # Day 5: More scripts
    "scripts/save_model.py",
    "scripts/load_model.py",
    "scripts/calibration/run_calibration_auto.py",
    "scripts/calibration/run_calibration_fast.py",
    "scripts/calibration/run_calibration_minimal.py",
    
    # Day 6: Calibration & tests
    "scripts/calibration/run_full_calibration.py",
    "scripts/calibration/calibrate_simple.py",
    "tests/test_api.py",
    "tests/test_calibration_system.py",
    "tests/test_enhanced_app.py",
    
    # Day 7: More tests
    "tests/test_existing_model.py",
    "tests/test_feature_extraction.py",
    "tests/test_gpu.py",
    "tests/quick_test.py",
    "tests/test_data/test_batch.csv",
    
    # Day 8: Test data & examples
    "tests/test_data/debug_test_batch.csv",
    "tests/test_data/test_urls.csv",
    "tests/test_data/sample_features.json",
    "examples/example_usage.py",
    "examples/example_debugger.py",
    
    # Day 9: Examples & web
    "examples/analyze_youtube_prediction.py",
    "examples/check_values.py",
    "examples/create_test_batch.py",
    "web/test_page.html",
    "web/test_working.html",
    
    # Day 10: Documentation
    "docs/QUICKSTART.md",
    "docs/QUICK_REFERENCE.md",
    "docs/DEPLOYMENT_GUIDE.md",
    "docs/IMPLEMENTATION_GUIDE.md",
    "docs/FEATURE_EXTRACTION_GUIDE.md",
    
    # Day 11: More docs
    "docs/CALIBRATION_README.md",
    "docs/DEBUGGER_GUIDE.md",
    "docs/WEB_APP_README.md",
    "docs/WEB_APP_USER_GUIDE.md",
    "docs/MODEL_SUMMARY.md",
    
    # Day 12: Technical docs
    "docs/technical/PIPELINE_FLOW.txt",
    "docs/technical/FEATURE_MISMATCH_ISSUE.md",
    "docs/technical/README_FEATURE_EXTRACTION.md",
    "docs/technical/IMPLEMENTATION_CHECKLIST.md",
    "docs/technical/SYSTEM_COMPLETE.md",
    
    # Day 13: Final docs & models
    "docs/technical/DELIVERY_COMPLETE.md",
    "docs/technical/DELIVERY_SUMMARY.md",
    "docs/technical/UPDATE_SUMMARY.md",
    "docs/technical/FINAL_STATUS_REPORT.md",
    "docs/technical/WEBAPP_READY.md",
    
    # Day 14: Model files (metadata only, actual models in .gitignore)
    "models/feature_names.json",
    "models/model_info.json",
]

def load_state():
    """Load commit state from file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"committed_count": 0, "completed": False, "day": 0}

def save_state(state):
    """Save commit state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def run_git(args):
    """Run a git command."""
    result = subprocess.run(
        ["git"] + args,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout + result.stderr

def init_repo():
    """Initialize git repo if not exists."""
    git_dir = os.path.join(PROJECT_DIR, ".git")
    if not os.path.exists(git_dir):
        print("Initializing git repository...")
        run_git(["init"])
        run_git(["remote", "add", "origin", REMOTE_URL])
        run_git(["branch", "-M", "main"])
        return True
    return False

def main():
    os.chdir(PROJECT_DIR)
    
    # Load state
    state = load_state()
    
    if state["completed"]:
        print("=" * 50)
        print("ALL FILES ALREADY COMMITTED!")
        print("Total commits made over", state["day"], "days")
        print("=" * 50)
        
        # Disable the scheduled task
        subprocess.run(
            ["schtasks", "/change", "/tn", "GitHub_Daily_Streak", "/disable"],
            capture_output=True
        )
        print("Scheduled task disabled.")
        return
    
    # Initialize repo if needed
    init_repo()
    
    # Calculate which files to commit today
    start_idx = state["committed_count"]
    end_idx = min(start_idx + FILES_PER_DAY, len(ALL_FILES))
    
    if start_idx >= len(ALL_FILES):
        state["completed"] = True
        save_state(state)
        print("All files committed!")
        return
    
    files_today = ALL_FILES[start_idx:end_idx]
    state["day"] += 1
    today = datetime.now().strftime("%Y-%m-%d")
    
    print("=" * 50)
    print(f"DAY {state['day']} - {today}")
    print(f"Committing files {start_idx + 1} to {end_idx} of {len(ALL_FILES)}")
    print("=" * 50)
    
    # Update streak log
    with open(os.path.join(PROJECT_DIR, "STREAK_LOG.md"), "a") as f:
        f.write(f"### Day {state['day']} - {today}\n")
        f.write(f"Files committed: {', '.join([os.path.basename(f) for f in files_today])}\n\n")
    
    # Add files
    for file in files_today:
        filepath = os.path.join(PROJECT_DIR, file)
        if os.path.exists(filepath):
            success, output = run_git(["add", file])
            print(f"  + {file}")
        else:
            print(f"  ! {file} (not found, skipping)")
    
    # Also add the streak log
    run_git(["add", "STREAK_LOG.md"])
    run_git(["add", ".commit_state.json"])
    
    # Commit
    commit_msg = f"Day {state['day']}: Add {', '.join([os.path.basename(f) for f in files_today[:3]])}..."
    success, output = run_git(["commit", "-m", commit_msg])
    
    if success:
        print(f"\nCommit successful!")
        
        # Push
        success, output = run_git(["push", "-u", "origin", "main"])
        if success:
            print("Pushed to GitHub!")
        else:
            print(f"Push output: {output}")
            # Try force push if first time
            run_git(["push", "-u", "origin", "main", "--force"])
    else:
        print(f"Commit output: {output}")
    
    # Update state
    state["committed_count"] = end_idx
    if end_idx >= len(ALL_FILES):
        state["completed"] = True
        print("\n" + "=" * 50)
        print("ALL FILES COMMITTED! Project upload complete!")
        print(f"Total: {len(ALL_FILES)} files over {state['day']} days")
        print("=" * 50)
        
        # Disable scheduled task
        subprocess.run(
            ["schtasks", "/change", "/tn", "GitHub_Daily_Streak", "/disable"],
            capture_output=True
        )
    
    save_state(state)
    
    remaining = len(ALL_FILES) - end_idx
    if remaining > 0:
        days_left = (remaining + FILES_PER_DAY - 1) // FILES_PER_DAY
        print(f"\nRemaining: {remaining} files ({days_left} more days)")

if __name__ == "__main__":
    main()
