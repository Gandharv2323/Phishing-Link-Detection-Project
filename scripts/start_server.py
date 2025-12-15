"""
Simple Flask App Starter - No Debug Mode
"""
from app import app, load_model

# Load model once
success, message = load_model()
if success:
    print(f"✓ {message}")
else:
    print(f"⚠ {message}")

print("\n" + "="*70)
print("HSEF WEB APPLICATION")
print("="*70)
print("\n🌐 Open your browser and go to: http://127.0.0.1:5000")
print("\n⏹  Press CTRL+C to stop\n")

# Run without debugger (prevents reloading issues)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
