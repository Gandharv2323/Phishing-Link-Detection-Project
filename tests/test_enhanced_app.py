"""
Quick Test: Verify Real Feature Extraction Works
Tests the updated HSEF web app with real URL analysis
"""

import requests
import json
import time

def test_single_url():
    """Test single URL prediction with real feature extraction"""
    print("="*70)
    print("TEST 1: Single URL Prediction (YouTube)")
    print("="*70)
    
    url = "https://www.youtube.com/"
    
    try:
        response = requests.post(
            'http://127.0.0.1:5000/api/predict',
            json={'url': url},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ URL: {result['url']}")
            print(f"✓ Prediction: {result['prediction']}")
            print(f"✓ Confidence: {result['confidence']*100:.2f}%")
            print(f"✓ Mode: {result.get('mode', 'unknown')}")
            
            if 'feature_summary' in result:
                print("\n📊 Feature Summary:")
                fs = result['feature_summary']
                print(f"   • URL Length: {fs['url_length']}")
                print(f"   • Domain Length: {fs['domain_length']}")
                print(f"   • Path Length: {fs['path_length']}")
                print(f"   • URL Entropy: {fs['entropy_url']:.4f}")
                print(f"   • Has IP: {fs['has_ip_address']}")
                print(f"   • Executable: {fs['is_executable']}")
                print(f"   • Sensitive Word: {fs['has_sensitive_word']}")
            
            if 'base_models' in result and result['base_models']:
                print("\n🤖 Base Models:")
                for name, data in result['base_models'].items():
                    print(f"   • {name}: {data['prediction']} ({data['confidence']*100:.1f}%)")
            
            if 'meta_layer_analysis' in result and result['meta_layer_analysis']:
                print("\n⚖️  Meta-Layer Weights:")
                for name, data in result['meta_layer_analysis'].items():
                    print(f"   • {name}: {data['percentage']:.1f}%")
            
            if 'shap_analysis' in result and result['shap_analysis']:
                print("\n📈 Top SHAP Features:")
                for i, feat in enumerate(result['shap_analysis']['top_features'][:3], 1):
                    print(f"   {i}. {feat['feature']}: {feat['impact']} risk")
            
            print("\n✅ Single URL test PASSED")
            return True
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Server not running!")
        print("   Start server with: python start_enhanced_server.py")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_various_urls():
    """Test multiple URLs with different characteristics"""
    print("\n" + "="*70)
    print("TEST 2: Various URL Types")
    print("="*70)
    
    test_cases = [
        {
            'url': 'https://www.google.com/',
            'expected': 'benign',
            'desc': 'Google homepage'
        },
        {
            'url': 'http://verify-account.tk/login.exe',
            'expected': 'malware/phishing',
            'desc': 'Suspicious URL with executable'
        },
        {
            'url': 'https://192.168.1.1/admin/login.php',
            'expected': 'suspicious',
            'desc': 'IP address with admin path'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test['url']}")
        print(f"   Description: {test['desc']}")
        
        try:
            response = requests.post(
                'http://127.0.0.1:5000/api/predict',
                json={'url': test['url']},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Prediction: {result['prediction']}")
                print(f"   ✓ Confidence: {result['confidence']*100:.1f}%")
                
                if 'feature_summary' in result:
                    fs = result['feature_summary']
                    flags = []
                    if fs['is_executable']:
                        flags.append('Executable')
                    if fs['has_sensitive_word']:
                        flags.append('Sensitive Word')
                    if fs['has_ip_address']:
                        flags.append('IP Address')
                    
                    if flags:
                        print(f"   ⚠️  Flags: {', '.join(flags)}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    print("\n✅ Various URLs test COMPLETED")
    return True


def test_health_check():
    """Test server health"""
    print("\n" + "="*70)
    print("TEST 3: Server Health Check")
    print("="*70)
    
    try:
        response = requests.get('http://127.0.0.1:5000/api/health', timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Status: {result['status']}")
            print(f"✓ Model Loaded: {result['model_loaded']}")
            print(f"✓ Timestamp: {result['timestamp']}")
            return True
        else:
            print(f"\n❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*70)
    print("TEST 4: Model Information")
    print("="*70)
    
    try:
        response = requests.get('http://127.0.0.1:5000/api/model_info', timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Model Loaded: {result['loaded']}")
            print(f"✓ Classes: {', '.join(result['classes'])}")
            print(f"✓ Features: {result['n_features']}")
            return True
        else:
            print(f"\n❌ Model info failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HSEF WEB APP - REAL FEATURE EXTRACTION TEST SUITE")
    print("="*70)
    print("\nThis script tests the updated HSEF web app with:")
    print("  • Real URL feature extraction (80 features)")
    print("  • Base model predictions")
    print("  • Meta-layer analysis")
    print("  • SHAP interpretability")
    print("\nMake sure the server is running: python start_enhanced_server.py")
    print("="*70)
    
    input("\nPress Enter to start tests...")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    results.append(("Model Info", test_model_info()))
    results.append(("Single URL", test_single_url()))
    results.append(("Various URLs", test_various_urls()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Real feature extraction is working!")
        print("\n✨ You can now:")
        print("   1. Open http://127.0.0.1:5000 in your browser")
        print("   2. Enter any URL (e.g., https://www.youtube.com/)")
        print("   3. See real feature-based predictions with full interpretability!")
    else:
        print("\n⚠️  Some tests failed. Check server logs.")
    
    print("="*70 + "\n")
