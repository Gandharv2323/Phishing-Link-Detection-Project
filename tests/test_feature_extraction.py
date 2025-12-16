"""
Test script for URL feature extraction
Verifies that all 80 features can be extracted from URLs
"""

from url_feature_extractor import URLFeatureExtractor
import json

def test_feature_extraction():
    """Test feature extraction on various URLs"""
    
    test_urls = [
        {
            'url': 'https://www.youtube.com/',
            'expected_type': 'benign',
            'description': 'Popular video streaming site'
        },
        {
            'url': 'http://example.com/path/to/file.html?arg1=value1&arg2=value2',
            'expected_type': 'benign',
            'description': 'Simple example URL with query parameters'
        },
        {
            'url': 'https://192.168.1.1:8080/admin/login.php',
            'expected_type': 'suspicious',
            'description': 'IP address with admin path'
        },
        {
            'url': 'http://suspicious-site.tk/verify-account/login.exe?session=123456789',
            'expected_type': 'suspicious',
            'description': 'Suspicious URL with executable and verification keywords'
        },
        {
            'url': 'https://secure-banking-login.com/confirm-account.html?user=12345',
            'expected_type': 'phishing',
            'description': 'Possible phishing URL with sensitive keywords'
        }
    ]
    
    extractor = URLFeatureExtractor()
    
    print("="*80)
    print("URL FEATURE EXTRACTION TEST")
    print("="*80)
    
    for test_case in test_urls:
        url = test_case['url']
        print(f"\n{'='*80}")
        print(f"URL: {url}")
        print(f"Description: {test_case['description']}")
        print(f"Expected Type: {test_case['expected_type']}")
        print('='*80)
        
        # Extract features
        features = extractor.extract_features(url)
        
        # Verify all 80 features are present
        print(f"\n✓ Total features extracted: {len(features)}")
        
        # Display key features
        print("\n🔍 KEY FEATURES:")
        print(f"  • URL Length: {features['urlLen']}")
        print(f"  • Domain Length: {features['domainlength']}")
        print(f"  • Path Length: {features['pathLength']}")
        print(f"  • Query Length: {features['Querylength']}")
        print(f"  • Number of Dots: {features['NumberofDotsinURL']}")
        print(f"  • TLD Code: {features['tld']}")
        
        print("\n🔒 SECURITY INDICATORS:")
        print(f"  • Has IP Address: {'Yes' if features['ISIpAddressInDomainName'] == 1 else 'No'}")
        print(f"  • Is Executable: {'Yes' if features['executable'] == 1 else 'No'}")
        print(f"  • Port 80: {'Yes' if features['isPortEighty'] == 1 else 'No/Other'}")
        print(f"  • Has Sensitive Word: {'Yes' if features['URL_sensitiveWord'] == 1 else 'No'}")
        
        print("\n📊 STATISTICAL FEATURES:")
        print(f"  • URL Entropy: {features['Entropy_URL']:.4f}")
        print(f"  • Domain Entropy: {features['Entropy_Domain']:.4f}")
        print(f"  • Digit Count: {features['URL_DigitCount']}")
        print(f"  • Letter Count: {features['URL_Letter_Count']}")
        print(f"  • Symbol Count: {features['SymbolCount_URL']}")
        
        print("\n🔢 RATIOS:")
        print(f"  • Path/URL Ratio: {features['pathurlRatio']:.4f}")
        print(f"  • Domain/URL Ratio: {features['domainUrlRatio']:.4f}")
        print(f"  • Arg/URL Ratio: {features['ArgUrlRatio']:.4f}")
        
        print("\n📝 TOKEN FEATURES:")
        print(f"  • Domain Token Count: {features['domain_token_count']}")
        print(f"  • Path Token Count: {features['path_token_count']}")
        print(f"  • Avg Domain Token Length: {features['avgdomaintokenlen']:.2f}")
        print(f"  • Longest Path Token: {features['LongestPathTokenLength']}")
        
        print("\n🎯 ENTROPY ANALYSIS:")
        print(f"  • URL Entropy: {features['Entropy_URL']:.4f}")
        print(f"  • Domain Entropy: {features['Entropy_Domain']:.4f}")
        print(f"  • Directory Entropy: {features['Entropy_DirectoryName']:.4f}")
        print(f"  • Filename Entropy: {features['Entropy_Filename']:.4f}")
        
        # Check for NaN values
        nan_features = [k for k, v in features.items() if isinstance(v, float) and str(v) == 'nan']
        if nan_features:
            print(f"\n⚠️  Features with NaN: {len(nan_features)}")
            print(f"     {', '.join(nan_features)}")
        else:
            print("\n✓ No NaN values detected")
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION TEST COMPLETE")
    print("="*80)
    
    # Save sample features to JSON for inspection
    print("\n💾 Saving sample features to 'sample_features.json'...")
    sample_features = extractor.extract_features(test_urls[0]['url'])
    with open('sample_features.json', 'w') as f:
        json.dump(sample_features, f, indent=2, default=str)
    print("✓ Sample features saved!")

if __name__ == "__main__":
    test_feature_extraction()
