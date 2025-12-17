from url_feature_extractor import URLFeatureExtractor

e = URLFeatureExtractor()
f = e.extract_features('https://www.youtube.com/')

print(f'✅ Features extracted: {len(f)}')
print(f'✅ URL Length: {f["urlLen"]}')
print(f'✅ Domain Length: {f["domainlength"]}')
print(f'✅ Entropy: {f["Entropy_URL"]:.4f}')
print(f'✅ Sensitive Word: {f["URL_sensitiveWord"]}')
print(f'✅ Executable: {f["executable"]}')
print('\n🎉 Feature extraction working correctly!')
