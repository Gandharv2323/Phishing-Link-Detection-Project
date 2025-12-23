"""
URL Feature Extractor for HSEF Model
Computes all 80 handcrafted features from raw URLs for classification
"""

import re
import math
from urllib.parse import urlparse, parse_qs
from collections import Counter
import string
import tldextract
import numpy as np


class URLFeatureExtractor:
    """
    Comprehensive URL feature extraction matching the 80 features in All.csv
    
    Feature Categories:
    - Lexical: Length, token counts, character compositions
    - Statistical/Structural: Ratios, counts, distributions
    - Entropy: Information entropy of URL components
    - Semantic/Binary: Sensitive words, IP addresses, ports, executables
    """
    
    # Sensitive words commonly found in phishing URLs
    SENSITIVE_WORDS = {
        'login', 'signin', 'account', 'update', 'verify', 'secure', 'banking',
        'password', 'confirm', 'suspend', 'restricted', 'expires', 'click',
        'urgent', 'alert', 'notification', 'verification', 'authenticate',
        'webscr', 'cmd', 'submit', 'billing', 'paypal', 'ebay', 'apple',
        'amazon', 'security', 'wallet', 'transfer', 'reset'
    }
    
    # Common TLDs for reference
    COMMON_TLDS = {
        'com': 1, 'org': 2, 'net': 3, 'edu': 4, 'gov': 5, 
        'mil': 6, 'int': 7, 'info': 8, 'biz': 9, 'name': 10
    }
    
    # Executable file extensions
    EXECUTABLE_EXTENSIONS = {
        'exe', 'dll', 'bat', 'cmd', 'com', 'scr', 'vbs', 'js', 
        'jar', 'app', 'deb', 'rpm', 'dmg', 'apk', 'msi'
    }
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.reset()
    
    def reset(self):
        """Reset internal state"""
        self.url = ""
        self.parsed = None
        self.extracted = None
        self.features = {}
    
    def extract_features(self, url):
        """
        Extract all 80 features from a URL
        
        Args:
            url (str): The URL to extract features from
            
        Returns:
            dict: Dictionary containing all 80 features with their values
        """
        self.reset()
        self.url = url.strip()
        
        # Parse URL using standard library
        try:
            self.parsed = urlparse(self.url)
        except Exception as e:
            print(f"URL parsing error: {e}")
            self.parsed = urlparse("")
        
        # Extract TLD information
        try:
            self.extracted = tldextract.extract(self.url)
        except Exception as e:
            print(f"TLD extraction error: {e}")
            self.extracted = tldextract.extract("")
        
        # Extract all feature categories
        self._extract_basic_features()
        self._extract_token_features()
        self._extract_character_features()
        self._extract_digit_letter_features()
        self._extract_length_features()
        self._extract_ratio_features()
        self._extract_delimiter_features()
        self._extract_number_rate_features()
        self._extract_symbol_count_features()
        self._extract_entropy_features()
        self._extract_semantic_features()
        
        return self.features
    
    def _extract_basic_features(self):
        """Extract basic URL components and properties"""
        # URL length
        self.features['urlLen'] = len(self.url)
        
        # Domain and path lengths
        domain = self.parsed.netloc
        path = self.parsed.path
        query = self.parsed.query
        
        self.features['domainlength'] = len(domain)
        self.features['pathLength'] = len(path)
        self.features['Querylength'] = len(query)
        
        # Argument/query length
        self.features['ArgLen'] = len(query)
        
        # Number of dots in URL
        self.features['NumberofDotsinURL'] = self.url.count('.')
        
        # TLD encoding (map common TLDs to numbers)
        tld = self.extracted.suffix.lower()
        self.features['tld'] = self.COMMON_TLDS.get(tld.split('.')[-1], 0)
        
        # Port check
        port = self.parsed.port
        if port is None:
            # Check scheme for default port
            if self.parsed.scheme == 'http':
                self.features['isPortEighty'] = 1
            elif self.parsed.scheme == 'https':
                self.features['isPortEighty'] = -1  # Different from 80
            else:
                self.features['isPortEighty'] = -1
        else:
            self.features['isPortEighty'] = 1 if port == 80 else -1
        
        # Check if IP address in domain
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        self.features['ISIpAddressInDomainName'] = 1 if re.search(ip_pattern, domain) else -1
        
        # Executable file check
        path_lower = path.lower()
        is_executable = any(path_lower.endswith(f'.{ext}') for ext in self.EXECUTABLE_EXTENSIONS)
        self.features['executable'] = 1 if is_executable else 0
    
    def _extract_token_features(self):
        """Extract token-based features (word/segment counts)"""
        domain = self.extracted.domain + '.' + self.extracted.suffix if self.extracted.suffix else self.extracted.domain
        path = self.parsed.path
        
        # Tokenize domain (split by dots, hyphens, underscores)
        domain_tokens = re.split(r'[.\-_]', domain)
        domain_tokens = [t for t in domain_tokens if t]
        
        # Tokenize path (split by slashes, hyphens, underscores, dots)
        path_tokens = re.split(r'[/\-_.]', path)
        path_tokens = [t for t in path_tokens if t]
        
        self.features['domain_token_count'] = len(domain_tokens)
        self.features['path_token_count'] = len(path_tokens)
        
        # Average token lengths
        if domain_tokens:
            self.features['avgdomaintokenlen'] = sum(len(t) for t in domain_tokens) / len(domain_tokens)
            self.features['longdomaintokenlen'] = max(len(t) for t in domain_tokens)
            self.features['Domain_LongestWordLength'] = max(len(t) for t in domain_tokens)
        else:
            self.features['avgdomaintokenlen'] = 0
            self.features['longdomaintokenlen'] = 0
            self.features['Domain_LongestWordLength'] = 0
        
        if path_tokens:
            self.features['avgpathtokenlen'] = sum(len(t) for t in path_tokens) / len(path_tokens)
            self.features['LongestPathTokenLength'] = max(len(t) for t in path_tokens)
            self.features['Path_LongestWordLength'] = max(len(t) for t in path_tokens)
        else:
            self.features['avgpathtokenlen'] = 0
            self.features['LongestPathTokenLength'] = 0
            self.features['Path_LongestWordLength'] = 0
        
        # Subdirectory and filename features
        path_parts = [p for p in self.parsed.path.split('/') if p]
        
        if path_parts:
            # Subdirectory length (all parts except last)
            subdir = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else ''
            self.features['subDirLen'] = len(subdir)
            
            # Subdirectory longest word
            if subdir:
                subdir_tokens = re.split(r'[/\-_.]', subdir)
                subdir_tokens = [t for t in subdir_tokens if t]
                if subdir_tokens:
                    self.features['sub-Directory_LongestWordLength'] = max(len(t) for t in subdir_tokens)
                else:
                    self.features['sub-Directory_LongestWordLength'] = 0
            else:
                self.features['sub-Directory_LongestWordLength'] = 0
            
            # Filename (last part)
            filename = path_parts[-1]
            # Separate extension if present
            if '.' in filename:
                name_part, ext_part = filename.rsplit('.', 1)
                self.features['fileNameLen'] = len(name_part)
                self.features['this.fileExtLen'] = len(ext_part)
            else:
                self.features['fileNameLen'] = len(filename)
                self.features['this.fileExtLen'] = 0
        else:
            self.features['subDirLen'] = 0
            self.features['fileNameLen'] = 0
            self.features['this.fileExtLen'] = 0
            self.features['sub-Directory_LongestWordLength'] = 0
        
        # Arguments longest word
        query = self.parsed.query
        if query:
            query_tokens = re.split(r'[&=\-_.]', query)
            query_tokens = [t for t in query_tokens if t]
            if query_tokens:
                self.features['Arguments_LongestWordLength'] = max(len(t) for t in query_tokens)
            else:
                self.features['Arguments_LongestWordLength'] = 0
        else:
            self.features['Arguments_LongestWordLength'] = 0
        
        # Number of query variables
        if query:
            query_dict = parse_qs(query)
            self.features['URLQueries_variable'] = len(query_dict)
        else:
            self.features['URLQueries_variable'] = 0
        
        # Longest variable value
        if query:
            query_dict = parse_qs(query)
            if query_dict:
                max_val_len = max(len(str(v[0])) for v in query_dict.values() if v)
                self.features['LongestVariableValue'] = max_val_len
            else:
                self.features['LongestVariableValue'] = 0
        else:
            self.features['LongestVariableValue'] = -1  # No query
    
    def _extract_character_features(self):
        """Extract character composition features"""
        url = self.url
        
        # Vowel count
        vowels = 'aeiouAEIOU'
        self.features['charcompvowels'] = sum(1 for c in url if c in vowels)
        
        # Space count (usually 0, but handle URL-encoded spaces)
        self.features['charcompace'] = url.count(' ') + url.count('%20')
        
        # Special characters count
        special_chars = set(string.punctuation) - set('.-_/?&=')
        self.features['spcharUrl'] = sum(1 for c in url if c in special_chars)
        
        # Character continuity rate (consecutive identical characters)
        continuity_count = 0
        for i in range(len(url) - 1):
            if url[i] == url[i + 1]:
                continuity_count += 1
        self.features['CharacterContinuityRate'] = continuity_count / len(url) if len(url) > 0 else 0
    
    def _extract_digit_letter_features(self):
        """Extract digit and letter count features for URL components"""
        domain = self.parsed.netloc
        path = self.parsed.path
        query = self.parsed.query
        
        # Get filename and extension
        path_parts = [p for p in path.split('/') if p]
        if path_parts:
            filename = path_parts[-1]
            if '.' in filename:
                name_part, ext_part = filename.rsplit('.', 1)
            else:
                name_part, ext_part = filename, ""
        else:
            name_part, ext_part = "", ""
        
        # Directory (path without filename)
        if path_parts:
            directory = '/'.join(path_parts[:-1])
        else:
            directory = path
        
        # Digit counts
        self.features['URL_DigitCount'] = sum(c.isdigit() for c in self.url)
        self.features['host_DigitCount'] = sum(c.isdigit() for c in domain)
        self.features['Directory_DigitCount'] = sum(c.isdigit() for c in directory)
        self.features['File_name_DigitCount'] = sum(c.isdigit() for c in name_part)
        self.features['Extension_DigitCount'] = sum(c.isdigit() for c in ext_part)
        self.features['Query_DigitCount'] = sum(c.isdigit() for c in query) if query else -1
        
        # Letter counts
        self.features['URL_Letter_Count'] = sum(c.isalpha() for c in self.url)
        self.features['host_letter_count'] = sum(c.isalpha() for c in domain)
        self.features['Directory_LetterCount'] = sum(c.isalpha() for c in directory)
        self.features['Filename_LetterCount'] = sum(c.isalpha() for c in name_part)
        self.features['Extension_LetterCount'] = sum(c.isalpha() for c in ext_part)
        self.features['Query_LetterCount'] = sum(c.isalpha() for c in query) if query else -1
    
    def _extract_length_features(self):
        """Extract Levenshtein-based and other length features"""
        # ldl_* features: Longest digit sequence length
        self.features['ldl_url'] = self._longest_digit_sequence(self.url)
        self.features['ldl_domain'] = self._longest_digit_sequence(self.parsed.netloc)
        self.features['ldl_path'] = self._longest_digit_sequence(self.parsed.path)
        
        # Get filename
        path_parts = [p for p in self.parsed.path.split('/') if p]
        filename = path_parts[-1] if path_parts else ""
        query = self.parsed.query
        
        self.features['ldl_filename'] = self._longest_digit_sequence(filename)
        self.features['ldl_getArg'] = self._longest_digit_sequence(query) if query else 0
        
        # dld_* features: Longest letter sequence length
        self.features['dld_url'] = self._longest_letter_sequence(self.url)
        self.features['dld_domain'] = self._longest_letter_sequence(self.parsed.netloc)
        self.features['dld_path'] = self._longest_letter_sequence(self.parsed.path)
        self.features['dld_filename'] = self._longest_letter_sequence(filename)
        self.features['dld_getArg'] = self._longest_letter_sequence(query) if query else 0
    
    def _longest_digit_sequence(self, text):
        """Find the length of the longest consecutive digit sequence"""
        if not text:
            return 0
        max_len = 0
        current_len = 0
        for c in text:
            if c.isdigit():
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len
    
    def _longest_letter_sequence(self, text):
        """Find the length of the longest consecutive letter sequence"""
        if not text:
            return 0
        max_len = 0
        current_len = 0
        for c in text:
            if c.isalpha():
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len
    
    def _extract_ratio_features(self):
        """Extract ratio-based features"""
        url_len = len(self.url)
        domain_len = len(self.parsed.netloc)
        path_len = len(self.parsed.path)
        arg_len = len(self.parsed.query)
        
        # Avoid division by zero
        if url_len > 0:
            self.features['pathurlRatio'] = path_len / url_len
            self.features['ArgUrlRatio'] = arg_len / url_len
            self.features['domainUrlRatio'] = domain_len / url_len
        else:
            self.features['pathurlRatio'] = 0
            self.features['ArgUrlRatio'] = 0
            self.features['domainUrlRatio'] = 0
        
        if domain_len > 0:
            self.features['argDomanRatio'] = arg_len / domain_len
            self.features['pathDomainRatio'] = path_len / domain_len
        else:
            self.features['argDomanRatio'] = 0
            self.features['pathDomainRatio'] = 0
        
        if path_len > 0:
            self.features['argPathRatio'] = arg_len / path_len
        else:
            self.features['argPathRatio'] = 0
    
    def _extract_delimiter_features(self):
        """Extract delimiter count features"""
        domain = self.parsed.netloc
        path = self.parsed.path
        
        # Common delimiters
        delimiters = ['-', '_', '.', '/', '&', '=', '?']
        
        # Domain delimiters (., -, _)
        domain_delims = sum(domain.count(d) for d in ['.', '-', '_'])
        self.features['delimeter_Domain'] = domain_delims
        
        # Path delimiters (/, -, _, .)
        path_delims = sum(path.count(d) for d in ['/', '-', '_', '.'])
        self.features['delimeter_path'] = path_delims
        
        # Total delimiter count
        total_delims = sum(self.url.count(d) for d in delimiters)
        self.features['delimeter_Count'] = total_delims
    
    def _extract_number_rate_features(self):
        """Extract number/digit rate (percentage) features"""
        url = self.url
        domain = self.parsed.netloc
        path = self.parsed.path
        query = self.parsed.query
        
        # Get path components
        path_parts = [p for p in path.split('/') if p]
        if path_parts:
            filename = path_parts[-1]
            if '.' in filename:
                name_part, ext_part = filename.rsplit('.', 1)
            else:
                name_part, ext_part = filename, ""
            directory = '/'.join(path_parts[:-1])
        else:
            directory, name_part, ext_part = "", "", ""
        
        # Calculate digit rates (digits / total_chars)
        self.features['NumberRate_URL'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
        self.features['NumberRate_Domain'] = sum(c.isdigit() for c in domain) / len(domain) if len(domain) > 0 else 0
        self.features['NumberRate_DirectoryName'] = sum(c.isdigit() for c in directory) / len(directory) if len(directory) > 0 else 0
        self.features['NumberRate_FileName'] = sum(c.isdigit() for c in name_part) / len(name_part) if len(name_part) > 0 else 0
        
        # Extension rate (handle NaN in original data)
        if ext_part:
            self.features['NumberRate_Extension'] = sum(c.isdigit() for c in ext_part) / len(ext_part)
        else:
            self.features['NumberRate_Extension'] = np.nan  # Match original data
        
        # After path (query)
        if query:
            self.features['NumberRate_AfterPath'] = sum(c.isdigit() for c in query) / len(query)
        else:
            self.features['NumberRate_AfterPath'] = -1  # No query
    
    def _extract_symbol_count_features(self):
        """Extract symbol/special character count features"""
        url = self.url
        domain = self.parsed.netloc
        path = self.parsed.path
        query = self.parsed.query
        
        # Get path components
        path_parts = [p for p in path.split('/') if p]
        if path_parts:
            filename = path_parts[-1]
            if '.' in filename:
                name_part, ext_part = filename.rsplit('.', 1)
            else:
                name_part, ext_part = filename, ""
            directory = '/'.join(path_parts[:-1])
        else:
            directory, name_part, ext_part = "", "", ""
        
        # Symbols are non-alphanumeric characters
        def count_symbols(text):
            return sum(1 for c in text if not c.isalnum())
        
        self.features['SymbolCount_URL'] = count_symbols(url)
        self.features['SymbolCount_Domain'] = count_symbols(domain)
        self.features['SymbolCount_Directoryname'] = count_symbols(directory)
        self.features['SymbolCount_FileName'] = count_symbols(name_part)
        self.features['SymbolCount_Extension'] = count_symbols(ext_part)
        
        if query:
            self.features['SymbolCount_Afterpath'] = count_symbols(query)
        else:
            self.features['SymbolCount_Afterpath'] = -1
    
    def _extract_entropy_features(self):
        """Extract Shannon entropy features for URL components"""
        url = self.url
        domain = self.parsed.netloc
        path = self.parsed.path
        query = self.parsed.query
        
        # Get path components
        path_parts = [p for p in path.split('/') if p]
        if path_parts:
            filename = path_parts[-1]
            if '.' in filename:
                name_part, ext_part = filename.rsplit('.', 1)
            else:
                name_part, ext_part = filename, ""
            directory = '/'.join(path_parts[:-1])
        else:
            directory, name_part, ext_part = "", "", ""
        
        # Calculate entropy for each component
        self.features['Entropy_URL'] = self._calculate_entropy(url)
        self.features['Entropy_Domain'] = self._calculate_entropy(domain)
        self.features['Entropy_DirectoryName'] = self._calculate_entropy(directory)
        self.features['Entropy_Filename'] = self._calculate_entropy(name_part)
        
        # Extension entropy (handle empty case)
        if ext_part:
            self.features['Entropy_Extension'] = self._calculate_entropy(ext_part)
        else:
            self.features['Entropy_Extension'] = np.nan  # Match original data
        
        # After path entropy
        if query:
            self.features['Entropy_Afterpath'] = self._calculate_entropy(query)
        else:
            self.features['Entropy_Afterpath'] = -1
    
    def _calculate_entropy(self, text):
        """
        Calculate Shannon entropy of a text string
        H(X) = -Σ p(x) * log2(p(x))
        """
        if not text or len(text) == 0:
            return 0.0
        
        # Count character frequencies
        char_counts = Counter(text)
        length = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _extract_semantic_features(self):
        """Extract semantic and binary features"""
        url_lower = self.url.lower()
        
        # Check for sensitive words
        has_sensitive = any(word in url_lower for word in self.SENSITIVE_WORDS)
        self.features['URL_sensitiveWord'] = 1 if has_sensitive else 0
    
    def get_feature_vector(self, feature_names):
        """
        Get features in the correct order matching the model's expected input
        
        Args:
            feature_names (list): List of feature names in the expected order
            
        Returns:
            list: Feature values in the correct order
        """
        return [self.features.get(name, 0) for name in feature_names]
    
    def get_feature_dict(self):
        """
        Get all features as a dictionary
        
        Returns:
            dict: Dictionary of all features
        """
        return self.features.copy()


# Convenience function for quick extraction
def extract_url_features(url):
    """
    Extract all features from a URL
    
    Args:
        url (str): URL to extract features from
        
    Returns:
        dict: Dictionary of all 80 features
    """
    extractor = URLFeatureExtractor()
    return extractor.extract_features(url)


if __name__ == "__main__":
    # Test the feature extractor
    test_urls = [
        "https://www.youtube.com/",
        "http://example.com/path/to/file.html?arg1=value1&arg2=value2",
        "https://192.168.1.1:8080/admin/login.php",
        "http://suspicious-site.tk/verify-account/login.exe?session=123456789"
    ]
    
    extractor = URLFeatureExtractor()
    
    for url in test_urls:
        print(f"\n{'='*70}")
        print(f"URL: {url}")
        print('='*70)
        
        features = extractor.extract_features(url)
        
        # Print some key features
        print(f"\nKey Features:")
        print(f"  URL Length: {features['urlLen']}")
        print(f"  Domain Length: {features['domainlength']}")
        print(f"  Path Length: {features['pathLength']}")
        print(f"  Number of Dots: {features['NumberofDotsinURL']}")
        print(f"  Has IP Address: {features['ISIpAddressInDomainName']}")
        print(f"  Is Executable: {features['executable']}")
        print(f"  Has Sensitive Word: {features['URL_sensitiveWord']}")
        print(f"  URL Entropy: {features['Entropy_URL']:.4f}")
        print(f"  Domain Entropy: {features['Entropy_Domain']:.4f}")
        print(f"  Digit Count: {features['URL_DigitCount']}")
        print(f"  Letter Count: {features['URL_Letter_Count']}")
        
        print(f"\nTotal features extracted: {len(features)}")
