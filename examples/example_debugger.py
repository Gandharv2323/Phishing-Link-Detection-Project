"""
Example: Using HSEF Debugger to Analyze Misclassifications
"""

import pandas as pd
from hsef_debugger import HSEFDebugger

print("="*70)
print("HSEF DEBUGGER - EXAMPLE USAGE")
print("="*70)

# Initialize debugger
debugger = HSEFDebugger()

# Example 1: Analyze a single URL
print("\n" + "="*70)
print("EXAMPLE 1: Single URL Analysis")
print("="*70)

# Load a sample from the dataset
df = pd.read_csv('All.csv')
sample = df.sample(n=1, random_state=42)
features = sample.drop('URL_Type_obf_Type', axis=1).iloc[0]
actual_class = sample['URL_Type_obf_Type'].values[0]

print(f"\nAnalyzing a random URL with actual class: {actual_class}")

# Analyze
result = debugger.analyze_url(
    url_features=features,
    url_name="Example_URL_1",
    actual_class=actual_class
)

print("\n" + "="*70)
print("Analysis complete! Check 'debug_results/' folder for outputs:")
print("  - PNG visualization with 4 plots")
print("  - JSON detailed report")
print("  - CSV summary")
print("="*70)

# Example 2: Analyze a batch of URLs
print("\n\n" + "="*70)
print("EXAMPLE 2: Batch Analysis")
print("="*70)

# Create a small test set
print("\nCreating test set with 10 URLs...")
test_df = df.sample(n=10, random_state=123)
test_df.to_csv('debug_test_batch.csv', index=False)
print("✓ Saved to debug_test_batch.csv")

# Analyze the batch
print("\nAnalyzing batch of 10 URLs...")
results = debugger.analyze_csv('debug_test_batch.csv')

print("\n" + "="*70)
print("BATCH ANALYSIS COMPLETE")
print("="*70)
print(f"Analyzed {len(results)} URLs")
print("\nOutputs generated:")
print("  - Individual analysis for each URL (PNG, JSON, CSV)")
print("  - Aggregate summary CSV")
print(f"  - All saved in: {debugger.output_dir}/")

# Example 3: Analyze misclassifications only
print("\n\n" + "="*70)
print("EXAMPLE 3: Finding Misclassifications")
print("="*70)

misclassified = [r for r in results 
                 if r.get('misclassification_analysis', {}).get('is_misclassified', False)]

if misclassified:
    print(f"\nFound {len(misclassified)} misclassification(s):")
    for i, mc in enumerate(misclassified, 1):
        print(f"\n{i}. {mc['url']}")
        print(f"   Actual: {mc['actual_class']}")
        print(f"   Predicted: {mc['predicted_class']} ({mc['confidence']:.1%} confidence)")
        
        # Show insights
        insights = mc['misclassification_analysis'].get('insights', [])
        if insights:
            print(f"   Key insights:")
            for insight in insights[:2]:
                print(f"     - {insight}")
else:
    print("\n✓ No misclassifications found in this batch!")

print("\n" + "="*70)
print("EXAMPLES COMPLETE")
print("="*70)
print("\nYou can now:")
print("1. Check the 'debug_results/' folder for all visualizations and reports")
print("2. Use the debugger on your own URLs or CSV files")
print("3. Integrate with the web app for live debugging")
