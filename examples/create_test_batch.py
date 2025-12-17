import pandas as pd

# Create a test batch file with 20 samples
df = pd.read_csv('All.csv')
test_df = df.head(20)
test_df.to_csv('test_batch.csv', index=False)

print(f"✓ Created test_batch.csv with {len(test_df)} samples")
print(f"  Classes in sample: {test_df['URL_Type_obf_Type'].value_counts().to_dict()}")
print("\nYou can now upload this file in the web interface for batch classification!")
