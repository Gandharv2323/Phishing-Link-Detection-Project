import pandas as pd

df = pd.read_csv('All.csv')
sample = df.head(1)

print("Training data sample values:")
print(f"urlLen: {sample['urlLen'].values[0]}")
print(f"Entropy_URL: {sample['Entropy_URL'].values[0]}")
print(f"Entropy_Domain: {sample['Entropy_Domain'].values[0]}")
print(f"pathurlRatio: {sample['pathurlRatio'].values[0]}")
print(f"NumberRate_URL: {sample['NumberRate_URL'].values[0]}")
