import pandas as pd

# Load dataset 
df = pd.read_excel("MCO2_Assigned dataset.xlsx")

# Show some rows
print(df.head())

# Keep only important columns
df = df[['word', 'label']].dropna()

# Normalize label names
df['label'] = df['label'].replace({
    'SYM': 'OTH',
    'UNK': 'OTH',
    'EXPR': 'OTH',
    'ABB': 'OTH',
    'NUM': 'OTH'
})

print(df.head())
print(df['label'].value_counts())

# Extract features here maybe ?