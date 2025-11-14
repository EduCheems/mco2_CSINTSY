import pandas as pd

# Load dataset 
df1 = pd.read_excel("MCO2_Assigned dataset.xlsx")
df2 = pd.read_excel("validated dataset 2548-2600.xlsx")
df3 = pd.read_excel("MCO2 Dataset (G29).xlsx")
df4 = pd.read_excel("g3 datasettt.xlsx")

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
