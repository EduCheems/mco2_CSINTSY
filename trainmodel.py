import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib 
import re 

def extract_prefix_suffix(word, n=3):
    # Returns the first n characters (prefix) and last n characters (suffix).
    word_l = str(word).lower()
    return pd.Series({
        f'prefix_{n}': word_l[:n],
        f'suffix_{n}': word_l[-n:]
    })

# Load datasets
df1 = pd.read_excel("MCO2_Assigned dataset.xlsx")
df2 = pd.read_excel("validated dataset 2548-2600.xlsx")
df3 = pd.read_excel("MCO2 Dataset (G29).xlsx")
df4 = pd.read_excel("g3-datasettt.xlsx")

# Combine datasets 
df = pd.concat([df1, df2, df3, df4], ignore_index = True)
print(df.head())

# Keep only important columns
df = df[['word', 'label']].dropna()
original_total_size = len(df)

# Normalize label names
df['label'] = df['label'].replace({
    'Fil': 'FIL',
    'Eng': 'ENG',
    'CS': 'FIL',
    'Fil-NE': 'FIL',
    'Eng-NE': 'ENG',
    'NE': 'OTH',
    'SYM': 'OTH',
    'UNK': 'OTH',
    'EXPR': 'OTH',
    'ABB': 'OTH',
    'NUM': 'OTH' 
})

# BEGIN: DYNAMIC CAPPING OF OVERREPRESENTED CLASSES
CATEGORY_COL = 'label'
# Define the maximum imbalance factor (e.g., 5 means larger classes can be up to 5x the smallest class)
MAX_IMBALANCE_FACTOR = 6

print(f"\nBefore Capping: Original Data Size: {original_total_size}")
print(f"Original Class Counts:\n{df[CATEGORY_COL].value_counts()}")

# 1. Count class frequencies and find the size of the smallest class
class_counts = df[CATEGORY_COL].value_counts()
N_min = class_counts.min()
MAX_ALLOWED_COUNT = N_min * MAX_IMBALANCE_FACTOR

print(f"\nSmallest class size (N_min): {N_min}")
print(f"Max allowed count for any class (N_min * {MAX_IMBALANCE_FACTOR}): {MAX_ALLOWED_COUNT}")

# 2. Perform the selective undersampling
capped_df_list = []
for category_value, current_count in class_counts.items():
    category_data = df[df[CATEGORY_COL] == category_value]
    
    # Check if this class is overrepresented and needs capping
    if current_count > MAX_ALLOWED_COUNT:
        # Cap the count by sampling down
        target_count = MAX_ALLOWED_COUNT
        sampled_data = category_data.sample(n=target_count, random_state=42)
        print(f"Capping '{category_value}' from {current_count} to {target_count}")
    else:
        # Keep the class as is (including the smallest class)
        sampled_data = category_data
        
    capped_df_list.append(sampled_data)

# 3. Combine and shuffle the resulting data
df = pd.concat(capped_df_list).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter Capping: Final Data Size: {len(df)}")
print(f"Final Class Counts:\n{df[CATEGORY_COL].value_counts()}")
# END: DYNAMIC CAPPING OF OVERREPRESENTED CLASSES

X = df['word']
y = df['label']

# Data Split (70-15-15) - Now based on the capped dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.70, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)


# 1. Extract Prefix/Suffix Features (N=3)
X_train_cat = X_train.apply(extract_prefix_suffix).reset_index(drop=True)
X_val_cat = X_val.apply(extract_prefix_suffix).reset_index(drop=True)
X_test_cat = X_test.apply(extract_prefix_suffix).reset_index(drop=True)

# 2. One-Hot Encode Categorical Features
# Fit the encoder ONLY on the training data
ohe_train = pd.get_dummies(X_train_cat, prefix=['pre', 'suf'])

# Reindex validation/test data using training columns
ohe_val = pd.get_dummies(X_val_cat, prefix=['pre', 'suf']).reindex(columns=ohe_train.columns, fill_value=0)
ohe_test = pd.get_dummies(X_test_cat, prefix=['pre', 'suf']).reindex(columns=ohe_train.columns, fill_value=0)

# 3. Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# 4. Combine Features (TF-IDF + One-Hot Encoded Prefixes/Suffixes)
X_train_combined = hstack([X_train_vec, ohe_train.astype(int).values])
X_val_combined = hstack([X_val_vec, ohe_val.astype(int).values])
X_test_combined = hstack([X_test_vec, ohe_test.astype(int).values])

# Model Training and Evaluation
model = MultinomialNB()
model.fit(X_train_combined, y_train)

print("\nTraining Data Class Balance")
# This prints the count of each label (FIL, ENG, OTH) in the 70% training data.
print(y_train.value_counts())

print("\nTraining Data Class Proportions")
print(y_train.value_counts(normalize=True), "\n")

val_predictions = model.predict(X_val_combined)
test_predictions = model.predict(X_test_combined)

print('val predictions test result')
print(classification_report(y_val, val_predictions))

print('test prediction result')
print(classification_report(y_test, test_predictions))

# Save Model and Vectorizers for pinoybot.py
joblib.dump(model, 'pinoybot_model.pkl')
joblib.dump(vectorizer, 'pinoybot_vectorizer.pkl')
joblib.dump(ohe_train.columns.tolist(), 'pinoybot_ohe_cols.pkl')