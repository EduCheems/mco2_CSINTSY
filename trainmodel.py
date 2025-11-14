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

X = df['word']
y = df['label']

# Data Split (70-15-15)
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
# FIX: Explicitly cast the OHE DataFrames to int to ensure NumPy array compatibility.
X_train_combined = hstack([X_train_vec, ohe_train.astype(int).values])
X_val_combined = hstack([X_val_vec, ohe_val.astype(int).values])
X_test_combined = hstack([X_test_vec, ohe_test.astype(int).values])

# Model Training and Evaluation
model = MultinomialNB()
model.fit(X_train_combined, y_train)

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