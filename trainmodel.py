import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

import joblib 

# Load datasets
df1 = pd.read_excel("MCO2_Assigned dataset.xlsx")
df2 = pd.read_excel("validated dataset 2548-2600.xlsx")
df3 = pd.read_excel("MCO2 Dataset (G29).xlsx")
df4 = pd.read_excel("g3-datasettt.xlsx")

# Combine/align datasets then clean up to avoid overlapping with ignore index 
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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.70, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

val_predictions = model.predict(X_val_vec)
test_predictions = model.predict(X_test_vec)

print('val predictions test result')
print(classification_report(y_val, val_predictions))

print('test prediction result')
print(classification_report(y_test, test_predictions))

joblib.dump(model, 'pinoybot_model.pkl')
joblib.dump(vectorizer, 'pinoybot_vectorizer.pkl')



