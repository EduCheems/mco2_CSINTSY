"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import joblib 
import re
import pandas as pd
from scipy.sparse import hstack
from typing import List

# Helper function to extract prefix/suffix (Must match trainmodel.py)
def extract_prefix_suffix(word, n=3):
	# Returns the prefix and suffix for a single word token
	word_l = str(word).lower()
	return pd.Series({
		f'prefix_{n}': word_l[:n],
		f'suffix_{n}': word_l[-n:]
	})

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
	# Check for empty input
	if not tokens:
		return []

	# File paths for the saved components
	model_path = 'pinoybot_model.pkl'
	vectorizer_path = 'pinoybot_vectorizer.pkl'
	ohe_cols_path = 'pinoybot_ohe_cols.pkl'

	# Minimal file existence check
	if not all(os.path.exists(p) for p in [model_path, vectorizer_path, ohe_cols_path]):
		return ['OTH'] * len(tokens) 

	try:
		# 1. Load trained model and transformers
		model = joblib.load(model_path)
		vectorizer = joblib.load(vectorizer_path)
		ohe_cols_list = joblib.load(ohe_cols_path)
	except Exception:
		return ['OTH'] * len(tokens)

	# 2. Extract Prefix/Suffix features and align columns
	df_tokens = pd.DataFrame({'word': tokens})
	X_pred_cat_raw = df_tokens['word'].apply(extract_prefix_suffix).reset_index(drop=True)
	X_pred_ohe = pd.get_dummies(X_pred_cat_raw, prefix=['pre', 'suf'])

	# Reindex OHE matrix to match training columns (fixes the feature count mismatch)
	X_pred_ohe_aligned = X_pred_ohe.reindex(columns=ohe_cols_list, fill_value=0)

	# 3. Vectorize text features
	X_pred_vec = vectorizer.transform(tokens)

	# 4. Combine features (TF-IDF + OHE)
	X_combined_pred = hstack([X_pred_vec, X_pred_ohe_aligned.astype(int).values])

	# 5. Use the model to predict the tags
	predicted_tags = model.predict(X_combined_pred)
	
	# 6. Return the list of tags
	return predicted_tags.tolist()

if __name__ == "__main__":
	# Example usage
	example_tokens = ["Love", "kita", "."]
	print("Tokens:", example_tokens)
	tags = tag_language(example_tokens)
	print("Predicted Tags:", tags)
	
	# Add a second test case for better verification
	example_tokens_2 = ["nag-lunch", "sa", "park"]
	print("\nTokens:", example_tokens_2)
	tags_2 = tag_language(example_tokens_2)
	print("Predicted Tags:", tags_2)