import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the preprocessed data from the CSV file
preprocessed_df = pd.read_csv("data/processed/IMDB Dataset Preprocessed.csv")

vectorizer = TfidfVectorizer()
tfidf_features = vectorizer.fit_transform(preprocessed_df['preprocessed_review'])

from sklearn.model_selection import train_test_split

# Get the sentiment labels from the preprocessed DataFrame
labels = preprocessed_df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')
f1 = f1_score(y_test, y_pred, pos_label='positive')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")