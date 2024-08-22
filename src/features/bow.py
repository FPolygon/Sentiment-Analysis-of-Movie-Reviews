import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the preprocessed data from the CSV file
preprocessed_df = pd.read_csv("data/processed/IMDB Dataset Preprocessed.csv")

# Get the sentiment labels from the preprocessed DataFrame
vectorizer = CountVectorizer()

# Fit and transform the preprocessed reviews
bow_features = vectorizer.fit_transform(preprocessed_df['preprocessed_review'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get the sentiment labels from the preprocessed DataFrame
labels = preprocessed_df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bow_features, labels, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to allow more iterations
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')  # Specify the positive label
recall = recall_score(y_test, y_pred, pos_label='positive')  # Specify the positive label
f1 = f1_score(y_test, y_pred, pos_label='positive')  # Specify the positive label

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)