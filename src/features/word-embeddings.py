import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
import pandas as pd

# Load the pre-trained Word2Vec model from Gensim's API
path = api.load("word2vec-google-news-300", return_path=True)
word2vec_model = KeyedVectors.load_word2vec_format(path, binary=True)

# Load the preprocessed data from the CSV file
preprocessed_df = pd.read_csv("data/processed/IMDB Dataset Preprocessed.csv")

def get_review_embedding(review):
    words = review.split()
    word_embeddings = [word2vec_model[word] for word in words if word in word2vec_model.key_to_index]
    if len(word_embeddings) > 0:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

# Assuming 'text' is the column in your DataFrame containing the reviews
review_embeddings = [get_review_embedding(review) for review in preprocessed_df['preprocessed_review']]

# Convert the list of embeddings to a numpy array for further processing
review_embeddings = np.array(review_embeddings)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming 'sentiment' is the column in your DataFrame containing the labels (0 for negative, 1 for positive)
labels = preprocessed_df['sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(review_embeddings, labels, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')
f1 = f1_score(y_test, y_pred, pos_label='positive')

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

