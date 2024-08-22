import re
import pandas as pd
import time
import nltk
from multiprocessing import Pool, cpu_count
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):

    # Remove HTML tags and special characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    #Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]

    # Lemmatize the tokens
    lemmanizer = WordNetLemmatizer()
    lemmanized_tokens = [lemmanizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back together
    preprocessed_text = ' '.join(lemmanized_tokens)

    return preprocessed_text


def preprocess_batch(batch):
    batch['preprocessed_review'] = batch['review'].apply(lambda x: preprocess_text(x))
    return batch

if __name__ == '__main__':
    # Record time to preprocess the data for optimization
    start_time = time.time()

    # Read in the data
    df = pd.read_csv("data/raw/IMDB Dataset.csv")
    print("Original DataFrame: ")
    print(df.head())

    # Set the batch size
    batch_size = 1000

    # Split the DataFrame into batches
    batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]

    # Create a pool of worker processes
    with Pool(processes=cpu_count()) as pool:
        # Process each batch in parallel
        processed_batches = pool.map(preprocess_batch, batches)

    # Concatenate the processed batches
    preprocessed_df = pd.concat(processed_batches)
    print("\nPreprocessed DataFrame: ")
    print(preprocessed_df.head())

    # Save the preprocessed data
    preprocessed_df.to_csv("data/processed/IMDB Dataset Preprocessed.csv", index=False)

    # Record end time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")