import json
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from text_processing import preprocess_dataframe

def train_save_model(data, model_store_file):
    # Load the JSON data into a Python list
    # data = json.loads(data) # If data is in JSON format as a string, uncomment this line

    # Convert the data to a Pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Preprocess DataFrame using the function from text_processing module
    df = preprocess_dataframe(df)

    # Vectorize text data using Bag-of-Words (BoW)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Text'])

    # Split data into Features (X) and Labels (y)
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and Train the Sentiment Analysis Model (Example: Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate Model
    accuracy = model.score(X_test, y_test)
    print("Model Accuracy:", accuracy)

    # Save the trained model to a file
    with open(model_store_file, 'wb') as model_store_file:
        pickle.dump(model, model_store_file)

# Execute training and model saving if this script is directly run
if __name__ == "__main__":
    print("Error. This file needs to be run from another file, with necessary arguments.")
