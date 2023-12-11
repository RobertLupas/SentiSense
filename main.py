import json

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from text_processing import preprocess_dataframe

# Load the JSON file into a Python list
with open('dataset.json', 'r') as file:
  data = json.load(file)

# Convert the data to a Pandas DataFrame for easier manipulation
df = pd.DataFrame(data)

# Preprocess DataFrame using the function from text_processing module
df = preprocess_dataframe(df)

# Vectorize text data using Bag-of-Words (BoW)
vectorizer = CountVectorizer()  # Or use TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])

# Split data into Features (X) and Labels (y)
y = df['Sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Build and Train the Sentiment Analysis Model (Example: Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
