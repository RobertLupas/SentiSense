import re

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
  text = text.lower()
  text = re.sub(r'[^\w\s]', '', text)
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  return tokens


def preprocess_dataframe(df):
  df['Tokens'] = df['Text'].apply(preprocess_text)
  return df
