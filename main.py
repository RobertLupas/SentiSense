import json
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from text_processing import preprocess_dataframe
from train_model import train_save_model

# Check if the trained model exists
try:
    with open('trained_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Training model...")
    
    # Train and save the model
    with open('dataset.json', 'r') as file:
        dataset = json.load(file)
    train_save_model(dataset, 'trained_model.pkl')
    with open('trained_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
