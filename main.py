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
    
    # Call the function from the train_model module to train and save the model
    train_save_model('dataset.json', 'trained_model.pkl')
    # Load the newly trained model
    with open('trained_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
