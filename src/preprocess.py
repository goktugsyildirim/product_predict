import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import os
import pickle

def load_data():
    # Load and preprocess the data files
    categories = pd.read_csv('data/Product_Categories.txt', sep=';', header=None, names=['Product_ID', 'Category'])
    explanations = pd.read_csv('data/Product_Explanation.txt', sep=';', header=None, names=['Product_ID', 'Description'])
    data = pd.merge(categories, explanations, on='Product_ID')
    data = data.dropna(subset=['Category', 'Description'])
    return data

def preprocess_data(data):
    # Define text cleaning function
    def clean_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
        tokens = text.split()
        tokens = [word for word in tokens if word not in stopwords.words('turkish')]
        return ' '.join(tokens)
    
    data['Cleaned_Description'] = data['Description'].apply(clean_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['Cleaned_Description'])
    y = data['Category']
    return X, y, vectorizer

def save_preprocessed_data(X, y, vectorizer):
    # Save preprocessed data
    with open('preprocessed/preprocessed_X.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('preprocessed/preprocessed_y.pkl', 'wb') as f:
        pickle.dump(y, f)
    with open('preprocessed/preprocessed_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

def load_preprocessed_data():
    # Load preprocessed data
    with open('preprocessed/preprocessed_X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('preprocessed/preprocessed_y.pkl', 'rb') as f:
        y = pickle.load(f)
    with open('preprocessed/preprocessed_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return X, y, vectorizer
