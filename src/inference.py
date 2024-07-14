import pickle
import os

# Load the best model and vectorizer
model_path = 'model/best_model.pkl'
vectorizer_path = 'model/vectorizer.pkl'

with open(model_path, 'rb') as model_file, open(vectorizer_path, 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    vectorizer = pickle.load(vectorizer_file)

def predict(description):
    # Transform the input description using the loaded vectorizer
    description_vectorized = vectorizer.transform([description])
    # Predict the category using the loaded model
    prediction = model.predict(description_vectorized)
    return prediction[0]
