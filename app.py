from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Download NLTK resources
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function for text preprocessing
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization and stemming
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split() if word not in set(stopwords.words('english'))])
    return text

# Function to predict news authenticity
def predict_news_authenticity(news):
    processed_news = preprocess_text(news)
    vectorized_news = vectorizer.transform([processed_news])
    prediction = model.predict(vectorized_news)
    if prediction[0] == 0:
        return "Real"
    else:
        return "Fake"

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        prediction = predict_news_authenticity(news_text)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=False,host=='0.0.0.0')
