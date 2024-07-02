from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline
import pandas as pd
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    score = result['score']
    label = result['label']

    # Map the sentiment to custom categories
    if label == 'NEGATIVE':
        if score > 0.9:
            sentiment = 'worst'
        else:
            sentiment = 'negative'
    elif label == 'POSITIVE':
        if score > 0.9:
            sentiment = 'excellent'
        else:
            sentiment = 'good'
    else:
        sentiment = 'average'  # Adjust as necessary based on your model's outputs

    return sentiment

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['message']
    sentiment = analyze_sentiment(text)
    return render_template('front.html', message=text, sentiment=sentiment)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Process the CSV file and get sentiment analysis
        review_counts = process_csv(file_path)
        return render_template('front.html', review_counts=review_counts)
    return redirect(url_for('home'))

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    sentiments = df['review'].apply(analyze_sentiment)
    review_counts = sentiments.value_counts().to_dict()
    return review_counts

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(port=2000, debug=True)
