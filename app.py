from flask import Flask, request, render_template
from transformers import pipeline

# Initialize the Flask application
app = Flask(__name__)

# Load the sentiment analysis pipeline
pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define sentiment mapping
def map_sentiment(label):
    star_rating = int(label.split()[0])
    if star_rating in [1, 2]:
        return 'Negative'
    elif star_rating == 3:
        return 'Neutral'
    elif star_rating in [4, 5]:
        return 'Positive'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Get model prediction
        result = pipe(text)[0]
        sentiment = map_sentiment(result['label'])
        
        return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
