from flask import Flask, request
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

#command
#python -m venv venv
#source venv/bin/activate  # On Windows use `venv\Scripts\activate`
#pip install Flask
#pip install transformers
#pip install torch
#pip install tokenizers .optional
#pip install Flask transformers torch
#

# Initialize the Flask application
app = Flask(__name__)

# Load the sentiment analysis pipeline
pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

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
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Historical Sentiment Analyzer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                width: 500px;
                text-align: center;
            }
            h1 {
                color: #333;
            }
            textarea {
                width: 100%;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ccc;
                margin-bottom: 20px;
                resize: vertical;
            }
            input[type="submit"] {
                background-color: #007BFF;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            input[type="submit"]:hover {
                background-color: #0056b3;
            }
            .result {
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ccc;
                margin-top: 20px;
                text-align: left;
            }
            .back-link {
                display: inline-block;
                margin-top: 20px;
                color: #007BFF;
                text-decoration: none;
            }
            .back-link:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Historical Sentiment Analyser</h1>
            <form action="/predict" method="post">
                <label for="text">Enter text:</label><br><br>
                <textarea id="text" name="text" rows="4" cols="50"></textarea><br><br>
                <input type="submit" value="Analyze it!">
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Get model prediction
        result = pipe(text)[0]
        sentiment = map_sentiment(result['label'])
        
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sentiment Analysis Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    width: 500px;
                    text-align: center;
                }}
                h1 {{
                    color: #333;
                }}
                .result {{
                    background-color: #f9f9f9;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                    margin-top: 20px;
                    text-align: left;
                }}
                .back-link {{
                    display: inline-block;
                    margin-top: 20px;
                    color: #007BFF;
                    text-decoration: none;
                }}
                .back-link:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Sentiment Analysis Result</h1>
                <div class="result">
                    <p><strong>Input text:</strong> {text}</p>
                    <p><strong>Predicted Sentiment:</strong> {sentiment}</p>
                </div>
                <a class="back-link" href="/">Reset</a>
            </div>
        </body>
        </html>
        '''

if __name__ == '__main__':
    app.run(debug=True)
