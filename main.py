from flask import Flask
from flask import render_template

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()