from flask import Flask, render_template, request
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorization = pickle.load(file)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news_text = request.form["news_text"]
        cleaned_text = clean_text(news_text)
        text_vectorized = vectorization.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        prediction = "Fake News" if prediction == 0 else "Not a Fake News"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
