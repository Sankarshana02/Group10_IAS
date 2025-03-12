from flask import Flask, request, render_template, jsonify
import pickle
import requests
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup

# Initialize Flask App
app = Flask(__name__)

# Get the absolute path to the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

# Load vectorizer and model using the correct path
tfidf = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

# Google Fact Check API
FACT_CHECK_API_KEY = "AIzaSyBPpHQNpjdmV7mh5xysZ3PxIWQrbdQnWFs"  
FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# ClaimBuster API
CLAIMBUSTER_API_KEY = '2e01209b7eba4826bca84b0cf8306b9c'
CLAIMBUSTER_SCORE_API_URL = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
CLAIMBUSTER_KB_API_URL = "https://idir.uta.edu/claimbuster/api/v2/query/knowledge_bases/"

# Initialize NLP components
ps = PorterStemmer()
nltk.download("stopwords")

# Text Processing Function
def text_processing(text):
    text = str(text)
    token = WordPunctTokenizer()
    stop_words = set(stopwords.words("english"))
    
    # Remove non-alphabetic characters and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    
    # Tokenization, Stopword Removal, and Stemming
    text = [ps.stem(word) for word in token.tokenize(text.lower()) if word not in stop_words]
    
    return " ".join(text)

# Function to Query Google Fact Check API
def check_fact_google(query_text):
    params = {
        "query": query_text,
        "key": FACT_CHECK_API_KEY,
        "languageCode": "en"
    }

    response = requests.get(FACT_CHECK_URL, params=params)
    data = response.json()

    if "claims" in data:
        claim_results = []
        for claim in data["claims"]:
            claim_text = claim["text"]
            claim_rating = claim["claimReview"][0]["textualRating"] if "claimReview" in claim else "Unknown"

            claim_results.append({
                "claim_text": claim_text,
                "rating": claim_rating
            })
        
        return claim_results
    else:
        return None  # No fact-checking results found
    
# Function to Query ClaimBuster Score API (Check-Worthiness)
def check_claimbuster_score(query_text):
    headers = {"x-api-key": CLAIMBUSTER_API_KEY}
    data = {"input_text": query_text}

    try:
        response = requests.post(CLAIMBUSTER_SCORE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        return [{"text": query_text, "score": "Error fetching ClaimBuster score"}]

# Function to Query ClaimBuster Knowledge-Based Check API
def check_claimbuster_knowledge(query_text):
    url = CLAIMBUSTER_KB_API_URL+ query_text
    headers = {"x-api-key": CLAIMBUSTER_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        knowledge_data = response.json()

        # Extract fact-checking results
        results = []
        if "justification" in knowledge_data:
            for justification in knowledge_data["justification"]:
                results.append({
                    "claim_text": knowledge_data.get("claim", "No claim text found"),
                    "verdict": justification.get("truth_rating", "Indeterminable"),
                    "source": justification.get("source", "Unknown source"),
                    "justification": justification.get("justification", "No justification available"),
                    "url": knowledge_data.get("url", "#")
                })
        print("Results: ", results)
        return results if results else [{"claim_text": query_text, "verdict": "No relevant fact-checks found."}]

    except Exception as e:
        return [{"claim_text": "Error fetching ClaimBuster Knowledge Base", "verdict": str(e)}]

# Flask Home Route
@app.route("/")
def home():
    return render_template("index.html")

# API Route for Prediction & Fact Checking
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_text = request.form["text"]

        # Run Text Processing
        processed_text = text_processing(input_text)
        ip_vec = tfidf.transform([processed_text])
        model_result = model.predict(ip_vec)[0]

        # Model Prediction
        model_prediction = "Genuine" if model_result == 1 else "Fake"

        # Query Google Fact Check API
        fact_check_results = check_fact_google(input_text)

        # Query ClaimBuster for check-worthy claims
        claimbuster_score = check_claimbuster_score(input_text)

        # Query ClaimBuster Knowledge Base
        claimbuster_kb_results = check_claimbuster_knowledge(input_text)

        # Response Data
        response_data = {
            "model_prediction": model_prediction,
            "fact_check_results": fact_check_results,
            "claimbuster_score": claimbuster_score,
            "claimbuster_kb_results": claimbuster_kb_results
        }

        return jsonify(response_data)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
