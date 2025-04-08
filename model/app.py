from flask import Flask, request, render_template, jsonify
import pickle
import requests
import re
import nltk
import os
import sqlite3
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "feedback_database.db")

def setup_database():
    db_exists = os.path.isfile(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if db_exists:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            cursor.execute("PRAGMA table_info(user_feedback)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'correct_classification' not in columns or 'confidence_level' not in columns:
                cursor.execute("DROP TABLE user_feedback")
                table_exists = False

    if not db_exists or not table_exists:
        cursor.execute('''
        CREATE TABLE user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_text TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            prediction_result TEXT,
            correct_classification TEXT,
            confidence_level INTEGER,
            ip_address TEXT
        )
        ''')

    conn.commit()
    conn.close()

setup_database()

# tfidf = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
# model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

FACT_CHECK_API_KEY = "AIzaSyBPpHQNpjdmV7mh5xysZ3PxIWQrbdQnWFs"
FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

CLAIMBUSTER_API_KEY = '2e01209b7eba4826bca84b0cf8306b9c'
CLAIMBUSTER_SCORE_API_URL = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
CLAIMBUSTER_KB_API_URL = "https://idir.uta.edu/claimbuster/api/v2/query/knowledge_bases/"

ps = PorterStemmer()
nltk.download("stopwords")

# Define device and relative path to the multilingual model directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODEL_PATH = os.path.join(BASE_DIR, "../multilingual_model/best")

# Load tokenizer and model from Hugging Face format
tokenizer = AutoTokenizer.from_pretrained(ML_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(ML_MODEL_PATH).to(device)

def text_processing(text):
    text = str(text)
    token = WordPunctTokenizer()
    stop_words = set(stopwords.words("english"))

    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"[0-9]", " ", text)

    text = [ps.stem(word) for word in token.tokenize(text.lower()) if word not in stop_words]

    return " ".join(text)

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
        return None

def check_claimbuster_score(query_text):
    headers = {"x-api-key": CLAIMBUSTER_API_KEY}
    data = {"input_text": query_text}

    try:
        response = requests.post(CLAIMBUSTER_SCORE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        return [{"text": query_text, "score": "Error fetching ClaimBuster score"}]

def check_claimbuster_knowledge(query_text):
    url = CLAIMBUSTER_KB_API_URL+ query_text
    headers = {"x-api-key": CLAIMBUSTER_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        knowledge_data = response.json()

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
    
def useMultilingual(someText):
    inputs = tokenizer(someText, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_class_id = logits.argmax(dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_id]
    
    return predicted_label

testString = "選挙制度の変更をめぐっては、香港の反体制派が実質的に消滅するとの批判が出ている 採択された決定案では、香港の選挙において、民主的に選出される候補者を減らし、中国政府寄りの委員会が審査で認める「愛国的な」候補者を増やす。 この案の採択は広く予想されていた。 この日の全人代では、賛成2895、反対0、棄権1で採択された。 詳細な内容はこれから作成され、数カ月以内に香港で施行されるとみられる。 全人代には数千人の代表者が集まった 昨年の全人代では、香港国家安全維持法（国安法）案を可決した。実質的に香港の自治権を弱め、デモ参加者の処罰を簡単にする内容で、やはり強い批判を招いていた。 昨年6月に国案法が施行されて以降、香港では何十人もが同法違反容疑で拘束されている。 毎年開催される全人代は5日に開幕。中国で最も影響力がある政治諮問機関、中国人民政治協商会議（CPPCC）とほぼ並行して開催されてきた。 それらは2大会議として知られ、1週間にわたって続く。 今年の全人代では、経済成長目標や環境政策も政府によって議題とされ、ともに承認された。"
predicted_label = useMultilingual(testString)
print(f"Predicted Label: {predicted_label}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_text = request.form["text"]

        processed_text = text_processing(input_text)
        # ip_vec = tfidf.transform([processed_text])
        # model_result = model.predict(ip_vec)[0]

        # model_prediction = "Genuine" if model_result == 1 else "Fake"
        model_prediction = "N/A"

        fact_check_results = check_fact_google(input_text)

        claimbuster_score = check_claimbuster_score(input_text)

        claimbuster_kb_results = check_claimbuster_knowledge(input_text)

        response_data = {
            "model_prediction": model_prediction,
            "fact_check_results": fact_check_results,
            "claimbuster_score": claimbuster_score,
            "claimbuster_kb_results": claimbuster_kb_results
        }

        return jsonify(response_data)

@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    if request.method == "POST":
        try:
            data = request.json

            claim_text = data.get('claim_text', '')
            feedback_type = data.get('feedback_type', 'classification')  # Default to 'classification'
            timestamp = data.get('timestamp', datetime.now().isoformat())

            prediction_result = data.get('prediction_result', '')

            correct_classification = data.get('correct_classification', '')
            confidence_level = data.get('confidence_level', 0)

            ip_address = request.remote_addr

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute(
                '''INSERT INTO user_feedback 
                   (claim_text, feedback_type, timestamp, prediction_result, 
                   correct_classification, confidence_level, ip_address) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (claim_text, feedback_type, timestamp, prediction_result,
                 correct_classification, confidence_level, ip_address)
            )

            conn.commit()
            conn.close()

            return jsonify({"status": "success", "message": "Feedback recorded successfully"})

        except Exception as e:
            print(f"Error saving feedback: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)