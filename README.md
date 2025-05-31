# 🧠 Fake Content Detection System for Social Media

A comprehensive system for detecting **fake text and manipulated images** on social media using **machine learning**, **deep learning**, multilingual models, and **trusted fact-checking APIs**.

## 📽️ Demo

Watch the demo video here 👉 [https://drive.google.com/file/d/1gNQrnnWag5iNafPVhE2rEZmbcoiwSf5Y/view?usp=sharing]

---

## 📌 Features

- ✅ Text classification using **XGBoost**
- 🖼️ Image manipulation detection using **CNN**
- 🌍 Multilingual text detection using **XLM-RoBERTa**
- 🔍 Fact-checking via **Google Fact Check API** and **Claimbuster API**
- 💬 Feedback-based retraining for continuous learning
- 🌐 Frontend interface with Flask and SQLite integration

---

## 🏗️ System Architecture

The system includes:
1. **Text detection model** (XGBoost + XLM-RoBERTa)
2. **Image manipulation detection** (CNN)
3. **Multilingual classification**   
4. **Fact-checking layer** using APIs
5. **User feedback loop** to retrain models dynamically
6. **Frontend interface** for user interaction and feedback submission

---

## 🧰 Tech Stack

| Category           | Tools & Technologies                             |
|--------------------|--------------------------------------------------|
| Programming Language | Python                                         |
| ML/DL Libraries     | TensorFlow, Keras, Scikit-learn, Transformers   |
| Datasets (Text)     | ISOT, Gossipcop                                 |
| Datasets (Image)    | CASIA2, CIPLab Real and Fake Face               |
| Datasets (MultiLang)| HC3, M4GT-Bench, MixSet, CUDRT, Urdu FND       |
| NLP Models          | XGBoost, XLM-RoBERTa                            |
| Backend             | Flask                                           |
| Database            | SQLite                                          |
| APIs                | Google Fact Check, Claimbuster                  |

---

## 📊 Model Performance

### 🔠 Text Classification (XGBoost)
- **Accuracy**: 93.89%
- **Precision**: 91.80%
- **Recall**: 97.94%
- **F1-Score**: 94.77%

### 🖼️ Image Classification (CNN)
- **Accuracy**: 99.10%
- **Validation Accuracy**: 92.44%
- **Manipulation Detection Accuracy**: 93.38%

### 🌐 Multilingual Model (XLM-RoBERTa)
- **Accuracy**: 86.30%
- **Supports detection of**: Machine-generated vs Human-written content

---

## 📁 Project Structure

```
├── Datasets/ 
├── models/ 
├── processed/ 
└── README.md 
```
---

## 🔄 Future Enhancements

- ✅ Image-based fact-checking integration
- ✅ Manipulated video detection
- ✅ Real-time analysis for livestreams
- ✅ Improved multilingual accuracy
- ✅ Scalable backend infrastructure

---


## 👨‍💻 How to Run Locally

1. **Clone this repo**  
   ```bash
   git clone https://github.com/ShivamGS/Fake-Content-Detection-System-for-Social-Media.git
   
   cd fake-content-detection
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Flask server**  
   ```bash
   python app.py
   ```
4. **Access the system ats**
   ```bash
   http://localhost:5000
   ```
---



Google Drive for files:

[https://drive.google.com/drive/folders/1eanlIveRv26J1DVrBXNlPbKMc6j-0ffR?usp=sharing](https://drive.google.com/drive/folders/1EvkhU-lYAnrAhBtKXw27TRbL_ahL8RAr?usp=sharing)
