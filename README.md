# 🛒 Customer Review Sentiment Analyzer (NLP Project)

A Natural Language Processing (NLP) web app that analyzes Amazon product reviews and predicts whether the sentiment is **Positive 😊** or **Negative 😞**.
The project demonstrates end-to-end NLP pipeline: preprocessing → feature extraction → model training → evaluation → deployment.

🔗 **Live App:** [https://huggingface.co/spaces/manishadharmik/customer-review-sentiment-analyzer](https://huggingface.co/spaces/manishadharmik/customer-review-sentiment-analyzer)
📦 **Repository:** [https://github.com/manishadharmik7/amazon-sentiment](https://github.com/manishadharmik7/amazon-sentiment)

---

## 📖 Overview

E-commerce platforms receive millions of customer reviews.
Manually analyzing sentiment is slow and inefficient—this app automates the process using **TF-IDF + Logistic Regression**, enabling fast and accurate insights.

---

## 🎯 Objective

* Automatically classify review sentiment
* Help businesses understand customer opinion at scale
* Build a lightweight, fast, deployable NLP model

---

## ⚙️ Workflow

### **1️⃣ Data Source**

* Amazon Product Reviews (FastText format)
* Labels:

  * `__label__1` → Positive
  * `__label__2` → Negative

### **2️⃣ Data Preprocessing**

* Lowercasing
* Removing punctuation & digits
* Strip extra spaces
* Tokenization

### **3️⃣ Feature Engineering**

* **TF-IDF Vectorizer**
* `max_features = 10,000`

### **4️⃣ Model Building**

* **Logistic Regression → ~92% accuracy**
* **XGBoost → ~90% accuracy**

### **5️⃣ Deployment**

* Interactive **Streamlit** UI
* Deployed on **Hugging Face Spaces**

---

## 💻 Tech Stack

* Python
* Streamlit
* scikit-learn
* XGBoost
* pandas, numpy
* joblib
* Hugging Face Spaces

---

## 🧠 How It Works

1. User enters a review
2. Text is preprocessed
3. Converted into vectors using **TF-IDF**
4. Model predicts sentiment
5. Output displayed instantly:

   * 😊 **Positive**
   * 😞 **Negative**

---

## 📂 Folder Structure

```
amazon-sentiment/
│── app.py                      # Streamlit application
│── sentiment_lr_model.pkl      # Trained Logistic Regression model
│── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
│── requirements.txt
│── README.md
└── data/                       # (optional) dataset for local testing
```

---

## 📊 Results

| Model               | Accuracy | Features |
| ------------------- | -------- | -------- |
| Logistic Regression | 92%      | TF-IDF   |
| XGBoost             | 90%      | TF-IDF   |

---

## 🏆 Skills Demonstrated

* Natural Language Processing
* Text Preprocessing
* TF-IDF Vectorization
* Logistic Regression & XGBoost
* Model Evaluation
* Streamlit App Development
* Deployment on Hugging Face

---
