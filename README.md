# 🎬 Movie Review Rating Prediction System

## 📌 Overview
This project is a **deep learning-based sentiment analysis system** that predicts whether a movie review is **positive or negative**.

It is built using an **LSTM (Long Short-Term Memory) network**, which effectively captures sequential patterns in textual data. The system provides real-time predictions through a web interface.

This project demonstrates a complete **end-to-end pipeline** including:
- Data preprocessing
- Model training
- Evaluation
- Deployment

---

## 🧠 Model Details
- Model Type: LSTM (Deep Learning)
- Embedding Dimension: 64
- LSTM Units: 64
- Dropout: 0.3
- Output: Binary Classification (Positive / Negative)
- Optimizer: Adam
- Loss Function: Binary Crossentropy

---

## 📊 Dataset
- Dataset: IMDb 50K Movie Reviews
- 25,000 Positive Reviews
- 25,000 Negative Reviews
- Balanced dataset for unbiased training :contentReference[oaicite:1]{index=1}

---

## ⚙️ Workflow

1. Input movie review text  
2. Text preprocessing (cleaning, removing HTML tags)  
3. Tokenization and sequence padding  
4. Embedding representation  
5. LSTM-based prediction  
6. Output sentiment (Positive / Negative)

---

## 🔍 Preprocessing Steps
- Removal of HTML tags (`<br/>`)
- Lowercasing text
- Removing special characters
- Tokenization (10,000 vocabulary size)
- Padding to fixed length (200 tokens) :contentReference[oaicite:2]{index=2}

---

## 📈 Model Performance

| Metric | Value |
|------|------|
| Test Accuracy | **86.73%** |
| Validation Accuracy | ~87% |
| Training Accuracy | ~95% |
| Loss | 0.3177 |

👉 Model shows strong performance with slight overfitting controlled using Dropout and EarlyStopping :contentReference[oaicite:3]{index=3}

---

## 🌐 Deployment Architecture

- **Frontend**: HTML (Hosted on Vercel)
- **Backend**: FastAPI (Hosted on Render)
- **Containerization**: Docker

The system is designed with **separate frontend and backend components**, ensuring scalability and modularity :contentReference[oaicite:4]{index=4}

---

## 📂 Project Structure
