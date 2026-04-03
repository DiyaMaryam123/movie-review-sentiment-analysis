import re, json, pickle, time, os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE = Path(__file__).parent

print("Loading tokenizer and config...")
tokenizer = pickle.load(open(BASE / "tokenizer.pkl", "rb"))
config    = json.load(open(BASE / "model_config.json"))
MAX_LEN   = config["max_len"]

print("Loading model...")
import keras
model = keras.models.load_model(str(BASE / "sentiment_model_fixed.keras"))
print(f"Ready! MAX_LEN={MAX_LEN}")

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return re.sub(r'\s+', ' ', text).lower().strip()

def predict(text):
    seq    = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    score  = float(model.predict(padded, verbose=0)[0][0])
    pos    = score >= 0.5
    conf   = score if pos else 1 - score
    return {
        "sentiment":  "positive" if pos else "negative",
        "confidence": round(conf, 4),
        "score":      round(score, 4),
        "stars":      round(1 + conf * 4)
    }

app = FastAPI(title="Sentiment API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ReviewIn(BaseModel):
    review: str

@app.get("/")
def root(): return {"status": "ok"}

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/predict")
def predict_route(req: ReviewIn):
    if len(req.review.strip()) < 5:
        raise HTTPException(400, "Review too short")
    t0 = time.perf_counter()
    r  = predict(req.review)
    r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return r
