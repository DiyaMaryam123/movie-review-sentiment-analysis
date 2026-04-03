# Run this ONCE to convert your model
# Command: python3 convert_model.py

import os, sys

print("Fixing Keras version conflict...")
os.system(f"{sys.executable} -m pip install 'keras==2.13.1' --quiet")
os.system(f"{sys.executable} -m pip install 'tf-keras==2.13.1' --quiet 2>/dev/null || true")

# Force use of tf.keras not standalone keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

print("Loading model with legacy keras...")
import tf_keras as keras_legacy

try:
    import tf_keras
    model = tf_keras.models.load_model(str(BASE / "sentiment_model(1).keras"), compile=False)
    print("Loaded with tf_keras!")
except Exception as e1:
    print(f"tf_keras failed: {e1}")
    try:
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        import tensorflow as tf
        model = tf.keras.models.load_model(str(BASE / "sentiment_model(1).keras"), compile=False)
        print("Loaded with tf.keras legacy!")
    except Exception as e2:
        print(f"Also failed: {e2}")
        print("\nTrying numpy workaround...")
        # Last resort — rebuild model from scratch and copy weights
        import pickle, json
        tokenizer = pickle.load(open(BASE / "tokenizer(1).pkl", "rb"))
        config    = json.load(open(BASE / "model_config(1).json"))
        MAX_LEN   = config["max_len"]
        MAX_WORDS = config["max_words"]

        import tensorflow as tf
        new_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Try to load weights only
        try:
            new_model.load_weights(str(BASE / "sentiment_model(1).keras"))
            model = new_model
            print("Loaded weights only!")
        except Exception as e3:
            print(f"Weight load failed too: {e3}")
            print("\nCannot convert. Please re-run Colab training.")
            sys.exit(1)

print("Saving as .h5...")
model.save(str(BASE / "sentiment_model.h5"))
print("\nDone! sentiment_model.h5 saved.")
print("Now run: uvicorn app:app --reload --port 8000")
