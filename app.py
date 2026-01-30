from flask import Flask, render_template, request, jsonify
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_loader import load_emotion_assets

app = Flask(__name__)

# --- Initialize AI Engine ---
# Load assets globally so they stay in memory for fast inference
MODEL, TOKENIZER, LABEL_ENCODER = load_emotion_assets()
MAX_LENGTH = 40

def clean_text(text):
    """
    Standardizes input text to match the preprocessing done during training.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- Page Routes ---

@app.route('/')
def home():
    """Render the main Analyzer page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the project information page."""
    return render_template('about.html')

@app.route('/workflow')
def workflow():
    """Render the technical pipeline explanation page."""
    return render_template('workflow.html')

# --- API Route for Predictions ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles AJAX requests from the frontend, processes text, 
    and returns emotion predictions.
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    data = request.get_json()
    user_text = data.get('text', '')
    
    if not user_text.strip():
        return jsonify({'error': 'No text provided'}), 400

    # 1. Preprocess
    cleaned_text = clean_text(user_text)
    
    # 2. Tokenize & Pad
    sequence = TOKENIZER.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    
    # 3. Model Inference
    prediction_scores = MODEL.predict(padded_sequence, verbose=0)
    class_index = np.argmax(prediction_scores)
    
    # 4. Decode Result
    emotion = LABEL_ENCODER.inverse_transform([class_index])[0]
    confidence = float(np.max(prediction_scores))

    return jsonify({
        'emotion': emotion,
        'confidence': f"{confidence*100:.2f}%"
    })

if __name__ == '__main__':
    # Local execution
    app.run(debug=True)