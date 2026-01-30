import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Minimizes TensorFlow logs in console
from tensorflow.keras.models import load_model

def load_emotion_assets():
    """
    Loads the H5 model, the tokenizer, and the label encoder from the root directory.
    Returns: (model, tokenizer, label_encoder) or (None, None, None) on failure.
    """
    try:
        # Load the pre-trained Keras/TensorFlow model
        # Ensure the file 'emotion_model.h5' is in the root folder
        model = load_model('emotion_model.h5')
        
        # Load the Tokenizer used during training
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Load the Label Encoder to map indices back to emotion names
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        print("Model assets loaded successfully.")
        return model, tokenizer, label_encoder
    except Exception as e:
        print(f"Error loading model assets: {e}")
        return None, None, None