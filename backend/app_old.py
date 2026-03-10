from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer
import numpy as np
import time
import tensorflow.keras.backend as K

# Make tf and K available globally for Lambda layers
import sys
sys.modules['__main__'].tf = tf
sys.modules['__main__'].K = K

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load tokenizer
# Try loading from local directory first, fallback to downloading from HuggingFace
import os
tokenizer_path = './tokenizer'
try:
    if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print("Tokenizer loaded from local directory")
    else:
        raise FileNotFoundError("Local tokenizer directory not found")
except Exception as e:
    print(f"Could not load local tokenizer: {e}")
    print("Downloading tokenizer from HuggingFace...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Tokenizer loaded from HuggingFace")

# Custom layer to replace Lambda (for loading fixed model)
class SumLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

# Load the baseline Keras model
model = None
try:
    # Provide custom objects for Lambda layers that may reference tf
    import tensorflow.keras.backend as K
    
    # Add to globals for Lambda layer execution
    globals()['tf'] = tf
    globals()['K'] = K
    
    # Create a comprehensive custom_objects dict for Lambda layers
    custom_objects = {
        'tf': tf,
        'K': K,
        'SumLayer': SumLayer,
        'reduce_mean': tf.math.reduce_mean,
        'reduce_sum': tf.math.reduce_sum,
        'reduce_max': tf.math.reduce_max,
        'sqrt': tf.math.sqrt,
        'square': tf.math.square,
        'expand_dims': tf.expand_dims,
        'squeeze': tf.squeeze,
        'concat': tf.concat,
        'stack': tf.stack,
        'transpose': tf.transpose,
    }
    
    # Try loading fixed model first, then fallback to .keras, then .h5
    try:
        model = keras.models.load_model(
            './model/model_fixed.keras', 
            custom_objects=custom_objects,
            safe_mode=False
        )
        print(f"Baseline model loaded successfully from model_fixed.keras")
    except:
        try:
            model = keras.models.load_model(
                './model/model.keras', 
                custom_objects=custom_objects,
                safe_mode=False
            )
            print(f"Baseline model loaded successfully from model.keras")
        except:
            model = keras.models.load_model(
                './model/model.h5', 
                custom_objects=custom_objects,
                compile=False, 
                safe_mode=False
            )
            print(f"Baseline model loaded successfully from model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Model will return dummy predictions until the model file is loaded correctly")
    model = None

def predict_sarcasm(text):
    """
    Predict sarcasm for a given text using the loaded model
    """
    if model is None:
        # Return dummy prediction if model failed to load
        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': 'Model not loaded'
        }
    
    start_time = time.time()
    
    try:
        # Tokenize input
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=50,  # Model expects length 50
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'  # Changed to 'tf' for TensorFlow
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Get prediction from Keras model (only pass input_ids based on model architecture)
        prediction = model.predict(input_ids, verbose=0)
        
        # Extract probability (assuming binary classification output)
        probability = float(prediction[0][0])
        
        # Determine if sarcastic (threshold = 0.5)
        is_sarcastic = probability > 0.5
        confidence = probability * 100 if is_sarcastic else (1 - probability) * 100
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'isSarcastic': bool(is_sarcastic),
            'confidence': round(confidence, 2),
            'probabilities': {
                'not_sarcastic': round((1 - probability) * 100, 2),
                'sarcastic': round(probability * 100, 2)
            },
            'processingTime': f'{round(processing_time, 0)}ms'
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': str(e)
        }

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict sarcasm
    Expects JSON: {"text": "Your text here"}
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        result = predict_sarcasm(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    API endpoint to predict sarcasm for multiple texts
    Expects JSON: {"texts": ["text1", "text2", ...]}
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        results = []
        for text in texts:
            if text and text.strip():
                result = predict_sarcasm(text)
                results.append(result)
            else:
                results.append({'error': 'Empty text'})
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """
    Check if the API is running and model is loaded
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'backend': 'tensorflow/keras'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
