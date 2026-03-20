from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer, BertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
import os
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Device configuration for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}")

# ============================================================================
# TOKENIZERS
# ============================================================================

# Load BERT tokenizer for Proposed Model
bert_tokenizer_path = './tokenizer'
try:
    if os.path.exists(bert_tokenizer_path) and os.path.isdir(bert_tokenizer_path):
        bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        print("BERT tokenizer loaded from local directory")
    else:
        raise FileNotFoundError("Local tokenizer directory not found")
except Exception as e:
    print(f"Could not load local BERT tokenizer: {e}")
    print("Downloading BERT tokenizer from HuggingFace...")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("BERT tokenizer loaded from HuggingFace")

# Load Keras tokenizer for Baseline Model
baseline_tokenizer = None
max_len = 50  # Must match the training parameter
try:
    baseline_tokenizer_path = './tokenizer/baseline_tokenizer.pkl'
    with open(baseline_tokenizer_path, 'rb') as f:
        baseline_tokenizer = pickle.load(f)
    print(" Baseline (Keras) tokenizer loaded")
except Exception as e:
    print(f" Could not load baseline tokenizer: {e}")
    print("   Please copy 'baseline_tokenizer.pkl' to backend/tokenizer/ directory")
    baseline_tokenizer = None

# ============================================================================
# PROPOSED MODEL (PyTorch - BERT + CNN + BiLSTM + Multi-Head Attention)
# ============================================================================

class SarcasmDetectorProposed(nn.Module):
    """Proposed Model: BERT + CNN + BiLSTM + Multi-Head Attention"""
    def __init__(self, bert_model='bert-base-uncased', hidden_size=128, num_classes=2):
        super(SarcasmDetectorProposed, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        bert_hidden = 768
        
        # CNN layers (matching saved model)
        self.conv1 = nn.Conv1d(in_channels=bert_hidden, out_channels=32, kernel_size=5, padding=2)
        self.dropout_conv = nn.Dropout(0.3)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(32, hidden_size, bidirectional=True, batch_first=True, dropout=0.3)
        
        # Multi-head attention
        lstm_out_size = hidden_size * 2
        self.mha = nn.MultiheadAttention(embed_dim=lstm_out_size, num_heads=8, dropout=0.3, batch_first=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_out_size)
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_out_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state  # (batch_size, seq_len, 768)
        
        # CNN layers
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len) for Conv1d
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # Back to (batch, seq_len, channels)
        x = self.dropout_conv(x)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # Multi-head attention
        mha_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        x = self.layer_norm(lstm_out + mha_out)  # Residual connection
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        logits = self.output(x)
        
        return logits


# Load Proposed Model (PyTorch)
proposed_model = None
try:
    proposed_model = SarcasmDetectorProposed()
    proposed_model.load_state_dict(torch.load('./model/sarcasm_model.pt', map_location=device))
    proposed_model.to(device)
    proposed_model.eval()
    print(f"Proposed model (PyTorch) loaded successfully on {device}")
except Exception as e:
    print(f"Error loading proposed model: {e}")
    proposed_model = None

# ============================================================================
# BASELINE MODEL (TensorFlow/Keras - BiLSTM + Attention)
# ============================================================================

# Custom layer to replace Lambda (used in model_fixed.keras)
class SumLayer(keras.layers.Layer):
    """Custom layer that replaces Lambda(lambda x: tf.reduce_sum(x, axis=-2))"""
    def call(self, inputs, mask=None):
        # Accept mask parameter like Lambda does
        return tf.reduce_sum(inputs, axis=-2)
    
    def compute_mask(self, inputs, mask=None):
        # Lambda doesn't output a mask, so neither should we
        return None
    
    def get_config(self):
        return super().get_config()

# Load Baseline Model (Keras)
baseline_model = None
try:
    # Load model_fixed.keras - converted from your model.h5 with Lambda replaced by SumLayer
    print("Loading baseline model from model_fixed.keras...")
    baseline_model = keras.models.load_model(
        './model/model_fixed.keras',
        custom_objects={'SumLayer': SumLayer},
        compile=False,
        safe_mode=False
    )
    baseline_model.compile(optimizer='adam', loss='binary_crossentropy')
    print(f"Baseline model loaded successfully from model_fixed.keras")
    
except Exception as e:
    print(f" Failed to load model_fixed.keras: {e}")
    baseline_model = None

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_proposed(text):
    """Predict using Proposed model (PyTorch - BERT + CNN + BiLSTM + MHA)"""
    if proposed_model is None:
        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': 'Proposed model not loaded'
        }
    
    start_time = time.time()
    
    try:
        # Tokenize input with BERT tokenizer
        encoded = bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=50,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = proposed_model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)[0]
            probability = probabilities[1].item()  # Probability of sarcastic class
            prediction = 1 if probability > 0.5 else 0
            confidence = probability * 100 if prediction == 1 else (1 - probability) * 100
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'isSarcastic': bool(prediction == 1),
            'confidence': round(confidence, 2),
            'probabilities': {
                'not_sarcastic': round((1 - probability) * 100, 2),
                'sarcastic': round(probability * 100, 2)
            },
            'processingTime': f'{round(processing_time, 0)}ms',
            'model': 'proposed'
        }
    except Exception as e:
        print(f"Proposed model prediction error: {e}")
        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': str(e)
        }


def predict_baseline(text):
    """Predict using Baseline model (Keras - GloVe + CNN + BiLSTM + Attention)"""
    if baseline_model is None:
        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': 'Baseline model not loaded'
        }
    
    if baseline_tokenizer is None:
        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': 'Baseline tokenizer not loaded'
        }
    
    start_time = time.time()
    
    try:
        # Tokenize input with Keras tokenizer (same as training)
        sequence = baseline_tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        
        # Get prediction
        prediction = baseline_model.predict(padded_sequence, verbose=0)
        probability = float(prediction[0][0])
        
        is_sarcastic = probability > 0.5
        confidence = probability * 100 if is_sarcastic else (1 - probability) * 100
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'isSarcastic': bool(is_sarcastic),
            'confidence': round(confidence, 2),
            'probabilities': {
                'not_sarcastic': round((1 - probability) * 100, 2),
                'sarcastic': round(probability * 100, 2)
            },
            'processingTime': f'{round(processing_time, 0)}ms',
            'model': 'baseline'
        }
    except Exception as e:
        print(f"Baseline model prediction error: {e}")
        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': str(e)
        }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Default prediction endpoint (uses proposed model by default)
    Expects JSON: {"text": "Your text here", "model": "proposed|baseline|both"}
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_choice = data.get('model', 'proposed')  # Default to proposed
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        if model_choice == 'both':
            return jsonify({
                'proposed': predict_proposed(text),
                'baseline': predict_baseline(text)
            })
        elif model_choice == 'baseline':
            return jsonify(predict_baseline(text))
        else:  # proposed or default
            return jsonify(predict_proposed(text))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/proposed', methods=['POST'])
def predict_proposed_endpoint():
    """Predict using Proposed model only"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        result = predict_proposed(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/baseline', methods=['POST'])
def predict_baseline_endpoint():
    """Predict using Baseline model only"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        result = predict_baseline(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/compare', methods=['POST'])
def predict_compare():
    """Compare predictions from both models"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        proposed_result = predict_proposed(text)
        baseline_result = predict_baseline(text)
        
        return jsonify({
            'text': text,
            'proposed': proposed_result,
            'baseline': baseline_result,
            'agreement': proposed_result.get('isSarcastic') == baseline_result.get('isSarcastic')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    Expects JSON: {"texts": ["text1", "text2", ...], "model": "proposed|baseline"}
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        model_choice = data.get('model', 'proposed')
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        results = []
        predict_fn = predict_proposed if model_choice == 'proposed' else predict_baseline
        
        for text in texts:
            if text and text.strip():
                result = predict_fn(text)
                results.append(result)
            else:
                results.append({'error': 'Empty text'})
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Check if the API is running and which models are loaded"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'proposed': {
                'loaded': proposed_model is not None,
                'type': 'PyTorch (BERT + CNN + BiLSTM + MHA)',
                'device': str(device)
            },
            'baseline': {
                'loaded': baseline_model is not None,
                'type': 'TensorFlow/Keras (BiLSTM + Attention)'
            }
        }
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get both baseline and proposed model performance metrics"""
    import json
    try:
        baseline_metrics = None
        proposed_metrics = None
        
        # Load baseline metrics
        baseline_path = './model/model_metrics.json'
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                baseline_metrics = json.load(f)
        
        # Load proposed metrics
        proposed_path = './model/proposed_model_metrics.json'
        if os.path.exists(proposed_path):
            with open(proposed_path, 'r') as f:
                proposed_metrics = json.load(f)
        
        if baseline_metrics is None and proposed_metrics is None:
            return jsonify({
                'status': 'not_found',
                'message': 'No metrics files found. Train the models first to generate metrics.'
            }), 404
        
        return jsonify({
            'status': 'success',
            'baseline': baseline_metrics,
            'proposed': proposed_metrics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 DUAL MODEL SARCASM DETECTION API")
    print("="*80)
    print(f" Proposed Model: {'Loaded' if proposed_model else 'Not Loaded'}")
    print(f" Baseline Model: {'Loaded' if baseline_model else 'Not Loaded'}")
    print("\nAvailable Endpoints:")
    print("  POST /api/predict              - Use proposed model (or specify 'model' param)")
    print("  POST /api/predict/proposed     - Use proposed model only")
    print("  POST /api/predict/baseline     - Use baseline model only")
    print("  POST /api/predict/compare      - Compare both models")
    print("  POST /api/predict_batch        - Batch predictions")
    print("  GET  /api/health               - Health check")
    print("  GET  /api/metrics              - Get baseline model performance metrics")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000)
