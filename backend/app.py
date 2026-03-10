from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer, BertModel
import numpy as np
import time
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Device configuration for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}")

# Load tokenizer (shared by both models)
tokenizer_path = './tokenizer'
try:
    if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print("✓ Tokenizer loaded from local directory")
    else:
        raise FileNotFoundError("Local tokenizer directory not found")
except Exception as e:
    print(f"Could not load local tokenizer: {e}")
    print("Downloading tokenizer from HuggingFace...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded from HuggingFace")

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
    print(f"✓ Proposed model (PyTorch) loaded successfully on {device}")
except Exception as e:
    print(f"❌ Error loading proposed model: {e}")
    proposed_model = None

# ============================================================================
# BASELINE MODEL (TensorFlow/Keras - BiLSTM + Attention)
# ============================================================================

class SumLayer(keras.layers.Layer):
    """Custom layer to replace Lambda"""
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

# Load Baseline Model (Keras)
baseline_model = None
try:
    custom_objects = {
        'tf': tf,
        'K': tf.keras.backend,
        'SumLayer': SumLayer,
    }
    
    baseline_model = keras.models.load_model(
        './model/model_fixed.keras', 
        custom_objects=custom_objects,
        safe_mode=False
    )
    print(f"✓ Baseline model (Keras) loaded successfully")
except Exception as e:
    print(f"❌ Error loading baseline model: {e}")
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
        # Tokenize input
        encoded = tokenizer.encode_plus(
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
    """Predict using Baseline model (Keras - BiLSTM + Attention)"""
    if baseline_model is None:
        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': 'Baseline model not loaded'
        }
    
    start_time = time.time()
    
    try:
        # Tokenize input
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=50,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        
        input_ids = encoded['input_ids']
        
        # Get prediction
        prediction = baseline_model.predict(input_ids, verbose=0)
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


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 DUAL MODEL SARCASM DETECTION API")
    print("="*80)
    print(f"✓ Proposed Model: {'Loaded' if proposed_model else 'Not Loaded'}")
    print(f"✓ Baseline Model: {'Loaded' if baseline_model else 'Not Loaded'}")
    print("\nAvailable Endpoints:")
    print("  POST /api/predict              - Use proposed model (or specify 'model' param)")
    print("  POST /api/predict/proposed     - Use proposed model only")
    print("  POST /api/predict/baseline     - Use baseline model only")
    print("  POST /api/predict/compare      - Compare both models")
    print("  POST /api/predict_batch        - Batch predictions")
    print("  GET  /api/health               - Health check")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000)
