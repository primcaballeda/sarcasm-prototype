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
import sys
import urllib.request
import tempfile
import shutil
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)
CORS(app)

# ============================================================================
# BASE DIRECTORY
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROPOSED_MODEL_LOAD_ERROR = None
BASELINE_MODEL_LOAD_ERROR = None
BASELINE_TOKENIZER_LOAD_ERROR = None
BERT_TOKENIZER_SOURCE = None
PROPOSED_MODEL_PATH = None

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}")

# ============================================================================
# PREPROCESSING TOOLS
# ============================================================================


def preprocess_text(text):

    # convert to lowercase
    text = str(text).lower()

    # remove punctuation, numbers, special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # tokenize
    words = text.split()

    # join back into sentence
    text = " ".join(words)

    return text

# ============================================================================
# TOKENIZERS
# ============================================================================

bert_tokenizer_path = os.path.join(BASE_DIR, 'tokenizer')

try:

    if os.path.exists(bert_tokenizer_path) and os.path.isdir(bert_tokenizer_path):

        bert_tokenizer = BertTokenizer.from_pretrained(
            bert_tokenizer_path
        )

        BERT_TOKENIZER_SOURCE = "local"

        print("BERT tokenizer loaded from local directory")

    else:
        raise FileNotFoundError("Local tokenizer directory not found")

except Exception as e:

    print(f"Could not load local BERT tokenizer: {e}")

    print("Downloading BERT tokenizer from HuggingFace...")

    bert_tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased'
    )

    BERT_TOKENIZER_SOURCE = "huggingface"

    print("BERT tokenizer loaded from HuggingFace")

baseline_tokenizer = None
max_len = 50

try:

    baseline_tokenizer_path = os.path.join(
        BASE_DIR,
        'tokenizer',
        'baseline_tokenizer.pkl'
    )

    with open(baseline_tokenizer_path, 'rb') as f:
        baseline_tokenizer = pickle.load(f)

    print("Baseline tokenizer loaded")

except Exception as e:

    print(f"Could not load baseline tokenizer: {e}")

    BASELINE_TOKENIZER_LOAD_ERROR = str(e)

    baseline_tokenizer = None

# ============================================================================
# PROPOSED MODEL
# ============================================================================

class SarcasmDetectorProposed(nn.Module):

    def __init__(
        self,
        bert_model='bert-base-uncased',
        hidden_size=128,
        num_classes=2
    ):

        super(SarcasmDetectorProposed, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model)

        for param in self.bert.parameters():
            param.requires_grad = False

        bert_hidden = 768

        self.conv1 = nn.Conv1d(
            in_channels=bert_hidden,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.dropout_conv = nn.Dropout(0.5)

        self.bilstm = nn.LSTM(
            32,
            hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )

        lstm_out_size = hidden_size * 2

        self.mha = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=2,
            dropout=0.5,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(lstm_out_size)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(lstm_out_size, 256)

        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)

        self.output = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = bert_output.last_hidden_state

        x = x.permute(0, 2, 1)

        x = self.relu(self.conv1(x))

        x = x.permute(0, 2, 1)

        x = self.dropout_conv(x)

        lstm_out, _ = self.bilstm(x)

        mha_out, _ = self.mha(
            lstm_out,
            lstm_out,
            lstm_out
        )

        x = self.layer_norm(lstm_out + mha_out)

        x = x.mean(dim=1)

        x = self.dropout(x)

        x = self.relu(self.fc1(x))

        x = self.dropout1(x)

        x = self.relu(self.fc2(x))

        logits = self.output(x)

        return logits

# ============================================================================
# LOAD PROPOSED MODEL
# ============================================================================

proposed_model = None

try:

    proposed_model = SarcasmDetectorProposed()

    model_path = os.path.join(
        BASE_DIR,
        'model',
        'sarcasm_model.pt'
    )

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    proposed_model.load_state_dict(state_dict)

    proposed_model.to(device)

    proposed_model.eval()

    print(f"Proposed model loaded on {device}")

except Exception as e:

    print(f"Error loading proposed model: {e}")

    PROPOSED_MODEL_LOAD_ERROR = str(e)

    proposed_model = None

# ============================================================================
# BASELINE MODEL
# ============================================================================

class SumLayer(keras.layers.Layer):

    def call(self, inputs, mask=None):
        return tf.reduce_sum(inputs, axis=-2)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return super().get_config()

baseline_model = None

try:

    baseline_model = keras.models.load_model(
        os.path.join(BASE_DIR, 'model', 'baseline_model.keras'),
        custom_objects={'SumLayer': SumLayer},
        compile=False,
        safe_mode=False
    )

    baseline_model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    print("Baseline model loaded successfully")

except Exception as e:

    print(f"Failed to load baseline model: {e}")

    BASELINE_MODEL_LOAD_ERROR = str(e)

    baseline_model = None

# ============================================================================
# PROPOSED PREDICTION
# ============================================================================

def predict_proposed(text):

    if proposed_model is None:

        return {
            'isSarcastic': False,
            'confidence': 0.0,
            'error': 'Proposed model not loaded'
        }

    start_time = time.time()

    try:

        # ============================================================
        # APPLY PREPROCESSING
        # ============================================================

        text = preprocess_text(text)

        # ============================================================
        # BERT TOKENIZATION
        # ============================================================

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

        # ============================================================
        # MODEL PREDICTION
        # ============================================================

        with torch.no_grad():

            logits = proposed_model(
                input_ids,
                attention_mask
            )

            probabilities_tensor = torch.softmax(
                logits,
                dim=1
            )[0]

            prediction = torch.argmax(
                probabilities_tensor
            ).item()

            prob_not_sarcastic = probabilities_tensor[0].item() * 100

            prob_sarcastic = probabilities_tensor[1].item() * 100

            confidence = (
                prob_sarcastic
                if prediction == 1
                else prob_not_sarcastic
            )

        processing_time = (
            time.time() - start_time
        ) * 1000

        return {

            'isSarcastic': bool(prediction == 1),

            'confidence': round(confidence, 2),

            'probabilities': {

                'not_sarcastic': round(
                    prob_not_sarcastic,
                    2
                ),

                'sarcastic': round(
                    prob_sarcastic,
                    2
                )
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

# ============================================================================
# BASELINE PREDICTION
# ============================================================================

def predict_baseline(text):

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

        sequence = baseline_tokenizer.texts_to_sequences([text])

        padded_sequence = pad_sequences(
            sequence,
            maxlen=max_len,
            padding='post'
        )

        prediction = baseline_model.predict(
            padded_sequence,
            verbose=0
        )

        probability = float(prediction[0][0])

        is_sarcastic = probability > 0.5

        confidence = (
            probability * 100
            if is_sarcastic
            else (1 - probability) * 100
        )

        processing_time = (
            time.time() - start_time
        ) * 1000

        return {

            'isSarcastic': bool(is_sarcastic),

            'confidence': round(confidence, 2),

            'probabilities': {

                'not_sarcastic': round(
                    (1 - probability) * 100,
                    2
                ),

                'sarcastic': round(
                    probability * 100,
                    2
                )
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