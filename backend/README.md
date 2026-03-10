# Sarcasm Detection Backend

This backend server loads and runs the PyTorch sarcasm detection model.

## Setup

1. Install Python 3.8 or higher

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Place your model file:
   - Copy your trained model file (.pt) to `backend/model/sarcasm_model.pt`

6. Place tokenizer files:
   - Copy `tokenizer.json` and `tokenizer_config.json` to `backend/tokenizer/`

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /api/predict
Predict sarcasm for a single text.

Request:
```json
{
  "text": "Your text here"
}
```

Response:
```json
{
  "isSarcastic": true,
  "confidence": 89.5,
  "probabilities": {
    "not_sarcastic": 10.5,
    "sarcastic": 89.5
  },
  "processingTime": "187ms"
}
```

### POST /api/predict_batch
Predict sarcasm for multiple texts.

Request:
```json
{
  "texts": ["text1", "text2", "text3"]
}
```

### GET /api/health
Check if the API is running and model is loaded.
