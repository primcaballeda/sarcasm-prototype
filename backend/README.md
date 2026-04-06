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

  Note: this repo intentionally ignores `backend/model/*.pt` (see `backend/.gitignore`) because the file is very large.
  For local runs, that's fine. For Streamlit Cloud deployments, you must either:
  - Use Git LFS to track the `.pt` file, OR
  - Host the `.pt` somewhere and provide a direct-download URL via `SARCASM_PROPOSED_MODEL_URL`.

6. Place tokenizer files:
   - Copy `tokenizer.json` and `tokenizer_config.json` to `backend/tokenizer/`

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## Running The Streamlit App (Python-Only UI)

Use this when you want the full UI in Python instead of React.

```bash
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

If that command is not found, make sure you activated your virtual environment first.

The Streamlit app provides the same core workflows as the React version:
- Single text sarcasm detection with baseline and proposed model comparison
- Example text shortcuts
- Dataset upload (CSV/JSON) and batch processing
- Dataset metrics and confusion matrices
- Model performance comparison view

When running Streamlit, Flask does not need to be started separately because predictions are called directly from Python.

## Streamlit Cloud Deployment Notes

- If the **Proposed model** does not appear, it's almost always because `backend/model/sarcasm_model.pt` is missing in the deployed environment.
- Recommended fix (no Git LFS): upload the file to a host that provides a direct-download URL (e.g., a private file host) and set:
  - `SARCASM_PROPOSED_MODEL_URL` = the URL to `sarcasm_model.pt`
  - Optional: `SARCASM_MODEL_CACHE_DIR` = where to cache the download (defaults to `~/.cache/sarcasm-prototype`)
- The app will download the weights on first start, then reuse the cached file on subsequent restarts.

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
