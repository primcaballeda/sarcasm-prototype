# Sarcasm Detection Setup Guide (Python-only)

This guide helps you run the Python backend and the Streamlit UI using your model.

## Project Structure

```
sarcasm-prototype/
├── backend/                    # Python backend + Streamlit UI
│   ├── app.py                 # Flask API (optional)
│   ├── streamlit_app.py       # Streamlit UI (recommended)
│   ├── requirements.txt       # Python dependencies
│   ├── tokenizer/             # Tokenizer files
│   └── model/                 # Model directory
│       └── sarcasm_model.pt   # YOUR MODEL FILE (copy here)
└── SETUP.md
```

## Step 1: Set Up the Backend (Python)

### 1.1 Navigate to backend directory

```powershell
cd "c:\Coding stuff\sarcasm-prototype\backend"
```

### 1.2 Create a virtual environment

```powershell
python -m venv venv
```

### 1.3 Activate the virtual environment

```powershell
.\venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

### 1.4 Install Python dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- Flask-CORS (for handling cross-origin requests)
- PyTorch (for running the model)
- Transformers (for BERT tokenizer)
- NumPy (for numerical operations)

### 1.5 Copy your model file

Copy your trained PyTorch model file (`.pt` extension) to the `backend/model/` directory and rename it to `sarcasm_model.pt`:

```powershell
# Example - replace with your actual model file path
Copy-Item "path\to\your\model.pt" -Destination "model\sarcasm_model.pt"
```

### 1.6 Start the backend server (Flask API, optional)

```powershell
python app.py
```

You should see:
```
Model loaded successfully on cpu (or cuda)
* Running on http://127.0.0.1:5000
```

**Keep this terminal window open** - the server needs to keep running.

### 1.7 Test the backend (optional)

Open a new terminal and test the API:

```powershell
# Test health endpoint
Invoke-WebRequest -Uri "http://localhost:5000/api/health" -Method GET

# Test prediction endpoint
$body = @{ text = "Oh great, another meeting!" } | ConvertTo-Json
Invoke-WebRequest -Uri "http://localhost:5000/api/predict" -Method POST -Body $body -ContentType "application/json"
```

## Step 2: Set Up the Frontend (React)

The React frontend has been removed. Use the Streamlit UI instead.

### 2.1 Start the Streamlit UI (recommended)

From `backend/` (with your venv activated):

```powershell
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

## Step 3: Using the Application

### Single Text Analysis

1. Type or paste text into the input box
2. Click "Analyze Sarcasm"
3. View results from both models:
   - **Baseline**: GloVe+CNN+BiLSTM+Attention (static detection)
   - **Proposed**: BERT+CNN+BiLSTM+MHA (your PyTorch model)

### Batch Dataset Analysis

1. Click "Choose File" in the "Dataset Analysis" section
2. Upload a CSV or JSON file with text data
3. Click "Process Dataset with Both Models"
4. View the comparative results

## Troubleshooting

### Backend Issues

**Error: "Model not loaded"**
- Make sure your model file is at `backend/model/sarcasm_model.pt`
- Check that the model architecture in `app.py` matches your trained model
- Check the terminal for error messages

**Error: "Module not found"**
- Make sure the virtual environment is activated
- Run `pip install -r requirements.txt` again

**Port 5000 already in use**
- Change the port in `app.py` (last line): `app.run(debug=True, port=5001)`
- Update the API URL in `SarcasmDetector.jsx` to match

### Frontend Issues

N/A (React frontend removed).

## Model Architecture Notes

The backend expects a model with this architecture:
- BERT base model (768 hidden dimensions)
- CNN layers (Conv1d with kernel_size=3)
- Bidirectional LSTM (hidden_size=256)
- Multi-head attention (8 heads)
- Fully connected layers for classification

If your model architecture is different, you'll need to modify the `SarcasmDetector` class in `app.py` to match your model's structure.

## Production Deployment

For production deployment, consider:
1. Deploying the Streamlit app (simplest) or the Flask API behind Gunicorn
2. Providing the model weights via Git LFS or `SARCASM_PROPOSED_MODEL_URL` (see `backend/README.md`)
3. Adding rate limiting/auth if exposing the API publicly

## Support

If you encounter issues:
1. Check the backend terminal for Python errors
2. Check the browser console for JavaScript errors
3. Verify the model file format and architecture
4. Ensure all dependencies are correctly installed
