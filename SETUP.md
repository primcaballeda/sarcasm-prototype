# Sarcasm Detection Setup Guide

This guide will help you set up the full-stack sarcasm detection application with your PyTorch model.

## Project Structure

```
sarcasm-prototype/
├── backend/                    # Python Flask backend
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── tokenizer/             # Tokenizer files (already copied)
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── model/                 # Model directory
│       └── sarcasm_model.pt   # YOUR MODEL FILE (you need to copy this)
├── src/                       # React frontend
│   └── components/
│       └── SarcasmDetector.jsx
└── package.json               # Node.js dependencies
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

### 1.6 Start the backend server

```powershell
w
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

### 2.1 Open a NEW terminal window

Keep the backend running in the first terminal.

### 2.2 Navigate to the project root

```powershell
cd "c:\Coding stuff\sarcasm-prototype"
```

### 2.3 Install Node.js dependencies (if not already installed)

```powershell
npm install
```

### 2.4 Start the React development server

```powershell
npm start
```

The application will open in your browser at `http://localhost:3000`

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

**"Network error" or "API unavailable"**
- Make sure the backend server is running
- Check that the backend URL in `SarcasmDetector.jsx` matches your backend port
- Check browser console for detailed error messages

**CORS errors**
- The backend already has CORS enabled via Flask-CORS
- If issues persist, try restarting both servers

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
1. Using Gunicorn or uWSGI instead of Flask's development server
2. Setting up proper environment variables for API keys and configurations
3. Using a reverse proxy (nginx) to serve both frontend and backend
4. Deploying the frontend build to a CDN
5. Implementing rate limiting and authentication

### Custom Domain Setup for `sarcasm-detector.dev`

If you want the app live on your domain, split it into two parts:
1. Host the React frontend on Vercel, Netlify, or a similar static host.
2. Host the Flask backend separately on Render, Railway, Fly.io, or a small VPS.

Recommended setup:
1. Point `sarcasm-detector.dev` to the frontend host.
2. Point `api.sarcasm-detector.dev` to the backend host.
3. Set `REACT_APP_API_BASE_URL` in the frontend deployment to your backend URL, for example `https://api.sarcasm-detector.dev`.
4. Keep CORS enabled in Flask so the frontend can call the API across subdomains.

DNS checklist:
1. Add the domain in your frontend host dashboard.
2. Add the DNS records your host provides. For Vercel, that usually means the apex A record plus a `www` CNAME.
3. Add a separate DNS record for `api` if your backend uses its own subdomain.

Important:
1. The frontend currently calls the API with a configurable base URL instead of `localhost`.
2. Do not deploy the frontend with `REACT_APP_API_BASE_URL` left blank unless the backend is served from the same origin.

## Support

If you encounter issues:
1. Check the backend terminal for Python errors
2. Check the browser console for JavaScript errors
3. Verify the model file format and architecture
4. Ensure all dependencies are correctly installed
