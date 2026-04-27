# Sarcasm Detector Prototype (Python)

Python backend + Streamlit UI for sarcasm detection.

This repo previously included a React frontend; all React/JavaScript code has been removed to keep the project Python-only.

## What to run

### Option A: Streamlit UI (recommended)

From the repo root:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

### Option B: Flask API (for programmatic access)

```powershell
cd backend
.\venv\Scripts\activate
python app.py
```

API runs on `http://127.0.0.1:5000`.

## More details

- Backend documentation: see `backend/README.md`
- Setup walkthrough: see `SETUP.md`


