# Deployment Guide

This guide explains how to deploy the AI Delivery Availability Predictor to Streamlit Community Cloud and run it locally.

## Prerequisites
- GitHub repository with this code (you already have it)
- API key(s) for live weather (optional):
  - `OPENWEATHER_API_KEY` (primary)
  - `WEATHERAPI_KEY` (fallback)

## 1. Streamlit Community Cloud

1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Click "New app" and select your repo and branch (main).
3. App path: `streamlit_app.py`
4. Advanced settings → Secrets → paste TOML:
   ```toml
   OPENWEATHER_API_KEY = "your_openweather_key"
   WEATHERAPI_KEY = "your_weatherapi_key"
   WEATHER_CACHE_TTL = 600
   ```
5. Deploy. Streamlit will build and run your app.

Notes:
- Large files should be avoided in the repo. If you add big datasets/models, consider object storage (S3/GCS) and fetch at runtime.
- Do not set `server.port` in config on Streamlit Cloud.

## 2. Local Development

1. Python 3.12 recommended
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Add local secrets for live weather:
   - Create `.streamlit/secrets.toml` (do NOT commit) with:
     ```toml
     OPENWEATHER_API_KEY = "your_openweather_key"
     WEATHERAPI_KEY = "your_weatherapi_key"
     WEATHER_CACHE_TTL = 600
     ```
4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 3. Configuration (optional)
Create `.streamlit/config.toml` locally to pin server and theme:
```toml
[server]
headless = true
# port = 8501
# address = "0.0.0.0"

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#1f77b4"
font = "sans serif"
```

## 4. CI (optional)
A GitHub Actions workflow `.github/workflows/ci.yml` runs basic lint and compile checks on pushes/PRs to `main`.

## 5. Troubleshooting
- If you see `StreamlitSecretNotFoundError`, ensure secrets are set in Cloud or `.streamlit/secrets.toml` locally.
- If port conflicts arise locally, specify a different port:
  ```bash
  streamlit run streamlit_app.py --server.port 8510
  ```
- For faster reloading:
  ```bash
  pip install watchdog
  ```
