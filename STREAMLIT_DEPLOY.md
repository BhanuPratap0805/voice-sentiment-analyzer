# Deploying Voice Sentiment Analyzer to Streamlit

This guide explains how to deploy your application to **Streamlit Community Cloud** (free).

## Prerequisites
- A GitHub account.
- This project pushed to a GitHub repository.

## Steps

### 1. Push Code to GitHub
Ensure your latest changes (including `streamlit_app.py` and `requirements.txt`) are committed and pushed.

```bash
git add .
git commit -m "Add Streamlit app"
git push
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **New app**.
3. Select your repository, branch (usually `main` or `master`), and the file path:
   - **Main file path**: `streamlit_app.py`
4. Click **Deploy!**

### 3. Configuration (Optional)
If you run into memory issues (Whisper + Emotion model is heavy), Streamlit Cloud might reboot the app. 
- The *Base* tier has resource limits (approx 1GB RAM). 
- **Whisper Base** model is ~150MB, **Emotion** model is ~260MB. It *should* fit, but if it crashes, try switching to `whisper.load_model("tiny")` in `streamlit_app.py`.

### 4. Local Testing
To run the app locally before deploying:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
