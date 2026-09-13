---
title: Indian Traffic Sign Detector
emoji: 🚦
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Indian Traffic Sign Detector

A custom HTML traffic sign detection app powered by Flask and TensorFlow. Upload one or more road sign images, run predictions, review top-5 scores, and display bilingual assist labels for demos.

## Stack

- Flask backend
- Custom HTML, CSS, and JavaScript frontend
- TensorFlow Keras `.h5` model
- Pillow-based image preprocessing
- Gunicorn for production serving

## Features

- Custom responsive frontend, no Streamlit
- Multi-image upload in one request
- Confidence-aware prediction states
- Top-5 predictions with progress bars
- Session-based recent prediction history
- Hindi assist labels
- Safety tips and sign-family insights

## Project files

- `app.py` - Flask application and inference logic
- `templates/index.html` - server-rendered frontend
- `static/style.css` - custom UI styling
- `static/app.js` - lightweight frontend interactions
- `class_labels.json` - label mapping for model outputs
- `indian_traffic_sign_model.h5` - trained sign detection model

## Run locally

Use Python 3.10 for the smoothest TensorFlow setup.

### Windows PowerShell

```powershell
cd C:\Users\pakal\Documents\IndianTrafficSignProject
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Deploy to Hugging Face Spaces

Hugging Face currently supports Docker Spaces for fully custom web apps like this one.

### 1. Create a Docker Space

Create a new Space and choose:

- SDK: `Docker`
- Visibility: your choice

### 2. Upload project files

Upload or push:

- `app.py`
- `templates/`
- `static/`
- `requirements.txt`
- `Dockerfile`
- `class_labels.json`
- `indian_traffic_sign_model.h5`
- `README.md`

### 3. Let the Space build

The Docker container will:

- install Python dependencies
- start Gunicorn
- expose the app on port `7860`

## Docker run locally

```powershell
docker build -t indian-traffic-sign .
docker run -p 7860:7860 indian-traffic-sign
```

Then open:

```text
http://localhost:7860
```

## Notes

- The local prediction history is stored in the browser session.
- The model file is large, so the first load can take time.
- If you later want webcam live capture instead of file upload with `capture`, we can add a JavaScript camera module next.
