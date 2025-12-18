# 🚀 HSEF Web Application Deployment Guide

## Overview

This guide covers deploying the HSEF URL Classification System as a web application.

---

## 📋 Prerequisites

1. **Trained HSEF Model** - Run `python hsef_model.py` first
2. **Python 3.8+** installed
3. **All dependencies** installed

---

## 🔧 Quick Setup

### Step 1: Install Dependencies

```bash
pip install flask flask-cors
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 2: Ensure Model is Saved

The model is automatically saved during training. Check if these files exist:

```
models/
├── hsef_model.pkl
├── hsef_scaler.pkl
├── feature_names.json
└── model_info.json
```

If not, run the training script:
```bash
python hsef_model.py
```

### Step 3: Start the Web Server

```bash
python app.py
```

### Step 4: Access the Application

Open your browser and go to:
```
http://127.0.0.1:5000
```

---

## 🌐 Application Features

### 1. **Single URL Classification**
- Enter any URL
- Get instant classification
- See confidence scores
- View probability distribution for all classes

### 2. **Batch Classification**
- Upload CSV file with URLs
- Process multiple URLs at once
- Download results
- Supports large batches

### 3. **Visual Interface**
- Modern, responsive design
- Real-time predictions
- Interactive probability charts
- Mobile-friendly

---

## 📊 API Endpoints

### Health Check
```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-24T13:00:00"
}
```

### Single URL Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "url": "https://example.com/path"
}
```

**Response:**
```json
{
  "url": "https://example.com/path",
  "prediction": "benign",
  "confidence": 0.9845,
  "probabilities": {
    "benign": 0.9845,
    "phishing": 0.0102,
    "malware": 0.0031,
    "defacement": 0.0015,
    "spam": 0.0007
  },
  "timestamp": "2025-10-24T13:00:00"
}
```

### Batch Prediction
```bash
POST /api/predict_batch
Content-Type: multipart/form-data

file: urls.csv
```

**CSV Format Option 1** (URLs only):
```csv
url
https://example.com
https://test.com
```

**CSV Format Option 2** (Pre-extracted features):
```csv
urlLen,domainlength,pathLength,...
100,15,50,...
120,20,60,...
```

**Response:**
```json
{
  "total": 2,
  "results": [
    {
      "index": 0,
      "prediction": "benign",
      "confidence": 0.9845,
      "probabilities": {...}
    },
    {
      "index": 1,
      "prediction": "phishing",
      "confidence": 0.9523,
      "probabilities": {...}
    }
  ],
  "timestamp": "2025-10-24T13:00:00"
}
```

### Model Information
```bash
GET /api/model_info
```

**Response:**
```json
{
  "loaded": true,
  "info": {
    "model_type": "HSEF",
    "base_learners": ["Random Forest", "XGBoost", "SVM"],
    "meta_learner": "Logistic Regression",
    "n_features": 80,
    "n_classes": 5,
    "gpu_enabled": true
  },
  "classes": ["benign", "phishing", "malware", "defacement", "spam"],
  "n_features": 80
}
```

---

## 🔐 Production Deployment

### Using Gunicorn (Linux/Mac)

1. Install Gunicorn:
```bash
pip install gunicorn
```

2. Run with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Options:
- `-w 4`: 4 worker processes
- `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000
- `--timeout 120`: Increase timeout for batch processing

### Using Waitress (Windows)

1. Install Waitress:
```bash
pip install waitress
```

2. Create `serve.py`:
```python
from waitress import serve
from app import app

if __name__ == '__main__':
    print("Starting HSEF Web Application with Waitress...")
    print("Access at: http://127.0.0.1:5000")
    serve(app, host='0.0.0.0', port=5000, threads=4)
```

3. Run:
```bash
python serve.py
```

---

## 🐳 Docker Deployment

### Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ templates/
COPY models/ models/

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

### Build and Run:

```bash
# Build image
docker build -t hsef-webapp .

# Run container
docker run -p 5000:5000 hsef-webapp
```

---

## ☁️ Cloud Deployment

### Heroku

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Create `runtime.txt`:
```
python-3.9.18
```

3. Deploy:
```bash
heroku create hsef-url-classifier
git push heroku main
```

### AWS Elastic Beanstalk

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize:
```bash
eb init -p python-3.9 hsef-webapp
```

3. Create environment and deploy:
```bash
eb create hsef-env
eb deploy
```

### Azure App Service

1. Create web app:
```bash
az webapp up --name hsef-classifier --runtime "PYTHON:3.9"
```

2. Configure startup:
```bash
az webapp config set --startup-file "gunicorn --bind=0.0.0.0 --timeout 120 app:app"
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```bash
FLASK_APP=app.py
FLASK_ENV=production
MODEL_DIR=models
MAX_FILE_SIZE=16777216  # 16MB
PORT=5000
HOST=0.0.0.0
```

### Load in `app.py`:
```python
from dotenv import load_dotenv
import os

load_dotenv()

app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE', 16*1024*1024))
PORT = int(os.getenv('PORT', 5000))
HOST = os.getenv('HOST', '0.0.0.0')
```

---

## 🔒 Security Considerations

### 1. **HTTPS**
Use HTTPS in production:
```python
from flask_talisman import Talisman
Talisman(app, force_https=True)
```

### 2. **Rate Limiting**
```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["200 per day", "50 per hour"])

@app.route('/api/predict')
@limiter.limit("10 per minute")
def api_predict():
    ...
```

### 3. **CORS**
```python
from flask_cors import CORS

CORS(app, origins=["https://yourdomain.com"])
```

### 4. **Input Validation**
```python
from urllib.parse import urlparse

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
```

---

## 📈 Performance Optimization

### 1. **Caching**
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/predict')
@cache.memoize(timeout=300)
def api_predict():
    ...
```

### 2. **Async Processing**
For batch jobs:
```python
from celery import Celery

celery = Celery(app.name, broker='redis://localhost:6379')

@celery.task
def process_batch(file_data):
    # Process in background
    ...
```

### 3. **Model Loading**
Load model once at startup, not per request.

---

## 🐛 Troubleshooting

### Model Not Loading

**Issue:** "Model file not found"

**Solution:**
1. Run training: `python hsef_model.py`
2. Check `models/` directory exists
3. Verify files: `ls models/`

### Memory Issues

**Issue:** "Out of memory during batch processing"

**Solution:**
1. Reduce batch size
2. Process in chunks
3. Increase server RAM
4. Use streaming for large files

### Slow Predictions

**Issue:** Predictions take too long

**Solution:**
1. Use GPU if available
2. Enable model caching
3. Reduce feature extraction complexity
4. Use fast_mode in training

---

## 📊 Monitoring

### Application Logs
```python
import logging

logging.basicConfig(
    filename='hsef_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route('/api/predict')
def api_predict():
    logging.info(f"Prediction request: {request.json}")
    ...
```

### Metrics
Track:
- Request count
- Response time
- Prediction accuracy
- Error rate
- Resource usage

---

## 📱 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### API Testing
```bash
# Health check
curl http://localhost:5000/api/health

# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Model info
curl http://localhost:5000/api/model_info
```

---

## 📝 Maintenance

### Update Model
1. Retrain with new data
2. Save new model files
3. Restart application
4. Verify predictions

### Backup
```bash
# Backup model files
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz *.log
```

---

## 📞 Support

- **Documentation**: README.md, QUICKSTART.md
- **Issues**: Report bugs and issues on GitHub
- **Questions**: Contact the development team

---

## ✅ Deployment Checklist

- [ ] Model trained and saved
- [ ] Dependencies installed
- [ ] Web application tested locally
- [ ] API endpoints verified
- [ ] Security measures implemented
- [ ] Production server configured
- [ ] HTTPS enabled
- [ ] Monitoring setup
- [ ] Backup strategy in place
- [ ] Documentation updated

---

**Deployment Status:** ✅ Ready for Production

**Last Updated:** October 24, 2025
