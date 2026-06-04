# 🌍 Karachi AQI Predictor

A fully automated machine learning system that predicts Karachi's Air Quality Index (AQI) and PM2.5 levels up to **3 days ahead** using weather and air-quality data from Open-Meteo.

## 🚀 Live Demo

- Dashboard: [https://your-streamlit-app.streamlit.app](https://aqipredictorkarachi-4gwrl3ccmh5jcstmoj3uhe.streamlit.app/)
- API:[ https://your-huggingface-space.hf.space](https://sabihshasmi50-karachi-aqi-predictor.hf.space)

---

## 📌 Features

- Real-time AQI monitoring for Karachi
- 1-day, 2-day, and 3-day AQI forecasting
- Historical AQI trend analysis
- SHAP-based model explainability
- Automated feature engineering pipeline
- Hourly data synchronization
- Daily model retraining
- FastAPI REST API
- Interactive Streamlit dashboard
- Fully automated CI/CD with GitHub Actions

---

## 🏗 System Architecture

```text
Open-Meteo APIs
       │
       ▼
Feature Engineering
       │
       ▼
Hopsworks Feature Store
       │
       ▼
Model Training Pipeline
       │
       ▼
Hopsworks Model Registry
       │
       ▼
FastAPI Prediction Service
       │
       ▼
Streamlit Dashboard
```

---

## 🛠 Tech Stack

| Component | Technology |
|------------|------------|
| Data Source | Open-Meteo APIs |
| Feature Store | Hopsworks |
| Model Registry | Hopsworks |
| ML Models | Ridge, Random Forest, MLP |
| Explainability | SHAP |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| Automation | GitHub Actions |
| Deployment | Hugging Face Spaces, Streamlit Cloud |
| Language | Python 3.11 |

---

## 📂 Project Structure

```text
aqi_predictor_karachi/
│
├── app/
│   ├── api.py
│   └── dashboard.py
│
├── scripts/
│   ├── backfill_features.py
│   └── train_model.py
│
├── src/
│   ├── config.py
│   ├── hopsworks_utils.py
│   └── open_meteo.py
│
├── .github/workflows/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/aqi_predictor_karachi.git
cd aqi_predictor_karachi
```

### Create Virtual Environment

```bash
py -3.11 -m venv .venv311
.venv311\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT=your_project
FEATURE_GROUP_NAME=karachi_aqi_features
MODEL_NAME=karachi_aqi_model
```

---

## 📊 Model Pipeline

### Feature Pipeline

- Fetch AQI and weather data from Open-Meteo
- Generate lag features
- Generate rolling statistics
- Store features in Hopsworks

### Training Pipeline

For each forecast horizon:

- Day 1 prediction
- Day 2 prediction
- Day 3 prediction

Models trained:

- Ridge Regression
- Random Forest Regressor
- MLP Regressor

Best model selected using RMSE.

---

## 🔮 API Endpoints

| Endpoint | Description |
|-----------|-------------|
| `/` | Health Check |
| `/health` | Detailed Status |
| `/predict-latest` | Latest 3-Day Forecast |
| `/history` | Historical AQI Data |
| `/shap-importance` | Feature Importance |
| `/reload-model` | Refresh Model Cache |

Example:

```bash
GET /predict-latest
```

Response:

```json
{
  "predictions": [
    {
      "date": "2026-06-05",
      "aqi": 117
    }
  ]
}
```

---

## 📈 Features Used

### Air Quality Features

- PM2.5
- PM10
- Carbon Monoxide
- Nitrogen Dioxide
- Sulphur Dioxide
- Ozone
- Dust
- UV Index

### Weather Features

- Temperature
- Humidity
- Pressure
- Wind Speed
- Wind Direction
- Precipitation

### Engineered Features

- Lag Variables
- Rolling Means
- Day of Week
- Month
- Day of Year
- Weekend Indicator

---

## 🤖 Model Explainability

SHAP values are calculated for each forecast horizon, allowing users to understand:

- Which features influenced predictions
- Relative feature importance
- Forecast behavior across horizons

---

## 🔄 Automation

### Hourly

- Feature ingestion
- Feature store updates

### Daily

- Model training
- Model evaluation
- Registry updates

Implemented using GitHub Actions.

---

## 📦 Deployment

### API

Deployed using:

- Hugging Face Spaces
- Docker
- FastAPI

### Dashboard

Deployed using:

- Streamlit Cloud

---

## 📉 Sample Performance

| Horizon | RMSE | MAE | R² |
|----------|------|------|------|
| Day 1 | 8.9 | 6.5 | 0.41 |
| Day 2 | 9.7 | 7.2 | 0.35 |
| Day 3 | 10.4 | 7.8 | 0.29 |

*(Results may vary after retraining.)*

---

## 📍 Location

Karachi, Pakistan

Coordinates:

```text
24.8607° N
67.0011° E
```

---

## 👨‍💻 Author

**Sabih Shamsi**

Bachelor of Computer Science (BCS)

Machine Learning • Data Engineering • MLOps

---

