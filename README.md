# 🚇 HydMetro Pro — AI Powered Hyderabad Metro Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-AI-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-brightgreen)
![Render](https://img.shields.io/badge/Deploy-Render-purple)

---

# 🌐 Live Application

## 🔗 https://hydmetro-pro.onrender.com/

---

# 📌 Project Overview

HydMetro Pro is an AI-powered smart metro transportation and analytics platform developed for the Hyderabad Metro Rail system.

The project combines:

- 🤖 Artificial Intelligence
- 📊 Machine Learning
- 🚉 GTFS Metro Simulation
- 🌦️ Weather Intelligence
- 🗺️ Smart Route Planning
- 📱 Responsive Web UI
- ⚡ Flask APIs
- ☁️ Cloud Deployment on Render

The application supports:

- Live train simulation
- AI crowd prediction
- Smart route planning
- Metro schedule intelligence
- Weather-based congestion analysis
- Nearest station detection
- Real-time metro visualization

---

# 🏙️ About Hyderabad Metro

Hyderabad Metro is one of India’s largest smart metro transportation systems connecting major commercial, residential, educational, and IT corridors across Hyderabad.

This project digitally simulates the Hyderabad Metro ecosystem using GTFS transportation data and AI-powered prediction systems.

---

# 🚇 Hyderabad Metro Network

The project models the Hyderabad Metro network with:

- 🔴 Red Line
- 🔵 Blue Line
- 🟢 Green Line

---

# 📍 Metro Statistics

| Feature | Value |
|---|---|
| Metro Lines | 3 |
| Total Stations | 57+ |
| Major Interchanges | 3 |
| AI Models | 6 |
| APIs | 7+ |
| Deployment | Render Cloud |

---

# 📂 About metro.2.ipynb.txt

The file `metro.2.ipynb.txt` is the core Artificial Intelligence and Machine Learning notebook used in the HydMetro Pro project.

It contains:

- GTFS Metro Data Processing
- Data Cleaning & Feature Engineering
- Ridership Prediction Logic
- Crowd Forecasting System
- Metro Traffic Intelligence
- Model Training & Evaluation
- Flask Integration Logic
- Live Metro Simulation APIs
- Weather-based Prediction System
- Route Planning Intelligence

This notebook powers the complete AI backend of the Hyderabad Metro Smart Transportation Platform.

---

# 📊 Dataset Information

The dataset is generated using GTFS (General Transit Feed Specification) metro data and engineered features.

## Features Used

| Feature | Description |
|---|---|
| stop_name | Metro station |
| hour | Time of travel |
| day_of_week | Weekday indicator |
| is_peak | Peak hour flag |
| is_weekend | Weekend flag |
| is_it_hub | IT corridor indicator |
| temperature | Weather temperature |
| rainfall | Rainfall value |
| is_festival | Festival indicator |
| platform | Platform number |
| ridership | Target variable |

Dataset contains:
- 500+ training samples
- 12 engineered features
- Realistic metro simulation data

---

# 🤖 Machine Learning Models Used

The project trains and compares multiple ML models for ridership forecasting and congestion prediction.

## 📈 Model List

| Model | Type | Purpose |
|---|---|---|
| Linear Regression | Statistical | Baseline Prediction |
| Random Forest | Ensemble Bagging | Ridership Forecasting |
| XGBoost | Gradient Boosting | Peak Crowd Prediction |
| LightGBM | Gradient Boosting | Fast Prediction |
| Stacking Regressor | Meta Ensemble | Final Deployment Model |
| ANN (MLP) | Neural Network | Deep Learning Forecast |

---

# 🏆 Model Accuracy Results

## 📊 Final Performance Comparison

| Rank | Model | MAE | R² Score | Accuracy Level |
|---|---|---|---|---|
| 🥇 | Random Forest | 7.9747 | 0.9652 | Excellent |
| 🥈 | Stacking Regressor | 7.9747 | 0.9652 | Deployed |
| 🥉 | XGBoost | 7.9751 | 0.9652 | Excellent |
| 4 | LightGBM | 7.9759 | 0.9652 | Excellent |
| 5 | Linear Regression | 7.9909 | 0.9651 | Baseline |
| 6 | ANN (MLP) | 8.4852 | 0.9605 | Moderate |

---

# ⭐ Best Model Selected

## 🏆 Stacking Regressor

Reasons:
- Highest generalization capability
- Combines strengths of all models
- Lowest prediction error
- Stable cross-validation performance
- Best suited for metro ridership forecasting

### Final Metrics
- **R² Score:** 0.9652
- **MAE:** 7.9747

---

# 🧠 AI Features

## 🚦 Crowd Prediction
Predicts:
- High crowd
- Medium crowd
- Low crowd

Based on:
- Time
- Weather
- Peak hours
- Festivals
- IT hub traffic

---

## 🌦️ Weather Intelligence

Uses Open-Meteo API for:
- Temperature
- Humidity
- Visibility
- AQI estimation

---

## 🗺️ Smart Route Planner

Provides:
- Fastest route
- Interchange guidance
- Fare estimation
- Travel duration

---

## 🚉 Live Train Simulation

Includes:
- Live moving trains
- ETA system
- Active trip tracking
- Dynamic station updates

---

# 🌐 Flask Web Application

The backend is developed using Flask.

## 🔗 Major API Endpoints

| Endpoint | Function |
|---|---|
| `/` | Main dashboard |
| `/api/live-map` | Live metro map |
| `/api/predict` | Ridership prediction |
| `/api/schedule` | Train schedule |
| `/api/route` | Route planner |
| `/api/weather` | Weather information |
| `/api/feedback` | User feedback |

---

# 🛠️ Technology Stack

## Backend
- Python 3.11
- Flask

## AI/ML
- scikit-learn
- XGBoost
- LightGBM
- NumPy
- Pandas

## Frontend
- HTML5
- Tailwind CSS
- JavaScript
- SVG Metro Maps

## Deployment
- Render Cloud

---

# 📱 Application Features

## ✨ Interactive Metro Map
- Animated train movement
- Station highlighting
- Metro line visualization

---

## 📊 AI Dashboard
- Crowd analytics
- Weather dashboard
- Peak-hour analysis

---

## 🎟️ Smart Ticketing
- QR ticket simulation
- Journey saving

---

## 📍 Nearest Station Detection
- Uses latitude & longitude
- Walking distance estimation

---

## ☁️ Live Deployment

Application deployed on Render Cloud:
- 24/7 availability
- Cloud-hosted APIs
- Real-time predictions

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/hydmetro-pro.git
cd hydmetro-pro
