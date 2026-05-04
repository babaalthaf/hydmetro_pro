# HydMetro Pro | Intelligence Dashboard

[![Deployment](https://img.shields.io/badge/Live-On%20Render-blue?style=for-the-badge&logo=render)](https://hydmetro-pro.onrender.com/)
[![Tech Stack](https://img.shields.io/badge/Python-Flask-black?style=for-the-badge&logo=python)](https://flask.palletsprojects.com/)

**HydMetro Pro** is a next-generation intelligence engine for the Hyderabad Metro Rail network. It combines real-time data engineering, satellite-synchronized scheduling, and neural load prediction to offer an unparalleled commuting experience.

## 🚀 Live Application
Access the production environment here: [https://hydmetro-pro.onrender.com/](https://hydmetro-pro.onrender.com/)

---

## 🚆 Dynamic Metro Network Map
The dashboard features a live SVG-based network topology map. 
- **Red Line**: Miyapur ↔ LB Nagar (27 Stations)
- **Blue Line**: Raidurg ↔ Nagole (21 Stations)
- **Green Line**: JBS ↔ MGBS (9 Stations)

### Interactive Flux Map
- **Live Train Tracking**: Real-time position simulation based on GTFS datasets.
- **Interchange Nodes**: Detailed guides for Ameerpet, MGBS, and Parade Ground.
- **IT Hub Intelligence**: Optimized routing for Hitech City, Madhapur, and Raidurg commuters.

---

## 📊 AI & Data Engineering (Jupyter Powered)
The system operates on a high-frequency dynamic dataset generated through robust simulation logic.

### Neural Load Prediction
The engine calculates "Atmosphere" and "Neural Load" using:
- **Peak Hour Analysis**: (7-10 AM, 5-9 PM)
- **Weather Influence**: Real-time temperature and rainfall metrics from Open-Meteo.
- **Ridership Matrices**: 500+ hourly data points simulating city flux.

### Jupyter Integration
The project includes automated CSV dataset generation (`final_metro_dataset.csv`) which serves as the training matrix for the weight-based prediction algorithm.

---

## 🛠️ Key Features
- **Path Architect**: Multi-line route planning with interchange instructions.
- **Station Intelligence Directory**: A comprehensive searchable index of the 57-station network with amenity logging and spatial filtering.
- **Enhanced Intel Overlays**: Deep-dive station statistics including satellite-synced first/last train timings and landmark proximity logic.
- **Digital Vault**: Secure ticket generation with QR codes and session-persistent history.
- **Neural Sentiment Hub**: Feedback system with local matrix persistence and cloud synchronization.
- **Mobile Flux UI**: Fully responsive bento-grid interface optimized for on-the-go tracking.

---

## 🏗️ Local Setup
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. Launch the engine: `python hydmetro_pro.py`
4. Access `http://localhost:3000`

---
*Developed for the HydMetro Network Matrix. Neural processing active.*
