 HydMetro Pro: Neural Transit Intelligence Engine

[![Live System](https://img.shields.io/badge/Status-🟢%20Live-emerald)](https://hydmetro-pro.onrender.com/)
[![Stack](https://img.shields.io/badge/Stack-Python%20|%20Vite%20|%20SVG-blue)](https://hydmetro-pro.onrender.com/)

HydMetro Pro is an advanced urban mobility dashboard designed for the Hyderabad Metro Rail network. It combines real-time GTFS simulation with an AI-driven crowd prediction engine to offer commuters an "optimized flux" path across the city.

**Live Deployment:** [https://hydmetro-pro.onrender.com/](https://hydmetro-pro.onrender.com/)

---

## 🚇 Core Capabilities

### 🗺️ Dynamic Network Topology
- **SVG Flux Map:** A custom-engineered SVG map of all 57 stations across Red, Blue, and Green lines.
- **Live Train Tracking:** Real-time interpolation of train positions between nodes based on GTFS schedules.
- **Interchange Logic:** Precise transfer guides for Ameerpet (Red/Blue), MGBS (Red/Green), and Parade Ground (Blue/Green).

### 🧠 Neural Flux Processor (AI Load Prediction)
Our intelligence engine predicts station congestion levels using high-frequency simulated datasets.

### 💳 Path Architect & Digital Token
- **Fare Matrix:** Official fare calculations based on distance segments (₹11 to ₹69).
- **UPI Integration:** Deep-linked payment support for Google Pay, PhonePe, and Paytm.
- **Sentiment Hub:** Persistent feedback system using Browser Local Matrix (Local Storage).

---

## 🧪 Data Science & Logic Explanation (Jupyter Context)

In our "Neural Engine," we simulate years of metropolitan transit data to refine our prediction accuracy. Below is the logic used in our core processing:

### 1. The Synthetic Dataset Formula
We generate a `final_metro_dataset.csv` using a Gaussian noise-infused ridership formula:
```python
# Base Ridership = 50
# Peak Multiplier = +100 (7-10 AM & 5-9 PM)
# IT Hub Weight = +80 (Hitech City, Madhapur, Raidurg)
# Weather Influence = +25 (Rain induces transport shift to Metro)
# Gaussian Noise = Dynamic variance for real-world stochasticity
```

### 2. The Prediction Model (Logic Block)
The prediction logic utilizes a weight-based classification system:
- **RED (High Rush):** Score > 200. Peak hours in IT clusters.
- **YELLOW (Manageable):** Score 140-200. Off-peak in high-density nodes.
- **GREEN (Optimal):** Score < 100. Late-night/Early-morning transitions.

### 3. Coordinate Interpolation
Mapping Latitude/Longitude to SVG `(x, y)` space is handled via linear interpolation across the network bounds:
```python
x = minX + (lng - minLng) / (maxLng - minLng) * (maxX - minX)
y = minY + (maxLat - lat) / (maxLat - minLat) * (maxY - minY)
```

---

## 🛠️ Technical Stack
- **Backend:** Python 3.x / Flask
- **Frontend:** Tailwind CSS / Lucide Icons / SVG
- **Data:** GTFS (General Transit Feed Specification) Simulation
- **Storage:** Browser Local Matrix (Local Storage) for User Sentiment
- **Deployment:** Render / Cloud Run

---

## 🚀 Getting Started
1. Clone the repository.
2. Install dependencies: `pip install flask requests`.
3. Run the engine: `python hydmetro_pro.py`.
4. Access the dashboard at `http://localhost:3000`.

---
*Developed for the future of urban mobility in Hyderabad.*
