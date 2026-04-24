# HydMetro Pro | Intelligence Transit Dashboard

HydMetro Pro is a sophisticated, AI-enhanced transit management and visualization platform for the Hyderabad Metro Rail network. It combines real-time data engineering, satellite-mapped topology, and predictive analytics to provide commuters with a high-fidelity "Intelligence Engine" for their daily travels.

![HydMetro Preview](https://tse4.mm.bing.net/th/id/OIP.1NBA8PENs-qr3CIho0o5oAHaDt?pid=Api&P=0&h=180)

## 🚀 Key Features

- **Neural Flux Dashboard**: Real-time departure boards with high-frequency synchronization and satellite-synced clocks.
- **AI Load Prediction**: Predictive crowd density logic that analyzes peak hours, IT hub activity, and festivals to suggest optimal travel times.
- **Smart Path Architect**: Advanced BFS-based route planning that computes the most efficient path across Red, Blue, and Green lines.
- **Precise Fare Calculation**: Distance-aware fare engine utilizing the revised official Hyderabad Metro Rail fare matrix.
- **Dynamic Network Topology**: A clean, high-precision SVG map focusing on station nodes and inter-station connectivity.
- **Integrated Environment Metrics**: Live weather telemetry (temperature, humidity, visibility) fetched via the Open-Meteo API.
- **Interchange Intelligence**: Detailed step-by-step transfer guides for complex hubs like Ameerpet and MGBS.

## 🛠️ Tech Stack

- **Backend**: Python (Flask)
- **Frontend**: Tailwind CSS, Lucide Icons, Vanilla JavaScript
- **Data Logic**: Haversine distance formulas, BFS Pathfinding, Synthetic GTFS Data Generation
- **Mapping**: Scalable Vector Graphics (SVG)

## 📦 Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python hydmetro_pro.py
   ```
   The app will be available at `http://localhost:3000`.

## 🧠 Intelligence Logic

### Crowd Density (Neural Load)
The system calculates a "Load Score" based on:
- **Peak Hours**: 7-10 AM and 5-9 PM.
- **Station Type**: High impact for IT Hubs (Hitech City, Madhapur, Raidurg).
- **Temporal Factors**: Weekend adjustments and festival buffers.

### Fare Matrix
Fares are calculated using point-to-point Haversine distance mapped to the official zone chart:
- `Up to 2km`: ₹11
- `2 - 4km`: ₹17
- `4 - 6km`: ₹28
- ...up to `24km+`: ₹69

## 📄 License
This project is for educational and simulation purposes.
