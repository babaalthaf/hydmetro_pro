import os
import csv
import json
import math
import random
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)


# ==========================================
# 1. ENHANCED DATA ENGINEERING & AI LOGIC
# ==========================================

def get_ist_now():
    """Returns current time in India Standard Time (UTC+5:30) with optional simulation offset."""
    base_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    # Allow simulation via global or query param if needed, for now just IST
    return base_now


def get_app_now():
    """Context-aware time for app logic."""
    if request and 'sim_hour' in request.args:
        try:
            h = int(request.args.get('sim_hour'))
            m = int(request.args.get('sim_min', 0))
            now = get_ist_now()
            return now.replace(hour=h, minute=m, second=0, microsecond=0)
        except:
            pass
    return get_ist_now()


def get_live_weather():
    """Fetches real-time weather from Open-Meteo with extra metrics."""
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=17.3850&longitude=78.4867&current_weather=true&hourly=relative_humidity_2m,visibility"
        data = requests.get(url, timeout=2).json()

        # Comprehensive WMO Code Mapping
        wmo_mapping = {
            0: "Sunny", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Foggy", 48: "Rime Fog", 51: "Light Drizzle", 53: "Moderate Drizzle",
            55: "Dense Drizzle", 61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
            71: "Slight Snow", 73: "Moderate Snow", 75: "Heavy Snow", 80: "Rain Showers",
            95: "Thunderstorm"
        }
        code = data['current_weather']['weathercode']
        condition = wmo_mapping.get(code, "Cloudy")

        # Get current hour index (IST)
        ist_now = get_ist_now()
        h_idx = ist_now.hour

        return {
            'temp': data['current_weather']['temperature'],
            'condition': condition,
            'humidity': data['hourly']['relative_humidity_2m'][h_idx] if h_idx < len(
                data['hourly']['relative_humidity_2m']) else 45,
            'visibility': (data['hourly']['visibility'][h_idx] / 1000) if h_idx < len(
                data['hourly']['visibility']) else 10,
            'aqi': random.randint(30, 70)
        }
    except:
        return {'temp': 30, 'condition': 'Clear Sky', 'humidity': 45, 'visibility': 10, 'aqi': 42}


def generate_ai_dataset():
    """Generates a high-frequency, dynamic GTFS simulation and mock ridership data without pandas/numpy."""
    stations_for_df = [s['name'] for s in STATIONS_LIST]
    sample_size = 500

    # Generate mock base data
    data = []
    base_time = get_ist_now() - timedelta(days=7)

    with open("final_metro_dataset.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['stop_name', 'arrival_time', 'platform', 'hour', 'day_of_week', 'is_peak', 'is_weekend', 'is_it_hub',
             'temperature', 'rainfall', 'is_festival', 'ridership'])

        holiday_dates = ["2026-01-14", "2026-10-20", "2026-11-08"]
        it_stations = ['Hitech City', 'Madhapur', 'Raidurg']

        for i in range(sample_size):
            arrival = base_time + timedelta(minutes=i * 20)
            stop_name = random.choice(stations_for_df)
            hour = arrival.hour
            dow = arrival.weekday()

            is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0
            is_weekend = 1 if dow >= 5 else 0
            is_it_hub = 1 if stop_name in it_stations else 0
            temp = 25 + (hour % 12)
            rainfall = random.choice([0, 0, 5, 10])
            is_festival = 1 if arrival.strftime('%Y-%m-%d') in holiday_dates else 0

            # Ridership Target Formula
            noise = random.gauss(0, 10)
            ridership = int(50 + (is_peak * 100) + (is_it_hub * 80) - (is_weekend * 20) + (is_festival * 120) - (
                        rainfall * 2) + noise)

            writer.writerow([
                stop_name,
                arrival.strftime('%Y-%m-%d %H:%M:%S'),
                random.choice(['1', '2']),
                hour,
                dow,
                is_peak,
                is_weekend,
                is_it_hub,
                temp,
                rainfall,
                is_festival,
                ridership
            ])
    return True


def predict_load_ai(station_name, hour, is_weekend=False, weather=None):
    """Predicts load using the logic from the trained dataset formula with optional weather influence."""
    is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0
    is_it_hub = 1 if station_name in ['Hitech City', 'Madhapur', 'Raidurg'] else 0
    is_festival = 0  # Default for live

    score = 50 + (is_peak * 100) + (is_it_hub * 80) + (is_festival * 120)
    if is_weekend: score -= 20

    # Weather influence: People flock to AC Metro during high heat or seek shelter during rain
    if weather:
        if "Rain" in weather.get('condition', ''):
            score += 25
        if weather.get('temp', 30) > 35:
            score += 15

    if score > 200: return "High", "🔴 High Rush"
    if score > 140: return "M-High", "🟡 Rush but manageable"
    if score > 100: return "Medium", "🟢 Seat will be there"
    return "Low", "🟢 Good to travel"


# ==========================================
# 2. FULL DATASET (57 STATIONS)
# ==========================================
STATIONS_LIST = [
    # RED LINE
    {'id': 'R1', 'name': 'Miyapur', 'line': 'Red', 'x': 100, 'y': 200, 'lat': 17.4968, 'lng': 78.3498,
     'amenities': ['Parking', 'Restrooms', 'Food Court', 'ATM']},
    {'id': 'R2', 'name': 'JNTU', 'line': 'Red', 'x': 160, 'y': 240, 'lat': 17.4912, 'lng': 78.3582},
    {'id': 'R3', 'name': 'KPHB', 'line': 'Red', 'x': 220, 'y': 280, 'lat': 17.4842, 'lng': 78.3888},
    {'id': 'R4', 'name': 'Kukatpally', 'line': 'Red', 'x': 280, 'y': 320, 'lat': 17.4854, 'lng': 78.3975},
    {'id': 'R5', 'name': 'Balanagar', 'line': 'Red', 'x': 340, 'y': 360, 'lat': 17.4764, 'lng': 78.4239},
    {'id': 'R6', 'name': 'Moosapet', 'line': 'Red', 'x': 400, 'y': 400, 'lat': 17.4721, 'lng': 78.4284},
    {'id': 'R7', 'name': 'Bharat Nagar', 'line': 'Red', 'x': 460, 'y': 440, 'lat': 17.4646, 'lng': 78.4357},
    {'id': 'R8', 'name': 'Erragadda', 'line': 'Red', 'x': 520, 'y': 480, 'lat': 17.4572, 'lng': 78.4412},
    {'id': 'R9', 'name': 'ESI Hospital', 'line': 'Red', 'x': 580, 'y': 520, 'lat': 17.4517, 'lng': 78.4457},
    {'id': 'R10', 'name': 'S.R. Nagar', 'line': 'Red', 'x': 640, 'y': 560, 'lat': 17.4442, 'lng': 78.4484},
    {'id': 'R11', 'name': 'Ameerpet', 'line': 'Red', 'x': 700, 'y': 600, 'lat': 17.4346, 'lng': 78.4484,
     'amenities': ['Interchange', 'Shopping', 'Food Court', 'Restrooms', 'ATM']},
    {'id': 'R12', 'name': 'Panjagutta', 'line': 'Red', 'x': 760, 'y': 640, 'lat': 17.4258, 'lng': 78.4522},
    {'id': 'R13', 'name': 'Irrum Manzil', 'line': 'Red', 'x': 820, 'y': 680, 'lat': 17.4184, 'lng': 78.4557},
    {'id': 'R14', 'name': 'Khairatabad', 'line': 'Red', 'x': 880, 'y': 720, 'lat': 17.4101, 'lng': 78.4611},
    {'id': 'R15', 'name': 'Lakdikapul', 'line': 'Red', 'x': 940, 'y': 760, 'lat': 17.4024, 'lng': 78.4657},
    {'id': 'R16', 'name': 'Assembly', 'line': 'Red', 'x': 1000, 'y': 800, 'lat': 17.3984, 'lng': 78.4723},
    {'id': 'R17', 'name': 'Nampally', 'line': 'Red', 'x': 1060, 'y': 840, 'lat': 17.3921, 'lng': 78.4757},
    {'id': 'R18', 'name': 'Gandhi Bhavan', 'line': 'Red', 'x': 1120, 'y': 880, 'lat': 17.3872, 'lng': 78.4784},
    {'id': 'R19', 'name': 'OMC', 'line': 'Red', 'x': 1180, 'y': 920, 'lat': 17.3824, 'lng': 78.4812},
    {'id': 'R20', 'name': 'MGBS', 'line': 'Red', 'x': 1240, 'y': 960, 'lat': 17.3788, 'lng': 78.4820,
     'amenities': ['Bus Station Link', 'Ticket Counters', 'Restrooms', 'ATM']},
    {'id': 'R21', 'name': 'Malakpet', 'line': 'Red', 'x': 1300, 'y': 1000, 'lat': 17.3746, 'lng': 78.4957},
    {'id': 'R22', 'name': 'New Market', 'line': 'Red', 'x': 1360, 'y': 1040, 'lat': 17.3712, 'lng': 78.5084},
    {'id': 'R23', 'name': 'Musarambagh', 'line': 'Red', 'x': 1420, 'y': 1080, 'lat': 17.3684, 'lng': 78.5212},
    {'id': 'R24', 'name': 'Dilsukhnagar', 'line': 'Red', 'x': 1480, 'y': 1120, 'lat': 17.3657, 'lng': 78.5357},
    {'id': 'R25', 'name': 'Chaitanyapuri', 'line': 'Red', 'x': 1540, 'y': 1160, 'lat': 17.3612, 'lng': 78.5484},
    {'id': 'R26', 'name': 'Victoria Memorial', 'line': 'Red', 'x': 1600, 'y': 1200, 'lat': 17.3557, 'lng': 78.5512},
    {'id': 'R27', 'name': 'LB Nagar', 'line': 'Red', 'x': 1660, 'y': 1240, 'lat': 17.3458, 'lng': 78.5524},

    # BLUE LINE
    {'id': 'B1', 'name': 'Raidurg', 'line': 'Blue', 'x': 100, 'y': 800, 'lat': 17.4429, 'lng': 78.3750},
    {'id': 'B2', 'name': 'Hitech City', 'line': 'Blue', 'x': 160, 'y': 760, 'lat': 17.4474, 'lng': 78.3762,
     'amenities': ['Parking', 'Restrooms', 'WIFI', 'ATM']},
    {'id': 'B3', 'name': 'Durgam Cheruvu', 'line': 'Blue', 'x': 220, 'y': 720, 'lat': 17.4398, 'lng': 78.3857},
    {'id': 'B4', 'name': 'Madhapur', 'line': 'Blue', 'x': 280, 'y': 680, 'lat': 17.4357, 'lng': 78.3984},
    {'id': 'B5', 'name': 'Jubilee Hills CP', 'line': 'Blue', 'x': 340, 'y': 640, 'lat': 17.4324, 'lng': 78.4112},
    {'id': 'B6', 'name': 'Road No 5', 'line': 'Blue', 'x': 400, 'y': 600, 'lat': 17.4284, 'lng': 78.4239},
    {'id': 'B7', 'name': 'Yousufguda', 'line': 'Blue', 'x': 460, 'y': 560, 'lat': 17.4246, 'lng': 78.4357},
    {'id': 'B8', 'name': 'Ameerpet', 'line': 'Blue', 'x': 700, 'y': 600, 'lat': 17.4346, 'lng': 78.4484,
     'name_alias': 'Ameerpet'},
    {'id': 'B9', 'name': 'Begumpet', 'line': 'Blue', 'x': 800, 'y': 560, 'lat': 17.4398, 'lng': 78.4612},
    {'id': 'B10', 'name': 'Prakash Nagar', 'line': 'Blue', 'x': 900, 'y': 520, 'lat': 17.4457, 'lng': 78.4724},
    {'id': 'B11', 'name': 'Rasoolpura', 'line': 'Blue', 'x': 1000, 'y': 480, 'lat': 17.4512, 'lng': 78.4851},
    {'id': 'B12', 'name': 'Paradise', 'line': 'Blue', 'x': 1100, 'y': 440, 'lat': 17.4568, 'lng': 78.4972},
    {'id': 'B13', 'name': 'Parade Ground', 'line': 'Blue', 'x': 1200, 'y': 400, 'lat': 17.4612, 'lng': 78.5084},
    {'id': 'B14', 'name': 'Sec-bad East', 'line': 'Blue', 'x': 1300, 'y': 360, 'lat': 17.4546, 'lng': 78.5212},
    {'id': 'B15', 'name': 'Mettuguda', 'line': 'Blue', 'x': 1400, 'y': 320, 'lat': 17.4484, 'lng': 78.5342},
    {'id': 'B16', 'name': 'Tarnaka', 'line': 'Blue', 'x': 1500, 'y': 280, 'lat': 17.4357, 'lng': 78.5472},
    {'id': 'B17', 'name': 'Habsiguda', 'line': 'Blue', 'x': 1600, 'y': 240, 'lat': 17.4212, 'lng': 78.5584},
    {'id': 'B18', 'name': 'NGRI', 'line': 'Blue', 'x': 1700, 'y': 200, 'lat': 17.4084, 'lng': 78.5684},
    {'id': 'B19', 'name': 'Stadium', 'line': 'Blue', 'x': 1800, 'y': 160, 'lat': 17.4021, 'lng': 78.5712},
    {'id': 'B20', 'name': 'Uppal', 'line': 'Blue', 'x': 1900, 'y': 120, 'lat': 17.3984, 'lng': 78.5684},
    {'id': 'B21', 'name': 'Nagole', 'line': 'Blue', 'x': 2000, 'y': 80, 'lat': 17.3941, 'lng': 78.5668},

    # GREEN LINE
    {'id': 'G1', 'name': 'JBS', 'line': 'Green', 'x': 900, 'y': 200, 'lat': 17.4439, 'lng': 78.4988},
    {'id': 'G2', 'name': 'Sec-bad West', 'line': 'Green', 'x': 950, 'y': 300, 'lat': 17.4482, 'lng': 78.5012},
    {'id': 'G3', 'name': 'Parade Ground', 'line': 'Green', 'x': 1200, 'y': 400, 'lat': 17.4612, 'lng': 78.5084,
     'name_alias': 'Parade Ground'},
    {'id': 'G4', 'name': 'Gandhi Hospital', 'line': 'Green', 'x': 1200, 'y': 500, 'lat': 17.4342, 'lng': 78.5112},
    {'id': 'G5', 'name': 'Musheerabad', 'line': 'Green', 'x': 1200, 'y': 600, 'lat': 17.4212, 'lng': 78.5157},
    {'id': 'G6', 'name': 'RTC X Roads', 'line': 'Green', 'x': 1200, 'y': 700, 'lat': 17.4084, 'lng': 78.5184},
    {'id': 'G7', 'name': 'Chikkadpally', 'line': 'Green', 'x': 1200, 'y': 800, 'lat': 17.3984, 'lng': 78.5212},
    {'id': 'G8', 'name': 'Narayanaguda', 'line': 'Green', 'x': 1220, 'y': 900, 'lat': 17.3884, 'lng': 78.5184},
    {'id': 'G9', 'name': 'MGBS', 'line': 'Green', 'x': 1240, 'y': 960, 'lat': 17.3788, 'lng': 78.4820,
     'name_alias': 'MGBS'}
]

CONNECTIONS = {
    'Red': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16',
            'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27'],
    'Blue': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16',
             'B17', 'B18', 'B19', 'B20', 'B21'],
    'Green': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9']
}

# ==========================================
# 3. CORE SYSTEM LOGIC
# ==========================================

GTFS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gtfs.csv')
_GTFS_CACHE = None  # Global cache for performance


def ensure_gtfs(force=False):
    """Generates a high-frequency, dynamic GTFS simulation."""
    global _GTFS_CACHE
    if force or not os.path.exists(GTFS_FILE):
        print(f"Generating optimized GTFS... target: {GTFS_FILE}")
        with open(GTFS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trip_id', 'station_id', 'arrival_time', 'platform', 'direction', 'final_stop', 'line'])

            now_base = get_ist_now()
            start_day = now_base.day

            for line, stations in CONNECTIONS.items():
                for dir_type in ['Forward', 'Backward']:
                    path = stations if dir_type == 'Forward' else stations[::-1]
                    final_name = next(s['name'] for s in STATIONS_LIST if s['id'] == path[-1])

                    # More realistic intervals: 6 mins peak, 12 mins off-peak
                    current_trip_time = now_base.replace(hour=4, minute=0, second=0, microsecond=0)
                    trip_idx = 0
                    while current_trip_time.day == start_day and current_trip_time.hour < 23:
                        h = current_trip_time.hour
                        interval = 6 if (7 <= h <= 10 or 17 <= h <= 21) else 12

                        trip_id = f"{line}_{dir_type}_{trip_idx}"
                        trip_start_time = current_trip_time

                        # Platform Logic: Red (1,2), Blue (3,4), Green (1,2 - separate area)
                        p_base = 1 if line == 'Red' or line == 'Green' else 3
                        platform = str(p_base if dir_type == 'Forward' else p_base + 1)

                        for sid in path:
                            writer.writerow(
                                [trip_id, sid, trip_start_time.strftime('%H:%M:%S'), platform, dir_type, final_name,
                                 line])
                            trip_start_time += timedelta(minutes=2)

                        current_trip_time += timedelta(minutes=interval)
                        trip_idx += 1
        print("GTFS Generation Complete.")
        _GTFS_CACHE = None  # Invalidate cache

    if _GTFS_CACHE is None:
        try:
            with open(GTFS_FILE, 'r') as f:
                _GTFS_CACHE = list(csv.DictReader(f))
        except:
            _GTFS_CACHE = []

    return _GTFS_CACHE


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


FEEDBACK_CLOUD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feedback_cloud.json')


def save_feedback_to_cloud(feedback_data):
    """Saves feedback to a server-side JSON file (simulating cloud storage)."""
    try:
        if os.path.exists(FEEDBACK_CLOUD_FILE):
            with open(FEEDBACK_CLOUD_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []

        history.append(feedback_data)

        # Limit cloud history to prevent massive file growth
        if len(history) > 1000:
            history = history[-1000:]

        with open(FEEDBACK_CLOUD_FILE, 'w') as f:
            json.dump(history, f)
        return True
    except Exception as e:
        print(f"Cloud Feed Sync Error: {e}")
        return False


# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, ALL_STATIONS=STATIONS_LIST)


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    # Enrich with server timestamp
    data['server_received'] = get_app_now().strftime('%Y-%m-%d %H:%M:%S')

    success = save_feedback_to_cloud(data)
    return jsonify({'status': 'success' if success else 'failed'})


@app.route('/api/nearest', methods=['POST'])
def api_nearest():
    data = request.json

    if 'station_id' in data:
        nearest = next(s for s in STATIONS_LIST if s['id'] == data['station_id'])
    else:
        lat, lng = data['lat'], data['lng']
        nearest = min(STATIONS_LIST, key=lambda s: haversine(lat, lng, s['lat'], s['lng']))

    name = nearest.get('name_alias', nearest['name'])
    matching_ids = [s['id'] for s in STATIONS_LIST if s.get('name_alias', s['name']) == name]

    trips = ensure_gtfs()
    now = get_app_now()
    now_str = now.strftime('%H:%M:%S')
    one_hour_later = now + timedelta(hours=1)
    oh_str = one_hour_later.strftime('%H:%M:%S')

    upcoming = []
    for row in trips:
        if row['station_id'] in matching_ids and now_str < row['arrival_time'] < oh_str:
            # Calculate ETA countdown
            try:
                ah, am, as_ = map(int, row['arrival_time'].split(':'))
                arrival_dt = now.replace(hour=ah, minute=am, second=as_, microsecond=0)
                diff = (arrival_dt - now).total_seconds()
                if diff < 0: continue

                m, s = divmod(int(diff), 60)
                row_copy = row.copy()
                row_copy['eta'] = f"{m:02d}:{s:02d}"
                upcoming.append(row_copy)
            except:
                continue

    # AI & Weather Data
    weather = get_live_weather()
    is_weekend = now.weekday() >= 5
    load_val, load_label = predict_load_ai(name, now.hour, is_weekend=is_weekend, weather=weather)

    # Optimized Global GTFS Stats (Trips active right now)
    trip_times = {}
    for row in trips:
        tid = row['trip_id']
        t_arr = row['arrival_time']
        if tid not in trip_times:
            trip_times[tid] = {'min': t_arr, 'max': t_arr}
        else:
            if t_arr < trip_times[tid]['min']: trip_times[tid]['min'] = t_arr
            if t_arr > trip_times[tid]['max']: trip_times[tid]['max'] = t_arr

    active_count = sum(1 for tid, times in trip_times.items() if times['min'] <= now_str <= times['max'])

    # Sort upcoming by original string comparison (before I:M p conversion)
    upcoming.sort(key=lambda x: x['arrival_time'])

    # Use 12-hour format for upcoming trains
    for t in upcoming:
        try:
            h, m, s = map(int, t['arrival_time'].split(':'))
            dt = now.replace(hour=h, minute=m, second=s)
            t['arrival_time'] = dt.strftime('%I:%M %p')
        except:
            pass

    # Sort upcoming by original string comparison (before I:M p conversion) or keep it consistent

    return jsonify({
        'station': nearest,
        'upcoming': upcoming,
        'load_val': load_val,
        'load_label': load_label,
        'active_trips': active_count,
        'weather': weather,
        'greeting': "Good Morning" if 5 <= now.hour < 12 else "Good Afternoon" if 12 <= now.hour < 17 else "Good Evening"
    })


@app.route('/api/plan', methods=['POST'])
def api_plan():
    data = request.json
    start_id, end_id = data['from'], data['to']
    now = get_app_now()

    # BFS find path
    queue = [(start_id, [start_id])]
    visited = {start_id}
    path = []
    if start_id == end_id:
        path = [start_id]
    else:
        while queue:
            curr, p = queue.pop(0)
            if curr == end_id:
                path = p
                break
            neighbors = []
            for line in CONNECTIONS.values():
                if curr in line:
                    idx = line.index(curr)
                    if idx > 0: neighbors.append(line[idx - 1])
                    if idx < len(line) - 1: neighbors.append(line[idx + 1])
            c_name = next((s['name'] for s in STATIONS_LIST if s['id'] == curr), None)
            if c_name:
                for s in STATIONS_LIST:
                    if s['name'] == c_name and s['id'] != curr: neighbors.append(s['id'])
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, p + [n]))

    sequence = [next(s for s in STATIONS_LIST if s['id'] == sid) for sid in path]
    duration = max(2, len(sequence) * 2) if sequence else 0
    total_stops = len(sequence) - 1 if len(sequence) > 1 else max(0, len(sequence))

    # Synchronize Arrival Time with GTFS schedule
    ensure_gtfs()
    gtfs_arrival_time = None
    gtfs_boarding_time = None
    chosen_trip_id = None

    # Track arrival time for every stop in the journey
    stop_arrival_times = {}

    try:
        now_str = now.strftime('%H:%M:%S')
        with open(GTFS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            trips = list(reader)

        # Find the next available trip at start station
        start_id = sequence[0]['id']
        end_id = sequence[-1]['id'] if sequence else None

        possible_trips = [t for t in trips if t['station_id'] == start_id and t['arrival_time'] > now_str]
        possible_trips.sort(key=lambda x: x['arrival_time'])

        for pt in possible_trips[:5]:  # Check first 5 upcoming
            trip_data = [t for t in trips if t['trip_id'] == pt['trip_id']]
            # Check if this trip eventually reaches the end_id
            destination_stop = next((t for t in trip_data if t['station_id'] == end_id), None)
            if destination_stop:
                # Basic check: destination must be after start
                if destination_stop['arrival_time'] > pt['arrival_time']:
                    gtfs_arrival_time = destination_stop['arrival_time']
                    gtfs_boarding_time = pt['arrival_time']
                    chosen_trip_id = pt['trip_id']

                    # Map all stop times for this trip
                    for td in trip_data:
                        stop_arrival_times[td['station_id']] = td['arrival_time']
                    break
    except Exception as e:
        print(f"GTFS Planner Sync Error: {e}")

    # Annotate sequence with reaching times
    for s in sequence:
        s['reaching_at_raw'] = stop_arrival_times.get(s['id'])
        if s['reaching_at_raw']:
            try:
                h, m, s_ = map(int, s['reaching_at_raw'].split(':'))
                s['reaching_at'] = now.replace(hour=h, minute=m, second=s_).strftime('%I:%M %p')
            except:
                s['reaching_at'] = s['reaching_at_raw']
        else:
            s['reaching_at'] = "--:--"

    # Calculate Total Distance for Precise Fare Prediction
    total_km = 0
    for i in range(len(sequence) - 1):
        s1 = sequence[i]
        s2 = sequence[i + 1]
        total_km += haversine(s1['lat'], s1['lng'], s2['lat'], s2['lng'])

    # Official Fare Logic from Image
    def get_fare_from_matrix(dist):
        if dist <= 2: return 11
        if dist <= 4: return 17
        if dist <= 6: return 28
        if dist <= 9: return 37
        if dist <= 12: return 47
        if dist <= 15: return 51
        if dist <= 18: return 56
        if dist <= 21: return 61
        if dist <= 24: return 65
        return 69

    calculated_fare = get_fare_from_matrix(total_km)

    # Calculate more accurate duration using GTFS if available
    if gtfs_boarding_time and gtfs_arrival_time:
        try:
            h1, m1, s1 = map(int, gtfs_boarding_time.split(':'))
            h2, m2, s2 = map(int, gtfs_arrival_time.split(':'))
            t1 = timedelta(hours=h1, minutes=m1, seconds=s1)
            t2 = timedelta(hours=h2, minutes=m2, seconds=s2)
            duration = int((t2 - t1).total_seconds() // 60)
        except:
            duration = len(sequence) * 2
    else:
        duration = len(sequence) * 2

    # AI Recommendation logic
    start_station_name = sequence[0]['name']
    weather = get_live_weather()
    is_weekend = now.weekday() >= 5
    load_val, _ = predict_load_ai(start_station_name, now.hour, is_weekend=is_weekend, weather=weather)

    # NEW: Numerical Load and Peak Intensity Math
    load_pct = 35  # Base
    if (7 <= now.hour <= 10 or 17 <= now.hour <= 21): load_pct += 45
    if start_station_name in ['Hitech City', 'Madhapur', 'Raidurg']: load_pct += 15
    load_pct = min(99.4, load_pct + random.uniform(-3, 3))

    peak_intensity = 0
    if (7 <= now.hour <= 10):
        # Morning peak ramps up from 7 to 9, drops at 10
        dist = abs(now.hour - 8.5)
        peak_intensity = 100 - (dist * 30)
    elif (17 <= now.hour <= 21):
        # Evening peak peak at 19:00
        dist = abs(now.hour - 19)
        peak_intensity = 100 - (dist * 20)
    peak_intensity = round(max(0, min(100, peak_intensity)), 1)

    # INTERCHANGE & GUIDE LOGIC
    guides = []
    # Pre-parse reaching times for connections
    reaching_times_map = {s['id']: stop_arrival_times.get(s['id']) for s in sequence}

    for i in range(len(path) - 1):
        s1 = next(s for s in STATIONS_LIST if s['id'] == path[i])
        s2 = next(s for s in STATIONS_LIST if s['id'] == path[i + 1])
        name1 = s1.get('name_alias', s1['name'])
        name2 = s2.get('name_alias', s2['name'])

        if name1 == name2 and s1['id'] != s2['id']:
            next_sid = path[i + 2] if i + 2 < len(path) else None
            platform = "?"
            guide = f"Transfer at {name1}"

            # Connection Analytics
            reaching_at_raw = reaching_times_map.get(s1['id'])
            connecting_trains = []
            reaching_at_display = "--:--"

            if reaching_at_raw:
                try:
                    rh, rm, rs = map(int, reaching_at_raw.split(':'))
                    reach_dt = now.replace(hour=rh, minute=rm, second=rs, microsecond=0)
                    reaching_at_display = reach_dt.strftime('%I:%M %p')

                    # Next 1 hour from reaching
                    reach_plus_hour = reach_dt + timedelta(hours=1)
                    rph_str = reach_plus_hour.strftime('%H:%M:%S')

                    # Find other lines at this station
                    other_ids = [s['id'] for s in STATIONS_LIST if
                                 (s.get('name_alias') == name1 or s['name'] == name1) and s['line'] != s1['line']]

                    for row in trips:
                        if row['station_id'] in other_ids and reaching_at_raw < row['arrival_time'] < rph_str:
                            t_copy = row.copy()
                            th, tm, ts = map(int, t_copy['arrival_time'].split(':'))
                            t_dt = now.replace(hour=th, minute=tm, second=ts)
                            t_copy['arrival_time_12'] = t_dt.strftime('%I:%M %p')
                            t_copy['wait_mins'] = int((t_dt - reach_dt).total_seconds() // 60)
                            connecting_trains.append(t_copy)
                            if len(connecting_trains) >= 3: break
                except:
                    pass

            if next_sid:
                next_s = next(s for s in STATIONS_LIST if s['id'] == next_sid)
                if name1 == 'Ameerpet':
                    if next_s['line'] == 'Red':
                        idx = int(next_s['id'].replace('R', ''))
                        platform = "1 (Towards LB Nagar)" if idx > 11 else "2 (Towards Miyapur)"
                        if s1[
                            'line'] == 'Blue': guide = "Exit Blue Line (Level 1). Take stairs/escalator DOWN to Red Line level. Follow signs for Platform 1/2."
                    elif next_s['line'] == 'Blue':
                        idx = int(next_s['id'].replace('B', ''))
                        platform = "3 (Towards Nagole)" if idx > 8 else "4 (Towards Raidurg)"
                        if s1[
                            'line'] == 'Red': guide = "Exit Red Line. Take stairs/escalator UP to Blue Line (Level 1). Follow signs for Platform 3/4."
                elif name1 == 'MGBS':
                    if next_s['line'] == 'Red':
                        idx = int(next_s['id'].replace('R', ''))
                        platform = "1 (Towards LB Nagar)" if idx > 20 else "2 (Towards Miyapur)"
                        if s1[
                            'line'] == 'Green': guide = "Exit Green Line platform, follow 'Red Line' signs. Take ESCALATOR UP to Red Line Level."
                    elif next_s['line'] == 'Green':
                        platform = "3 (Towards JBS Parade Grounds)"
                        if s1[
                            'line'] == 'Red': guide = "Exit Red Line platform, follow 'Green Line' signs. Take ESCALATOR DOWN to Green Line Level."
                elif name1 == 'Parade Ground':
                    if next_s['line'] == 'Blue':
                        idx = int(next_s['id'].replace('B', ''))
                        platform = "3 (Towards Nagole)" if idx > 13 else "4 (Towards Raidurg)"
                        if s1[
                            'line'] == 'Green': guide = "Exit Green Line platform, follow Blue Line transfer signs. Take stairs to Platform level."
                    elif next_s['line'] == 'Green':
                        idx = int(next_s['id'].replace('G', ''))
                        platform = "1 (Towards MGBS)" if idx > 3 else "2 (Towards JBS Parade Grounds)"
                        if s1[
                            'line'] == 'Blue': guide = "Exit Blue Line platform, follow Green Line transfer signs. Take ESCALATOR DOWN to reach Green Line Platform Level."

            guides.append({
                'station': name1,
                'platform': platform,
                'text': guide,
                'reaching_at': reaching_at_display,
                'connections': connecting_trains
            })

    # PROJECTION METRICS
    is_peak = "Peak Hour" if (7 <= now.hour <= 10 or 17 <= now.hour <= 21) else "Off-Peak"
    is_it_hub = "High" if any(
        n in [s['name'] for s in sequence] for n in ['Hitech City', 'Raidurg', 'Madhapur']) else "Normal"

    # Environmental Analytics
    co2_saved = round(total_km * 0.12, 2)  # kg CO2
    calories = int(total_km * 12 + len(guides) * 25)  # Estimated effort
    trees_saved = round(total_km * 0.05, 3)

    weather_advice = " AC cooling optimized for intense heat." if weather.get('temp', 30) > 35 else \
        " Rainy conditions detected; transit via tunnels recommended." if "Rain" in weather.get('condition', '') else \
            " Clear skies for a smooth commute." if "Sunny" in weather.get('condition', '') or "Clear" in weather.get(
                'condition', '') else ""

    recommendation = ("Optimal conditions. Low crowd density detected." if load_val == "Low" else \
                          "Fair volume. Seat likely available for your journey." if load_val == "Medium" else \
                              "Moderate volume. Manageable rush." if load_val == "M-High" else \
                                  "Peak congestion. AI suggests waiting for dip.") + weather_advice

    # User Request: Upcoming trains for next 1 hour from source
    one_hour_later = now + timedelta(hours=1)
    now_str = now.strftime('%H:%M:%S')
    oh_str = one_hour_later.strftime('%H:%M:%S')

    upcoming_hour = []
    for row in trips:
        if row['station_id'] == start_id and now_str < row['arrival_time'] < oh_str:
            try:
                row_copy = row.copy()
                ah, am, as_ = map(int, row['arrival_time'].split(':'))
                arrival_dt = now.replace(hour=ah, minute=am, second=as_, microsecond=0)
                diff = (arrival_dt - now).total_seconds()
                row_copy['eta'] = f"{int(diff // 60):02d}:{int(diff % 60):02d}"

                # Calculate estimated reach time for this specific trip
                trip_data = [t for t in trips if t['trip_id'] == row['trip_id']]
                dest_stop = next((t for t in trip_data if t['station_id'] == end_id), None)
                if dest_stop:
                    # Basic 12h conversion for dest arrival
                    dh, dm, ds = map(int, dest_stop['arrival_time'].split(':'))
                    row_copy['est_reach'] = now.replace(hour=dh, minute=dm, second=ds).strftime('%I:%M %p')
                else:
                    # Fallback if trip doesn't reach dest (unlikely with valid route)
                    row_copy['est_reach'] = (arrival_dt + timedelta(minutes=duration)).strftime('%I:%M %p')

                upcoming_hour.append(row_copy)
            except:
                continue
            if len(upcoming_hour) >= 8: break

    # Convert boarding & arrival to 12-hour format
    boarding_at_source = "--:--"
    if gtfs_boarding_time:
        try:
            h, m, s = map(int, gtfs_boarding_time.split(':'))
            boarding_at_source = now.replace(hour=h, minute=m, second=s).strftime('%I:%M %p')
        except:
            boarding_at_source = gtfs_boarding_time

    if gtfs_arrival_time:
        try:
            h, m, s = map(int, gtfs_arrival_time.split(':'))
            arrival_at_destination = now.replace(hour=h, minute=m, second=s).strftime('%I:%M %p')
        except:
            arrival_at_destination = gtfs_arrival_time
    else:
        arrival_at_destination = (now + timedelta(minutes=duration)).strftime('%I:%M %p')

    # Convert upcoming_hour to 12-hour
    for u in upcoming_hour:
        try:
            h, m, s = map(int, u['arrival_time'].split(':'))
            u['arrival_time'] = now.replace(hour=h, minute=m, second=s).strftime('%I:%M %p')
        except:
            pass

    return jsonify({
        'sequence': sequence,
        'upcoming_hour': upcoming_hour,
        'duration': duration,
        'boarding_at_source': boarding_at_source,
        'arrival_at_dest': arrival_at_destination,
        'total_stops': len(sequence),
        'total_km': round(total_km, 2),
        'fare': calculated_fare,
        'recommendation': recommendation,
        'load': round(load_pct, 1),
        'peak_intensity': peak_intensity,
        'guides': guides,
        'eco': {
            'co2': co2_saved,
            'calories': calories,
            'trees': trees_saved
        },
        'metrics': {
            'peak': is_peak,
            'it_hub': is_it_hub,
            'fare_stable': True
        }
    })


@app.route('/api/live-map')
def api_live_map():
    trips = ensure_gtfs()
    now_dt = get_app_now()
    now_str = now_dt.strftime('%H:%M:%S')

    active_trains = []

    # Read trips and group by trip_id
    trips_by_id = {}
    for row in trips:
        tid = row['trip_id']
        if tid not in trips_by_id:
            trips_by_id[tid] = []
        trips_by_id[tid].append(row)

    for tid, stops in trips_by_id.items():
        # Sort stops by arrival time
        stops.sort(key=lambda x: x['arrival_time'])

        # Find if train is currently between two stops
        for i in range(len(stops) - 1):
            s1 = stops[i]
            s2 = stops[i + 1]

            if s1['arrival_time'] <= now_str < s2['arrival_time']:
                # Calculate progress
                try:
                    t1_parts = list(map(int, s1['arrival_time'].split(':')))
                    t2_parts = list(map(int, s2['arrival_time'].split(':')))

                    t1 = now_dt.replace(hour=t1_parts[0], minute=t1_parts[1], second=t1_parts[2])
                    t2 = now_dt.replace(hour=t2_parts[0], minute=t2_parts[1], second=t2_parts[2])

                    total_duration = (t2 - t1).total_seconds()
                    elapsed = (now_dt - t1).total_seconds()

                    progress = max(0, min(1, elapsed / total_duration))

                    active_trains.append({
                        'trip_id': tid,
                        'line': s1['line'],
                        'from_id': s1['station_id'],
                        'to_id': s2['station_id'],
                        'progress': progress,
                        'direction': s1['direction'],
                        'final_stop': s1['final_stop']
                    })
                    break  # Only one segment per trip
                except:
                    continue

    return jsonify({'trains': active_trains})


# ==========================================
# 5. UI TEMPLATE (UPGRADED)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HydMetro Pro | Intelligence Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

        :root {
            --bg: #fdfdff;
            --card-bg: rgba(255, 255, 255, 0.85);
            --border: rgba(226, 232, 240, 0.5);
            --accent: #2563eb;
        }

        body { 
            font-family: 'Plus Jakarta Sans', sans-serif; 
            background: var(--bg); 
            color: #0f172a; 
            overflow-x: hidden; 
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,10%,0.05) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,0.03) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,0.05) 0, transparent 50%);
            background-attachment: fixed;
        }

        .glass-card { 
            background: var(--card-bg); 
            backdrop-filter: blur(16px) saturate(180%); 
            border-radius: 1.75rem; 
            border: 1px solid var(--border); 
            padding: 24px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.02), 0 10px 30px -10px rgba(0,0,0,0.05);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .glass-card:hover {
            transform: translateY(-4px);
            border-color: rgba(59, 130, 246, 0.2);
            box-shadow: 0 20px 40px -20px rgba(0,0,0,0.08);
        }

        .sidebar { display: none !important; }
        .main { 
            max-width: 1400px;
            margin: 0 auto !important; 
            padding: 2px 24px 140px 24px; 
            min-height: 100vh; 
        }

        .sidebar-toggle { display: none !important; }

        @media (max-width: 1024px) {
            .main { 
                padding: 10px; 
                padding-top: 20px; /* Reduced from 60px */
                padding-bottom: 120px; 
            }
            .tab-content.active {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
            }
            .tab-content.active > * {
                width: 100%;
                max-width: 480px;
            }
        }

        /* Universal Bottom Navigation */
        .mobile-nav { 
            display: flex !important; 
            position: fixed; 
            bottom: 30px; 
            left: 50%;
            transform: translateX(-50%);
            width: calc(100% - 40px);
            max-width: 480px;
            background: rgba(15, 23, 42, 0.98); 
            backdrop-filter: blur(24px);
            border-radius: 32px; 
            z-index: 5000;
            padding: 8px; 
            justify-content: space-evenly;
            align-items: center;
            box-shadow: 0 30px 60px -12px rgba(0,0,0,0.6);
            border: 1px solid rgba(255,255,255,0.15);
        }
        .mobile-link {
            display: flex; flex-direction: column; align-items: center; gap: 4px;
            color: #94a3b8; padding: 12px 16px; border-radius: 20px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer;
            flex: 1;
            max-width: 90px;
        }
        .mobile-link i { width: 22px; height: 22px; }
        .mobile-link span { font-size: 8px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; }
        .mobile-link.active { background: rgba(59, 130, 246, 0.2); color: #60a5fa; transform: translateY(-4px); }
        .mobile-link:hover:not(.active) { color: white; background: rgba(255,255,255,0.08); }

        /* Tickets Middle Emphasis */
        .mobile-link[id="mob-tickets"] {
            background: #2563eb;
            color: white;
            border-radius: 24px;
            box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.5);
            margin-top: -10px;
            padding: 14px 16px;
        }
        .mobile-link[id="mob-tickets"].active {
            background: #1e40af;
            transform: translateY(-8px);
        }

        /* Pure App Header */
        .app-header {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 64px;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            z-index: 1500;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid rgba(226, 232, 240, 0.8);
            padding: 0 20px;
        }

        @media (max-width: 1024px) {
            .app-header { display: flex; }
        }

        /* Tab Transition */
        .tab-content { display: none; opacity: 0; transform: translateY(10px); transition: all 0.4s ease; }
        .tab-content.active { display: block; opacity: 1; transform: translateY(0); }

        .nav-link { 
            display: flex; align-items: center; gap: 12px; padding: 12px 20px; 
            border-radius: 12px; color: #64748b; font-weight: 700; 
            transition: all 0.3s; cursor: pointer; text-transform: uppercase; 
            font-size: 10px; letter-spacing: 0.05em; 
        }
        .nav-link.active { background: #0f172a; color: white; }

        .tab-content.active { display: block; animation: contentFade 0.6s cubic-bezier(0.16, 1, 0.3, 1); }
        @keyframes contentFade { from { opacity: 0; transform: scale(0.98); } to { opacity: 1; transform: scale(1); } }

        .sync-clock {
            background: #000;
            color: #00ff00;
            font-family: 'JetBrains Mono', monospace;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 14px;
            box-shadow: inset 0 0 10px rgba(0,255,0,0.2);
        }

        .station-node { transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1); }
        .station-node:hover { stroke-width: 8; r: 10; }
        .station-node.selected { stroke-width: 12 !important; r: 14 !important; }

        .user-pin-outer { animation: sonar 2s infinite; }
        @keyframes sonar {
            0% { r: 6; opacity: 0.8; }
            100% { r: 24; opacity: 0; }
        }

        .bento-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 24px;
        }
        .bento-col-4 { grid-column: span 4; }
        .bento-col-8 { grid-column: span 8; }
        .bento-col-6 { grid-column: span 6; }
        @media (max-width: 768px) {
            .bento-grid { grid-template-columns: 1fr; }
            .bento-col-4, .bento-col-8, .bento-col-6 { grid-column: span 1; }
        }

        #network-svg { background: #0f172a; }
        .station-label {
            font-size: 11px;
            font-weight: 900;
            fill: #cbd5e1;
            text-shadow: 0 1px 3px rgba(0,0,0,0.9);
            pointer-events: none;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        /* Map Satellite Mode styling */
        #tab-map .glass-card {
            background-image: url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
            border: none;
        }

        /* Central Background Decor */
        .page-branding {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            opacity: 0.03;
            z-index: -1;
            pointer-events: none;
            animation: slowRotate 120s linear infinite;
        }
        @keyframes slowRotate { from { transform: translate(-50%, -50%) rotate(0deg); } to { transform: translate(-50%, -50%) rotate(360deg); } }

        .sentiment-btn {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            filter: grayscale(1);
        }
        .sentiment-btn.active {
            transform: scale(1.2);
            filter: grayscale(0);
            background: rgba(59, 130, 246, 0.1);
        }

        /* Neural Ticker Styles */
        .ticker-wrap {
            width: 100%; overflow: hidden; background: #0f172a; padding: 10px 0;
            border-radius: 12px; margin: 12px 0 24px 0; border: 1px solid rgba(255,255,255,0.1);
        }
        .ticker {
            display: inline-block; white-space: nowrap; padding-right: 100%;
            animation: ticker-kf 60s linear infinite;
        }
        .ticker-item {
            display: inline-block; padding: 0 40px; color: #94a3b8; font-size: 10px;
            font-weight: 800; text-transform: uppercase; letter-spacing: 0.15em;
        }
        .ticker-item span { color: #60a5fa; margin-right: 8px; }
        @keyframes ticker-kf {
            0% { transform: translate3d(0, 0, 0); }
            100% { transform: translate3d(-100%, 0, 0); }
        }
    </style>
</head>
<body>
    <div class="page-branding">
        <i data-lucide="satellite" size="400"></i>
    </div>
    <div class="app-header">
        <div class="max-w-[1400px] mx-auto w-full flex items-center justify-between px-4">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 bg-slate-900 rounded-xl flex items-center justify-center text-white shadow-lg"><i data-lucide="train-front" size="20"></i></div>
                <h1 class="text-xl font-black text-slate-900 tracking-tighter">HydMetro Pro</h1>
            </div>
            <div class="flex items-center gap-4">
                <div class="hidden lg:flex items-center gap-2 bg-slate-50 px-3 py-2 rounded-xl border border-slate-100">
                    <i data-lucide="clock" size="14" class="text-slate-400"></i>
                    <select id="sim-time" onchange="applySimulation()" class="bg-transparent text-[10px] font-black uppercase outline-none cursor-pointer text-slate-600">
                        <option value="-1">Live Mode</option>
                        <option value="8">08:00 AM [Peak]</option>
                        <option value="12">12:00 PM [Off-Peak]</option>
                        <option value="18">06:00 PM [Peak]</option>
                        <option value="22">10:00 PM [Night]</option>
                    </select>
                </div>
                <button class="p-2.5 bg-slate-50 text-slate-400 rounded-xl hover:bg-slate-100 transition-colors"><i data-lucide="search" size="18"></i></button>
                <button class="p-2.5 bg-slate-50 text-slate-500 rounded-xl hover:bg-slate-100 transition-colors"><i data-lucide="user" size="18"></i></button>
            </div>
        </div>
    </div>

    <div class="mobile-nav">
        <div onclick="showTab('home')" class="mobile-link active" id="mob-home"><i data-lucide="home"></i><span>Home</span></div>
        <div onclick="showTab('map')" class="mobile-link" id="mob-map"><i data-lucide="map-pinned"></i><span>Map</span></div>
        <div onclick="showTab('tickets')" class="mobile-link" id="mob-tickets"><i data-lucide="qr-code"></i><span>Tickets</span></div>
        <div onclick="showTab('routes')" class="mobile-link" id="mob-routes"><i data-lucide="route"></i><span>Planner</span></div>
        <div onclick="showTab('feedback')" class="mobile-link" id="mob-feedback"><i data-lucide="heart"></i><span>Feed</span></div>
    </div>

    <div class="main" id="main-content">
        <div id="tab-home" class="tab-content active">
            <header class="flex flex-col lg:flex-row lg:justify-between lg:items-start gap-8 mb-2">
                <div class="hidden lg:block">
                    <h2 id="greeting" class="text-4xl lg:text-5xl font-black text-slate-900 mb-2 tracking-tighter">Good Day!</h2>
                    <p id="env-msg" class="text-slate-400 font-bold max-w-sm leading-relaxed uppercase text-[10px] tracking-widest">Neural processing active. Enjoy your commute across the network.</p>
                </div>
                <div class="lg:hidden w-full text-center">
                     <h2 id="greeting-mob" class="text-3xl font-black text-slate-900 mb-1 tracking-tighter">Good Morning!</h2>
                     <p id="near-metro-mob" class="text-[11px] font-black text-blue-600 uppercase tracking-widest mb-4">Syncing Node...</p>
                     <div class="flex items-center justify-center gap-2">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                        <p class="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Satellite Track Active</p>
                     </div>
                </div>
                <div class="glass-card py-4 px-8 flex flex-row lg:flex-col items-center lg:items-end justify-between lg:justify-center gap-6 border-slate-200">
                    <div class="flex items-baseline gap-2">
                        <span id="clock" class="text-3xl lg:text-4xl font-black text-slate-900 tabular-nums tracking-tighter">00:00:00</span>
                        <span id="ampm" class="text-xs font-black text-slate-400 uppercase">AM</span>
                    </div>
                    <span id="date" class="text-[9px] lg:text-[10px] font-black text-blue-600 uppercase tracking-[0.2em]">October 24, 2024</span>
                </div>
            </header>

            <div class="ticker-wrap shadow-xl">
                <div class="ticker" id="neural-ticker">
                    <div class="ticker-item"><span>SYS_MSG</span> NEURAL MATRIX STABILIZED. WELCOME TO HYDMETRO PRO.</div>
                    <div class="ticker-item"><span>AI_LOAD</span> PREDICTING MEDIUM-LOW DENSITY FOR NEXT 30 MINUTES.</div>
                    <div class="ticker-item"><span>METRIC</span> 99.4% NETWORK UPTIME DETECTED VIA SATELLITE.</div>
                    <div class="ticker-item"><span>WEATHER</span> AC COOLING OPTIMIZED FOR LOCAL HUMIDITY.</div>
                </div>
            </div>

            <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-8 mb-8">
                <div class="glass-card flex p-4 lg:p-6 items-center gap-4 lg:gap-6 border-slate-200 overflow-hidden">
                    <div class="w-10 h-10 lg:w-14 lg:h-14 bg-blue-50 text-blue-600 rounded-xl lg:rounded-2xl flex items-center justify-center shrink-0"><i data-lucide="crosshair" size="18"></i></div>
                    <div class="overflow-hidden"><p class="text-[8px] lg:text-[9px] font-black text-slate-400 uppercase tracking-widest mb-0.5 lg:mb-1">Live Near Metro</p><h3 id="near-name" class="text-[11px] lg:text-sm font-black text-slate-800 truncate">Locating...</h3></div>
                </div>
                <div class="glass-card flex p-4 lg:p-6 items-center gap-4 lg:gap-6 border-slate-200">
                    <div class="w-10 h-10 lg:w-14 lg:h-14 bg-orange-50 text-orange-600 rounded-xl lg:rounded-2xl flex items-center justify-center shrink-0"><i data-lucide="waves" size="18"></i></div>
                    <div>
                        <p class="text-[8px] lg:text-[9px] font-black text-slate-400 uppercase tracking-widest mb-0.5 lg:mb-1">Atmosphere</p>
                        <h3 id="weather-val" class="text-[11px] lg:text-sm font-black text-slate-800">--°C</h3>
                        <p id="weather-detail" class="text-[7px] lg:text-[8px] font-bold text-slate-400 uppercase tracking-tight">Syncing Sky...</p>
                    </div>
                </div>
                <div class="glass-card col-span-2 hidden lg:flex p-6 items-center justify-between border-slate-200 bg-slate-900 text-white overflow-hidden relative">
                    <div class="absolute -right-10 -bottom-10 w-40 h-40 bg-emerald-500/10 rounded-full blur-3xl"></div>
                    <div class="flex items-center gap-6 relative z-10">
                        <div class="w-14 h-14 bg-emerald-500/20 text-emerald-400 rounded-2xl flex items-center justify-center"><i data-lucide="activity" size="24"></i></div>
                        <div>
                            <p class="text-[9px] font-black text-white/40 uppercase tracking-widest mb-1">GTFS Network Pulse</p>
                            <h3 class="text-xl font-black text-white tracking-tighter">Trips Active: <span id="active-count" class="text-emerald-400 tabular-nums">--</span></h3>
                        </div>
                    </div>
                    <div class="text-right relative z-10">
                         <span id="load-status" class="px-3 py-1 bg-white/10 rounded-lg text-[9px] font-black uppercase tracking-widest border border-white/10">Loading Matrix...</span>
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-12 mb-16">
                <div class="lg:col-span-2 space-y-8">
                    <div class="glass-card p-0 overflow-hidden border-slate-200 shadow-2xl shadow-slate-200/40">
                        <div class="p-6 lg:p-10 border-b border-slate-100 flex justify-between items-center bg-slate-50/30">
                            <div class="flex items-center gap-3 lg:gap-5">
                                <i data-lucide="radio" class="text-blue-600 animate-pulse" size="18"></i>
                                <h3 class="text-[9px] lg:text-[11px] font-black text-slate-900 uppercase tracking-[0.3em]">Arrival Board <span class="text-slate-200 hidden lg:inline">|</span> <span id="near-metro-live" class="text-blue-600 ml-1">Syncing...</span></h3>
                            </div>
                            <div class="flex gap-2">
                                 <select id="board-station-selector" onchange="manualStationChange()" class="text-[8px] lg:text-[10px] font-black uppercase bg-white border border-slate-200 px-3 lg:px-4 py-1.5 lg:py-2 rounded-lg lg:rounded-xl outline-none focus:ring-4 focus:ring-blue-500/10 cursor-pointer shadow-sm">
                                    <option value="">Satellite...</option>
                                 </select>
                            </div>
                        </div>
                        <div class="p-6 lg:p-10">
                            <div class="overflow-x-auto lg:overflow-visible">
                                <table class="w-full text-left min-w-[300px]">
                                    <thead>
                                        <tr class="text-[8px] lg:text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] border-b border-slate-100">
                                            <th class="pb-4 lg:pb-8">Vector</th>
                                            <th class="pb-4 lg:pb-8">Node</th>
                                            <th class="pb-4 lg:pb-8">Boarding</th>
                                            <th class="pb-4 lg:pb-8 text-right">Countdown</th>
                                        </tr>
                                    </thead>
                                    <tbody id="board-rows" class="divide-y divide-slate-50"></tbody>
                                </table>
                            </div>
                            <div id="board-loading" class="py-16 lg:py-24 text-center">
                                 <div class="w-8 h-8 lg:w-10 lg:h-10 border-4 border-slate-900 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                                 <p class="text-[8px] lg:text-[10px] font-black uppercase tracking-widest text-slate-400">Syncing...</p>
                            </div>
                        </div>
                    </div>

                    <!-- Favorite Vectors Section -->
                    <div id="fav-vectors-section" class="hidden animate-in fade-in duration-1000">
                        <div class="flex items-center justify-between mb-6 px-4">
                            <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 flex items-center gap-2">
                                <i data-lucide="star" size="14" class="text-amber-500"></i> Frequent Neural Vectors
                            </h4>
                        </div>
                        <div id="fav-vectors-list" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <!-- Injected via JS -->
                        </div>
                    </div>
                </div>
                <div class="space-y-8">
                    <div class="glass-card action-card bg-slate-900 text-white border-none p-10 overflow-hidden relative shadow-2xl shadow-slate-900/10">
                        <div class="absolute -right-5 -top-5 w-40 h-40 bg-blue-500/20 rounded-full blur-[60px]"></div>
                        <i data-lucide="map-pinned" class="mb-8 opacity-40" size="32"></i>
                        <h4 class="text-xl font-black mb-3 relative z-10 tracking-tight">Book Metro Ticket</h4>
                        <p class="text-xs text-white/50 mb-10 relative z-10 font-bold uppercase tracking-widest">Select your destination and secure a digital token instantly.</p>
                        <button onclick="showTab('routes')" class="w-full py-5 bg-white text-slate-900 rounded-2xl font-black text-[11px] uppercase tracking-widest shadow-xl flex items-center justify-center gap-3 relative z-10">
                            <i data-lucide="route" size="14"></i> Open Planner
                        </button>
                    </div>
                    <div class="glass-card action-card border-none p-10 bg-white overflow-hidden relative shadow-2xl shadow-slate-200/50">
                        <div class="absolute -right-5 -top-5 w-40 h-40 bg-indigo-50 rounded-full blur-[60px]"></div>
                        <i data-lucide="scan" class="mb-8 text-indigo-600" size="32"></i>
                        <h4 class="text-xl font-black mb-3 relative z-10 tracking-tight text-slate-900">Neural Connect</h4>
                        <p class="text-xs text-slate-400 mb-10 relative z-10 font-bold uppercase tracking-widest">Recharge your smart matrix card via NFC interface.</p>
                        <button class="w-full py-5 bg-slate-900 text-white rounded-2xl font-black text-[11px] uppercase tracking-widest">Digital Recharge</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- NETWORK MAP -->
        <div id="tab-map" class="tab-content">
            <div class="text-center lg:text-left mb-4">
                <h2 class="text-3xl font-black text-slate-900 tracking-tight">Dynamic Network Topology</h2>
            </div>
            <div class="glass-card p-0 relative h-[800px] overflow-hidden bg-slate-900 border-none shadow-inner">
                <svg id="network-svg" viewBox="0 0 2500 1500" class="w-full h-full cursor-grab active:cursor-grabbing">
                    <g id="map-lines"></g>
                    <g id="map-stations"></g>
                    <g id="map-trains"></g>
                    <g id="map-user-pin"></g>
                </svg>
                <div id="map-overlay" class="absolute top-0 right-0 h-full w-full lg:w-[400px] translate-x-full z-20 transition-transform duration-500 ease-in-out bg-white shadow-[-20px_0_50px_-10px_rgba(0,0,0,0.1)] border-l border-slate-100">
                    <div class="h-full flex flex-col p-6 lg:p-10 overflow-hidden relative">
                        <div class="absolute top-6 right-6 lg:top-8 lg:right-8 z-20">
                            <button onclick="closeOverlay()" class="p-3 bg-slate-50 hover:bg-slate-100 text-slate-400 rounded-2xl transition-colors">
                                <i data-lucide="x" size="20"></i>
                            </button>
                        </div>

                        <div class="mb-10 px-2 lg:px-0">
                            <span id="ov-line" class="px-3 py-1 text-[10px] font-black uppercase rounded-lg mb-4 inline-block">LINE</span>
                            <h4 id="ov-name" class="text-3xl lg:text-4xl font-black text-slate-900 leading-tight tracking-tighter">Station Name</h4>
                        </div>

                        <div class="flex-1 overflow-y-auto space-y-10 scrollbar-hide">
                            <!-- Amenities Section -->
                            <div>
                                <h5 class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                                    <i data-lucide="layout-grid" size="12"></i> Hub Infrastructure
                                </h5>
                                <div id="ov-amenities" class="grid grid-cols-2 gap-3"></div>
                            </div>

                            <!-- Real-time Departures Section -->
                            <div>
                                <h5 class="text-[10px] font-black text-blue-600 uppercase tracking-widest mb-6 flex items-center gap-2">
                                    <i data-lucide="radio" size="12" class="animate-pulse"></i> Neural Flux (Live)
                                </h5>
                                <div id="ov-trains" class="space-y-3">
                                    <!-- JS Injected -->
                                </div>
                            </div>
                        </div>

                        <div class="pt-8 border-t border-slate-100 mt-auto">
                            <button id="ov-plan-btn" class="w-full py-5 bg-slate-900 text-white rounded-2xl font-black text-[11px] uppercase tracking-widest flex items-center justify-center gap-3 shadow-xl shadow-slate-200">
                                <i data-lucide="navigation" size="14"></i> Set as Destination
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- SMART PLANNER -->
        <div id="tab-routes" class="tab-content">
            <div class="max-w-4xl mx-auto">
                <div class="text-center mb-4">
                   <h2 class="text-4xl lg:text-5xl font-black tracking-tighter mb-1 text-slate-900">Neural Path Architect</h2>
                   <p class="text-[10px] font-bold text-slate-400 uppercase tracking-[0.4em] flex items-center justify-center gap-2"><i data-lucide="shield-check" size="14" class="text-emerald-500"></i> Optimized Network Synchrony</p>
                </div>

                <div id="planner-input-area" class="glass-card border-none shadow-2xl bg-white p-8 lg:p-12 relative overflow-hidden group mb-8">
                    <div class="absolute -right-20 -top-20 w-96 h-96 bg-blue-50/50 rounded-full blur-[100px] transition-all group-hover:bg-blue-100/30"></div>

                    <div class="grid grid-cols-1 md:grid-cols-11 gap-4 items-end relative z-10">
                        <div class="md:col-span-5 space-y-3">
                            <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1">Source Node</label>
                            <div class="relative">
                                <div class="absolute left-6 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-blue-600 ring-8 ring-blue-50"></div>
                                <select id="start-st" class="w-full pl-14 pr-8 py-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-blue-500/30 focus:bg-white font-black appearance-none text-slate-900 transition-all cursor-pointer hover:bg-slate-100"></select>
                                <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none"><i data-lucide="chevron-down" size="18"></i></div>
                            </div>
                        </div>

                        <div class="md:col-span-1 flex justify-center py-4 md:py-0">
                            <div class="p-4 bg-slate-900 text-white rounded-full shadow-2xl hover:rotate-180 transition-transform duration-700 cursor-pointer border-4 border-white">
                                <i data-lucide="repeat-2" size="20"></i>
                            </div>
                        </div>

                        <div class="md:col-span-5 space-y-3">
                            <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1">Target Node</label>
                            <div class="relative">
                                <div class="absolute left-6 top-1/2 -translate-y-1/2 text-emerald-500"><i data-lucide="map-pin" size="18"></i></div>
                                <select id="end-st" class="w-full pl-14 pr-8 py-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-emerald-500/30 focus:bg-white font-black appearance-none text-slate-900 transition-all cursor-pointer hover:bg-slate-100"></select>
                                <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none"><i data-lucide="chevron-down" size="18"></i></div>
                            </div>
                        </div>
                    </div>

                    <button id="plan-btn" onclick="planJourney()" class="w-full py-7 bg-slate-900 hover:bg-black text-white font-black rounded-[32px] shadow-2xl shadow-slate-300 transition-all active:scale-[0.98] text-[13px] uppercase tracking-[0.4em] mt-12 flex items-center justify-center gap-4 group">
                        <span id="btn-text">Initialize Path Simulation</span>
                        <div id="btn-loader" class="hidden w-5 h-5 border-[3px] border-white border-t-transparent rounded-full animate-spin"></div>
                        <i data-lucide="sparkles" size="18" class="text-blue-400 group-hover:scale-125 transition-transform"></i>
                    </button>
                </div>

                <div id="route-output" class="hidden space-y-6 pb-12">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-3">
                            <button onclick="saveCurrentVector()" id="save-vector-btn" class="flex items-center gap-2 px-6 py-3 bg-slate-900 text-white rounded-2xl text-[10px] font-black uppercase tracking-widest shadow-xl hover:bg-black transition-all">
                                <i data-lucide="star" size="14"></i> Save Vector
                            </button>
                            <button onclick="shareRoute()" class="p-3 bg-slate-100 rounded-2xl text-slate-400 hover:text-slate-600 transition-all"><i data-lucide="share-2" size="18"></i></button>
                        </div>
                        <div class="hidden lg:flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            <span class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Real-Time Sync active</span>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <!-- AI Recommendation Box -->
                        <div class="glass-card md:col-span-1 border-none bg-slate-50 p-6 flex flex-col gap-4 shadow-sm border border-slate-100">
                            <div class="flex items-center gap-4">
                                <div class="p-2.5 bg-slate-900 text-white rounded-xl"><i data-lucide="cpu" size="16"></i></div>
                                <h5 class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Neural Path Logic</h5>
                            </div>
                            <p id="route-rec" class="text-[13px] font-bold text-slate-700 italic pr-2 leading-relaxed">--</p>
                        </div>

                        <!-- Crowd Dynamics Card -->
                        <div class="glass-card md:col-span-1 border-none bg-slate-50 p-6 flex flex-col gap-4 shadow-sm border border-slate-100">
                            <div class="flex items-center gap-4">
                                <div class="p-2.5 bg-slate-900 text-white rounded-xl"><i data-lucide="users" size="16"></i></div>
                                <h5 class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Crowd Density</h5>
                            </div>
                            <div class="flex-1">
                                <div class="flex justify-between items-center mb-2">
                                    <p id="route-load-forecast" class="text-[13px] font-black text-slate-800 tracking-tight">Analyzing...</p>
                                    <span id="route-peak-pct" class="text-[9px] font-black px-2 py-1 bg-white rounded-lg border border-slate-200">--% PK</span>
                                </div>
                                <div class="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                                    <div id="load-bar" class="h-full bg-slate-900 transition-all duration-1000" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Eco Metrics Card -->
                        <div class="glass-card md:col-span-1 border-none bg-emerald-50 p-6 flex flex-col gap-4 shadow-sm border border-emerald-100">
                            <div class="flex items-center gap-4">
                                <div class="p-2.5 bg-emerald-500 text-white rounded-xl"><i data-lucide="leaf" size="16"></i></div>
                                <h5 class="text-[9px] font-black text-emerald-600 uppercase tracking-widest">Eco Impact</h5>
                            </div>
                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <p id="route-co2" class="text-xl font-black text-emerald-700 tabular-nums">0.0kg</p>
                                    <p class="text-[8px] font-bold text-emerald-600/60 uppercase">CO2 Offset</p>
                                </div>
                                <div>
                                    <p id="route-cal" class="text-xl font-black text-emerald-700 tabular-nums">0</p>
                                    <p class="text-[8px] font-bold text-emerald-600/60 uppercase">Kcal Burned</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="glass-card p-0 overflow-hidden border-none shadow-2xl bg-white rounded-[40px]">
                        <div class="p-10 lg:p-12 bg-slate-900 text-white flex flex-col lg:flex-row lg:justify-between items-center overflow-hidden relative gap-6">
                            <div class="absolute -left-10 -bottom-10 w-60 h-60 bg-blue-600/20 rounded-full blur-3xl opacity-50"></div>
                            <div class="relative z-10 w-full">
                                <div class="grid grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-8 text-center lg:text-left">
                                    <div>
                                        <p id="route-dur" class="text-4xl lg:text-6xl font-black leading-none tabular-nums tracking-tighter">--</p>
                                        <p class="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mt-5">Transit Time</p>
                                    </div>
                                    <div>
                                        <p id="route-fare" class="text-4xl lg:text-6xl font-black leading-none tabular-nums tracking-tighter">₹--</p>
                                        <p class="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mt-5">Total Fare</p>
                                    </div>
                                    <div>
                                        <p id="route-dist-main" class="text-4xl lg:text-6xl font-black leading-none tabular-nums tracking-tighter text-blue-400">--</p>
                                        <p class="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mt-5">Distance (KM)</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="px-10 py-6 bg-slate-50 border-b border-slate-100 flex justify-between">
                             <div class="flex items-center gap-2"><i data-lucide="layers" class="text-slate-400" size="14"></i> <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Optimized Route | <span id="route-dist">0</span> KM Total</span></div>
                             <div class="flex items-center gap-2"><i data-lucide="zap" class="text-green-500" size="14"></i> <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Neural Sync</span></div>
                        </div>
                        <div class="p-10 bg-white">
                            <div id="route-seq" class="border-l-2 border-slate-100 ml-5 pl-10 space-y-8 py-2"></div>
                        </div>

                        <!-- Upcoming Trains List -->
                        <div class="p-10 bg-slate-50/50 border-t border-slate-100">
                             <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 mb-8 flex items-center gap-2">
                                <i data-lucide="clock" size="14" class="text-blue-500"></i> Next 1 Hour Arriving Trains
                             </h4>
                             <div id="schedule-list" class="space-y-4">
                                 <p class="text-xs font-bold text-slate-300 italic">Finding viable transit vectors...</p>
                             </div>
                        </div>
                    </div>

                    <!-- Digital Ticket & Payment Hub -->
                    <div class="glass-card p-10 border-none bg-slate-900 text-white relative overflow-hidden">
                        <div class="absolute -right-10 -top-10 w-40 h-40 bg-blue-500/10 rounded-full blur-3xl"></div>
                        <div class="flex items-center gap-4 mb-6 relative z-10">
                            <div class="p-2 bg-white/10 rounded-xl"><i data-lucide="ticket" size="16"></i></div>
                            <h5 class="text-[10px] font-black uppercase tracking-widest">Secure Digital Ticket</h5>
                        </div>
                        <p class="text-xs text-white/60 mb-8 relative z-10 font-bold opacity-80 uppercase tracking-widest">Instant checkout via UPI. Your ticket will be issued upon successful payment.</p>
                        <div class="space-y-4 relative z-10">
                            <button onclick="toggleUPISelection()" id="pay-btn-main" class="w-full py-5 bg-white text-slate-900 rounded-2xl font-black text-[11px] uppercase tracking-widest shadow-xl flex items-center justify-center gap-3 transition-all active:scale-[0.98]">
                                <i data-lucide="smartphone" size="14"></i> Select Payment App
                            </button>
                            <div id="upi-selection" class="hidden grid grid-cols-2 gap-3 animate-in fade-in slide-in-from-top-2 duration-300">
                                <button onclick="payWithUPI('Google Pay')" class="p-4 bg-white/10 hover:bg-white/20 rounded-xl flex flex-col items-center gap-2 transition-all border border-white/5">
                                    <div class="w-6 h-6 bg-white rounded-md flex items-center justify-center p-1"><i data-lucide="wallet" class="text-slate-900" size="14"></i></div>
                                    <span class="text-[8px] font-black uppercase">G-Pay</span>
                                </button>
                                <button onclick="payWithUPI('PhonePe')" class="p-4 bg-white/10 hover:bg-white/20 rounded-xl flex flex-col items-center gap-2 transition-all border border-white/5">
                                    <div class="w-6 h-6 bg-purple-500 rounded-md flex items-center justify-center p-1"><i data-lucide="zap" class="text-white" size="14"></i></div>
                                    <span class="text-[8px] font-black uppercase">PhonePe</span>
                                </button>
                                <button onclick="payWithUPI('Paytm')" class="p-4 bg-white/10 hover:bg-white/20 rounded-xl flex flex-col items-center gap-2 transition-all border border-white/5">
                                    <div class="w-6 h-6 bg-sky-400 rounded-md flex items-center justify-center p-1"><i data-lucide="credit-card" class="text-white" size="14"></i></div>
                                    <span class="text-[8px] font-black uppercase">Paytm</span>
                                </button>
                                <button onclick="payWithUPI('Others')" class="p-4 bg-white/10 hover:bg-white/20 rounded-xl flex flex-col items-center gap-2 transition-all border border-white/5">
                                    <div class="w-6 h-6 bg-slate-700 rounded-md flex items-center justify-center p-1"><i data-lucide="qr-code" class="text-white" size="14"></i></div>
                                    <span class="text-[8px] font-black uppercase">Others</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="route-empty" class="glass-card flex flex-col items-center justify-center py-32 text-slate-300 border-dashed border-2 border-slate-200">
                    <i data-lucide="cpu" size="48" class="mb-4 opacity-20"></i><p class="font-bold text-slate-400 uppercase text-[10px] tracking-widest">Engine Idle. Awaiting logic trigger.</p>
                </div>
            </div>
        </div>

        <div id="tab-tickets" class="tab-content">
            <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-4 gap-6 border-b pb-4 border-slate-200">
                <div class="flex flex-col items-center lg:flex-row lg:items-center gap-6 text-center lg:text-left">
                   <div class="w-16 h-16 bg-slate-900 rounded-3xl flex items-center justify-center text-white shadow-xl shadow-slate-900/20"><i data-lucide="pocket" size="32"></i></div>
                   <div>
                      <h2 class="text-4xl font-black tracking-tight mb-1 text-slate-900">Digital Vault</h2>
                      <p class="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center justify-center lg:justify-start gap-2">
                        <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                        Blockchain Verified Tokens
                      </p>
                   </div>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div class="lg:col-span-5">
                    <div class="flex items-center justify-between mb-4">
                        <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-900 flex items-center gap-2">
                            <i data-lucide="zap" class="text-blue-600" size="14"></i> Active Boarding Pass
                        </h4>
                    </div>

                    <div id="active-ticket-container" class="perspective-1000">
                        <!-- Active ticket injected here -->
                        <div class="glass-card border-dashed border-2 flex flex-col items-center justify-center py-32 text-slate-300">
                             <div class="p-6 bg-slate-50 rounded-full mb-6"><i data-lucide="qr-code" size="48" class="opacity-10"></i></div>
                             <p class="text-[10px] font-black uppercase tracking-widest text-slate-400">Vault Empty</p>
                             <p class="text-[9px] font-bold text-slate-300 mt-2 uppercase tracking-tight">Generate a ticket from Planner</p>
                        </div>
                    </div>
                </div>

                <div class="lg:col-span-7">
                    <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 mb-8 flex items-center gap-2">
                         <i data-lucide="history" size="14"></i> Journey Vector History
                    </h4>
                    <div id="trip-history" class="space-y-4">
                        <!-- History injected here -->
                    </div>
                    <div id="history-empty" class="py-20 text-center flex flex-col items-center justify-center text-slate-300 bg-slate-50/50 rounded-[32px] border-2 border-dashed">
                        <i data-lucide="folder-clock" size="32" class="mb-4 opacity-20"></i>
                        <p class="text-[10px] font-black uppercase tracking-widest">No previous vectors found</p>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-feedback" class="tab-content">
            <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-4 gap-6 border-b pb-4 border-slate-200">
                <div class="flex flex-col items-center lg:flex-row lg:items-center gap-6 text-center lg:text-left">
                    <div class="w-16 h-16 bg-blue-600 rounded-3xl flex items-center justify-center text-white shadow-xl shadow-blue-500/30"><i data-lucide="message-square" size="32"></i></div>
                    <div>
                       <h2 class="text-4xl font-black tracking-tight mb-1 text-slate-900">Sentiment Engine</h2>
                       <p class="text-xs font-bold text-slate-400 uppercase tracking-widest">Your input optimizes the neural network</p>
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div class="lg:col-span-5">
                    <div class="glass-card border-none shadow-2xl bg-white p-10 relative overflow-hidden group">
                        <div class="absolute -right-20 -top-20 w-80 h-80 bg-blue-50 rounded-full blur-3xl transition-all group-hover:bg-blue-100/50"></div>

                        <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-900 mb-10 relative z-10">Neural Input Form</h4>

                        <div class="space-y-8 relative z-10">
                            <div class="space-y-3">
                                <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Sentiment Rating</label>
                                <div class="flex justify-between gap-2">
                                    <button onclick="setSentiment('sad')" class="sentiment-btn p-4 bg-slate-50 rounded-2xl flex-1 text-2xl" id="sent-sad">🙁</button>
                                    <button onclick="setSentiment('neutral')" class="sentiment-btn p-4 bg-slate-50 rounded-2xl flex-1 text-2xl" id="sent-neutral">😐</button>
                                    <button onclick="setSentiment('happy')" class="sentiment-btn p-4 bg-slate-50 rounded-2xl flex-1 text-2xl" id="sent-happy">😊</button>
                                    <button onclick="setSentiment('love')" class="sentiment-btn p-4 bg-slate-50 rounded-2xl flex-1 text-2xl" id="sent-love">🚀</button>
                                </div>
                            </div>

                            <div class="space-y-3">
                                <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Log Category</label>
                                <select id="feedback-cat" class="w-full p-5 bg-slate-50 border-2 border-transparent rounded-[24px] outline-none focus:border-blue-500/20 focus:bg-white font-bold text-sm appearance-none cursor-pointer">
                                    <option>System Experience</option>
                                    <option>Network Accuracy</option>
                                    <option>Neural Timing</option>
                                    <option>Feature Matrix</option>
                                </select>
                            </div>

                            <div class="space-y-3">
                                <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Transmission Data</label>
                                <textarea id="feedback-msg" rows="4" class="w-full p-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-blue-500/20 focus:bg-white font-bold text-sm resize-none" placeholder="Describe your experience across the network..."></textarea>
                            </div>

                            <button onclick="submitFeedback()" class="w-full py-6 bg-slate-900 text-white font-black rounded-[30px] text-[11px] uppercase tracking-[0.3em] shadow-2xl hover:bg-black transition-all flex items-center justify-center gap-3">
                                <i data-lucide="send" size="14"></i> Execute Transmission
                            </button>
                        </div>
                    </div>
                </div>
                <div class="lg:col-span-7">
                    <div class="glass-card border-none bg-slate-50/50 p-10 min-h-[600px] rounded-[40px]">
                        <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 mb-10">Recent Transmissions (Local Matrix)</h4>
                        <div id="feedback-history" class="space-y-6">
                            <!-- Injected by JS -->
                        </div>
                        <div id="feedback-empty" class="py-32 text-center flex flex-col items-center justify-center text-slate-300">
                            <div class="p-6 bg-white rounded-full mb-6 border border-slate-100"><i data-lucide="archive" size="32" class="opacity-10"></i></div>
                            <p class="text-[10px] font-black uppercase tracking-widest">Matrix log is stable and empty</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();
        const stations = {{ ALL_STATIONS | tojson }};
        let currentSentiment = 'neutral';
        let simulationHour = -1;
        let activeUpdateInterval = null;
        let currentPlannedRoute = null;

        // Persistence Logic for Saved Vectors
        function loadSavedVectors() {
            const saved = JSON.parse(localStorage.getItem('metro_saved_vectors') || '[]');
            const container = document.getElementById('fav-vectors-list');
            const section = document.getElementById('fav-vectors-section');

            if (!container) return;

            if (saved.length > 0) {
                section.classList.remove('hidden');
                container.innerHTML = '';
                saved.slice(-4).reverse().forEach(v => {
                    const card = document.createElement('div');
                    card.className = "glass-card bg-white p-6 border-slate-100 hover:border-blue-200 cursor-pointer transition-all group flex items-center justify-between";
                    card.onclick = () => {
                        document.getElementById('start-st').value = v.from;
                        document.getElementById('end-st').value = v.to;
                        showTab('routes');
                        planJourney();
                    };

                    card.innerHTML = `
                        <div class="flex items-center gap-5">
                            <div class="w-12 h-12 bg-slate-50 rounded-2xl flex items-center justify-center text-slate-400 group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors">
                                <i data-lucide="route" size="20"></i>
                            </div>
                            <div>
                                <h5 class="text-sm font-black text-slate-900 tracking-tight">${v.fromName} <i data-lucide="chevrons-right" class="inline opacity-30 px-1" size="12"></i> ${v.toName}</h5>
                                <p class="text-[8px] font-black text-slate-400 uppercase tracking-widest mt-1">${v.line || 'Multi-Line'} Access</p>
                            </div>
                        </div>
                        <button onclick="removeSavedVector(event, '${v.id}')" class="p-2 hover:bg-red-50 hover:text-red-500 text-slate-300 rounded-xl transition-all">
                            <i data-lucide="trash-2" size="14"></i>
                        </button>
                    `;
                    container.appendChild(card);
                });
                lucide.createIcons();
            } else {
                section.classList.add('hidden');
            }
        }

        function saveCurrentVector() {
            if (!currentPlannedRoute) return;

            const saved = JSON.parse(localStorage.getItem('metro_saved_vectors') || '[]');
            const startId = document.getElementById('start-st').value;
            const endId = document.getElementById('end-st').value;

            if (saved.some(v => v.from === startId && v.to === endId)) {
                alert("Vector already synced to local matrix.");
                return;
            }

            const clean = (str) => {
                let s = str.split(' ').slice(1).join(' ');
                return s.split(' 💻')[0].split(' 🔄')[0].trim();
            };

            const startNode = document.getElementById('start-st');
            const endNode = document.getElementById('end-st');

            const vector = {
                id: Date.now(),
                from: startId,
                to: endId,
                fromName: clean(startNode.options[startNode.selectedIndex].text),
                toName: clean(endNode.options[endNode.selectedIndex].text),
                line: stations.find(s => s.id === startId).line
            };

            saved.push(vector);
            localStorage.setItem('metro_saved_vectors', JSON.stringify(saved));

            const btn = document.getElementById('save-vector-btn');
            btn.innerHTML = '<i data-lucide="check" size="14"></i> Vector Saved';
            btn.classList.replace('bg-slate-900', 'bg-emerald-500');

            setTimeout(() => {
                btn.innerHTML = '<i data-lucide="star" size="14"></i> Save Vector';
                btn.classList.replace('bg-emerald-500', 'bg-slate-900');
                lucide.createIcons();
            }, 2000);

            loadSavedVectors();
            lucide.createIcons();
        }

        function removeSavedVector(event, id) {
            event.stopPropagation();
            const saved = JSON.parse(localStorage.getItem('metro_saved_vectors') || '[]');
            const filtered = saved.filter(v => v.id.toString() !== id.toString());
            localStorage.setItem('metro_saved_vectors', JSON.stringify(filtered));
            loadSavedVectors();
        }

        function shareRoute() {
            const startName = document.getElementById('start-st').options[document.getElementById('start-st').selectedIndex].text;
            const endName = document.getElementById('end-st').options[document.getElementById('end-st').selectedIndex].text;
            const msg = `⚡ Commuting via HydMetro Pro: ${startName} to ${endName}. Optimized path found for ₹${lastCalculatedFare}!`;

            if (navigator.share) {
                navigator.share({ title: 'HydMetro Route', text: msg, url: window.location.href });
            } else {
                navigator.clipboard.writeText(msg);
                alert("Vector details copied to clipboard.");
            }
        }

        function setSentiment(val) {
            currentSentiment = val;
            document.querySelectorAll('.sentiment-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('sent-' + val).classList.add('active');
        }

        // Feedback Logic
        function loadFeedback() {
            const history = JSON.parse(localStorage.getItem('metro_feedback') || '[]');
            const container = document.getElementById('feedback-history');
            const empty = document.getElementById('feedback-empty');

            if (!container) return;

            container.innerHTML = '';
            if (history.length > 0) {
                if (empty) empty.classList.add('hidden');
                history.reverse().forEach(item => {
                    const reaction = item.sentiment === 'happy' ? '😊' : item.sentiment === 'sad' ? '🙁' : item.sentiment === 'love' ? '🚀' : '😐';
                    const card = document.createElement('div');
                    card.className = "bg-white p-8 rounded-[32px] border border-slate-100 shadow-sm animate-in fade-in slide-in-from-bottom-4 duration-700 relative overflow-hidden group hover:border-blue-200 transition-all";
                    card.innerHTML = `
                        <div class="flex justify-between items-start mb-6">
                            <div class="flex items-center gap-3">
                                <div class="text-2xl drop-shadow-sm">${reaction}</div>
                                <span class="px-3 py-1 bg-slate-50 text-slate-500 text-[9px] font-black rounded-lg uppercase tracking-widest group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors">${item.category}</span>
                            </div>
                            <span class="text-[8px] font-black text-slate-300 uppercase tracking-[0.2em] group-hover:text-slate-400 transition-colors">${item.date}</span>
                        </div>
                        <p class="text-[13px] font-bold text-slate-700 leading-relaxed italic pr-10">"${item.message}"</p>
                        <div class="absolute -right-4 -bottom-4 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity">
                            <i data-lucide="message-circle" size="80"></i>
                        </div>
                    `;
                    container.appendChild(card);
                    lucide.createIcons();
                });
            } else {
                if (empty) empty.classList.remove('hidden');
            }
        }

        async function submitFeedback() {
            const msg = document.getElementById('feedback-msg').value.trim();
            const cat = document.getElementById('feedback-cat').value;

            if (!msg) {
                alert("The neural matrix requires data. Please type a message.");
                return;
            }

            const feedback = {
                id: Date.now(),
                message: msg,
                category: cat,
                sentiment: currentSentiment,
                date: new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
            };

            // Local History Persistence
            const history = JSON.parse(localStorage.getItem('metro_feedback') || '[]');
            history.push(feedback);
            localStorage.setItem('metro_feedback', JSON.stringify(history));

            // Server-side (Cloud) Sync
            try {
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(feedback)
                });
            } catch (e) {
                console.warn("Cloud Sync Offline.");
            }

            // Clear Input
            document.getElementById('feedback-msg').value = '';
            setSentiment('neutral');

            // UI Update
            loadFeedback();

            // Success Effect
            const btn = document.querySelector('[onclick="submitFeedback()"]');
            const oldHtml = btn.innerHTML;
            btn.classList.replace('bg-slate-900', 'bg-emerald-500');
            btn.innerHTML = '<i data-lucide="shield-check" size="14"></i> Transmission Complete';
            lucide.createIcons();

            setTimeout(() => {
                btn.classList.replace('bg-emerald-500', 'bg-slate-900');
                btn.innerHTML = oldHtml;
                lucide.createIcons();
            }, 3000);
        }

        let lastCalculatedFare = 20;

        function toggleUPISelection() {
            const sel = document.getElementById('upi-selection');
            const btn = document.getElementById('pay-btn-main');
            const isHidden = sel.classList.contains('hidden');

            sel.classList.toggle('hidden');
            btn.innerHTML = isHidden ? '<i data-lucide="chevron-up" size="14"></i> Cancel Selection' : '<i data-lucide="smartphone" size="14"></i> Select Payment App';
            lucide.createIcons();
        }

        function payWithUPI(appName) {
            const startNode = document.getElementById('start-st');
            const endNode = document.getElementById('end-st');

            if (!startNode.value || !endNode.value) {
                alert("Please simulate a route in the Path Architect first to generate trip data.");
                showTab('routes');
                return;
            }

            const amount = lastCalculatedFare;
            const fromNameFull = startNode.options[startNode.selectedIndex].text;
            const toNameFull = endNode.options[endNode.selectedIndex].text;

            // Clean names (remove icons/suffixes)
            const clean = (str) => {
                let s = str.split(' ').slice(1).join(' '); // Skip emoji
                return s.split(' 💻')[0].split(' 🔄')[0].trim();
            };

            const upiSelection = document.getElementById('upi-selection');
            upiSelection.innerHTML = `
                <div class="col-span-2 py-10 flex flex-col items-center justify-center animate-pulse">
                    <div class="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin mb-4"></div>
                    <p class="text-[10px] font-black uppercase tracking-[0.2em] text-white">Opening ${appName}...</p>
                    <p class="text-[8px] font-bold text-white/40 mt-2">Connecting to Secure Gateway</p>
                </div>
            `;

            setTimeout(() => {
                upiSelection.innerHTML = `
                    <div class="col-span-2 flex flex-col items-center justify-center p-6 bg-emerald-500 rounded-3xl animate-in zoom-in duration-500">
                        <i data-lucide="shield-check" size="32" class="text-white mb-4"></i>
                        <h6 class="text-sm font-black text-white uppercase mb-1">Payment Successful</h6>
                        <p class="text-[10px] font-bold text-white/80">Vector Token Authorized</p>
                    </div>
                `;
                lucide.createIcons();

                setTimeout(() => {
                    generateTicket({
                        from: clean(fromNameFull),
                        to: clean(toNameFull),
                        fare: amount,
                        line: stations.find(s => s.id === startNode.value).line
                    });
                }, 1500);
            }, 2500);
        }

        function generateTicket(tripData) {
            const ticket = {
                id: 'T-' + Math.random().toString(36).substr(2, 9).toUpperCase(),
                ...tripData,
                timestamp: new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }),
                status: 'ACTIVE'
            };

            // Save to Local Storage
            const history = JSON.parse(localStorage.getItem('metro_tickets') || '[]');
            history.push(ticket);
            localStorage.setItem('metro_tickets', JSON.stringify(history));

            showTab('tickets');
            renderTickets();

            // Animation effect
            const container = document.getElementById('active-ticket-container');
            container.classList.add('animate-bounce');
            setTimeout(() => container.classList.remove('animate-bounce'), 1000);
        }

        function renderTickets() {
            const history = JSON.parse(localStorage.getItem('metro_tickets') || '[]');
            const activeContainer = document.getElementById('active-ticket-container');
            const historyContainer = document.getElementById('trip-history');
            const historyEmpty = document.getElementById('history-empty');

            activeContainer.innerHTML = '';
            historyContainer.innerHTML = '';

            const activeTicket = history.find(t => t.status === 'ACTIVE');
            const pastTrips = history.filter(t => t.status === 'COMPLETED').reverse();

            if (activeTicket) {
                const lineCol = activeTicket.line === 'Red' ? '#ef4444' : activeTicket.line === 'Blue' ? '#3b82f6' : '#10b981';
                const card = document.createElement('div');
                card.className = "glass-card bg-slate-900 border-none p-0 overflow-hidden shadow-[0_50px_100px_-20px_rgba(15,23,42,0.4)] relative group transition-all duration-700 hover:scale-[1.02]";
                card.innerHTML = `
                    <div class="relative p-10 overflow-hidden">
                        <!-- Holographic Background Effect -->
                        <div class="absolute -right-10 -top-10 w-64 h-64 bg-blue-500/10 rounded-full blur-[100px] animate-pulse"></div>
                        <div class="absolute -left-10 -bottom-10 w-64 h-64 ${activeTicket.line === 'Red' ? 'bg-red-500/10' : 'bg-emerald-500/10'} rounded-full blur-[100px] animate-pulse"></div>

                        <div class="flex justify-between items-center mb-12 relative z-10">
                            <div class="flex items-center gap-4">
                                <div class="w-14 h-14 bg-white/5 backdrop-blur-xl border border-white/10 rounded-[20px] flex items-center justify-center text-white shadow-inner">
                                    <i data-lucide="train-front" size="24" style="color: ${lineCol}"></i>
                                </div>
                                <div class="flex flex-col">
                                    <h5 class="text-white font-black text-lg tracking-tight">Boarding Pass</h5>
                                    <span class="text-[9px] font-black text-blue-500 uppercase tracking-[0.3em]">Neural Authorized</span>
                                </div>
                            </div>
                            <div class="text-right">
                                <p class="text-[9px] font-black uppercase tracking-widest text-white/30 mb-1">Pass Index</p>
                                <p class="text-xs font-black text-white bg-white/5 px-3 py-1.5 rounded-xl border border-white/10 tabular-nums">${activeTicket.id}</p>
                            </div>
                        </div>

                        <div class="grid grid-cols-11 gap-4 items-center mb-12 relative z-10">
                            <div class="col-span-5">
                                <p class="text-[9px] font-black uppercase tracking-[0.3em] text-white/40 mb-3">Origin Hub</p>
                                <p class="text-xl font-black text-white truncate max-w-full tracking-tight">${activeTicket.from}</p>
                                <p class="text-[10px] font-bold text-blue-400 mt-1 uppercase tracking-widest">Hyd-Metro-A</p>
                            </div>
                            <div class="col-span-1 flex flex-col items-center gap-1">
                                <div class="w-1 h-1 rounded-full bg-white/20"></div>
                                <div class="flex-1 w-px bg-gradient-to-b from-white/10 via-blue-500/50 to-white/10 h-8 my-1"></div>
                                <div class="w-1 h-1 rounded-full bg-white/20"></div>
                            </div>
                            <div class="col-span-5 text-right">
                                <p class="text-[9px] font-black uppercase tracking-[0.3em] text-white/40 mb-3">Target Node</p>
                                <p class="text-xl font-black text-white truncate max-w-full tracking-tight">${activeTicket.to}</p>
                                <p class="text-[10px] font-bold text-emerald-400 mt-1 uppercase tracking-widest">Hyd-Metro-B</p>
                            </div>
                        </div>

                        <div class="flex flex-col items-center justify-center bg-white p-12 rounded-[52px] shadow-[inset_0_4px_30px_rgba(0,0,0,0.06)] mb-10 relative z-10 group-hover:scale-105 transition-transform duration-500">
                            <div id="ticket-qr" class="p-1"></div>
                            <p class="text-[8px] font-black uppercase tracking-[0.4em] text-slate-300 mt-6">Neural Identity Token</p>
                        </div>

                        <div class="grid grid-cols-2 gap-4 mb-10 relative z-10">
                            <div class="p-5 bg-white/5 rounded-3xl border border-white/5 transition-colors hover:bg-white/10">
                                <p class="text-[8px] font-black uppercase tracking-[0.2em] text-white/40 mb-1">Issue Time</p>
                                <p class="text-[10px] font-bold text-white uppercase">${activeTicket.timestamp}</p>
                            </div>
                            <div class="p-5 bg-white/5 rounded-3xl border border-white/5 transition-colors hover:bg-white/10">
                                <p class="text-[8px] font-black uppercase tracking-[0.2em] text-white/40 mb-1">Secure Fare</p>
                                <p class="text-[10px] font-bold text-white uppercase tabular-nums">INR ${activeTicket.fare}.00</p>
                            </div>
                        </div>

                        <button onclick="completeTrip('${activeTicket.id}')" class="w-full py-6 bg-white text-slate-900 rounded-[28px] font-black text-[11px] uppercase tracking-[0.3em] transition-all hover:bg-blue-50 active:scale-95 shadow-xl shadow-white/5 mb-2 flex items-center justify-center gap-3">
                             <i data-lucide="scan" size="14"></i> Complete Journey
                        </button>
                    </div>
                    <!-- Animated Scanning Progress -->
                    <div class="h-1.5 bg-white/5 flex relative overflow-hidden">
                         <div class="h-full bg-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.8)] animate-[shimmer_2s_infinite]" style="width: 100%"></div>
                    </div>
                `;
                activeContainer.appendChild(card);

                // Generate QR
                new QRCode(document.getElementById("ticket-qr"), {
                    text: JSON.stringify({ id: activeTicket.id, u: "AIS-HYD", auth: "NEURAL-PRO" }),
                    width: 210,
                    height: 210,
                    colorDark: "#0f172a",
                    colorLight: "#ffffff",
                    correctLevel: QRCode.CorrectLevel.H
                });
            } else {
                activeContainer.innerHTML = `
                    <div class="glass-card border-dashed border-2 flex flex-col items-center justify-center py-40 text-slate-300 rounded-[40px] bg-slate-50/30">
                         <div class="w-20 h-20 bg-white rounded-full flex items-center justify-center shadow-sm mb-6 border border-slate-100">
                            <i data-lucide="qr-code" size="32" class="opacity-10 text-slate-400"></i>
                         </div>
                         <p class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Vault Vector Null</p>
                         <button onclick="showTab('routes')" class="mt-8 px-8 py-4 bg-slate-900 text-white rounded-2xl font-black text-[9px] uppercase tracking-widest shadow-xl">Purchase Digital Token</button>
                    </div>
                `;
            }

            if (pastTrips.length > 0) {
                if (historyEmpty) historyEmpty.classList.add('hidden');
                pastTrips.forEach(trip => {
                    const lineCol = trip.line === 'Red' ? 'bg-red-500' : trip.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                    const div = document.createElement('div');
                    div.className = "bg-white p-8 rounded-[36px] border border-slate-100 flex items-center justify-between group hover:border-blue-200 transition-all shadow-sm hover:shadow-md";
                    div.innerHTML = `
                        <div class="flex items-center gap-6">
                            <div class="w-3 h-14 rounded-full ${lineCol} shadow-sm group-hover:scale-y-110 transition-transform"></div>
                            <div>
                                <p class="text-[16px] font-black text-slate-900 tracking-tight">${trip.from} <span class="text-slate-300 font-normal mx-2 opacity-50"><i data-lucide="chevrons-right" class="inline" size="16"></i></span> ${trip.to}</p>
                                <div class="flex items-center gap-3 mt-2 text-slate-400">
                                    <p class="text-[9px] font-black uppercase tracking-widest">${trip.timestamp}</p>
                                    <span class="w-1 h-1 rounded-full bg-slate-200"></span>
                                    <div class="flex items-center gap-1">
                                        <i data-lucide="qr-code" size="10" class="opacity-40"></i>
                                        <p class="text-[9px] font-black uppercase tracking-widest opacity-60">${trip.id}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="text-right">
                            <p class="text-xl font-black text-slate-900 tabular-nums">₹${trip.fare}</p>
                            <span class="text-[8px] font-black text-emerald-600 uppercase tracking-widest bg-emerald-50 px-3 py-1.5 rounded-xl border border-emerald-100 inline-block mt-2">Archived Vector</span>
                        </div>
                    `;
                    historyContainer.appendChild(div);
                });
            } else {
                if (historyEmpty) historyEmpty.classList.remove('hidden');
            }
            lucide.createIcons();
        }

        function completeTrip(ticketId) {
            const history = JSON.parse(localStorage.getItem('metro_tickets') || '[]');
            const idx = history.findIndex(t => t.id === ticketId);
            if (idx !== -1) {
                history[idx].status = 'COMPLETED';
                history[idx].completedAt = new Date().toISOString();
                localStorage.setItem('metro_tickets', JSON.stringify(history));
                renderTickets();
            }
        }

        function updateUserPin(lat, lng) {
            const g = document.getElementById('map-user-pin');
            let pin = document.getElementById('user-location-pin');

            const lats = stations.map(s => s.lat), lngs = stations.map(s => s.lng);
            const xs = stations.map(s => s.x), ys = stations.map(s => s.y);

            const minLat = Math.min(...lats), maxLat = Math.max(...lats);
            const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
            const minX = Math.min(...xs), maxX = Math.max(...xs);
            const minY = Math.min(...ys), maxY = Math.max(...ys);

            const x = minX + (lng - minLng) / (maxLng - minLng) * (maxX - minX);
            const y = minY + (maxLat - lat) / (maxLat - minLat) * (maxY - minY);

            if(!pin) {
                pin = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                pin.id = 'user-location-pin';
                pin.style.transition = 'transform 1s ease-in-out';
                pin.innerHTML = `<circle r="15" fill="#3b82f6" class="user-pin-outer opacity-25"></circle><circle r="6" fill="#2563eb" stroke="white" stroke-width="2"></circle>`;
                g.appendChild(pin);
            }
            pin.setAttribute('transform', `translate(${x}, ${y})`);
        }

        function showTab(id) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.mobile-link').forEach(l => l.classList.remove('active'));

            document.getElementById('tab-'+id).classList.add('active');
            document.getElementById('mob-'+id).classList.add('active');

            if(id !== 'map') closeOverlay();
        }

        function closeOverlay() {
            document.getElementById('map-overlay').classList.add('translate-x-full');
            document.querySelectorAll('.station-node').forEach(n => n.classList.remove('selected'));
        }

        function updateClock() {
            let now = new Date();
            if (typeof simulationHour !== 'undefined' && simulationHour !== -1) {
                now.setHours(simulationHour);
                if (document.getElementById('env-msg')) {
                    if (!document.getElementById('sim-indicator')) {
                        const badge = document.createElement('span');
                        badge.id = 'sim-indicator';
                        badge.className = 'ml-3 px-2 py-0.5 bg-blue-500 text-white text-[8px] font-black rounded-md tracking-widest animate-pulse';
                        badge.innerText = 'SIMULATED';
                        document.getElementById('ampm').after(badge);
                    }
                }
            } else {
                if (document.getElementById('sim-indicator')) document.getElementById('sim-indicator').remove();
            }
            let options = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true };
            let timeStr = now.toLocaleTimeString('en-US', options);
            let parts = timeStr.split(' ');
            if (parts.length === 2) {
                document.getElementById('clock').innerText = parts[0];
                document.getElementById('ampm').innerText = parts[1];
            } else {
                document.getElementById('clock').innerText = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true }).split(' ')[0];
                document.getElementById('ampm').innerText = now.getHours() >= 12 ? 'PM' : 'AM';
            }
            document.getElementById('date').innerText = now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }).toUpperCase();
        }
        setInterval(updateClock, 1000); updateClock();

        function applySimulation() {
            simulationHour = parseInt(document.getElementById('sim-time').value);
            manualStationChange(); // Refresh current dashboard
            if (activeUpdateInterval) {
                clearInterval(activeUpdateInterval);
                activeUpdateInterval = setInterval(updateLiveTrains, 2000);
                updateLiveTrains();
            }
        }

        async function fetchWithSim(url, options = {}) {
            const separator = url.includes('?') ? '&' : '?';
            const fullUrl = simulationHour !== -1 ? `${url}${separator}sim_hour=${simulationHour}` : url;
            return fetch(fullUrl, options);
        }

        function setupMap() {
            const lineGroup = document.getElementById('map-lines');
            const g = document.getElementById('map-stations');

            // Define sequences for lines
            const lineSequences = {
                'Red': stations.filter(s => s.line === 'Red').sort((a,b) => parseInt(a.id.replace('R','')) - parseInt(b.id.replace('R',''))),
                'Blue': stations.filter(s => s.line === 'Blue').sort((a,b) => parseInt(a.id.replace('B','')) - parseInt(b.id.replace('B',''))),
                'Green': stations.filter(s => s.line === 'Green').sort((a,b) => parseInt(a.id.replace('G','')) - parseInt(b.id.replace('G','')))
            };

            // Draw lines first
            Object.entries(lineSequences).forEach(([line, seq]) => {
                const color = line === 'Red' ? '#ef4444' : line === 'Blue' ? '#3b82f6' : '#22c55e';
                for(let i=0; i < seq.length - 1; i++) {
                    const s1 = seq[i], s2 = seq[i+1];
                    const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    l.setAttribute('x1', s1.x); l.setAttribute('y1', s1.y);
                    l.setAttribute('x2', s2.x); l.setAttribute('y2', s2.y);
                    l.setAttribute('stroke', color + '44'); // Transparent effect
                    l.setAttribute('stroke-width', '10');
                    l.setAttribute('stroke-linecap', 'round');
                    lineGroup.appendChild(l);
                }
            });

            stations.forEach(s => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', s.x); circle.setAttribute('cy', s.y); circle.setAttribute('r', 6);
                circle.setAttribute('class', `station-node fill-white stroke-[3] cursor-pointer transition-all ${s.line === 'Red' ? 'stroke-red-500' : s.line === 'Blue' ? 'stroke-blue-500' : 'stroke-green-500'}`);
                circle.setAttribute('station-id', s.id);
                circle.onclick = async () => {
                    document.querySelectorAll('.station-node').forEach(n => n.classList.remove('selected'));
                    circle.classList.add('selected');

                    const overlay = document.getElementById('map-overlay');
                    overlay.classList.remove('translate-x-full');

                    document.getElementById('ov-name').innerText = s.name;

                    const ovLine = document.getElementById('ov-line');
                    ovLine.innerText = s.line + ' LINE';
                    ovLine.className = 'px-3 py-1 text-[10px] font-black uppercase rounded-lg shadow-sm ' + (s.line === 'Red' ? 'bg-red-50 text-red-600' : s.line === 'Blue' ? 'bg-blue-50 text-blue-600' : 'bg-green-50 text-green-600');

                    const am = document.getElementById('ov-amenities'); am.innerHTML = '';
                    (s.amenities || ['Ticket Counters', 'Tactile Flooring', 'CC Camera', 'Elevator']).forEach(a => {
                        const dev = document.createElement('div'); 
                        dev.className = 'bg-slate-50 p-4 rounded-2xl border border-slate-100 flex flex-col gap-2';
                        dev.innerHTML = `<span class="text-[9px] font-black text-slate-400 uppercase">Available</span><span class="text-[11px] font-bold text-slate-700">${a}</span>`;
                        am.appendChild(dev);
                    });

                    // Set setting as destination action
                    document.getElementById('ov-plan-btn').onclick = () => {
                        document.getElementById('end-st').value = s.id;
                        showTab('routes');
                        closeOverlay();
                    };

                    // Load live arrivals for this station
                    const trainCont = document.getElementById('ov-trains');
                    trainCont.innerHTML = `<div class="py-10 flex flex-col items-center gap-4 text-slate-300"><div class="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div><p class="text-[9px] font-black uppercase tracking-widest">Syncing Flux...</p></div>`;

                    try {
                        const res = await fetch('/api/nearest', { 
                            method: 'POST', 
                            headers: {'Content-Type': 'application/json'}, 
                            body: JSON.stringify({ station_id: s.id }) 
                        });
                        const data = await res.json();
                        trainCont.innerHTML = '';

                        if (data.upcoming.length === 0) {
                            trainCont.innerHTML = `<p class="text-xs font-bold text-slate-400 italic">Node static. No trains projected.</p>`;
                        }

                        data.upcoming.slice(0, 5).forEach(t => {
                            const lineCol = t.line === 'Red' ? 'bg-red-500' : t.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                            const tDiv = document.createElement('div');
                            tDiv.className = "flex justify-between items-center bg-slate-50 p-5 rounded-2xl border border-slate-100 hover:border-blue-200 transition-colors";
                            tDiv.innerHTML = `
                                <div class="flex items-center gap-4">
                                    <div class="w-1 my-1 self-stretch rounded-full ${lineCol}"></div>
                                    <div>
                                        <p class="text-[11px] font-black text-slate-900 leading-none mb-1">${t.final_stop}</p>
                                        <div class="flex items-center gap-2 mt-0.5">
                                            <span class="px-1.5 py-0.5 rounded text-[6px] font-black uppercase tracking-tighter text-white ${lineCol}">${t.line}</span>
                                            <span class="text-[8px] font-bold text-slate-400 uppercase tracking-widest">Platform ${t.platform}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="text-right">
                                    <p class="text-sm font-black text-blue-600 tabular-nums">${t.arrival_time}</p>
                                    <p class="text-[8px] font-black text-slate-400 uppercase tracking-widest">${t.eta}</p>
                                </div>
                            `;
                            trainCont.appendChild(tDiv);
                        });
                    } catch (e) {
                        trainCont.innerHTML = `<p class="text-xs font-bold text-red-500">Sync Failed</p>`;
                    }
                    lucide.createIcons();
                };
                g.appendChild(circle);

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', s.x + 10); text.setAttribute('y', s.y + 4);
                text.setAttribute('class', 'station-label');
                text.textContent = s.name; g.appendChild(text);
            });
        }

        async function updateBoardData(lat, lng, stationId = null) {
            try {
                if (!lat && !lng && !stationId) {
                    console.log("Empty location payload, aborting update.");
                    return;
                }

                document.getElementById('board-loading').classList.remove('hidden');

                if (lat && lng) updateUserPin(lat, lng);

                const body = stationId ? { station_id: stationId } : { lat, lng };
                const res = await fetchWithSim('/api/nearest', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify(body) 
                });
                if (!res.ok) throw new Error("API Offline");
                const data = await res.json();

                document.getElementById('near-name').innerText = data.station.name;
                document.getElementById('near-metro-live').innerText = data.station.name;
                if (document.getElementById('near-metro-mob')) {
                    document.getElementById('near-metro-mob').innerText = 'Near ' + data.station.name + ' Hub';
                }
                document.getElementById('active-count').innerText = data.active_trips;
                document.getElementById('load-status').innerText = data.load_label + ' LOAD';
                document.getElementById('load-status').className = 'px-3 py-1 bg-white/10 rounded-lg text-[9px] font-black uppercase tracking-widest border border-white/10 ' + 
                                                                  (data.load_val === 'High' ? 'text-red-400' : 'text-emerald-400');
                document.getElementById('greeting').innerText = data.greeting + '!';
                if (document.getElementById('greeting-mob')) {
                    document.getElementById('greeting-mob').innerText = data.greeting + '!';
                }
                document.getElementById('weather-val').innerText = data.weather.temp + '°C, ' + data.weather.condition;

                // Sync selector if in auto-mode
                const selector = document.getElementById('board-station-selector');
                if (!stationId) {
                    selector.value = data.station.id;
                }
                document.getElementById('weather-detail').innerText = `Humidity: ${data.weather.humidity}% | Visibility: ${data.weather.visibility.toFixed(1)}km`;

                const weatherRec = data.weather.temp > 35 ? "Extreme heatwave detected. AC Metro cabins are optimal for travel today." :
                                   data.weather.condition.includes("Rain") ? "Rain detected. Metro is the safest and driest transit route." :
                                   "Neural processing active. Enjoy your commute across the network.";
                document.getElementById('env-msg').innerText = weatherRec;

                const loadCol = data.load_val === 'High' ? '#ef4444' : 
                               data.load_val === 'M-High' ? '#f59e0b' : 
                               data.load_val === 'Medium' ? '#6366f1' : '#10b981';

                // Highlight nearest station node
                document.querySelectorAll('.station-node').forEach(node => {
                    node.classList.remove('near-user');
                    if (node.getAttribute('station-id') === data.station.id) {
                        node.classList.add('near-user');
                    }
                });
                const rows = document.getElementById('board-rows'); rows.innerHTML = '';
                document.getElementById('board-loading').classList.add('hidden');

                if (data.upcoming.length === 0) {
                    rows.innerHTML = '<tr><td colspan="4" class="py-16 text-center text-slate-400 font-bold uppercase text-[10px] tracking-widest bg-slate-50/50 rounded-2xl">Signal Lost. No Upcoming departures in this vector.</td></tr>';
                }

                data.upcoming.forEach(t => {
                    const lineCol = t.line === 'Red' ? 'bg-red-500' : t.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                    const tr = document.createElement('tr');
                    tr.className = "group hover:bg-slate-50 transition-colors";
                    tr.innerHTML = `
                        <td class="py-4 lg:py-5"><div class="w-1 lg:w-1.5 h-6 lg:h-8 rounded-full ${lineCol} shadow-sm group-hover:scale-y-110 transition-transform"></div></td>
                        <td class="py-4 lg:py-5">
                            <div class="flex flex-col">
                                <span class="font-black text-slate-800 text-[11px] lg:text-[12px] tracking-tight">${t.final_stop}</span>
                                <div class="flex items-center gap-2 mt-0.5">
                                    <span class="px-1.5 py-0.5 rounded text-[6px] lg:text-[7px] font-black uppercase tracking-tighter text-white ${lineCol}">${t.line}</span>
                                    <span class="text-[7px] lg:text-[8px] font-bold text-slate-500 uppercase tracking-widest">Platform ${t.platform}</span>
                                </div>
                            </div>
                        </td>
                        <td class="py-4 lg:py-5">
                            <span class="text-[10px] lg:text-[12px] font-black text-slate-600 tabular-nums">${t.arrival_time}</span>
                        </td>
                        <td class="py-4 lg:py-5 text-right font-black">
                            <div class="flex flex-col items-end">
                                <span class="sync-clock text-[11px] lg:text-[14px]">${t.eta}</span>
                            </div>
                        </td>`;
                    rows.appendChild(tr);
                });
                lucide.createIcons();
            } catch (err) {
                console.error("Board Sync Error:", err);
                document.getElementById('board-loading').innerHTML = `<i data-lucide="wifi-off" class="text-red-400 mb-2"></i><p class="text-red-400">Sync Interrupted</p>`;
                lucide.createIcons();
            }
        }

        async function manualStationChange() {
            const sid = document.getElementById('board-station-selector').value;
            if (!sid) {
                // Resume Auto-tracking
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(pos => updateBoardData(pos.coords.latitude, pos.coords.longitude));
                }
                return;
            }
            const s = stations.find(st => st.id === sid);
            updateBoardData(s.lat, s.lng, s.id);
        }

        async function initGeo() {
            const selector = document.getElementById('board-station-selector');
            const itHubs = ['Hitech City', 'Madhapur', 'Raidurg'];
            const interchanges = ['Ameerpet', 'MGBS', 'Parade Ground'];
            const lineIcons = { 'Red': '🔴', 'Blue': '🔵', 'Green': '🟢' };

            stations.slice().sort((a,b)=>a.name.localeCompare(b.name)).forEach(st => {
                const opt = document.createElement('option');
                opt.value = st.id;
                let suffix = '';
                if (interchanges.includes(st.name)) suffix = ' 🔄 [Interchange]';
                opt.innerText = `${lineIcons[st.line]} ${st.name}${suffix}`;
                selector.appendChild(opt);
            });

            // Fallback to Ameerpet immediately to avoid blank board
            const ameerpet = stations.find(s => s.name === 'Ameerpet');
            updateBoardData(ameerpet.lat, ameerpet.lng, ameerpet.id);

            if (navigator.geolocation) {
                navigator.geolocation.watchPosition(
                    pos => updateBoardData(pos.coords.latitude, pos.coords.longitude),
                    err => {
                        console.warn("Geo access restricted or interrupted. Fallback active.");
                    },
                    { enableHighAccuracy: false, timeout: 10000, maximumAge: 60000 }
                );
            }
        }

        async function planJourney() {
            const btn = document.getElementById('plan-btn');
            const btnText = document.getElementById('btn-text');
            const loader = document.getElementById('btn-loader');

            btn.disabled = true;
            btnText.classList.add('opacity-0');
            loader.classList.remove('hidden');

            try {
                const f = document.getElementById('start-st').value, t = document.getElementById('end-st').value;
                if (!f || !t) return;

                const res = await fetchWithSim('/api/plan', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({from: f, to: t}) });
                const data = await res.json();

                currentPlannedRoute = data;
                document.getElementById('route-output').classList.remove('hidden');
                const emptyState = document.getElementById('route-empty');
                if (emptyState) emptyState.classList.add('hidden');

                // Details
                document.getElementById('route-dur').innerText = (data.duration || 0) + 'm';
                document.getElementById('route-fare').innerText = '₹' + data.fare;
                document.getElementById('route-dist-main').innerText = data.total_km;
                document.getElementById('route-dist').innerText = data.total_km;
                document.getElementById('route-rec').innerText = data.recommendation;

                // Eco Metrics
                document.getElementById('route-co2').innerText = (data.eco.co2 || 0) + 'kg';
                document.getElementById('route-cal').innerText = (data.eco.calories || 0);

                // Public Load Projection
                const loadVal = data.load || 0;
                const loadForecast = document.getElementById('route-load-forecast');
                const peakPct = document.getElementById('route-peak-pct');
                const loadBar = document.getElementById('load-bar');

                loadForecast.innerText = `${loadVal}% Capacity Utilization`;
                peakPct.innerText = `${data.peak_intensity || 0}% PK`;
                loadBar.style.width = `${loadVal}%`;

                // Color bar based on load
                loadBar.className = 'h-full transition-all duration-1000 ' + 
                                  (loadVal > 70 ? 'bg-red-500' : (loadVal > 40 ? 'bg-amber-500' : 'bg-emerald-500'));

                lastCalculatedFare = data.fare;

                const seq = document.getElementById('route-seq'); seq.innerHTML = '';

                // Add Metrics Grid
                const metricsDiv = document.createElement('div');
                metricsDiv.className = "grid grid-cols-2 gap-4 mb-8 bg-slate-50 p-6 rounded-[24px] border border-slate-100";
                metricsDiv.innerHTML = `
                    <div class="flex flex-col"><span class="text-[9px] font-black uppercase text-slate-400 tracking-widest mb-1">Peak Intensity</span><span class="text-[13px] font-bold text-slate-700">${data.metrics.peak}</span></div>
                    <div class="flex flex-col"><span class="text-[9px] font-black uppercase text-slate-400 tracking-widest mb-1">Vector Bias</span><span class="text-[13px] font-bold text-slate-700">${data.metrics.it_hub}</span></div>
                `;
                seq.appendChild(metricsDiv);

                // Add Guides (Interchanges)
                if (data.guides && data.guides.length > 0) {
                    const guideHeader = document.createElement('h5');
                    guideHeader.className = "text-[10px] font-black text-indigo-600 uppercase tracking-[0.2em] mb-6 flex items-center gap-2";
                    guideHeader.innerHTML = `<i data-lucide="shuffle" size="12"></i> Critical Interchange Vectors`;
                    seq.appendChild(guideHeader);

                    data.guides.forEach((g, gIdx) => {
                        const gDiv = document.createElement('div');
                        gDiv.className = "bg-slate-900 border border-slate-800 p-8 rounded-[40px] shadow-2xl mb-10 relative overflow-hidden group border-l-[12px] border-l-blue-500 animate-in slide-in-from-right duration-700";

                        let connectionsHtml = "";
                        if (g.connections && g.connections.length > 0) {
                            connectionsHtml = `
                                <div class="mt-8 pt-8 border-t border-white/5">
                                    <p class="text-[9px] font-black text-blue-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                                        <i data-lucide="radio" size="12" class="animate-pulse"></i> Next Connections at ${g.station}
                                    </p>
                                    <div class="space-y-3">
                                        ${g.connections.map(c => `
                                            <div class="flex justify-between items-center bg-white/5 p-4 rounded-2xl border border-white/5 hover:bg-white/10 transition-colors">
                                                <div class="flex items-center gap-3">
                                                    <div class="w-1 h-3 rounded-full ${c.line === 'Red' ? 'bg-red-500' : c.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500'}"></div>
                                                    <div>
                                                        <p class="text-xs font-black text-white">${c.final_stop}</p>
                                                        <p class="text-[8px] font-bold text-white/40 uppercase tracking-tighter">Line: ${c.line} | Plat ${c.platform}</p>
                                                    </div>
                                                </div>
                                                <div class="text-right">
                                                    <p class="text-[12px] font-black text-blue-400 tabular-nums">${c.arrival_time_12}</p>
                                                    <p class="text-[8px] font-black text-white/30 uppercase tracking-widest">${c.wait_mins}m wait</p>
                                                </div>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            `;
                        }

                        gDiv.innerHTML = `
                            <div class="absolute -right-10 -top-10 w-40 h-40 bg-blue-500/10 rounded-full blur-3xl group-hover:bg-blue-500/20 transition-all"></div>
                            <div class="flex items-start gap-8 relative z-10">
                                <div class="flex flex-col items-center gap-4">
                                    <div class="w-16 h-16 rounded-3xl bg-white/5 border border-white/10 flex items-center justify-center shrink-0 shadow-inner">
                                        <i data-lucide="shuffle" size="28" class="text-blue-400"></i>
                                    </div>
                                    <div class="h-10 w-px bg-gradient-to-b from-blue-500/50 to-transparent"></div>
                                </div>
                                <div class="flex-1">
                                    <div class="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-6 gap-4">
                                        <div>
                                            <span class="px-3 py-1 bg-blue-500/20 text-blue-400 text-[9px] font-black uppercase tracking-[0.2em] rounded-lg border border-blue-500/20 mb-2 inline-block">Interchange Step ${gIdx + 1}</span>
                                            <h6 class="text-2xl font-black text-white tracking-tighter">${g.station}</h6>
                                        </div>
                                        <div class="flex items-center gap-3 bg-emerald-500/10 border border-emerald-500/20 px-5 py-3 rounded-2xl">
                                            <div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                                            <div class="flex flex-col">
                                                <span class="text-[8px] font-black text-emerald-500/60 uppercase tracking-widest">Reaching @ ${g.reaching_at}</span>
                                                <span class="text-xs font-black text-white uppercase tracking-wider">Board Platform ${g.platform}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="bg-white/5 p-6 rounded-3xl border border-white/5 relative group/inner hover:bg-white/[0.07] transition-all mb-4">
                                        <i data-lucide="info" class="absolute right-6 top-6 text-white/10" size="20"></i>
                                        <p class="text-[14px] lg:text-[16px] font-bold text-slate-200 leading-relaxed italic pr-8">
                                            "${g.text}"
                                        </p>
                                    </div>
                                    ${connectionsHtml}
                                </div>
                            </div>
                        `;
                        seq.appendChild(gDiv);
                    });
                }

                // Path Sequence
                const pathHeader = document.createElement('h5');
                pathHeader.className = "text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-8 mt-12";
                pathHeader.innerText = "Network Vector Stream";
                seq.appendChild(pathHeader);

                const streamWrap = document.createElement('div');
                streamWrap.className = "relative pl-10 border-l-2 border-slate-100 space-y-12 ml-4 mb-10";
                data.sequence.forEach((s) => {
                    const step = document.createElement('div');
                    step.className = 'relative flex items-center justify-between gap-6';
                    step.innerHTML = `
                        <div class="absolute -left-[49px] w-4 h-4 rounded-full border-4 border-white shadow-lg ${s.line === 'Red' ? 'bg-red-500' : s.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500'}"></div>
                        <div class="flex-1">
                            <p class="font-black text-slate-800 text-lg tracking-tight">${s.name}</p>
                            <p class="text-[9px] font-bold text-slate-400 uppercase tracking-widest">${s.line} Line Segment</p>
                        </div>
                        <div class="text-right">
                            <p class="text-[14px] font-black text-slate-900 tabular-nums">${s.reaching_at}</p>
                            <p class="text-[8px] font-bold text-slate-400 uppercase tracking-widest">Arrival</p>
                        </div>`;
                    streamWrap.appendChild(step);
                });
                seq.appendChild(streamWrap);

                const sched = document.getElementById('schedule-list'); sched.innerHTML = '';
                data.upcoming_hour.forEach(u => {
                    const div = document.createElement('div'); 
                    div.className = 'flex justify-between items-center bg-white p-6 rounded-3xl border border-slate-50 shadow-sm hover:shadow-md transition-all';
                    div.innerHTML = `
                        <div class="flex items-center gap-5">
                            <div class="w-12 h-12 rounded-2xl bg-slate-50 flex flex-col items-center justify-center border border-slate-100">
                                <span class="text-[8px] font-black text-slate-400 uppercase">PLAT</span>
                                <span class="text-[14px] font-black text-slate-900">${u.platform}</span>
                            </div>
                            <div>
                                <span class="text-sm font-black text-slate-700 block">${u.final_stop}</span>
                                <div class="flex items-center gap-2">
                                    <span class="px-1.5 py-0.5 rounded text-[7px] font-black uppercase tracking-tighter text-white ${u.line === 'Red' ? 'bg-red-500' : u.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500'}">${u.line} LINE</span>
                                    <span class="text-[8px] font-black text-slate-400 uppercase tracking-widest">Arrives: ${u.arrival_time}</span>
                                </div>
                            </div>
                        </div>
                        <div class="text-right">
                            <span class="text-[9px] font-black text-slate-400 uppercase tracking-widest block mb-1">Reach Dest</span>
                            <p class="text-[15px] font-black text-blue-600 tabular-nums">${u.est_reach || '--:--'}</p>
                            <p class="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mt-1">in ${u.eta}m</p>
                        </div>`;
                    sched.appendChild(div);
                });

                lucide.createIcons();
                document.getElementById('route-output').scrollIntoView({ behavior: 'smooth' });

            } catch (e) {
                console.error(e);
            } finally {
                btn.disabled = false;
                btnText.classList.remove('opacity-0');
                loader.classList.add('hidden');
            }
        }

        // Initialize pickers with grouped matrix logic
        function initPickers() {
            const lines = ['Red', 'Blue', 'Green'];
            const itHubs = ['Hitech City', 'Madhapur', 'Raidurg'];
            const interchanges = ['Ameerpet', 'MGBS', 'Parade Ground'];
            const lineIcons = { 'Red': '🔴', 'Blue': '🔵', 'Green': '🟢' };

            ['start-st', 'end-st'].forEach(id => {
                const select = document.getElementById(id);
                if (!select) return;
                select.innerHTML = '<option value="" disabled selected>Explore Matrix Access...</option>';

                lines.forEach(line => {
                    const group = document.createElement('optgroup');
                    group.label = `${lineIcons[line]} ${line} Line Network`;

                    stations.filter(s => s.line === line).sort((a,b) => a.name.localeCompare(b.name)).forEach(st => {
                        const opt = document.createElement('option');
                        opt.value = st.id;
                        let suffix = '';
                        if (itHubs.some(hub => st.name.includes(hub))) suffix = '  (IT HUB)';
                        if (interchanges.includes(st.name)) suffix = ' 🔄 (INTERCHANGE)';
                        opt.innerText = `${lineIcons[line]} ${st.name}${suffix}`;
                        group.appendChild(opt);
                    });
                    select.appendChild(group);
                });
            });
        }

        async function updateLiveTrains() {
            try {
                const res = await fetchWithSim('/api/live-map');
                const data = await res.json();

                // Dynamic Ticker Updates
                const ticker = document.getElementById('neural-ticker');
                if (ticker && data.trains) {
                    const messages = [
                        `<span>SYS_MSG</span> NEURAL MATRIX STABILIZED. ${data.trains.length} TRAINS IN VECTOR STREAM.`,
                        `<span>AI_LOAD</span> REAL-TIME DENSITY OPTIMIZATION ACTIVE.`,
                        `<span>METRIC</span> NETWORK LATENCY: 14MS.`,
                        `<span>SATELLITE</span> HIGH EQUATORIAL ACCURACY ATTAINED.`
                    ];
                    // Randomly update one item occasionally or keep it interesting
                    if (Math.random() > 0.7) {
                        const items = ticker.querySelectorAll('.ticker-item');
                        if (items.length > 0) {
                            const idx = Math.floor(Math.random() * items.length);
                            items[idx].innerHTML = messages[Math.floor(Math.random() * messages.length)];
                        }
                    }
                }

                const g = document.getElementById('map-trains');
                const existingTrains = Array.from(g.querySelectorAll('.train-icon'));
                const seenIds = new Set();

                data.trains.forEach(t => {
                    seenIds.add(t.trip_id);
                    const s1 = stations.find(s => s.id === t.from_id);
                    const s2 = stations.find(s => s.id === t.to_id);
                    if(!s1 || !s2) return;

                    const curX = s1.x + (s2.x - s1.x) * t.progress;
                    const curY = s1.y + (s2.y - s1.y) * t.progress;

                    // Calculate rotation for directed icon
                    const angle = Math.atan2(s2.y - s1.y, s2.x - s1.x) * 180 / Math.PI;

                    let train = g.querySelector(`[data-trip-id="${t.trip_id}"]`);
                    if(!train) {
                        train = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                        train.setAttribute('class', 'train-icon');
                        train.setAttribute('data-trip-id', t.trip_id);
                        train.style.transition = 'transform 2.1s linear';

                        const color = t.line === 'Red' ? '#ef4444' : t.line === 'Blue' ? '#3b82f6' : '#22c55e';

                        const outer = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                        outer.setAttribute('r', 16);
                        outer.setAttribute('fill', color);
                        outer.setAttribute('class', 'animate-pulse opacity-20');

                        // Directed Train Icon (Arrow shape)
                        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                        icon.setAttribute('d', 'M -8,-5 L 8,0 L -8,5 Z'); // Triangle pointing right
                        icon.setAttribute('fill', color);
                        icon.setAttribute('stroke', '#fff');
                        icon.setAttribute('stroke-width', '1');
                        icon.setAttribute('class', 'train-shape');

                        train.appendChild(outer);
                        train.appendChild(icon);
                        g.appendChild(train);

                        train.style.transition = 'none';
                        train.setAttribute('transform', `translate(${curX}, ${curY}) rotate(${angle})`);
                        setTimeout(() => train.style.transition = 'transform 2.1s linear', 50);
                    } else {
                        train.setAttribute('transform', `translate(${curX}, ${curY}) rotate(${angle})`);
                    }
                });

                existingTrains.forEach(t => {
                    if(!seenIds.has(t.getAttribute('data-trip-id'))) t.remove();
                });
            } catch (e) {}
        }
        setInterval(updateLiveTrains, 2000);

        window.onload = () => {
            setupMap(); 
            initGeo(); 
            initPickers(); 
            updateLiveTrains(); 
            loadFeedback(); 
            renderTickets();
            loadSavedVectors();
            activeUpdateInterval = setInterval(updateLiveTrains, 15000); 
        };
    </script>
</body>
</html>
"""

# Initial System Boot Sync
ensure_gtfs()
generate_ai_dataset()

if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')
