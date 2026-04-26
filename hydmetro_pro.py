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
    """Returns current time in India Standard Time (UTC+5:30)"""
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


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
    {'id': 'R1', 'name': 'Miyapur', 'line': 'Red', 'x': 50, 'y': 50, 'lat': 17.4968, 'lng': 78.3498,
     'amenities': ['Parking', 'Restrooms', 'Food Court', 'ATM']},
    {'id': 'R2', 'name': 'JNTU', 'line': 'Red', 'x': 80, 'y': 70, 'lat': 17.4912, 'lng': 78.3582},
    {'id': 'R3', 'name': 'KPHB', 'line': 'Red', 'x': 110, 'y': 90, 'lat': 17.4842, 'lng': 78.3888},
    {'id': 'R4', 'name': 'Kukatpally', 'line': 'Red', 'x': 140, 'y': 110, 'lat': 17.4854, 'lng': 78.3975},
    {'id': 'R5', 'name': 'Balanagar', 'line': 'Red', 'x': 170, 'y': 130, 'lat': 17.4764, 'lng': 78.4239},
    {'id': 'R6', 'name': 'Moosapet', 'line': 'Red', 'x': 200, 'y': 150, 'lat': 17.4721, 'lng': 78.4284},
    {'id': 'R7', 'name': 'Bharat Nagar', 'line': 'Red', 'x': 230, 'y': 170, 'lat': 17.4646, 'lng': 78.4357},
    {'id': 'R8', 'name': 'Erragadda', 'line': 'Red', 'x': 260, 'y': 190, 'lat': 17.4572, 'lng': 78.4412},
    {'id': 'R9', 'name': 'ESI Hospital', 'line': 'Red', 'x': 290, 'y': 210, 'lat': 17.4517, 'lng': 78.4457},
    {'id': 'R10', 'name': 'S.R. Nagar', 'line': 'Red', 'x': 320, 'y': 230, 'lat': 17.4442, 'lng': 78.4484},
    {'id': 'R11', 'name': 'Ameerpet', 'line': 'Red', 'x': 350, 'y': 250, 'lat': 17.4346, 'lng': 78.4484,
     'amenities': ['Interchange', 'Shopping', 'Food Court', 'Restrooms', 'ATM']},
    {'id': 'R12', 'name': 'Panjagutta', 'line': 'Red', 'x': 380, 'y': 270, 'lat': 17.4258, 'lng': 78.4522},
    {'id': 'R13', 'name': 'Irrum Manzil', 'line': 'Red', 'x': 410, 'y': 290, 'lat': 17.4184, 'lng': 78.4557},
    {'id': 'R14', 'name': 'Khairatabad', 'line': 'Red', 'x': 440, 'y': 310, 'lat': 17.4101, 'lng': 78.4611},
    {'id': 'R15', 'name': 'Lakdikapul', 'line': 'Red', 'x': 470, 'y': 330, 'lat': 17.4024, 'lng': 78.4657},
    {'id': 'R16', 'name': 'Assembly', 'line': 'Red', 'x': 500, 'y': 350, 'lat': 17.3984, 'lng': 78.4723},
    {'id': 'R17', 'name': 'Nampally', 'line': 'Red', 'x': 530, 'y': 370, 'lat': 17.3921, 'lng': 78.4757},
    {'id': 'R18', 'name': 'Gandhi Bhavan', 'line': 'Red', 'x': 560, 'y': 390, 'lat': 17.3872, 'lng': 78.4784},
    {'id': 'R19', 'name': 'OMC', 'line': 'Red', 'x': 590, 'y': 410, 'lat': 17.3824, 'lng': 78.4812},
    {'id': 'R20', 'name': 'MGBS', 'line': 'Red', 'x': 620, 'y': 430, 'lat': 17.3788, 'lng': 78.4820,
     'amenities': ['Bus Station Link', 'Ticket Counters', 'Restrooms', 'ATM']},
    {'id': 'R21', 'name': 'Malakpet', 'line': 'Red', 'x': 650, 'y': 450, 'lat': 17.3746, 'lng': 78.4957},
    {'id': 'R22', 'name': 'New Market', 'line': 'Red', 'x': 680, 'y': 470, 'lat': 17.3712, 'lng': 78.5084},
    {'id': 'R23', 'name': 'Musarambagh', 'line': 'Red', 'x': 710, 'y': 490, 'lat': 17.3684, 'lng': 78.5212},
    {'id': 'R24', 'name': 'Dilsukhnagar', 'line': 'Red', 'x': 740, 'y': 510, 'lat': 17.3657, 'lng': 78.5357},
    {'id': 'R25', 'name': 'Chaitanyapuri', 'line': 'Red', 'x': 770, 'y': 530, 'lat': 17.3612, 'lng': 78.5484},
    {'id': 'R26', 'name': 'Victoria Memorial', 'line': 'Red', 'x': 800, 'y': 550, 'lat': 17.3557, 'lng': 78.5512},
    {'id': 'R27', 'name': 'LB Nagar', 'line': 'Red', 'x': 830, 'y': 570, 'lat': 17.3458, 'lng': 78.5524},

    # BLUE LINE
    {'id': 'B1', 'name': 'Raidurg', 'line': 'Blue', 'x': 50, 'y': 350, 'lat': 17.4429, 'lng': 78.3750},
    {'id': 'B2', 'name': 'Hitech City', 'line': 'Blue', 'x': 80, 'y': 330, 'lat': 17.4474, 'lng': 78.3762,
     'amenities': ['Parking', 'Restrooms', 'WIFI', 'ATM']},
    {'id': 'B3', 'name': 'Durgam Cheruvu', 'line': 'Blue', 'x': 110, 'y': 310, 'lat': 17.4398, 'lng': 78.3857},
    {'id': 'B4', 'name': 'Madhapur', 'line': 'Blue', 'x': 140, 'y': 290, 'lat': 17.4357, 'lng': 78.3984},
    {'id': 'B5', 'name': 'Jubilee Hills CP', 'line': 'Blue', 'x': 170, 'y': 270, 'lat': 17.4324, 'lng': 78.4112},
    {'id': 'B6', 'name': 'Road No 5', 'line': 'Blue', 'x': 200, 'y': 250, 'lat': 17.4284, 'lng': 78.4239},
    {'id': 'B7', 'name': 'Yousufguda', 'line': 'Blue', 'x': 230, 'y': 230, 'lat': 17.4246, 'lng': 78.4357},
    {'id': 'B8', 'name': 'Ameerpet', 'line': 'Blue', 'x': 350, 'y': 250, 'lat': 17.4346, 'lng': 78.4484,
     'name_alias': 'Ameerpet'},
    {'id': 'B9', 'name': 'Begumpet', 'line': 'Blue', 'x': 400, 'y': 230, 'lat': 17.4398, 'lng': 78.4612},
    {'id': 'B10', 'name': 'Prakash Nagar', 'line': 'Blue', 'x': 450, 'y': 210, 'lat': 17.4457, 'lng': 78.4724},
    {'id': 'B11', 'name': 'Rasoolpura', 'line': 'Blue', 'x': 500, 'y': 190, 'lat': 17.4512, 'lng': 78.4851},
    {'id': 'B12', 'name': 'Paradise', 'line': 'Blue', 'x': 550, 'y': 170, 'lat': 17.4568, 'lng': 78.4972},
    {'id': 'B13', 'name': 'Parade Ground', 'line': 'Blue', 'x': 600, 'y': 150, 'lat': 17.4612, 'lng': 78.5084},
    {'id': 'B14', 'name': 'Sec-bad East', 'line': 'Blue', 'x': 650, 'y': 130, 'lat': 17.4546, 'lng': 78.5212},
    {'id': 'B15', 'name': 'Mettuguda', 'line': 'Blue', 'x': 700, 'y': 110, 'lat': 17.4484, 'lng': 78.5342},
    {'id': 'B16', 'name': 'Tarnaka', 'line': 'Blue', 'x': 750, 'y': 90, 'lat': 17.4357, 'lng': 78.5472},
    {'id': 'B17', 'name': 'Habsiguda', 'line': 'Blue', 'x': 800, 'y': 70, 'lat': 17.4212, 'lng': 78.5584},
    {'id': 'B18', 'name': 'NGRI', 'line': 'Blue', 'x': 850, 'y': 50, 'lat': 17.4084, 'lng': 78.5684},
    {'id': 'B19', 'name': 'Stadium', 'line': 'Blue', 'x': 900, 'y': 30, 'lat': 17.4021, 'lng': 78.5712},
    {'id': 'B20', 'name': 'Uppal', 'line': 'Blue', 'x': 950, 'y': 10, 'lat': 17.3984, 'lng': 78.5684},
    {'id': 'B21', 'name': 'Nagole', 'line': 'Blue', 'x': 1000, 'y': -10, 'lat': 17.3941, 'lng': 78.5668},

    # GREEN LINE
    {'id': 'G1', 'name': 'JBS', 'line': 'Green', 'x': 450, 'y': 50, 'lat': 17.4439, 'lng': 78.4988},
    {'id': 'G2', 'name': 'Sec-bad West', 'line': 'Green', 'x': 475, 'y': 100, 'lat': 17.4482, 'lng': 78.5012},
    {'id': 'G3', 'name': 'Parade Ground', 'line': 'Green', 'x': 600, 'y': 150, 'lat': 17.4612, 'lng': 78.5084,
     'name_alias': 'Parade Ground'},
    {'id': 'G4', 'name': 'Gandhi Hospital', 'line': 'Green', 'x': 600, 'y': 200, 'lat': 17.4342, 'lng': 78.5112},
    {'id': 'G5', 'name': 'Musheerabad', 'line': 'Green', 'x': 600, 'y': 250, 'lat': 17.4212, 'lng': 78.5157},
    {'id': 'G6', 'name': 'RTC X Roads', 'line': 'Green', 'x': 600, 'y': 300, 'lat': 17.4084, 'lng': 78.5184},
    {'id': 'G7', 'name': 'Chikkadpally', 'line': 'Green', 'x': 600, 'y': 350, 'lat': 17.3984, 'lng': 78.5212},
    {'id': 'G8', 'name': 'Narayanaguda', 'line': 'Green', 'x': 610, 'y': 400, 'lat': 17.3884, 'lng': 78.5184},
    {'id': 'G9', 'name': 'MGBS', 'line': 'Green', 'x': 620, 'y': 430, 'lat': 17.3788, 'lng': 78.4820,
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


def ensure_gtfs(force=False):
    """Generates a high-frequency, dynamic GTFS simulation."""
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


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, ALL_STATIONS=STATIONS_LIST)


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

    ensure_gtfs()
    now = get_ist_now()
    now_str = now.strftime('%H:%M:%S')
    one_hour_later = now + timedelta(hours=1)
    oh_str = one_hour_later.strftime('%H:%M:%S')

    upcoming = []
    with open(GTFS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['station_id'] in matching_ids and now_str < row['arrival_time'] < oh_str:
                # Calculate ETA countdown
                ah, am, as_ = map(int, row['arrival_time'].split(':'))
                arrival_dt = now.replace(hour=ah, minute=am, second=as_, microsecond=0)
                diff = (arrival_dt - now).total_seconds()
                if diff < 0: continue

                m, s = divmod(int(diff), 60)
                row['eta'] = f"{m:02d}:{s:02d}"
                upcoming.append(row)

    # Sort upcoming by time
    upcoming.sort(key=lambda x: x['arrival_time'])

    weather = get_live_weather()
    is_weekend = now.weekday() >= 5
    load_val, load_label = predict_load_ai(name, now.hour, is_weekend=is_weekend, weather=weather)

    return jsonify({
        'station': nearest,
        'upcoming': upcoming,
        'load_val': load_val,
        'load_label': load_label,
        'weather': weather,
        'greeting': "Good Morning" if 5 <= now.hour < 12 else "Good Afternoon" if 12 <= now.hour < 17 else "Good Evening"
    })


@app.route('/api/plan', methods=['POST'])
def api_plan():
    data = request.json
    start_id, end_id = data['from'], data['to']
    now = get_ist_now()

    # BFS find path
    queue = [(start_id, [start_id])]
    visited = {start_id}
    path = []
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
        c_name = next(s['name'] for s in STATIONS_LIST if s['id'] == curr)
        for s in STATIONS_LIST:
            if s['name'] == c_name and s['id'] != curr: neighbors.append(s['id'])
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                queue.append((n, p + [n]))

    sequence = [next(s for s in STATIONS_LIST if s['id'] == sid) for sid in path]

    # Synchronize Arrival Time with GTFS schedule
    ensure_gtfs()
    gtfs_arrival_time = None
    try:
        now_str = now.strftime('%H:%M:%S')
        with open(GTFS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            trips = list(reader)

        # Find the next available trip at start station
        start_id = sequence[0]['id']
        next_id = sequence[1]['id'] if len(sequence) > 1 else None

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
                    break
    except Exception as e:
        print(f"GTFS Planner Sync Error: {e}")

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
    duration = len(sequence) * 2

    # Use GTFS time if found, fallback to calculation
    if gtfs_arrival_time:
        arrival_at_destination = gtfs_arrival_time
    else:
        arrival_at_destination = (now + timedelta(minutes=duration)).strftime('%H:%M:%S')

    # AI Recommendation logic
    start_station_name = sequence[0]['name']
    weather = get_live_weather()
    is_weekend = now.weekday() >= 5
    load_val, _ = predict_load_ai(start_station_name, now.hour, is_weekend=is_weekend, weather=weather)

    # INTERCHANGE & GUIDE LOGIC (Mirrored from React)
    interchanges = []
    guides = []
    for i in range(len(path) - 1):
        s1 = next(s for s in STATIONS_LIST if s['id'] == path[i])
        s2 = next(s for s in STATIONS_LIST if s['id'] == path[i + 1])
        name1 = s1.get('name_alias', s1['name'])
        name2 = s2.get('name_alias', s2['name'])

        if name1 == name2 and s1['id'] != s2['id']:
            interchanges.append(name1)
            # Find next stop to determine platform
            next_sid = path[i + 2] if i + 2 < len(path) else None
            platform = "?"
            guide = f"Transfer at {name1}"
            if next_sid:
                next_s = next(s for s in STATIONS_LIST if s['id'] == next_sid)
                if name1 == 'Ameerpet':
                    if next_s['line'] == 'Red':
                        idx = int(next_s['id'].replace('R', ''))
                        platform = "1 (Towards LB Nagar)" if idx > 11 else "2 (Towards Miyapur)"
                        if s1['line'] == 'Blue': guide = "Go DOWN from Blue Line (Level 1) to Red Line"
                    elif next_s['line'] == 'Blue':
                        idx = int(next_s['id'].replace('B', ''))
                        platform = "3 (Towards Nagole)" if idx > 8 else "4 (Towards Raidurg)"
                        if s1['line'] == 'Red': guide = "Go UP from Red Line level to Blue Line (Level 1)"
                elif name1 == 'MGBS':
                    if next_s['line'] == 'Red':
                        idx = int(next_s['id'].replace('R', ''))
                        platform = "1 (Towards LB Nagar)" if idx > 20 else "2 (Towards Miyapur)"
                        if s1['line'] == 'Green': guide = "Go UP from Green Line to reach Red Line Platform"
                    elif next_s['line'] == 'Green':
                        platform = "3 (Towards JBS Parade Grounds)"
                        if s1['line'] == 'Red': guide = "Go DOWN from Red Line to reach Green Line Platform"
            guides.append({'station': name1, 'platform': platform, 'text': guide})

    # PROJECTION METRICS
    is_peak = "Peak Hour" if (7 <= now.hour <= 10 or 17 <= now.hour <= 21) else "Off-Peak"
    is_it_hub = "High" if any(
        n in [s['name'] for s in sequence] for n in ['Hitech City', 'Raidurg', 'Madhapur']) else "Normal"
    recommendation = "Optimal conditions. Good to travel. Low crowd density detected." if load_val == "Low" else \
        "Fair volume. Seat will be there for your journey." if load_val == "Medium" else \
            "Moderate volume. Rush but manageable. Proceed with caution." if load_val == "M-High" else \
                "Peak congestion. High rush detected. AI suggests waiting for off-peak dip."

    # User Request: Upcoming trains for next 1 hour from source
    ensure_gtfs()
    one_hour_later = now + timedelta(hours=1)
    now_str = now.strftime('%H:%M:%S')
    oh_str = one_hour_later.strftime('%H:%M:%S')

    upcoming_hour = []
    with open(GTFS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['station_id'] == start_id and now_str < row['arrival_time'] < oh_str:
                # Calculate ETA
                try:
                    ah, am, as_ = map(int, row['arrival_time'].split(':'))
                    arrival_dt = now.replace(hour=ah, minute=am, second=as_, microsecond=0)
                    diff = (arrival_dt - now).total_seconds()
                    row['eta'] = f"{int(diff // 60):02d}:{int(diff % 60):02d}"
                except:
                    row['eta'] = "Soon"
                upcoming_hour.append(row)
                if len(upcoming_hour) >= 8: break

    return jsonify({
        'sequence': sequence,
        'upcoming_hour': upcoming_hour,
        'duration': duration,
        'arrival_at_dest': arrival_at_destination,
        'total_stops': len(sequence),
        'total_km': round(total_km, 2),
        'fare': calculated_fare,
        'recommendation': recommendation,
        'guides': guides,
        'metrics': {
            'peak': is_peak,
            'it_hub': is_it_hub,
            'fare_stable': True
        }
    })


@app.route('/api/live-map')
def api_live_map():
    ensure_gtfs()
    now_dt = get_ist_now()
    now_str = now_dt.strftime('%H:%M:%S')

    active_trains = []

    # Read trips and group by trip_id
    trips_by_id = {}
    with open(GTFS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
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

        .sidebar { 
            height: 100vh; width: 260px; position: fixed; 
            background: rgba(255, 255, 255, 0.8); 
            backdrop-filter: blur(40px);
            border-right: 1px solid #f1f5f9; z-index: 50; 
            display: flex; flex-direction: column; 
            transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .sidebar.collapsed {
            transform: translateX(-100%);
        }
        .main { 
            margin-left: 260px; 
            padding: 48px; 
            min-height: 100vh; 
            transition: margin-left 0.4s cubic-bezier(0.4, 0, 0.2, 1); 
        }
        .main.full-width {
            margin-left: 0;
        }

        .sidebar-toggle {
            position: fixed;
            top: 20px;
            left: 280px;
            z-index: 1000;
            background: #0f172a;
            color: white;
            border: none;
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            display: none; /* Desktop only */
        }
        .sidebar-toggle:hover {
            transform: scale(1.05);
            background: #1e293b;
        }
        .sidebar-toggle.sidebar-collapsed {
            left: 20px;
        }
        @media (min-width: 1025px) {
            .sidebar-toggle { display: flex; }
        }

        @media (max-width: 1024px) {
            .sidebar { display: none; }
            .sidebar.collapsed { transform: none; }
            .main { margin-left: 0; padding: 24px; padding-bottom: 120px; }
            .mobile-nav { display: flex; }
        }

        .mobile-nav { 
            display: none; position: fixed; bottom: 20px; left: 20px; right: 20px; 
            background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(20px);
            border-radius: 20px; z-index: 100;
            padding: 10px; justify-content: space-around;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        }

        .nav-link { 
            display: flex; align-items: center; gap: 12px; padding: 12px 20px; 
            border-radius: 12px; color: #64748b; font-weight: 700; 
            transition: all 0.3s; cursor: pointer; text-transform: uppercase; 
            font-size: 10px; letter-spacing: 0.05em; 
        }
        .nav-link.active { background: #0f172a; color: white; }

        .tab-content { display: none; }
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

        #network-svg { background: transparent; }
    </style>
</head>
<body>
    <div class=\"sidebar p-10 flex flex-col justify-between shadow-2xl shadow-slate-200/50\">
        <div>
            <div class=\"flex items-center gap-4 mb-16\">
                <div class=\"w-14 h-14 bg-slate-900 rounded-2xl flex items-center justify-center text-white shadow-2xl shadow-slate-400/20\"><i data-lucide=\"train-front\"></i></div>
                <div><h1 class=\"text-2xl font-black text-slate-900 tracking-tighter\">HydMetro</h1><p class=\"text-[9px] font-black text-blue-600 uppercase tracking-widest leading-none\">Intelligence Engine</p></div>
            </div>
            <nav class=\"space-y-3\">
                <div onclick=\"showTab('home')\" id=\"btn-home\" class=\"nav-link active\"><i data-lucide=\"layout-grid\"></i> Dashboard</div>
                <div onclick=\"showTab('map')\" id=\"btn-map\" class=\"nav-link\"><i data-lucide=\"globe\"></i> Network Flux</div>
                <div onclick=\"showTab('routes')\" id=\"btn-routes\" class=\"nav-link\"><i data-lucide=\"route\"></i> Path Architect</div>
            </nav>
        </div>
        <div class=\"bg-indigo-50/50 p-6 rounded-[32px] border border-indigo-100\">
            <div class=\"flex items-center gap-3 mb-3\"><div class=\"w-2.5 h-2.5 bg-emerald-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)]\"></div><span class=\"text-[10px] font-black text-indigo-900 uppercase tracking-widest\">System Ready</span></div>
            <p class=\"text-[11px] text-slate-500 font-bold leading-relaxed\">Neural tracking active. Coordinates synchronized with satellite clock.</p>
        </div>
    </div>

    <div class=\"mobile-nav\">
        <div onclick=\"showTab('home')\" class=\"mobile-link active\" id=\"mob-home\"><i data-lucide=\"layout-grid\"></i><span>Home</span></div>
        <div onclick=\"showTab('map')\" class=\"mobile-link\" id=\"mob-map\"><i data-lucide=\"globe\"></i><span>Map</span></div>
        <div onclick=\"showTab('routes')\" class=\"mobile-link\" id=\"mob-routes\"><i data-lucide=\"route\"></i><span>Planner</span></div>
    </div>

    <div class=\"main\" id=\"main-content\">
        <button onclick=\"toggleSidebar()\" class=\"sidebar-toggle\" id=\"sidebar-toggle\">
            <i data-lucide=\"panel-left-close\"></i>
        </button>
        <!-- HOME HUB -->
        <div id=\"tab-home\" class=\"tab-content active\">
            <header class=\"flex flex-col lg:flex-row lg:justify-between lg:items-start gap-8 mb-16\">
                <div>
                    <h1 class=\"text-2xl font-black lg:hidden mb-4 flex items-center gap-3\"><i data-lucide=\"train-front\"></i> HydMetro</h1>
                    <h2 id=\"greeting\" class=\"text-4xl lg:text-5xl font-black text-slate-900 mb-2 tracking-tighter\">Good Day!</h2>
                    <p id=\"env-msg\" class=\"text-slate-400 font-bold max-w-sm leading-relaxed uppercase text-[10px] tracking-widest\">Neural processing active. Enjoy your commute across the network.</p>
                </div>
                <div class=\"glass-card py-4 px-8 flex flex-row lg:flex-col items-center lg:items-end justify-between lg:justify-center gap-6 border-slate-200\">
                    <div class=\"flex items-baseline gap-2\">
                        <span id=\"clock\" class=\"text-3xl lg:text-4xl font-black text-slate-900 tabular-nums tracking-tighter\">00:00:00</span>
                        <span id=\"ampm\" class=\"text-xs font-black text-slate-400 uppercase\">AM</span>
                    </div>
                    <span id=\"date\" class=\"text-[9px] lg:text-[10px] font-black text-blue-600 uppercase tracking-[0.2em]\">October 24, 2024</span>
                </div>
            </header>

            <div class=\"grid grid-cols-1 md:grid-cols-4 gap-8 mb-12\">
                <div class=\"glass-card flex items-center gap-6 border-slate-200\">
                    <div class=\"w-14 h-14 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center shrink-0\"><i data-lucide=\"map-pin\"></i></div>
                    <div class=\"overflow-hidden\"><p class=\"text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1\">Local Matrix</p><h3 id=\"near-name\" class=\"text-sm font-black text-slate-800 truncate\">Locating...</h3></div>
                </div>
                <div class=\"glass-card flex items-center gap-6 border-slate-200\">
                    <div class=\"w-14 h-14 bg-orange-50 text-orange-600 rounded-2xl flex items-center justify-center shrink-0\"><i data-lucide=\"waves\"></i></div>
                    <div>
                        <p class=\"text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1\">Atmosphere</p>
                        <h3 id=\"weather-val\" class=\"text-sm font-black text-slate-800\">--°C</h3>
                        <p id=\"weather-detail\" class=\"text-[8px] font-bold text-slate-400 uppercase\">H: --% | V: --km</p>
                    </div>
                </div>
                <div id=\"load-card\" class=\"glass-card flex items-center gap-6 border-slate-200 border-l-[6px] border-l-emerald-500\">
                    <div class=\"w-14 h-14 bg-emerald-50 text-emerald-600 rounded-2xl flex items-center justify-center shrink-0\"><i data-lucide=\"activity\"></i></div>
                    <div><p class=\"text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1\">Neural Load</p><h3 id=\"load-val\" class=\"text-sm font-black text-slate-800\">Analyzing...</h3></div>
                </div>
                <div class=\"glass-card flex items-center gap-6 border-slate-200\">
                    <div class=\"w-14 h-14 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center shrink-0\"><i data-lucide=\"zap\"></i></div>
                    <div><p class=\"text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1\">Flux Logic</p><h3 class=\"text-sm font-black text-slate-800 uppercase\">Optimal</h3></div>
                </div>
            </div>

            <div class=\"grid grid-cols-1 lg:grid-cols-3 gap-12 mb-16\">
                <div class=\"lg:col-span-2 glass-card p-0 overflow-hidden border-slate-200 shadow-2xl shadow-slate-200/40\">
                    <div class=\"p-10 border-b border-slate-100 flex justify-between items-center bg-slate-50/30\">
                        <div class=\"flex items-center gap-5\">
                            <i data-lucide=\"radio\" class=\"text-blue-600 animate-pulse\"></i>
                            <h3 class=\"text-[11px] font-black text-slate-900 uppercase tracking-[0.3em]\">Network Activity Stream</h3>
                            <select id=\"board-station-selector\" onchange=\"manualStationChange()\" class=\"text-[10px] font-black uppercase bg-white border border-slate-200 px-4 py-2 rounded-xl outline-none focus:ring-4 focus:ring-blue-500/10 cursor-pointer shadow-sm\">
                                <option value=\"\">Satellite Track...</option>
                            </select>
                        </div>
                        <span class=\"px-3 py-1 bg-slate-900 text-white text-[9px] font-black rounded-lg uppercase tracking-widest\">Real-time</span>
                    </div>
                    <div class=\"p-10\">
                        <table class=\"w-full text-left\">
                            <thead>
                                <tr class=\"text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] border-b border-slate-100\">
                                    <th class=\"pb-8\">Vector</th>
                                    <th class=\"pb-8\">Target Node</th>
                                    <th class=\"pb-8\">Terminal</th>
                                    <th class=\"pb-8 text-right\">Countdown (MM:SS)</th>
                                </tr>
                            </thead>
                            <tbody id=\"board-rows\" class=\"divide-y divide-slate-50\"></tbody>
                        </table>
                        <div id=\"board-loading\" class=\"py-24 text-center\">
                             <div class=\"w-10 h-10 border-4 border-slate-900 border-t-transparent rounded-full animate-spin mx-auto mb-4\"></div>
                             <p class=\"text-[10px] font-black uppercase tracking-widest text-slate-400\">Synchronizing Local Nodes...</p>
                        </div>
                    </div>
                </div>
                <div class=\"space-y-8\">
                    <div class=\"glass-card action-card bg-slate-900 text-white border-none p-10 overflow-hidden relative shadow-2xl shadow-slate-900/10\">
                        <div class=\"absolute -right-5 -top-5 w-40 h-40 bg-blue-500/20 rounded-full blur-[60px]\"></div>
                        <i data-lucide=\"ticket\" class=\"mb-8 opacity-40\" size=\"32\"></i>
                        <h4 class=\"text-xl font-black mb-3 relative z-10 tracking-tight\">Book Metro Ticket</h4>
                        <p class=\"text-xs text-white/50 mb-10 relative z-10 font-bold uppercase tracking-widest\">Instant checkout via preferred UPI gateway.</p>
                        <div class=\"space-y-4 relative z-10\">
                            <button onclick=\"toggleUPISelection()\" id=\"pay-btn-main\" class=\"w-full py-5 bg-white text-slate-900 rounded-2xl font-black text-[11px] uppercase tracking-widest shadow-xl flex items-center justify-center gap-3\">
                                <i data-lucide=\"smartphone\" size=\"14\"></i> Select Payment App
                            </button>
                            <div id=\"upi-selection\" class=\"hidden grid grid-cols-2 gap-3 animate-in fade-in slide-in-from-top-2 duration-300\">
                                <button onclick=\"payWithUPI('Google Pay')\" class=\"p-4 bg-white/10 hover:bg-white/20 rounded-xl flex flex-col items-center gap-2 transition-all border border-white/5\">
                                    <div class=\"w-6 h-6 bg-white rounded-md flex items-center justify-center p-1\"><i data-lucide=\"wallet\" class=\"text-slate-900\" size=\"14\"></i></div>
                                    <span class=\"text-[8px] font-black uppercase\">G-Pay</span>
                                </button>
                                <button onclick=\"payWithUPI('PhonePe')\" class=\"p-4 bg-white/10 hover:bg-white/20 rounded-xl flex flex-col items-center gap-2 transition-all border border-white/5\">
                                    <div class=\"w-6 h-6 bg-purple-500 rounded-md flex items-center justify-center p-1\"><i data-lucide=\"zap\" class=\"text-white\" size=\"14\"></i></div>
                                    <span class=\"text-[8px] font-black uppercase\">PhonePe</span>
                                </button>
                                <button onclick=\"payWithUPI('Paytm')\" class=\"p-4 bg-white/10 hover:bg-white/20 rounded-xl flex flex-col items-center gap-2 transition-all border border-white/5\">
                                    <div class=\"w-6 h-6 bg-sky-400 rounded-md flex items-center justify-center p-1\"><i data-lucide=\"credit-card\" class=\"text-white\" size=\"14\"></i></div>
                                    <span class=\"text-[8px] font-black uppercase\">Paytm</span>
                                </button>
                                <button onclick=\"payWithUPI('Others')\" class=\"p-4 bg-white/10 hover:bg-white/20 rounded-xl flex flex-col items-center gap-2 transition-all border border-white/5\">
                                    <div class=\"w-6 h-6 bg-slate-700 rounded-md flex items-center justify-center p-1\"><i data-lucide=\"qr-code\" class=\"text-white\" size=\"14\"></i></div>
                                    <span class=\"text-[8px] font-black uppercase\">Others</span>
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class=\"glass-card action-card border-none p-10 bg-white overflow-hidden relative shadow-2xl shadow-slate-200/50\">
                        <div class=\"absolute -right-5 -top-5 w-40 h-40 bg-indigo-50 rounded-full blur-[60px]\"></div>
                        <i data-lucide=\"wallet-minimal\" class=\"mb-8 text-indigo-600\" size=\"32\"></i>
                        <h4 class=\"text-xl font-black mb-3 relative z-10 tracking-tight text-slate-900\">Neural Connect</h4>
                        <p class=\"text-xs text-slate-400 mb-10 relative z-10 font-bold uppercase tracking-widest\">Recharge your smart matrix card via NFC interface.</p>
                        <button class=\"w-full py-5 bg-slate-900 text-white rounded-2xl font-black text-[11px] uppercase tracking-widest\">Sync Hardware</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- NETWORK MAP -->
        <div id=\"tab-map\" class=\"tab-content\">
            <h2 class=\"text-3xl font-black text-slate-900 mb-8 tracking-tight\">Dynamic Network Topology</h2>
            <div class=\"glass-card p-0 relative h-[700px] overflow-hidden bg-white border-2\">
                <svg id=\"network-svg\" viewBox=\"0 0 1100 700\" class=\"w-full h-full cursor-grab active:cursor-grabbing\">
                    <g id=\"map-lines\"></g>
                    <g id=\"map-stations\"></g>
                    <g id=\"map-trains\"></g>
                    <g id=\"map-user-pin\"></g>
                </svg>
                <div id=\"map-overlay\" class=\"absolute top-0 right-0 h-full w-full lg:w-[400px] translate-x-full z-20 transition-transform duration-500 ease-in-out bg-white shadow-[-20px_0_50px_-10px_rgba(0,0,0,0.1)] border-l border-slate-100\">
                    <div class=\"h-full flex flex-col p-6 lg:p-10 overflow-hidden relative\">
                        <div class=\"absolute top-6 right-6 lg:top-8 lg:right-8 z-20\">
                            <button onclick=\"closeOverlay()\" class=\"p-3 bg-slate-50 hover:bg-slate-100 text-slate-400 rounded-2xl transition-colors\">
                                <i data-lucide=\"x\" size=\"20\"></i>
                            </button>
                        </div>

                        <div class=\"mb-10 px-2 lg:px-0\">
                            <span id=\"ov-line\" class=\"px-3 py-1 text-[10px] font-black uppercase rounded-lg mb-4 inline-block\">LINE</span>
                            <h4 id=\"ov-name\" class=\"text-3xl lg:text-4xl font-black text-slate-900 leading-tight tracking-tighter\">Station Name</h4>
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
            <div class="flex items-center justify-between mb-10 text-slate-900 border-b pb-8 border-slate-200">
                <div>
                   <h2 class="text-4xl font-black tracking-tight mb-2">Neural Flux Architect</h2>
                   <p class="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2"><i data-lucide="shield-check" size="14" class="text-emerald-500"></i> Optimized for lowest congestion</p>
                </div>
                <div class="flex gap-4">
                    <div class="px-6 py-4 bg-white shadow-sm border border-slate-100 rounded-2xl flex flex-col items-center">
                        <span class="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1">Status</span>
                        <span class="text-[11px] font-black text-emerald-600 uppercase">Live Mapping</span>
                    </div>
                </div>
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-12">
                <div class="space-y-8">
                    <div class="glass-card border-none shadow-2xl bg-white p-10 relative overflow-hidden group">
                        <div class="absolute -right-20 -top-20 w-80 h-80 bg-blue-50 rounded-full blur-[100px] transition-all group-hover:bg-blue-100/50 pulse"></div>

                        <div class="flex items-center gap-4 mb-12">
                            <div class="w-12 h-12 rounded-2xl bg-slate-900 flex items-center justify-center text-white"><i data-lucide="cpu" size="20"></i></div>
                            <div>
                                <h4 class="text-[10px] font-black uppercase tracking-[0.4em] text-slate-900">Vector Coordinates</h4>
                                <p class="text-xs text-slate-400 font-bold">Configure journey parameters</p>
                            </div>
                        </div>

                        <div class="space-y-8 relative z-10">
                            <div class="space-y-3">
                                <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1 font-bold">Source Matrix</label>
                                <div class="relative group/input">
                                    <div class="absolute left-6 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-blue-600 ring-8 ring-blue-50"></div>
                                    <select id="start-st" class="w-full pl-14 pr-8 py-6 bg-slate-50 border-2 border-transparent rounded-[24px] outline-none focus:border-blue-500/30 focus:bg-white font-black appearance-none text-slate-900 transition-all cursor-pointer hover:bg-slate-100 hover:scale-[1.01]"></select>
                                    <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300"><i data-lucide="chevron-down" size="18"></i></div>
                                </div>
                            </div>

                            <div class="flex justify-center -my-6 relative z-10">
                                <div class="p-5 bg-slate-900 text-white rounded-full shadow-2xl hover:rotate-180 transition-transform duration-700 cursor-pointer border-8 border-white group-hover:scale-110">
                                    <i data-lucide="arrow-down-up" size="22"></i>
                                </div>
                            </div>

                            <div class="space-y-3">
                                <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1 font-bold">Target Matrix</label>
                                <div class="relative group/input">
                                    <div class="absolute left-6 top-1/2 -translate-y-1/2"><i data-lucide="map-pin" size="18" class="text-emerald-500"></i></div>
                                    <select id="end-st" class="w-full pl-14 pr-8 py-6 bg-slate-50 border-2 border-transparent rounded-[24px] outline-none focus:border-emerald-500/30 focus:bg-white font-black appearance-none text-slate-900 transition-all cursor-pointer hover:bg-slate-100 hover:scale-[1.01]"></select>
                                    <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300"><i data-lucide="chevron-down" size="18"></i></div>
                                </div>
                            </div>

                            <button id="plan-btn" onclick="planJourney()" class="w-full py-7 bg-slate-900 hover:bg-black text-white font-black rounded-[28px] shadow-2xl shadow-slate-300 transition-all active:scale-[0.98] text-[12px] uppercase tracking-[0.3em] mt-8 flex items-center justify-center gap-4 group">
                                <span id="btn-text">Compute Optimized Path</span>
                                <div id="btn-loader" class="hidden w-5 h-5 border-[3px] border-white border-t-transparent rounded-full animate-spin"></div>
                                <i data-lucide="zap" size="16" class="fill-indigo-400 group-hover:scale-125 transition-transform"></i>
                            </button>
                        </div>
                    </div>

                    <div id="route-schedule" class="hidden glass-card scrollbar-hide max-h-[400px] overflow-y-auto">
                        <h4 class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6 border-b pb-4">Upcoming Next Hour</h4>
                        <div id="schedule-list" class="space-y-3"></div>
                    </div>

                    <div class="glass-card p-8 border-none bg-slate-50">
                         <h4 class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6">General Transit Guide</h4>
                         <div class="space-y-4">
                              <div class="flex items-start gap-4"><div class="w-10 h-10 bg-white rounded-xl shadow-sm flex items-center justify-center shrink-0"><i data-lucide="info" size="18" class="text-blue-600"></i></div><p class="text-xs font-semibold text-slate-600 leading-relaxed">Interchanges at Ameerpet (Red/Blue), MGBS (Red/Green), and Parade Ground (Blue/Green).</p></div>
                              <div class="flex items-start gap-4"><div class="w-10 h-10 bg-white rounded-xl shadow-sm flex items-center justify-center shrink-0"><i data-lucide="phone-call" size="18" class="text-red-500"></i></div><p class="text-xs font-semibold text-slate-600 leading-relaxed">Emergency Helpline: 155370. Medical SOS available at all major hubs.</p></div>
                         </div>
                    </div>
                           <div id=\"route-output\" class=\"hidden space-y-6\">
                    <!-- AI Recommendation Box -->
                    <div class=\"glass-card border-none bg-indigo-50 border-l-4 border-indigo-500 p-6 flex items-start gap-4\">
                        <div class=\"p-2 bg-indigo-500 text-white rounded-xl\"><i data-lucide=\"cpu\" size=\"16\"></i></div>
                        <div>
                            <h5 class=\"text-[10px] font-black text-indigo-600 uppercase tracking-widest mb-1\">AI Recommendation</h5>
                            <p id=\"route-rec\" class=\"text-xs font-bold text-slate-600\">--</p>
                        </div>
                    </div>

                    <div class=\"glass-card p-0 overflow-hidden border-none shadow-2xl\">
                        <div class=\"p-6 lg:p-10 bg-slate-900 text-white flex flex-col lg:flex-row lg:justify-between lg:items-end overflow-hidden relative gap-8 lg:gap-0\">
                            <div class=\"absolute -left-10 -bottom-10 w-40 h-40 bg-blue-600/30 rounded-full blur-3xl\"></div>
                            <div>
                                <p id=\"route-dur\" class=\"text-4xl lg:text-5xl font-black leading-none\">--</p>
                                <p class=\"text-[10px] font-bold uppercase tracking-widest opacity-40 mt-4\">Total Transit Time</p>
                            </div>
                            <div class=\"flex justify-between lg:block lg:text-right z-10 w-full lg:w-auto\">
                                <div class=\"flex flex-col lg:items-end lg:mb-4\">
                                    <p id=\"route-dest-time\" class=\"text-lg lg:text-xl font-black\">--:--</p>
                                    <p class=\"text-[9px] font-bold uppercase tracking-widest opacity-40\">Est. Arrival</p>
                                </div>
                                <div class=\"flex flex-col items-end\">
                                    <p id=\"route-fare\" class=\"text-lg lg:text-xl font-black\">₹--</p>
                                    <p class=\"text-[9px] font-bold uppercase tracking-widest opacity-40\">Locked Fare</p>
                                </div>
                            </div>
                        </div>
                        <div class="px-10 py-6 bg-slate-50 border-b border-slate-100 flex justify-between">
                             <div class="flex items-center gap-2"><i data-lucide="layers" class="text-slate-400" size="14"></i> <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest"><span id="route-stops">0</span> Stops | <span id="route-dist">0</span> KM Total</span></div>
                             <div class="flex items-center gap-2"><i data-lucide="zap" class="text-green-500" size="14"></i> <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Optimized Path</span></div>
                        </div>
                        <div class="p-10 bg-white">
                            <div id="route-seq" class="border-l-2 border-slate-100 ml-5 pl-10 space-y-8 py-2"></div>
                        </div>
                    </div>
                </div>

                <div id="route-empty" class="glass-card flex flex-col items-center justify-center py-32 text-slate-300 border-dashed border-2 border-slate-200">
                    <i data-lucide="cpu" size="48" class="mb-4 opacity-20"></i><p class="font-bold text-slate-400 uppercase text-[10px] tracking-widest">Engine Idle. Awaiting logic trigger.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();
        const stations = {{ ALL_STATIONS | tojson }};

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
            const amount = lastCalculatedFare;
            const upiId = "metro@upi"; // Mock Merchant ID
            const merchantName = "HydMetro Authority";
            const transactionNote = "Metro Ticket Booking";

            // Standard UPI Deep link
            const upiUrl = `upi://pay?pa=${upiId}&pn=${encodeURIComponent(merchantName)}&am=${amount}&cu=INR&tn=${encodeURIComponent(transactionNote)}`;

            console.log(`Initiating ${appName} payment for ₹${amount}...`);

            // In a real device environment, this triggers the UPI app selector or the specific app
            window.location.href = upiUrl;

            // UI Feedback
            alert(`Redirecting to ${appName} to pay ₹${amount}. \n(Note: UPI deep links require a mobile device with a UPI app installed)`);
        }

        function updateUserPin(lat, lng) {
            const g = document.getElementById('map-user-pin');
            let pin = document.getElementById('user-location-pin');

            // Geographic mapping based on station distribution
            const lats = stations.map(s => s.lat);
            const lngs = stations.map(s => s.lng);
            const xs = stations.map(s => s.x);
            const ys = stations.map(s => s.y);

            const minLat = Math.min(...lats), maxLat = Math.max(...lats);
            const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
            const minX = Math.min(...xs), maxX = Math.max(...xs);
            const minY = Math.min(...ys), maxY = Math.max(...ys);

            // Interpolate coordinate mapping
            const x = minX + (lng - minLng) / (maxLng - minLng) * (maxX - minX);
            const y = minY + (maxLat - lat) / (maxLat - minLat) * (maxY - minY);

            if(!pin) {
                pin = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                pin.id = 'user-location-pin';
                pin.style.transition = 'transform 1s cubic-bezier(0.4, 0, 0.2, 1)';

                const outer = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                outer.setAttribute('r', 15);
                outer.setAttribute('fill', '#3b82f6');
                outer.setAttribute('class', 'user-pin-outer opacity-25');

                const inner = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                inner.setAttribute('r', 6);
                inner.setAttribute('fill', '#2563eb');
                inner.setAttribute('stroke', 'white');
                inner.setAttribute('stroke-width', '2');

                pin.appendChild(outer);
                pin.appendChild(inner);
                g.appendChild(pin);
            }

            pin.setAttribute('transform', `translate(${x}, ${y})`);
        }

        function toggleSidebar() {
            const sidebar = document.querySelector('.sidebar');
            const main = document.getElementById('main-content');
            const btn = document.getElementById('sidebar-toggle');

            sidebar.classList.toggle('collapsed');
            main.classList.toggle('full-width');
            btn.classList.toggle('sidebar-collapsed');

            const isCollapsed = sidebar.classList.contains('collapsed');
            btn.innerHTML = isCollapsed ? '<i data-lucide="panel-left-open"></i>' : '<i data-lucide="panel-left-close"></i>';
            lucide.createIcons();
        }

        function showTab(id) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            document.querySelectorAll('.mobile-link').forEach(l => l.classList.remove('active'));

            document.getElementById('tab-'+id).classList.add('active');
            document.getElementById('btn-'+id).classList.add('active');
            document.getElementById('mob-'+id).classList.add('active');

            if(id !== 'map') closeOverlay();
        }

        function closeOverlay() {
            document.getElementById('map-overlay').classList.add('translate-x-full');
            document.querySelectorAll('.station-node').forEach(n => n.classList.remove('selected'));
        }

        function updateClock() {
            const now = new Date();
            // Force 12h formatting for the clock span to avoid "13:17:21 PM"
            let options = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true };
            let timeStr = now.toLocaleTimeString('en-US', options);
            // Split to get just the time and the AM/PM
            let parts = timeStr.split(' ');
            if (parts.length === 2) {
                document.getElementById('clock').innerText = parts[0];
                document.getElementById('ampm').innerText = parts[1];
            } else {
                // Fallback if formatting differs
                document.getElementById('clock').innerText = now.toLocaleTimeString('en-US', { hour12: false });
                document.getElementById('ampm').innerText = now.getHours() >= 12 ? 'PM' : 'AM';
            }
            document.getElementById('date').innerText = now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }).toUpperCase();
        }
        setInterval(updateClock, 1000); updateClock();

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
                        dev.innerHTML = `<span class=\"text-[9px] font-black text-slate-400 uppercase\">Available</span><span class=\"text-[11px] font-bold text-slate-700\">${a}</span>`;
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
                    trainCont.innerHTML = `<div class=\"py-10 flex flex-col items-center gap-4 text-slate-300\"><div class=\"w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin\"></div><p class=\"text-[9px] font-black uppercase tracking-widest\">Syncing Flux...</p></div>`;

                    try {
                        const res = await fetch('/api/nearest', { 
                            method: 'POST', 
                            headers: {'Content-Type': 'application/json'}, 
                            body: JSON.stringify({ station_id: s.id }) 
                        });
                        const data = await res.json();
                        trainCont.innerHTML = '';

                        if (data.upcoming.length === 0) {
                            trainCont.innerHTML = `<p class=\"text-xs font-bold text-slate-400 italic\">Node static. No trains projected.</p>`;
                        }

                        data.upcoming.slice(0, 5).forEach(t => {
                            const lineCol = t.line === 'Red' ? 'bg-red-500' : t.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                            const tDiv = document.createElement('div');
                            tDiv.className = \"flex justify-between items-center bg-slate-50 p-5 rounded-2xl border border-slate-100 hover:border-blue-200 transition-colors\";
                            tDiv.innerHTML = `
                                <div class=\"flex items-center gap-4\">
                                    <div class=\"w-1 my-1 self-stretch rounded-full ${lineCol}\"></div>
                                    <div>
                                        <p class=\"text-[11px] font-black text-slate-900 leading-none mb-1\">${t.final_stop}</p>
                                        <p class=\"text-[9px] font-bold text-slate-400 uppercase tracking-widest\">Platform ${t.platform}</p>
                                    </div>
                                </div>
                                <div class=\"text-right\">
                                    <p class=\"text-sm font-black text-blue-600 tabular-nums\">${t.arrival_time}</p>
                                    <p class=\"text-[8px] font-black text-slate-400 uppercase tracking-widest\">${t.eta}</p>
                                </div>
                            `;
                            trainCont.appendChild(tDiv);
                        });
                    } catch (e) {
                        trainCont.innerHTML = `<p class=\"text-xs font-bold text-red-500\">Sync Failed</p>`;
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
                const res = await fetch('/api/nearest', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify(body) 
                });
                if (!res.ok) throw new Error("API Offline");
                const data = await res.json();

                document.getElementById('near-name').innerText = data.station.name;
                document.getElementById('load-val').innerText = data.load_label;
                document.getElementById('greeting').innerText = data.greeting + '!';
                document.getElementById('weather-val').innerText = data.weather.temp + '°C, ' + data.weather.condition;

                // Sync selector if in auto-mode
                const selector = document.getElementById('board-station-selector');
                if (!stationId) {
                    selector.value = data.station.id;
                }
                document.getElementById('weather-detail').innerText = `Humidity: ${data.weather.humidity}% | Visibility: ${data.weather.visibility.toFixed(1)}km`;

                const lCard = document.getElementById('load-card');
                const loadCol = data.load_val === 'High' ? '#ef4444' : 
                               data.load_val === 'M-High' ? '#f59e0b' : 
                               data.load_val === 'Medium' ? '#6366f1' : '#10b981';
                lCard.style.borderLeft = `6px solid ${loadCol}`;

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
                        <td class="py-5"><div class="w-1.5 h-8 rounded-full ${lineCol} shadow-sm group-hover:scale-y-125 transition-transform"></div></td>
                        <td class="py-5">
                            <div class="flex flex-col">
                                <span class="font-black text-slate-800 text-[12px] tracking-tight">${t.final_stop}</span>
                                <span class="text-[8px] font-bold text-slate-400 uppercase tracking-widest">${t.line} Line Matrix</span>
                            </div>
                        </td>
                        <td class="py-5"><span class="px-3 py-1 bg-slate-900 text-white rounded-lg text-[9px] font-black tracking-widest border-2 border-slate-800">PLATFORM ${t.platform}</span></td>
                        <td class="py-5 text-right font-black">
                            <div class="flex flex-col items-end">
                                <span class="sync-clock">${t.eta}</span>
                                <span class="text-[8px] text-slate-400 uppercase tracking-widest mt-1">Countdown Mode</span>
                            </div>
                        </td>`;
                    rows.appendChild(tr);
                });
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
            stations.slice().sort((a,b)=>a.name.localeCompare(b.name)).forEach(st => {
                const opt = document.createElement('option'); opt.value = st.id; opt.innerText = st.name; selector.appendChild(opt);
            });

            if (navigator.geolocation) {
                navigator.geolocation.watchPosition(
                    pos => updateBoardData(pos.coords.latitude, pos.coords.longitude),
                    err => {
                        console.warn("Geo blocked, falling back to Ameerpet.");
                        const ameerpet = stations.find(s => s.name === 'Ameerpet');
                        updateBoardData(ameerpet.lat, ameerpet.lng, ameerpet.id);
                    },
                    { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
                );
            } else {
                const ameerpet = stations.find(s => s.name === 'Ameerpet');
                updateBoardData(ameerpet.lat, ameerpet.lng, ameerpet.id);
            }
        }

        async function planJourney() {
            const btn = document.getElementById('plan-btn');
            const btnText = document.getElementById('btn-text');
            const loader = document.getElementById('btn-loader');

            btn.disabled = true;
            btnText.innerText = 'Calculating...';
            loader.classList.remove('hidden');

            try {
                const f = document.getElementById('start-st').value, t = document.getElementById('end-st').value;
                const res = await fetch('/api/plan', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({from: f, to: t}) });
                const data = await res.json();

                document.getElementById('route-empty').classList.add('hidden');
                document.getElementById('route-output').classList.remove('hidden');
                document.getElementById('route-schedule').classList.remove('hidden');

                document.getElementById('route-dur').innerText = data.duration + 'm';
                document.getElementById('route-fare').innerText = '₹' + data.fare;
                document.getElementById('route-stops').innerText = data.total_stops;
                document.getElementById('route-dist').innerText = data.total_km;
                document.getElementById('route-dest-time').innerText = data.arrival_at_dest;
                document.getElementById('route-rec').innerText = data.recommendation;

                lastCalculatedFare = data.fare;

                // Add Metrics Grid
                const seq = document.getElementById('route-seq'); seq.innerHTML = '';
                const metricsDiv = document.createElement('div');
                metricsDiv.className = "grid grid-cols-2 gap-4 mb-8 bg-slate-50 p-4 rounded-2xl border border-slate-100";
                metricsDiv.innerHTML = `
                    <div class="flex flex-col"><span class="text-[8px] font-black uppercase text-slate-400 tracking-widest mb-1">Peak Status</span><span class="text-xs font-bold text-slate-700">${data.metrics.peak}</span></div>
                    <div class="flex flex-col"><span class="text-[8px] font-black uppercase text-slate-400 tracking-widest mb-1">IT Activity</span><span class="text-xs font-bold text-slate-700">${data.metrics.it_hub}</span></div>
                    <div class="flex flex-col"><span class="text-[8px] font-black uppercase text-slate-400 tracking-widest mb-1">Fare Logic</span><span class="text-xs font-bold text-green-600">Stable</span></div>
                    <div class="flex flex-col"><span class="text-[8px] font-black uppercase text-slate-400 tracking-widest mb-1">Processing</span><span class="text-xs font-bold text-blue-600">Neural Sync</span></div>
                `;
                seq.appendChild(metricsDiv);

                // Add Guides if any (NEW)
                if (data.guides && data.guides.length > 0) {
                    const guideHeader = document.createElement('h5');
                    guideHeader.className = "text-[10px] font-black text-blue-600 uppercase tracking-[0.2em] mb-4 flex items-center gap-2";
                    guideHeader.innerHTML = `<i data-lucide="navigation" size="12"></i> Critical Transfer Guides`;
                    seq.appendChild(guideHeader);

                    data.guides.forEach((g, gIdx) => {
                        const gDiv = document.createElement('div');
                        gDiv.className = "bg-amber-50 border border-amber-200/50 p-5 rounded-2xl shadow-sm mb-6";
                        gDiv.innerHTML = `
                            <div class="flex items-start gap-4">
                                <div class="w-8 h-8 rounded-xl bg-amber-100 flex items-center justify-center shrink-0"><i data-lucide="arrow-up-right" size="16" class="text-amber-700"></i></div>
                                <div>
                                    <p class="text-xs font-black text-slate-800 leading-tight">Step ${gIdx+1}: ${g.station} Interchange</p>
                                    <p class="text-[10px] font-semibold text-amber-900 mt-2 leading-relaxed opacity-80">${g.text}</p>
                                    <div class="mt-3 inline-flex items-center gap-2 px-3 py-1 bg-white rounded-lg border border-amber-200 shadow-sm">
                                        <span class="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse"></span>
                                        <span class="text-[9px] font-black text-amber-700 uppercase tracking-widest">Platform ${g.platform}</span>
                                    </div>
                                </div>
                            </div>
                        `;
                        seq.appendChild(gDiv);
                    });
                }

                const pathHeader = document.createElement('h5');
                pathHeader.className = "text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4";
                pathHeader.innerText = "Network Vector Stream";
                seq.appendChild(pathHeader);

                data.sequence.forEach((s) => {
                    const step = document.createElement('div'); step.className = 'relative flex items-center gap-6';
                    step.innerHTML = `<div class=\"absolute -left-[48px] w-4 h-4 rounded-full border-4 border-white shadow-lg ${s.line === 'Red' ? 'bg-red-500' : s.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500'}\"></div>
                        <div><p class=\"font-black text-slate-800 text-lg\">${s.name}</p><p class=\"text-[10px] font-bold text-slate-400 uppercase tracking-widest\">${s.line} Line Hub</p></div>`;
                    seq.appendChild(step);
                });

                const sched = document.getElementById('schedule-list'); sched.innerHTML = '';
                data.upcoming_hour.forEach(u => {
                    const div = document.createElement('div'); div.className = 'flex justify-between items-center bg-white p-5 rounded-2xl border border-slate-100 shadow-sm';
                        div.innerHTML = `
                        <div class="flex items-center gap-4">
                            <div class="flex flex-col">
                                <span class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Platform</span>
                                <span class="font-black text-slate-900">P${u.platform}</span>
                            </div>
                            <div class="w-px h-8 bg-slate-100"></div>
                            <div class="flex flex-col">
                                <span class="text-sm font-black text-slate-700">${u.final_stop}</span>
                                <span class="text-[9px] font-bold text-slate-400 uppercase">${u.line} Line</span>
                            </div>
                        </div>
                        <div class="text-right">
                            <p class="sync-clock mb-1">${u.eta}</p>
                            <p class="text-[9px] font-bold text-slate-400 uppercase">Synchronization</p>
                        </div>`;
                    sched.appendChild(div);
                });
                lucide.createIcons();
            } catch (e) {
                console.error(e);
            } finally {
                btn.disabled = false;
                btnText.innerText = 'Compute Optimized path';
                loader.classList.add('hidden');
            }
        }

        // Initialize pickers with grouped matrix logic
        function initPickers() {
            const lines = ['Red', 'Blue', 'Green'];
            ['start-st', 'end-st'].forEach(id => {
                const select = document.getElementById(id);
                select.innerHTML = '<option value=\"\" disabled selected>Select Matrix Node...</option>';

                lines.forEach(line => {
                    const group = document.createElement('optgroup');
                    group.label = `${line} Line Network`;

                    stations.filter(s => s.line === line).sort((a,b) => a.name.localeCompare(b.name)).forEach(st => {
                        const opt = document.createElement('option');
                        opt.value = st.id;
                        opt.innerText = st.name;
                        group.appendChild(opt);
                    });
                    select.appendChild(group);
                });
            });
        }

        async function updateLiveTrains() {
            try {
                const res = await fetch('/api/live-map');
                const data = await res.json();
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

                    let train = g.querySelector(`[data-trip-id="${t.trip_id}"]`);
                    if(!train) {
                        train = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                        train.setAttribute('class', 'train-icon');
                        train.setAttribute('data-trip-id', t.trip_id);
                        train.style.transition = 'transform 2.1s linear'; // Slightly longer than interval to hide latency

                        const color = t.line === 'Red' ? '#ef4444' : t.line === 'Blue' ? '#3b82f6' : '#22c55e';

                        const outer = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                        outer.setAttribute('r', 12);
                        outer.setAttribute('fill', color);
                        outer.setAttribute('class', 'animate-pulse opacity-25');

                        const inner = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                        inner.setAttribute('r', 6);
                        inner.setAttribute('fill', color);
                        inner.setAttribute('stroke', '#fff');
                        inner.setAttribute('stroke-width', '2');

                        train.appendChild(outer);
                        train.appendChild(inner);
                        g.appendChild(train);

                        // Initial position without transition
                        train.style.transition = 'none';
                        train.setAttribute('transform', `translate(${curX}, ${curY})`);
                        setTimeout(() => train.style.transition = 'transform 2.1s linear', 50);
                    } else {
                        train.setAttribute('transform', `translate(${curX}, ${curY})`);
                    }
                });

                existingTrains.forEach(t => {
                    if(!seenIds.has(t.getAttribute('data-trip-id'))) t.remove();
                });
            } catch (e) {}
        }
        setInterval(updateLiveTrains, 2000);

        setupMap(); initGeo(); initPickers(); updateLiveTrains();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    ensure_gtfs(force=True)
    generate_ai_dataset()
    app.run(debug=True, port=3000, host='0.0.0.0')
