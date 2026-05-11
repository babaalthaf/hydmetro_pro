import os
import csv
import json
import math
import random
import urllib.request
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

# ==========================================
# 1. SMART DATA & PREDICTION LOGIC
# ==========================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def get_ist_now():
    """Returns current time in India."""
    base_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return base_now

def get_app_now():
    """Context-aware time for app logic."""
    sim_hour = -1
    sim_min = 0
    
    # Check if a specific time was sent in the request body (for planning)
    if request:
        try:
            if request.is_json:
                data = request.get_json(silent=True)
                if data and 'planned_time' in data and data['planned_time']:
                    # Format: "HH:MM"
                    h, m = map(int, data['planned_time'].split(':'))
                    now = get_ist_now()
                    return now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            if 'sim_hour' in request.args:
                sim_hour = int(request.args.get('sim_hour'))
                sim_min = int(request.args.get('sim_min', 0))
            elif request.is_json:
                data = request.get_json(silent=True)
                if data and 'sim_hour' in data:
                    sim_hour = int(data.get('sim_hour', -1))
                    sim_min = int(data.get('sim_min', 0))
        except: pass
    
    if sim_hour != -1:
        try:
            now = get_ist_now()
            return now.replace(hour=sim_hour, minute=sim_min, second=0, microsecond=0)
        except: pass
    return get_ist_now()

def get_live_weather(lat=None, lng=None):
    """Fetches real-time weather from Open-Meteo with extra metrics."""
    if lat is None or lng is None:
        lat, lng = 17.3850, 78.4867 # Default Hyderabad
        
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true&hourly=relative_humidity_2m,visibility"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
        
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
            'humidity': data['hourly']['relative_humidity_2m'][h_idx] if h_idx < len(data['hourly']['relative_humidity_2m']) else 45,
            'visibility': (data['hourly']['visibility'][h_idx] / 1000) if h_idx < len(data['hourly']['visibility']) else 10,
            'aqi': random.randint(30, 70)
        }
    except:
        return {'temp': 30, 'condition': 'Clear Sky', 'humidity': 45, 'visibility': 10, 'aqi': 42}

def generate_ai_dataset():
    """Writes the user provided ridership dataset to final_metro_dataset.csv."""
    data = [
        ["Punjagutta","2026-05-03 14:26:13","1","14","6","0","1","0","27","0","0","40"],
        ["Begumpet","2026-04-30 08:52:45","2","8","3","1","0","0","32","12","0","145"],
        ["HITEC City","2026-10-12 18:05:30","1","18","0","1","0","1","34","0","0","210"],
        ["Ameerpet","2026-06-15 17:30:00","1","17","0","1","0","0","31","5","0","185"],
        ["Miyapur","2026-03-22 10:15:20","2","10","6","1","1","0","29","0","0","110"],
        ["MG Bus Station","2026-11-08 20:45:10","2","20","6","1","1","0","26","0","1","195"],
        ["Tarnaka","2026-08-04 07:40:00","1","7","1","1","0","0","28","0","0","95"],
        ["Raidurg","2026-12-24 16:20:45","1","16","3","0","0","1","25","0","0","160"],
        ["Kukatpally","2026-01-14 09:10:05","2","9","2","1","0","0","24","0","1","175"],
        ["Dilsukhnagar","2026-09-12 11:55:30","1","11","5","0","1","0","33","0","0","130"],
        ["L.B. Nagar","2026-02-18 19:10:00","2","19","2","1","0","0","30","0","0","155"],
        ["Nagole","2026-07-30 21:05:40","1","21","3","1","0","0","31","0","0","140"],
        ["Secunderabad East","2026-05-05 08:20:15","2","8","1","1","0","0","29","0","0","165"],
        ["JNTU College","2026-10-21 13:45:00","1","13","2","0","0","0","35","0","0","85"],
        ["Parade Ground","2026-11-02 18:50:30","2","18","0","1","0","0","27","0","0","205"]
    ]
    with open('final_metro_dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["stop_name","arrival_time","platform","hour","day_of_week","is_peak","is_weekend","is_it_hub","temperature","rainfall","is_festival","ridership"])
        writer.writerows(data)
    return True

def predict_load_ai(station_name, hour, is_weekend=False, weather=None):
    """Predicts load using historical CSV data if available, fallback to heuristic."""
    load_val_num = 0
    found_in_csv = False
    
    try:
        if os.path.exists('final_metro_dataset.csv'):
            with open('final_metro_dataset.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['stop_name'] == station_name and int(row['hour']) == hour:
                        load_val_num = int(row['ridership'])
                        found_in_csv = True
                        break
    except: pass

    if found_in_csv:
        # Normalize CSV ridership to a 0-100 scale for UI usage
        score = min(100, (load_val_num / 250)) 
        if load_val_num > 1500: return "High", score
        if load_val_num > 800: return "M-High", score
        if load_val_num > 300: return "Medium", score
        return "Low", score

    # Heuristic Fallback
    is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0
    is_it_hub = 1 if any(h in station_name for h in ['HITEC City', 'Madhapur', 'Raidurg', 'Ameerpet']) else 0
    
    score = 40 + (is_peak * 40) + (is_it_hub * 20)
    if is_weekend: score -= 15
    
    if weather:
        if "Rain" in weather.get('condition', ''): score += 15
        if weather.get('temp', 30) > 35: score += 10
    
    score = min(100, max(5, score + random.uniform(-5, 5)))
    
    if score > 80: return "High", score
    if score > 60: return "M-High", score
    if score > 35: return "Medium", score
    return "Low", score

# ==========================================
# 2. FULL DATASET (59 STATIONS - UPDATED)
# ==========================================
STATIONS_LIST = [
       {'id': 'R1', 'name': 'Miyapur', 'line': 'Red', 'x': 100, 'y': 200, 'lat': 17.496441625059354, 'lng': 78.37291458173314, 'amenities': ['Large Parking Hub', 'Food Court (Subway, KFC)', 'Medical Center', 'HDFC Bank ATM'], 'description': 'Miyapur is a major residential hub and the western terminal of the metro. It is popular among IT employees due to its affordability and act as a crucial daily commute hub.'},
    {'id': 'R2', 'name': 'JNTU College', 'line': 'Red', 'x': 160, 'y': 240, 'lat': 17.49870028606833, 'lng': 78.388279, 'amenities': ['Student Discount Kiosk', 'Cycle Stand', 'Coffee Day', 'Restrooms'], 'description': 'Serves the Jawaharlal Nehru Technological University and surrounding academic hubs.'},
    {'id': 'R3', 'name': 'KPHB Colony', 'line': 'Red', 'x': 220, 'y': 280, 'lat': 17.49379645973468, 'lng': 78.40169333899651, 'amenities': ['Main Mall Access', 'Electronics Outlet', 'Escalators'], 'description': 'A bustling residential and commercial node with heavy student and shopper footfall.'},
    {'id': 'R4', 'name': 'Kukatpally', 'line': 'Red', 'x': 280, 'y': 320, 'lat': 17.485070512388596, 'lng': 78.4115403827053, 'amenities': ['Elevated Concourse', 'Local Market Link', 'Elevators', 'CCTV'], 'description': 'A dense residential area with numerous shopping centers and markets nearby. It is a major hub for affordable housing and retail.'},
    {'id': 'R5', 'name': 'Dr. B.R. Ambedkar Balanagar', 'line': 'Red', 'x': 340, 'y': 360, 'lat': 17.476746484051, 'lng': 78.42202741151657, 'amenities': ['Industrial Access Gate', 'Parking', 'Help Desk'], 'description': 'Important industrial and commercial connection point in the Balanagar zone, connecting the city to the industrial corridors.'},
    {'id': 'R6', 'name': 'Moosapet', 'line': 'Red', 'x': 400, 'y': 400, 'lat': 17.47209873592423, 'lng': 78.42594732033717, 'amenities': ['Food Kiosks', 'Restrooms', 'Security Gate'], 'description': 'Serves the residential clusters of Moosapet and nearby commercial establishments like movie theaters and local shops.'},
    {'id': 'R7', 'name': 'Bharat Nagar', 'line': 'Red', 'x': 460, 'y': 440, 'lat': 17.466626, 'lng': 78.429572, 'description': 'Connects the Bharat Nagar residential area and provides access to local markets.'},
    {'id': 'R8', 'name': 'Erragadda', 'line': 'Red', 'x': 520, 'y': 480, 'lat': 17.459406, 'lng': 78.433426, 'description': 'Located near major healthcare facilities and residential neighborhoods.'},
    {'id': 'R9', 'name': 'ESI Hospital', 'line': 'Red', 'x': 580, 'y': 520, 'lat': 17.447638, 'lng': 78.438267, 'description': 'Essential stop for the Employees State Insurance (ESI) Hospital and healthcare staff.'},
    {'id': 'R10', 'name': 'SR Nagar', 'line': 'Red', 'x': 640, 'y': 560, 'lat': 17.441797, 'lng': 78.441615, 'description': 'A central residential and educational hub known for its student population.'},
    {'id': 'R11', 'name': 'Ameerpet', 'line': 'Red', 'x': 700, 'y': 600, 'lat': 17.435808, 'lng': 78.444675, 'amenities': ['Interchange (Blue Line)', 'Ameerpet Metro Mall', 'Fine Dining Court', 'Apollo Clinic', 'SBI Full Branch'], 'description': 'The busiest interchange station connecting the city\'s two longest corridors. Houses numerous shopping outlets and eateries within the station premises. Landmarks include Next Galleria Mall and various coaching institutes. Reach any part of the city within 15-20 mins.'},
    {'id': 'R12', 'name': 'Punjagutta', 'line': 'Red', 'x': 760, 'y': 640, 'lat': 17.428756, 'lng': 78.451199, 'amenities': ['Next Galleria Mall', 'PVR Cinemas Link', 'Starbucks', 'Valet Parking', 'Luxury Retail Hub'], 'description': 'Directly connected to major malls like Next Galleria. Serving as a crucial hub for luxury shopping, entertainment, and the business district.'},
    {'id': 'R13', 'name': 'Irrum Manzil', 'line': 'Red', 'x': 820, 'y': 680, 'lat': 17.420983, 'lng': 78.456019, 'description': 'Direct connectivity to the Next Galleria mall and proximity to government offices.'},
    {'id': 'R14', 'name': 'Khairatabad', 'line': 'Red', 'x': 880, 'y': 720, 'lat': 17.41152, 'lng': 78.460739, 'description': 'Located near the Khairatabad flyover and provides access to several government departments.'},
    {'id': 'R15', 'name': 'Lakdikapul', 'line': 'Red', 'x': 940, 'y': 760, 'lat': 17.404477, 'lng': 78.46516, 'description': 'A key node for government offices and heritage sites in the core city.'},
    {'id': 'R16', 'name': 'Assembly', 'line': 'Red', 'x': 1000, 'y': 800, 'lat': 17.398754, 'lng': 78.470811, 'description': 'Serves the Telangana State Legislative Assembly and Public Gardens area.'},
    {'id': 'R17', 'name': 'Nampally', 'line': 'Red', 'x': 1060, 'y': 840, 'lat': 17.385988, 'lng': 78.472925, 'description': 'Direct link to the historical Nampally Railway Station and busy commercial markets.'},
    {'id': 'R18', 'name': 'Gandhi Bhavan', 'line': 'Red', 'x': 1120, 'y': 880, 'lat': 17.386037, 'lng': 78.472983, 'description': 'Serves the Gandhi Bhavan and the administrative centers of political entities.'},
    {'id': 'R19', 'name': 'Osmania Medical College', 'line': 'Red', 'x': 1180, 'y': 920, 'lat': 17.383006, 'lng': 78.480965, 'description': 'Primary stop for Osmania Medical College and surrounding heritage market areas.'},
    {'id': 'R20', 'name': 'MG Bus Station', 'line': 'Red', 'x': 1240, 'y': 960, 'lat': 17.378583, 'lng': 78.485514, 'amenities': ['Interstate Bus Hub', 'Interchange (Green Line)', 'Multi-level Parking', 'Dormitory Services', 'Cloak Room'], 'description': 'The primary gateway for those using intercity bus services via MGBS. It is the nearest operational hub for the Mahatma Gandhi Bus Station, connecting the city to the entire state.'},
    {'id': 'R21', 'name': 'Malakpet', 'line': 'Red', 'x': 1300, 'y': 1000, 'lat': 17.377354, 'lng': 78.493668, 'description': 'Serves the residential and commercial areas of Malakpet and Saidabad.'},
    {'id': 'R22', 'name': 'New Market', 'line': 'Red', 'x': 1360, 'y': 1040, 'lat': 17.373504, 'lng': 78.503109, 'description': 'Located in a busy commercial zone near historical fruit and grain markets.'},
    {'id': 'R23', 'name': 'Musarambagh', 'line': 'Red', 'x': 1420, 'y': 1080, 'lat': 17.371088, 'lng': 78.511864, 'description': 'Provides access to the dense residential layouts of Musarambagh.'},
    {'id': 'R24', 'name': 'Dilsukhnagar', 'line': 'Red', 'x': 1480, 'y': 1120, 'lat': 17.369204, 'lng': 78.525511, 'amenities': ['Shopping Arcade', 'Bus Junction Link', 'Snack Counters', 'Escalators'], 'description': 'A massive commercial hub on the eastern side with significant student traffic.'},
    {'id': 'R25', 'name': 'Chaitanyapuri', 'line': 'Red', 'x': 1540, 'y': 1160, 'lat': 17.368876, 'lng': 78.535982, 'description': 'Residential gateway for eastern Hyderabad colonies.'},
    {'id': 'R26', 'name': 'Victoria Memorial', 'line': 'Red', 'x': 1600, 'y': 1200, 'lat': 17.362445, 'lng': 78.54375, 'description': 'Named after the nearby historical landmark on the way to LB Nagar.'},
    {'id': 'R27', 'name': 'LB Nagar', 'line': 'Red', 'x': 1660, 'y': 1240, 'lat': 17.349663, 'lng': 78.548073, 'amenities': ['South Hub Terminal', 'Auto Stand', 'Public Restrooms', 'Security Post'], 'description': 'The eastern terminal station, connecting to key highway exits and outskirts.'},

    # BLUE LINE
    {'id': 'B1', 'name': 'Raidurg', 'line': 'Blue', 'x': 100, 'y': 800, 'lat': 17.442269, 'lng': 78.377124, 'amenities': ['IT Hub Terminal', 'Shopping Link', 'Premium Restrooms', 'Escalators', 'Bike-Sharing Dock'], 'description': 'The gateway to the Mindspace IT park and a major terminal for commuters. Surrounded by major tech companies and luxury hotels.'},
    {'id': 'B2', 'name': 'HITEC City', 'line': 'Blue', 'x': 167, 'y': 778, 'lat': 17.449209, 'lng': 78.383004, 'amenities': ['Tech-Park Shuttle', 'Starbucks', 'Gigabit WIFI', 'Digital Kiosks', 'Covered Walkway'], 'description': 'The pulse of the city IT sector. Features high-tech facilities and serves as the transit backbone for Cyberabad employees.'},
    {'id': 'B3', 'name': 'Durgam Cheruvu', 'line': 'Blue', 'x': 233, 'y': 756, 'lat': 17.443118, 'lng': 78.3877, 'amenities': ['Skywalk to Inorbit Mall', 'Scenic View deck', 'Elevators', 'ATM', 'Food Stalls'], 'description': 'Features a direct skywalk to the Inorbit Mall. Popular for its views of the cable bridge and proximity to recreational lake areas.'},
    {'id': 'B4', 'name': 'Madhapur', 'line': 'Blue', 'x': 300, 'y': 733, 'lat': 17.436929, 'lng': 78.400809, 'description': 'A central point for the IT industry and vibrant nightlife clusters.'},
    {'id': 'B5', 'name': 'Peddamma Temple', 'line': 'Blue', 'x': 367, 'y': 711, 'lat': 17.430523, 'lng': 78.4085, 'description': 'Serves the popular Peddamma Temple, receiving heavy pilgrim traffic on special days.'},
    {'id': 'B6', 'name': 'Jubilee Hills Check Post', 'line': 'Blue', 'x': 433, 'y': 689, 'lat': 17.428041, 'lng': 78.413704, 'description': 'Strategically located at the core of the upscale Jubilee Hills residential area.'},
    {'id': 'B7', 'name': 'Road No 5 Jubilee Hills', 'line': 'Blue', 'x': 500, 'y': 667, 'lat': 17.430662, 'lng': 78.423145, 'description': 'Access point for several corporate offices and premium food outlets in Jubilee Hills.'},
    {'id': 'B8', 'name': 'Yousufguda', 'line': 'Blue', 'x': 567, 'y': 644, 'lat': 17.435616, 'lng': 78.427308, 'description': 'Serves the dense residential neighborhoods around Yousufguda and Krishna Nagar.'},
    {'id': 'B9', 'name': 'Taruni Madhura Nagar', 'line': 'Blue', 'x': 633, 'y': 622, 'lat': 17.437499, 'lng': 78.438723, 'description': 'A uniquely themed station catering specially to women commuters and small businesses.'},
    {'id': 'B10', 'name': 'Ameerpet', 'line': 'Blue', 'x': 700, 'y': 600, 'lat': 17.435808, 'lng': 78.444675, 'name_alias': 'Ameerpet', 'description': 'The busiest interchange connecting Red and Blue corridors. Houses numerous shopping outlets.'},
    {'id': 'B11', 'name': 'Begumpet', 'line': 'Blue', 'x': 800, 'y': 560, 'lat': 17.438114, 'lng': 78.456833, 'description': 'Connects with the Begumpet Railway Station and serves major commercial entities.'},
    {'id': 'B12', 'name': 'Prakash Nagar', 'line': 'Blue', 'x': 900, 'y': 520, 'lat': 17.445579, 'lng': 78.4659, 'description': 'A key node for commuters working in the central Begumpet business district.'},
    {'id': 'B13', 'name': 'Rasoolpura', 'line': 'Blue', 'x': 1000, 'y': 480, 'lat': 17.444054, 'lng': 78.476578, 'description': 'Located in the military cantonment vicinity, serving many corporate staff.'},
    {'id': 'B14', 'name': 'Paradise', 'line': 'Blue', 'x': 1100, 'y': 440, 'lat': 17.443638, 'lng': 78.486203, 'amenities': ['Paradise Biryani Link', 'Commercial Center', 'Restrooms', 'ATM'], 'description': 'The gateway to the famous Paradise Biryani and Secunderabad commercial hubs.'},
    {'id': 'B15', 'name': 'JBS Parade Ground', 'line': 'Blue', 'x': 1200, 'y': 400, 'lat': 17.443846, 'lng': 78.497317, 'amenities': ['Green Line Interchange', 'Army Area Access', 'Parking', 'Escalators', 'Medical Kiosk'], 'description': 'Major interchange hub between Blue and Green lines. Directly serves the Jubilee Bus Station and provides access to the Secunderabad Cantonment area.'},
    {'id': 'B16', 'name': 'Secunderabad East', 'line': 'Blue', 'x': 1300, 'y': 360, 'lat': 17.436604, 'lng': 78.505235, 'amenities': ['Railway Station Link', 'Porter Services', 'Waiting Lounge', 'Food Court'], 'description': 'Direct connection to the main Secunderabad Railway Station for easy transfers.'},
    {'id': 'B17', 'name': 'Mettuguda', 'line': 'Blue', 'x': 1400, 'y': 320, 'lat': 17.436138, 'lng': 78.519492, 'description': 'Serves the residential and railway workforce colonies of Mettuguda.'},
    {'id': 'B18', 'name': 'Tarnaka', 'line': 'Blue', 'x': 1500, 'y': 280, 'lat': 17.429023, 'lng': 78.528636, 'description': 'A gateway to the Osmania University campus and numerous research institutions.'},
    {'id': 'B19', 'name': 'Habsiguda', 'line': 'Blue', 'x': 1600, 'y': 240, 'lat': 17.420163, 'lng': 78.540478, 'description': 'Connects the major residential and commercial hubs of Habsiguda.'},
    {'id': 'B20', 'name': 'NGRI', 'line': 'Blue', 'x': 1700, 'y': 200, 'lat': 17.414877, 'lng': 78.545991, 'description': 'Named after the National Geophysical Research Institute located adjacent.'},
    {'id': 'B21', 'name': 'Stadium', 'line': 'Blue', 'x': 1800, 'y': 160, 'lat': 17.408047, 'lng': 78.554102, 'description': 'Located right next to the Uppal International Cricket Stadium.'},
    {'id': 'B22', 'name': 'Uppal', 'line': 'Blue', 'x': 1900, 'y': 120, 'lat': 17.400176, 'lng': 78.560099, 'description': 'A major terminal on the eastern side, facilitating travel to the outer ring road.'},
    {'id': 'B23', 'name': 'Nagole', 'line': 'Blue', 'x': 2000, 'y': 80, 'lat': 17.39079859846158, 'lng': 78.55884857717045, 'description': 'The terminal of the Blue Line, serving the growing residential clusters of Nagole.'},

    # GREEN LINE
    {'id': 'G1', 'name': 'JBS Parade Ground', 'line': 'Green', 'x': 1200, 'y': 400, 'lat': 17.443845, 'lng': 78.497357, 'name_alias': 'JBS Parade Ground', 'amenities': ['Bus Terminal Link', 'Blue Line Interchange', 'Ticket Vending', 'Restrooms', 'Information Center'], 'description': 'Northern terminal of the Green Line. Gateway to the northern bus hub of the city, connecting Secunderabad to North Telangana.'},
    {'id': 'G2', 'name': 'Secunderabad West', 'line': 'Green', 'x': 1205, 'y': 470, 'lat': 17.434119, 'lng': 78.499223, 'amenities': ['Railway Concourse Link', 'Cycle Stand', 'CCTV Support', 'ATM'], 'description': 'Vital for railway commuters, offering seamless transit to Secunderabad West station entrances.'},
    {'id': 'G3', 'name': 'Gandhi Hospital', 'line': 'Green', 'x': 1210, 'y': 540, 'lat': 17.425631, 'lng': 78.501819, 'description': 'Dedicated station for patients and staff of the Gandhi Medical Hospital.'},
    {'id': 'G4', 'name': 'Musheerabad', 'line': 'Green', 'x': 1215, 'y': 610, 'lat': 17.417875, 'lng': 78.499305, 'description': 'Serves the historical Musheerabad area and busy residential wards.'},
    {'id': 'G5', 'name': 'RTC X Roads', 'line': 'Green', 'x': 1220, 'y': 680, 'lat': 17.407373, 'lng': 78.496575, 'description': 'The cinema hub of the city, surrounded by iconic theaters and food joints.'},
    {'id': 'G6', 'name': 'Chikkadpally', 'line': 'Green', 'x': 1225, 'y': 750, 'lat': 17.400781, 'lng': 78.494846, 'description': 'A dense cultural and residential node known for its bookstores and libraries.'},
    {'id': 'G7', 'name': 'Narayanaguda', 'line': 'Green', 'x': 1230, 'y': 820, 'lat': 17.39546, 'lng': 78.490409, 'description': 'Key stop for the numerous educational institutes located around Narayanaguda.'},
    {'id': 'G8', 'name': 'Sultan Bazar', 'line': 'Green', 'x': 1235, 'y': 890, 'lat': 17.384045, 'lng': 78.483664, 'description': 'A major commercial hub serving the historic Sultan Bazar and Koti markets.'},
    {'id': 'G9', 'name': 'MG Bus Station', 'line': 'Green', 'x': 1240, 'y': 960, 'lat': 17.378583, 'lng': 78.485514, 'name_alias': 'MG Bus Station', 'description': 'Interchange with Red Line and direct access back to MGBS terminal.'}
]

CONNECTIONS = {
    'Red': [f'R{i}' for i in range(1, 28)],
    'Blue': [f'B{i}' for i in range(1, 24)],
    'Green': [f'G{i}' for i in range(1, 10)]
}

INTERCHANGE_DATA = {
    'Ameerpet': {
        'lines': ['Red', 'Blue'],
        'time_estimate': '3-5 mins',
        'platforms': [
            {'pair': 'Blue to Red', 'text': 'Downstairs to Platform 1 (LB Nagar) or 2 (Miyapur)'},
            {'pair': 'Red to Blue', 'text': 'Upstairs to Level 1, Platform 3 (Nagole) or 4 (Raidurg)'}
        ],
        'guidance': [
            "Arrive at Ameerpet (Level L2 for Blue, L1 for Red).",
            "Follow 'Interchange' signs on the floor or pillars.",
            "Use Escalators towards your target line (Blue/Red).",
            "Ameerpet is a bi-level station. Red Line is below Blue Line."
        ]
    },
    'MG Bus Station': {
        'lines': ['Red', 'Green'],
        'time_estimate': '4-6 mins',
        'platforms': [
            {'pair': 'Green to Red', 'text': 'Take Escalator UP to Red Line Level, Platform 1/2'},
            {'pair': 'Red to Green', 'text': 'Take Escalator DOWN to Green Line Level, Platform 3'}
        ],
        'guidance': [
            "MG Bus Station connects Red and Green corridors.",
            "Look for 'Connecting to Green Line' signs near center escalators.",
            "Green line is at the lowest level of this facility.",
            "Allow extra time as MGBS is one of the largest metro hubs."
        ]
    },
    'JBS Parade Ground': {
        'lines': ['Blue', 'Green'],
        'time_estimate': '2-3 mins',
        'platforms': [
            {'pair': 'Green to Blue', 'text': 'Stairs to Platform level 3 (Nagole) / 4 (Raidurg)'},
            {'pair': 'Blue to Green', 'text': 'Escalator DOWN to reach Green Line Platform Level'}
        ],
        'guidance': [
            "Transition between Secunderabad East area lines.",
            "Short walking distance but requires level change.",
            "Green line is the originating point here for the corridor.",
            "Follow floor markers to avoid confusion with exit gates."
        ]
    }
}

# ==========================================
# 3. CORE SYSTEM LOGIC
# ==========================================

GTFS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gtfs.csv')
RIDERSHIP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_metro_dataset.csv')
_GTFS_CACHE = None  # Global cache for performance
_GTFS_INDEX = {} # Map trip_id -> list of stop_times
_GTFS_STATION_INDEX = {} # Map station_id -> list of trips passing through

def convert_to_12h(t_str):
    """Converts 24h string to 12h AM/PM format."""
    try:
        h, m, s = map(int, t_str.split(':'))
        dt = datetime.now().replace(hour=h, minute=m, second=s)
        return dt.strftime('%I:%M %p')
    except: return t_str

def ensure_gtfs(force=False):
    """Generates a high-frequency, dynamic GTFS simulation."""
    global _GTFS_CACHE, _GTFS_INDEX, _GTFS_STATION_INDEX
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
                            writer.writerow([trip_id, sid, trip_start_time.strftime('%H:%M:%S'), platform, dir_type, final_name, line])
                            trip_start_time += timedelta(minutes=2)
                        
                        current_trip_time += timedelta(minutes=interval)
                        trip_idx += 1
        print("GTFS Generation Complete.")
        _GTFS_CACHE = None # Invalidate cache

    if _GTFS_CACHE is None:
        try:
            with open(GTFS_FILE, 'r') as f:
                _GTFS_CACHE = list(csv.DictReader(f))
            
            # Build indices for O(1) lookups
            _GTFS_INDEX = {}
            _GTFS_STATION_INDEX = {}
            for row in _GTFS_CACHE:
                tid = row['trip_id']
                sid = row['station_id']
                if tid not in _GTFS_INDEX: _GTFS_INDEX[tid] = []
                _GTFS_INDEX[tid].append(row)
                
                if sid not in _GTFS_STATION_INDEX: _GTFS_STATION_INDEX[sid] = []
                _GTFS_STATION_INDEX[sid].append(row)
            
            # Pre-sort by time
            for sid in _GTFS_STATION_INDEX:
                _GTFS_STATION_INDEX[sid].sort(key=lambda x: x['arrival_time'])
                
            print(f"GTFS cache indexed: {len(_GTFS_CACHE)} rows, {len(_GTFS_INDEX)} trips.")
        except Exception as e:
            print(f"Crit: GTFS Load Failed: {e}")
            _GTFS_CACHE = []
    
    return _GTFS_CACHE

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def get_fare_from_matrix(dist):
    if dist <= 2: return 12
    if dist <= 4: return 18
    if dist <= 6: return 30
    if dist <= 9: return 40
    if dist <= 12: return 50
    if dist <= 15: return 55
    if dist <= 18: return 60
    if dist <= 21: return 66
    if dist <= 24: return 70
    return 75

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
    initial_tab = request.args.get('tab', 'home')
    return render_template_string(HTML_TEMPLATE, ALL_STATIONS=STATIONS_LIST, GEMINI_API_KEY=GEMINI_API_KEY, initial_tab=initial_tab)

@app.route('/planner')
def planner_page():
    return render_template_string(HTML_TEMPLATE, ALL_STATIONS=STATIONS_LIST, GEMINI_API_KEY=GEMINI_API_KEY, initial_tab='routes')

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
    dist = 0
    
    if 'station_id' in data:
        nearest = next(s for s in STATIONS_LIST if s['id'] == data['station_id'])
        lat, lng = nearest['lat'], nearest['lng']
    else:
        lat, lng = data['lat'], data['lng']
        nearest = min(STATIONS_LIST, key=lambda s: haversine(lat, lng, s['lat'], s['lng']))
        dist = haversine(lat, lng, nearest['lat'], nearest['lng'])
    
    # Urban Walking Adjustment: approx 1.35x crow-flies distance for city streets
    walk_dist = dist * 1.35
    walking_mins = int((walk_dist / 5.0) * 60) # Assume 5km/h walking speed
    if walking_mins < 1: walking_mins = 1
    
    name = nearest.get('name_alias', nearest['name'])
    matching_ids = [s['id'] for s in STATIONS_LIST if s.get('name_alias', s['name']) == name]
    to_id = data.get('to_id')
    
    trips_data = ensure_gtfs()
    now = get_app_now()
    now_str = now.strftime('%H:%M:%S')
    one_hour_later = now + timedelta(hours=2) # Check 2 hours for better coverage
    oh_str = one_hour_later.strftime('%H:%M:%S')
    
    upcoming = []
    for mid in matching_ids:
        st_trips = _GTFS_STATION_INDEX.get(mid, [])
        for row in st_trips:
            # Direction Filter: If destination is selected, check if trip reaches it AFTER this station
            if to_id:
                trip_stops = _GTFS_INDEX.get(row['trip_id'], [])
                dest_stop = next((s for s in trip_stops if s['station_id'] == to_id), None)
                if not dest_stop or dest_stop['arrival_time'] <= row['arrival_time']:
                    continue

            # Handle midnight wrap-around for time window
            if now_str < row['arrival_time'] < oh_str or (oh_str < now_str and (row['arrival_time'] > now_str or row['arrival_time'] < oh_str)):
                # Calculate ETA countdown
                try:
                    ah, am, as_ = map(int, row['arrival_time'].split(':'))
                    # Determine if arrival is today or tomorrow (if oh_str < now_str and arrival < oh_str)
                    arrival_dt = now.replace(hour=ah, minute=am, second=as_, microsecond=0)
                    if oh_str < now_str and row['arrival_time'] < oh_str:
                        arrival_dt += timedelta(days=1)
                        
                    diff = (arrival_dt - now).total_seconds()
                    if diff < 0: continue
                    
                    m, s = divmod(int(diff), 60)
                    row_copy = row.copy()
                    row_copy['eta'] = f"{m:02d}:{s:02d}"
                    row_copy['sort_time'] = row['arrival_time']
                    # Keep original for sorting consistency
                    upcoming.append(row_copy)
                except: continue
    
    # Sort correctly by 24h time before conversion, awareness of midnight wrap
    # Logic: if sort_time is less than now_str, it means it's for 'tomorrow' in the 1h window (e.g. now 23:30, train 00:05)
    upcoming.sort(key=lambda x: (0 if x['sort_time'] >= now_str else 1, x['sort_time']))
    upcoming = upcoming[:10]
    
    # AI & Weather Data
    weather = get_live_weather(lat=lat, lng=lng)
    is_weekend = now.weekday() >= 5
    load_val, load_label = predict_load_ai(name, now.hour, is_weekend=is_weekend, weather=weather)
    
    # Optimized Global GTFS Stats (Trips active right now)
    trip_times = {}
    for row in trips_data:
        tid = row['trip_id']
        t_arr = row['arrival_time']
        if tid not in trip_times:
            trip_times[tid] = {'min': t_arr, 'max': t_arr}
        else:
            if t_arr < trip_times[tid]['min']: trip_times[tid]['min'] = t_arr
            if t_arr > trip_times[tid]['max']: trip_times[tid]['max'] = t_arr

    active_count = sum(1 for tid, times in trip_times.items() if times['min'] <= now_str <= times['max'])

    # Final conversion to 12h for the selected top 10
    for t in upcoming:
        t['arrival_time'] = convert_to_12h(t['arrival_time'])
    
    return jsonify({
        'station': nearest, 
        'distance': round(dist, 2),
        'walk_dist': round(walk_dist, 2),
        'walking_mins': walking_mins,
        'upcoming': upcoming, 
        'load_val': load_val, 
        'load_label': load_label,
        'active_trips': active_count,
        'weather': weather,
        'greeting': "Good Morning" if 5 <= now.hour < 12 else "Good Afternoon" if 12 <= now.hour < 17 else "Good Evening"
    })

@app.route('/api/weather', methods=['POST'])
def api_weather():
    data = request.json
    lat, lng = data.get('lat'), data.get('lng')
    weather = get_live_weather(lat=lat, lng=lng)
    return jsonify(weather)

@app.route('/api/plan', methods=['POST'])
def api_plan():
    data = request.json
    start_id, end_id = data['from'], data['to']
    user_prefs = data.get('prefs', {}) # {'comfort': bool, 'speed': bool, 'scenic': bool}
    now = get_app_now()
    
    # BFS find multiple paths (up to 3 distinct paths to provide alternatives)
    all_paths = []
    queue = [(start_id, [start_id])]
    
    if start_id == end_id:
        all_paths = [[start_id]]
    else:
        max_results = 3
        while queue and len(all_paths) < max_results:
            curr, p = queue.pop(0)
            if curr == end_id:
                all_paths.append(p)
                continue
            if len(p) > 25: continue
            
            neighbors = []
            for line in CONNECTIONS.values():
                if curr in line:
                    idx = line.index(curr)
                    if idx > 0: neighbors.append(line[idx-1])
                    if idx < len(line)-1: neighbors.append(line[idx+1])
            c_name = next((s['name'] for s in STATIONS_LIST if s['id'] == curr), None)
            if c_name:
                for s in STATIONS_LIST:
                    if s['name'] == c_name and s['id'] != curr: neighbors.append(s['id'])
            
            for n in neighbors:
                if n not in p:
                    queue.append((n, p + [n]))

    def get_route_details(path_ids):
        # We use copies to avoid mutating global station objects
        sequence = []
        for sid in path_ids:
            s_orig = next(s for s in STATIONS_LIST if s['id'] == sid)
            sequence.append(s_orig.copy())
            
        start_sid = sequence[0]['id']
        end_sid = sequence[-1]['id']
        
        gtfs_boarding_time = None
        chosen_trip_id = None
        stop_arrival_times = {} 
        
        try:
            possible_start_trips = [t for t in _GTFS_STATION_INDEX.get(start_sid, []) if t['arrival_time'] > now.strftime('%H:%M:%S')]
            for pt in possible_start_trips[:10]:
                trip_data = _GTFS_INDEX.get(pt['trip_id'], [])
                destination_stop = next((t for t in trip_data if t['station_id'] == end_sid), None)
                if destination_stop and destination_stop['arrival_time'] > pt['arrival_time']:
                    gtfs_boarding_time = pt['arrival_time']
                    chosen_trip_id = pt['trip_id']
                    for td in trip_data:
                        stop_arrival_times[td['station_id']] = td['arrival_time']
                    break
        except: pass
        if not gtfs_boarding_time: gtfs_boarding_time = now.strftime('%H:%M:%S')

        guides = []
        interchanges = 0
        total_duration_calculated = 0
        max_load = 0
        prev_dt = None
        try:
            h, m, s_ = map(int, gtfs_boarding_time.split(':'))
            prev_dt = now.replace(hour=h, minute=m, second=s_, microsecond=0)
        except: pass

        for i, s in enumerate(sequence):
            reaching_at_raw = stop_arrival_times.get(s['id'])
            current_dt = None
            reach_hour = now.hour
            if reaching_at_raw:
                try:
                    h, m, s_ = map(int, reaching_at_raw.split(':'))
                    reach_hour = h
                    current_dt = now.replace(hour=h, minute=m, second=s_)
                    s['reaching_at'] = current_dt.strftime('%I:%M %p')
                except:
                    s['reaching_at'] = reaching_at_raw
            else:
                if prev_dt:
                    current_dt = prev_dt + timedelta(minutes=2)
                    s['reaching_at'] = current_dt.strftime('%I:%M %p')
                    reach_hour = current_dt.hour
                else: s['reaching_at'] = "--:--"
            
            if i > 0:
                if prev_dt and current_dt:
                    total_duration_calculated += max(1, (current_dt - prev_dt).total_seconds() / 60)
                else: total_duration_calculated += 2

            # Load Prediction
            is_wknd = now.weekday() >= 5
            loc_weather = get_live_weather(lat=s['lat'], lng=s['lng'])
            _, load_pct = predict_load_ai(s['name'], reach_hour, is_weekend=is_wknd, weather=loc_weather)
            s['predicted_load'] = load_pct
            max_load = max(max_load, load_pct)
            
            # Guides
            if i < len(path_ids) - 1:
                s1 = sequence[i]
                s2 = sequence[i+1]
                if s1['name'] == s2['name'] and s1['id'] != s2['id']:
                    interchanges += 1
                    i_data = INTERCHANGE_DATA.get(s1['name'], {})
                    guides.append({
                        'station': s1['name'],
                        'platform': "Interchange Hub",
                        'text': i_data.get('guidance', ["Follow transfer signs"])[0],
                        'steps': i_data.get('guidance', []),
                        'time_estimate': i_data.get('time_estimate', '5 mins'),
                        'reaching_at': s['reaching_at']
                    })
            prev_dt = current_dt

        total_km = 0
        for i in range(1, len(sequence)):
            total_km += haversine(sequence[i-1]['lat'], sequence[i-1]['lng'], sequence[i]['lat'], sequence[i]['lng'])

        duration = int(total_duration_calculated) if total_duration_calculated > 0 else len(sequence)*2
        fare = get_fare_from_matrix(total_km)
        
        # Personalized Advices logic
        advices = []
        if max_load > 70:
            advices.append({'type': 'congestion', 'title': 'High Density Alert', 'text': 'Severe congestion predicted. Suggest first coach for less crowding.', 'icon': 'users'})
        if interchanges > 1:
            advices.append({'type': 'itinerary', 'title': 'Multi-Hub Route', 'text': 'Multiple transfers required. Visual guides active for platforms.', 'icon': 'shuffle'})
        if now.hour > 21:
            advices.append({'type': 'time', 'title': 'Night Protocol', 'text': 'Lower frequency service active. Stay near station security.', 'icon': 'moon'})

        return {
            'sequence': sequence,
            'duration': duration,
            'total_km': round(total_km, 2),
            'fare': fare,
            'load': round(max_load, 1),
            'interchanges': interchanges,
            'guides': guides,
            'chosen_trip_id': chosen_trip_id,
            'personalized_advices': advices,
            'eco': {
                'co2': round(total_km * 0.15, 2),
                'trees': round(total_km * 0.02, 3),
                'calories': int(total_km * 12)
            }
        }

    results = []
    for p_ids in all_paths:
        try:
            results.append(get_route_details(p_ids))
        except Exception as e:
            print(f"Error processing path: {e}")

    if not results:
        return jsonify({'status': 'no_trains', 'message': 'No viable vectors found for this trajectory.'})

    # Sort based on user preference if provided
    if user_prefs.get('comfort'):
        results.sort(key=lambda x: x['load'])
    elif user_prefs.get('speed'):
        results.sort(key=lambda x: x['duration'])
    
    # Enrichment for primary
    primary = results[0]
    primary['peak_intensity'] = round(35 + (30 if (7<=now.hour<=10 or 17<=now.hour<=21) else 0), 1)
    primary['recommendation'] = "Optimal Neural Trajectory"
    
    return jsonify({
        'status': 'success',
        'primary': primary,
        'alternatives': results[1:] if len(results) > 1 else []
    })

@app.route('/api/live-map')
def api_live_map():
    trips = ensure_gtfs()
    now_dt = get_app_now()
    now_str = now_dt.strftime('%H:%M:%S')
    
    active_trains = []
    
    # Use pre-built index for speed
    for tid, stops in _GTFS_INDEX.items():
        # Find if train is currently between two stops
        for i in range(len(stops) - 1):
            s1 = stops[i]
            s2 = stops[i+1]
            
            if s1['arrival_time'] <= now_str < s2['arrival_time']:
                # Calculate progress
                try:
                    t1_parts = list(map(int, s1['arrival_time'].split(':')))
                    t2_parts = list(map(int, s2['arrival_time'].split(':')))
                    
                    t1 = now_dt.replace(hour=t1_parts[0], minute=t1_parts[1], second=t1_parts[2])
                    t2 = now_dt.replace(hour=t2_parts[0], minute=t2_parts[1], second=t2_parts[2])
                    
                    total_duration = (t2 - t1).total_seconds()
                    elapsed = (now_dt - t1).total_seconds()
                    
                    # Realism: Dwell time at station (30 seconds)
                    dwell_time = 30
                    if elapsed < dwell_time:
                        progress = 0
                        is_at_station = True
                    else:
                        progress = max(0, min(1, (elapsed - dwell_time) / (total_duration - dwell_time)))
                        is_at_station = False

                    # Calculate speed
                    st1 = next((s for s in STATIONS_LIST if s['id'] == s1['station_id']), None)
                    st2 = next((s for s in STATIONS_LIST if s['id'] == s2['station_id']), None)
                    speed = 0
                    if st1 and st2 and total_duration > dwell_time:
                        dist = haversine(st1['lat'], st1['lng'], st2['lat'], st2['lng'])
                        speed = (dist / ((total_duration - dwell_time) / 3600.0))
                    
                    active_trains.append({
                        'trip_id': tid,
                        'line': s1['line'],
                        'from_id': s1['station_id'],
                        'to_id': s2['station_id'],
                        'progress': progress,
                        'is_at_station': is_at_station,
                        't1_epoch': t1.timestamp(),
                        'duration': total_duration,
                        'dwell_time': dwell_time,
                        'direction': s1['direction'],
                        'final_stop': s1['final_stop'],
                        'speed': round(speed, 1) if speed < 110 else 75
                    })
                    break # Only one segment per trip
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
    <script type="importmap">
      {
        "imports": {
          "@google/genai": "https://esm.run/@google/genai"
        }
      }
    </script>
    <script type="module">
      import { GoogleGenAI } from "@google/genai";
      window.GoogleGenAI = GoogleGenAI;
    </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script type="module">
        import { GoogleGenAI } from 'https://esm.run/@google/genai';
        window.GoogleGenAI = GoogleGenAI;
    </script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
        
        :root {
            --bg: #f8fafc;
            --card-bg: rgba(255, 255, 255, 0.9);
            --border: rgba(226, 232, 240, 0.8);
            --accent: #3b82f6;
        }

        body { 
            font-family: 'Plus Jakarta Sans', sans-serif; 
            background: var(--bg); 
            color: #0f172a; 
            overflow-x: hidden; 
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,10%,0.02) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,0.02) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,0.02) 0, transparent 50%);
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
            max-width: 500px;
            background: rgba(255, 255, 255, 0.9); 
            backdrop-filter: blur(24px);
            border-radius: 32px; 
            z-index: 5000;
            padding: 8px; 
            justify-content: space-evenly;
            align-items: center;
            box-shadow: 0 20px 50px -10px rgba(0,0,0,0.15);
            border: 1px solid rgba(255,255,255,1);
        }
        .mobile-link {
            display: flex; flex-direction: column; align-items: center; gap: 4px;
            color: #64748b; padding: 12px 10px; border-radius: 20px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer;
            flex: 1;
        }
        .mobile-link i { width: 22px; height: 22px; }
        .mobile-link span { font-size: 8px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; }
        .mobile-link.active { background: #2563eb; color: white; transform: translateY(-4px); box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.3); }
        .mobile-link:hover:not(.active) { color: #0f172a; background: rgba(0,0,0,0.04); }

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

        .highlighted-train-halo {
            pointer-events: none;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80px;
            height: 80px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.4) 0%, rgba(59, 130, 246, 0) 70%);
            border-radius: 50%;
            z-index: -1;
            animation: pulse-highlight 2s infinite;
        }

        @keyframes pulse-highlight {
            0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.5; }
            50% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.8; }
            100% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.5; }
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

        #metro-map { 
            background: #f1f5f9; 
            border-radius: 2rem;
            z-index: 10;
        }
        .leaflet-container { background: #f1f5f9 !important; }
        .leaflet-bar a { background-color: #ffffff !important; color: #475569 !important; border: 1px solid rgba(0,0,0,0.1) !important; }
        .station-label-marker { background: transparent; border: none; box-shadow: none; }
        .station-label-text {
            font-size: 9px;
            font-weight: 800;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-shadow: 0 1px 2px rgba(255,255,255,0.8);
            white-space: nowrap;
            pointer-events: none;
        }
        .train-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));
        }
        .train-shape-inner { transition: transform 0.1s linear; }
        .train-shape-inner.at-station {
            animation: station-pulse 2s infinite ease-in-out;
        }
        @keyframes station-pulse {
            0%, 100% { transform: scale(1); filter: brightness(1); }
            50% { transform: scale(1.05); filter: brightness(1.2); }
        }
        /* Map Satellite Mode styling */
        #tab-map .glass-card {
            background-color: #f8fafc;
            background-image: radial-gradient(#e2e8f0 1.5px, transparent 1.5px);
            background-size: 30px 30px;
            border: 1px solid rgba(226, 232, 240, 0.8);
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
            width: 100%; overflow: hidden; background: #0f172a; padding: 12px 0;
            border-radius: 20px; margin: 12px 0 32px 0; border: none;
            box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.3);
        }
        #neural-ticker {
            display: inline-block; white-space: nowrap;
            animation: ticker-kf 60s linear infinite;
        }
        .ticker-item {
            display: inline-block; padding: 0 40px; color: #94a3b8; font-size: 10px;
            font-weight: 800; text-transform: uppercase; letter-spacing: 2px;
            border-right: 1px solid rgba(255,255,255,0.1);
        }
        .ticker-item strong { color: #3b82f6; }
        @keyframes ticker-kf {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        .planner-opt-btn.active {
            background-color: #2563eb !important;
            color: white !important;
            box-shadow: 0 15px 30px -5px rgba(37, 99, 235, 0.4);
            transform: translateY(-2px);
        }
            font-weight: 800; text-transform: uppercase; letter-spacing: 0.15em;
        }
        .ticker-item span { color: #2563eb; margin-right: 8px; }
        @keyframes ticker-kf {
            0% { transform: translate3d(0, 0, 0); }
            100% { transform: translate3d(-100%, 0, 0); }
        }

        /* Train Animation & Movement Styles */
        .train-shape-inner {
            transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            will-change: transform;
            transform-origin: center center;
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
        }
        
        .train-icon {
            z-index: 1000 !important;
        }

        .at-station svg {
            animation: pulse-station 2s infinite ease-in-out;
        }

        @keyframes pulse-station {
            0% { filter: drop-shadow(0 0 0px rgba(59, 130, 246, 0)); transform: scale(1); }
            50% { filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.4)); transform: scale(1.05); }
            100% { filter: drop-shadow(0 0 0px rgba(59, 130, 246, 0)); transform: scale(1); }
        }

        /* Speed stretch effect */
        .moving-fast {
            transform: scaleX(1.1);
        }
        
        .station-node-geo {
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .station-node-geo:hover {
            r: 10;
            stroke-width: 5;
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
                <div class="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white shadow-lg"><i data-lucide="train-front" size="20"></i></div>
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
        <div onclick="showTab('home')" class="mobile-link" id="mob-home"><i data-lucide="home"></i><span>Home</span></div>
        <div onclick="showTab('map')" class="mobile-link" id="mob-map"><i data-lucide="map-pinned"></i><span>Map</span></div>
        <div onclick="showTab('routes')" class="mobile-link" id="mob-routes"><i data-lucide="route"></i><span>Planner</span></div>
        <div onclick="showTab('details')" class="mobile-link" id="mob-details"><i data-lucide="info"></i><span>Stations</span></div>
        <div onclick="showTab('history')" class="mobile-link" id="mob-history"><i data-lucide="qr-code"></i><span>Tickets</span></div>
    </div>

    <div class="main" id="main-content">
        <div id="tab-home" class="tab-content active">
            <header class="flex flex-col lg:flex-row lg:justify-between lg:items-start gap-8 mb-2">
                <div class="hidden lg:block">
                    <h2 id="greeting" class="text-4xl lg:text-5xl font-black text-slate-900 mb-2 tracking-tighter">Welcome!</h2>
                    <p id="env-msg" class="text-slate-400 font-bold max-w-sm leading-relaxed uppercase text-[10px] tracking-widest">Network is active. Have a great journey.</p>
                </div>
                <div class="lg:hidden w-full text-center">
                     <h2 id="greeting-mob" class="text-3xl font-black text-slate-900 mb-1 tracking-tighter">Hello!</h2>
                     <p id="near-metro-mob" class="text-[11px] font-black text-blue-600 uppercase tracking-widest mb-4">Finding Station...</p>
                     <div class="flex items-center justify-center gap-2">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                        <p class="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Live Updates On</p>
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

            <div class="ticker-wrap shadow-xl bg-slate-900 overflow-hidden relative border-none">
                <div class="absolute inset-y-0 left-0 w-20 bg-gradient-to-r from-slate-900 to-transparent z-10"></div>
                <div class="absolute inset-y-0 right-0 w-20 bg-gradient-to-l from-slate-900 to-transparent z-10"></div>
                <div id="neural-ticker" class="whitespace-nowrap flex py-2">
                    <!-- Dynamic items injected here -->
                </div>
            </div>

            <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-8 mb-8">
                <div class="glass-card flex p-4 lg:p-6 items-center gap-4 lg:gap-6 border-slate-200 overflow-hidden relative group">
                    <div class="w-10 h-10 lg:w-14 lg:h-14 bg-blue-50 text-blue-600 rounded-xl lg:rounded-2xl flex items-center justify-center shrink-0"><i data-lucide="crosshair" size="18"></i></div>
                    <div class="overflow-hidden">
                        <p class="text-[8px] lg:text-[9px] font-black text-slate-400 uppercase tracking-widest mb-0.5 lg:mb-1 flex items-center gap-2">
                             Live Near Metro <span id="near-dist" class="text-blue-500 font-bold border-l border-slate-200 pl-2">-- km</span>
                        </p>
                        <h3 id="near-name" class="text-[11px] lg:text-sm font-black text-slate-800 truncate">Locating...</h3>
                        <p id="near-walk-time" class="text-[8px] font-bold text-emerald-600 mt-0.5 uppercase tracking-widest hidden">-- min walk</p>
                    </div>
                    <div class="absolute right-2 top-2 flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-all">
                        <button onclick="manualRefreshGeo()" class="p-1.5 bg-white shadow-sm rounded-lg border border-slate-100 text-blue-600 hover:bg-blue-50" title="Refresh GPS">
                            <i data-lucide="refresh-cw" size="12"></i>
                        </button>
                        <button id="nav-btn" onclick="openGoogleMaps()" class="p-1.5 bg-blue-600 text-white shadow-sm rounded-lg hover:bg-blue-700 hidden" title="Google Maps Directions">
                            <i data-lucide="navigation" size="12"></i>
                        </button>
                    </div>
                </div>
                <div class="glass-card flex p-4 lg:p-6 items-center gap-4 lg:gap-6 border-slate-200">
                    <div class="w-10 h-10 lg:w-14 lg:h-14 bg-orange-50 text-orange-600 rounded-xl lg:rounded-2xl flex items-center justify-center shrink-0"><i data-lucide="waves" size="18"></i></div>
                    <div>
                        <p class="text-[8px] lg:text-[9px] font-black text-slate-400 uppercase tracking-widest mb-0.5 lg:mb-1">Atmosphere</p>
                        <h3 id="weather-val" class="text-[11px] lg:text-sm font-black text-slate-800">--°C</h3>
                        <p id="weather-detail" class="text-[7px] lg:text-[8px] font-bold text-slate-400 uppercase tracking-tight">Syncing Sky...</p>
                        <div id="weather-indicator" class="hidden"><span class="text-[8px] font-bold text-emerald-500 uppercase tracking-[0.2em] flex items-center gap-1 mt-1"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>Live</span></div>
                    </div>
                </div>
                <div class="glass-card col-span-2 hidden lg:flex p-6 items-center justify-between border-blue-100 bg-blue-600 text-white overflow-hidden relative">
                    <div class="absolute -right-10 -bottom-10 w-40 h-40 bg-white/10 rounded-full blur-3xl"></div>
                    <div class="flex items-center gap-6 relative z-10">
                        <div class="w-14 h-14 bg-white/20 text-white rounded-2xl flex items-center justify-center"><i data-lucide="activity" size="24"></i></div>
                        <div>
                            <p class="text-[9px] font-black text-white/70 uppercase tracking-widest mb-1">Network Status</p>
                            <h3 class="text-xl font-black text-white tracking-tighter">Trips Active: <span id="active-count" class="text-white tabular-nums">--</span></h3>
                        </div>
                    </div>
                    <div class="text-right relative z-10">
                         <span id="load-status" class="px-3 py-1 bg-white/20 rounded-lg text-[9px] font-black uppercase tracking-widest border border-white/20">System Status Active</span>
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
                                            <th class="pb-4 lg:pb-8">Line</th>
                                            <th class="pb-4 lg:pb-8">Station</th>
                                            <th class="pb-4 lg:pb-8">Time</th>
                                            <th class="pb-4 lg:pb-8 text-right">Status</th>
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

                    <!-- Saved Routes Section -->
                    <div id="fav-vectors-section" class="hidden animate-in fade-in duration-1000">
                        <div class="flex items-center justify-between mb-6 px-4">
                            <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 flex items-center gap-2">
                                <i data-lucide="star" size="14" class="text-amber-500"></i> Frequent Routes
                            </h4>
                        </div>
                        <div id="fav-vectors-list" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <!-- Injected via JS -->
                        </div>
                    </div>
                </div>
                <div class="space-y-8">
                    <div class="glass-card action-card bg-blue-600 text-white border-none p-10 overflow-hidden relative shadow-2xl shadow-blue-500/10">
                        <div class="absolute -right-5 -top-5 w-40 h-40 bg-white/10 rounded-full blur-[60px]"></div>
                        <i data-lucide="map-pinned" class="mb-8 opacity-60" size="32"></i>
                        <h4 class="text-xl font-black mb-3 relative z-10 tracking-tight">Book Metro Ticket</h4>
                        <p class="text-xs text-white/70 mb-10 relative z-10 font-bold uppercase tracking-widest">Select your destination and secure a digital token instantly.</p>
                        <button onclick="showTab('routes')" class="w-full py-5 bg-white text-blue-600 rounded-2xl font-black text-[11px] uppercase tracking-widest shadow-xl flex items-center justify-center gap-3 relative z-10">
                            <i data-lucide="route" size="14"></i> Open Planner
                        </button>
                    </div>
                    <div class="glass-card action-card border-none p-10 bg-white overflow-hidden relative shadow-2xl shadow-slate-200/50">
                        <div class="absolute -right-5 -top-5 w-40 h-40 bg-blue-50 rounded-full blur-[60px]"></div>
                        <i data-lucide="scan" class="mb-8 text-blue-600" size="32"></i>
                        <h4 class="text-xl font-black mb-3 relative z-10 tracking-tight text-slate-900">Smart Card</h4>
                        <p class="text-xs text-slate-400 mb-10 relative z-10 font-bold uppercase tracking-widest">Recharge your metro card instantly from your phone.</p>
                        <button class="w-full py-5 bg-blue-600 text-white rounded-2xl font-black text-[11px] uppercase tracking-widest">Quick Recharge</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- NETWORK MAP -->
        <div id="tab-map" class="tab-content h-full">
            <div id="interchange-modal" class="hidden fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-400/20 backdrop-blur-sm animate-in fade-in duration-300">
                <div class="glass-card bg-white w-full max-w-lg p-0 overflow-hidden shadow-2xl border-none">
                    <div id="modal-header" class="p-8 pb-4 flex justify-between items-start">
                        <div>
                            <h2 id="modal-title" class="text-2xl font-black text-slate-900 tracking-tight">Interchange Hub</h2>
                            <p id="modal-subtitle" class="text-[10px] font-black text-blue-600 uppercase tracking-[0.2em] mt-1">Multi-Level Terminal Guidance</p>
                        </div>
                        <button onclick="closeInterchangeModal()" class="p-2 hover:bg-slate-100 rounded-xl transition-all">
                            <i data-lucide="x" size="20"></i>
                        </button>
                    </div>
                    <div id="modal-content" class="p-8 pt-0 space-y-6">
                        <div id="line-badges" class="flex gap-2 mb-6"></div>
                        <div class="space-y-4">
                            <h4 class="text-[9px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                                <i data-lucide="layers" size="14"></i> Platform Matrix
                            </h4>
                            <div id="platform-info" class="space-y-2">
                                <!-- Platform data injected -->
                            </div>
                        </div>
                        <div class="space-y-4">
                            <h4 class="text-[9px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                                <i data-lucide="shuffle" size="14"></i> Transfer Protocol
                            </h4>
                            <div id="transfer-guidance" class="bg-slate-50 p-5 rounded-2xl border border-slate-100 space-y-4">
                                <!-- Guidance steps will be injected here -->
                            </div>
                        </div>
                    </div>
                    <div class="p-6 bg-slate-50 text-slate-800 flex justify-between items-center mt-4 border-t border-slate-200">
                        <div class="flex items-center gap-3">
                            <i data-lucide="info" class="text-blue-400" size="16"></i>
                            <span class="text-[9px] font-black uppercase tracking-widest italic">Live station map available via terminal kiosk</span>
                        </div>
                        <button onclick="closeInterchangeModal()" class="px-5 py-2.5 bg-white/10 hover:bg-white/20 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all">Acknowledge</button>
                    </div>
                </div>
            </div>

            <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-6 mb-8">
                <div>
                    <h2 class="text-3xl font-black text-slate-900 tracking-tight">Live Station Map</h2>
                    <p class="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mt-1">Real-time train tracking active</p>
                </div>
                <div class="relative group w-full lg:w-96" id="map-search-container">
                    <div class="absolute inset-y-0 left-4 flex items-center pointer-events-none text-slate-400 group-focus-within:text-blue-600 transition-colors">
                        <i data-lucide="search" size="18"></i>
                    </div>
                    <input type="text" id="map-search" placeholder="Search Neural Node (e.g. Miyapur)..." 
                        class="w-full pl-12 pr-12 py-4 bg-white border-2 border-slate-100 rounded-2xl text-sm font-bold text-slate-700 outline-none focus:border-blue-600 focus:ring-4 focus:ring-blue-100 transition-all shadow-sm"
                        oninput="filterMapStations(this.value)"
                        onfocus="showMapSuggestions(this.value)">
                    <div class="absolute right-4 top-1/2 -translate-y-1/2 flex gap-2">
                        <button id="search-clear" onclick="clearMapSearch()" class="hidden p-1 hover:bg-slate-100 rounded-lg text-slate-400 transition-all">
                            <i data-lucide="circle-x" size="16"></i>
                        </button>
                        <span class="hidden lg:block px-2 py-1 bg-slate-50 border border-slate-100 rounded-md text-[8px] font-black text-slate-400 uppercase">⌘ K</span>
                    </div>
                    <!-- Search Suggestions -->
                    <div id="map-suggestions" class="hidden absolute top-full left-0 right-0 mt-2 bg-white rounded-2xl border border-slate-100 shadow-2xl z-[110] max-h-80 overflow-y-auto p-2 scrollbar-none">
                        <!-- Suggestions appended here -->
                    </div>
                </div>
            </div>
            <div class="glass-card p-0 relative h-[800px] overflow-hidden bg-white border-slate-200 shadow-inner">
                <div id="metro-map" class="w-full h-full"></div>
                <div id="map-overlay" class="absolute top-0 right-0 h-full w-full lg:w-[400px] translate-x-full z-[1000] transition-transform duration-500 ease-in-out bg-white shadow-[-20px_0_50px_-10px_rgba(0,0,0,0.1)] border-l border-slate-100">
                    <div class="h-full flex flex-col p-6 lg:p-10 overflow-hidden relative">
                        <div class="absolute top-6 right-6 lg:top-8 lg:right-8 z-20">
                            <button onclick="closeOverlay()" class="p-3 bg-slate-50 hover:bg-slate-100 text-slate-400 rounded-2xl transition-colors">
                                <i data-lucide="x" size="20"></i>
                            </button>
                        </div>

                        <div class="mb-10 px-2 lg:px-0">
                            <div class="flex items-center gap-2 mb-4">
                                <span id="ov-line" class="px-3 py-1 text-[10px] font-black uppercase rounded-lg">LINE</span>
                                <div id="ov-weather" class="hidden flex items-center gap-1.5 px-3 py-1 bg-emerald-50 text-emerald-600 text-[10px] font-black uppercase rounded-lg border border-emerald-100">
                                    <i data-lucide="cloud-sun" size="10"></i>
                                    <span id="ov-weather-val">--°C</span>
                                </div>
                            </div>
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
                        </div>

                        <div class="pt-8 border-t border-slate-100 mt-auto">
                            <button id="ov-plan-btn" class="w-full py-5 bg-blue-600 text-white rounded-2xl font-black text-[11px] uppercase tracking-widest flex items-center justify-center gap-3 shadow-xl shadow-blue-200">
                                <i data-lucide="navigation" size="14"></i> Set as Destination
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        {% if not hide_planner %}
        <!-- SMART PLANNER -->
        <div id="tab-routes" class="tab-content">
            <div class="max-w-4xl mx-auto">
                <div class="text-center mb-4">
                   <h2 class="text-4xl lg:text-5xl font-black tracking-tighter mb-1 text-slate-900">Trip Planner</h2>
                   <p class="text-[10px] font-bold text-slate-400 uppercase tracking-[0.4em] flex items-center justify-center gap-2"><i data-lucide="shield-check" size="14" class="text-emerald-500"></i> Always on time</p>
                </div>
                
                <div id="planner-input-area" class="glass-card border-none shadow-2xl bg-white p-8 lg:p-12 relative overflow-hidden group mb-8">
                    <div class="absolute -right-20 -top-20 w-96 h-96 bg-blue-50/50 rounded-full blur-[100px] transition-all group-hover:bg-blue-100/30"></div>
                    
                    <!-- Input Matrix -->
                    <div class="grid grid-cols-1 md:grid-cols-11 items-center gap-2 mb-6">
                        <div class="md:col-span-2 space-y-3">
                            <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1">Travel Time</label>
                            <div class="relative">
                                <input type="time" id="plan-time" onchange="autoPlan()" class="w-full pl-6 pr-6 py-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-indigo-500/30 focus:bg-white font-black text-slate-900 transition-all cursor-pointer hover:bg-slate-100">
                            </div>
                        </div>

                        <div class="md:col-span-4 space-y-3">
                            <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1">From Station</label>
                            <div class="relative">
                                <div class="absolute left-6 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-blue-600 ring-8 ring-blue-50"></div>
                                <select id="start-st" onchange="autoPlan()" class="w-full pl-14 pr-8 py-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-blue-500/30 focus:bg-white font-black appearance-none text-slate-900 transition-all cursor-pointer hover:bg-slate-100"></select>
                                <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none"><i data-lucide="chevron-down" size="18"></i></div>
                            </div>
                        </div>

                        <div class="md:col-span-1 flex justify-center py-4 md:py-0 text-center relative">
                            <button onclick="swapStations(); autoPlan();" class="p-4 bg-blue-600 text-white rounded-full shadow-2xl hover:rotate-180 transition-transform duration-700 cursor-pointer border-4 border-white z-20">
                                <i data-lucide="repeat-2" size="20"></i>
                            </button>
                        </div>

                        <div class="md:col-span-4 space-y-3">
                            <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1">To Station</label>
                            <div class="relative">
                                <div class="absolute left-6 top-1/2 -translate-y-1/2 text-emerald-500"><i data-lucide="map-pin" size="18"></i></div>
                                <select id="end-st" onchange="autoPlan()" class="w-full pl-14 pr-8 py-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-emerald-500/30 focus:bg-white font-black appearance-none text-slate-900 transition-all cursor-pointer hover:bg-slate-100"></select>
                                <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none"><i data-lucide="chevron-down" size="18"></i></div>
                            </div>
                        </div>
                    </div>

                    <!-- User Preferences Segment -->
                    <div class="flex flex-wrap items-center gap-4 px-2">
                        <span class="text-[9px] font-black uppercase text-slate-400 tracking-widest mr-2">Route Optimization:</span>
                        
                        <label class="flex items-center gap-2 cursor-pointer group">
                            <input type="checkbox" id="pref-comfort" onchange="autoPlan()" class="hidden peer">
                            <div class="px-4 py-2 bg-slate-50 border border-slate-100 rounded-xl text-[10px] font-black text-slate-400 uppercase tracking-widest peer-checked:bg-blue-50 peer-checked:border-blue-200 peer-checked:text-blue-600 transition-all group-hover:bg-slate-100">
                                Low Crowds
                            </div>
                        </label>
                        
                        <label class="flex items-center gap-2 cursor-pointer group">
                            <input type="checkbox" id="pref-speed" onchange="autoPlan()" class="hidden peer">
                            <div class="px-4 py-2 bg-slate-50 border border-slate-100 rounded-xl text-[10px] font-black text-slate-400 uppercase tracking-widest peer-checked:bg-emerald-50 peer-checked:border-emerald-200 peer-checked:text-emerald-600 transition-all group-hover:bg-slate-100">
                                Fast Path
                            </div>
                        </label>
                        
                        <label class="flex items-center gap-2 cursor-pointer group">
                            <input type="checkbox" id="pref-scenic" onchange="autoPlan()" class="hidden peer">
                            <div class="px-4 py-2 bg-slate-50 border border-slate-100 rounded-xl text-[10px] font-black text-slate-400 uppercase tracking-widest peer-checked:bg-amber-50 peer-checked:border-amber-200 peer-checked:text-amber-600 transition-all group-hover:bg-slate-100">
                                Scenic Views
                            </div>
                        </label>
                    </div>

                    </div>
                </div>

                <div id="route-output" class="hidden space-y-6 pb-12">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-3">
                            <button onclick="saveCurrentVector()" id="save-vector-btn" class="flex items-center gap-2 px-6 py-3 bg-slate-100 text-slate-700 rounded-2xl text-[10px] font-black uppercase tracking-widest shadow-sm hover:bg-slate-200 transition-all">
                                <i data-lucide="star" size="14"></i> Save Route
                            </button>
                            <button onclick="shareRoute()" class="p-3 bg-slate-100 rounded-2xl text-slate-400 hover:text-slate-600 transition-all"><i data-lucide="share-2" size="18"></i></button>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            <span class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Live Updates</span>
                        </div>
                    </div>

                    <!-- Metrics Grid -->
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div class="bg-white p-5 rounded-[28px] border border-slate-100 shadow-sm flex flex-col gap-1">
                            <span class="text-[9px] font-black uppercase tracking-widest text-slate-400">Time</span>
                            <span id="route-dur" class="text-xl font-black text-slate-900">--</span>
                        </div>
                        <div class="bg-white p-5 rounded-[28px] border border-slate-100 shadow-sm flex flex-col gap-1">
                            <span class="text-[9px] font-black uppercase tracking-widest text-slate-400">Distance</span>
                            <span id="route-dist-km" class="text-xl font-black text-slate-900">-- KM</span>
                        </div>
                        <div class="bg-blue-600 p-5 rounded-[28px] shadow-lg flex flex-col gap-1 text-white relative overflow-hidden group/fare">
                            <div class="absolute -right-2 -top-2 w-16 h-16 bg-white/5 rounded-full blur-xl group-hover/fare:bg-white/10 transition-all"></div>
                            <div class="flex items-center justify-between relative z-10">
                                <span class="text-[9px] font-black uppercase tracking-widest opacity-70">Estimated Fare</span>
                                <div class="flex items-center gap-1.5 px-2 py-1 bg-white/10 rounded-lg border border-white/10 backdrop-blur-sm">
                                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]"></span>
                                    <span class="text-[7px] font-black uppercase tracking-widest opacity-90">Real-time</span>
                                </div>
                            </div>
                            <span id="route-fare" class="text-2xl font-black relative z-10 flex items-baseline gap-1">₹<span id="route-fare-val">--</span></span>
                        </div>
                        <div class="bg-slate-900 p-5 rounded-[28px] shadow-lg flex flex-col gap-1 text-white">
                            <span class="text-[9px] font-black uppercase tracking-widest opacity-70">Ridership</span>
                            <div class="flex items-baseline gap-2">
                                <span id="route-load-val" class="text-xl font-black">--%</span>
                                <span id="route-load-label" class="text-[8px] font-black uppercase opacity-60">Wait...</span>
                            </div>
                        </div>
                    </div>

                    <!-- Station Insight -->
                    <div id="route-rec-container" class="bg-indigo-50 p-6 rounded-[32px] border border-indigo-100 mb-6 flex items-start gap-4">
                        <div class="w-10 h-10 bg-indigo-600 text-white rounded-2xl flex items-center justify-center shrink-0 shadow-lg">
                            <i data-lucide="info" size="20"></i>
                        </div>
                        <div>
                            <h4 class="text-[10px] font-black text-indigo-900 uppercase tracking-widest mb-1">Transit Advisory</h4>
                            <p id="route-rec" class="text-sm font-bold text-indigo-800 leading-tight">--</p>
                        </div>
                    </div>

                    <div id="personalized-recommendations-wrapper" class="hidden mt-8 mb-6">
                        <h5 class="text-[11px] font-black uppercase text-indigo-400 tracking-widest pl-2 mb-4 flex items-center gap-2">
                            <i data-lucide="sparkles" size="14"></i> Personalized AI Intelligence
                        </h5>
                        <div id="personalized-recommendations" class="space-y-3">
                            <!-- Dynamic Advices -->
                        </div>
                    </div>

                    <div id="route-transfer-container" class="hidden mt-8 mb-6">
                        <h5 class="text-[11px] font-black uppercase text-blue-600 tracking-widest pl-2 mb-4 flex items-center gap-2">
                            <i data-lucide="shuffle" size="14"></i> Transfer Intelligence
                        </h5>
                        <div id="route-transfer-list" class="space-y-4">
                            <!-- Dynamic Transfers -->
                        </div>
                    </div>

                    <div id="track-vector-container" class="hidden mt-6 mb-6">
                        <!-- Tracking Card Injection Point -->
                    </div>

                    <!-- Station Stop List -->
                    <div class="bg-white rounded-[32px] border border-slate-100 shadow-xl overflow-hidden mt-6">
                        <div class="p-6 bg-slate-50 border-b border-slate-100 flex items-center justify-between">
                            <h4 class="text-[10px] font-black uppercase tracking-widest text-slate-400">Intermediate Stations</h4>
                            <div class="flex items-center gap-2 text-slate-300">
                                <i data-lucide="route" size="12"></i>
                                <span class="text-[9px] font-black uppercase tracking-widest">Network Flow</span>
                            </div>
                        </div>
                        <div id="route-stops-list" class="p-6 space-y-4">
                            <!-- Injected Stations -->
                        </div>
                    </div>

                    <!-- Alternatives Container -->
                    <div id="route-alternatives-wrapper" class="hidden mt-10 mb-8 border-t border-slate-100 pt-10">
                        <h5 class="text-[11px] font-black uppercase text-slate-400 tracking-widest pl-2 mb-6 flex items-center gap-2">
                            <i data-lucide="split" size="14"></i> AI Suggested Alternatives
                        </h5>
                        <div id="alternatives-list" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <!-- Injected by JS -->
                        </div>
                    </div>

                </div>

                <!-- Digital Ticket & Payment Hub -->
                    <div class="glass-card p-10 border-none bg-blue-600 text-white relative overflow-hidden">
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
        {% endif %}

        <div id="tab-history" class="tab-content">
            <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-12 gap-6 border-b pb-8 border-slate-200">
                <div class="flex flex-col items-center lg:flex-row lg:items-center gap-6 text-center lg:text-left">
                    <div class="w-20 h-20 bg-slate-900 rounded-[32px] flex items-center justify-center text-white shadow-2xl shadow-slate-200"><i data-lucide="vault" size="40"></i></div>
                    <div>
                        <h2 class="text-4xl lg:text-5xl font-black tracking-tighter mb-1 text-slate-900">Digital Vault</h2>
                        <p class="text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]">Neural Authorized Journey Ledger</p>
                    </div>
                </div>
                <div class="flex items-center gap-4">
                    <button onclick="clearHistory()" class="px-6 py-3 bg-red-50 text-red-600 rounded-2xl text-[10px] font-black uppercase tracking-widest border border-red-100 hover:bg-red-100 transition-all">Destroy Ledger</button>
                    <button onclick="showTab('routes')" class="px-6 py-3 bg-blue-600 text-white rounded-2xl text-[10px] font-black uppercase tracking-widest shadow-xl flex items-center gap-2">
                        <i data-lucide="plus" size="14"></i> New Token
                    </button>
                </div>
            </div>

            <div class="max-w-4xl mx-auto">
                <div id="unified-tickets-list" class="space-y-6 mb-20">
                    <!-- Tickets & History merged list -->
                </div>

                <div id="history-empty" class="hidden py-40 text-center flex flex-col items-center justify-center text-slate-300 bg-white rounded-[50px] border-2 border-dashed border-slate-100 shadow-sm">
                    <div class="w-24 h-24 bg-slate-50 rounded-full flex items-center justify-center mb-8 border border-slate-100">
                        <i data-lucide="folder-search" size="48" class="opacity-20 text-slate-400"></i>
                    </div>
                    <h4 class="text-2xl font-black text-slate-400 mb-2 tracking-tighter">Vault is Silent</h4>
                    <p class="text-[10px] font-black uppercase tracking-[0.4em] opacity-40">No boarding vectors detected in matrix</p>
                </div>
            </div>

            <!-- Sentiment Engine (Visible only in History Tab) -->
            <div id="sentiment-section" class="mt-12 pt-12 border-t border-slate-100 pb-20 hidden">
                <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-8 gap-6">
                    <div class="flex items-center gap-6">
                        <div class="w-16 h-16 bg-blue-600 rounded-3xl flex items-center justify-center text-white shadow-xl shadow-blue-500/30"><i data-lucide="message-square" size="32"></i></div>
                        <div>
                           <h2 class="text-4xl font-black tracking-tight mb-1 text-slate-900">Sentiment Engine</h2>
                           <p class="text-xs font-bold text-slate-400 uppercase tracking-widest">Your input optimizes the neural network</p>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    <div class="lg:col-span-12">
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

                                        <button onclick="submitFeedback()" class="w-full py-6 bg-blue-600 text-white font-black rounded-[30px] text-[11px] uppercase tracking-[0.3em] shadow-2xl hover:bg-blue-700 transition-all flex items-center justify-center gap-3">
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
            </div>
        </div>

        <div id="tab-details" class="tab-content">
            <div class="max-w-6xl mx-auto px-4">
                <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-12 gap-6">
                    <div class="flex items-center gap-6">
                        <div class="w-16 h-16 bg-blue-600 rounded-[28px] flex items-center justify-center text-white shadow-2xl shadow-blue-500/20"><i data-lucide="info" size="32"></i></div>
                        <div>
                            <h2 class="text-4xl font-black tracking-tight text-slate-900 mb-1">Station Directory</h2>
                            <p class="text-xs font-bold text-slate-400 uppercase tracking-widest">Network Knowledge Matrix</p>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-12">
                    <div class="bg-white p-6 rounded-[28px] border border-slate-100 shadow-sm flex items-center gap-5">
                        <div class="w-12 h-12 bg-red-50 text-red-600 rounded-2xl flex items-center justify-center shrink-0">
                            <i data-lucide="route" size="24"></i>
                        </div>
                        <div>
                            <p class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Network Span</p>
                            <p class="text-xl font-black text-slate-900">72.4 KM</p>
                        </div>
                    </div>
                    <div class="bg-white p-6 rounded-[28px] border border-slate-100 shadow-sm flex items-center gap-5">
                        <div class="w-12 h-12 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center shrink-0">
                            <i data-lucide="layers" size="24"></i>
                        </div>
                        <div>
                            <p class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Active Lines</p>
                            <p class="text-xl font-black text-slate-900">3 Specialized</p>
                        </div>
                    </div>
                    <div class="bg-white p-6 rounded-[28px] border border-slate-100 shadow-sm flex items-center gap-5">
                        <div class="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-2xl flex items-center justify-center shrink-0">
                            <i data-lucide="users" size="24"></i>
                        </div>
                        <div>
                            <p class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Daily Flux</p>
                            <p class="text-xl font-black text-slate-900">~600K Leads</p>
                        </div>
                    </div>
                    <div class="bg-white p-6 rounded-[28px] border border-slate-100 shadow-sm flex items-center gap-5">
                        <div class="w-12 h-12 bg-amber-50 text-amber-600 rounded-2xl flex items-center justify-center shrink-0">
                            <i data-lucide="zap" size="24"></i>
                        </div>
                        <div>
                            <p class="text-[10px] font-black text-slate-400 uppercase tracking-widest">AI Readiness</p>
                            <p class="text-xl font-black text-slate-900">100% Sync</p>
                        </div>
                    </div>
                </div>

                <div class="flex flex-col md:flex-row gap-4 mb-10">
                    <div class="relative flex-1">
                        <div class="absolute inset-y-0 left-4 flex items-center pointer-events-none text-slate-400">
                            <i data-lucide="search" size="18"></i>
                        </div>
                        <input type="text" id="directory-search" placeholder="Search station name..." 
                            class="w-full pl-12 pr-4 py-4 bg-white border-2 border-slate-100 rounded-2xl text-sm font-bold text-slate-700 outline-none focus:border-blue-600 transition-all shadow-sm"
                            oninput="renderStationDirectory()">
                    </div>
                    <div class="relative w-full md:w-64">
                        <div class="absolute inset-y-0 left-4 flex items-center pointer-events-none text-slate-400">
                            <i data-lucide="layers" size="18"></i>
                        </div>
                        <select id="directory-line-filter" 
                            class="w-full pl-12 pr-8 py-4 bg-white border-2 border-slate-100 rounded-2xl text-sm font-black text-slate-700 outline-none focus:border-blue-600 appearance-none cursor-pointer shadow-sm uppercase tracking-widest"
                            onchange="renderStationDirectory()">
                            <option value="all">All Lines</option>
                            <option value="Red">Red Line</option>
                            <option value="Blue">Blue Line</option>
                            <option value="Green">Green Line</option>
                        </select>
                        <div class="absolute right-4 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none">
                            <i data-lucide="chevron-down" size="18"></i>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-20" id="station-details-list">
                    <!-- Injected via JS -->
                </div>

                <!-- Fare Structure Information -->
                <div id="fare-structure" class="mt-20 border-t border-slate-100 pt-20 pb-10">
                    <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-900 mb-10 flex items-center gap-2">
                        <i data-lucide="calculator" class="text-blue-600" size="14"></i> Official Fare Matrix (2026)
                    </h4>
                    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">0-2 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹12</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">2-4 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹18</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">4-6 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹30</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">6-9 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹40</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">9-12 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹50</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">12-15 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹55</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">15-18 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹60</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">18-21 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹66</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">21-24 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹70</p>
                        </div>
                        <div class="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm text-center transform transition-transform hover:scale-105">
                            <p class="text-[8px] font-black text-slate-400 uppercase mb-2 tracking-widest">> 24 KM</p>
                            <p class="text-2xl font-black text-blue-600">₹75</p>
                        </div>
                    </div>
                    <div class="bg-blue-50 p-6 rounded-[32px] mt-10 border border-blue-100 flex items-center gap-6">
                        <div class="w-12 h-12 bg-blue-600 text-white rounded-2xl flex items-center justify-center shrink-0">
                            <i data-lucide="contactless" size="24"></i>
                        </div>
                        <p class="text-[10px] font-bold text-blue-800 uppercase tracking-[0.1em] leading-relaxed">
                            <strong class="text-blue-900">Smart Choice:</strong> A 10% discount is automatically applied to all fares when using a validated Hyderabad Metro Smart Card or Digital QR ticket.
                        </p>
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
        let lastUserLoc = null;
        let lastStationLoc = null;
        let weatherInterval = null;
        let notificationInterval = null;
        let plannerOpt = 'speed';
        let plannerSimTime = -1;
        let trainStates = new Map(); // Store live train data for smooth interpolation
        let trainMarkers = new Map(); // Store Leaflet marker objects
        let map = null;
        let stationMarkers = [];
        let linePolylines = {};
        let userMarker = null;
        let trainAnimationId = null;
        let tabState = 'home';
        let followingTrainId = null; // Global for "Follow mode"
        let plannedRoutePolyline = null;
        let routeMarkers = [];

        function trackTrain(tripId) {
            followingTrainId = tripId;
            showTab('map');
            
            // Highlight the specific marker
            trainMarkers.forEach((m, tid) => {
                const el = m.getElement();
                if (el) {
                    if (tid === tripId) {
                        el.style.filter = 'drop-shadow(0 0 15px rgba(59, 130, 246, 0.8))';
                        el.classList.add('tracking-active');
                    } else {
                        el.style.filter = 'grayscale(0.5) opacity(0.5)';
                        el.classList.remove('tracking-active');
                    }
                }
            });
            
            // Add a "Stop Tracking" floating button if not exists
            if (!document.getElementById('stop-tracking-ctrl')) {
                const ctrl = document.createElement('div');
                ctrl.id = 'stop-tracking-ctrl';
                ctrl.className = "fixed bottom-24 left-1/2 -translate-x-1/2 z-[2000] bg-slate-900 text-white px-6 py-3 rounded-full shadow-2xl flex items-center gap-3 animate-in slide-in-from-bottom-4 duration-500 border border-slate-700";
                ctrl.innerHTML = `
                    <div class="flex items-center gap-2">
                        <div class="w-2 h-2 bg-blue-500 rounded-full animate-ping"></div>
                        <span class="text-[10px] font-black uppercase tracking-widest">Tracking Live Vector</span>
                    </div>
                    <button onclick="stopTracking()" class="pl-4 border-l border-white/20 text-[10px] font-black text-red-400 uppercase tracking-widest hover:text-red-300">Stop</button>
                `;
                document.body.appendChild(ctrl);
            }
        }

        function stopTracking() {
            followingTrainId = null;
            const ctrl = document.getElementById('stop-tracking-ctrl');
            if (ctrl) ctrl.remove();
            
            trainMarkers.forEach((m) => {
                const el = m.getElement();
                if (el) {
                    el.style.filter = '';
                    el.classList.remove('tracking-active');
                }
            });
        }

        function initLeafletMap() {
            if (map) return;
            
            // Center on Hyderabad (Ameerpet area approx)
            map = L.map('metro-map', {
                center: [17.4374, 78.4482],
                zoom: 13,
                zoomControl: false,
                attributionControl: false
            });

            // Light Modern Tiles (CartoDB Voyager)
            L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
                maxZoom: 19
            }).addTo(map);

            L.control.zoom({ position: 'bottomright' }).addTo(map);

            // Draw Lines
            const redCoords = stations.filter(s => s.line === 'Red').sort((a,b) => parseInt(a.id.replace('R','')) - parseInt(b.id.replace('R',''))).map(s => [s.lat, s.lng]);
            const blueCoords = stations.filter(s => s.line === 'Blue').sort((a,b) => parseInt(a.id.replace('B','')) - parseInt(b.id.replace('B',''))).map(s => [s.lat, s.lng]);
            const greenCoords = stations.filter(s => s.line === 'Green').sort((a,b) => parseInt(a.id.replace('G','')) - parseInt(b.id.replace('G',''))).map(s => [s.lat, s.lng]);

            linePolylines['Red'] = L.polyline(redCoords, { color: '#ef4444', weight: 4, opacity: 0.8, lineCap: 'round' }).addTo(map);
            linePolylines['Blue'] = L.polyline(blueCoords, { color: '#3b82f6', weight: 4, opacity: 0.8, lineCap: 'round' }).addTo(map);
            linePolylines['Green'] = L.polyline(greenCoords, { color: '#22c55e', weight: 4, opacity: 0.8, lineCap: 'round' }).addTo(map);

            // Add Stations
            stations.forEach(s => {
                const color = s.line === 'Red' ? '#ef4444' : s.line === 'Blue' ? '#3b82f6' : '#22c55e';
                
                const marker = L.circleMarker([s.lat, s.lng], {
                    radius: 6,
                    fillColor: '#fff',
                    color: color,
                    weight: 3,
                    opacity: 1,
                    fillOpacity: 1,
                    className: 'station-node-geo'
                }).addTo(map);

                marker.on('click', () => {
                    handleStationClick(s);
                });

                // Station Labels (Only high zoom)
                const labelIcon = L.divIcon({
                    className: 'station-label-marker',
                    html: `<div class="station-label-text" style="margin-left: 12px; margin-top: -6px;">${s.name}</div>`,
                    iconSize: [0, 0]
                });
                const label = L.marker([s.lat, s.lng], { icon: labelIcon, interactive: false }).addTo(map);
                
                stationMarkers.push({ id: s.id, marker, label });
            });

            map.on('zoomend', () => {
                const zoom = map.getZoom();
                stationMarkers.forEach(sm => {
                    if (zoom < 13) map.removeLayer(sm.label);
                    else if (!map.hasLayer(sm.label)) map.addLayer(sm.label);
                });
            });
        }

        async function handleStationClick(s) {
            const overlay = document.getElementById('map-overlay');
            overlay.classList.remove('translate-x-full');
            
            document.getElementById('ov-name').innerText = s.name;
            document.getElementById('ov-weather').classList.add('hidden');
            
            const ovLine = document.getElementById('ov-line');
            ovLine.innerText = s.line + ' LINE';
            ovLine.className = 'px-3 py-1 text-[10px] font-black uppercase rounded-lg shadow-sm ' + (s.line === 'Red' ? 'bg-red-50 text-red-600' : s.line === 'Blue' ? 'bg-blue-50 text-blue-600' : 'bg-green-50 text-green-600');
            
            const interchanges = Object.keys(interchangeData);
            const oldBtn = document.getElementById('ov-inter-btn');
            if (oldBtn) oldBtn.remove();

            if (interchanges.includes(s.name)) {
                // Prominent Interchange Summary in Overlay
                const iData = interchangeData[s.name];
                const summaryBox = document.createElement('div');
                summaryBox.className = "mb-6 p-6 bg-slate-900 text-white rounded-[32px] shadow-2xl relative overflow-hidden group";
                summaryBox.innerHTML = `
                    <div class="absolute -right-4 -top-4 w-20 h-20 bg-blue-500/10 rounded-full blur-2xl group-hover:bg-blue-500/20 transition-all"></div>
                    <div class="relative z-10">
                        <div class="flex items-center justify-between mb-4">
                            <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Interchange Core</p>
                            <span class="flex items-center gap-1.5 px-2 py-0.5 bg-blue-500 text-white text-[7px] font-black rounded uppercase tracking-widest border border-blue-400">
                                <i data-lucide="clock" size="8"></i> ${iData.time} Est.
                            </span>
                        </div>
                        <div class="space-y-3">
                            ${iData.platforms.map(p => `
                                <div class="p-3 bg-white/5 rounded-2xl border border-white/10 hover:bg-white/10 transition-colors">
                                    <p class="text-[7px] font-black text-blue-400 uppercase tracking-tighter mb-1">${p.pair}</p>
                                    <p class="text-[10px] font-bold text-slate-100 leading-tight">${p.text}</p>
                                </div>
                            `).join('')}
                        </div>
                        <div class="mt-4 pt-4 border-t border-white/10 uppercase tracking-[0.1em] text-[7px] font-black text-blue-300 flex items-center gap-2">
                            <i data-lucide="zap" size="10"></i>
                            Optimal Transit: Follow ${iData.lines.join(' & ')} color-coded paths.
                        </div>
                    </div>
                `;
                document.getElementById('ov-name').after(summaryBox);

                openInterchangeModal(s);
                const interBtn = document.createElement('button');
                interBtn.id = 'ov-inter-btn';
                interBtn.onclick = () => openInterchangeModal(s);
                interBtn.className = "w-full mt-4 py-4 bg-indigo-600 text-white rounded-2xl font-black text-[10px] uppercase tracking-widest flex items-center justify-center gap-3 shadow-xl shadow-indigo-200 border border-indigo-500 hover:bg-indigo-700 transition-all transform hover:scale-[1.02] active:scale-95 mb-6";
                interBtn.innerHTML = `
                    <div class="flex items-center gap-2">
                        <i data-lucide="shuffle" size="14"></i> View Deep Transfer Logic
                    </div>
                    <span class="px-2 py-0.5 bg-white/20 rounded text-[7px] border border-white/30">PLATFORM MAP</span>
                `;
                document.getElementById('ov-name').after(interBtn);
            }

            try {
                const wRes = await fetch('/api/weather', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ lat: s.lat, lng: s.lng })
                });
                const wData = await wRes.json();
                const wEl = document.getElementById('ov-weather');
                const wVal = document.getElementById('ov-weather-val');
                wVal.innerText = `${wData.temp}°C, ${wData.condition}`;
                wEl.classList.remove('hidden');
                lucide.createIcons();
            } catch (e) { console.warn("Station weather fetch failed"); }
            
            const am = document.getElementById('ov-amenities'); am.innerHTML = '';
            const amenities = s.amenities || ['Express Check-in', 'Tactile Pathing', 'HD Surveillance', 'Emergency Ops Center'];
            
            amenities.forEach(a => {
                const dev = document.createElement('div'); 
                dev.className = 'bg-white p-4 rounded-2xl border border-slate-100 shadow-sm flex items-start gap-4 transition-all hover:border-blue-200 group';
                let icon = 'info';
                if (a.includes('Food') || a.includes('KFC')) icon = 'utensils';
                if (a.includes('Parking')) icon = 'parking-circle';
                if (a.includes('ATM')) icon = 'credit-card';
                if (a.includes('Medical')) icon = 'heart-pulse';
                dev.innerHTML = `
                    <div class="w-8 h-8 bg-slate-50 text-slate-400 group-hover:bg-blue-50 group-hover:text-blue-600 rounded-lg flex items-center justify-center shrink-0 transition-colors">
                        <i data-lucide="${icon}" size="14"></i>
                    </div>
                    <div>
                        <h4 class="text-[11px] font-black text-slate-800 uppercase tracking-tight">${a}</h4>
                        <p class="text-[8px] font-bold text-slate-400 uppercase mt-0.5 tracking-widest">Verified Service</p>
                    </div>
                `;
                am.appendChild(dev);
            });

            document.getElementById('ov-plan-btn').onclick = () => {
                document.getElementById('end-st').value = s.id;
                showTab('routes');
                closeOverlay();
            };

            const trainCont = document.getElementById('ov-trains');
            trainCont.innerHTML = `<div class="py-10 flex flex-col items-center gap-4 text-slate-300"><div class="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div><p class="text-[9px] font-black uppercase tracking-widest\">Syncing Flux...</p></div>`;
            
            try {
                const res = await fetch('/api/nearest', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify({ station_id: s.id }) 
                });
                const data = await res.json();
                trainCont.innerHTML = '';
                
                // Group by Platform
                const platforms = {};
                data.upcoming.forEach(t => {
                    if(!platforms[t.platform]) platforms[t.platform] = [];
                    platforms[t.platform].push(t);
                });

                Object.entries(platforms).forEach(([pNum, tList]) => {
                    const pContainer = document.createElement('div');
                    pContainer.className = "mb-6";
                    pContainer.innerHTML = `
                        <h6 class="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                             Platform ${pNum} <span class="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                        </h6>
                    `;
                    
                    const list = document.createElement('div');
                    list.className = "space-y-2";
                    
                    tList.slice(0, 4).forEach(t => {
                        const lineCol = t.line === 'Red' ? 'bg-red-500' : t.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                        const tDiv = document.createElement('div');
                        tDiv.className = "flex justify-between items-center bg-slate-50 p-4 rounded-xl border border-slate-100 hover:border-blue-200 transition-colors";
                        tDiv.innerHTML = `
                            <div class="flex items-center gap-3">
                                <div class="w-1 rounded-full ${lineCol} h-6"></div>
                                <div>
                                    <p class="text-[10px] font-black text-slate-900 mb-0.5">${t.final_stop}</p>
                                    <span class="text-[7px] font-bold text-slate-400 uppercase">${t.direction}</span>
                                </div>
                            </div>
                            <div class="text-right">
                                <p class="text-[11px] font-black text-blue-600 leading-tight">${t.arrival_time}</p>
                                <p class="text-[7px] font-black text-slate-400 uppercase">${t.eta}</p>
                            </div>
                        `;
                        list.appendChild(tDiv);
                    });
                    pContainer.appendChild(list);
                    trainCont.appendChild(pContainer);
                });

                if (Object.keys(platforms).length === 0) {
                    trainCont.innerHTML = `<p class="text-xs font-bold text-slate-300 italic text-center py-10">No active transit vectors found.</p>`;
                }
            } catch (e) { trainCont.innerHTML = `<p class="text-xs font-bold text-red-500">Sync Failed</p>`; }
            lucide.createIcons();
            
            map.flyTo([s.lat, s.lng], 15, { duration: 1.5 });
        }

        function animateTrains() {
            if (tabState !== 'map' || !map) {
                trainAnimationId = requestAnimationFrame(animateTrains);
                return;
            }

            const now = Date.now();
            trainStates.forEach((t, tid) => {
                const s1 = stations.find(s => s.id === t.from_id);
                const s2 = stations.find(s => s.id === t.to_id);
                if(!s1 || !s2) return;

                const elapsed = now / 1000 - t.t1_epoch;
                let progress = 0;
                
                if (elapsed < t.dwell_time) {
                    progress = 0;
                } else if (elapsed > t.duration) {
                    progress = 1;
                } else {
                    progress = (elapsed - t.dwell_time) / (t.duration - t.dwell_time);
                }
                
                // Enhanced easing for fluid motion
                const easedProgress = Math.max(0, Math.min(1, 
                    progress < 0.5 ? 4 * progress * progress * progress : 1 - Math.pow(-2 * progress + 2, 3) / 2
                ));

                const curLat = s1.lat + (s2.lat - s1.lat) * easedProgress;
                const curLng = s1.lng + (s2.lng - s1.lng) * easedProgress;
                
                // Centering for follow mode
                if (followingTrainId === tid) {
                    map.panTo([curLat, curLng]);
                }
                
                // Smoothed rotation calculation
                const targetAngle = Math.atan2(s2.lat - s1.lat, s2.lng - s1.lng) * 180 / Math.PI;

                let marker = trainMarkers.get(tid);
                if (marker) {
                    marker.setLatLng([curLat, curLng]);
                    const iconEl = marker.getElement();
                    if (iconEl) {
                        const inner = iconEl.querySelector('.train-shape-inner');
                        if (inner) {
                            // Apply rotation based on path
                            inner.style.transform = `rotate(${-targetAngle}deg)`;
                            
                            // Visual speed feedback: Stretch train slightly when moving fast
                            if (t.speed > 50 && easedProgress > 0.1 && easedProgress < 0.9) {
                                const stretch = 1 + (t.speed / 500); 
                                inner.style.transform += ` scaleX(${stretch})`;
                            }

                            // Dynamic shadow based on state
                            if (elapsed < t.dwell_time) {
                                inner.classList.add('at-station');
                            } else {
                                inner.classList.remove('at-station');
                            }

                            // Follow mode centering
                            if (followingTrainId === tid) {
                                map.panTo([curLat, curLng], { animate: true, duration: 0.1 });
                            }
                        }
                    }
                }
            });

            trainAnimationId = requestAnimationFrame(animateTrains);
        }

        function swapStations() {
            const start = document.getElementById('start-st');
            const end = document.getElementById('end-st');
            const tmp = start.value;
            start.value = end.value;
            end.value = tmp;
            lucide.createIcons();
            updateLiveStationFeed(start.value);
        }

        function setPlannerOpt(opt) {
            plannerOpt = opt;
            document.querySelectorAll('.planner-opt-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('opt-' + opt).classList.add('active');
        }

        function updateSimTime(hour) {
            plannerSimTime = parseInt(hour);
            const display = document.getElementById('sim-time-display');
            if (plannerSimTime === -1) {
                display.innerText = 'Live Time';
            } else {
                const h = plannerSimTime % 24;
                const ampm = h >= 12 ? 'PM' : 'AM';
                const dispH = h % 12 || 12;
                display.innerText = `${dispH}:00 ${ampm}`;
            }
        }

        function injectAINotifications() {
            const ticker = document.getElementById('neural-ticker');
            if (!ticker) return;
            
            const randomStation = stations[Math.floor(Math.random() * stations.length)];
            const messages = [
                `LOAD PREDICTION: <strong>${randomStation.name}</strong> expecting normal flux in next 20 mins.`,
                `NETWORK STATUS: 100% Neural Sync active across <strong>${stations.length}</strong> nodes.`,
                `AI ADVICE: Use <strong>Green Travel</strong> routes to reduce carbon footprint by 40%.`,
                `SATELLITE DATA: Clear weather reported for <strong>Ameerpet</strong> hub. Optimal transit conditions.`,
                `GTFS TELEMETRY: Train <strong>H-42</strong> moving at standard velocity towards <strong>${randomStation.name}</strong>.`,
                `CROWD LOGIC: Managing high transit peaks at <strong>HITEC City</strong>. Dispatching relief vectors.`
            ];
            
            ticker.innerHTML = '';
            // Duplicate messages to ensure seamless loop
            [...messages, ...messages].forEach(msg => {
                const item = document.createElement('span');
                item.className = 'ticker-item';
                item.innerHTML = msg;
                ticker.appendChild(item);
            });
        }

        function startAINotifications() {
            injectAINotifications();
            if (notificationInterval) clearInterval(notificationInterval);
            notificationInterval = setInterval(injectAINotifications, 15000);
        }

        async function refreshWeather() {
            const updateWeather = async (lat, lng) => {
                try {
                    const res = await fetch('/api/weather', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ lat, lng })
                    });
                    const data = await res.json();
                    document.getElementById('weather-val').innerText = data.temp + '°C, ' + data.condition;
                    document.getElementById('weather-detail').innerText = `Humidity: ${data.humidity}% | Visibility: ${data.visibility.toFixed(1)}km`;
                    
                    const weatherRec = data.temp > 35 ? "Extreme heatwave detected. AC Metro cabins are optimal for travel today." :
                                       data.condition.includes("Rain") ? "Rain detected. Metro is the safest and driest transit route." :
                                       "Environment synchronized. Live updates active for your location.";
                    document.getElementById('env-msg').innerText = weatherRec;
                    
                    const weatherIndicator = document.getElementById('weather-indicator');
                    if (weatherIndicator) {
                        const timeStr = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                        weatherIndicator.innerHTML = `<span class="text-[8px] font-bold text-emerald-500 uppercase tracking-[0.2em] flex items-center gap-1 mt-1"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>Live: ${timeStr}</span>`;
                        weatherIndicator.classList.remove('hidden');
                    }
                } catch (e) { console.warn("Weather Refresh Fail", e); }
            };

            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    pos => updateWeather(pos.coords.latitude, pos.coords.longitude),
                    err => {
                        if (lastUserLoc) updateWeather(lastUserLoc.lat, lastUserLoc.lng);
                    },
                    { enableHighAccuracy: true, timeout: 10000 }
                );
            } else if (lastUserLoc) {
                updateWeather(lastUserLoc.lat, lastUserLoc.lng);
            }
        }

        function renderStationDirectory() {
            const list = document.getElementById('station-details-list');
            if(!list) return;
            
            const search = document.getElementById('directory-search')?.value.toLowerCase() || '';
            const lineFilter = document.getElementById('directory-line-filter')?.value || 'all';

            list.innerHTML = '';
            
            stations.filter(s => {
                const matchesSearch = s.name.toLowerCase().includes(search);
                const matchesLine = lineFilter === 'all' || s.line === lineFilter;
                return matchesSearch && matchesLine;
            }).sort((a, b) => a.name.localeCompare(b.name)).forEach(s => {
                const card = document.createElement('div');
                card.className = "glass-card bg-white p-8 border-slate-100 hover:border-blue-200 transition-all group flex flex-col gap-6";
                
                const color = s.line === 'Red' ? 'text-red-600' : s.line === 'Blue' ? 'text-blue-600' : 'text-emerald-600';
                const bg = s.line === 'Red' ? 'bg-red-50' : s.line === 'Blue' ? 'bg-blue-50' : 'bg-emerald-50';
                
                const amenitiesHtml = (s.amenities || ['Standard Token Kiosk', 'Tactile Flooring', 'CCTV Surveillance']).map(a => `
                    <span class="px-2 py-1 bg-slate-50 text-[8px] font-black text-slate-400 uppercase rounded-md border border-slate-100">${a}</span>
                `).join('');

                card.innerHTML = `
                    <div class="flex items-start justify-between">
                        <div class="flex-1">
                            <h4 class="text-xl font-black text-slate-900 tracking-tight mb-2">${s.name}</h4>
                            <div class="flex items-center gap-2">
                                <span class="px-2 py-0.5 ${bg} ${color} text-[8px] font-black uppercase rounded-md border border-current opacity-70">${s.line} Line</span>
                                <span class="text-[8px] font-bold text-slate-400 uppercase tracking-widest">Node ID: ${s.id}</span>
                            </div>
                        </div>
                        <div class="w-12 h-12 ${bg} ${color} rounded-2xl flex items-center justify-center shadow-sm">
                            <i data-lucide="info" size="20"></i>
                        </div>
                    </div>
                    
                    <div class="space-y-6">
                        <div>
                            <h5 class="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                                <i data-lucide="align-left" size="12"></i> Locality Intelligence
                            </h5>
                            <p class="text-[12px] font-bold text-slate-500 leading-relaxed">${s.description || 'Information regarding this matrix node is currently restricted.'}</p>
                        </div>
                        
                        <div class="space-y-3">
                            <h5 class="text-[9px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                                <i data-lucide="layout-grid" size="12"></i> Hub Infrastructure
                            </h5>
                            <div class="flex flex-wrap gap-2">
                                ${amenitiesHtml}
                            </div>
                        </div>
                    </div>
                    
                    <div class="pt-6 border-t border-slate-50 mt-auto">
                        <button onclick="handleStationClickById('${s.id}')" class="w-full py-4 bg-slate-900 text-white rounded-2xl font-black text-[9px] uppercase tracking-widest hover:bg-blue-600 transition-all flex items-center justify-center gap-3 shadow-lg shadow-slate-200">
                            <i data-lucide="map-pinned" size="14"></i> Locate on Neural Map
                        </button>
                    </div>
                `;
                list.appendChild(card);
            });
            lucide.createIcons();
        }

        function handleStationClickById(id) {
            const s = stations.find(st => st.id === id);
            if(s) {
                showTab('map');
                setTimeout(() => handleStationClick(s), 300);
            }
        }

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
                                <i data-lucide="${v.hasInterchange ? 'shuffle' : 'route'}" size="20"></i>
                            </div>
                            <div>
                                <h5 class="text-sm font-black text-slate-900 tracking-tight">${v.fromName} <i data-lucide="chevrons-right" class="inline opacity-30 px-1" size="12"></i> ${v.toName}</h5>
                                <div class="flex items-center gap-2 mt-1">
                                    <p class="text-[8px] font-black text-slate-400 uppercase tracking-widest">${v.line || 'Multi-Line'} Access</p>
                                    ${v.hasInterchange ? `<span class="px-1.5 py-0.5 bg-blue-50 text-blue-600 text-[6px] font-black rounded uppercase tracking-widest border border-blue-100">${v.interchanges.length} Transfer${v.interchanges.length > 1 ? 's' : ''}</span>` : ''}
                                </div>
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
                line: stations.find(s => s.id === startId).line,
                hasInterchange: currentPlannedRoute.guides && currentPlannedRoute.guides.length > 0,
                interchanges: currentPlannedRoute.guides ? currentPlannedRoute.guides.map(g => g.station) : []
            };

            saved.push(vector);
            localStorage.setItem('metro_saved_vectors', JSON.stringify(saved));
            
            const btn = document.getElementById('save-vector-btn');
            btn.innerHTML = '<i data-lucide="check" size="14"></i> Vector Saved';
            btn.classList.replace('bg-blue-600', 'bg-emerald-500');
            
            setTimeout(() => {
                btn.innerHTML = '<i data-lucide="star" size="14"></i> Save Vector';
                btn.classList.replace('bg-emerald-500', 'bg-blue-600');
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
            btn.classList.replace('bg-blue-600', 'bg-emerald-500');
            btn.innerHTML = '<i data-lucide="shield-check" size="14"></i> Transmission Complete';
            lucide.createIcons();
            
            setTimeout(() => {
                btn.classList.replace('bg-emerald-500', 'bg-blue-600');
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

        function quickCompleteJourney() {
            if (!currentPlannedRoute) return;
            
            const startId = document.getElementById('start-st').value;
            const endId = document.getElementById('end-st').value;
            
            const clean = (str) => {
                let s = str.split(' ').slice(1).join(' ');
                return s.split(' 💻')[0].split(' 🔄')[0].trim();
            };

            const startNode = document.getElementById('start-st');
            const endNode = document.getElementById('end-st');

            const journey = {
                id: 'H-' + Math.random().toString(36).substr(2, 9).toUpperCase(),
                from: clean(startNode.options[startNode.selectedIndex].text),
                to: clean(endNode.options[endNode.selectedIndex].text),
                fare: currentPlannedRoute.fare,
                line: stations.find(s => s.id === startId).line,
                timestamp: new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }),
                status: 'COMPLETED'
            };

            const history = JSON.parse(localStorage.getItem('metro_tickets') || '[]');
            history.push(journey);
            localStorage.setItem('metro_tickets', JSON.stringify(history));
            
            const btn = document.getElementById('complete-journey-btn');
            btn.innerHTML = '<i data-lucide="check" size="14"></i> Logged to History';
            btn.classList.replace('bg-blue-600', 'bg-emerald-500');
            
            setTimeout(() => {
                btn.innerHTML = '<i data-lucide="check-circle" size="14"></i> Complete Journey';
                btn.classList.replace('bg-emerald-500', 'bg-blue-600');
                lucide.createIcons();
                showTab('history');
            }, 1000);

            renderTickets();
        }

        function clearHistory() {
            if (confirm("Are you sure you want to purge the neural archive? This cannot be undone.")) {
                const history = JSON.parse(localStorage.getItem('metro_tickets') || '[]');
                const remaining = history.filter(t => t.status === 'ACTIVE');
                localStorage.setItem('metro_tickets', JSON.stringify(remaining));
                renderTickets();
            }
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

            showTab('history');
            renderTickets();
            
            // Animation effect
            const container = document.getElementById('active-ticket-container');
            container.classList.add('animate-bounce');
            setTimeout(() => container.classList.remove('animate-bounce'), 1000);
        }

        function renderTickets() {
            const history = JSON.parse(localStorage.getItem('metro_tickets') || '[]');
            const listCont = document.getElementById('unified-tickets-list');
            const historyEmpty = document.getElementById('history-empty');
            
            if (listCont) listCont.innerHTML = '';

            if (history.length === 0) {
                if (historyEmpty) historyEmpty.classList.remove('hidden');
                return;
            }
            if (historyEmpty) historyEmpty.classList.add('hidden');

            const sortedHistory = [...history].sort((a,b) => {
                if (a.status === 'ACTIVE' && b.status !== 'ACTIVE') return -1;
                if (b.status === 'ACTIVE' && a.status !== 'ACTIVE') return 1;
                return b.id - a.id;
            });

            sortedHistory.forEach(ticket => {
                const isActive = ticket.status === 'ACTIVE';
                const lineCol = ticket.line === 'Red' ? '#ef4444' : ticket.line === 'Blue' ? '#3b82f6' : '#10b981';
                const card = document.createElement('div');
                
                if (isActive) {
                    card.className = "bg-slate-900 text-white p-0 rounded-[40px] border border-slate-800 shadow-2xl relative overflow-hidden group mb-10";
                    card.innerHTML = `
                        <div class="relative p-10 overflow-hidden">
                            <div class="absolute -right-20 -top-20 w-80 h-80 bg-blue-500/10 rounded-full blur-[100px] animate-pulse"></div>
                            
                            <div class="flex justify-between items-center mb-10 relative z-10">
                                <div class="flex items-center gap-4">
                                    <div class="w-16 h-16 bg-white/5 backdrop-blur-3xl border border-white/10 rounded-3xl flex items-center justify-center text-white">
                                        <i data-lucide="zap" size="28" style="color: ${lineCol}"></i>
                                    </div>
                                    <div>
                                        <h5 class="text-xl font-black tracking-tight">Active Pass</h5>
                                        <div class="flex items-center gap-2">
                                            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                                            <span class="text-[9px] font-black text-white/40 uppercase tracking-[0.3em]">Valid for Journey</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="text-right">
                                    <p class="text-[9px] font-black uppercase tracking-widest text-white/30 mb-1">Pass ID</p>
                                    <p class="text-xs font-black bg-white/5 px-3 py-1.5 rounded-xl border border-white/10 tabular-nums">${ticket.id}</p>
                                </div>
                            </div>

                            <div class="grid grid-cols-1 md:grid-cols-11 gap-6 items-center mb-10 relative z-10">
                                <div class="md:col-span-5 text-left">
                                    <p class="text-[9px] font-black uppercase text-white/30 tracking-widest mb-2">Starting Node</p>
                                    <p class="text-2xl font-black">${ticket.from}</p>
                                </div>
                                <div class="md:col-span-1 flex justify-center py-4 md:py-0">
                                    <i data-lucide="arrow-right-circle" class="text-white/20" size="32"></i>
                                </div>
                                <div class="md:col-span-5 text-right">
                                    <p class="text-[9px] font-black uppercase text-white/30 tracking-widest mb-2">Target Node</p>
                                    <p class="text-2xl font-black">${ticket.to}</p>
                                </div>
                            </div>

                            <div class="flex flex-col md:flex-row items-center justify-between gap-8 py-8 border-t border-white/10 relative z-10">
                                <div class="flex gap-8">
                                    <div>
                                        <p class="text-[9px] font-black uppercase text-white/30 tracking-widest mb-1">Net Fee</p>
                                        <p class="text-xl font-black tabular-nums">₹${ticket.fare}</p>
                                    </div>
                                    <div>
                                        <p class="text-[9px] font-black uppercase text-white/30 tracking-widest mb-1">Issue Time</p>
                                        <p class="text-xl font-black tabular-nums">${new Date(ticket.id).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</p>
                                    </div>
                                </div>
                                <div class="bg-white p-4 rounded-3xl" id="qrcode-${ticket.id}"></div>
                            </div>

                            <button onclick="completeTrip(${ticket.id})" class="w-full py-6 bg-white text-slate-900 rounded-[30px] font-black text-[11px] uppercase tracking-[0.3em] hover:bg-blue-50 transition-all flex items-center justify-center gap-3 mt-4">
                                <i data-lucide="check-circle" size="16"></i> End Journey & Verify
                            </button>
                        </div>
                    `;
                    setTimeout(() => {
                        new QRCode(document.getElementById(`qrcode-${ticket.id}`), {
                            text: `METRO-${ticket.id}-${ticket.from}-${ticket.to}`,
                            width: 100, height: 100,
                            colorDark: "#0f172a", colorLight: "#ffffff",
                            correctLevel: QRCode.CorrectLevel.H
                        });
                    }, 100);
                } else {
                    card.className = "bg-white p-8 rounded-[40px] border border-slate-100 flex flex-col lg:flex-row lg:items-center justify-between group hover:border-slate-300 transition-all shadow-sm gap-6 mb-4";
                    card.innerHTML = `
                        <div class="flex items-center gap-8">
                            <div class="w-14 h-14 rounded-3xl bg-slate-50 flex items-center justify-center text-slate-300 group-hover:bg-blue-50 group-hover:text-blue-600 transition-all">
                                <i data-lucide="archive" size="24"></i>
                            </div>
                            <div class="flex-1">
                                <div class="flex items-center gap-4 mb-2">
                                    <h5 class="text-xl font-black text-slate-900 tracking-tighter">${ticket.from}</h5>
                                    <i data-lucide="chevrons-right" class="text-slate-200" size="14"></i>
                                    <h5 class="text-xl font-black text-slate-900 tracking-tighter">${ticket.to}</h5>
                                </div>
                                <div class="flex items-center gap-4 text-slate-400">
                                    <p class="text-[9px] font-black uppercase tracking-widest">${new Date(ticket.id).toLocaleDateString('en-GB')}</p>
                                    <span class="w-1 h-1 rounded-full bg-slate-200"></span>
                                    <p class="text-[9px] font-black uppercase tracking-widest text-slate-500">₹${ticket.fare} Finalized</p>
                                </div>
                            </div>
                        </div>
                        <div class="flex items-center gap-4">
                            <span class="px-4 py-2 bg-slate-100 text-[8px] font-black text-slate-400 uppercase rounded-xl tracking-widest border border-slate-200">Vector Logged</span>
                            <div class="w-10 h-10 border border-slate-100 rounded-xl flex items-center justify-center text-slate-200 group-hover:text-blue-500 transition-colors">
                                <i data-lucide="chevron-right" size="16"></i>
                            </div>
                        </div>
                    `;
                }
                listCont.appendChild(card);
            });
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
                
                // Show Sentiment Engine after a trip is completed
                const sentimentSect = document.getElementById('sentiment-section');
                if (sentimentSect) {
                    sentimentSect.classList.remove('hidden');
                    sentimentSect.scrollIntoView({ behavior: 'smooth' });
                }
            }
        }

        function updateUserPin(lat, lng) {
            if (!map) return;
            if (!userMarker) {
                userMarker = L.circleMarker([lat, lng], {
                    radius: 8,
                    fillColor: '#3b82f6',
                    color: '#fff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 1
                }).addTo(map);
                userMarker.bindTooltip("You are here", { permanent: false, direction: 'top' });
            } else {
                userMarker.setLatLng([lat, lng]);
            }
        }

        function showTab(id) {
            tabState = id;
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.mobile-link').forEach(l => l.classList.remove('active'));
            
            const tab = document.getElementById('tab-'+id);
            const mob = document.getElementById('mob-'+id);
            if (tab) tab.classList.add('active');
            if (mob) mob.classList.add('active');
            
            // State persistence
            const newUrl = id === 'home' ? '/' : `/?tab=${id}`;
            window.history.pushState({tab: id}, '', newUrl);

            if(id === 'map') {
                initLeafletMap();
                setTimeout(() => map.invalidateSize(), 200);
            } else {
                closeOverlay();
            }
            if(id === 'routes') {
                const startSt = document.getElementById('start-st');
                if(startSt && startSt.value) updateLiveStationFeed(startSt.value);
            }
            if(id === 'history') renderTickets();
            if(id === 'details') renderStationDirectory();
        }

        function closeOverlay() {
            document.getElementById('map-overlay').classList.add('translate-x-full');
            document.querySelectorAll('.station-node').forEach(n => n.classList.remove('selected'));
        }

        function updateClock() {
            try {
                let now = new Date();
                if (typeof simulationHour !== 'undefined' && simulationHour !== -1) {
                    now.setHours(simulationHour);
                    if (document.getElementById('env-msg')) {
                        if (!document.getElementById('sim-indicator')) {
                            const badge = document.createElement('span');
                            badge.id = 'sim-indicator';
                            badge.className = 'ml-3 px-2 py-0.5 bg-blue-500 text-white text-[8px] font-black rounded-md tracking-widest animate-pulse';
                            badge.innerText = 'SIMULATED';
                            const ampmEl = document.getElementById('ampm');
                            if (ampmEl) ampmEl.after(badge);
                        }
                    }
                } else {
                    const simInd = document.getElementById('sim-indicator');
                    if (simInd) simInd.remove();
                }
                
                const clockEl = document.getElementById('clock');
                const ampmEl = document.getElementById('ampm');
                const dateEl = document.getElementById('date');

                if (clockEl) {
                    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
                    const parts = timeStr.split(' ');
                    clockEl.innerText = parts[0];
                    if (ampmEl && parts[1]) ampmEl.innerText = parts[1];
                }
                if (dateEl) {
                    dateEl.innerText = now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }).toUpperCase();
                }
            } catch (e) { console.error("Clock error", e); }
        }

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

        const interchangeData = {
            'Ameerpet': {
                lines: ['Red', 'Blue'],
                time: '3-4 mins',
                platforms: [
                    { pair: 'Red Line (L1)', text: 'Plat 1: Miyapur | Plat 2: LB Nagar' },
                    { pair: 'Blue Line (L2)', text: 'Plat 3: Nagole | Plat 4: Raidurg' }
                ],
                guidance: [
                    "Red Line platforms are situated on Level 1.",
                    "Blue Line platforms are situated on Level 2.",
                    "Use the central concourse escalators for rapid transition between levels.",
                    "Check dynamic signage for platform occupancy real-time."
                ]
            },
            'MG Bus Station': {
                lines: ['Red', 'Green'],
                time: '5-6 mins',
                platforms: [
                    { pair: 'Red Line (L1)', text: 'Plat 1: Miyapur | Plat 2: LB Nagar' },
                    { pair: 'Green Line (L2)', text: 'Plat 3: JBS | Plat 4: Falaknuma' }
                ],
                guidance: [
                    "Transfer between the terminal hubs via the dedicated Interchange Walkway.",
                    "Red Line serves the North-South corridor (Level 1).",
                    "Green Line serves the East corridor (Adjacent Wing).",
                    "Follow the floor-haptic strips for low-visibility guidance."
                ]
            },
            'JBS Parade Ground': {
                lines: ['Blue', 'Green'],
                time: '4-5 mins',
                platforms: [
                    { pair: 'Blue Line (L1)', text: 'Plat 1: Nagole | Plat 2: Raidurg' },
                    { pair: 'Green Line (L2)', text: 'Plat 3: JBS | Plat 4: MGBS' }
                ],
                guidance: [
                    "Blue Line services operate from the Primary Concourse (Level 1).",
                    "Green Line platforms are elevated at Level 2.",
                    "The transition typically takes 3-4 minutes via the designated escalators.",
                    "Neural projection shows optimal boarding at the center of the train for faster exit."
                ]
            }
        };

        function closeInterchangeModal() {
            document.getElementById('interchange-modal').classList.add('hidden');
        }

        function openInterchangeModal(s) {
            const data = interchangeData[s.name];
            if (!data) return;

            document.getElementById('modal-title').innerText = s.name + ' Interchange';
            const badges = document.getElementById('line-badges');
            badges.innerHTML = `<span class="px-3 py-1 bg-blue-50 text-blue-600 rounded-lg text-[9px] font-black uppercase tracking-widest border border-blue-100">Est. Transfer: ${data.time || data.time_estimate}</span>`;
            data.lines.forEach(line => {
                const b = document.createElement('span');
                const color = line === 'Red' ? 'bg-red-500' : line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                b.className = `px-3 py-1 rounded-full text-[10px] font-black text-white uppercase tracking-widest ${color}`;
                b.innerText = line + ' Line';
                badges.appendChild(b);
            });

            // Add Platform Information
            const platCont = document.getElementById('platform-info');
            if (platCont) {
                platCont.innerHTML = '';
                if (data.platforms) {
                    data.platforms.forEach(p => {
                        const div = document.createElement('div');
                        div.className = "p-3 bg-slate-50 rounded-xl border border-slate-100 mb-2";
                        div.innerHTML = `<p class="text-[9px] font-black text-blue-600 uppercase mb-1">${p.pair}</p><p class="text-[11px] font-bold text-slate-700">${p.text}</p>`;
                        platCont.appendChild(div);
                    });
                }
            }

            const guidance = document.getElementById('transfer-guidance');
            guidance.innerHTML = '';
            data.guidance.forEach(step => {
                const div = document.createElement('div');
                div.className = "mb-4";
                div.innerHTML = `
                    <div class="flex gap-6 items-start p-4 bg-slate-50 rounded-2xl border border-slate-100 group hover:border-blue-300 transition-all">
                        <div class="w-8 h-8 bg-blue-600 text-white rounded-lg flex items-center justify-center shrink-0 shadow-lg group-hover:scale-110 transition-transform">
                            <span class="text-[10px] font-black">${data.guidance.indexOf(step) + 1}</span>
                        </div>
                        <div class="flex-1">
                            <p class="text-[12px] font-black text-slate-700 leading-snug tracking-tight mb-1">${step}</p>
                            <div class="flex items-center gap-2">
                                <div class="w-1 h-1 rounded-full bg-emerald-500"></div>
                                <span class="text-[7px] font-black text-slate-400 uppercase tracking-[0.2em]">Verified Path</span>
                            </div>
                        </div>
                    </div>
                `;
                guidance.appendChild(div);
            });

            document.getElementById('interchange-modal').classList.remove('hidden');
            lucide.createIcons();
        }

        function clearMapSearch() {
            const input = document.getElementById('map-search');
            input.value = '';
            filterMapStations('');
            document.getElementById('map-suggestions').classList.add('hidden');
            document.getElementById('search-clear').classList.add('hidden');
        }

        function showMapSuggestions(val) {
            const sugg = document.getElementById('map-suggestions');
            const q = val.toLowerCase().trim();
            
            if (q.length < 1) {
                sugg.classList.add('hidden');
                return;
            }

            const matches = stations.filter(s => s.name.toLowerCase().includes(q));
            if (matches.length === 0) {
                sugg.classList.add('hidden');
                return;
            }

            sugg.innerHTML = '';
            matches.forEach(m => {
                const div = document.createElement('div');
                div.className = "flex items-center gap-4 p-3 hover:bg-slate-50 rounded-xl cursor-pointer transition-all group";
                const color = m.line === 'Red' ? 'bg-red-500' : m.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                div.innerHTML = `
                    <div class="w-8 h-8 ${color} rounded-lg flex items-center justify-center text-white shrink-0 group-hover:scale-110 transition-transform">
                        <i data-lucide="map-pin" size="14"></i>
                    </div>
                    <div class="flex-1">
                        <p class="text-xs font-black text-slate-800 uppercase tracking-tight">${m.name}</p>
                        <p class="text-[8px] font-black text-slate-400 uppercase tracking-widest">${m.line} Line Terminal</p>
                    </div>
                    <i data-lucide="chevron-right" class="text-slate-200 group-hover:text-blue-500" size="14"></i>
                `;
                div.onclick = () => {
                    handleStationClick(m);
                    clearMapSearch();
                };
                sugg.appendChild(div);
            });

            sugg.classList.remove('hidden');
            lucide.createIcons();
        }

        function filterMapStations(query) {
            const q = query.toLowerCase().trim();
            const clearBtn = document.getElementById('search-clear');
            
            if (q === '') clearBtn.classList.add('hidden');
            else clearBtn.classList.remove('hidden');

            showMapSuggestions(query);
            
            if (map) {
                stationMarkers.forEach(sm => {
                    const st = stations.find(s => s.id === sm.id);
                    if (q === '' || st.name.toLowerCase().includes(q)) {
                        sm.marker.setStyle({ opacity: 1, fillOpacity: 1 });
                    } else {
                        sm.marker.setStyle({ opacity: 0.1, fillOpacity: 0.1 });
                    }
                });
            }
        }

        function setupMap() {
            const lineGroup = document.getElementById('map-lines');
            const g = document.getElementById('map-stations');
            
            const interchanges = Object.keys(interchangeData);
            
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
                    document.getElementById('ov-weather').classList.add('hidden');
                    
                    const ovLine = document.getElementById('ov-line');
                    ovLine.innerText = s.line + ' LINE';
                    ovLine.className = 'px-3 py-1 text-[10px] font-black uppercase rounded-lg shadow-sm ' + (s.line === 'Red' ? 'bg-red-50 text-red-600' : s.line === 'Blue' ? 'bg-blue-50 text-blue-600' : 'bg-green-50 text-green-600');
                    
                    // Cleanup old interchange button
                    const oldBtn = document.getElementById('ov-inter-btn');
                    if (oldBtn) oldBtn.remove();

                    // Show interchange details if applicable
                    if (interchanges.includes(s.name)) {
                        openInterchangeModal(s);
                        const interBtn = document.createElement('button');
                        interBtn.id = 'ov-inter-btn';
                        interBtn.onclick = () => openInterchangeModal(s);
                        interBtn.className = "w-full mt-4 py-3 bg-blue-50 text-blue-600 rounded-xl font-black text-[9px] uppercase tracking-widest border border-blue-100 mb-6 flex items-center justify-center gap-2";
                        interBtn.innerHTML = `<i data-lucide="shuffle" size="12"></i> View Interchange Guidance`;
                        document.getElementById('ov-name').after(interBtn);
                    }

                    // Fetch Station Specific Weather
                    try {
                        const wRes = await fetch('/api/weather', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({ lat: s.lat, lng: s.lng })
                        });
                        const wData = await wRes.json();
                        const wEl = document.getElementById('ov-weather');
                        const wVal = document.getElementById('ov-weather-val');
                        wVal.innerText = `${wData.temp}°C, ${wData.condition}`;
                        wEl.classList.remove('hidden');
                        lucide.createIcons();
                    } catch (e) { console.warn("Station weather fetch failed"); }
                    
                    const am = document.getElementById('ov-amenities'); am.innerHTML = '';
                    const amenities = s.amenities || ['Express Check-in', 'Tactile Pathing', 'HD Surveillance', 'Emergency Ops Center'];
                    
                    amenities.forEach(a => {
                        const dev = document.createElement('div'); 
                        dev.className = 'bg-white p-4 rounded-2xl border border-slate-100 shadow-sm flex items-start gap-4 transition-all hover:border-blue-200 group';
                        
                        // Select icon based on keyword
                        let icon = 'info';
                        if (a.includes('Food') || a.includes('KFC') || a.includes('Snack') || a.includes('Court') || a.includes('Coffee')) icon = 'utensils';
                        if (a.includes('Parking') || a.includes('Stand')) icon = 'parking-circle';
                        if (a.includes('ATM') || a.includes('Bank')) icon = 'credit-card';
                        if (a.includes('Clinic') || a.includes('Medical')) icon = 'heart-pulse';
                        if (a.includes('Mall') || a.includes('Shopping') || a.includes('Arcade')) icon = 'shopping-bag';
                        if (a.includes('Interchange') || a.includes('Terminal') || a.includes('Link')) icon = 'shuffle';
                        if (a.includes('Escalator') || a.includes('Elevator')) icon = 'arrow-up-circle';
                        
                        dev.innerHTML = `
                            <div class="w-8 h-8 bg-slate-50 text-slate-400 group-hover:bg-blue-50 group-hover:text-blue-600 rounded-lg flex items-center justify-center shrink-0 transition-colors">
                                <i data-lucide="${icon}" size="14"></i>
                            </div>
                            <div>
                                <h4 class="text-[11px] font-black text-slate-800 uppercase tracking-tight">${a}</h4>
                                <p class="text-[8px] font-bold text-slate-400 uppercase mt-0.5 tracking-widest">Verified Service</p>
                            </div>
                        `;
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
                
                lastStationLoc = { lat: data.station.lat, lng: data.station.lng };
                if (lat && lng) lastUserLoc = { lat, lng };

                document.getElementById('near-name').innerText = data.station.name;
                document.getElementById('near-dist').innerText = data.distance + ' km away';
                
                const walkTimeEl = document.getElementById('near-walk-time');
                if (walkTimeEl) {
                    walkTimeEl.innerText = `${data.walking_mins} min walk (${data.walk_dist}km)`;
                    walkTimeEl.classList.remove('hidden');
                }
                
                const navBtn = document.getElementById('nav-btn');
                if (navBtn) navBtn.classList.remove('hidden');

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
                const now = new Date();
                const h = now.getHours();
                const isClosed = h >= 23 || h < 4;
                const msg = isClosed ? "Network in Maintenance Sleep. Limited vectors available." : "Signal Lost. No upcoming departures projected in this window.";
                rows.innerHTML = `<tr><td colspan="4" class="py-16 text-center text-slate-400 font-bold uppercase text-[10px] tracking-widest bg-slate-50/50 rounded-2xl">${msg}</td></tr>`;
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
                manualRefreshGeo();
                return;
            }
            const s = stations.find(st => st.id === sid);
            updateBoardData(s.lat, s.lng, s.id);
        }

        function manualRefreshGeo() {
            if (navigator.geolocation) {
                document.getElementById('near-name').innerText = "Refreshing...";
                navigator.geolocation.getCurrentPosition(
                    pos => {
                        lastUserLoc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                        updateBoardData(pos.coords.latitude, pos.coords.longitude);
                        refreshWeather();
                    },
                    err => alert("GPS Restricted. Please check permissions."),
                    { enableHighAccuracy: true, timeout: 5000 }
                );
            }
        }

        async function initGeo() {
            const selector = document.getElementById('board-station-selector');
            const interchanges = ['Ameerpet', 'MG Bus Station', 'JBS Parade Ground'];
            const lineIcons = { 'Red': '🔴', 'Blue': '🔵', 'Green': '🟢' };

            if (selector && selector.options.length <= 1) {
                stations.slice().sort((a,b)=>a.name.localeCompare(b.name)).forEach(st => {
                    const opt = document.createElement('option');
                    opt.value = st.id;
                    let suffix = '';
                    if (interchanges.includes(st.name)) suffix = ' 🔄 [Interchange]';
                    opt.innerText = `${lineIcons[st.line]} ${st.name}${suffix}`;
                    selector.appendChild(opt);
                });
            }

            // Initial state
            const nearNameEl = document.getElementById('near-name');
            if (nearNameEl) nearNameEl.innerText = "Acquiring Fix...";

            // Geolocation Timeout Failsafe (2.5s)
            const geoTimeout = setTimeout(() => {
                if (nearNameEl && nearNameEl.innerText === "Acquiring Fix...") {
                    console.log("Geo Timeout - Forcing Default Location");
                    const ameerpet = stations.find(s => s.name === 'Ameerpet');
                    lastUserLoc = { lat: ameerpet.lat, lng: ameerpet.lng };
                    updateBoardData(ameerpet.lat, ameerpet.lng, ameerpet.id);
                }
            }, 2500);

            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    pos => {
                        clearTimeout(geoTimeout);
                        lastUserLoc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                        updateBoardData(pos.coords.latitude, pos.coords.longitude);
                        refreshWeather();
                    },
                    err => {
                        clearTimeout(geoTimeout);
                        if (nearNameEl && nearNameEl.innerText === "Acquiring Fix...") {
                             const ameerpet = stations.find(s => s.name === 'Ameerpet');
                             lastUserLoc = { lat: ameerpet.lat, lng: ameerpet.lng };
                             updateBoardData(ameerpet.lat, ameerpet.lng, ameerpet.id);
                        }
                    },
                    { enableHighAccuracy: true, timeout: 5000 }
                );

                navigator.geolocation.watchPosition(
                    pos => {
                        lastUserLoc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                        updateBoardData(pos.coords.latitude, pos.coords.longitude);
                        refreshWeather();
                    },
                    err => {
                        console.warn("Geo Watch failed");
                    },
                    { enableHighAccuracy: true, timeout: 20000, maximumAge: 5000 }
                );
            } else {
                clearTimeout(geoTimeout);
                const ameerpet = stations.find(s => s.name === 'Ameerpet');
                lastUserLoc = { lat: ameerpet.lat, lng: ameerpet.lng };
                updateBoardData(ameerpet.lat, ameerpet.lng, ameerpet.id);
            }

            // Atmosphere Refresh (60s)
            if (weatherInterval) clearInterval(weatherInterval);
            weatherInterval = setInterval(refreshWeather, 60000);

            // Global shortcut for Map Search
            window.addEventListener('keydown', (e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                    if (tabState === 'map') {
                        e.preventDefault();
                        document.getElementById('map-search')?.focus();
                    }
                }
            });

            // Set default time to now
            const nowTimeInit = new Date();
            const timeStrInit = `${String(nowTimeInit.getHours()).padStart(2, '0')}:${String(nowTimeInit.getMinutes()).padStart(2, '0')}`;
            const timeInputInit = document.getElementById('plan-time');
            if (timeInputInit) {
                timeInputInit.value = timeStrInit;
            }
        }

        function autoPlan() {
            const f = document.getElementById('start-st').value;
            const t = document.getElementById('end-st').value;
            if (f && t) {
                planJourney();
            }
        }

        async function updateLiveStationFeed(stationId) {
            const preview = document.getElementById('quick-train-preview');
            const list = document.getElementById('quick-train-list');
            const planned_time = document.getElementById('plan-time').value;
            const endId = document.getElementById('end-st').value;

            if(!stationId) {
                preview.classList.add('hidden');
                return;
            }

            preview.classList.remove('hidden');
            
            const endName = endId ? (stations.find(s => s.id === endId)?.name || '') : null;
            const directionText = endName ? `Towards ${endName}` : 'All Directions';
            
            list.innerHTML = `<div class="col-span-full py-6 flex flex-col items-center gap-2">
                <div class="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                <p class="text-[8px] font-black uppercase text-slate-400">Syncing ${directionText} Forecast...</p>
            </div>`;

            try {
                const res = await fetch('/api/nearest', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify({ 
                        station_id: stationId,
                        planned_time: planned_time,
                        to_id: endId // Send to_id to help filter direction
                    }) 
                });
                const data = await res.json();
                list.innerHTML = '';
                
                if (data.upcoming && data.upcoming.length > 0) {
                    // Neural Flux Card highlighting Ridership
                    const fluxCard = document.createElement('div');
                    fluxCard.className = "col-span-full mb-6 bg-slate-900 p-8 rounded-[40px] text-white flex justify-between items-center relative overflow-hidden group shadow-2xl shadow-slate-200";
                    fluxCard.innerHTML = `
                        <div class="relative z-10">
                            <p class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500 mb-2">Platform Intelligence</p>
                            <div class="flex items-center gap-3 mb-4">
                                <span class="text-3xl font-black italic tracking-tighter">${data.load || 30}% Intensity</span>
                                <span class="px-3 py-1 bg-white/10 rounded-full text-[9px] font-black uppercase border border-white/20">${targetStopName ? `Filtered: ${targetStopName}` : 'Optimal Flux'}</span>
                            </div>
                            <div class="flex items-center gap-6">
                                <div class="flex items-center gap-2">
                                    <div class="w-1.5 h-1.5 rounded-full ${data.load < 40 ? 'bg-emerald-400' : 'bg-amber-400'} animate-pulse"></div>
                                    <span class="text-[9px] font-black uppercase tracking-widest text-slate-400">Live Ridership</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <i data-lucide="zap" size="12" class="text-blue-400"></i>
                                    <span class="text-[9px] font-black uppercase tracking-widest text-slate-400">Real-time Vector Sync</span>
                                </div>
                            </div>
                        </div>
                        <div class="absolute -right-12 -top-12 w-48 h-48 bg-blue-600/10 rounded-full blur-3xl group-hover:scale-150 transition-transform duration-1000"></div>
                        <div class="text-right relative z-10">
                            <div class="w-16 h-16 bg-white/5 rounded-[28px] border border-white/10 flex items-center justify-center mb-2 ml-auto shadow-inner">
                                <i data-lucide="radio" size="32" class="text-blue-500 animate-pulse"></i>
                            </div>
                            <p class="text-[9px] font-black uppercase tracking-widest text-slate-500">Telemetry Feed</p>
                        </div>
                    `;
                    list.appendChild(fluxCard);

                    // Update header
                    const previewHeader = preview.querySelector('h4');
                    if (previewHeader) {
                        previewHeader.innerHTML = `<span class="w-2 h-2 rounded-full bg-blue-600 animate-pulse"></span> Next 10 Arrivals ${targetStopName ? `Filtered for ${targetStopName}` : 'at Station'}`;
                    }

                    data.upcoming.forEach(t => {
                        const card = document.createElement('div');
                        card.className = "bg-white p-5 rounded-[24px] border border-slate-100 flex justify-between items-center group hover:border-blue-500 shadow-sm transition-all hover:shadow-lg hover:-translate-y-0.5";
                        card.innerHTML = `
                            <div class="flex items-center gap-4">
                                <div class="w-1.5 h-10 rounded-full ${t.line === 'Red' ? 'bg-red-500' : t.line === 'Blue' ? 'bg-blue-500' : 'bg-emerald-500'}"></div>
                                <div>
                                    <p class="text-xs font-black text-slate-900">${t.final_stop}</p>
                                    <div class="flex items-center gap-2 mt-0.5">
                                        <span class="px-1.5 py-0.5 bg-slate-900 text-white text-[7px] font-black rounded uppercase">Platform ${t.platform}</span>
                                        <span class="text-[7px] font-bold text-slate-400 uppercase tracking-widest">${t.line} Line</span>
                                    </div>
                                </div>
                            </div>
                            <div class="text-right">
                                <p class="text-xs font-black text-blue-600">${t.arrival_time}</p>
                                <p class="text-[8px] font-black text-slate-400 uppercase tracking-tighter mt-0.5">ETA: ${t.eta}</p>
                            </div>
                        `;
                        list.appendChild(card);
                    });
                } else {
                    list.innerHTML = `<div class="col-span-full py-12 text-center bg-slate-50 rounded-[32px] border-2 border-dashed border-slate-100">
                        <i data-lucide="wifi-off" class="mx-auto mb-3 text-slate-300" size="24"></i>
                        <p class="text-[10px] font-black uppercase text-slate-400 tracking-widest">No matching vectors found for this trajectory</p>
                        <p class="text-[8px] font-medium text-slate-300 mt-1 uppercase">Try adjusting your travel time or destination</p>
                    </div>`;
                }
                lucide.createIcons();
            } catch (e) {
                list.innerHTML = `<p class="col-span-full text-[8px] font-black text-red-400 uppercase text-center">Sync Matrix Offline</p>`;
            }
        }

        async function planJourney() {
            const outArea = document.getElementById('route-output');
            const plannedTime = document.getElementById('plan-time').value;
            
            // Get Preferences
            const userPrefs = {
                comfort: document.getElementById('pref-comfort')?.checked || false,
                speed: document.getElementById('pref-speed')?.checked || false,
                scenic: document.getElementById('pref-scenic')?.checked || false
            };

            try {
                const f = document.getElementById('start-st').value, t = document.getElementById('end-st').value;
                if (!f || !t) return;

                const res = await fetchWithSim('/api/plan', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify({
                        from: f, 
                        to: t,
                        planned_time: plannedTime,
                        user_prefs: userPrefs
                    }) 
                });
                const responseData = await res.json();
                
                if (responseData.status === 'closed' || responseData.status === 'no_trains' || responseData.status === 'error') {
                    alert(responseData.message || "Route planning failed");
                    return;
                }

                // Primary Route Processing
                renderRouteDisplay(responseData.primary);
                
                // Alternative Routes Processing
                renderAlternatives(responseData.alternatives || []);

            } catch (e) {
                console.error("Plan Journey error:", e);
            }
        }

        function renderRouteDisplay(data) {
            const outArea = document.getElementById('route-output');
            currentPlannedRoute = data;
            outArea.classList.remove('hidden');
            
            const emptyState = document.getElementById('route-empty');
            if (emptyState) emptyState.classList.add('hidden');

            // Draw route on map
            if (map) {
                if (plannedRoutePolyline) map.removeLayer(plannedRoutePolyline);
                routeMarkers.forEach(m => map.removeLayer(m));
                routeMarkers = [];

                const latlngs = data.sequence.map(s => [s.lat, s.lng]);
                plannedRoutePolyline = L.polyline(latlngs, {
                    color: data.sequence[0].line === 'Red' ? '#ef4444' : data.sequence[0].line === 'Blue' ? '#3b82f6' : '#22c55e',
                    weight: 6,
                    opacity: 0.8,
                    dashArray: '10, 10',
                    lineCap: 'round',
                    lineJoin: 'round'
                }).addTo(map);

                data.sequence.forEach((s, idx) => {
                    const isInterchange = !!interchangeData[s.name];
                    if (isInterchange || idx === 0 || idx === data.sequence.length - 1) {
                        const icon = L.divIcon({
                            className: 'route-node-marker',
                            html: `<div class="relative group">
                                        <div class="w-10 h-10 ${isInterchange ? 'bg-blue-600' : 'bg-slate-900'} rounded-2xl flex items-center justify-center text-white shadow-2xl border-2 border-white transform transition-transform group-hover:scale-110">
                                            <i data-lucide="${idx === 0 ? 'play' : idx === data.sequence.length - 1 ? 'square' : 'shuffle'}" size="${isInterchange ? 20 : 16}"></i>
                                        </div>
                                        <div class="absolute top-full left-1/2 -translate-x-1/2 mt-2 bg-slate-900 text-white text-[8px] font-black px-2 py-1 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-[1000] shadow-xl">
                                            ${s.name} ${isInterchange ? '(Hub)' : ''}
                                        </div>
                                   </div>`,
                            iconSize: [40, 40],
                            iconAnchor: [20, 20]
                        });
                        const marker = L.marker([s.lat, s.lng], { icon }).addTo(map);
                        routeMarkers.push(marker);
                    }
                });

                if (typeof lucide !== 'undefined') lucide.createIcons();
                map.fitBounds(plannedRoutePolyline.getBounds(), { padding: [50, 50] });
            }

            // Metrics Update
            const durEl = document.getElementById('route-dur');
            const distEl = document.getElementById('route-dist-km');
            const fareValEl = document.getElementById('route-fare-val');
            const loadValEl = document.getElementById('route-load-val');
            const loadLabelEl = document.getElementById('route-load-label');
            const recEl = document.getElementById('route-rec');
            const routeTransferCont = document.getElementById('route-transfer-container');
            const routeTransferList = document.getElementById('route-transfer-list');
            const stopsCont = document.getElementById('route-stops-list');

            if (followingTrainId) stopTracking();

            if (durEl) durEl.innerText = (data.duration || '--') + 'm';
            if (distEl) distEl.innerText = (data.total_km || '--') + ' KM';
            if (fareValEl) {
                fareValEl.innerText = data.fare || '--';
                // Trigger a small highlight effect on update
                fareValEl.parentElement.classList.add('animate-pulse');
                setTimeout(() => fareValEl.parentElement.classList.remove('animate-pulse'), 1000);
            }
            
            const loadVal = Math.round(data.load || 35);
            if (loadValEl) loadValEl.innerText = loadVal + '%';
            if (loadLabelEl) {
                if (loadVal < 35) loadLabelEl.innerText = "Smooth";
                else if (loadVal < 65) loadLabelEl.innerText = "Active";
                else loadLabelEl.innerText = "Dense";
            }
            if (recEl) recEl.innerText = data.recommendation || 'No advisory for this route.';
            
            // Track Train Card
            const trackCont = document.getElementById('track-vector-container');
            if (trackCont) {
                trackCont.innerHTML = '';
                if (data.chosen_trip_id) {
                    trackCont.classList.remove('hidden');
                    const card = document.createElement('div');
                    card.className = "col-span-full bg-slate-900 rounded-[40px] p-8 text-white relative overflow-hidden group shadow-2xl mb-6 border border-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500";
                    card.innerHTML = `
                        <div class="absolute -right-20 -top-20 w-60 h-60 bg-blue-500/10 rounded-full blur-3xl group-hover:bg-blue-500/20 transition-all duration-700"></div>
                        <div class="relative z-10">
                            <div class="flex items-center justify-between mb-6">
                                <div>
                                    <p class="text-[10px] font-black text-blue-400 uppercase tracking-[0.3em] mb-2">Live Telemetry Active</p>
                                    <h3 class="text-3xl font-black italic tracking-tighter">Vector ${data.chosen_trip_id.split('_').pop() || 'N/A'}</h3>
                                </div>
                                <div class="w-14 h-14 bg-white/5 rounded-2xl flex items-center justify-center border border-white/10 group-hover:bg-blue-600 group-hover:border-blue-500 transition-all">
                                    <i data-lucide="radio" size="24" class="animate-pulse"></i>
                                </div>
                            </div>
                            <div class="grid grid-cols-2 gap-4 mb-8">
                                <div class="p-4 bg-white/5 rounded-2xl border border-white/10">
                                    <p class="text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">Path Fidelity</p>
                                    <p class="text-xs font-black">99.8% Sync</p>
                                </div>
                                <div class="p-4 bg-white/5 rounded-2xl border border-white/10">
                                    <p class="text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">Status</p>
                                    <p class="text-xs font-black text-emerald-400">On Track</p>
                                </div>
                            </div>
                            <button onclick="trackTrain('${data.chosen_trip_id}')" class="w-full py-5 bg-blue-600 rounded-3xl font-black text-[11px] uppercase tracking-widest hover:bg-blue-500 transition-all transform hover:scale-[1.02] active:scale-95 shadow-xl shadow-blue-900/40">
                                Track Live Vector on Map
                            </button>
                        </div>
                    `;
                    trackCont.appendChild(card);
                } else {
                    trackCont.classList.add('hidden');
                }
            }

            // Gemini Dynamic Advice (Client Side)
            if (window.GoogleGenAI && "{{ GEMINI_API_KEY }}") {
                async function updateAiAdvice() {
                    try {
                        const recEl = document.getElementById('route-rec');
                        if (!recEl) return;
                        recEl.innerText = "Analyzing neural path logic...";
                        
                        const ai = new window.GoogleGenAI({ apiKey: "{{ GEMINI_API_KEY }}" });
                        const model = ai.getGenerativeModel({ model: 'gemini-1.5-flash' });

                        const prompt = `Analyze this route: ${data.sequence[0].name} -> ${data.sequence[data.sequence.length - 1].name}. 
                        Stops: ${data.total_stops}, Distance: ${data.total_km}km, Load: ${data.load}%, Conditions: ${data.weather_desc || 'Normal'}.
                        Provide ONE helpful, futuristic transit advisory (max 20 words). Mention a specific station if relevant.`;
                        
                        const resAI = await model.generateContent(prompt);
                        const responseText = resAI.response.text();
                        if (responseText) {
                            recEl.innerText = responseText.trim().replace(/^"|"$/g, '');
                        }
                    } catch (e) { console.error("AI Insight error:", e); }
                }
                updateAiAdvice();
            }

            // Personnelized Advice Cards
            const persRecCont = document.getElementById('personalized-recommendations');
            const persRecWrapper = document.getElementById('personalized-recommendations-wrapper');
            if (persRecCont && data.personalized_advices) {
                persRecCont.innerHTML = '';
                if (data.personalized_advices.length > 0) {
                    persRecWrapper?.classList.remove('hidden');
                    data.personalized_advices.forEach((adv, i) => {
                        const card = document.createElement('div');
                        card.className = "bg-indigo-600 p-6 rounded-[32px] text-white flex items-start gap-5 border border-indigo-400 relative overflow-hidden animate-in fade-in slide-in-from-left-4 duration-500 shadow-xl";
                        card.style.animationDelay = `${i * 100}ms`;
                        card.innerHTML = `
                             <div class="w-12 h-12 bg-white/10 rounded-2xl flex items-center justify-center shrink-0 border border-white/20">
                                <i data-lucide="${adv.icon}" size="20"></i>
                            </div>
                            <div class="flex-1">
                                <h4 class="text-[8px] font-black uppercase tracking-widest text-indigo-200 mb-1">${adv.title}</h4>
                                <p class="text-[13px] font-black leading-tight">${adv.text}</p>
                            </div>
                        `;
                        persRecCont.appendChild(card);
                    });
                } else { persRecWrapper?.classList.add('hidden'); }
            }

            // Transfers Intelligence
            if (routeTransferList) {
                routeTransferList.innerHTML = '';
                if (data.guides && data.guides.length > 0) {
                    routeTransferCont?.classList.remove('hidden');
                    data.guides.forEach(g => {
                        const card = document.createElement('div');
                        card.className = "bg-white p-6 rounded-[32px] border-2 border-blue-100 shadow-sm relative overflow-hidden group hover:border-blue-500 transition-all";
                        card.innerHTML = `
                            <div class="flex items-center justify-between mb-4">
                                <div class="flex items-center gap-3">
                                    <div class="w-10 h-10 bg-blue-600 text-white rounded-xl flex items-center justify-center shadow-lg"><i data-lucide="shuffle" size="20"></i></div>
                                    <div>
                                        <p class="text-[8px] font-black text-blue-600 uppercase tracking-widest">Interchange Operation</p>
                                        <h4 class="text-base font-black text-slate-900">${g.station}</h4>
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="bg-blue-600 text-white px-3 py-1.5 rounded-xl text-center shadow-sm">
                                        <p class="text-[7px] font-black uppercase tracking-widest opacity-60">Platform</p>
                                        <p class="text-xs font-black">${g.platform}</p>
                                    </div>
                                </div>
                            </div>
                            <p class="text-xs font-bold text-slate-600 mb-4 bg-slate-50 p-3 rounded-xl border border-slate-100">${g.text}</p>
                            ${g.steps ? `
                                <div class="space-y-2">
                                    ${g.steps.map((s, i) => `
                                        <div class="flex gap-3 items-center">
                                            <span class="w-5 h-5 bg-blue-50 text-blue-600 rounded text-[9px] font-black flex items-center justify-center">${i+1}</span>
                                            <p class="text-[10px] font-bold text-slate-500">${s}</p>
                                        </div>
                                    `).join('')}
                                </div>
                            ` : ''}
                        `;
                        routeTransferList.appendChild(card);
                    });
                } else { routeTransferCont?.classList.add('hidden'); }
            }

            // Intermediate Stops
            if (stopsCont) {
                stopsCont.innerHTML = '';
                data.sequence.forEach((s, i) => {
                    const row = document.createElement('div');
                    row.className = "flex items-start gap-4 pb-6 relative group";
                    const isHub = !!interchangeData[s.name];
                    const lineCol = s.line === 'Red' ? 'bg-red-500' : s.line === 'Blue' ? 'bg-blue-500' : 'bg-emerald-500';
                    row.innerHTML = `
                        <div class="flex flex-col items-center shrink-0">
                            <div class="w-4 h-4 ${lineCol} rounded-full ring-4 ${lineCol.replace('bg-', 'ring-')}/20 z-10"></div>
                            ${i < data.sequence.length - 1 ? `<div class="w-[2px] h-full ${lineCol} opacity-20 absolute top-2"></div>` : ''}
                        </div>
                        <div class="flex-1 -mt-1">
                            <div class="flex justify-between">
                                <p class="text-sm font-black ${isHub ? 'text-blue-600' : 'text-slate-900'}">${s.name} ${isHub ? '🔄' : ''}</p>
                                <span class="text-[11px] font-black text-slate-400 font-mono">${s.reaching_at}</span>
                            </div>
                            <div class="flex items-center gap-2 mt-1">
                                <span class="px-1.5 py-0.5 rounded text-[7px] font-black text-white ${lineCol}">${s.line}</span>
                                <span class="text-[8px] font-black text-slate-300 uppercase">${s.dist_km} KM</span>
                                ${s.predicted_load ? `<span class="px-1.5 py-0.5 rounded text-[7px] font-black uppercase ${s.predicted_load > 70 ? 'bg-red-50 text-red-500' : 'bg-emerald-50 text-emerald-500'}">Load: ${s.predicted_load}%</span>` : ''}
                            </div>
                        </div>
                    `;
                    stopsCont.appendChild(row);
                });
            }

            if (typeof lucide !== 'undefined') lucide.createIcons();
            lastCalculatedFare = data.fare;

            // Update Upcoming Hour Schedule
            const sched = document.getElementById('schedule-list');
            if (sched && data.upcoming_hour) {
                sched.innerHTML = '';
                data.upcoming_hour.forEach(u => {
                    const div = document.createElement('div');
                    div.className = 'flex justify-between items-center bg-white p-6 rounded-3xl border border-slate-50 shadow-sm hover:shadow-md transition-all animate-in fade-in slide-in-from-right-4 duration-500';
                    div.innerHTML = `
                        <div class="flex items-center gap-5">
                            <div class="w-12 h-12 rounded-2xl bg-slate-50 flex flex-col items-center justify-center border border-slate-100 shrink-0">
                                <span class="text-[8px] font-black text-slate-400 uppercase">PLAT</span>
                                <span class="text-[14px] font-black text-slate-900">${u.platform}</span>
                            </div>
                            <div>
                                <span class="text-sm font-black text-slate-700 block line-clamp-1">${u.final_stop}</span>
                                <div class="flex items-center gap-2 mt-1">
                                    <span class="px-1.5 py-0.5 rounded text-[7px] font-black uppercase tracking-tighter text-white ${u.line === 'Red' ? 'bg-red-500' : u.line === 'Blue' ? 'bg-blue-500' : 'bg-emerald-500'}">${u.line} LINE</span>
                                    <span class="text-[8px] font-black text-slate-400 uppercase tracking-widest">Arrives: ${u.arrival_time}</span>
                                </div>
                            </div>
                        </div>
                        <div class="text-right shrink-0">
                            <span class="text-[9px] font-black text-slate-400 uppercase tracking-widest block mb-0.5">Reach Dest</span>
                            <p class="text-[15px] font-black text-blue-600 tabular-nums">${u.est_reach || '--:--'}</p>
                            <p class="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mt-0.5">in ${u.eta}m</p>
                        </div>`;
                    sched.appendChild(div);
                });
            }
            if (typeof lucide !== 'undefined') lucide.createIcons();
        }

        function renderAlternatives(alternatives) {
            const wrapper = document.getElementById('route-alternatives-wrapper');
            const list = document.getElementById('alternatives-list');
            if (!list || !alternatives || alternatives.length === 0) {
                wrapper?.classList.add('hidden');
                return;
            }
            wrapper?.classList.remove('hidden');
            list.innerHTML = '';

            alternatives.forEach((alt, i) => {
                const card = document.createElement('div');
                card.className = "bg-white p-6 rounded-[32px] border border-slate-100 shadow-sm hover:shadow-xl hover:border-blue-200 transition-all cursor-pointer group animate-in fade-in slide-in-from-bottom-4 duration-500";
                card.style.animationDelay = `${i * 150}ms`;
                card.onclick = () => {
                    renderRouteDisplay(alt);
                    window.scrollTo({ top: document.getElementById('planner-input-area').offsetTop - 20, behavior: 'smooth' });
                };
                card.innerHTML = `
                    <div class="flex items-center justify-between mb-4">
                        <div class="w-10 h-10 bg-slate-50 text-slate-400 rounded-2xl flex items-center justify-center group-hover:bg-blue-600 group-hover:text-white transition-colors">
                            <i data-lucide="route" size="18"></i>
                        </div>
                        <div class="px-3 py-1 bg-blue-50 text-blue-600 rounded-lg text-[10px] font-black">₹${alt.fare}</div>
                    </div>
                    <div class="mb-4">
                        <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1">AI Recommendation ${i+1}</p>
                        <h4 class="text-sm font-black text-slate-900 line-clamp-1">${alt.sequence[0].name} to ${alt.sequence[alt.sequence.length-1].name}</h4>
                    </div>
                    <div class="grid grid-cols-2 gap-3 mb-4">
                        <div class="p-3 bg-slate-50 rounded-2xl text-center">
                            <p class="text-[7px] font-black text-slate-400 uppercase mb-1">Time</p>
                            <p class="text-xs font-black text-slate-900">${alt.duration}m</p>
                        </div>
                        <div class="p-3 bg-slate-50 rounded-2xl text-center">
                            <p class="text-[7px] font-black text-slate-400 uppercase mb-1">Load</p>
                            <p class="text-xs font-black ${alt.load > 70 ? 'text-red-500' : 'text-emerald-500'}">${alt.load}%</p>
                        </div>
                    </div>
                    <p class="text-[9px] font-bold text-slate-400 leading-tight line-clamp-2 italic">${alt.personalized_advices[0]?.text || 'Optimized path logic.'}</p>
                `;
                list.appendChild(card);
            });
            if (typeof lucide !== 'undefined') lucide.createIcons();
        }

        // Initialize pickers with grouped matrix logic
        function initPickers() {
            const lines = ['Red', 'Blue', 'Green'];
            const itHubs = ['HITEC City', 'Madhapur', 'Raidurg'];
            const interchanges = ['Ameerpet', 'MG Bus Station', 'JBS Parade Ground'];
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
                
                if (data.trains && map) {
                    const seenIds = new Set();
                    data.trains.forEach(t => {
                        seenIds.add(t.trip_id);
                        trainStates.set(t.trip_id, t);
                        
                        // Highlight logic for planned route
                        const isPlannedTrain = currentPlannedRoute && t.trip_id === currentPlannedRoute.chosen_trip_id;
                        
                        const color = t.line === 'Red' ? '#ef4444' : t.line === 'Blue' ? '#3b82f6' : '#10b981';
                        const accent = t.line === 'Red' ? '#fee2e2' : t.line === 'Blue' ? '#dbeafe' : '#dcfce7';
                        let marker = trainMarkers.get(t.trip_id);
                        
                        if(!marker) {
                            // High-fidelity train marker with direction indicator and line branding
                            const color = t.line === 'Red' ? '#ef4444' : t.line === 'Blue' ? '#3b82f6' : '#10b981';
                            const accent = t.line === 'Red' ? '#fee2e2' : t.line === 'Blue' ? '#dbeafe' : '#dcfce7';
                            
                            const trainIcon = L.divIcon({
                                className: 'train-icon',
                                html: `
                                    <div class="train-halo-container hidden absolute inset-0 flex items-center justify-center pointer-events-none">
                                        <div class="highlighted-train-halo"></div>
                                    </div>
                                    <div class="train-shape-inner">
                                    <svg width="60" height="40" viewBox="0 0 60 40">
                                        <!-- Shadow -->
                                        <rect x="12" y="16" width="36" height="12" rx="4" fill="black" fill-opacity="0.15"/>
                                        
                                        <!-- Train Body -->
                                        <rect x="10" y="8" width="40" height="18" rx="4" fill="${color}" stroke="white" stroke-width="2"/>
                                        
                                        <!-- Windows -->
                                        <rect x="14" y="12" width="6" height="5" rx="1" fill="white" fill-opacity="0.3"/>
                                        <rect x="22" y="12" width="6" height="5" rx="1" fill="white" fill-opacity="0.3"/>
                                        <rect x="30" y="12" width="6" height="5" rx="1" fill="white" fill-opacity="0.3"/>
                                        
                                        <!-- Line Code -->
                                        <text x="38" y="19" font-family="Plus Jakarta Sans" font-size="10" font-weight="900" fill="white" fill-opacity="0.9">${t.line[0]}</text>
                                        
                                        <!-- Direction Arrow (Nose) -->
                                        <path d="M 50,12 L 58,17 L 50,22 Z" fill="#ffffff" stroke="${color}" stroke-width="1"/>
                                        
                                        <!-- Roof details -->
                                        <rect x="15" y="6" width="20" height="2" rx="1" fill="white" fill-opacity="0.5"/>
                                    </svg>
                                </div>`,
                                iconSize: [60, 40],
                                iconAnchor: [30, 20]
                            });
                            marker = L.marker([0, 0], { icon: trainIcon, zIndexOffset: isPlannedTrain ? 2000 : 1000 }).addTo(map);
                            trainMarkers.set(t.trip_id, marker);
                        }

                        // Apply persistent highlight for planned train/followed train
                        const iconEl = marker.getElement();
                        if (iconEl) {
                            const tid = t.trip_id;
                            const halo = iconEl.querySelector('.train-halo-container');
                            if (isPlannedTrain || tid === followingTrainId) {
                                if (halo) halo.classList.remove('hidden');
                                iconEl.style.filter = 'drop-shadow(0 0 20px rgba(59, 130, 246, 0.9))';
                                iconEl.style.zIndex = "2001";
                            } else {
                                if (halo) halo.classList.add('hidden');
                                iconEl.style.filter = '';
                                iconEl.style.zIndex = "1000";
                            }
                        }
                    });

                    // Cleanup
                    trainMarkers.forEach((m, tid) => {
                        if(!seenIds.has(tid)) {
                            map.removeLayer(m);
                            trainMarkers.delete(tid);
                            trainStates.delete(tid);
                        }
                    });
                }
            } catch (e) { console.error("Sim Matrix Error:", e); }
        }

        window.addEventListener('DOMContentLoaded', () => {
            const urlParams = new URLSearchParams(window.location.search);
            const tabUrl = urlParams.get('tab');
            const initialTab = "{{ initial_tab|default('home') }}";
            
            showTab(tabUrl || initialTab);

            if (typeof lucide !== 'undefined') lucide.createIcons();
            
            initLeafletMap();
            initGeo(); 
            initPickers(); 
            updateLiveTrains(); 
            loadFeedback(); 
            renderTickets();
            loadSavedVectors();
            startAINotifications();
            
            activeUpdateInterval = setInterval(updateLiveTrains, 5000); 
            trainAnimationId = requestAnimationFrame(animateTrains);
            
            setInterval(updateClock, 1000); 
            updateClock();
        });
    </script>
</body>
</html>
"""

# Initial System Boot Sync
ensure_gtfs()
generate_ai_dataset()

if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')
