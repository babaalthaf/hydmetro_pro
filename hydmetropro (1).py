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
    if request and 'sim_hour' in request.args:
        try:
            h = int(request.args.get('sim_hour'))
            m = int(request.args.get('sim_min', 0))
            now = get_ist_now()
            return now.replace(hour=h, minute=m, second=0, microsecond=0)
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
    """Predicts load with more descriptive results for AI reasoning."""
    is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0
    is_it_hub = 1 if any(h in station_name for h in ['HITEC City', 'Madhapur', 'Raidurg']) else 0
    
    score = 50 + (is_peak * 100) + (is_it_hub * 80)
    if is_weekend: score -= 30
    
    # Weather influence: People flock to AC Metro during high heat or seek shelter during rain
    weather_str = ""
    if weather:
        if "Rain" in weather.get('condition', ''):
            score += 25
            weather_str = "Raining outdoors"
        if weather.get('temp', 30) > 35:
            score += 20
            weather_str = "Intense heat"
    
    if score > 180: return "High", f"🔴 High Crowd ({weather_str})" if weather_str else "🔴 High Crowd"
    if score > 130: return "M-High", f"🟡 Moderate Rush ({weather_str})" if weather_str else "🟡 Moderate Rush"
    if score > 80: return "Medium", "🟢 Manageable Crowd"
    return "Low", "🟢 Calm Condition"

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
        except:
            _GTFS_CACHE = []
    
    return _GTFS_CACHE

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
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
    return render_template_string(HTML_TEMPLATE, ALL_STATIONS=STATIONS_LIST, GEMINI_API_KEY=GEMINI_API_KEY, hide_planner=False, initial_tab='home')

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
            except: continue
    
    # AI & Weather Data
    weather = get_live_weather(lat=lat, lng=lng)
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
        except: pass
    
    # Sort upcoming by original string comparison (before I:M p conversion) or keep it consistent
    
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
                    if idx > 0: neighbors.append(line[idx-1])
                    if idx < len(line)-1: neighbors.append(line[idx+1])
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
        
        for pt in possible_trips[:5]: # Check first 5 upcoming
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

    if not gtfs_boarding_time:
        # Check if outside hours (typical Hyd Metro: 6 AM to 11 PM)
        is_closed = False
        if now.hour < 6 or now.hour >= 23:
            is_closed = True
        
        return jsonify({
            'status': 'closed' if is_closed else 'no_trains',
            'message': 'Metro services are currently closed. They will resume at 06:00 AM.' if is_closed else 'No trains found for this route at this time.',
            'resume_time': '06:00 AM'
        })

    # Annotate sequence with reaching times and predict load for each station
    for s in sequence:
        s['reaching_at_raw'] = stop_arrival_times.get(s['id'])
        reach_hour = now.hour
        if s['reaching_at_raw']:
            try:
                h, m, s_ = map(int, s['reaching_at_raw'].split(':'))
                reach_hour = h
                s['reaching_at'] = now.replace(hour=h, minute=m, second=s_).strftime('%I:%M %p')
            except: 
                s['reaching_at'] = s['reaching_at_raw']
        else:
            s['reaching_at'] = "--:--"
        
        # Add AI Predicted Load for every stop
        is_weekend = now.weekday() >= 5
        weather = get_live_weather(lat=s['lat'], lng=s['lng'])
        load_label, _ = predict_load_ai(s['name'], reach_hour, is_weekend=is_weekend, weather=weather)
        s['predicted_load'] = load_label

    # Calculate Total Distance for Precise Fare Prediction
    total_km = 0
    for i in range(len(sequence)):
        if i == 0:
            sequence[i]['segment_km'] = 0
            sequence[i]['dist_km'] = 0
        else:
            s1 = sequence[i-1]
            s2 = sequence[i]
            d = haversine(s1['lat'], s1['lng'], s2['lat'], s2['lng'])
            total_km += d
            sequence[i]['segment_km'] = round(d, 2)
            sequence[i]['dist_km'] = round(total_km, 2)
    
    # Official Fare Logic (2026 Updated Chart)
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

    calculated_fare = get_fare_from_matrix(total_km)
    digital_fare = round(calculated_fare * 0.9, 1) # 10% Smart Card Discount
    
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
    weather = get_live_weather(lat=sequence[0]['lat'], lng=sequence[0]['lng'])
    is_weekend = now.weekday() >= 5
    load_val, _ = predict_load_ai(start_station_name, now.hour, is_weekend=is_weekend, weather=weather)
    
    # NEW: Numerical Load and Peak Intensity Math
    load_pct = 35 # Base
    if (7 <= now.hour <= 10 or 17 <= now.hour <= 21): load_pct += 45
    if start_station_name in ['HITEC City', 'Madhapur', 'Raidurg']: load_pct += 15
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
        s2 = next(s for s in STATIONS_LIST if s['id'] == path[i+1])
        name1 = s1.get('name_alias', s1['name'])
        name2 = s2.get('name_alias', s2['name'])
        
        if name1 == name2 and s1['id'] != s2['id']:
            next_sid = path[i+2] if i+2 < len(path) else None
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
                    other_ids = [s['id'] for s in STATIONS_LIST if (s.get('name_alias') == name1 or s['name'] == name1) and s['line'] != s1['line']]
                    
                    for row in trips:
                        if row['station_id'] in other_ids and reaching_at_raw < row['arrival_time'] < rph_str:
                            t_copy = row.copy()
                            th, tm, ts = map(int, t_copy['arrival_time'].split(':'))
                            t_dt = now.replace(hour=th, minute=tm, second=ts)
                            t_copy['arrival_time_12'] = t_dt.strftime('%I:%M %p')
                            t_copy['wait_mins'] = int((t_dt - reach_dt).total_seconds() // 60)
                            connecting_trains.append(t_copy)
                            if len(connecting_trains) >= 3: break
                except: pass

            if next_sid:
                next_s = next(s for s in STATIONS_LIST if s['id'] == next_sid)
                i_data = INTERCHANGE_DATA.get(name1, {})
                platform = "?"
                steps = i_data.get('guidance', [])
                time_est = i_data.get('time_estimate', '2-3 mins')

                if name1 == 'Ameerpet':
                    if next_s['line'] == 'Red':
                        idx = int(next_s['id'].replace('R',''))
                        platform = "1 (Towards LB Nagar)" if idx > 11 else "2 (Towards Miyapur)"
                        if s1['line'] == 'Blue': guide = "Exit Blue Line (Level 1). Take stairs/escalator DOWN to Red Line level. Follow signs for Platform 1/2."
                    elif next_s['line'] == 'Blue':
                        idx = int(next_s['id'].replace('B',''))
                        platform = "3 (Towards Nagole)" if idx > 8 else "4 (Towards Raidurg)"
                        if s1['line'] == 'Red': guide = "Exit Red Line. Take stairs/escalator UP to Blue Line (Level 1). Follow signs for Platform 3/4."
                elif name1 == 'MG Bus Station':
                    if next_s['line'] == 'Red':
                        idx = int(next_s['id'].replace('R',''))
                        platform = "1 (Towards LB Nagar)" if idx > 20 else "2 (Towards Miyapur)"
                        if s1['line'] == 'Green': guide = "Exit Green Line platform, follow 'Red Line' signs. Take ESCALATOR UP to Red Line Level."
                    elif next_s['line'] == 'Green':
                        platform = "3 (Towards JBS Parade Ground)"
                        if s1['line'] == 'Red': guide = "Exit Red Line platform, follow 'Green Line' signs. Take ESCALATOR DOWN to Green Line Level."
                elif name1 == 'JBS Parade Ground':
                    if next_s['line'] == 'Blue':
                        idx = int(next_s['id'].replace('B',''))
                        platform = "3 (Towards Nagole)" if idx > 13 else "4 (Towards Raidurg)"
                        if s1['line'] == 'Green': guide = "Exit Green Line platform, follow Blue Line transfer signs. Take stairs to Platform level."
                    elif next_s['line'] == 'Green':
                        idx = int(next_s['id'].replace('G',''))
                        platform = "1 (Towards MG Bus Station)" if idx > 3 else "2 (Towards JBS Parade Ground)"
                        if s1['line'] == 'Blue': guide = "Exit Blue Line platform, follow Green Line transfer signs. Take ESCALATOR DOWN to reach Green Line Platform Level."
            
                guides.append({
                    'station': name1, 
                    'platform': platform, 
                    'text': guide, 
                    'steps': steps,
                    'time_estimate': time_est,
                    'reaching_at': reaching_at_display, 
                    'connections': connecting_trains
                })

    # PROJECTION METRICS
    is_peak = "Peak Hour" if (7 <= now.hour <= 10 or 17 <= now.hour <= 21) else "Off-Peak"
    is_it_hub = "High" if any(n in [s['name'] for s in sequence] for n in ['HITEC City', 'Raidurg', 'Madhapur']) else "Normal"
    
    # Environmental Analytics
    co2_saved = round(total_km * 0.12, 2) # kg CO2
    calories = int(total_km * 12 + len(guides) * 25) # Estimated effort
    trees_saved = round(total_km * 0.05, 3)

    weather_advice = " AC cooling optimized for intense heat." if weather.get('temp', 30) > 35 else \
                     " Rainy conditions detected; transit via tunnels recommended." if "Rain" in weather.get('condition', '') else \
                     " Clear skies for a smooth commute." if "Sunny" in weather.get('condition', '') or "Clear" in weather.get('condition', '') else ""

    recommendation = ("Optimal conditions. Low crowd density detected." if load_val == "Low" else \
                     "Fair volume. Seat likely available for your journey." if load_val == "Medium" else \
                     "Moderate volume. Manageable rush." if load_val == "M-High" else \
                     "Peak congestion. AI suggests waiting for dip.") + weather_advice

    # Personalized Recommendations List
    personalized_advices = []
    
    # Enhanced AI Suggestion Logic based on Crowd & Weather
    crowd_context = "higher" if load_pct > 65 else "moderate" if load_pct > 40 else "low"
    interchange_hit = next((s['name'] for s in sequence if s['name'] in INTERCHANGE_DATA), None)
    
    if interchange_hit and load_pct > 70:
        personalized_advices.append({
            'type': 'ai-insight',
            'title': 'AI Insight',
            'text': f"Expect {crowd_context} crowds at {interchange_hit} due to peak hour rush. Consider an alternate platform if available for faster boarding.",
            'icon': 'brain'
        })
    elif weather.get('temp', 30) > 38:
        personalized_advices.append({
            'type': 'weather',
            'title': 'Weather Alert',
            'text': f"Severe heat detected ({weather.get('temp')}°C). Hydrating at {start_station_name} before boarding is strongly recommended.",
            'icon': 'sun'
        })
    
    if load_pct > 75:
        personalized_advices.append({
            'type': 'congestion',
            'title': 'High Density Alert',
            'text': 'Severe congestion predicted. Suggest taking a 15-min earlier train for a guaranteed seat.',
            'icon': 'users'
        })
    elif load_pct > 50:
        personalized_advices.append({
            'type': 'congestion',
            'title': 'Moderate Flux',
            'text': 'Steady boarding volume. Board mid-train coaches for less crowding.',
            'icon': 'info'
        })

    if weather.get('temp', 30) > 36:
        personalized_advices.append({
            'type': 'weather',
            'title': 'Heat Protocol',
            'text': 'High external temperatures. Station cooling optimized; stay hydrated at platform kiosks.',
            'icon': 'thermometer'
        })
    elif "Rain" in weather.get('condition', ''):
        personalized_advices.append({
            'type': 'weather',
            'title': 'Rain Advisory',
            'text': 'Slippery floors possible at terminal exits. Multi-modal links might be delayed.',
            'icon': 'cloud-rain'
        })
    
    if is_peak == "Peak Hour" and any(n in [s['name'] for s in sequence] for n in ['HITEC City', 'Madhapur', 'Raidurg']):
         personalized_advices.append({
            'type': 'it-hub',
            'title': 'Tech Hub Rush',
            'text': 'Heavy IT corridor movement detected. Use the dedicated express entry for faster access.',
            'icon': 'zap'
        })

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
            except: continue
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
        except: pass

    return jsonify({
        'sequence': sequence,
        'upcoming_hour': upcoming_hour,
        'duration': duration,
        'boarding_at_source': boarding_at_source,
        'arrival_at_dest': arrival_at_destination,
        'total_stops': len(sequence),
        'total_km': round(total_km, 2),
        'fare': calculated_fare,
        'digital_fare': digital_fare,
        'recommendation': recommendation,
        'personalized_advices': personalized_advices,
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
        <div onclick="showTab('routes')" class="mobile-link" id="mob-routes"><i data-lucide="route"></i><span>Route</span></div>
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
                            <i data-lucide="route" size="14"></i> Find Route
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
                            <button id="ov-plan-btn" class="w-full py-5 bg-blue-600 text-white rounded-2xl font-black text-[11px] uppercase tracking-widest flex items-center justify-center gap-3 shadow-xl shadow-blue-200">
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
                   <h2 class="text-4xl lg:text-5xl font-black tracking-tighter mb-1 text-slate-900">Route Finder</h2>
                   <p class="text-[10px] font-bold text-slate-400 uppercase tracking-[0.4em] flex items-center justify-center gap-2"><i data-lucide="shield-check" size="14" class="text-emerald-500"></i> Always on time</p>
                </div>
                
                <div id="planner-input-area" class="glass-card border-none shadow-2xl bg-white p-8 lg:p-12 relative overflow-hidden group mb-8">
                    <div class="absolute -right-20 -top-20 w-96 h-96 bg-blue-50/50 rounded-full blur-[100px] transition-all group-hover:bg-blue-100/30"></div>
                    
                    <!-- Input Matrix -->
                    <div class="grid grid-cols-1 md:grid-cols-11 items-center gap-2 mb-10">
                        <div class="md:col-span-5 space-y-3">
                            <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1">From Station</label>
                            <div class="relative">
                                <div class="absolute left-6 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-blue-600 ring-8 ring-blue-50"></div>
                                <select id="start-st" class="w-full pl-14 pr-8 py-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-blue-500/30 focus:bg-white font-black appearance-none text-slate-900 transition-all cursor-pointer hover:bg-slate-100"></select>
                                <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none"><i data-lucide="chevron-down" size="18"></i></div>
                            </div>
                        </div>

                        <div class="md:col-span-1 flex justify-center py-4 md:py-0 text-center relative">
                            <button onclick="swapStations()" class="p-4 bg-blue-600 text-white rounded-full shadow-2xl hover:rotate-180 transition-transform duration-700 cursor-pointer border-4 border-white z-20">
                                <i data-lucide="repeat-2" size="20"></i>
                            </button>
                        </div>

                        <div class="md:col-span-5 space-y-3">
                            <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1">To Station</label>
                            <div class="relative">
                                <div class="absolute left-6 top-1/2 -translate-y-1/2 text-emerald-500"><i data-lucide="map-pin" size="18"></i></div>
                                <select id="end-st" class="w-full pl-14 pr-8 py-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-emerald-500/30 focus:bg-white font-black appearance-none text-slate-900 transition-all cursor-pointer hover:bg-slate-100"></select>
                                <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none"><i data-lucide="chevron-down" size="18"></i></div>
                            </div>
                        </div>
                    </div>

                    <!-- Optimization Matrix -->
                    <div class="mb-10 relative z-10">
                        <label class="text-[10px] font-black text-slate-500 uppercase block tracking-[0.2em] pl-1 mb-4">Route Optimization Logic</label>
                        <div class="grid grid-cols-3 gap-4">
                            <button onclick="setPlannerOpt('speed')" id="opt-speed" class="planner-opt-btn active p-5 bg-slate-50 text-slate-400 rounded-3xl border-2 border-transparent flex flex-col items-center gap-2 transition-all hover:bg-slate-100">
                                <i data-lucide="zap" size="20"></i>
                                <span class="text-[10px] font-black uppercase tracking-widest">Fastest</span>
                            </button>
                            <button onclick="setPlannerOpt('comfort')" id="opt-comfort" class="planner-opt-btn p-5 bg-slate-50 text-slate-400 rounded-3xl border-2 border-transparent flex flex-col items-center gap-2 transition-all hover:bg-slate-100">
                                <i data-lucide="armchair" size="20"></i>
                                <span class="text-[10px] font-black uppercase tracking-widest">Comfort</span>
                            </button>
                            <button onclick="setPlannerOpt('direct')" id="opt-direct" class="planner-opt-btn p-5 bg-slate-50 text-slate-400 rounded-3xl border-2 border-transparent flex flex-col items-center gap-2 transition-all hover:bg-slate-100">
                                <i data-lucide="git-branch" size="20"></i>
                                <span class="text-[10px] font-black uppercase tracking-widest">Direct</span>
                            </button>
                        </div>
                    </div>

                    <!-- Departure Window -->
                    <div class="bg-slate-50 p-8 rounded-[32px] border border-slate-100 mb-10 relative z-10">
                        <div class="flex items-center justify-between mb-6">
                            <label class="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Departure Window</label>
                            <div class="flex items-center gap-2 bg-white px-4 py-2 rounded-xl shadow-sm border border-slate-100">
                                <i data-lucide="clock" size="14" class="text-blue-600"></i>
                                <span id="sim-time-display" class="text-xs font-black text-slate-900">Live Time</span>
                            </div>
                        </div>
                        <input type="range" min="-1" max="23" value="-1" id="time-slider" oninput="updateSimTime(this.value)" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600">
                        <div class="flex justify-between mt-4 text-[9px] font-black text-slate-400 uppercase tracking-widest px-1">
                            <span>Now</span>
                            <span>Morning</span>
                            <span>Noon</span>
                            <span>Evening</span>
                            <span>Night</span>
                        </div>
                    </div>

                    <button id="plan-btn" onclick="planJourney()" class="w-full py-7 bg-blue-600 hover:bg-blue-700 text-white font-black rounded-[32px] shadow-2xl shadow-blue-100 transition-all active:scale-[0.98] text-[13px] uppercase tracking-[0.4em] flex items-center justify-center gap-4 group relative z-10">
                        <span id="btn-text">Generate Neural Path</span>
                        <div id="btn-loader" class="hidden w-5 h-5 border-[3px] border-white border-t-transparent rounded-full animate-spin"></div>
                        <i data-lucide="sparkles" size="18" class="text-blue-400 group-hover:scale-125 transition-transform"></i>
                    </button>
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
                    <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                        <div class="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm flex flex-col gap-2">
                            <div class="flex items-center gap-2 text-blue-600">
                                <i data-lucide="users" size="14"></i>
                                <span class="text-[9px] font-black uppercase tracking-widest">Crowd Density</span>
                            </div>
                            <div class="flex flex-col">
                                <span id="route-load-val" class="text-lg font-black text-slate-900">--%</span>
                                <div class="w-full h-1 bg-slate-100 rounded-full mt-1 overflow-hidden">
                                    <div id="route-load-bar" class="h-full bg-blue-600 w-0 transition-all duration-1000"></div>
                                </div>
                            </div>
                        </div>

                        <div class="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm flex flex-col gap-2">
                            <div class="flex items-center gap-2 text-indigo-600">
                                <i data-lucide="brain" size="14"></i>
                                <span class="text-[9px] font-black uppercase tracking-widest">Neural Path Logic</span>
                            </div>
                            <p id="route-rec" class="text-[11px] font-bold text-slate-600 leading-tight line-clamp-2">--</p>
                        </div>

                        <div class="bg-emerald-50 p-5 rounded-3xl border border-emerald-100 shadow-sm flex flex-col gap-2">
                            <div class="flex items-center gap-2 text-emerald-600">
                                <i data-lucide="leaf" size="14"></i>
                                <span class="text-[9px] font-black uppercase tracking-widest">Eco Impact</span>
                            </div>
                            <div class="flex justify-between items-end">
                                <span id="route-eco-val" class="text-lg font-black text-emerald-700">--</span>
                                <span class="text-[8px] font-black text-emerald-600/60 uppercase pb-1">Saved</span>
                            </div>
                        </div>

                        <div class="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm flex flex-col gap-2">
                            <div class="flex items-center gap-2 text-slate-400">
                                <i data-lucide="clock" size="14"></i>
                                <span class="text-[9px] font-black uppercase tracking-widest">Transit Time</span>
                            </div>
                            <span id="route-dur" class="text-lg font-black text-slate-900">--</span>
                        </div>

                        <div class="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm flex flex-col gap-2">
                            <div class="flex items-center gap-2 text-slate-400">
                                <i data-lucide="map" size="14"></i>
                                <span class="text-[9px] font-black uppercase tracking-widest">Distance (KM)</span>
                            </div>
                            <span id="route-dist-km" class="text-lg font-black text-slate-900">-- KM</span>
                        </div>

                        <div class="bg-blue-600 p-5 rounded-3xl shadow-lg flex flex-col gap-2 text-white">
                            <div class="flex items-center gap-2 opacity-80">
                                <i data-lucide="credit-card" size="14"></i>
                                <span class="text-[9px] font-black uppercase tracking-widest">Ticket Fare</span>
                            </div>
                            <span id="route-fare" class="text-xl font-black">₹--</span>
                            <span id="route-digital-fare" class="text-[9px] font-bold opacity-70 bg-black/20 px-2 py-1 rounded-md">Smart: ₹--</span>
                        </div>
                    </div>

                    <!-- Station Stop List -->
                    <div class="bg-white rounded-[32px] border border-slate-100 shadow-xl overflow-hidden mt-2">
                        <div class="p-6 bg-slate-50 border-b border-slate-100">
                            <h4 class="text-[10px] font-black uppercase tracking-widest text-slate-400">Stops on your way</h4>
                        </div>
                        <div id="route-stops-list" class="p-6 space-y-4">
                            <!-- Injected Stations -->
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

        <div id="tab-history" class="tab-content">
            <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-8 gap-6 border-b pb-6 border-slate-200">
                <div class="flex flex-col items-center lg:flex-row lg:items-center gap-6 text-center lg:text-left">
                    <div class="w-16 h-16 bg-blue-600 rounded-3xl flex items-center justify-center text-white shadow-xl shadow-blue-500/30"><i data-lucide="qr-code" size="32"></i></div>
                    <div>
                        <h2 class="text-4xl font-black tracking-tight mb-1 text-slate-900">Tickets & History</h2>
                        <p class="text-xs font-bold text-slate-400 uppercase tracking-widest">Digital Vault & Journey Archive</p>
                    </div>
                </div>
                <button onclick="clearHistory()" class="px-5 py-2.5 bg-red-50 text-red-600 rounded-xl text-[10px] font-black uppercase tracking-widest border border-red-100 hover:bg-red-100 transition-all">Clear Archive</button>
            </div>

            <div class="max-w-4xl mx-auto space-y-12">
                <!-- Active Tickets Section -->
                <div>
                    <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-900 mb-6 flex items-center gap-2">
                        <i data-lucide="zap" class="text-blue-600" size="14"></i> Active Boarding Pass
                    </h4>
                    <div id="active-ticket-container" class="perspective-1000">
                        <!-- Active ticket injected here -->
                    </div>
                </div>

                <!-- History Section -->
                <div id="history-container">
                    <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 mb-6 flex items-center gap-2">
                        <i data-lucide="history" size="14"></i> Journey History
                    </h4>
                    <div id="trip-history" class="space-y-4">
                        <!-- History injected here -->
                    </div>
                    <div id="history-empty" class="py-32 text-center flex flex-col items-center justify-center text-slate-300 bg-slate-50/50 rounded-[40px] border-2 border-dashed border-slate-200 shadow-inner">
                        <i data-lucide="folder-clock" size="48" class="mb-6 opacity-20 text-slate-400"></i>
                        <h4 class="text-xl font-black text-slate-400 mb-2">Vector Log Empty</h4>
                        <p class="text-[10px] font-black uppercase tracking-widest opacity-60">Complete a journey in the Path Architect to log data</p>
                        <button onclick="showTab('routes')" class="mt-10 px-8 py-4 bg-blue-600 text-white rounded-2xl font-black text-[10px] uppercase tracking-[0.2em] shadow-xl">Find Route</button>
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
                openInterchangeModal(s);
                const interBtn = document.createElement('button');
                interBtn.id = 'ov-inter-btn';
                interBtn.onclick = () => openInterchangeModal(s);
                interBtn.className = "w-full mt-4 py-3 bg-blue-50 text-blue-600 rounded-xl font-black text-[9px] uppercase tracking-widest border border-blue-100 mb-6 flex items-center justify-center gap-2";
                interBtn.innerHTML = `<i data-lucide="shuffle" size="12"></i> View Interchange Guidance`;
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
                } else {
                    progress = (elapsed - t.dwell_time) / (t.duration - t.dwell_time);
                }
                
                // Add easing for even more realism (smooth start/stop)
                const easedProgress = Math.max(0, Math.min(1, 
                    progress < 0.5 ? 2 * progress * progress : 1 - Math.pow(-2 * progress + 2, 2) / 2
                ));

                const curLat = s1.lat + (s2.lat - s1.lat) * easedProgress;
                const curLng = s1.lng + (s2.lng - s1.lng) * easedProgress;
                
                // Angle calculation with smoothing
                const targetAngle = Math.atan2(s2.lat - s1.lat, s2.lng - s1.lng) * 180 / Math.PI;

                let marker = trainMarkers.get(tid);
                if (marker) {
                    marker.setLatLng([curLat, curLng]);
                    const iconEl = marker.getElement();
                    if (iconEl) {
                        const inner = iconEl.querySelector('.train-shape-inner');
                        if (inner) {
                            // Only rotate if moving
                            if (easedProgress > 0 && easedProgress < 1) {
                                inner.style.transform = `rotate(${-targetAngle}deg)`;
                            }
                            
                            // Visual feedback if at station
                            if (elapsed < t.dwell_time) {
                                inner.classList.add('at-station');
                            } else {
                                inner.classList.remove('at-station');
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
            const activeCont = document.getElementById('active-ticket-container');
            const historyCont = document.getElementById('trip-history');
            const historyEmpty = document.getElementById('history-empty');
            
            // Clean containers first
            if (activeCont) activeCont.innerHTML = '';
            if (historyCont) historyCont.innerHTML = '';

            const activeTicket = history.find(t => t.status === 'ACTIVE');
            const pastTrips = history.filter(t => t.status === 'COMPLETED').reverse();

            // Render Active Ticket
            if (activeTicket) {
                const lineCol = activeTicket.line === 'Red' ? '#ef4444' : activeTicket.line === 'Blue' ? '#3b82f6' : '#10b981';
                activeCont.innerHTML = `
                    <div class="glass-card bg-white border-slate-200 p-0 overflow-hidden shadow-[0_20px_40px_-5px_rgba(0,0,0,0.05)] relative group transition-all duration-700 hover:scale-[1.02] border-l-[10px]" style="border-left-color: ${lineCol}">
                        <div class="relative p-10 overflow-hidden">
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
                                <div class="col-span-11 flex flex-col items-center">
                                     <div class="flex items-center justify-between w-full mb-4 px-4 bg-white/5 py-4 rounded-[28px] border border-white/5">
                                        <div class="text-left flex-1">
                                            <p class="text-[8px] font-black uppercase text-white/30 tracking-widest mb-1">Source</p>
                                            <p class="text-[14px] font-black text-white truncate">${activeTicket.from}</p>
                                        </div>
                                        <div class="px-6 flex flex-col items-center">
                                            <i data-lucide="arrow-right" class="text-blue-500" size="14"></i>
                                        </div>
                                        <div class="text-right flex-1">
                                            <p class="text-[8px] font-black uppercase text-white/30 tracking-widest mb-1">Target</p>
                                            <p class="text-[14px] font-black text-white truncate">${activeTicket.to}</p>
                                        </div>
                                     </div>
                                </div>
                            </div>

                            <div class="flex flex-col items-center justify-center bg-white p-12 rounded-[52px] shadow-[inset_0_4px_30px_rgba(0,0,0,0.06)] mb-10 relative z-10 group-hover:scale-105 transition-transform duration-500">
                                <div id="ticket-qr" class="p-1"></div>
                                <p class="text-[8px] font-black uppercase tracking-[0.4em] text-slate-300 mt-6">Digital Ticket ID</p>
                            </div>

                            <button onclick="completeTrip('${activeTicket.id}')" class="w-full py-6 bg-white text-slate-900 rounded-[28px] font-black text-[11px] uppercase tracking-[0.3em] transition-all hover:bg-blue-50 active:scale-95 shadow-xl shadow-white/5 mb-2 flex items-center justify-center gap-3">
                                 <i data-lucide="scan-face" size="14"></i> Complete Journey
                            </button>
                        </div>
                    </div>
                `;
                new QRCode(document.getElementById("ticket-qr"), {
                    text: JSON.stringify({ id: activeTicket.id, u: "AIS-HYD", auth: "NEURAL-PRO" }),
                    width: 210, height: 210, colorDark: "#0f172a", colorLight: "#ffffff",
                    correctLevel: QRCode.CorrectLevel.H
                });
            } else {
                activeCont.innerHTML = `
                    <div class="glass-card border-dashed border-2 flex flex-col items-center justify-center py-40 text-slate-300 rounded-[40px] bg-slate-50/10">
                         <div class="w-20 h-20 bg-white rounded-full flex items-center justify-center shadow-sm mb-8 border border-slate-100">
                            <i data-lucide="qr-code" size="32" class="opacity-10 text-slate-400"></i>
                         </div>
                         <h4 class="text-xl font-black text-slate-400 mb-2">No active tickets</h4>
                         <p class="text-[10px] font-black uppercase tracking-widest opacity-40 mb-10">Purchase a ticket to start your journey</p>
                         <button onclick="showTab('routes')" class="px-8 py-4 bg-blue-600 text-white rounded-2xl font-black text-[9px] uppercase tracking-widest shadow-xl">Purchase Digital Token</button>
                    </div>
                `;
            }

            // Render History
            if (historyCont) {
                if (pastTrips.length === 0) {
                    if (historyEmpty) historyEmpty.classList.remove('hidden');
                } else {
                    if (historyEmpty) historyEmpty.classList.add('hidden');
                    pastTrips.forEach(trip => {
                        const lineCol = trip.line === 'Red' ? 'bg-red-500' : trip.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                        const div = document.createElement('div');
                        div.className = "bg-white p-8 rounded-[40px] border border-slate-100 flex flex-col lg:flex-row lg:items-center justify-between group hover:border-blue-200 transition-all shadow-sm hover:shadow-md gap-6 mb-4";
                        div.innerHTML = `
                            <div class="flex items-center gap-8">
                                <div class="w-12 h-12 rounded-[20px] ${lineCol} flex items-center justify-center text-white shadow-lg group-hover:scale-110 transition-transform">
                                    <i data-lucide="map-pin" size="20"></i>
                                </div>
                                <div>
                                    <div class="flex items-center gap-4 mb-2">
                                        <h5 class="text-[18px] font-black text-slate-900 tracking-tight">${trip.from}</h5>
                                        <i data-lucide="arrow-right" class="text-slate-300" size="14"></i>
                                        <h5 class="text-[18px] font-black text-slate-900 tracking-tight">${trip.to}</h5>
                                    </div>
                                    <div class="flex items-center gap-4 text-slate-400">
                                        <p class="text-[9px] font-black uppercase tracking-widest">${trip.timestamp}</p>
                                        <span class="w-1 h-1 rounded-full bg-slate-200"></span>
                                        <p class="text-[9px] font-black uppercase tracking-widest text-blue-600">₹${trip.fare} Network Fee</p>
                                    </div>
                                </div>
                            </div>
                            <div class="flex flex-col lg:items-end border-t lg:border-t-0 pt-4 lg:pt-0 border-slate-50">
                                <span class="text-[9px] font-black text-emerald-600 uppercase tracking-widest bg-emerald-50 px-4 py-2 rounded-xl border border-emerald-100 inline-block mb-2">Vector Archived</span>
                                <p class="text-[10px] font-black text-slate-300 tabular-nums">ID: ${trip.id}</p>
                            </div>
                        `;
                        historyCont.appendChild(div);
                    });
                }
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
            
            if(id === 'map') {
                initLeafletMap();
                setTimeout(() => map.invalidateSize(), 200);
            } else {
                closeOverlay();
            }
            if(id === 'history') renderTickets();
            if(id === 'details') renderStationDirectory();
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

        const interchangeData = {
            'Ameerpet': {
                lines: ['Red', 'Blue'],
                time: '3-4 mins',
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
            badges.innerHTML = `<span class="px-3 py-1 bg-blue-50 text-blue-600 rounded-lg text-[9px] font-black uppercase tracking-widest border border-blue-100">Est. Transfer: ${data.time_estimate}</span>`;
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
                div.className = "flex gap-4 items-start";
                div.innerHTML = `
                    <div class="w-5 h-5 bg-blue-100 rounded-full flex items-center justify-center shrink-0 mt-0.5">
                        <span class="text-[10px] font-black text-blue-600">${data.guidance.indexOf(step) + 1}</span>
                    </div>
                    <p class="text-sm font-bold text-slate-600 leading-relaxed">${step}</p>
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

            stations.slice().sort((a,b)=>a.name.localeCompare(b.name)).forEach(st => {
                const opt = document.createElement('option');
                opt.value = st.id;
                let suffix = '';
                if (interchanges.includes(st.name)) suffix = ' 🔄 [Interchange]';
                opt.innerText = `${lineIcons[st.line]} ${st.name}${suffix}`;
                selector.appendChild(opt);
            });

            // Initial state
            document.getElementById('near-name').innerText = "Acquiring Fix...";

            if (navigator.geolocation) {
                // Get one point immediately for speed
                navigator.geolocation.getCurrentPosition(
                    pos => {
                        lastUserLoc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                        updateBoardData(pos.coords.latitude, pos.coords.longitude);
                        refreshWeather();
                    },
                    err => {
                        console.warn("Rapid fix failed. Waiting for satellite stream.");
                    },
                    { enableHighAccuracy: true, timeout: 5000 }
                );

                // Start continuous watch
                navigator.geolocation.watchPosition(
                    pos => {
                        lastUserLoc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                        updateBoardData(pos.coords.latitude, pos.coords.longitude);
                        refreshWeather();
                    },
                    err => {
                        if (document.getElementById('near-name').innerText === "Acquiring Fix...") {
                             const ameerpet = stations.find(s => s.name === 'Ameerpet');
                             lastUserLoc = { lat: ameerpet.lat, lng: ameerpet.lng };
                             updateBoardData(ameerpet.lat, ameerpet.lng, ameerpet.id);
                        }
                    },
                    { enableHighAccuracy: true, timeout: 20000, maximumAge: 5000 }
                );
            } else {
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
        }

        async function planJourney() {
            const btn = document.getElementById('plan-btn');
            const btnText = document.getElementById('btn-text');
            const loader = document.getElementById('btn-loader');
            const outArea = document.getElementById('route-output');
            
            btn.disabled = true;
            btnText.classList.add('opacity-0');
            loader.classList.remove('hidden');

            try {
                const f = document.getElementById('start-st').value, t = document.getElementById('end-st').value;
                if (!f || !t) {
                    btn.disabled = false;
                    btnText.classList.remove('opacity-0');
                    loader.classList.add('hidden');
                    return;
                }

                const res = await fetchWithSim('/api/plan', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify({
                        from: f, 
                        to: t,
                        optimization: plannerOpt,
                        sim_hour: plannerSimTime // Ensure this is sent
                    }) 
                });
                const data = await res.json();
                
                if (data.status === 'closed') {
                    outArea.classList.remove('hidden');
                    document.getElementById('route-empty')?.classList.add('hidden');
                    outArea.innerHTML = `
                        <div class="bg-slate-900 p-10 rounded-[40px] text-center border border-slate-800 shadow-2xl relative overflow-hidden my-10 animate-in fade-in duration-700">
                            <div class="absolute top-0 left-0 w-full h-1 bg-blue-600"></div>
                            <div class="w-20 h-20 bg-blue-600/10 text-blue-500 rounded-full flex items-center justify-center mx-auto mb-8">
                                <i data-lucide="moon" size="40"></i>
                            </div>
                            <h3 class="text-2xl font-black text-white mb-4">Metro is Closed</h3>
                            <p class="text-slate-400 font-bold leading-relaxed mb-8">${data.message}</p>
                            <div class="bg-white/5 border border-white/10 p-6 rounded-3xl">
                                <p class="text-[10px] font-black text-blue-400 uppercase tracking-widest mb-1">Service Resumes At</p>
                                <p class="text-3xl font-black text-white">${data.resume_time}</p>
                            </div>
                        </div>
                    `;
                    lucide.createIcons();
                    btn.disabled = false;
                    btnText.classList.remove('opacity-0');
                    loader.classList.add('hidden');
                    return;
                }

                // If output was overridden by closed message, restore structure if needed (though we'll probably just reload next time)
                // For simplicity, let's assume valid data from here
                currentPlannedRoute = data;
                outArea.classList.remove('hidden');
                
                // Draw route on map
                if (map) {
                    if (plannedRoutePolyline) map.removeLayer(plannedRoutePolyline);
                    const latlngs = data.sequence.map(s => [s.lat, s.lng]);
                    plannedRoutePolyline = L.polyline(latlngs, {
                        color: data.sequence[0].line === 'Red' ? '#ef4444' : data.sequence[0].line === 'Blue' ? '#3b82f6' : '#22c55e',
                        weight: 6,
                        opacity: 0.8,
                        dashArray: '10, 10',
                        lineCap: 'round',
                        lineJoin: 'round'
                    }).addTo(map);
                    map.fitBounds(plannedRoutePolyline.getBounds(), { padding: [50, 50] });
                }

                const emptyState = document.getElementById('route-empty');
                if (emptyState) emptyState.classList.add('hidden');
                
                // Updates for new mobile UI metrics
                document.getElementById('route-dur').innerText = data.duration + 'm';
                document.getElementById('route-fare').innerText = '₹' + data.fare;
                document.getElementById('route-digital-fare').innerText = 'Smart Card: ₹' + (data.digital_fare || (data.fare * 0.9).toFixed(1));
                document.getElementById('route-dist-km').innerText = data.total_km + ' KM';
                document.getElementById('route-rec').innerText = data.recommendation;
                
                // Gemini Personalized Suggestion
                if (window.GoogleGenAI && "{{ GEMINI_API_KEY }}") {
                    try {
                        const loadingSub = "Analyzing neural path logic...";
                        document.getElementById('route-rec').innerText = loadingSub;
                        
                        const ai = new window.GoogleGenAI({ apiKey: "{{ GEMINI_API_KEY }}" });
                        const prompt = `You are the Hyderabad Metro AI Assistant. Analyze this route data and provide ONE short, personalized advice (max 25 words).
                        Route: ${data.sequence[0].name} to ${data.sequence[data.sequence.length - 1].name}
                        Stops: ${data.total_stops}
                        Peak Intensity: ${data.peak_intensity}%
                        Crowd Level: ${data.load}%
                        General Status: ${data.recommendation}
                        Note: Mention specific stations like ${data.sequence[0].name} or interchanges if any. Use a helpful, futuristic tone.`;
                        
                        const model = 'gemini-3-flash-preview';
                        const resAI = await ai.models.generateContent({
                            model: model,
                            contents: [{ role: 'user', parts: [{ text: prompt }] }]
                        });
                        if (resAI.text) {
                            document.getElementById('route-rec').innerText = resAI.text.trim();
                        }
                    } catch (aiErr) {
                        console.error("Gemini suggestion failed:", aiErr);
                    }
                }
                
                // Eco Metrics
                document.getElementById('route-eco-val').innerText = data.eco.co2 + 'kg';

                // Crowd Data
                const loadVal = Math.round(data.load || 35);
                document.getElementById('route-load-val').innerText = loadVal + '%';
                const loadBar = document.getElementById('route-load-bar');
                if (loadBar) {
                    loadBar.style.width = loadVal + '%';
                    loadBar.className = 'h-full transition-all duration-1000 ' + 
                                      (loadVal > 70 ? 'bg-red-500' : (loadVal > 40 ? 'bg-amber-500' : 'bg-emerald-500'));
                }
                
                // Personalized Recommendations
                const persRecCont = document.getElementById('personalized-recommendations');
                if (persRecCont) {
                    persRecCont.innerHTML = '';
                    if (data.personalized_advices && data.personalized_advices.length > 0) {
                        persRecCont.classList.remove('hidden');
                        data.personalized_advices.forEach(adv => {
                            const card = document.createElement('div');
                            card.className = "bg-white p-4 rounded-3xl border border-slate-100 shadow-sm flex items-start gap-4";
                            card.innerHTML = `
                                <div class="w-10 h-10 bg-slate-50 text-slate-600 rounded-xl flex items-center justify-center shrink-0">
                                    <i data-lucide="${adv.icon}" size="16"></i>
                                </div>
                                <div>
                                    <h4 class="text-[10px] font-black text-slate-800 uppercase mb-0.5">${adv.title}</h4>
                                    <p class="text-[9px] font-bold text-slate-500 leading-tight">${adv.text}</p>
                                </div>
                            `;
                            persRecCont.appendChild(card);
                        });
                        lucide.createIcons();
                    } else {
                        persRecCont.classList.add('hidden');
                    }
                }

                lastCalculatedFare = data.fare;
                
                // Render Stops List (In-between metro stations)
                const stopsCont = document.getElementById('route-stops-list');
                if (stopsCont) {
                    stopsCont.innerHTML = '';
                    data.sequence.forEach((s, idx) => {
                        const sDiv = document.createElement('div');
                        sDiv.className = "flex items-center gap-4 group";
                        
                        const lineCol = s.line === 'Red' ? 'bg-red-500' : s.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                        const dotSize = idx === 0 || idx === data.sequence.length - 1 ? 'w-4 h-4 ring-4' : 'w-2.5 h-2.5';
                        const ringCol = s.line === 'Red' ? 'ring-red-100' : s.line === 'Blue' ? 'ring-blue-100' : 'ring-green-100';

                        sDiv.innerHTML = `
                            <div class="relative flex flex-col items-center">
                                <div class="${dotSize} ${lineCol} ${ringCol} rounded-full z-10 transition-transform group-hover:scale-125"></div>
                                ${idx < data.sequence.length - 1 ? `<div class="absolute top-full w-0.5 h-4 ${lineCol} opacity-20"></div>` : ''}
                            </div>
                            <div class="flex-1">
                                <div class="flex justify-between items-center">
                                    <p class="text-xs font-black text-slate-800">${s.name}</p>
                                    <span class="text-[9px] font-black text-slate-300 uppercase">${s.reaching_at}</span>
                                </div>
                                <div class="flex items-center gap-2 mt-1">
                                    <span class="text-[8px] font-bold text-slate-400 uppercase tracking-tighter">${s.line} Line</span>
                                    ${s.segment_km > 0 ? `<span class="text-[8px] font-black text-slate-500 uppercase tracking-tighter">+${s.segment_km} KM</span>` : ''}
                                    ${s.dist_km > 0 ? `<span class="text-[8px] font-black text-blue-600 uppercase tracking-tighter">Total: ${s.dist_km} KM</span>` : ''}
                                    ${s.predicted_load ? `<span class="text-[7px] font-black px-1.5 py-0.5 bg-slate-100 rounded text-slate-500 uppercase tracking-tighter">${s.predicted_load} Crowd</span>` : ''}
                                </div>
                            </div>
                        `;
                        stopsCont.appendChild(sDiv);
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
                    
                    const loadColor = s.predicted_load === 'High' ? 'text-red-600 bg-red-50' : 
                                     s.predicted_load === 'Medium' ? 'text-amber-600 bg-amber-50' : 'text-emerald-600 bg-emerald-50';
                    
                    step.innerHTML = `
                        <div class="absolute -left-[49px] w-4 h-4 rounded-full border-4 border-white shadow-lg ${s.line === 'Red' ? 'bg-red-500' : s.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500'}"></div>
                        <div class="flex-1">
                            <div class="flex items-center gap-3">
                                <p class="font-black text-slate-800 text-lg tracking-tight">${s.name}</p>
                                <span class="px-2 py-0.5 rounded-md text-[8px] font-black uppercase tracking-widest ${loadColor}">${s.predicted_load} LOAD</span>
                            </div>
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
                        
                        const color = t.line === 'Red' ? '#ef4444' : t.line === 'Blue' ? '#3b82f6' : '#10b981';
                        const accent = t.line === 'Red' ? '#fee2e2' : t.line === 'Blue' ? '#dbeafe' : '#dcfce7';
                        let marker = trainMarkers.get(t.trip_id);
                        
                        if(!marker) {
                            // Line-specific train designs
                            let svgPath = '';
                            let extraElements = '';
                            
                            if (t.line === 'Red') {
                                // Red Line: Sleek, high-speed look
                                svgPath = `<rect x="10" y="8" width="30" height="14" rx="2" fill="${color}" stroke="white" stroke-width="1.5"/>
                                           <path d="M 40,8 L 48,15 L 40,22 Z" fill="${color}" stroke="white" stroke-width="1.5"/>`;
                                extraElements = `<text x="18" y="18" font-family="Plus Jakarta Sans" font-size="8" font-weight="900" fill="white" fill-opacity="0.9">R</text>`;
                            } else if (t.line === 'Blue') {
                                // Blue Line: Modern, aerodynamic
                                svgPath = `<path d="M 10,10 Q 15,8 35,8 L 45,15 L 35,22 Q 15,22 10,20 Z" fill="${color}" stroke="white" stroke-width="1.5"/>`;
                                extraElements = `<text x="18" y="18" font-family="Plus Jakarta Sans" font-size="8" font-weight="900" fill="white" fill-opacity="0.9">B</text>`;
                            } else {
                                // Green Line: Robust, eco-aesthetic
                                svgPath = `<rect x="8" y="8" width="35" height="14" rx="4" fill="${color}" stroke="white" stroke-width="1.5"/>
                                           <path d="M 43,10 L 48,15 L 43,20 Z" fill="${color}" stroke="white" stroke-width="1.5"/>`;
                                extraElements = `<circle cx="15" cy="15" r="4" fill="white" fill-opacity="0.2"/>
                                                 <text x="22" y="18" font-family="Plus Jakarta Sans" font-size="8" font-weight="900" fill="white" fill-opacity="0.9">G</text>`;
                            }

                            const trainIcon = L.divIcon({
                                className: 'train-icon',
                                html: `<div class="train-shape-inner">
                                    <svg width="60" height="40" viewBox="0 0 60 40">
                                        <filter id="glow-${t.trip_id}"><feGaussianBlur stdDeviation="2" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
                                        <!-- Shadow -->
                                        <rect x="12" y="12" width="36" height="16" rx="4" fill="black" fill-opacity="0.1"/>
                                        <!-- Main Train Body -->
                                        ${svgPath}
                                        <!-- Line Marker -->
                                        ${extraElements}
                                        <!-- Windows -->
                                        <rect x="14" y="11" width="4" height="4" fill="${accent}" fill-opacity="0.8" rx="1"/>
                                        <rect x="22" y="11" width="4" height="4" fill="${accent}" fill-opacity="0.8" rx="1"/>
                                        <rect x="30" y="11" width="4" height="4" fill="${accent}" fill-opacity="0.8" rx="1"/>
                                        <!-- Front Light -->
                                        <circle cx="44" cy="15" r="2" fill="#fff" filter="url(#glow-${t.trip_id})">
                                            <animate attributeName="opacity" values="0.4;1;0.4" dur="1s" repeatCount="indefinite" />
                                        </circle>
                                    </svg>
                                </div>`,
                                iconSize: [60, 40],
                                iconAnchor: [30, 20]
                            });
                            marker = L.marker([0, 0], { icon: trainIcon, zIndexOffset: 1000 }).addTo(map);
                            trainMarkers.set(t.trip_id, marker);
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

        window.onload = () => {
            const urlParams = new URLSearchParams(window.location.search);
            const tabUrl = urlParams.get('tab');
            const initialTab = "{{ initial_tab|default('home') }}";
            
            showTab(tabUrl || initialTab);

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
