import os
import csv
import json
import math
import random
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template_string, jsonify, request
from google import genai
from google.genai import types

app = Flask(__name__)

# ==========================================
# 0. API CONFIGURATION & CREDENTIALS
# ==========================================

# Pull API Keys from Environment (Managed in AI Studio Settings)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_PLATFORM_KEY', '')

# Initialize Gemini Client if key is available
ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Gemini Init Error: {e}")


# ==========================================
# 1. ENHANCED DATA ENGINEERING & AI LOGIC
# ==========================================

def get_ist_now():
    """Returns current time in India Standard Time (UTC+5:30) with optional simulation offset."""
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
        except:
            pass
    return get_ist_now()


def get_ai_insight(prompt):
    """Uses Gemini to generate real-time transit insights if key is provided."""
    if not ai_client:
        return None
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Inference Error: {e}")
        return None


def predict_load_ai(station_name, hour, is_weekend=False, weather=None):
    """Predicts load using the logic from the trained dataset formula with optional weather influence."""
    # Base logic (Fast fallback)
    is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0
    is_it_hub = 1 if station_name in ['Hitech City', 'Madhapur', 'Raidurg'] else 0
    is_festival = 0  # Default for live

    score = 50 + (is_peak * 100) + (is_it_hub * 80) + (is_festival * 120)
    if is_weekend: score -= 20

    if weather:
        if "Rain" in weather.get('condition', ''):
            score += 25
        if weather.get('temp', 30) > 35:
            score += 15

    recommendation = "🟢 Good to travel"
    if score > 200:
        load_lvl, recommendation = "High", "🔴 High Rush"
    elif score > 140:
        load_lvl, recommendation = "M-High", "🟡 Rush but manageable"
    elif score > 100:
        load_lvl, recommendation = "Medium", "🟢 Seat will be there"
    else:
        load_lvl, recommendation = "Low", "🟢 Good to travel"

    # Neural Enhancement: Attempt to get a real-world tip from Gemini
    if ai_client and random.random() < 0.3:  # Don't over-call during dev
        insight_prompt = f"Give a very short (10 words max) transit tip for someone at {station_name} metro station in Hyderabad at {hour}:00. Weather is {weather.get('condition', 'Unknown')} at {weather.get('temp', 'unknown')}C. Be witty."
        insight = get_ai_insight(insight_prompt)
        if insight:
            recommendation = f"{recommendation} | AI Insight: {insight}"

    return load_lvl, recommendation


def get_live_weather(lat=None, lng=None):
    """Fetches real-time weather from Open-Meteo with extra metrics."""
    if lat is None or lng is None:
        lat, lng = 17.3850, 78.4867  # Default Hyderabad

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true&hourly=relative_humidity_2m,visibility"
        data = requests.get(url, timeout=2).json()

        wmo_mapping = {
            0: "Sunny", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Foggy", 48: "Rime Fog", 51: "Light Drizzle", 53: "Moderate Drizzle",
            55: "Dense Drizzle", 61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
            71: "Slight Snow", 73: "Moderate Snow", 75: "Heavy Snow", 80: "Rain Showers",
            95: "Thunderstorm"
        }
        code = data['current_weather']['weathercode']
        condition = wmo_mapping.get(code, "Cloudy")
        temp = data['current_weather']['temperature']

        # New: Alert Categorization Logic
        alert = None
        severity = "normal"  # normal, caution, critical
        if temp > 40:
            alert = "Extreme Heatwave: Avoid non-essential outdoor travel. Metro stations are cooling hubs."
            severity = "critical"
        elif temp > 35:
            alert = "Intense Heat: Stay hydrated. Prefer AC cabins for travel."
            severity = "caution"

        if code in [61, 63, 65, 80, 81, 82]:
            alert = "Heavy Rain Detected: Visibility low. Expect minor delays. Stick to indoor platforms."
            severity = "caution"
            if code == 65:
                alert = "Relentless Rain: Storm conditions likely. Use Metro for safest dry passage."
                severity = "critical"
        elif code in [95, 96, 99]:
            alert = "Thunderstorm Warning: High voltage network active. Safety first inside stations."
            severity = "critical"

        ist_now = get_ist_now()
        h_idx = ist_now.hour

        return {
            'temp': temp,
            'condition': condition,
            'humidity': data['hourly']['relative_humidity_2m'][h_idx] if h_idx < len(
                data['hourly']['relative_humidity_2m']) else 45,
            'visibility': (data['hourly']['visibility'][h_idx] / 1000) if h_idx < len(
                data['hourly']['visibility']) else 10,
            'aqi': random.randint(30, 70),
            'alert': alert,
            'severity': severity
        }
    except:
        return {'temp': 30, 'condition': 'Clear Sky', 'humidity': 45, 'visibility': 10, 'aqi': 42, 'alert': None,
                'severity': 'normal'}


def generate_ai_dataset():
    """Generates a high-fidelity ridership dataset covering all 57 stations for the prediction model."""
    stations_for_df = [s['name'] for s in STATIONS_LIST]
    sample_size = 2500  # Increased for better 'training' resolution

    with open("final_metro_dataset.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['stop_name', 'arrival_time', 'platform', 'hour', 'day_of_week', 'is_peak', 'is_weekend', 'is_it_hub',
             'temperature', 'rainfall', 'is_festival', 'ridership'])

        it_stations = ['Hitech City', 'Madhapur', 'Raidurg']
        base_time = get_ist_now() - timedelta(days=30)  # 1 month of history

        for i in range(sample_size):
            # Spread across 30 days
            arrival = base_time + timedelta(minutes=i * 15)
            stop_name = random.choice(stations_for_df)
            hour = arrival.hour
            dow = arrival.weekday()

            is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0
            is_weekend = 1 if dow >= 5 else 0
            is_it_hub = 1 if stop_name in it_stations else 0
            temp = 25 + (hour % 12)
            rainfall = 1 if random.random() < 0.1 else 0
            is_festival = 1 if (arrival.day == 24 and arrival.month == 10) else 0  # Dussehra mock

            # Weighted Ridership Logic (Simulated Training Weights)
            base_ridership = 40
            if is_peak: base_ridership += 110
            if is_it_hub: base_ridership += 95
            if is_festival: base_ridership += 150
            if is_weekend: base_ridership -= 35

            # Station-specific density factor
            station_density = 1.2 if stop_name in ['Ameerpet', 'MGBS', 'Secunderabad East'] else 1.0

            noise = random.gauss(0, 5)
            ridership = int((base_ridership * station_density) + noise)

            writer.writerow(
                [stop_name, arrival.strftime('%Y-%m-%d %H:%M:%S'), random.choice(['1', '2']), hour, dow, is_peak,
                 is_weekend, is_it_hub, temp, rainfall, is_festival, max(0, ridership)])
    return True


# ==========================================
# 2. FULL DATASET (57 STATIONS)
# ==========================================
STATIONS_LIST = [
    # RED LINE (1-27)
    {'id': 'R1', 'name': 'Miyapur', 'line': 'Red', 'lat': 17.4933, 'lng': 78.3484,
     'amenities': ['Large Parking Area', 'Food Court', 'Medical Center'],
     'description': 'Miyapur is a major residential hub and the western terminal of the metro. Nearby you’ll find DMart, GSM Mall, and multiple local shopping complexes. TSRTC bus connectivity is strong with Miyapur Bus Depot nearby, while the closest MMTS access is via Hafeezpet (a short ride away). The area is mostly residential but growing rapidly with apartments, making it ideal for commuters rather than tourism.'},
    {'id': 'R2', 'name': 'JNTU College', 'line': 'Red', 'lat': 17.4877, 'lng': 78.3557,
     'description': 'This station serves Jawaharlal Nehru Technological University and is surrounded by student hostels, bookstores, and cafes. Nearby shopping is mostly local markets, and bus connectivity is frequent. The closest MMTS is Kukatpally/Hafeezpet. It’s not a tourist spot but a strong student ecosystem.'},
    {'id': 'R3', 'name': 'KPHB Colony', 'line': 'Red', 'lat': 17.4834, 'lng': 78.3883,
     'description': 'KPHB is a lively area with Forum Sujana Mall, South India Shopping Mall, Manjeera Mall (nearby Kukatpally). TSRTC buses are frequent, and Kukatpally bus stops are close. MMTS access is via Hafeezpet. It’s great for shopping, street food, and casual hangouts.'},
    {'id': 'R4', 'name': 'Kukatpally', 'line': 'Red', 'lat': 17.4842, 'lng': 78.3986,
     'description': 'One of the busiest areas, Kukatpally has Manjeera Mall, Forum Mall, and extensive street shopping. It is well connected by buses and close to Kukatpally Y Junction. MMTS is accessible via Hafeezpet. Not a tourist hub, but excellent for shopping and food.'},
    {'id': 'R5', 'name': 'Dr. B. R. Ambedkar Balanagar', 'line': 'Red', 'lat': 17.4776, 'lng': 78.4216,
     'description': 'Balanagar is an industrial area with limited malls but strong TSRTC bus connectivity. The closest MMTS is Balanagar MMTS station. It’s mainly functional for workers; not much tourism here.'},
    {'id': 'R6', 'name': 'Moosapet', 'line': 'Red', 'lat': 17.4727, 'lng': 78.4279,
     'description': 'Moosapet has small commercial complexes and easy bus connectivity. Nearby MMTS stations include Bharat Nagar and Balanagar. The area is practical for commuting but not tourist-oriented.'},
    {'id': 'R7', 'name': 'Bharat Nagar', 'line': 'Red', 'lat': 17.4646, 'lng': 78.4357,
     'description': 'This station connects to Bharat Nagar MMTS Station, making it a key multimodal point. It has local shops and moderate bus connectivity. No major malls or tourist spots.'},
    {'id': 'R8', 'name': 'Erragadda', 'line': 'Red', 'lat': 17.4572, 'lng': 78.4412,
     'description': 'Erragadda is known for hospitals like ESI Hospital Hyderabad. Nearby are small shopping areas and bus stops. MMTS is accessible via Bharat Nagar. It’s more of a healthcare hub than a tourist destination.'},
    {'id': 'R9', 'name': 'ESI Hospital', 'line': 'Red', 'lat': 17.4517, 'lng': 78.4457,
     'description': 'This station directly serves medical facilities including ESI Hospital. Bus connectivity is strong, and MMTS access is via nearby stations like Bharat Nagar. No malls or tourist attractions here.'},
    {'id': 'R10', 'name': 'S.R. Nagar', 'line': 'Red', 'lat': 17.4442, 'lng': 78.4484,
     'description': 'SR Nagar is full of coaching institutes and student hostels. It has small shopping complexes and strong bus connectivity. The nearest MMTS is Nature Cure (towards Begumpet side). It’s a student hub rather than a tourist place.'},
    {'id': 'R11', 'name': 'Ameerpet', 'line': 'Red', 'lat': 17.4334, 'lng': 78.4484,
     'amenities': ['Transfer Area', 'Ameerpet Metro Mall', 'Apollo Clinic'],
     'description': 'Ameerpet Metro Station is the busiest interchange. Nearby malls include GVK One (Punjagutta), Hyderabad Central Mall. Bus connectivity is excellent, and MMTS stations like Begumpet are nearby. It’s a commercial hotspot with shopping, coaching centers, and food.'},
    {'id': 'R12', 'name': 'Punjagutta', 'line': 'Red', 'lat': 17.4261, 'lng': 78.4522,
     'amenities': ['Next Galleria Mall', 'PVR Cinemas'],
     'description': 'Punjagutta is a premium commercial area with GVK One Mall, Next Galleria Mall. TSRTC buses are frequent, and MMTS is accessible via Begumpet. It’s great for shopping and dining.'},
    {'id': 'R13', 'name': 'Irrum Manzil', 'line': 'Red', 'lat': 17.4184, 'lng': 78.4557,
     'description': 'This station serves government offices and business areas. Limited shopping options but good bus connectivity. Nearest MMTS is Lakdikapul (via short travel). Not a tourist area.'},
    {'id': 'R14', 'name': 'Khairatabad', 'line': 'Red', 'lat': 17.4101, 'lng': 78.4611,
     'description': 'Khairatabad connects to Hussain Sagar, Birla Mandir Hyderabad, and Lumbini Park. Good TSRTC bus links and nearby MMTS Lakdikapul. This is a key tourist-friendly station.'},
    {'id': 'R15', 'name': 'Lakdi-ka-pul', 'line': 'Red', 'lat': 17.4024, 'lng': 78.4657,
     'description': 'Lakdikapul offers access to hotels, offices, and MMTS Lakdikapul station. Bus connectivity is strong. Tourist spots like Birla Mandir and Necklace Road are nearby.'},
    {'id': 'R16', 'name': 'Assembly', 'line': 'Red', 'lat': 17.3984, 'lng': 78.4723,
     'description': 'This station serves the Telangana Assembly area. Limited shopping, but strong bus routes. MMTS is nearby at Lakdikapul. Not tourist-focused.'},
    {'id': 'R17', 'name': 'Nampally', 'line': 'Red', 'lat': 17.3916, 'lng': 78.4757,
     'description': 'Nampally connects to Hyderabad Deccan Railway Station and Exhibition Grounds. Good bus services and railway connectivity. Tourist places include markets and heritage zones.'},
    {'id': 'R18', 'name': 'Gandhi Bhavan', 'line': 'Red', 'lat': 17.3872, 'lng': 78.4784,
     'description': 'A commercial and political area with local shops and bus access. MMTS is via Nampally. No major tourist attractions.'},
    {'id': 'R19', 'name': 'Osmania Medical College', 'name_alias': 'OMC', 'line': 'Red', 'lat': 17.3824, 'lng': 78.4812,
     'description': 'Serves Osmania Medical College and hospitals. Bus connectivity is strong. Not a shopping or tourist hub.'},
    {'id': 'R20', 'name': 'MG Bus Station', 'name_alias': 'MGBS', 'line': 'Red', 'lat': 17.3776, 'lng': 78.4815,
     'amenities': ['Interstate Bus Station', 'Multi-level Parking'],
     'description': 'Mahatma Gandhi Bus Station is a massive TSRTC hub and interchange with Green Line. Nearby attractions include Charminar, Laad Bazaar, and Salar Jung Museum. No malls, but rich heritage.'},
    {'id': 'R21', 'name': 'Malakpet', 'line': 'Red', 'lat': 17.3752, 'lng': 78.4907,
     'description': 'Malakpet has local markets and bus connectivity. Nearby MMTS Malakpet station is available. It’s a residential and cultural area.'},
    {'id': 'R22', 'name': 'New Market', 'line': 'Red', 'lat': 17.3712, 'lng': 78.5084,
     'description': 'Traditional shopping area with local vendors and bus access. No MMTS directly, but nearby Malakpet station. Good for budget shopping.'},
    {'id': 'R23', 'name': 'Musarambagh', 'line': 'Red', 'lat': 17.3684, 'lng': 78.5212,
     'description': 'Residential locality with TSRTC buses and limited shopping. No major tourist attractions.'},
    {'id': 'R24', 'name': 'Dilsukhnagar', 'line': 'Red', 'lat': 17.3657, 'lng': 78.5357,
     'description': 'Major commercial hub with shopping complexes, street food, and bus connectivity. MMTS Malakpet is nearby. It’s one of the busiest areas in east Hyderabad.'},
    {'id': 'R25', 'name': 'Chaitanyapuri', 'line': 'Red', 'lat': 17.3612, 'lng': 78.5484,
     'description': 'Residential area with small shops and good bus connectivity. Not a tourist spot.'},
    {'id': 'R26', 'name': 'Victoria Memorial', 'line': 'Red', 'lat': 17.3557, 'lng': 78.5512,
     'description': 'Quiet suburban station with minimal commercial activity. Mainly residential.'},
    {'id': 'R27', 'name': 'LB Nagar', 'line': 'Red', 'lat': 17.3458, 'lng': 78.5524,
     'amenities': ['South Terminal', 'Auto Stand'],
     'description': 'The terminal station with strong TSRTC bus depot connectivity and highway access. It has shopping complexes and serves as a major transit junction for eastern Hyderabad.'},

    # BLUE LINE (1-23)
    {'id': 'B1', 'name': 'Nagole', 'line': 'Blue', 'lat': 17.3941, 'lng': 78.5668,
     'description': 'Nagole is the eastern terminal of the Blue Line. It has a metro depot and good bus connectivity to outer Hyderabad. The area is developing with residential projects and small commercial centers.'},
    {'id': 'B2', 'name': 'Uppal', 'line': 'Blue', 'lat': 17.3984, 'lng': 78.5684,
     'description': 'A major residential and commercial hub with markets, shopping complexes, and strong TSRTC bus connectivity. MMTS access is available at Uppal station.'},
    {'id': 'B3', 'name': 'Stadium', 'line': 'Blue', 'lat': 17.4021, 'lng': 78.5712,
     'description': 'This station provides access to Rajiv Gandhi International Cricket Stadium. It becomes very busy during matches. Bus connectivity is strong, and MMTS is via Uppal.'},
    {'id': 'B4', 'name': 'NGRI', 'line': 'Blue', 'lat': 17.4084, 'lng': 78.5684,
     'description': 'Named after National Geophysical Research Institute, this station serves research facilities and residential areas. Limited shopping, but good bus connectivity.'},
    {'id': 'B5', 'name': 'Habsiguda', 'line': 'Blue', 'lat': 17.4212, 'lng': 78.5584,
     'description': 'A residential and research area with institutes like CCMB nearby. Bus connectivity is good, and MMTS is via Habsiguda/Jamai Osmania.'},
    {'id': 'B6', 'name': 'Tarnaka', 'line': 'Blue', 'lat': 17.4357, 'lng': 78.5472,
     'description': 'Tarnaka serves students and staff of Osmania University. It has local markets, eateries, and good bus connectivity. MMTS is accessible via Jamai Osmania station.'},
    {'id': 'B7', 'name': 'Mettuguda', 'line': 'Blue', 'lat': 17.4484, 'lng': 78.5342,
     'description': 'A residential area with railway connectivity nearby. Bus services are moderate, and MMTS access is via Secunderabad. No major malls or tourist spots.'},
    {'id': 'B8', 'name': 'Secunderabad East', 'line': 'Blue', 'lat': 17.4546, 'lng': 78.5212,
     'description': 'Close to Secunderabad Railway Station, this station is a major transit hub. Bus, rail, and metro connectivity converge here. Nearby shopping areas and hotels make it very active.'},
    {'id': 'B9', 'name': 'Parade Ground', 'line': 'Blue', 'lat': 17.4452, 'lng': 78.4985,
     'description': 'An important interchange with the Green Line. Located in Secunderabad cantonment area, it has strong bus connectivity and access to markets. MMTS is nearby at Secunderabad station.'},
    {'id': 'B10', 'name': 'Paradise', 'line': 'Blue', 'lat': 17.4568, 'lng': 78.4972,
     'description': 'Famous for Paradise Biryani, this station is a food hotspot. Nearby markets and shopping areas make it lively. Bus connectivity is strong, and MMTS is accessible via Secunderabad.'},
    {'id': 'B11', 'name': 'Rasoolpura', 'line': 'Blue', 'lat': 17.4502, 'lng': 78.4851,
     'description': 'A dense locality with local markets and small businesses. Bus services are available, and MMTS is accessible via Begumpet. It’s mainly a transit stop.'},
    {'id': 'B12', 'name': 'Prakash Nagar', 'line': 'Blue', 'lat': 17.4468, 'lng': 78.4720,
     'description': 'A smaller station serving residential and office areas. Limited shopping options but good bus connectivity. MMTS is via Begumpet. Not a tourist area.'},
    {'id': 'B13', 'name': 'Begumpet', 'line': 'Blue', 'lat': 17.4398, 'lng': 78.4612,
     'description': 'Begumpet is a business and residential hub with access to Begumpet Airport. Nearby are Hyderabad Central Mall and lifestyle stores. It connects directly to Begumpet MMTS Station, making it a key multimodal station.'},
    {'id': 'B14', 'name': 'Ameerpet', 'line': 'Blue', 'lat': 17.4334, 'lng': 78.4484, 'name_alias': 'Ameerpet',
     'description': 'Ameerpet Metro Station is the central interchange between Red and Blue lines. Nearby are Hyderabad Central Mall, GVK One Mall, and numerous coaching institutes. Bus connectivity is excellent, and MMTS stations like Begumpet are close.'},
    {'id': 'B15', 'name': 'Madhura Nagar', 'line': 'Blue', 'lat': 17.4280, 'lng': 78.4420,
     'description': 'A quiet residential locality with small shops and easy bus access. It mainly serves daily commuters. MMTS is accessible via nearby stations like Nature Cure.'},
    {'id': 'B16', 'name': 'Yousufguda', 'line': 'Blue', 'lat': 17.4246, 'lng': 78.4357,
     'description': 'A residential and sports-focused area, Yousufguda has local markets and stadium facilities. Bus connectivity is good, and MMTS is nearby at Begumpet. It’s more residential than commercial.'},
    {'id': 'B17', 'name': 'Road No. 5 Jubilee Hills', 'line': 'Blue', 'lat': 17.4284, 'lng': 78.4239,
     'description': 'This station connects to elite shopping streets and cafes in Jubilee Hills. It is close to KBR National Park, a major green space for walking and fitness. Bus services are frequent, and MMTS is via Begumpet.'},
    {'id': 'B18', 'name': 'Jubilee Hills Check Post', 'line': 'Blue', 'lat': 17.4310, 'lng': 78.4120,
     'description': 'A premium locality known for upscale residences, restaurants, and boutiques. Nearby malls include GVK One Mall and luxury dining places. Bus connectivity is available, and MMTS access is via Begumpet. It’s a high-end lifestyle area.'},
    {'id': 'B19', 'name': 'Peddamma Gudi', 'line': 'Blue', 'lat': 17.4330, 'lng': 78.4050,
     'description': 'This station is named after Peddamma Temple, a famous religious site. The area has local markets, eateries, and good bus connectivity. MMTS is accessible via nearby stations. It attracts devotees especially during festivals.'},
    {'id': 'B20', 'name': 'Madhapur', 'line': 'Blue', 'lat': 17.4357, 'lng': 78.3984,
     'description': 'Madhapur is a vibrant IT and nightlife hub filled with cafes, pubs, and coworking spaces. It has access to shopping complexes and is close to Inorbit Mall. TSRTC buses connect well, and MMTS is via HITEC City. It’s one of the most happening areas in Hyderabad.'},
    {'id': 'B21', 'name': 'Durgam Cheruvu', 'line': 'Blue', 'lat': 17.4398, 'lng': 78.3857,
     'description': 'Located near the scenic Durgam Cheruvu, this station is popular for evening outings. The cable bridge, lakefront parks, and cafes are key attractions. Bus connectivity is moderate, and MMTS is via HITEC City. No major malls right next to it, but Inorbit is nearby.'},
    {'id': 'B22', 'name': 'HITEC City', 'line': 'Blue', 'lat': 17.4474, 'lng': 78.3762,
     'description': 'This station serves HITEC City, one of India’s biggest tech hubs. Nearby are Inorbit Mall, Shilparamam Crafts Village, and multiple tech parks. Bus connectivity is strong, and HITEC City MMTS Station is within reach. It’s a major hotspot for offices, shopping, and nightlife.'},
    {'id': 'B23', 'name': 'Raidurg', 'line': 'Blue', 'lat': 17.4429, 'lng': 78.3750,
     'description': 'Raidurg is the western terminal of the Blue Line and sits at the edge of Hyderabad’s Financial District. It provides access to major IT campuses and offices. Nearby shopping includes Inorbit Mall and Sarath City Capital Mall (a short ride away). TSRTC buses connect Raidurg to Gachibowli and other IT areas, while the nearest MMTS access is via HITEC City station. Tourist spots include Durgam Cheruvu and the cable bridge nearby, making it a mix of work and leisure.'},

    # GREEN LINE (1-10)
    {'id': 'G1', 'name': 'JBS', 'line': 'Green', 'lat': 17.4510, 'lng': 78.5002,
     'description': 'Jubilee Bus Station is a major TSRTC terminal and the northern end of the Green Line. It has strong bus connectivity and local markets. MMTS access is via Secunderabad East.'},
    {'id': 'G2', 'name': 'Parade Ground', 'line': 'Green', 'lat': 17.4452, 'lng': 78.4985,
     'name_alias': 'Parade Ground',
     'description': 'An important interchange with the Blue Line. Located in Secunderabad cantonment area, it has strong bus connectivity and access to markets. MMTS is nearby at Secunderabad station.'},
    {'id': 'G3', 'name': 'Secunderabad West', 'line': 'Green', 'lat': 17.4410, 'lng': 78.5020,
     'description': 'Close to Secunderabad Railway Station, this station serves busy commercial areas. Bus and rail connectivity make it a major transit point. Nearby hotels and markets.'},
    {'id': 'G4', 'name': 'Gandhi Hospital', 'line': 'Green', 'lat': 17.4335, 'lng': 78.5020,
     'description': 'Directly serves Gandhi Hospital and medical college. Strong bus connectivity. Not a shopping or tourist area.'},
    {'id': 'G5', 'name': 'Musheerabad', 'line': 'Green', 'lat': 17.4215, 'lng': 78.5030,
     'description': 'A residential and commercial locality with local shops and bus access. MMTS is accessible via Secunderabad.'},
    {'id': 'G6', 'name': 'RTC X Roads', 'line': 'Green', 'lat': 17.4080, 'lng': 78.5040,
     'description': 'Famous for cinema halls and local eateries. It has strong bus connectivity and is a popular hangout spot for locals. No major malls, but excellent for food.'},
    {'id': 'G7', 'name': 'Chikkadpally', 'line': 'Green', 'lat': 17.4025, 'lng': 78.4965,
     'description': 'Residential area with local markets and coaching centers. Good bus connectivity. Not a tourist destination.'},
    {'id': 'G8', 'name': 'Narayanaguda', 'line': 'Green', 'lat': 17.3964, 'lng': 78.4893,
     'description': 'A student-heavy area with colleges and coaching centers. Local shops and bus connectivity are strong. MMTS is nearby at Kachiguda station.'},
    {'id': 'G9', 'name': 'Sultan Bazaar', 'line': 'Green', 'lat': 17.3888, 'lng': 78.4842,
     'description': 'Located near the historic Koti market area, this station is great for budget shopping. Heritage sites and heritage buildings are nearby. Strong bus link to Koti Bus Terminal.'},
    {'id': 'G10', 'name': 'MG Bus Station', 'line': 'Green', 'lat': 17.3776, 'lng': 78.4815, 'name_alias': 'MGBS',
     'description': 'Mahatma Gandhi Bus Station is a massive TSRTC hub and interchange with Blue Line. Nearby attractions include Charminar and Salar Jung Museum. No malls, but rich heritage.'}
]

LANDMARKS = [
    {'name': 'Charminar', 'lat': 17.3616, 'lng': 78.4747, 'type': 'Monuments'},
    {'name': 'Hussain Sagar', 'lat': 17.4239, 'lng': 78.4738, 'type': 'Nature'},
    {'name': 'Golconda Fort', 'lat': 17.3833, 'lng': 78.4011, 'type': 'Monuments'},
    {'name': 'Birla Mandir', 'lat': 17.4062, 'lng': 78.4691, 'type': 'Temple'},
    {'name': 'Salar Jung Museum', 'lat': 17.3714, 'lng': 78.4804, 'type': 'Museum'},
    {'name': 'Chowmahalla Palace', 'lat': 17.3579, 'lng': 78.4717, 'type': 'Palace'},
    {'name': 'Lumbini Park', 'lat': 17.4116, 'lng': 78.4735, 'type': 'Nature'},
    {'name': 'Nehru Zoological Park', 'lat': 17.3501, 'lng': 78.4516, 'type': 'Nature'},
    {'name': 'Osmania University', 'lat': 17.4137, 'lng': 78.5284, 'type': 'Education'},
    {'name': 'Secretariat', 'lat': 17.4101, 'lng': 78.4725, 'type': 'Government'}
]

CONNECTIONS = {
    'Red': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16',
            'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27'],
    'Blue': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16',
             'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23'],
    'Green': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
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
    ensure_gtfs(force=False)  # Only generate if missing
    now = get_app_now()
    return render_template_string(HTML_TEMPLATE,
                                  ALL_STATIONS=STATIONS_LIST,
                                  CONNECTIONS=CONNECTIONS,
                                  LANDMARKS=LANDMARKS,
                                  GOOGLE_MAPS_API_KEY=GOOGLE_MAPS_API_KEY,
                                  SERVER_TIME_ISO=now.isoformat(),
                                  SERVER_TIME_EPOCH=now.timestamp()
                                  )


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    # Enrich with server timestamp
    data['server_received'] = get_app_now().strftime('%Y-%m-%d %H:%M:%S')

    success = save_feedback_to_cloud(data)
    return jsonify({'status': 'success' if success else 'failed'})


@app.route('/api/nearest', methods=['GET', 'POST'])
def api_nearest():
    """Returns nearest station, departures, and predictive load based on simulated GTFS."""
    if request.method == 'POST':
        data = request.json or {}
    else:
        data = request.args

    dist = 0

    if 'station_id' in data:
        matches = [s for s in STATIONS_LIST if s['id'] == data['station_id']]
        if not matches:
            return jsonify({'error': 'Station not found'}), 404
        nearest = matches[0]
        lat, lng = nearest['lat'], nearest['lng']
    else:
        try:
            lat = float(data.get('lat', 17.3850))
            lng = float(data.get('lng', 78.4867))
        except:
            lat, lng = 17.3850, 78.4867
        nearest = min(STATIONS_LIST, key=lambda s: haversine(lat, lng, s['lat'], s['lng']))
        dist = haversine(lat, lng, nearest['lat'], nearest['lng'])

    # Urban Walking Adjustment: approx 1.35x crow-flies distance for city streets
    walk_dist = dist * 1.35
    walking_mins = int((walk_dist / 5.0) * 60)

    # Urban Driving Adjustment: approx 1.5x crow-flies distance for city streets
    drive_dist = dist * 1.5
    driving_mins = int((drive_dist / 25.0) * 60)  # Avg 25km/h in city traffic

    # Precise Geofence Check
    range_status = "In City"
    geofence_msg = "Google Cloud Uplink: Active | Location Accuracy: High"
    if dist > 300:
        range_status = "Remote"
        geofence_msg = "Remote Mode: You are far from Hyderabad. Displaying City Hub (Ameerpet)."
    elif dist > 50:
        range_status = "Out of City"
        geofence_msg = "External Range: You are outside the primary metro service boundary."
    elif dist > 15:
        range_status = "Peripheral"
        geofence_msg = "Edge Zone: You are in the Hyderabad suburbs."

    name = nearest.get('name_alias', nearest['name'])
    matching_ids = [s['id'] for s in STATIONS_LIST if s.get('name_alias', s['name']) == name]

    trips = ensure_gtfs()
    now = get_app_now()

    # Handle simulation override
    sim_h = data.get('sim_hour')
    if sim_h:
        try:
            now = now.replace(hour=int(sim_h), minute=30, second=0)
        except:
            pass

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
    weather = get_live_weather(lat=lat, lng=lng)
    is_weekend = now.weekday() >= 5
    load_val, load_label = predict_load_ai(name, now.hour, is_weekend=is_weekend, weather=weather)

    # Landmark Logic (Simplified for brevity)
    landmarks = [{'name': 'Local Hub', 'dist': 0.5}]
    if name == 'Ameerpet':
        landmarks = [{'name': 'Next Galleria Mall', 'dist': 0.2}]
    elif name == 'Hitech City':
        landmarks = [{'name': 'Cyber Towers', 'dist': 0.3}]

    # Timings
    first_train = '06:00 AM'
    last_train = '11:00 PM' if not is_weekend else '11:30 PM'

    # Optimized Global GTFS Stats (Trips active right now)
    trip_times = {}
    line_active_counts = {'Red': 0, 'Blue': 0, 'Green': 0}
    for row in trips:
        tid = row['trip_id']
        t_arr = row['arrival_time']
        line = row['line']
        if tid not in trip_times:
            trip_times[tid] = {'min': t_arr, 'max': t_arr, 'line': line}
        else:
            if t_arr < trip_times[tid]['min']: trip_times[tid]['min'] = t_arr
            if t_arr > trip_times[tid]['max']: trip_times[tid]['max'] = t_arr

    active_count = 0
    for tid, times in trip_times.items():
        if times['min'] <= now_str <= times['max']:
            active_count += 1
            line_active_counts[times['line']] += 1

    # Format upcoming for display
    upcoming.sort(key=lambda x: x['arrival_time'])
    for t in upcoming:
        try:
            h, m, s = map(int, t['arrival_time'].split(':'))
            dt = now.replace(hour=h, minute=m, second=s)
            t['arrival_time'] = dt.strftime('%I:%M %p')
        except:
            pass

    return jsonify({
        'station': nearest,
        'distance': round(dist, 2),
        'walk_dist': round(dist * 1.35, 2),
        'walking_mins': walking_mins,
        'drive_dist': round(dist * 1.5, 2),
        'driving_mins': driving_mins,
        'range_status': range_status,
        'geofence_msg': geofence_msg,
        'upcoming': upcoming,
        'load_val': load_val,
        'load_label': load_label,
        'active_trips': {
            'total': active_count,
            'lines': line_active_counts
        },
        'weather': weather,
        'landmarks': landmarks,
        'first_train': first_train,
        'last_train': last_train,
        'server_time': now.isoformat(),
        'server_epoch': now.timestamp(),
        'greeting': "Good Morning" if 5 <= now.hour < 12 else "Good Afternoon" if 12 <= now.hour < 17 else "Good Evening"
    })


@app.route('/api/weather', methods=['POST'])
def api_weather():
    data = request.json
    lat, lng = data.get('lat'), data.get('lng')
    now = get_app_now()
    weather = get_live_weather(lat=lat, lng=lng)
    weather['server_time'] = now.isoformat()
    weather['server_epoch'] = now.timestamp()
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

    weather = get_live_weather(lat=sequence[0]['lat'], lng=sequence[0]['lng'])
    weather_advice = " AC cooling optimized for intense heat." if weather.get('temp', 30) > 35 else \
        " Rainy conditions detected; transit via tunnels recommended." if "Rain" in weather.get('condition', '') else \
            " Clear skies for a smooth commute." if "Sunny" in weather.get('condition', '') or "Clear" in weather.get(
                'condition', '') else ""

    # AI Recommendation logic
    start_station_name = sequence[0]['name']
    end_station_name = sequence[-1]['name']
    is_weekend = now.weekday() >= 5
    load_val, base_load_label = predict_load_ai(start_station_name, now.hour, is_weekend=is_weekend, weather=weather)

    # NEW: Numerical Load and Peak Intensity Math
    load_pct = 35  # Base
    if (7 <= now.hour <= 10 or 17 <= now.hour <= 21): load_pct += 45
    if start_station_name in ['Hitech City', 'Madhapur', 'Raidurg']: load_pct += 15
    load_pct = min(99.4, load_pct + random.uniform(-3, 3))

    peak_intensity = 0
    if (7 <= now.hour <= 10):
        dist = abs(now.hour - 8.5)
        peak_intensity = 100 - (dist * 30)
    elif (17 <= now.hour <= 21):
        dist = abs(now.hour - 19)
        peak_intensity = 100 - (dist * 20)
    peak_intensity = round(max(0, min(100, peak_intensity)), 1)

    # Base recommendation
    recommendation = (
                         "Empty seats available. Very low crowd." if load_val == "Low" else \
                             "Not too busy. You might get a seat." if load_val == "Medium" else \
                                 "A bit busy. Stay alert." if load_val == "M-High" else \
                                     "Very busy. Try taking the next train if possible."
                     ) + weather_advice

    # Neural Path Logic: Use Gemini
    if ai_client:
        try:
            path_names = " -> ".join([s['name'] for s in sequence])
            ai_prompt = f"As a transit AI, give a one-sentence tip for a commute from {start_station_name} to {end_station_name} in Hyderabad Metro. Route: {path_names}. Time: {now.strftime('%H:%M')}. Weather: {weather.get('condition')}. Keep it helpful and under 20 words."
            ai_rec = get_ai_insight(ai_prompt)
            if ai_rec:
                recommendation = f"{base_load_label}. {ai_rec}"
        except:
            pass

    # INTERCHANGE & GUIDE LOGIC
    guides = []
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
    is_peak_val = "Peak Hour" if (7 <= now.hour <= 10 or 17 <= now.hour <= 21) else "Off-Peak"
    is_it_hub_val = "High" if any(
        n in [s['name'] for s in sequence] for n in ['Hitech City', 'Raidurg', 'Madhapur']) else "Normal"

    # Environmental Analytics
    co2_saved = round(total_km * 0.12, 2)  # kg CO2
    calories = int(total_km * 12 + len(guides) * 25)  # Estimated effort
    trees_saved = round(total_km * 0.05, 3)
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
            'peak': is_peak_val,
            'it_hub': is_it_hub_val,
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

                    # Calculate speed
                    st1 = next((s for s in STATIONS_LIST if s['id'] == s1['station_id']), None)
                    st2 = next((s for s in STATIONS_LIST if s['id'] == s2['station_id']), None)
                    speed = 0
                    if st1 and st2 and total_duration > 0:
                        dist = haversine(st1['lat'], st1['lng'], st2['lat'], st2['lng'])
                        speed = (dist / (total_duration / 3600.0))

                    active_trains.append({
                        'trip_id': tid,
                        'line': s1['line'],
                        'from_id': s1['station_id'],
                        'to_id': s2['station_id'],
                        'progress': progress,
                        't1_epoch': t1.timestamp(),
                        'duration': total_duration,
                        'direction': s1['direction'],
                        'final_stop': s1['final_stop'],
                        'speed': round(speed, 1) if speed < 110 else 75
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
    <title>HydMetro | Your Metro Guide</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <script src="https://maps.googleapis.com/maps/api/js?key={{ GOOGLE_MAPS_API_KEY }}&libraries=places"></script>
    <script>
      // Google Maps Geocoder instance
      let geocoder;
      function initGoogleSync() {
          if (typeof google !== 'undefined') {
              geocoder = new google.maps.Geocoder();
              console.log("Google Maps Sync Initialized");
          }
      }
    </script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <style>
        :root {
            --bg: #f8fafc;
            --card-bg: rgba(255, 255, 255, 0.9);
            --border: rgba(226, 232, 240, 0.7);
            --accent: #2563eb;
            --safe-bottom: env(safe-area-inset-bottom, 20px);
        }

        body { 
            font-family: 'Plus Jakarta Sans', sans-serif; 
            background: var(--bg); 
            color: #0f172a; 
            overflow-x: hidden; 
            -webkit-tap-highlight-color: transparent;
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,10%,0.03) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,0.02) 0, transparent 50%);
            background-attachment: fixed;
        }

        .glass-card { 
            background: var(--card-bg); 
            backdrop-filter: blur(12px) saturate(180%); 
            border-radius: 2rem; 
            border: 1px solid var(--border); 
            padding: 24px; 
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 10px 15px -3px rgba(0,0,0,0.03);
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .mobile-card {
            border-radius: 1.5rem;
            padding: 20px;
        }

        .main { 
            max-width: 1400px;
            margin: 0 auto; 
            padding: 0 20px 140px 20px; 
            min-height: 100vh; 
        }

        @media (max-width: 1024px) {
            .main { 
                padding: 16px; 
                padding-top: 80px;
                padding-bottom: 140px; 
            }
        }

        /* High-Fidelity Mobile Navigation */
        .mobile-nav { 
            display: flex !important; 
            position: fixed; 
            bottom: var(--safe-bottom); 
            left: 50%;
            transform: translateX(-50%);
            width: calc(100% - 32px);
            max-width: 440px;
            background: rgba(15, 23, 42, 0.95); 
            backdrop-filter: blur(20px) saturate(200%);
            border-radius: 40px; 
            z-index: 5000;
            padding: 6px; 
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 20px 50px -12px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .mobile-link {
            display: flex; flex-direction: column; align-items: center; gap: 4px;
            color: #64748b; padding: 12px 2px; border-radius: 32px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer;
            flex: 1;
            position: relative;
        }
        .mobile-link i { width: 22px; height: 22px; opacity: 0.7; }
        .mobile-link span { 
            font-size: 9px; font-weight: 700; text-transform: uppercase; 
            letter-spacing: 0.05em; opacity: 0.6;
        }

        .mobile-link.active { color: white; }
        .mobile-link.active i { opacity: 1; transform: scale(1.1); }
        .mobile-link.active span { opacity: 1; }
        .mobile-link.active::after {
            content: '';
            position: absolute;
            bottom: 4px;
            width: 4px;
            height: 4px;
            background: #3b82f6;
            border-radius: 50%;
            box-shadow: 0 0 8px #3b82f6;
        }

        /* Centered CTA Nav Item */
        .mobile-link-cta {
            background: #2563eb;
            color: white !important;
            padding: 16px !important;
            border-radius: 50% !important;
            flex: 0 0 60px !important;
            height: 60px;
            justify-content: center;
            box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.4);
            margin-top: -30px;
            border: 4px solid #1e293b;
        }
        .mobile-link-cta i { opacity: 1 !important; width: 24px; height: 24px; }
        .mobile-link-cta span { display: none; }
        .mobile-link-cta.active { background: #1d4ed8; }

        /* App Header Fix */
        .app-header {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 72px;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(20px);
            z-index: 1500;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid rgba(226, 232, 240, 0.5);
            padding: 0 24px;
        }

        @media (max-width: 1024px) {
            .app-header { display: flex; }
        }

        .tab-content { display: none; opacity: 0; transform: scale(0.98); transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
        .tab-content.active { display: block; opacity: 1; transform: scale(1); }

        .sync-clock {
            background: #000;
            color: #00ff00;
            font-family: 'JetBrains Mono', monospace;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 14px;
            box-shadow: inset 0 0 8px rgba(0,255,0,0.1);
        }

        /* Scrolling Ticker Animation */
        @keyframes ticker-scroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }
        .ticker-wrap {
            display: flex;
            width: fit-content;
            animation: ticker-scroll 25s linear infinite;
        }
        .ticker-wrap:hover {
            animation-play-state: paused;
        }
        .ticker-item {
            padding-right: 150px;
            display: flex;
            align-items: center;
            gap: 100px;
        }
        .ticker-glow {
            text-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
        }
            flex-shrink: 0;
            display: flex;
            align-items: center;
            gap: 64px;
        }

        #weather-alert-banner.hidden {
            display: none;
        }

        .alert-banner-critical {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            border-color: rgba(255, 255, 255, 0.2) !important;
        }

        .alert-banner-caution {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            border-color: rgba(255, 255, 255, 0.2) !important;
        }

        @keyframes alert-pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(220, 38, 38, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }

        .critical-pulse {
            animation: alert-pulse 2s infinite;
        }

        .user-location-glow-outer {
            fill: #06b6d4;
            animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
        }
        .user-location-glow-inner {
            fill: #22d3ee;
            animation: pulse-ring-inner 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
        }
        @keyframes pulse-ring {
            0% { r: 10; fill-opacity: 0.8; }
            100% { r: 40; fill-opacity: 0; }
        }
        @keyframes pulse-ring-inner {
            0% { r: 10; fill-opacity: 0.6; }
            100% { r: 30; fill-opacity: 0; }
        }
        .user-location-core {
            fill: #06b6d4;
            stroke: white;
            stroke-width: 3;
            filter: drop-shadow(0 0 12px rgba(6, 182, 212, 1));
        }
        .user-location-subcore {
            fill: white;
            fill-opacity: 0.9;
            pointer-events: none;
        }

        /* Ticker Animation */
        .ticker-wrap {
            width: 100%;
            overflow: hidden;
            background-color: transparent;
            box-sizing: content-box;
        }
        .ticker-item {
            display: inline-flex;
            white-space: nowrap;
            padding-right: 50%;
            animation: ticker 30s linear infinite;
        }
        @keyframes ticker {
            0% { transform: translate3d(0, 0, 0); }
            100% { transform: translate3d(-100%, 0, 0); }
        }
        .ticker-glow {
            text-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
        }

        .user-pulse-marker {
            background: none;
            border: none;
        }
        .user-pulse-marker .pulse {
            width: 20px;
            height: 20px;
            background: #06b6d4;
            border: 3px solid white;
            border-radius: 50%;
            position: relative;
            z-index: 10;
            box-shadow: 0 0 15px rgba(6, 182, 212, 0.8);
        }
        .user-pulse-marker .ring {
            position: absolute;
            top: -10px;
            left: -10px;
            width: 40px;
            height: 40px;
            border: 3px solid #06b6d4;
            border-radius: 50%;
            animation: sonar 1.5s infinite;
            opacity: 0;
        }
        @keyframes sonar {
            0% { transform: scale(0.5); opacity: 0; }
            50% { opacity: 0.5; }
            100% { transform: scale(3); opacity: 0; }
        }

        /* Specific Mobile Card Overrides */
        .mobile-hero {
            background: #0f172a;
            color: white;
            border-radius: 2.5rem;
            padding: 32px 24px;
            position: relative;
            overflow: hidden;
        }

        .mobile-action-btn {
            background: white;
            color: #0f172a;
            padding: 16px;
            border-radius: 1.25rem;
            font-weight: 800;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.1em;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            box-shadow: 0 10px 20px -5px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        .mobile-action-btn:active { transform: scale(0.96); }

        /* Status Pills */
        .status-pill {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Mobile Phone UI for Interchanges */
        .phone-frame {
            width: 100%;
            max-width: 340px;
            height: 600px;
            background: #1e293b;
            border: 12px solid #334155;
            border-radius: 48px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 50px 100px -20px rgba(0, 0, 0, 0.5);
            margin: 0 auto;
        }

        .phone-notch {
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 140px;
            height: 24px;
            background: #334155;
            border-bottom-left-radius: 16px;
            border-bottom-right-radius: 16px;
            z-index: 50;
        }

        .phone-screen {
            width: 100%;
            height: 100%;
            background: #f8fafc;
            border-radius: 36px;
            overflow-y: auto;
            position: relative;
        }

        .phone-screen::-webkit-scrollbar {
            width: 0px;
        }

        .phone-ui-header {
            background: white;
            padding: 40px 20px 20px 20px;
            border-bottom: 1px solid #e2e8f0;
        }

        .phone-ui-content {
            padding: 24px 20px;
        }
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Suppress Scrollbars */
        .scrollbar-none::-webkit-scrollbar { display: none; }
        .scrollbar-none { -ms-overflow-style: none; scrollbar-width: none; }

        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        .float-anim { animation: float 6s ease-in-out infinite; }

        /* Better Table for Mobile */
        .mobile-table-card {
            background: white;
            border-radius: 1.5rem;
            overflow: hidden;
            border: 1px solid #f1f5f9;
        }
        .mobile-table-row {
            padding: 18px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #f1f5f9;
        }
        .mobile-table-row:last-child { border-bottom: none; }

        #stations-grid .glass-card {
            background: white;
            border-radius: 1.5rem;
            border: 1px solid #f1f5f9;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }

        /* Digital Pass Styling */
        .digital-pass {
            background: white;
            border-radius: 2rem;
            overflow: hidden;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.1);
            position: relative;
        }
        .pass-cutout {
            position: absolute;
            top: 60%;
            width: 20px;
            height: 20px;
            background: var(--bg);
            border-radius: 50%;
            z-index: 10;
        }
        .pass-cutout-left { left: -10px; }
        .pass-cutout-right { right: -10px; }
        .pass-divider {
            position: absolute;
            top: 60%;
            left: 20px;
            right: 20px;
            border-top: 2px dashed #f1f5f9;
            margin-top: 10px;
        }

        .map-line-path {
            stroke-linecap: round;
            stroke-linejoin: round;
            fill: none;
            transition: all 0.3s ease;
        }
        .station-node {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .near-user {
            stroke-width: 6;
            stroke: #3b82f6;
            filter: drop-shadow(0 0 8px rgba(59, 130, 246, 0.5));
            animation: station-pulse 2s infinite;
        }
        @keyframes station-pulse {
            0% { r: 8; }
            50% { r: 12; }
            100% { r: 8; }
        }
        .station-node:hover {
            r: 10;
        }
        .station-label-custom {
            font-size: 10px;
            font-weight: 800;
            fill: #334155;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            pointer-events: none;
            transition: opacity 0.3s ease;
            text-shadow: 0 0 2px white;
        }
        .train-unit {
            cursor: pointer;
            transition: transform 0.2s linear;
        }
        .map-control-btn {
            width: 44px;
            height: 44px;
            background: white;
            border: 1px solid #f1f5f9;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748b;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        .map-control-btn:hover {
            color: #2563eb;
            background: #f8fafc;
            transform: translateY(-2px);
        }
        #metro-map-container {
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        #metro-map-canvas {
            width: 100%;
            height: 100%;
            cursor: grab;
            user-select: none;
            touch-action: none;
            background: radial-gradient(circle at center, #ffffff 0%, #f8fafc 100%);
        }
        #metro-map-canvas:active { cursor: grabbing; }

        /* Unified Map Overlay Logic */
        @media (max-width: 1024px) {
            #map-overlay {
                position: fixed !important;
                bottom: 0 !important;
                left: 0 !important;
                right: 0 !important;
                top: auto !important;
                width: 100% !important;
                height: 80vh !important;
                border-radius: 2.5rem 2.5rem 0 0 !important;
                transform: translateY(100%) !important;
                z-index: 5500 !important;
                border-left: none !important;
                border-top: 1px solid rgba(0,0,0,0.05) !important;
                box-shadow: 0 -20px 40px rgba(0,0,0,0.1) !important;
            }
            #map-overlay.active {
                transform: translateY(0) !important;
            }
        }
        @media (min-width: 1025px) {
            #map-overlay {
                transform: translateX(100%);
            }
            #map-overlay.active {
                transform: translateX(0);
            }
        }

        .user-pin-outer {
            animation: pulse-glow 2s infinite;
        }
        @keyframes pulse-glow {
            0% { r: 8; fill-opacity: 0.6; }
            100% { r: 24; fill-opacity: 0; }
        }
        .station-node {
            transition: r 0.2s, stroke-width 0.2s;
            cursor: pointer;
        }
        .station-node:hover {
            r: 10;
            stroke-width: 5;
        }

        .station-label-custom { 
            font-size: 11px;
            font-weight: 800;
            fill: #1e293b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            pointer-events: none;
            paint-order: stroke;
            stroke: #ffffff;
            stroke-width: 4px;
            transition: opacity 0.3s;
        }

        .map-line-path {
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            transition: stroke-width 0.2s;
        }

        .train-unit {
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
            transition: transform 0.1s linear;
        }

        .map-control-btn {
            width: 44px;
            height: 44px;
            background: white;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748b;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
            transition: all 0.2s;
        }
        .map-control-btn:active { transform: scale(0.9); }
        .map-control-btn i { width: 20px; height: 20px; }
        .station-tooltip {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            color: #334155;
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 4px 8px;
        }
        .leaflet-container {
            background: #f8fafc !important;
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
                <h1 class="text-xl font-black text-slate-900 tracking-tighter">HydMetro</h1>
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
        <div onclick="showTab('stations')" class="mobile-link" id="mob-stations"><i data-lucide="layout-grid"></i><span>Stations</span></div>
        <div onclick="showTab('tickets')" class="mobile-link mobile-link-cta" id="mob-tickets"><i data-lucide="qr-code"></i><span>Tickets</span></div>
        <div onclick="showTab('routes')" class="mobile-link" id="mob-routes"><i data-lucide="navigation"></i><span>Plan</span></div>
        <div onclick="showTab('map')" class="mobile-link" id="mob-map"><i data-lucide="map-pinned"></i><span>Live Map</span></div>
    </div>

    <div class="main" id="main-content">
        <div id="tab-home" class="tab-content active px-4 lg:px-10 pt-8 lg:pt-12">
            <!-- New: Weather Alert Banner -->
            <div id="weather-alert-banner" class="hidden max-w-[1500px] mx-auto mb-8 animate-in slide-in-from-top-4 duration-500">
                <div id="alert-body" class="rounded-[32px] p-6 flex flex-col lg:flex-row items-center gap-6 shadow-2xl border relative overflow-hidden">
                    <div class="absolute inset-0 bg-white/5 backdrop-blur-sm pointer-events-none"></div>
                    <div class="flex items-center gap-5 relative z-10">
                        <div id="alert-icon-wrap" class="w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg">
                            <i id="alert-icon" data-lucide="cloud-lightning" class="animate-bounce"></i>
                        </div>
                        <div>
                            <h4 id="alert-title" class="text-[11px] font-black uppercase tracking-[0.3em] mb-1">Weather Alert</h4>
                            <p id="alert-msg" class="text-sm font-bold leading-tight">Extreme conditions detected. Travel with caution.</p>
                        </div>
                    </div>
                    <div class="lg:ml-auto relative z-10 flex items-center gap-4">
                        <div class="px-5 py-2.5 bg-white/10 rounded-xl border border-white/10">
                            <span id="alert-advice" class="text-[10px] font-black uppercase tracking-widest text-white">Advice: Carry Umbrella</span>
                        </div>
                        <button onclick="dismissAlert()" class="p-3 hover:bg-white/10 rounded-xl transition-colors">
                            <i data-lucide="x" size="18" class="text-white/40"></i>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Dashboard Header -->
            <div class="flex flex-col lg:flex-row lg:items-end justify-between gap-6 mb-8 max-w-[1500px] mx-auto">
                <div class="space-y-1">
                    <h1 id="greeting" class="text-6xl lg:text-8xl font-black text-slate-900 tracking-tighter leading-none italic uppercase">Good Morning!</h1>
                    <p class="text-[11px] font-black text-slate-400 uppercase tracking-[0.4em] pl-1">Neural Processing Active. Enjoy your commute across the network.</p>
                </div>
                <div class="hidden lg:block">
                    <div class="bg-white border border-slate-100 rounded-[32px] p-6 pr-10 shadow-xl flex items-center gap-8 min-w-[320px]">
                        <div id="home-clock" class="text-5xl font-black text-slate-900 tabular-nums tracking-tighter">00:00:00</div>
                        <div class="flex flex-col items-start">
                            <span id="home-clock-ampm" class="text-sm font-black text-slate-400 uppercase">AM</span>
                            <span id="home-date" class="text-[9px] font-black text-blue-600 uppercase tracking-[0.2em] mt-1">MAY 8, 2026</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Dark Status Ticker -->
            <div class="max-w-[1500px] mx-auto mb-10 overflow-hidden">
                <div class="bg-[#0c1221] rounded-2xl p-4 lg:p-5 flex items-center border border-white/5 shadow-2xl relative overflow-hidden">
                    <div class="absolute left-0 top-0 bottom-0 w-1 bg-blue-600 z-10"></div>
                    <div class="ticker-wrap">
                        <div class="ticker-item font-black italic">
                            <div class="flex items-center gap-3 shrink-0">
                                <span class="text-[9px] font-black text-blue-400 uppercase tracking-widest whitespace-nowrap">SYS_MSG</span>
                                <p id="env-msg" class="text-[10px] font-bold text-slate-300 uppercase tracking-wider ticker-glow">Neural matrix stabilized. 38 trains in vector stream.</p>
                            </div>
                            <div class="flex items-center gap-3 shrink-0">
                                <span class="text-[9px] font-black text-blue-400 uppercase tracking-widest whitespace-nowrap">METRIC</span>
                                <p class="text-[10px] font-bold text-slate-300 uppercase tracking-wider ticker-glow">99.4% Network uptime detected via satellite.</p>
                            </div>
                            <div class="flex items-center gap-3 shrink-0">
                                <span class="text-[9px] font-black text-blue-400 uppercase tracking-widest whitespace-nowrap">WEATHER</span>
                                <p id="env-weather-msg" class="text-[10px] font-bold text-slate-300 uppercase tracking-wider ticker-glow">AC cooling optimized for low heat.</p>
                            </div>
                        </div>
                        <div class="ticker-item font-black italic">
                            <div class="flex items-center gap-3 shrink-0">
                                <span class="text-[9px] font-black text-blue-400 uppercase tracking-widest whitespace-nowrap">SYS_MSG</span>
                                <p id="env-msg-clone" class="text-[10px] font-bold text-slate-300 uppercase tracking-wider ticker-glow">Neural matrix stabilized. 38 trains in vector stream.</p>
                            </div>
                            <div class="flex items-center gap-3 shrink-0">
                                <span class="text-[9px] font-black text-blue-400 uppercase tracking-widest whitespace-nowrap">METRIC</span>
                                <p class="text-[10px] font-bold text-slate-300 uppercase tracking-wider ticker-glow">99.4% Network uptime detected via satellite.</p>
                            </div>
                            <div class="flex items-center gap-3 shrink-0">
                                <span class="text-[9px] font-black text-blue-400 uppercase tracking-widest whitespace-nowrap">WEATHER</span>
                                <p id="env-weather-msg-clone" class="text-[10px] font-bold text-slate-300 uppercase tracking-wider ticker-glow">AC cooling optimized for low heat.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Dash Grid -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 max-w-[1500px] mx-auto mb-12">
                <!-- Live Near Metro -->
                <div class="lg:col-span-3 bg-white border border-slate-100 rounded-[40px] p-8 shadow-xl relative overflow-hidden">
                    <div class="flex flex-col h-full justify-between">
                        <div class="flex items-center gap-6 mb-8">
                            <div class="w-16 h-16 bg-blue-600/5 rounded-[24px] flex items-center justify-center text-blue-600 border border-blue-100 group">
                                <i data-lucide="crosshair" size="24" class="group-hover:scale-110 transition-transform"></i>
                            </div>
                            <div class="space-y-1">
                                <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Live Near Metro</p>
                                <h3 id="near-name-dash" class="text-2xl font-black text-slate-900 tracking-tight leading-tight">--</h3>
                            </div>
                        </div>
                        <div class="flex items-end justify-between pt-6 border-t border-slate-50">
                            <div>
                                <p id="near-walk-dash" class="text-[11px] font-black text-emerald-500 uppercase tracking-widest mb-1">-- MIN WALK (--KM)</p>
                                <p id="near-dist-dash" class="text-[9px] font-bold text-slate-400 uppercase tracking-widest">-- KM AWAY</p>
                            </div>
                            <div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                        </div>
                    </div>
                </div>

                <!-- Atmosphere -->
                <div class="lg:col-span-3 bg-white border border-slate-100 rounded-[40px] p-8 shadow-xl relative overflow-hidden">
                     <div class="flex flex-col h-full justify-between">
                        <div class="flex items-center gap-6 mb-8">
                            <div class="w-16 h-16 bg-orange-600/5 rounded-[24px] flex items-center justify-center text-orange-500 border border-orange-100">
                                <i data-lucide="waves" size="24"></i>
                            </div>
                            <div class="space-y-1">
                                <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Atmosphere</p>
                                <h3 id="dash-weather-temp" class="text-2xl font-black text-slate-900 tracking-tight leading-tight">--°C, --</h3>
                            </div>
                        </div>
                        <p id="dash-weather-detail" class="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em]">Humidity: -- | Visibility: --</p>
                    </div>
                </div>

                <!-- GTFS Pulsar -->
                <div class="lg:col-span-6 bg-[#0c1221] rounded-[40px] p-8 shadow-2xl relative overflow-hidden group">
                    <div class="absolute -right-20 -top-20 w-80 h-80 bg-blue-600/5 rounded-full blur-3xl transition-all group-hover:bg-blue-600/10"></div>
                    <div class="flex items-center justify-between h-full">
                        <div class="flex items-center gap-8">
                            <div class="w-16 h-16 bg-emerald-500/10 text-emerald-500 rounded-[24px] flex items-center justify-center border border-emerald-500/20 shadow-inner">
                                <i data-lucide="activity" size="28" class="animate-pulse"></i>
                            </div>
                            <div>
                                <p class="text-[9px] font-black text-blue-400/60 uppercase tracking-[0.3em] mb-2">GTFS Network Pulse</p>
                                <h3 class="text-4xl font-black text-white tracking-tighter">Trips Active: <span id="home-train-count" class="text-emerald-400 tabular-nums">--</span></h3>
                            </div>
                        </div>
                        <div class="hidden lg:block">
                            <div id="load-pill-dash" class="px-6 py-2.5 bg-slate-800/80 rounded-full border border-white/5 flex items-center gap-3">
                                <div class="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_10px_#f59e0b]"></div>
                                <span id="load-label-dash" class="text-[10px] font-black text-slate-300 uppercase tracking-widest">Analyzing Load...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Arrival Board Section -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 max-w-[1500px] mx-auto pb-24">
                <div class="lg:col-span-8">
                    <div class="bg-white border border-slate-100 rounded-[48px] shadow-2xl overflow-hidden">
                        <div class="p-10 lg:p-12 border-b border-slate-50 flex items-center justify-between">
                            <div class="flex items-center gap-6">
                                <div class="w-12 h-12 bg-slate-900 rounded-2xl flex items-center justify-center text-white"><i data-lucide="radio-tower" size="20"></i></div>
                                <div>
                                    <h4 class="text-[11px] font-black text-slate-400 uppercase tracking-[0.4em] mb-1">Arrival Board</h4>
                                    <div class="flex items-center gap-3">
                                        <h3 id="board-station-name" class="text-2xl font-black text-slate-900 tracking-tight uppercase italic">--</h3>
                                    </div>
                                </div>
                            </div>
                            <!-- Station Switcher -->
                            <div class="relative group">
                                <select id="board-station-selector" onchange="updateBoardData(null, null, this.value)" class="pl-12 pr-10 py-4 bg-slate-50 border border-slate-100 rounded-2xl text-[10px] font-black uppercase outline-none focus:ring-4 focus:ring-blue-500/10 appearance-none cursor-pointer text-slate-600 transition-all">
                                    <option value="">Switch Station</option>
                                    {% for st in ALL_STATIONS %}
                                    <option value="{{ st.id }}">{{ st.name }}</option>
                                    {% endfor %}
                                </select>
                                <div class="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-red-500 border-2 border-white shadow-sm"></div>
                                <div class="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"><i data-lucide="chevron-down" size="14"></i></div>
                            </div>
                        </div>

                        <div class="p-4 lg:p-0">
                            <!-- Table Headers -->
                            <div class="hidden lg:grid grid-cols-4 px-12 py-8 bg-slate-50/50">
                                <span class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Vector</span>
                                <span class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Node</span>
                                <span class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Boarding</span>
                                <span class="text-[10px] font-black text-slate-400 uppercase tracking-widest text-right">Countdown</span>
                            </div>

                            <div id="arrival-board-list" class="divide-y divide-slate-50">
                                <!-- Board Rows Injected -->
                                <div class="p-20 text-center flex flex-col items-center">
                                    <div class="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
                                    <p class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Synchronizing Pulse...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Book Ticket Card -->
                <div class="lg:col-span-4 h-full">
                    <div class="bg-[#0c1221] rounded-[48px] p-10 lg:p-14 h-full relative overflow-hidden group border border-white/5 shadow-2xl flex flex-col justify-between">
                         <div class="absolute -left-10 -bottom-10 w-60 h-60 bg-blue-600/10 rounded-full blur-3xl"></div>
                         <div>
                            <div class="w-16 h-16 bg-white/5 rounded-[24px] flex items-center justify-center text-blue-400 mb-10 border border-white/10">
                                <i data-lucide="map-pinned" size="28"></i>
                            </div>
                            <h3 class="text-4xl lg:text-5xl font-black text-white tracking-tighter mb-6 leading-[1.1]">Book Metro Ticket</h3>
                            <p class="text-sm font-bold text-slate-400 leading-relaxed mb-12 uppercase tracking-wide opacity-80">Select your destination and secure a digital token instantly.</p>
                         </div>
                         <button onclick="showTab('routes')" class="w-full py-6 bg-white text-slate-900 rounded-[28px] font-black text-[12px] uppercase tracking-[0.2em] shadow-2xl hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-4 group">
                            <i data-lucide="navigation-2" size="20" class="group-hover:translate-x-1 transition-transform"></i>
                            Open Planner
                         </button>
                    </div>
                </div>
            </div>
        </div>
        <!-- NETWORK MAP -->
        <div id="tab-map" class="tab-content h-full">
            <div id="interchange-modal" class="hidden fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-300">
                <div class="phone-frame scale-90 sm:scale-100">
                    <div class="phone-notch"></div>
                    <div class="phone-screen">
                        <div class="phone-ui-header">
                            <div class="flex justify-between items-center mb-6">
                                <button onclick="closeInterchangeModal()" class="p-2 bg-slate-100 rounded-full hover:bg-slate-200 transition-colors">
                                    <i data-lucide="chevron-left" class="text-slate-600" size="18"></i>
                                </button>
                                <span class="text-[10px] font-black text-slate-400 uppercase tracking-widest">Station Guide</span>
                                <i data-lucide="more-horizontal" class="text-slate-300" size="18"></i>
                            </div>
                            <h3 id="modal-title" class="text-2xl font-black text-slate-900 tracking-tight leading-tight mb-2">Interchange</h3>
                            <div id="line-badges" class="flex gap-2"></div>
                        </div>
                        <div class="phone-ui-content">
                            <div class="bg-blue-600/5 border border-blue-100 p-5 rounded-3xl mb-8">
                                <div class="flex items-center gap-3 mb-4">
                                    <div class="p-2 bg-blue-600 text-white rounded-xl"><i data-lucide="shuffle" size="14"></i></div>
                                    <span class="text-[10px] font-black text-blue-600 uppercase tracking-widest">Transfer Vectors</span>
                                </div>
                                <div id="transfer-guidance" class="space-y-6 placeholder-slate-200">
                                    <div class="h-4 w-3/4 bg-slate-100 rounded animate-pulse"></div>
                                    <div class="h-4 w-1/2 bg-slate-100 rounded animate-pulse"></div>
                                </div>
                            </div>

                            <div class="space-y-4">
                                <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest pl-1">Live Load Projection</p>
                                <div class="bg-white border border-slate-100 p-5 rounded-3xl flex items-center justify-between">
                                    <div class="flex items-center gap-4">
                                        <div class="w-10 h-10 bg-emerald-50 text-emerald-500 rounded-xl flex items-center justify-center"><i data-lucide="users" size="18"></i></div>
                                        <div>
                                            <p class="text-xs font-black text-slate-900">Normal Flow</p>
                                            <p class="text-[8px] font-bold text-slate-400 uppercase">Current Density</p>
                                        </div>
                                    </div>
                                    <span class="text-[14px] font-black text-emerald-500">22%</span>
                                </div>
                            </div>

                            <button onclick="closeInterchangeModal()" class="w-full mt-10 py-5 bg-slate-900 text-white rounded-2xl font-black text-[11px] uppercase tracking-widest shadow-xl">
                                Return to Map
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div>
                <h2 class="text-3xl font-black text-slate-900 tracking-tight">Hyderabad Metro Network</h2>
                <p class="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mt-1">Real-time updates active</p>
            </div>
            <div class="flex flex-wrap items-center gap-4">
                <div class="relative group w-full lg:w-96" id="map-search-container">
                    <div class="absolute inset-y-0 left-4 flex items-center pointer-events-none text-slate-400 group-focus-within:text-blue-600 transition-colors">
                        <i data-lucide="search" size="18"></i>
                    </div>
                    <input type="text" id="map-search" placeholder="Search station (e.g. Miyapur)..." 
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
            <!-- Network Status Bar -->
            <div class="absolute top-6 left-6 flex items-center gap-3 z-50">
                <div class="px-4 py-2 bg-slate-900 border border-white/10 rounded-2xl shadow-2xl flex items-center gap-3 animate-in slide-in-from-left duration-700">
                    <span class="relative flex h-2 w-2">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    </span>
                    <span class="text-[9px] font-black text-white uppercase tracking-[0.2em] whitespace-nowrap">Live Sync: Active</span>
                </div>
                <div class="px-4 py-2 bg-white/90 backdrop-blur-md rounded-2xl shadow-xl flex items-center gap-2 border border-slate-100">
                    <i data-lucide="clock" size="12" class="text-blue-500"></i>
                    <span id="map-clock" class="text-[9px] font-black text-slate-800 uppercase tracking-widest tabular-nums">00:00:00</span>
                </div>
            </div>

            <div class="glass-card p-0 relative h-[600px] lg:h-[800px] overflow-hidden bg-slate-50 border-none shadow-2xl">
                <div id="leaflet-map" class="absolute inset-0 z-0"></div>

                <div id="map-legend" class="absolute bottom-24 left-6 glass-card p-6 border-none shadow-2xl bg-white/95 backdrop-blur-xl hidden lg:block z-[60]">
                    <h5 class="text-[10px] font-black uppercase tracking-[0.2em] mb-4 text-slate-400">Map Legend</h5>
                    <div class="space-y-3">
                        <div class="flex items-center gap-3">
                            <div class="w-4 h-1 bg-red-500 rounded-full"></div>
                            <span class="text-[9px] font-bold text-slate-700 uppercase">Red Line (Miyapur ↔ LB Nagar)</span>
                        </div>
                        <div class="flex items-center gap-3">
                            <div class="w-4 h-1 bg-blue-500 rounded-full"></div>
                            <span class="text-[9px] font-bold text-slate-700 uppercase">Blue Line (Raidurg ↔ Nagole)</span>
                        </div>
                        <div class="flex items-center gap-3">
                            <div class="w-4 h-1 bg-green-500 rounded-full"></div>
                            <span class="text-[9px] font-bold text-slate-700 uppercase">Green Line (JBS ↔ MGBS)</span>
                        </div>
                    </div>
                </div>

                <div class="absolute top-6 right-6 flex flex-col gap-3 z-[60]">
                    <button onclick="zoomInMap()" class="map-control-btn" title="Zoom In"><i data-lucide="zoom-in"></i></button>
                    <button onclick="zoomOutMap()" class="map-control-btn" title="Zoom Out"><i data-lucide="zoom-out"></i></button>
                    <button onclick="resetMap()" class="map-control-btn" title="Reset View"><i data-lucide="maximize"></i></button>
                    <button onclick="locateMeOnMap()" class="map-control-btn" title="Find My Location"><i data-lucide="locate"></i></button>
                </div>

                <div id="metro-map-container" class="absolute inset-0 w-full h-full overflow-hidden pointer-events-none z-10">
                    <svg id="metro-map-canvas" viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid slice" class="hidden">
                        <g id="map-viewport">
                            <g id="map-lines"></g>
                            <g id="map-stations"></g>
                            <g id="map-labels"></g>
                            <g id="map-trains"></g>
                            <g id="map-user"></g>
                        </g>
                    </svg>
                </div>

                <div id="map-overlay" class="absolute top-0 right-0 h-full w-full lg:w-[400px] z-[2000] transition-transform duration-500 ease-in-out bg-white shadow-[-20px_0_50px_-10px_rgba(0,0,0,0.1)] border-l border-slate-100 lg:rounded-none">
                    <div class="h-full flex flex-col p-6 lg:p-10 overflow-hidden relative">
                        <!-- Mobile Sheet Handle -->
                        <div class="lg:hidden w-12 h-1 bg-slate-200 rounded-full mx-auto mb-8 shrink-0"></div>

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
                            <!-- About this Area Section -->
                            <div id="ov-desc-container" class="hidden">
                                <h5 class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <i data-lucide="info" size="12"></i> About this Area
                                </h5>
                                <p id="ov-description" class="text-[11px] font-medium text-slate-600 leading-relaxed bg-slate-50 p-5 rounded-2xl border border-slate-100 italic"></p>
                            </div>

                            <!-- Amenities Section -->
                            <div>
                                <h5 class="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                                    <i data-lucide="layout-grid" size="12"></i> Station Facilities
                                </h5>
                                <div id="ov-amenities" class="grid grid-cols-2 gap-3"></div>
                            </div>

                            <!-- Real-time Departures Section -->
                            <div>
                                <h5 class="text-[10px] font-black text-blue-600 uppercase tracking-widest mb-6 flex items-center gap-2">
                                    <i data-lucide="radio" size="12" class="animate-pulse"></i> Live Departures
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
                <div class="text-center lg:text-left mb-8 px-4">
                   <h2 class="text-3xl lg:text-5xl font-black tracking-tight mb-1 text-slate-900 italic">Plan Your Trip</h2>
                   <div class="flex items-center justify-center lg:justify-start gap-2">
                       <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                       <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Optimized Metro Route</p>
                   </div>
                </div>

                <div id="planner-input-area" class="glass-card border-none bg-white p-6 lg:p-12 relative overflow-hidden mb-8 !rounded-[2.5rem]">
                    <div class="grid grid-cols-1 gap-6 relative z-10">
                        <div class="space-y-3">
                            <label class="text-[10px] font-black text-slate-400 uppercase block tracking-widest pl-2">Boarding From</label>
                            <div class="relative">
                                <div class="absolute left-6 top-1/2 -translate-y-1/2 w-4 h-4 rounded-full border-4 border-blue-500 bg-white"></div>
                                <select id="start-st" class="w-full pl-16 pr-8 py-6 bg-slate-50 border border-slate-100 rounded-[28px] outline-none focus:ring-4 focus:ring-blue-500/10 font-bold appearance-none text-slate-900 text-lg"></select>
                                <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none"><i data-lucide="chevron-down" size="20"></i></div>
                            </div>
                        </div>

                        <div class="flex items-center gap-4 px-2">
                            <div class="flex-1 h-px bg-slate-100"></div>
                            <button onclick="swapPlannerNodes()" class="p-4 bg-white border border-slate-100 rounded-2xl shadow-sm text-slate-400 hover:text-slate-900 transition-all active:scale-90"><i data-lucide="arrow-up-down" size="20"></i></button>
                            <div class="flex-1 h-px bg-slate-100"></div>
                        </div>

                        <div class="space-y-3">
                            <label class="text-[10px] font-black text-slate-400 uppercase block tracking-widest pl-2">Destination Station</label>
                            <div class="relative">
                                <div class="absolute left-6 top-1/2 -translate-y-1/2 text-emerald-500"><i data-lucide="map-pin" size="20"></i></div>
                                <select id="end-st" class="w-full pl-16 pr-8 py-6 bg-slate-50 border border-slate-100 rounded-[28px] outline-none focus:ring-4 focus:ring-emerald-500/10 font-bold appearance-none text-slate-900 text-lg"></select>
                                <div class="absolute right-8 top-1/2 -translate-y-1/2 text-slate-300 pointer-events-none"><i data-lucide="chevron-down" size="20"></i></div>
                            </div>
                        </div>
                    </div>

                    <button id="plan-btn" onclick="planJourney()" class="w-full py-6 mt-8 bg-slate-900 text-white font-black rounded-[30px] shadow-xl text-[12px] uppercase tracking-widest flex items-center justify-center gap-3 active:scale-[0.98] transition-all">
                        <span id="btn-text">Find Best Routes</span>
                        <div id="btn-loader" class="hidden w-5 h-5 border-[3px] border-white border-t-transparent rounded-full animate-spin"></div>
                        <i data-lucide="sparkles" size="18" class="text-blue-400"></i>
                    </button>
                </div>

                <div id="route-output" class="hidden space-y-6 pb-12">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-3">
                            <button onclick="saveCurrentVector()" id="save-route-btn" class="flex items-center gap-2 px-6 py-3 bg-slate-900 text-white rounded-2xl text-[10px] font-black uppercase tracking-widest shadow-xl hover:bg-black transition-all">
                                <i data-lucide="star" size="14"></i> Save Route
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
                                <h5 class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Route Advice</h5>
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
                                        <p class="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mt-5">Travel Time</p>
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
                             <div class="flex items-center gap-2"><i data-lucide="zap" class="text-green-500" size="14"></i> <span class="text-[10px] font-black text-slate-500 uppercase tracking-widest">Live Sync</span></div>
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
                                 <p class="text-xs font-bold text-slate-300 italic">Finding best routes...</p>
                             </div>
                        </div>
                    </div>

                    <!-- Ticket & Payment Center -->
                    <div class="glass-card p-10 border-none bg-slate-900 text-white relative overflow-hidden">
                        <div class="absolute -right-10 -top-10 w-40 h-40 bg-blue-500/10 rounded-full blur-3xl"></div>
                        <div class="flex items-center gap-4 mb-6 relative z-10">
                            <div class="p-2 bg-white/10 rounded-xl"><i data-lucide="ticket" size="16"></i></div>
                            <h5 class="text-[10px] font-black uppercase tracking-widest">Your Ticket</h5>
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
                        <i data-lucide="search" size="48" class="mb-4 opacity-20"></i><p class="font-bold text-slate-400 uppercase text-[10px] tracking-widest">Select stations to plan your trip.</p>
                    </div>
            </div>
        </div>

        <div id="tab-tickets" class="tab-content">
            <div class="max-w-lg mx-auto">
                <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-8 gap-6">
                   <div class="text-center lg:text-left">
                      <h2 class="text-3xl font-black tracking-tight mb-1 text-slate-900 italic uppercase">My Tickets</h2>
                      <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest">Metro Trip History</p>
                   </div>
                </div>

                <div class="mb-10">
                    <h4 class="text-[9px] font-black uppercase tracking-[0.3em] text-slate-400 mb-6 pl-2">Active Ticket</h4>
                    <div id="active-ticket-container" class="space-y-6">
                        <!-- Active ticket injected here -->
                    </div>
                </div>

                <div>
                    <h4 class="text-[9px] font-black uppercase tracking-[0.3em] text-slate-400 mb-6 pl-2">Past Trips</h4>
                    <div id="trip-history" class="space-y-4">
                        <!-- History injected here -->
                    </div>
                    <div id="history-empty" class="py-16 text-center flex flex-col items-center justify-center text-slate-300 bg-white rounded-[2rem] border border-slate-100">
                        <i data-lucide="folder-clock" size="32" class="mb-4 opacity-10 text-slate-400"></i>
                        <p class="text-[9px] font-black uppercase tracking-widest">Nothing here yet</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- STATIONS DIRECTORY -->
        <div id="tab-stations" class="tab-content">
            <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-8 gap-6">
                <div class="text-center lg:text-left">
                   <h2 class="text-4xl font-black tracking-tight mb-1 text-slate-900 italic uppercase">Metro Directory</h2>
                   <p id="stations-dir-subtitle" class="text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]">Explore all stations and facilities</p>
                </div>
                <div class="flex flex-col sm:flex-row items-center gap-4">
                    <div class="relative w-full sm:w-64">
                         <i data-lucide="search" class="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size="14"></i>
                         <input type="text" id="dir-search" oninput="renderStationsDirectory()" placeholder="Search stations..." class="w-full pl-12 pr-4 py-3 bg-slate-100 border-none rounded-2xl outline-none focus:ring-2 focus:ring-blue-500/20 font-bold text-sm">
                    </div>
                    <div class="flex shrink-0 gap-1 bg-slate-100 p-1.5 rounded-[24px]">
                        <button id="dir-filter-All" onclick="filterDirByLine('All')" class="dir-line-btn active px-6 py-2.5 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all">All</button>
                        <button id="dir-filter-Red" onclick="filterDirByLine('Red')" class="dir-line-btn px-6 py-2.5 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all hover:bg-white text-red-500">Red</button>
                        <button id="dir-filter-Blue" onclick="filterDirByLine('Blue')" class="dir-line-btn px-6 py-2.5 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all hover:bg-white text-blue-500">Blue</button>
                        <button id="dir-filter-Green" onclick="filterDirByLine('Green')" class="dir-line-btn px-6 py-2.5 rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all hover:bg-white text-green-500">Green</button>
                    </div>
                </div>
            </div>

            <div id="stations-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                <!-- Stations cards injected here -->
            </div>
        </div>

        <div id="tab-feedback" class="tab-content">
            <div class="flex flex-col lg:flex-row lg:items-center justify-between mb-4 gap-6 border-b pb-4 border-slate-200">
                <div class="flex flex-col items-center lg:flex-row lg:items-center gap-6 text-center lg:text-left">
                    <div class="w-16 h-16 bg-blue-600 rounded-3xl flex items-center justify-center text-white shadow-xl shadow-blue-500/30"><i data-lucide="message-square" size="32"></i></div>
                    <div>
                       <h2 class="text-4xl font-black tracking-tight mb-1 text-slate-900">Help & Feedback</h2>
                       <p class="text-xs font-bold text-slate-400 uppercase tracking-widest">Your feedback helps us improve the network</p>
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div class="lg:col-span-5">
                    <div class="glass-card border-none shadow-2xl bg-white p-10 relative overflow-hidden group">
                        <div class="absolute -right-20 -top-20 w-80 h-80 bg-blue-50 rounded-full blur-3xl transition-all group-hover:bg-blue-100/50"></div>

                        <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-900 mb-10 relative z-10">Feedback Form</h4>

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
                                <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Feedback Category</label>
                                <select id="feedback-cat" class="w-full p-5 bg-slate-50 border-2 border-transparent rounded-[24px] outline-none focus:border-blue-500/20 focus:bg-white font-bold text-sm appearance-none cursor-pointer">
                                    <option>General Experience</option>
                                    <option>Network Accuracy</option>
                                    <option>Train Timing</option>
                                    <option>App Features</option>
                                </select>
                            </div>

                            <div class="space-y-3">
                                <label class="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Your Message</label>
                                <textarea id="feedback-msg" rows="4" class="w-full p-6 bg-slate-50 border-2 border-transparent rounded-[28px] outline-none focus:border-blue-500/20 focus:bg-white font-bold text-sm resize-none" placeholder="Describe your experience with the metro..."></textarea>
                            </div>

                            <button onclick="submitFeedback()" class="w-full py-6 bg-slate-900 text-white font-black rounded-[30px] text-[11px] uppercase tracking-[0.3em] shadow-2xl hover:bg-black transition-all flex items-center justify-center gap-3">
                                <i data-lucide="send" size="14"></i> Send Message
                            </button>
                        </div>
                    </div>
                </div>
                <div class="lg:col-span-7">
                    <div class="glass-card border-none bg-slate-50/50 p-10 min-h-[600px] rounded-[40px]">
                        <h4 class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 mb-10">Recent Feedback</h4>
                        <div id="feedback-history" class="space-y-6">
                            <!-- Injected by JS -->
                        </div>
                        <div id="feedback-empty" class="py-32 text-center flex flex-col items-center justify-center text-slate-300">
                            <div class="p-6 bg-white rounded-full mb-6 border border-slate-100"><i data-lucide="archive" size="32" class="opacity-10"></i></div>
                            <p class="text-[10px] font-black uppercase tracking-widest">No history yet</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const stations = {{ ALL_STATIONS | tojson }};
        const connections = {{ CONNECTIONS | tojson }};
        const landmarksData = {{ LANDMARKS | tojson }};
        let trainStates = new Map(); 
        let syncSuccessOnce = false;
        window.shouldCenterOnNextFix = true;

        let leafletMap = null;
        let stationMarkers = new Map();
        let trainMarkers = new Map();
        let linePolylines = [];
        let userLeafletMarker = null;

        function initCustomMetroMap() {
            if (leafletMap) return;

            // Organic Maps style (OpenStreetMap)
            leafletMap = L.map('leaflet-map', {
                center: [17.4334, 78.4484], // Ameerpet
                zoom: 13,
                zoomControl: false,
                attributionControl: false
            });

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19
            }).addTo(leafletMap);

            // Draw Lines
            Object.entries(connections).forEach(([line, sidList]) => {
                const color = line === 'Red' ? '#ef4444' : line === 'Blue' ? '#3b82f6' : '#22c55e';
                const latlngs = sidList.map(sid => {
                    const s = stations.find(st => st.id === sid);
                    return s ? [s.lat, s.lng] : null;
                }).filter(p => p !== null);

                const poly = L.polyline(latlngs, {
                    color: color,
                    weight: 6,
                    opacity: 0.8,
                    lineJoin: 'round'
                }).addTo(leafletMap);
                linePolylines.push(poly);
            });

            // Draw Stations
            stations.forEach(s => {
                const color = s.line === 'Red' ? '#ef4444' : s.line === 'Blue' ? '#3b82f6' : '#22c55e';
                const isInter = ['Ameerpet', 'MG Bus Station', 'Parade Ground'].includes(s.name);

                const icon = L.divIcon({
                    className: 'custom-station-icon',
                    html: `<div class="w-4 h-4 rounded-full border-2 border-white shadow-lg ${isInter ? 'bg-slate-900 border-4 w-6 h-6 -ml-3 -mt-3' : ''}" style="background-color: ${color}"></div>`,
                    iconSize: [16, 16],
                    iconAnchor: [8, 8]
                });

                const m = L.marker([s.lat, s.lng], { icon: icon }).addTo(leafletMap);
                m.on('click', () => handleStationInteraction(s));

                if (leafletMap.getZoom() > 14 || isInter) {
                    m.bindTooltip(s.name, { 
                        permanent: true, 
                        direction: 'right', 
                        className: 'station-tooltip',
                        offset: [10, 0]
                    });
                }

                stationMarkers.set(s.id, m);
            });

            leafletMap.on('zoomend', () => {
                const zoom = leafletMap.getZoom();
                stationMarkers.forEach((m, sid) => {
                    const s = stations.find(st => st.id === sid);
                    const isInter = ['Ameerpet', 'MG Bus Station', 'Parade Ground'].includes(s.name);
                    if (zoom < 14 && !isInter) {
                        m.unbindTooltip();
                    } else if (!m.getTooltip()) {
                        m.bindTooltip(s.name, { permanent: true, direction: 'right', className: 'station-tooltip', offset: [10, 0] });
                    }
                });
            });

            updateTrainPositions();
        }

        function zoomInMap() { if (leafletMap) leafletMap.zoomIn(); }
        function zoomOutMap() { if (leafletMap) leafletMap.zoomOut(); }
        function resetMap() { 
            if (leafletMap) leafletMap.setView([17.4334, 78.4484], 13); 
            closeOverlay();
        }

        let leafletUserMarker = null;
        let routeHighlightLayer = null;

        function locateMeOnMap() {
            if (lastUserLoc && leafletMap) {
                // Check if within Telangana bounds
                const isWithinTelangana = (
                    lastUserLoc.lat >= 15.8 && lastUserLoc.lat <= 19.9 &&
                    lastUserLoc.lng >= 77.2 && lastUserLoc.lng <= 81.3
                );

                if (isWithinTelangana) {
                    leafletMap.setView([lastUserLoc.lat, lastUserLoc.lng], 15);
                } else {
                    console.log("User outside Telangana. Centering on Ameerpet instead.");
                    leafletMap.setView([17.4334, 78.4484], 13);
                }

                // Update or create user marker
                const userIcon = L.divIcon({
                    className: 'user-pulse-marker',
                    html: '<div class="ring"></div><div class="pulse"></div>',
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                });

                if (!leafletUserMarker) {
                    leafletUserMarker = L.marker([lastUserLoc.lat, lastUserLoc.lng], { icon: userIcon }).addTo(leafletMap);
                } else {
                    leafletUserMarker.setLatLng([lastUserLoc.lat, lastUserLoc.lng]);
                }

                updateSVGUserPin(lastUserLoc.lat, lastUserLoc.lng);
            } else {
                manualRefreshGeo();
            }
        }

        function highlightRouteOnMap(sequence) {
            if (!leafletMap || !sequence || sequence.length < 2) return;

            // Remove existing highlight
            if (routeHighlightLayer) {
                leafletMap.removeLayer(routeHighlightLayer);
            }

            const latlngs = sequence.map(s => [s.lat, s.lng]);

            routeHighlightLayer = L.polyline(latlngs, {
                color: '#2563eb',
                weight: 12,
                opacity: 0.7,
                lineCap: 'round',
                dashArray: '10, 15',
                className: 'route-highlight-p'
            }).addTo(leafletMap);

            // Add CSS for flow animation if not already there
            if (!document.getElementById('route-anim-style')) {
                const style = document.createElement('style');
                style.id = 'route-anim-style';
                style.innerHTML = `
                    .route-highlight-p {
                        animation: route-flow 2s linear infinite;
                    }
                    @keyframes route-flow {
                        from { stroke-dashoffset: 25; }
                        to { stroke-dashoffset: 0; }
                    }
                `;
                document.head.appendChild(style);
            }

            // Pan to the start of the route
            leafletMap.panTo([sequence[0].lat, sequence[0].lng]);

            // Highlight stations specifically on the board or map?
            // User requested visually highlight on the network map.
            // We'll zoom to fit the route.
            leafletMap.fitBounds(routeHighlightLayer.getBounds(), { padding: [50, 50] });
        }

        function updateTrainPositions() {
            if (tabState !== 'map' || !leafletMap) {
                requestAnimationFrame(updateTrainPositions);
                return;
            }

            const now = Date.now();

            trainStates.forEach((t, tid) => {
                const s1 = stations.find(s => s.id === t.from_id);
                const s2 = stations.find(s => s.id === t.to_id);
                if(!s1 || !s2) return;

                let progress = (now / 1000 - t.t1_epoch) / t.duration;
                progress = Math.max(0, Math.min(1.0, progress));

                const lat = s1.lat + (s2.lat - s1.lat) * progress;
                const lng = s1.lng + (s2.lng - s1.lng) * progress;

                let marker = trainMarkers.get(tid);
                const color = t.line === 'Red' ? '#ef4444' : t.line === 'Blue' ? '#3b82f6' : '#22c55e';

                if (!marker) {
                    const icon = L.divIcon({
                        className: 'train-marker-icon',
                        html: `
                            <div class="train-unit-leaflet" style="transform: rotate(${0}deg)">
                                <div class="w-8 h-4 bg-slate-900 border border-white rounded-md flex items-center justify-end px-1 shadow-xl">
                                    <div class="w-1.5 h-1.5 rounded-full" style="background-color: ${color}"></div>
                                </div>
                            </div>
                        `,
                        iconSize: [32, 16],
                        iconAnchor: [16, 8]
                    });
                    marker = L.marker([lat, lng], { icon: icon }).addTo(leafletMap);
                    marker.on('click', () => handleTrainInteraction(t, s2));
                    trainMarkers.set(tid, marker);
                } else {
                    marker.setLatLng([lat, lng]);
                    const angle = Math.atan2(s2.lat - s1.lat, s2.lng - s1.lng) * 180 / Math.PI;
                    // Note: Simplified rotation, Leaflet marker rotation usually needs a plugin or custom div icon update
                }
            });

            // Cleanup vanished trains
            trainMarkers.forEach((m, tid) => {
                if (!trainStates.has(tid)) {
                    leafletMap.removeLayer(m);
                    trainMarkers.delete(tid);
                }
            });

            requestAnimationFrame(updateTrainPositions);
        }

        function updateSVGUserPin(lat, lng) {
            if (!leafletMap) return;

            if (!userLeafletMarker) {
                const icon = L.divIcon({
                    className: 'user-pin-icon',
                    html: `
                        <div class="relative w-8 h-8 flex items-center justify-center">
                            <div class="absolute inset-0 bg-blue-500/20 rounded-full animate-ping"></div>
                            <div class="w-4 h-4 bg-blue-600 border-2 border-white rounded-full shadow-lg relative z-10"></div>
                        </div>
                    `,
                    iconSize: [32, 32],
                    iconAnchor: [16, 16]
                });
                userLeafletMarker = L.marker([lat, lng], { icon: icon }).addTo(leafletMap);
            } else {
                userLeafletMarker.setLatLng([lat, lng]);
            }
        }


        function toggleOverlaySide() {
            const overlay = document.getElementById('map-overlay');
            overlay.classList.toggle('left-side');
        }

        function closeOverlay() {
            const overlay = document.getElementById('map-overlay');
            overlay.classList.remove('active');
        }

        async function handleTrainInteraction(t, nextStop) {
            const overlay = document.getElementById('map-overlay');
            overlay.classList.add('active');

            document.getElementById('ov-name').innerText = `Train ${t.trip_id}`;
            document.getElementById('ov-weather').classList.add('hidden');

            const ovLine = document.getElementById('ov-line');
            ovLine.innerText = t.line + ' LINE';
            ovLine.className = 'px-3 py-1 text-[10px] font-black uppercase rounded-lg shadow-sm ' + (t.line === 'Red' ? 'bg-red-50 text-red-600' : t.line === 'Blue' ? 'bg-blue-50 text-blue-600' : 'bg-green-50 text-green-600');

            const oldBtn = document.getElementById('ov-inter-btn');
            if (oldBtn) oldBtn.remove();

            const am = document.getElementById('ov-amenities'); 
            am.innerHTML = `
                <div class="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm flex items-start gap-4 col-span-2">
                    <div class="p-3 bg-blue-50 text-blue-600 rounded-xl"><i data-lucide="navigation"></i></div>
                    <div>
                        <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1">Status</p>
                        <p class="text-xs font-bold text-slate-800">Heading to ${t.final_stop}</p>
                    </div>
                </div>
                <div class="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm flex items-start gap-4">
                    <div class="p-3 bg-emerald-50 text-emerald-600 rounded-xl"><i data-lucide="gauge"></i></div>
                    <div>
                        <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1">Speed</p>
                        <p class="text-xs font-bold text-slate-800">${t.speed} KM/H</p>
                    </div>
                </div>
                <div class="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm flex items-start gap-4">
                    <div class="p-3 bg-amber-50 text-amber-600 rounded-xl"><i data-lucide="map-pin"></i></div>
                    <div>
                        <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1">Next Stop</p>
                        <p class="text-xs font-bold text-slate-800">${nextStop.name}</p>
                    </div>
                </div>
            `;

            const trainsEl = document.getElementById('ov-trains');
            trainsEl.innerHTML = '<p class="text-[10px] font-bold text-slate-400 italic">Tracking current unit metrics...</p>';

            lucide.createIcons();

            // Re-center on train position (estimated)
            const s1 = stations.find(s => s.id === t.from_id);
            const s2 = stations.find(s => s.id === t.to_id);
            if (s1 && s2) {
                const now = Date.now();
                let progress = (now / 1000 - t.t1_epoch) / t.duration;
                progress = Math.max(0, Math.min(1, progress));
                const lat = s1.lat + (s2.lat - s1.lat) * progress;
                const lng = s1.lng + (s2.lng - s1.lng) * progress;
                centerMapOn(lat, lng);
            }
        }

        function centerMapOn(lat, lng) {
            const pt = project(lat, lng);
            mapViewport.scale = 2.5; 
            mapViewport.x = 500 - (pt.x * mapViewport.scale);
            mapViewport.y = 500 - (pt.y * mapViewport.scale);
            applyMapTransform();
        }

        async function handleStationInteraction(s) {
            const overlay = document.getElementById('map-overlay');
            overlay.classList.add('active');

            document.getElementById('ov-name').innerText = s.name;
            document.getElementById('ov-weather').classList.add('hidden');

            const ovLine = document.getElementById('ov-line');
            ovLine.innerText = s.line + ' LINE';
            ovLine.className = 'px-3 py-1 text-[10px] font-black uppercase rounded-lg shadow-sm ' + (s.line === 'Red' ? 'bg-red-50 text-red-600' : s.line === 'Blue' ? 'bg-blue-50 text-blue-600' : 'bg-green-50 text-green-600');

            centerMapOn(s.lat, s.lng);

            // Description Rendering
            const descCont = document.getElementById('ov-desc-container');
            const descEl = document.getElementById('ov-description');
            if (s.description) {
                descEl.innerText = s.description;
                descCont.classList.remove('hidden');
            } else {
                descCont.classList.add('hidden');
            }

            const oldBtn = document.getElementById('ov-inter-btn');
            if (oldBtn) oldBtn.remove();

            const interchanges = ['Ameerpet', 'MG Bus Station', 'Parade Ground'];
            if (interchanges.includes(s.name)) {
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
                if (a.includes('Food') || a.includes('KFC') || a.includes('Snack')) icon = 'utensils';
                if (a.includes('Parking')) icon = 'parking-circle';
                if (a.includes('ATM')) icon = 'credit-card';
                dev.innerHTML = `
                    <div class="w-8 h-8 bg-slate-50 text-slate-400 group-hover:bg-blue-50 group-hover:text-blue-600 rounded-lg flex items-center justify-center shrink-0 transition-colors">
                        <i data-lucide="${icon}" size="14"></i>
                    </div>
                    <div>
                        <h4 class="text-[11px] font-black text-slate-800 uppercase tracking-tight">${a}</h4>
                        <p class="text-[8px] font-bold text-slate-400 uppercase mt-0.5 tracking-widest">Verified Service</p>
                    </div>`;
                am.appendChild(dev);
            });

            document.getElementById('ov-plan-btn').onclick = () => {
                document.getElementById('end-st').value = s.id;
                showTab('routes');
                closeOverlay();
            };

            const trainCont = document.getElementById('ov-trains');
            trainCont.innerHTML = `<div class="py-10 flex flex-col items-center gap-4 text-slate-300"><div class="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div><p class="text-[9px] font-black uppercase tracking-widest">Loading departures...</p></div>`;

            try {
                const res = await fetch('/api/nearest', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ station_id: s.id }) });
                const data = await res.json();
                trainCont.innerHTML = '';
                data.upcoming.slice(0, 5).forEach(t => {
                    const lineCol = t.line === 'Red' ? 'bg-red-500' : t.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                    const tDiv = document.createElement('div');
                    tDiv.className = "flex justify-between items-center bg-slate-50 p-5 rounded-2xl border border-slate-100 hover:border-blue-200 transition-colors";
                    tDiv.innerHTML = `
                        <div class="flex items-center gap-4">
                            <div class="w-1 my-1 self-stretch rounded-full ${lineCol}"></div>
                            <div>
                                <p class="text-[11px] font-black text-slate-900 leading-none mb-1">${t.final_stop}</p>
                                <div class="flex items-center gap-2 mt-0.5"><span class="px-1.5 py-0.5 rounded text-[6px] font-black uppercase tracking-tighter text-white ${lineCol}">${t.line}</span><span class="text-[8px] font-bold text-slate-400 uppercase tracking-widest">Platform ${t.platform}</span></div>
                            </div>
                        </div>
                        <div class="text-right"><p class="text-sm font-black text-blue-600 tabular-nums">${t.arrival_time}</p><p class="text-[8px] font-black text-slate-400 uppercase tracking-widest">${t.eta}</p></div>`;
                    trainCont.appendChild(tDiv);
                });
            } catch (e) { trainCont.innerHTML = `<p class="text-xs font-bold text-red-500">Connection Failed</p>`; }
            lucide.createIcons();
        }

        function locateMeOnMap() {
            if (lastUserLoc) {
                // Telangana Bounds for centering check
                const latMin = 15.8, latMax = 19.9, lngMin = 77.2, lngMax = 81.3;
                const isOutOfTelangana = lastUserLoc.lat < latMin || lastUserLoc.lat > latMax || lastUserLoc.lng < lngMin || lastUserLoc.lng > lngMax;

                let targetLoc = lastUserLoc;
                if (isOutOfTelangana && window.lastNearestStation) {
                    targetLoc = window.lastNearestStation;
                    const status = document.getElementById('satellite-status');
                    if (status) status.innerText = "Showing Nearest Metro (User Outside State)";
                }

                const pt = project(targetLoc.lat, targetLoc.lng);

                // Centering logic: target point (pt.x, pt.y) should be at (500, 500) in coordinate space
                // Viewport translation: newX = (canvasWidth/2) - (pt.x * scale)
                mapViewport.scale = 2.0; 
                mapViewport.x = 500 - (pt.x * mapViewport.scale);
                mapViewport.y = 500 - (pt.y * mapViewport.scale);

                applyMapTransform();
                updateSVGUserPin(lastUserLoc.lat, lastUserLoc.lng);
            } else {
                window.shouldCenterOnNextFix = true;
                manualRefreshGeo();
                const status = document.getElementById('satellite-status');
                if (status) status.innerText = "Targeting User Vector...";
            }
        }

        let serverTimeOffset = ({{ SERVER_TIME_EPOCH }} * 1000) - Date.now();
        let currentSentiment = 'neutral';
        let tabState = 'home';
        let simulationHour = -1;
        let activeUpdateInterval = null;
        let currentPlannedRoute = null;
        let lastUserLoc = null;
        let lastStationLoc = null;
        let weatherInterval = null;
        let gpsFallbackMsg = null;

        const setElText = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.innerText = text;
            const elClone = document.getElementById(id + '-clone');
            if (elClone) elClone.innerText = text;
        };


        function openGoogleMaps() {
            if (!lastStationLoc) return;
            const dest = `${lastStationLoc.lat},${lastStationLoc.lng}`;
            const origin = lastUserLoc ? `${lastUserLoc.lat},${lastUserLoc.lng}` : "";
            const url = `https://www.google.com/maps/dir/?api=1&origin=${origin}&destination=${dest}&travelmode=walking`;
            window.open(url, '_blank');
        }

        async function refreshWeather() {
            if (!lastUserLoc) return;
            try {
                const res = await fetch('/api/weather', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ lat: lastUserLoc.lat, lng: lastUserLoc.lng })
                });
                const data = await res.json();

                if (data.server_epoch) {
                    serverTimeOffset = (data.server_epoch * 1000) - Date.now();
                }

                const vEl = document.getElementById('weather-val');
                const dEl = document.getElementById('weather-detail');
                const vElMob = document.getElementById('weather-val-mob');
                const cElMob = document.getElementById('weather-cond-mob');

                if (vEl) vEl.innerText = (data.temp || '--') + '°C';
                if (dEl) dEl.innerText = (data.condition || 'Clear') + ' in Hyderabad';
                if (vElMob) vElMob.innerText = (data.temp || '--') + '°C';
                if (cElMob) cElMob.innerText = data.condition || 'Cloudy';

                // Update extended stats if they exist
                setElText('stat-humidity', data.humidity + '%');
                setElText('stat-visibility', data.visibility + 'km');

                // Show Banner if Alert exists
                handleWeatherAlert(data);

                const weatherRec = (data.temp > 35) ? "Extreme heatwave detected. AC Metro cabins are optimal for travel today." :
                                   (data.condition || '').includes("Rain") ? "Rain detected. Metro is the safest and driest transit route." :
                                   "System is ready. Enjoy your commute across the network.";
                setElText('env-msg', weatherRec);
            } catch (e) { console.warn("Weather Refresh Fail", e); }
        }

        function handleWeatherAlert(data) {
            const banner = document.getElementById('weather-alert-banner');
            const alertMsg = document.getElementById('alert-msg');
            const alertAdvice = document.getElementById('alert-advice');
            const alertBody = document.getElementById('alert-body');
            const alertIconWrap = document.getElementById('alert-icon-wrap');
            const alertIcon = document.getElementById('alert-icon');

            if (data.alert && banner) {
                banner.classList.remove('hidden');
                alertMsg.innerText = data.alert;

                // Dynamic Advice mapping
                let advice = "Travel with caution";
                if (data.temp > 35) advice = "Visit AC Cooling Hubs";
                if (data.condition.includes("Rain")) advice = "Carry Umbrella";
                if (data.severity === "critical") advice = "Avoid Outdoor Travel";
                alertAdvice.innerText = "Advice: " + advice;

                // Severity Styling
                if (data.severity === "critical") {
                    alertBody.className = "rounded-[32px] p-6 flex flex-col lg:flex-row items-center gap-6 shadow-2xl border transition-all duration-500 alert-banner-critical text-white relative overflow-hidden critical-pulse";
                    alertIconWrap.className = "w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg bg-white/20 text-white";
                    alertIcon.setAttribute('data-lucide', 'alert-octagon');
                } else if (data.severity === "caution") {
                    alertBody.className = "rounded-[32px] p-6 flex flex-col lg:flex-row items-center gap-6 shadow-2xl border transition-all duration-500 alert-banner-caution text-white relative overflow-hidden";
                    alertIconWrap.className = "w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg bg-white/20 text-white";
                    alertIcon.setAttribute('data-lucide', 'cloud-lightning');
                } else {
                    banner.classList.add('hidden');
                    return;
                }
                if (window.lucide) lucide.createIcons();
            } else if (banner) {
                banner.classList.add('hidden');
            }
        }

        function dismissAlert() {
            const banner = document.getElementById('weather-alert-banner');
            if (banner) banner.style.display = 'none';
        }

        // Persistence Logic for Saved Routes
        function loadSavedVectors() {
            const saved = JSON.parse(localStorage.getItem('metro_saved_routes') || '[]');
            const container = document.getElementById('saved-routes-list');
            const section = document.getElementById('saved-routes-section');

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
                                <p class="text-[8px] font-black text-slate-400 uppercase tracking-widest mt-1">${v.line || 'Multi-Line'} Route</p>
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

            const saved = JSON.parse(localStorage.getItem('metro_saved_routes') || '[]');
            const startId = document.getElementById('start-st').value;
            const endId = document.getElementById('end-st').value;

            if (saved.some(v => v.from === startId && v.to === endId)) {
                alert("Route already saved.");
                return;
            }

            const clean = (str) => {
                let s = str.split(' ').slice(1).join(' ');
                return s.split(' 💻')[0].split(' 🔄')[0].trim();
            };

            const startNode = document.getElementById('start-st');
            const endNode = document.getElementById('end-st');

            const route = {
                id: Date.now(),
                from: startId,
                to: endId,
                fromName: clean(startNode.options[startNode.selectedIndex].text),
                toName: clean(endNode.options[endNode.selectedIndex].text),
                line: stations.find(s => s.id === startId).line
            };

            saved.push(route);
            localStorage.setItem('metro_saved_routes', JSON.stringify(saved));

            const btn = document.getElementById('save-route-btn');
            btn.innerHTML = '<i data-lucide="check" size="14"></i> Route Saved';
            btn.classList.replace('bg-slate-900', 'bg-emerald-500');

            setTimeout(() => {
                btn.innerHTML = '<i data-lucide="star" size="14"></i> Save Route';
                btn.classList.replace('bg-emerald-500', 'bg-slate-900');
                lucide.createIcons();
            }, 2000);

            loadSavedVectors();
            lucide.createIcons();
        }

        function removeSavedVector(event, id) {
            event.stopPropagation();
            const saved = JSON.parse(localStorage.getItem('metro_saved_routes') || '[]');
            const filtered = saved.filter(v => v.id.toString() !== id.toString());
            localStorage.setItem('metro_saved_routes', JSON.stringify(filtered));
            loadSavedVectors();
        }

        function shareRoute() {
            const startName = document.getElementById('start-st').options[document.getElementById('start-st').selectedIndex].text;
            const endName = document.getElementById('end-st').options[document.getElementById('end-st').selectedIndex].text;
            const msg = `⚡ Commuting via HydMetro: ${startName} to ${endName}. Guide found at ₹${lastCalculatedFare}!`;

            if (navigator.share) {
                navigator.share({ title: 'HydMetro Route', text: msg, url: window.location.href });
            } else {
                navigator.clipboard.writeText(msg);
                alert("Route details copied to clipboard.");
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
                        <p class="text-[10px] font-bold text-white/80">Ticket Verified</p>
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
                card.className = "digital-pass animate-in fade-in slide-in-from-bottom-8 duration-700 shadow-2xl overflow-hidden";
                card.innerHTML = `
                    <div class="bg-white p-8 pb-12 relative">
                        <div class="flex justify-between items-center mb-8">
                            <div class="flex items-center gap-3">
                                <div class="w-12 h-12 bg-slate-900 rounded-2xl flex items-center justify-center text-white shadow-lg">
                                    <i data-lucide="train-front" size="24"></i>
                                </div>
                                <div>
                                    <h3 class="text-base font-black text-slate-900 tracking-tight uppercase italic">Metro Ticket</h3>
                                    <p class="text-[9px] font-black text-blue-600 uppercase tracking-widest">Live Verified</p>
                                </div>
                            </div>
                            <div class="text-right">
                                <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-1">Ticket ID</p>
                                <p class="text-[11px] font-black text-slate-900 bg-slate-50 px-2 py-1 rounded-lg border border-slate-100 tabular-nums">${activeTicket.id}</p>
                            </div>
                        </div>

                        <div class="flex justify-between items-end gap-4 mb-4">
                            <div class="flex-1 overflow-hidden">
                                <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-3">From Station</p>
                                <h4 class="text-2xl font-black text-slate-900 truncate tracking-tight uppercase italic">${activeTicket.from}</h4>
                                <p class="text-[8px] font-bold text-blue-500 uppercase mt-1 tracking-widest opacity-60">Entrance</p>
                            </div>
                            <div class="flex flex-col items-center gap-1 pb-1">
                                <div class="w-2 h-2 rounded-full border-2 border-slate-200 bg-white"></div>
                                <div class="w-px h-8 bg-slate-100"></div>
                                <i data-lucide="arrow-right" size="18" class="text-blue-500"></i>
                                <div class="w-px h-8 bg-slate-100"></div>
                                <div class="w-2 h-2 rounded-full border-2 border-slate-200 bg-white"></div>
                            </div>
                            <div class="flex-1 text-right overflow-hidden">
                                <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-3">To Station</p>
                                <h4 class="text-2xl font-black text-slate-900 truncate tracking-tight uppercase italic">${activeTicket.to}</h4>
                                <p class="text-[8px] font-bold text-emerald-500 uppercase mt-1 tracking-widest opacity-60">Exit</p>
                            </div>
                        </div>
                    </div>

                    <div class="pass-cutout pass-cutout-left"></div>
                    <div class="pass-cutout pass-cutout-right"></div>
                    <div class="pass-divider"></div>

                    <div class="bg-white px-8 pt-12 pb-10 flex flex-col items-center">
                        <div id="ticket-qr" class="bg-white p-6 rounded-[3rem] border border-slate-50 shadow-[inset_0_4px_20px_rgba(0,0,0,0.03)] mb-8"></div>
                        <div class="grid grid-cols-2 gap-12 w-full mb-10 border-t border-slate-50 pt-8 mt-2 px-2">
                            <div>
                                <p class="text-[9px] font-black text-slate-300 uppercase tracking-widest mb-1 pl-1">Issue Time</p>
                                <p class="text-[11px] font-black text-slate-900 uppercase tabular-nums">${activeTicket.timestamp}</p>
                            </div>
                            <div class="text-right pr-1">
                                <p class="text-[9px] font-black text-slate-300 uppercase tracking-widest mb-1">Fare Value</p>
                                <p class="text-[11px] font-black text-blue-600 uppercase tabular-nums">INR ${activeTicket.fare}.00</p>
                            </div>
                        </div>
                        <button onclick="completeTrip('${activeTicket.id}')" class="w-full py-6 bg-slate-900 text-white rounded-[2rem] font-black text-[11px] uppercase tracking-widest flex items-center justify-center gap-3 shadow-xl active:scale-[0.97] transition-all">
                             <i data-lucide="check-circle" size="18"></i> Terminate Journey
                        </button>
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
                            <span class="text-[8px] font-black text-emerald-600 uppercase tracking-widest bg-emerald-50 px-3 py-1.5 rounded-xl border border-emerald-100 inline-block mt-2">Past Trip</span>
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

        let userLocationMarker;
        function updateUserPin(lat, lng) {
            if (!googleMap || !advancedMarkersLib) return;

            if (!userLocationMarker) {
                const pinContent = document.createElement('div');
                pinContent.innerHTML = `
                    <div style="position: relative; display: flex; align-items: center; justify-content: center;">
                        <div style="position: absolute; width: 60px; height: 60px; background: rgba(34, 211, 238, 0.4); border-radius: 50%; animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;"></div>
                        <div style="width: 16px; height: 16px; background: #06b6d4; border: 3px solid white; border-radius: 50%; box-shadow: 0 0 20px #06b6d4; z-index: 2; animation: pulse-dot 2s infinite;"></div>
                    </div>
                `;

                // Add keyframes if not exists
                if (!document.getElementById('user-pulse-style')) {
                    const style = document.createElement('style');
                    style.id = 'user-pulse-style';
                    style.innerHTML = `
                        @keyframes pulse-ring {
                            0% { transform: scale(0.33); opacity: 1; }
                            80%, 100% { transform: scale(2.5); opacity: 0; }
                        }
                        @keyframes pulse-dot {
                            0% { transform: scale(0.9); }
                            50% { transform: scale(1.1); }
                            100% { transform: scale(0.9); }
                        }
                    `;
                    document.head.appendChild(style);
                }

                userLocationMarker = new advancedMarkersLib.AdvancedMarkerElement({
                    map: googleMap,
                    position: { lat, lng },
                    content: pinContent,
                    title: "Your Live Position"
                });
            } else {
                userLocationMarker.position = { lat, lng };
            }

            const locText = document.getElementById('user-location-text');
            if (locText && (locText.innerText.includes('Finding') || locText.innerText.includes('Syncing'))) {
                locText.innerText = "Handshaking with Satellites...";
            }
        }

        function updateSVGUserPin(lat, lng) {
            const pt = project(lat, lng);
            let userPin = document.getElementById('user-location-pin');
            const userGroup = document.getElementById('map-user');

            if (!userPin && userGroup) {
                userPin = document.createElementNS("http://www.w3.org/2000/svg", "g");
                userPin.setAttribute("id", "user-location-pin");
                userPin.innerHTML = `
                    <circle cx="0" cy="0" r="12" class="user-location-glow-outer"></circle>
                    <circle cx="0" cy="0" r="12" class="user-location-glow-inner"></circle>
                    <circle cx="0" cy="0" r="8" class="user-location-core"></circle>
                    <circle cx="0" cy="0" r="3" class="user-location-subcore"></circle>
                `;
                userGroup.appendChild(userPin);
            }

            if (userPin) {
                userPin.setAttribute("transform", `translate(${pt.x}, ${pt.y})`);

                if (window.shouldCenterOnNextFix) {
                    locateMeOnMap(); // Recursive call but with lastUserLoc present
                    window.shouldCenterOnNextFix = false;
                }
            }
        }

        function focusOnMe() {
            if (!lastUserLoc) {
                window.shouldCenterOnNextFix = true;
                manualRefreshGeo();
                return;
            }
            locateMeOnMap();
        }

        function showTab(id) {
            tabState = id;
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.mobile-link').forEach(l => l.classList.remove('active'));

            const contentTab = document.getElementById('tab-'+id);
            const mobileTab = document.getElementById('mob-'+id);

            if (contentTab) contentTab.classList.add('active');
            if (mobileTab) mobileTab.classList.add('active');

            if(id === 'map') {
                if (!document.getElementById('map-lines').children.length) {
                    initCustomMetroMap();
                } else {
                    setTimeout(resetMap, 50);
                }
            } else if (id === 'stations') {
                renderStationsDirectory();
            } else if (id === 'tickets') {
                renderTickets();
            } else {
                closeOverlay();
            }
            window.scrollTo({ top: 0, behavior: 'smooth' });
            lucide.createIcons();
        }

        let currentDirFilter = 'All';
        function filterDirByLine(line) {
            currentDirFilter = line;
            document.querySelectorAll('.dir-line-btn').forEach(b => {
                b.classList.remove('active', 'bg-white', 'shadow-sm');
            });
            const btn = document.getElementById('dir-filter-' + line);
            btn.classList.add('active');
            if(line !== 'All') btn.classList.add('bg-white', 'shadow-sm');
            renderStationsDirectory();
        }

        function renderStationsDirectory() {
            const grid = document.getElementById('stations-grid');
            if (!grid) return;

            const subtitle = document.getElementById('stations-dir-subtitle');
            const searchQ = document.getElementById('dir-search').value.toLowerCase().trim();

            if (currentDirFilter === 'Blue' && !searchQ) {
                subtitle.innerText = "BLUE LINE (Raidurg → Nagole) — Full details with services";
                subtitle.className = "text-[10px] font-black text-blue-500 uppercase tracking-[0.3em]";
            } else if (currentDirFilter === 'Red' && !searchQ) {
                subtitle.innerText = "RED LINE (Miyapur → LB Nagar) — Full details with services";
                subtitle.className = "text-[10px] font-black text-red-500 uppercase tracking-[0.3em]";
            } else if (currentDirFilter === 'Green' && !searchQ) {
                subtitle.innerText = "GREEN LINE (JBS → MGBS) — Full details with services";
                subtitle.className = "text-[10px] font-black text-green-500 uppercase tracking-[0.3em]";
            } else {
                subtitle.innerText = "Explore all stations and facilities";
                subtitle.className = "text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]";
            }

            grid.innerHTML = '';

            let filteredSt = currentDirFilter === 'All' ? stations : stations.filter(s => s.line === currentDirFilter);
            if (searchQ) {
                filteredSt = filteredSt.filter(s => s.name.toLowerCase().includes(searchQ) || s.id.toLowerCase().includes(searchQ));
            }

            if (filteredSt.length === 0) {
                grid.innerHTML = '<div class="col-span-full py-32 text-center flex flex-col items-center justify-center text-slate-300"><i data-lucide="search-x" size="48" class="mb-4 opacity-10"></i><p class="text-[10px] font-black uppercase tracking-widest opacity-40">No stations detected in this sector</p></div>';
                lucide.createIcons();
                return;
            }

            const interchanges = ['Ameerpet', 'MG Bus Station', 'Parade Ground'];

            filteredSt.forEach((s, idx) => {
                const card = document.createElement('div');
                card.className = "glass-card p-6 flex flex-col gap-5 group cursor-pointer hover:border-blue-400 hover:shadow-xl hover:shadow-blue-500/5 transition-all active:scale-[0.98] animate-in fade-in slide-in-from-bottom-4 duration-500";
                card.style.animationDelay = `${idx * 30}ms`;
                card.onclick = () => {
                    showTab('map');
                    handleStationInteraction(s);
                };

                const lineColor = s.line === 'Red' ? 'bg-red-500' : s.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                const lineText = s.line === 'Red' ? 'text-red-500' : s.line === 'Blue' ? 'text-blue-500' : 'text-green-500';
                const isInter = interchanges.includes(s.name);

                card.innerHTML = `
                    <div class="flex justify-between items-start">
                        <div class="w-10 h-10 ${lineColor} rounded-2xl flex items-center justify-center text-white shadow-lg shadow-current/20 group-hover:scale-110 transition-transform">
                            <i data-lucide="${isInter ? 'shuffle' : 'train-front'}" size="18"></i>
                        </div>
                        <div class="flex flex-col items-end gap-1">
                            <span class="text-[9px] font-black uppercase ${lineText} tracking-[0.2em] bg-slate-50 px-3 py-1 rounded-full border border-slate-100">${s.line} Line</span>
                            ${isInter ? '<span class="text-[7px] font-black uppercase text-amber-500 tracking-widest bg-amber-50 px-2 py-0.5 rounded border border-amber-100">Transfer Station</span>' : ''}
                        </div>
                    </div>
                    <div>
                        <h4 class="text-xl font-black text-slate-900 tracking-tight group-hover:text-blue-600 transition-colors uppercase italic">${s.name}</h4>
                        <div class="flex items-center gap-2 mt-2">
                             <div class="w-1 h-1 rounded-full bg-slate-300"></div>
                             <p class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">${s.id} Station</p>
                        </div>
                        <p class="mt-4 text-[11px] font-medium text-slate-500 leading-relaxed line-clamp-3">${s.description || 'Station details and area description available in the interactive map view.'}</p>
                    </div>
                    <div class="flex flex-wrap gap-2 mt-2">
                        ${(s.amenities || ['Parking', 'ATM', 'Lift', 'Security']).slice(0, 3).map(a => `
                            <span class="px-2.5 py-1 bg-slate-50 border border-slate-100 rounded-lg text-[8px] font-black text-slate-500 uppercase tracking-tighter group-hover:bg-blue-50 group-hover:border-blue-100 transition-colors">${a}</span>
                        `).join('')}
                    </div>
                    <div class="pt-4 border-t border-slate-50 flex items-center justify-between mt-auto">
                         <span class="text-[8px] font-black text-slate-400 uppercase tracking-[0.2em]">View Details on Map</span>
                         <i data-lucide="arrow-right" size="12" class="text-slate-300 group-hover:translate-x-1 transition-transform group-hover:text-blue-500"></i>
                    </div>
                `;
                grid.appendChild(card);
            });
            lucide.createIcons();
        }

        function closeOverlay() {
            const overlay = document.getElementById('map-overlay');
            overlay.classList.remove('active');
            document.querySelectorAll('.station-node').forEach(n => n.classList.remove('selected'));
        }

        function updateClock() {
            try {
                let now = new Date(Date.now() + serverTimeOffset);
                if (typeof simulationHour !== 'undefined' && simulationHour !== -1) {
                    now.setHours(simulationHour);
                }

                const hours = now.getHours();
                const gText = hours < 12 ? "Good Morning" : hours < 17 ? "Good Afternoon" : "Good Evening";
                const baseGreeting = document.getElementById('base-greeting');
                if (baseGreeting) baseGreeting.innerText = gText + '!';
                const homeGreeting = document.getElementById('greeting');
                if (homeGreeting) homeGreeting.innerText = gText + '!';

                let options = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true };
                let timeStr = now.toLocaleTimeString('en-US', options);
                let parts = timeStr.split(/[\s\u00A0\u202F]/);

                // Dashboard Clock
                const hClock = document.getElementById('home-clock');
                if (hClock) hClock.innerText = parts[0];
                const hAmpm = document.getElementById('home-clock-ampm');
                if (hAmpm) hAmpm.innerText = parts[parts.length - 1];
                const hDate = document.getElementById('home-date');
                if (hDate) hDate.innerText = now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }).toUpperCase();

                // Other clocks (compat)
                const clockEl = document.getElementById('clock');
                if (clockEl && parts.length >= 2) clockEl.innerText = parts[0];
                const ampmEl = document.getElementById('ampm');
                if (ampmEl && parts.length >= 2) ampmEl.innerText = parts[parts.length - 1];
                const dateEl = document.getElementById('date');
                if (dateEl) dateEl.innerText = now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });

                const mapClock = document.getElementById('map-clock');
                if (mapClock) mapClock.innerText = now.toLocaleTimeString('en-US', { hour12: false });
            } catch (e) {
                console.error("Clock Update Failed:", e);
            }
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
                guidance: [
                    "Red Line platforms are situated on Level 1.",
                    "Blue Line platforms are situated on Level 2.",
                    "Use the central concourse escalators for rapid transition between levels.",
                    "Check dynamic signage for platform occupancy real-time."
                ]
            },
            'MG Bus Station': {
                lines: ['Red', 'Green'],
                guidance: [
                    "Transfer between the terminal stations via the dedicated Interchange Walkway.",
                    "Red Line serves the North-South corridor (Level 1).",
                    "Green Line serves the terminal wing (Adjacent).",
                    "Follow the floor-haptic strips for low-visibility guidance."
                ]
            },
            'Parade Ground': {
                lines: ['Blue', 'Green'],
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
            badges.innerHTML = '';
            data.lines.forEach(line => {
                const b = document.createElement('span');
                const color = line === 'Red' ? 'bg-red-500' : line === 'Blue' ? 'bg-blue-500' : 'bg-green-500';
                b.className = `px-3 py-1 rounded-full text-[10px] font-black text-white uppercase tracking-widest ${color}`;
                b.innerText = line + ' Line';
                badges.appendChild(b);
            });

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
                    handleStationInteraction(m);
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

            if (q === '') {
                clearBtn.classList.add('hidden');
            } else {
                clearBtn.classList.remove('hidden');
            }

            showMapSuggestions(query);

            // If query matches a station exactly (case insensitive), pan and zoom
            const match = stations.find(s => s.name.toLowerCase() === q);
            if (match) {
                handleStationInteraction(match);
            }
        }

        function initSearch() {
            window.addEventListener('keydown', (e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                    if (tabState === 'map') {
                        e.preventDefault();
                        document.getElementById('map-search').focus();
                    }
                }
            });
        }

        async function updateBoardData(lat, lng, stationId = null) {
            try {
                if (!lat && !lng && !stationId) return;

                const boardLoading = document.getElementById('board-loading');
                const boardContent = document.getElementById('board-content');

                // Show loading only if we haven't succeeded yet
                if (boardLoading && !syncSuccessOnce) {
                    boardLoading.classList.remove('hidden');
                }

                if (lat && lng) {
                    updateUserPin(lat, lng);
                    updateSVGUserPin(lat, lng);
                }

                const body = stationId ? { station_id: stationId } : { lat, lng };
                const res = await fetchWithSim('/api/nearest', { 
                    method: 'POST', 
                    headers: {'Content-Type': 'application/json'}, 
                    body: JSON.stringify(body) 
                });

                if (!res.ok) throw new Error("API Offline");
                const data = await res.json();
                window.lastNearestStation = data;

                if (data.server_epoch) {
                    serverTimeOffset = (data.server_epoch * 1000) - Date.now();
                }

                // Header Clock & Date Logic
                const now = new Date(Date.now() + (window.serverTimeOffset || 0));
                setElText('base-greeting', data.greeting + '!');
                setElText('greeting', data.greeting + '!');

                const timeStr = now.toLocaleTimeString('en-US', { hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' }).split(' ');
                setElText('home-clock', timeStr[0]);
                setElText('home-clock-ampm', timeStr[1]);
                setElText('home-date', now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }));

                if (!syncSuccessOnce && lat && lng && !stationId) {
                    setTimeout(locateMeOnMap, 500);
                }
                syncSuccessOnce = true;
                // keep the full data object here as other parts of the code expect .distance and .station
                window.lastNearestStation = data; 

                if (lat && lng && !stationId) lastUserLoc = { lat, lng };
                lastStationLoc = { lat: data.station.lat, lng: data.station.lng };

                // Update UI with real data context
                setElText('near-name', data.station.name);
                setElText('near-name-mob-2', data.station.name);
                setElText('near-dist', data.distance + ' km (Crow-flies)');
                setElText('near-dist-mob-2', data.distance + ' km away');

                let locDisplay = data.station.name + ' Station';
                if (data.range_status === 'Remote') locDisplay = "Remote Mode: " + data.station.name;
                else if (data.range_status === 'Out of City') locDisplay = data.station.name + ' (Out of bounds)';

                setElText('user-location-text', locDisplay);

                if (data.geofence_msg) {
                    setElText('env-msg', data.geofence_msg);
                }

                // Atmosphere & Dashboard Updates
                if (data.weather) {
                    handleWeatherAlert(data.weather);
                    setElText('weather-val', data.weather.temp + '°C');
                    setElText('weather-detail', data.weather.condition + ' in Hyderabad');
                    setElText('dash-weather-temp', data.weather.temp + '°C, ' + data.weather.condition);
                    setElText('dash-weather-detail', `Humidity: ${data.weather.humidity}% | Visibility: ${data.weather.visibility.toFixed(1)}km`);

                    const envWeatherMsg = document.getElementById('env-weather-msg');
                    if (envWeatherMsg) {
                        const msg = data.weather.temp > 35 ? "AC cooling optimized for low heat." : 
                                                 data.weather.condition.includes("Rain") ? "Metro tunnels are dry. Ideal for transit." : 
                                                 "Atmospheric conditions stable.";
                        setElText('env-weather-msg', msg);
                    }

                    // Arrivals Board weather indicator
                    const boardWeather = document.getElementById('board-weather-indicator');
                    const boardTemp = document.getElementById('board-weather-temp');
                    const mobWeather = document.getElementById('near-weather-mob');

                    if (boardWeather && boardTemp) {
                        boardTemp.innerText = data.weather.temp + '°C';
                        boardWeather.classList.remove('hidden');
                    }
                    if (mobWeather) {
                        mobWeather.innerText = data.weather.temp + '°C';
                        mobWeather.classList.remove('hidden');
                    }

                    // Update extended stats if they exist
                    setElText('stat-humidity', data.weather.humidity + '%');
                    setElText('stat-visibility', data.weather.visibility + 'km');
                }

                // Dash Card Location Update
                setElText('near-name-dash', data.station.name);
                setElText('near-dist-dash', data.distance + ' KM AWAY');
                setElText('near-walk-dash', `${data.walking_mins} MIN WALK (${data.walk_dist}KM)`);
                setElText('board-station-name', data.station.name);

                // GTFS Pulse & Load
                setElText('home-train-count', data.active_trips.total);
                setElText('load-label-dash', data.load_label);
                const loadPill = document.getElementById('load-pill-dash');
                if (loadPill) {
                    const dot = loadPill.querySelector('div');
                    if (data.load_val === 'High') {
                        dot.className = 'w-2 h-2 rounded-full bg-red-500 shadow-[0_0_10px_#ef4444]';
                    } else if (data.load_val === 'M-High') {
                        dot.className = 'w-2 h-2 rounded-full bg-amber-500 shadow-[0_0_10px_#f59e0b]';
                    } else {
                        dot.className = 'w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_#10b981]';
                    }
                }

                // Render Arrival Board
                const boardList = document.getElementById('arrival-board-list');
                if (boardList) {
                    if (data.upcoming && data.upcoming.length > 0) {
                        boardList.innerHTML = data.upcoming.map(item => {
                            const lineCol = item.line === 'Red' ? 'bg-red-600' : item.line === 'Blue' ? 'bg-blue-600' : 'bg-emerald-600';
                            return `
                                <div class="flex flex-col lg:grid lg:grid-cols-4 items-center gap-4 lg:gap-0 px-10 py-8 hover:bg-slate-50 transition-colors">
                                    <div class="flex items-center gap-3 w-full lg:w-auto">
                                        <div class="w-1 h-8 ${lineCol} rounded-full"></div>
                                        <span class="text-[9px] font-black text-slate-800 uppercase tracking-widest">${item.line} Line Entry</span>
                                    </div>
                                    <div class="w-full lg:w-auto">
                                        <h5 class="text-sm font-black text-slate-900">${item.final_stop}</h5>
                                        <div class="flex items-center gap-2 mt-1">
                                            <span class="px-2 py-0.5 ${lineCol} text-white text-[8px] font-black rounded uppercase">PL ${item.platform}</span>
                                            <span class="text-[8px] font-bold text-slate-400 uppercase tracking-widest">${item.direction}</span>
                                        </div>
                                    </div>
                                    <div class="w-full lg:w-auto">
                                        <p class="text-sm font-black text-slate-900 tabular-nums">${item.arrival_time}</p>
                                        <p class="text-[8px] font-bold text-slate-400 uppercase tracking-widest">Scheduled Arrival</p>
                                    </div>
                                    <div class="w-full lg:w-auto lg:text-right">
                                        <div class="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-xl">
                                            <span class="text-xs font-black tabular-nums">${item.eta}</span>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }).join('');
                    } else {
                        boardList.innerHTML = `<div class="p-20 text-center text-slate-400 text-xs font-bold uppercase tracking-widest italic">No trains scheduled in this sector.</div>`;
                    }
                }

                const walkContainer = document.getElementById('near-walk-container');
                const walkTimeEl = document.getElementById('near-walk-time');
                if (walkContainer && walkTimeEl) {
                    if (data.range_status === 'Out of City' || data.distance > 15) {
                        walkTimeEl.innerText = `Vehicle Required (${data.distance}km)`;
                    } else {
                        walkTimeEl.innerText = `${data.walking_mins} min walk (${data.walk_dist}km)`;
                    }
                    walkContainer.classList.remove('hidden');
                }

                // Update System Status
                setElText('live-train-count', (data.active_trips.total || data.active_trips) + ' Trains Active');

                const ridershipEl = document.getElementById('ridership-val');
                if (ridershipEl) {
                    const activeCount = data.active_trips.total || data.active_trips;
                    const estRiders = (activeCount * 450) + Math.floor(Math.random() * 200);
                    ridershipEl.innerText = estRiders.toLocaleString();
                }

                let statusLine = `${data.walking_mins} min walk | ${data.load_label}`;
                if (data.range_status === 'Out of City') statusLine = `Inter-city Location | ${data.distance}km away`;
                else if (data.range_status === 'Peripheral') statusLine = `Peripheral Area | ${data.distance}km away`;

                const netStatusEl = document.getElementById('near-dist-status');
                if (netStatusEl) {
                    if (gpsFallbackMsg) {
                        netStatusEl.innerHTML = `${gpsFallbackMsg} <button onclick="manualRefreshGeo()" class="px-2 py-0.5 bg-amber-500/20 rounded-md underline font-black cursor-pointer hover:bg-amber-500/40 transition-all ml-2">RETRY</button>`;
                        netStatusEl.classList.add('text-amber-400');
                    } else {
                        netStatusEl.innerText = statusLine;
                        netStatusEl.classList.remove('text-amber-400');
                    }
                }

                const navBtn = document.getElementById('nav-btn');
                if (navBtn) navBtn.classList.remove('hidden');

                const mapJumpBtn = document.getElementById('map-jump-btn');
                if (mapJumpBtn) mapJumpBtn.classList.remove('hidden');

                setElText('near-metro-live', data.station.name);
                setElText('near-metro-mob', 'Near ' + data.station.name + ' Station');
                const activeCountTotal = data.active_trips.total || data.active_trips;
                setElText('active-count', activeCountTotal);
                setElText('active-count-mob', activeCountTotal);
                if (document.getElementById('near-trains-mob')) document.getElementById('near-trains-mob').classList.remove('hidden');

                // Update Line-specific counts if elements exist
                if (data.active_trips.lines) {
                    setElText('red-line-active', data.active_trips.lines.Red);
                    setElText('blue-line-active', data.active_trips.lines.Blue);
                    setElText('green-line-active', data.active_trips.lines.Green);
                }

                // Update Network Stats
                setElText('stat-stations', data.stations_total || '57');
                setElText('stat-length', (data.line_length || '72') + ' km');
                setElText('stat-riders', (data.daily_riders || '420k') + '+');
                setElText('stat-speed', (data.top_speed || '80') + ' km/h');

                const loadStatusEl = document.getElementById('load-status');
                if (loadStatusEl) {
                    loadStatusEl.innerText = data.load_label;
                    loadStatusEl.className = 'px-3 py-1 bg-white/10 rounded-lg text-[9px] font-black uppercase tracking-widest border border-white/10 ' + 
                                            (data.load_val === 'High' ? 'text-red-400' : 'text-emerald-400');
                }

                setElText('greeting', data.greeting + '!');
                setElText('greeting-mob', data.greeting + '!');
                const weatherVal = document.getElementById('weather-val');
                if (weatherVal) {
                    weatherVal.innerText = data.weather.temp + '°C, ' + data.weather.condition;
                    weatherVal.classList.remove('animate-pulse');
                }

                // Sync selector if in auto-mode
                const selector = document.getElementById('board-station-selector');
                if (!stationId) {
                    selector.value = data.station.id;
                }

                const weatherDetail = document.getElementById('weather-detail');
                if (weatherDetail) {
                    weatherDetail.innerText = `Humidity: ${data.weather.humidity}% | Visibility: ${data.weather.visibility.toFixed(1)}km`;
                    weatherDetail.classList.remove('animate-pulse');
                }

                const weatherRec = data.weather.temp > 35 ? "Extreme heatwave detected. AC Metro cabins are optimal for travel today." :
                                   data.weather.condition.includes("Rain") ? "Rain detected. Metro is the safest and driest transit route." :
                                   "System is ready. Enjoy your commute across the network.";
                setElText('env-msg', weatherRec);

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
                const syncIcon = document.getElementById('sync-icon');
                if (syncIcon) {
                    syncIcon.classList.remove('text-blue-600', 'animate-pulse');
                    syncIcon.classList.add('text-red-500');
                }

                if (syncSuccessOnce) {
                    // Just show a subtle warning if we already have data
                    const netStatus = document.getElementById('near-dist-status');
                    if (netStatus) netStatus.innerText = "Sync Latency Detected. Reconnecting...";
                    return;
                }

                const boardRows = document.getElementById('board-rows');
                if (boardRows) boardRows.innerHTML = '';
                const boardLoading = document.getElementById('board-loading');
                if (boardLoading) {
                    boardLoading.classList.remove('hidden');
                    boardLoading.innerHTML = `
                    <div class="flex flex-col items-center gap-6 py-12 animate-in fade-in zoom-in duration-500">
                        <div class="w-20 h-20 bg-red-50 rounded-[40px] flex items-center justify-center text-red-500 shadow-inner relative">
                            <div class="absolute inset-0 bg-red-400 rounded-[40px] animate-ping opacity-20"></div>
                            <i data-lucide="wifi-off" size="32" class="relative"></i>
                        </div>
                        <div class="text-center px-6">
                            <h4 class="text-xl font-black text-slate-900 tracking-tight italic uppercase">Sync Vector Interrupted</h4>
                            <p class="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mt-3 max-w-xs leading-relaxed">
                                Satellite handshake failed. Check your data connection or terminal location permissions.
                            </p>
                        </div>
                        <div class="flex gap-4">
                            <button onclick="manualRefreshGeo()" class="px-8 py-4 bg-slate-900 text-white rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] hover:bg-black transition-all shadow-2xl active:scale-95 flex items-center gap-2">
                                <i data-lucide="refresh-cw" size="14"></i> Retry Sync
                            </button>
                            <button onclick="manualStationChange('R11')" class="px-8 py-4 bg-blue-50 text-blue-600 rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] border border-blue-100 hover:bg-blue-100 transition-all active:scale-95">
                                Select Ameerpet
                            </button>
                        </div>
                    </div>
                `;
                }
                lucide.createIcons();
            }
        }

        async function manualStationChange(forcedId) {
            const selector = document.getElementById('board-station-selector');
            const sid = forcedId || selector.value;
            if (!sid) {
                manualRefreshGeo();
                return;
            }
            if (forcedId) selector.value = forcedId;
            const s = stations.find(st => st.id === sid);
            if (s) updateBoardData(s.lat, s.lng, s.id);
        }

        function manualRefreshGeo() {
            performSatelliteHandshake();
        }

        async function manualRefreshGeoHome() {
            const btn = document.querySelector('button[onclick="manualRefreshGeoHome()"]');
            if (btn) {
                const originalContent = btn.innerHTML;
                btn.innerHTML = '<i data-lucide="refresh-cw" class="animate-spin" size="14"></i> Syncing...';
                btn.disabled = true;
                lucide.createIcons();
                await performSatelliteHandshake();
                btn.innerHTML = originalContent;
                btn.disabled = false;
                lucide.createIcons();
            } else {
                performSatelliteHandshake();
            }
        }

        function openGoogleMapsHome() {
            if (lastUserLoc && window.lastNearestStation && window.lastNearestStation.station) {
                const mode = window.lastNearestStation.distance <= 2 ? 'walking' : 'driving';
                const dest = window.lastNearestStation.station;
                const url = `https://www.google.com/maps/dir/?api=1&origin=${lastUserLoc.lat},${lastUserLoc.lng}&destination=${dest.lat},${dest.lng}&travelmode=${mode}`;
                window.open(url, '_blank');
            }
        }

        async function performSatelliteHandshake() {
            const status = document.getElementById('satellite-status');
            const userLocText = document.getElementById('user-location-text');
            const netStatus = document.getElementById('near-dist-status');
            const nearName = document.getElementById('near-name');
            const nearDist = document.getElementById('near-dist');

            // Quick Load from Cache
            try {
                const cached = localStorage.getItem('hyd_metro_cache_loc');
                if (cached) {
                    const { lat, lng, name, time } = JSON.parse(cached);
                    const ageInMins = (Date.now() - time) / 1000 / 60;
                    if (ageInMins < 60) {
                        console.log("Restoring Cached Location Fix...");
                        lastUserLoc = { lat, lng };
                        await updateBoardData(lat, lng);
                        if (netStatus) netStatus.innerText = "Cached Fix: " + (name || "Hyderabad");
                        if (userLocText) userLocText.classList.remove('animate-pulse');
                    }
                }
            } catch (e) { console.warn("Cache Restore Failed", e); }

            if (status) {
                status.innerText = "Syncing with Google Satellite...";
                status.classList.remove('text-amber-500');
                status.classList.add('text-emerald-500');
            }

            if (userLocText) {
                userLocText.innerText = "Syncing with Geo-Network...";
            }

            if (netStatus) netStatus.innerText = "Quantum Sync: Active";
            if (nearName) nearName.innerText = "Metro Grid...";
            if (nearDist) nearDist.innerText = "Calculated Lat/Lng...";

            try {
                if (!navigator.geolocation) throw new Error("GPS_NOT_SUPPORTED");

                // Rapid Synchronization
                const pos = await Promise.race([
                    new Promise((resolve, reject) => {
                        navigator.geolocation.getCurrentPosition(resolve, reject, { 
                            enableHighAccuracy: true, 
                            timeout: 10000, 
                            maximumAge: 0
                        });
                    }),
                    new Promise((_, reject) => {
                        setTimeout(() => reject(new Error("METRO_GPS_TIMEOUT")), 10500);
                    })
                ]);

                gpsFallbackMsg = null;
                const { latitude: lat, longitude: lng } = pos.coords;

                // Google Geocoding Sync
                if (typeof google !== 'undefined' && geocoder) {
                    geocoder.geocode({ location: { lat, lng } }, (results, status) => {
                        if (status === "OK" && results[0]) {
                            const address = results[0].formatted_address;
                            const cityName = address.split(',')[0];
                            console.log("Google Sync Success:", address);
                            const netStatusEl = document.getElementById('near-dist-status');
                            if (netStatusEl) {
                                netStatusEl.innerText = "Geo-Sync: " + cityName;
                                netStatusEl.classList.add('text-emerald-400');
                            }
                            // Save to cache for next visit
                            localStorage.setItem('hyd_metro_cache_loc', JSON.stringify({
                                lat, lng, name: cityName, time: Date.now()
                            }));
                        }
                    });
                }

                lastUserLoc = { lat, lng };
                console.log("GPS Fix Acquired:", lastUserLoc);
                await updateBoardData(lat, lng);

                if (status) {
                    status.innerText = "Satellite Mode: Active";
                    status.classList.remove('text-amber-500');
                    status.classList.add('text-emerald-500');
                }

                if (window.locWatchId) navigator.geolocation.clearWatch(window.locWatchId);
                window.locWatchId = navigator.geolocation.watchPosition(
                    p => {
                        lastUserLoc = { lat: p.coords.latitude, lng: p.coords.longitude };
                        updateBoardData(p.coords.latitude, p.coords.longitude);
                    },
                    err => console.warn("Watch update failed", err),
                    { enableHighAccuracy: true }
                );

            } catch (err) {
                console.warn("Satellite Handshake Fallback:", err.message);
                let errorMsg = "GPS Signal Fragmented";
                if (err.code === 1) errorMsg = "GPS Permission Denied";
                if (err.message === "METRO_GPS_TIMEOUT" || err.code === 3) errorMsg = "GPS Timeout Happened";

                gpsFallbackMsg = errorMsg + " | Fallback Mode";

                if (status) {
                    status.innerText = `${errorMsg}. Ameerpet Hub Active.`;
                    status.classList.remove('text-emerald-500');
                    status.classList.add('text-amber-500');
                }

                if (userLocText) {
                    userLocText.classList.remove('animate-pulse');
                }

                // Try IP Fallback as a second layer of defense
                try {
                    const ipRes = await fetch('https://ipapi.co/json/');
                    const ipData = await ipRes.json();
                    if (ipData.latitude && ipData.longitude) {
                        lastUserLoc = { lat: ipData.latitude, lng: ipData.longitude };
                        console.log("IP Fallback Success:", lastUserLoc);
                        gpsFallbackMsg = "IP Location Active | Fallback Mode";
                        if (status) status.innerText = "IP Geo-Sync Active. Centering Map.";
                        await updateBoardData(ipData.latitude, ipData.longitude);
                        return;
                    }
                } catch (e) { console.warn("IP Fallback failed", e); }

                // Final fallback to Ameerpet default
                await updateBoardData(17.4334, 78.4484); 
            }
        }

        function fallbackToDefault() {
            const ameerpet = stations.find(s => s.name === 'Ameerpet');
            if (ameerpet) {
                lastUserLoc = { lat: ameerpet.lat, lng: ameerpet.lng };
                updateBoardData(ameerpet.lat, ameerpet.lng, ameerpet.id);
            }
        }

        async function initGeo() {
            const selector = document.getElementById('board-station-selector');
            const interchanges = ['Ameerpet', 'MG Bus Station', 'Parade Ground'];
            const lineIcons = { 'Red': '🔴', 'Blue': '🔵', 'Green': '🟢' };

            stations.slice().sort((a,b)=>a.name.localeCompare(b.name)).forEach(st => {
                const opt = document.createElement('option');
                opt.value = st.id;
                let suffix = '';
                if (interchanges.includes(st.name)) suffix = ' 🔄 [Interchange]';
                opt.innerText = `${lineIcons[st.line]} ${st.name}${suffix}`;
                selector.appendChild(opt);
            });

            performSatelliteHandshake();

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

        function swapPlannerNodes() {
            const start = document.getElementById('start-st');
            const end = document.getElementById('end-st');
            const temp = start.value;
            start.value = end.value;
            end.value = temp;
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
                    <div class="flex flex-col"><span class="text-[9px] font-black uppercase text-slate-400 tracking-widest mb-1">Rush Info</span><span class="text-[13px] font-bold text-slate-700">${data.metrics.it_hub}</span></div>
                `;
                seq.appendChild(metricsDiv);

                // Add Guides (Interchanges)
                if (data.guides && data.guides.length > 0) {
                    const guideHeader = document.createElement('h5');
                    guideHeader.className = "text-[11px] font-black text-blue-600 uppercase tracking-[0.2em] mb-8 mt-12 flex items-center gap-2 pl-2";
                    guideHeader.innerHTML = `<i data-lucide="smartphone" size="14"></i> Critical Interchange Vectors`;
                    seq.appendChild(guideHeader);

                    data.guides.forEach((g, gIdx) => {
                        const outer = document.createElement('div');
                        outer.className = "mb-12";
                        outer.innerHTML = `
                            <div class="phone-frame scale-90 sm:scale-100 !h-[550px] border-[6px]">
                                <div class="phone-notch scale-75"></div>
                                <div class="phone-screen !bg-white">
                                    <div class="phone-ui-header !pt-10 !pb-6">
                                        <div class="flex justify-between items-center mb-4 text-slate-400">
                                            <i data-lucide="wifi" size="12"></i>
                                            <span class="text-[9px] font-black tabular-nums">12:42 PM</span>
                                            <i data-lucide="battery" size="12"></i>
                                        </div>
                                        <div class="flex items-center gap-3">
                                            <div class="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white">
                                                <i data-lucide="shuffle" size="18"></i>
                                            </div>
                                            <div>
                                                <h6 class="text-[16px] font-black text-slate-900 tracking-tight">${g.station}</h6>
                                                <p class="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Transfer Hub</p>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="phone-ui-content">
                                        <div class="bg-indigo-900 text-white p-6 rounded-3xl shadow-xl mb-6 relative overflow-hidden">
                                            <div class="absolute -right-4 -top-4 w-20 h-20 bg-white/5 rounded-full"></div>
                                            <div class="flex items-center gap-2 mb-4">
                                                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                                                <span class="text-[9px] font-black uppercase tracking-widest text-indigo-200">Live Guidance</span>
                                            </div>
                                            <p class="text-sm font-bold leading-relaxed mb-6">${g.text}</p>
                                            <div class="flex items-center justify-between pt-4 border-t border-white/10">
                                                <div class="flex flex-col">
                                                    <span class="text-[8px] font-black uppercase text-indigo-300">Target</span>
                                                    <span class="text-lg font-black tracking-tight">Platform ${g.platform}</span>
                                                </div>
                                                <div class="text-right">
                                                    <span class="text-[8px] font-black uppercase text-indigo-300">ETA</span>
                                                    <p class="text-lg font-black tracking-tight">${g.reaching_at}</p>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="space-y-3">
                                            <p class="text-[9px] font-black text-slate-400 uppercase tracking-widest pl-1">Connecting Services</p>
                                            <div class="grid grid-cols-1 gap-2">
                                                ${(g.connections || []).slice(0, 3).map(c => `
                                                    <div class="p-4 bg-slate-50 border border-slate-100 rounded-2xl flex items-center justify-between">
                                                        <div class="flex items-center gap-3">
                                                            <div class="w-1 h-8 rounded-full ${c.line === 'Red' ? 'bg-red-500' : c.line === 'Blue' ? 'bg-blue-500' : 'bg-green-500'}"></div>
                                                            <div>
                                                                <p class="text-[10px] font-black text-slate-900 uppercase truncate max-w-[120px]">${c.final_stop}</p>
                                                                <p class="text-[8px] font-bold text-slate-400 uppercase">${c.line} Line</p>
                                                            </div>
                                                        </div>
                                                        <span class="text-xs font-black text-slate-600 tabular-nums">${c.arrival_time_12}</span>
                                                    </div>
                                                `).join('')}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                        seq.appendChild(outer);
                    });
                }

                // Path Sequence
                const pathHeader = document.createElement('h5');
                pathHeader.className = "text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-8 mt-12";
                pathHeader.innerText = "Metro Map View";
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

                // Highlight on network map (SVG)
                if (data.sequence) {
                    highlightRouteOnMap(data.sequence);
                }

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
            const interchanges = ['Ameerpet', 'MG Bus Station', 'Parade Ground'];
            const lineIcons = { 'Red': '🔴', 'Blue': '🔵', 'Green': '🟢' };

            ['start-st', 'end-st'].forEach(id => {
                const select = document.getElementById(id);
                if (!select) return;
                select.innerHTML = '<option value="" disabled selected>Select a station...</option>';

                lines.forEach(line => {
                    const group = document.createElement('optgroup');
                    group.label = `${lineIcons[line]} ${line} Line Network`;

                    stations.filter(s => s.line === line).sort((a,b) => a.name.localeCompare(b.name)).forEach(st => {
                        const opt = document.createElement('option');
                        opt.value = st.id;
                        let suffix = '';
                        if (itHubs.some(hub => st.name.includes(hub))) suffix = '  (IT AREA)';
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

                const seenIds = new Set();
                if (data.trains) {
                    data.trains.forEach(t => {
                        seenIds.add(t.trip_id);
                        trainStates.set(t.trip_id, t);
                    });
                }

                // Cleanup vanished trains
                trainStates.forEach((state, id) => {
                    if (!seenIds.has(id)) {
                        trainStates.delete(id);
                    }
                });

            } catch (e) { console.error("Sim Matrix Error:", e); }
        }

        window.onload = () => {
            try {
                initGoogleSync();
                if (window.lucide) {
                    lucide.createIcons();
                }
                updateClock();
                setInterval(updateClock, 1000);
                initGeo(); 
                initPickers(); 
                updateLiveTrains(); 
                loadFeedback(); 
                renderTickets();
                loadSavedVectors();
                if (activeUpdateInterval) clearInterval(activeUpdateInterval);
                activeUpdateInterval = setInterval(updateLiveTrains, 5000); 
            } catch (err) {
                console.error("Critical Boot Error:", err);
            }
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
