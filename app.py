import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    h1, h2, h3 { color: #004AAD !important; }
    .stButton button {
        background-color: #004AAD !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    .stButton button:hover {
        background-color: #5DE0E6 !important;
        color: #004AAD !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('delay_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

try:
    model, encoders = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ==================== AIRLINE CODES ====================
model_airline_codes = list(encoders['AIRLINE'].classes_)

CODE_TO_NAME = {
    'AA': 'American Airlines', 'AS': 'Alaska Airlines', 'B6': 'JetBlue Airways',
    'DL': 'Delta Air Lines', 'EV': 'ExpressJet', 'F9': 'Frontier Airlines',
    'HA': 'Hawaiian Airlines', 'MQ': 'Envoy Air', 'NK': 'Spirit Airlines',
    'OO': 'SkyWest Airlines', 'UA': 'United Airlines', 'WN': 'Southwest Airlines',
    'G4': 'Allegiant Air', '9E': 'Endeavor Air', 'QX': 'Horizon Air',
    'YV': 'Mesa Airlines', 'PT': 'PSA Airlines', 'YX': 'Republic Airline'
}

def get_airline_display(code):
    name = CODE_TO_NAME.get(code, code)
    return f"{name} ({code})"

def get_airline_code(display_with_code):
    if '(' in display_with_code and ')' in display_with_code:
        return display_with_code.split('(')[-1].split(')')[0]
    return display_with_code

airline_display_options = [get_airline_display(code) for code in model_airline_codes]

# ==================== AIRPORT NAMES ====================
AIRPORT_CITY_MAP = {
    'ABE': 'Allentown', 'ABI': 'Abilene', 'ABQ': 'Albuquerque', 'ABR': 'Aberdeen',
    'ABY': 'Albany', 'ACK': 'Nantucket', 'ACT': 'Waco', 'ACV': 'Arcata/Eureka',
    'ACY': 'Atlantic City', 'ADQ': 'Kodiak', 'AEX': 'Alexandria', 'AGS': 'Augusta',
    'AKN': 'King Salmon', 'ALB': 'Albany', 'ALO': 'Waterloo', 'ALS': 'Alamosa',
    'ALW': 'Walla Walla', 'AMA': 'Amarillo', 'ANC': 'Anchorage', 'APN': 'Alpena',
    'ASE': 'Aspen', 'ATL': 'Atlanta', 'ATW': 'Appleton', 'ATY': 'Watertown',
    'AUS': 'Austin', 'AVL': 'Asheville', 'AVP': 'Scranton/Wilkes-Barre', 'AZA': 'Phoenix-Mesa',
    'AZO': 'Kalamazoo', 'BDL': 'Hartford', 'BET': 'Bethel', 'BFF': 'Scottsbluff',
    'BFL': 'Bakersfield', 'BFM': 'Mobile', 'BGM': 'Binghamton', 'BGR': 'Bangor',
    'BHM': 'Birmingham', 'BIL': 'Billings', 'BIS': 'Bismarck', 'BJI': 'Bemidji',
    'BLI': 'Bellingham', 'BLV': 'Belleville', 'BMI': 'Bloomington', 'BNA': 'Nashville',
    'BOI': 'Boise', 'BOS': 'Boston', 'BPT': 'Beaumont', 'BQK': 'Brunswick',
    'BQN': 'Aguadilla', 'BRD': 'Brainerd', 'BRO': 'Brownsville', 'BRW': 'Barrow',
    'BTM': 'Butte', 'BTR': 'Baton Rouge', 'BTV': 'Burlington', 'BUF': 'Buffalo',
    'BUR': 'Burbank', 'BWI': 'Baltimore', 'BZN': 'Bozeman', 'CAE': 'Columbia',
    'CAK': 'Akron', 'CDB': 'Cold Bay', 'CDC': 'Cedar City', 'CDV': 'Cordova',
    'CGI': 'Cape Girardeau', 'CHA': 'Chattanooga', 'CHO': 'Charlottesville', 'CHS': 'Charleston',
    'CID': 'Cedar Rapids', 'CIU': 'Sault Ste Marie', 'CKB': 'Clarksburg', 'CLE': 'Cleveland',
    'CLL': 'College Station', 'CLT': 'Charlotte', 'CMH': 'Columbus', 'CMI': 'Champaign',
    'CMX': 'Hancock', 'CNY': 'Moab', 'COD': 'Cody', 'COS': 'Colorado Springs',
    'COU': 'Columbia', 'CPR': 'Casper', 'CRP': 'Corpus Christi', 'CRW': 'Charleston',
    'CSG': 'Columbus', 'CVG': 'Cincinnati', 'CWA': 'Mosinee', 'CYS': 'Cheyenne',
    'DAB': 'Daytona Beach', 'DAL': 'Dallas Love', 'DAY': 'Dayton', 'DBQ': 'Dubuque',
    'DCA': 'Washington', 'DDC': 'Dodge City', 'DEC': 'Decatur', 'DEN': 'Denver',
    'DFW': 'Dallas/Fort Worth', 'DHN': 'Dothan', 'DIK': 'Dickinson', 'DLG': 'Dillingham',
    'DLH': 'Duluth', 'DRO': 'Durango', 'DRT': 'Del Rio', 'DSM': 'Des Moines',
    'DTW': 'Detroit', 'DVL': 'Devils Lake', 'EAR': 'Kearney', 'EAT': 'Wenatchee',
    'EAU': 'Eau Claire', 'ECP': 'Panama City', 'EGE': 'Vail', 'EKO': 'Elko',
    'ELM': 'Elmira', 'ELP': 'El Paso', 'ERI': 'Erie', 'ESC': 'Escanaba',
    'EUG': 'Eugene', 'EVV': 'Evansville', 'EWN': 'New Bern', 'EWR': 'Newark',
    'EYW': 'Key West', 'FAI': 'Fairbanks', 'FAR': 'Fargo', 'FAT': 'Fresno',
    'FAY': 'Fayetteville', 'FCA': 'Kalispell', 'FLG': 'Flagstaff', 'FLL': 'Fort Lauderdale',
    'FNT': 'Flint', 'FOD': 'Fort Dodge', 'FSD': 'Sioux Falls', 'FSM': 'Fort Smith',
    'FWA': 'Fort Wayne', 'GCC': 'Gillette', 'GCK': 'Garden City', 'GEG': 'Spokane',
    'GFK': 'Grand Forks', 'GGG': 'Longview', 'GJT': 'Grand Junction', 'GNV': 'Gainesville',
    'GPT': 'Gulfport', 'GRB': 'Green Bay', 'GRI': 'Grand Island', 'GRK': 'Killeen',
    'GRR': 'Grand Rapids', 'GSO': 'Greensboro', 'GSP': 'Greenville', 'GST': 'Gustavus',
    'GTF': 'Great Falls', 'GTR': 'Columbus', 'GUC': 'Gunnison', 'GUM': 'Guam',
    'HDN': 'Hayden', 'HHH': 'Hilton Head', 'HIB': 'Hibbing', 'HLN': 'Helena',
    'HNL': 'Honolulu', 'HOB': 'Hobbs', 'HOU': 'Houston Hobby', 'HPN': 'White Plains',
    'HRL': 'Harlingen', 'HSV': 'Huntsville', 'HTS': 'Huntington', 'HVN': 'New Haven',
    'HYA': 'Hyannis', 'HYS': 'Hays', 'IAD': 'Washington Dulles', 'IAG': 'Niagara Falls',
    'IAH': 'Houston', 'ICT': 'Wichita', 'IDA': 'Idaho Falls', 'ILG': 'Wilmington',
    'ILM': 'Wilmington', 'IMT': 'Iron Mountain', 'IND': 'Indianapolis', 'INL': 'International Falls',
    'ISN': 'Williston', 'ISP': 'Islip', 'ITH': 'Ithaca', 'ITO': 'Hilo',
    'JAC': 'Jackson Hole', 'JAN': 'Jackson', 'JAX': 'Jacksonville', 'JFK': 'New York',
    'JLN': 'Joplin', 'JMS': 'Jamestown', 'JNU': 'Juneau', 'JST': 'Johnstown',
    'KOA': 'Kona', 'KTN': 'Ketchikan', 'LAN': 'Lansing', 'LAR': 'Laramie',
    'LAS': 'Las Vegas', 'LAW': 'Lawton', 'LAX': 'Los Angeles', 'LBB': 'Lubbock',
    'LBE': 'Latrobe', 'LBF': 'North Platte', 'LBL': 'Liberal', 'LCH': 'Lake Charles',
    'LCK': 'Columbus', 'LEX': 'Lexington', 'LFT': 'Lafayette', 'LGA': 'New York',
    'LGB': 'Long Beach', 'LIH': 'Lihue', 'LIT': 'Little Rock', 'LNK': 'Lincoln',
    'LRD': 'Laredo', 'LSE': 'La Crosse', 'LWB': 'Lewisburg', 'LWS': 'Lewiston',
    'LYH': 'Lynchburg', 'MAF': 'Midland', 'MBS': 'Saginaw', 'MCI': 'Kansas City',
    'MCO': 'Orlando', 'MCW': 'Mason City', 'MDT': 'Harrisburg', 'MDW': 'Chicago',
    'MEI': 'Meridian', 'MEM': 'Memphis', 'MFE': 'McAllen', 'MFR': 'Medford',
    'MGM': 'Montgomery', 'MHK': 'Manhattan', 'MHT': 'Manchester', 'MIA': 'Miami',
    'MKE': 'Milwaukee', 'MKG': 'Muskegon', 'MLB': 'Melbourne', 'MLI': 'Moline',
    'MLU': 'Monroe', 'MMH': 'Mammoth Lakes', 'MOB': 'Mobile', 'MOT': 'Minot',
    'MQT': 'Marquette', 'MRY': 'Monterey', 'MSN': 'Madison', 'MSO': 'Missoula',
    'MSP': 'Minneapolis', 'MSY': 'New Orleans', 'MTJ': 'Montrose', 'MVY': 'Martha\'s Vineyard',
    'MYR': 'Myrtle Beach', 'OAJ': 'Jacksonville', 'OAK': 'Oakland', 'OGG': 'Kahului',
    'OGS': 'Ogdensburg', 'OKC': 'Oklahoma City', 'OMA': 'Omaha', 'OME': 'Nome',
    'ONT': 'Ontario', 'ORD': 'Chicago', 'ORF': 'Norfolk', 'ORH': 'Worcester',
    'OTH': 'North Bend', 'OTZ': 'Kotzebue', 'PAE': 'Everett', 'PAH': 'Paducah',
    'PBG': 'Plattsburgh', 'PBI': 'West Palm Beach', 'PDX': 'Portland', 'PGD': 'Punta Gorda',
    'PHF': 'Newport News', 'PHL': 'Philadelphia', 'PHX': 'Phoenix', 'PIA': 'Peoria',
    'PIB': 'Laurel', 'PIE': 'St Petersburg', 'PIH': 'Pocatello', 'PIR': 'Pierre',
    'PIT': 'Pittsburgh', 'PLN': 'Pellston', 'PNS': 'Pensacola', 'PPG': 'Pago Pago',
    'PRC': 'Prescott', 'PSC': 'Pasco', 'PSE': 'Ponce', 'PSG': 'Petersburg',
    'PSM': 'Portsmouth', 'PSP': 'Palm Springs', 'PUB': 'Pueblo', 'PUW': 'Pullman',
    'PVD': 'Providence', 'PVU': 'Provo', 'PWM': 'Portland', 'RAP': 'Rapid City',
    'RDD': 'Redding', 'RDM': 'Redmond', 'RDU': 'Raleigh/Durham', 'RFD': 'Rockford',
    'RHI': 'Rhinelander', 'RIC': 'Richmond', 'RIW': 'Riverton', 'RKS': 'Rock Springs',
    'RNO': 'Reno', 'ROA': 'Roanoke', 'ROC': 'Rochester', 'ROW': 'Roswell',
    'RST': 'Rochester', 'RSW': 'Fort Myers', 'SAF': 'Santa Fe', 'SAN': 'San Diego',
    'SAT': 'San Antonio', 'SAV': 'Savannah', 'SBA': 'Santa Barbara', 'SBN': 'South Bend',
    'SBP': 'San Luis Obispo', 'SCC': 'Deadhorse', 'SCE': 'State College', 'SCK': 'Stockton',
    'SDF': 'Louisville', 'SEA': 'Seattle', 'SFB': 'Orlando Sanford', 'SFO': 'San Francisco',
    'SGF': 'Springfield', 'SGU': 'St George', 'SHD': 'Staunton', 'SHR': 'Sheridan',
    'SHV': 'Shreveport', 'SIT': 'Sitka', 'SJC': 'San Jose', 'SJT': 'San Angelo',
    'SJU': 'San Juan', 'SLC': 'Salt Lake City', 'SLN': 'Salina', 'SMF': 'Sacramento',
    'SMX': 'Santa Maria', 'SNA': 'Santa Ana', 'SPI': 'Springfield', 'SPN': 'Saipan',
    'SPS': 'Wichita Falls', 'SRQ': 'Sarasota', 'STC': 'St Cloud', 'STL': 'St Louis',
    'STS': 'Santa Rosa', 'STT': 'St Thomas', 'STX': 'St Croix', 'SUN': 'Sun Valley',
    'SUX': 'Sioux City', 'SWF': 'Newburgh', 'SWO': 'Stillwater', 'SYR': 'Syracuse',
    'TBN': 'Fort Leonard Wood', 'TLH': 'Tallahassee', 'TOL': 'Toledo', 'TPA': 'Tampa',
    'TRI': 'Bristol', 'TTN': 'Trenton', 'TUL': 'Tulsa', 'TUS': 'Tucson',
    'TVC': 'Traverse City', 'TWF': 'Twin Falls', 'TXK': 'Texarkana', 'TYR': 'Tyler',
    'TYS': 'Knoxville', 'UIN': 'Quincy', 'USA': 'Concord', 'VCT': 'Victoria',
    'VEL': 'Vernal', 'VLD': 'Valdosta', 'VPS': 'Fort Walton Beach', 'WRG': 'Wrangell',
    'WYS': 'West Yellowstone', 'XNA': 'Fayetteville', 'XWA': 'Williston', 'YAK': 'Yakutat',
    'YKM': 'Yakima', 'YUM': 'Yuma'
}

def get_airport_display(code):
    city = AIRPORT_CITY_MAP.get(code, code)
    return f"{city} ({code})"

def get_airport_code(display_name):
    if '(' in display_name and ')' in display_name:
        return display_name.split('(')[-1].split(')')[0]
    return display_name

all_airport_codes = sorted([a for a in encoders['ORIGIN_AIRPORT'].classes_ if len(str(a)) <= 4])
airport_display_names = [get_airport_display(code) for code in all_airport_codes]

# ==================== HELPER FUNCTIONS ====================
def get_day_advice(day_of_week):
    advice = {
        2: "Tuesday - Best day to fly", 3: "Wednesday - Excellent choice",
        4: "Thursday - Good day", 1: "Monday - Average day",
        5: "Friday - Higher delays expected", 6: "Saturday - Weekend crowds",
        7: "Sunday - Busiest day"
    }
    return advice.get(day_of_week, "Average day")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ✈️ Flight Delay Predictor")
    st.markdown("---")
    st.markdown("**Model Statistics**")
    st.metric("Model Accuracy", "89%")
    st.metric("Airlines", len(model_airline_codes))
    st.metric("Airports", len(all_airport_codes))
    st.markdown("---")
    st.markdown("**Best Travel Tips**")
    st.info("📅 Best days: Tuesday, Wednesday")
    st.info("⏰ Best time: 6-8 AM")
    st.warning("⚠️ Worst day: Sunday")

# ==================== MAIN APP ====================
st.title("✈️ Flight Delay Prediction System")
st.markdown("Predicts probability of flight delay (30+ minutes)")

st.markdown("---")
st.subheader("📝 Flight Information")

col1, col2 = st.columns(2)

with col1:
    departure_date = st.date_input(
        "Departure Date",
        min_value=datetime.now().date(),
        max_value=datetime.now().date() + timedelta(days=30)
    )
    month = departure_date.month
    day_of_week = departure_date.weekday() + 1
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    st.caption(f"📅 {day_names[day_of_week-1]}, {departure_date.strftime('%B %d, %Y')}")

with col2:
    dep_hour = st.slider("Departure Time (24-hour)", 0, 23, 9)

col3, col4 = st.columns(2)

with col3:
    origin_display = st.selectbox("Origin Airport", airport_display_names)
    origin_code = get_airport_code(origin_display)

with col4:
    destination_display = st.selectbox("Destination Airport", airport_display_names)
    destination_code = get_airport_code(destination_display)

# Airline selection
airline_display_selected = st.selectbox("Airline", airline_display_options)
airline_code = get_airline_code(airline_display_selected)

# Day Advice
day_advice = get_day_advice(day_of_week)
if day_of_week in [2, 3]:
    st.success(f"💡 Tip: {day_advice}")
elif day_of_week == 7:
    st.error(f"💡 Tip: {day_advice}")
else:
    st.info(f"💡 Tip: {day_advice}")

# Predict Button
if st.button(" Predict Delay Risk", type="primary", use_container_width=True):
    
    try:
        # Encode inputs
        airline_encoded = encoders['AIRLINE'].transform([airline_code])[0]
        origin_encoded = encoders['ORIGIN_AIRPORT'].transform([origin_code])[0]
        dest_encoded = encoders['DESTINATION_AIRPORT'].transform([destination_code])[0]
        
        # Model prediction
        input_data = pd.DataFrame([[
            month, day_of_week, airline_encoded, origin_encoded, dest_encoded
        ]], columns=['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
        
        base_risk = model.predict_proba(input_data)[0][1] * 100
        
        # Time of day adjustment
        if dep_hour < 8:
            hour_adjustment = -10
        elif dep_hour > 18:
            hour_adjustment = 15
        elif dep_hour > 12:
            hour_adjustment = 5
        else:
            hour_adjustment = 0
        
        final_risk = base_risk + hour_adjustment
        final_risk = max(0, min(100, final_risk))
        
        st.markdown("---")
        st.subheader("📊 Delay Risk Assessment")
        
        col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
        with col_g2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_risk,
                title={"text": "Probability of Delay (30+ minutes)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#004AAD"},
                    'steps': [
                        {'range': [0, 30], 'color': "#27ae60"},
                        {'range': [30, 60], 'color': "#f39c12"},
                        {'range': [60, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 2}, 'value': 50}
                }
            ))
            fig.update_layout(height=320, margin=dict(t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        if final_risk < 30:
            st.success(f"✅ Low Risk ({final_risk:.0f}%) - Flight likely on time")
        elif final_risk < 60:
            st.warning(f"⚠️ Medium Risk ({final_risk:.0f}%) - Moderate delay chance")
        else:
            st.error(f"🔴 High Risk ({final_risk:.0f}%) - Significant delay probability")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        recs = []
        if dep_hour < 8:
            recs.append("Early morning departure - best time to fly")
        elif dep_hour > 16:
            recs.append("Evening departure - consider morning flight")
        if day_of_week == 7:
            recs.append("Sunday - busiest day, allow extra time")
        elif day_of_week in [2, 3]:
            recs.append("Midweek travel - excellent choice")
        if not recs:
            recs.append("Good flight selection - low risk")
        for rec in recs:
            st.write(f"• {rec}")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption(f"Model: Random Forest | Accuracy: 89% | Based on US DOT 2019-2023 data")
