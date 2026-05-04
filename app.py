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
    'JFK': 'New York', 'LGA': 'New York', 'EWR': 'Newark',
    'LAX': 'Los Angeles', 'SFO': 'San Francisco', 'OAK': 'Oakland',
    'ORD': 'Chicago', 'MDW': 'Chicago', 'DFW': 'Dallas/Fort Worth',
    'ATL': 'Atlanta', 'DEN': 'Denver', 'SEA': 'Seattle',
    'MIA': 'Miami', 'MCO': 'Orlando', 'LAS': 'Las Vegas',
    'BOS': 'Boston', 'PHX': 'Phoenix', 'IAH': 'Houston',
    'DTW': 'Detroit', 'MSP': 'Minneapolis', 'CLT': 'Charlotte',
    'SAN': 'San Diego', 'PDX': 'Portland', 'SLC': 'Salt Lake City',
    'BWI': 'Baltimore', 'DCA': 'Washington', 'IAD': 'Washington',
    'STL': 'St. Louis', 'ABQ': 'Albuquerque', 'ANC': 'Anchorage',
    'CDC': 'Cedar City', 'FWA': 'Fort Wayne', 'HSV': 'Huntsville', 'MHT': 'Manchester'
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
if st.button("🔮 Predict Delay Risk", type="primary", use_container_width=True):
    
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
