import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import requests
import time

# Page configuration
st.set_page_config(
    page_title="AI Delivery Availability Predictor — Created by Braj Patel",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .prediction-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    .prediction-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }

    /* Floating clock (top-right) */
    .floating-clock {
        position: fixed;
        top: 18px;
        right: 18px;
        z-index: 9999;
        background: linear-gradient(135deg, rgba(31,119,180,0.95), rgba(31,119,180,0.75));
        color: #fff;
        padding: 10px 14px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(6px);
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji';
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .floating-clock .clock-icon {
        font-size: 18px;
        line-height: 1;
    }
    .floating-clock .clock-time {
        font-weight: 700;
        letter-spacing: 0.4px;
        font-size: 14px;
        white-space: nowrap;
    }
    .floating-clock .clock-date {
        font-size: 12px;
        opacity: 0.95;
        white-space: nowrap;
    }
    @media (max-width: 768px) {
        .floating-clock { top: 12px; right: 12px; padding: 8px 10px; }
        .floating-clock .clock-time { font-size: 13px; }
        .floating-clock .clock-date { font-size: 11px; }
    }
</style>
""", unsafe_allow_html=True)

# Clock renderer (theme-aware, optional timezone)
def render_clock(show: bool, timezone: str = ""):
    if not show:
        return
    # Add dark-mode adjustments
    st.markdown(
        """
        <style>
        @media (prefers-color-scheme: dark) {
            .floating-clock { background: linear-gradient(135deg, rgba(31,119,180,0.75), rgba(31,119,180,0.55)); border-color: rgba(255,255,255,0.2); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tz = timezone or ""
    js_tz = ("'" + tz.replace("'", "\\'") + "'") if tz else "''"
    template = """
        <div class="floating-clock">
          <span class="clock-icon">🕒</span>
          <div>
            <div class="clock-time" id="clock-time">--:--:--</div>
            <div class="clock-date" id="clock-date">--, -- --- ----</div>
          </div>
        </div>
        <script>
          const _tz = {TZ_JS};
          function updateClock() {{
            const now = new Date();
            const optsTime = {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }};
            const optsDate = {{ weekday: 'short', year: 'numeric', month: 'short', day: '2-digit' }};
            if (_tz) {{ optsTime.timeZone = _tz; optsDate.timeZone = _tz; }}
            let time, date;
            try {{
              time = now.toLocaleTimeString([], optsTime);
              date = now.toLocaleDateString([], optsDate);
            }} catch (e) {{
              // Fallback without timezone
              time = now.toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }});
              date = now.toLocaleDateString([], {{ weekday: 'short', year: 'numeric', month: 'short', day: '2-digit' }});
            }}
            const t = document.getElementById('clock-time');
            const d = document.getElementById('clock-date');
            if (t && d) {{ t.textContent = time; d.textContent = date; }}
          }}
          updateClock();
          setInterval(updateClock, 1000);
        </script>
    """
    st.markdown(template.replace("{TZ_JS}", js_tz), unsafe_allow_html=True)

# Safe secrets access (handles missing secrets.toml gracefully)
def _get_secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

@st.cache_data(ttl=3600)
def load_data():
    """Load and cache the datasets"""
    try:
        customers_df = pd.read_csv('customers.csv')
        delivery_df = pd.read_csv('delivery_history.csv')
        calendar_df = pd.read_csv('calendar_data.csv')
        return customers_df, delivery_df, calendar_df
    except FileNotFoundError:
        st.error("Dataset files not found. Please run generate_dataset.py first.")
        return None, None, None

# ----------------------------
# Weather utilities
# ----------------------------
def _map_openweather_to_category(main: str) -> str:
    """Map OpenWeather 'main' field to our app's categories."""
    if not main:
        return 'sunny'
    main = main.lower()
    if main in ['clear']:
        return 'sunny'
    if main in ['clouds', 'mist', 'haze', 'fog', 'smoke', 'dust']:
        return 'cloudy'
    if main in ['rain', 'drizzle', 'thunderstorm']:
        return 'rainy'
    if main in ['snow', 'sleet']:
        return 'snowy'
    return 'cloudy'

@st.cache_data(show_spinner=False)
def fetch_city_candidates(query: str, api_key: str, cache_bucket: int, limit: int = 5):
    """Search city candidates using OpenWeather Geocoding API.

    Returns a list of dicts: {name, state, country, lat, lon}
    """
    try:
        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {"q": query, "limit": limit, "appid": api_key}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json() or []
        result = []
        for item in data:
            result.append({
                "name": item.get("name"),
                "state": item.get("state"),
                "country": item.get("country"),
                "lat": item.get("lat"),
                "lon": item.get("lon"),
            })
        return result
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def fetch_live_weather_city(city_query: str, api_key: str, cache_bucket: int):
    """Fetch current weather by city name from OpenWeather.

    Returns: (category:str, details:dict) or (None, error:str)
    """
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_query, "appid": api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        main = (data.get('weather') or [{}])[0].get('main', '')
        category = _map_openweather_to_category(main)
        return category, {
            "source": "openweather",
            "mode": "city",
            "main": main,
            "category": category,
            "temp_c": (data.get('main') or {}).get('temp'),
            "city": data.get('name'),
            "country": (data.get('sys') or {}).get('country'),
        }
    except Exception as e:
        return None, str(e)

# WeatherAPI fallback
def _map_weatherapi_to_category(text: str) -> str:
    if not text:
        return 'sunny'
    t = text.lower()
    if 'snow' in t or 'sleet' in t or 'blizzard' in t:
        return 'snowy'
    if 'rain' in t or 'drizzle' in t or 'shower' in t or 'thunder' in t:
        return 'rainy'
    if 'cloud' in t or 'overcast' in t or 'mist' in t or 'fog' in t or 'haze' in t:
        return 'cloudy'
    if 'clear' in t or 'sunny' in t:
        return 'sunny'
    return 'cloudy'

@st.cache_data(show_spinner=False)
def fetch_weatherapi_city(city_query: str, api_key: str, cache_bucket: int):
    try:
        url = "https://api.weatherapi.com/v1/current.json"
        params = {"key": api_key, "q": city_query, "aqi": "no"}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        text = ((data.get('current') or {}).get('condition') or {}).get('text', '')
        category = _map_weatherapi_to_category(text)
        loc = data.get('location') or {}
        return category, {
            "source": "weatherapi",
            "mode": "city",
            "main": text,
            "category": category,
            "temp_c": (data.get('current') or {}).get('temp_c'),
            "city": loc.get('name'),
            "country": loc.get('country'),
        }
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def fetch_weatherapi_coords(lat: float, lon: float, api_key: str, cache_bucket: int):
    try:
        url = "https://api.weatherapi.com/v1/current.json"
        params = {"key": api_key, "q": f"{lat},{lon}", "aqi": "no"}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        text = ((data.get('current') or {}).get('condition') or {}).get('text', '')
        category = _map_weatherapi_to_category(text)
        loc = data.get('location') or {}
        return category, {
            "source": "weatherapi",
            "mode": "coords",
            "main": text,
            "category": category,
            "temp_c": (data.get('current') or {}).get('temp_c'),
            "city": loc.get('name'),
            "country": loc.get('country'),
            "lat": lat,
            "lon": lon,
        }
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def fetch_live_weather_coords(lat: float, lon: float, api_key: str, cache_bucket: int):
    """Fetch current weather by coordinates from OpenWeather.

    Returns: (category:str, details:dict) or (None, error:str)
    """
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        main = (data.get('weather') or [{}])[0].get('main', '')
        category = _map_openweather_to_category(main)
        return category, {
            "source": "openweather",
            "mode": "coords",
            "main": main,
            "category": category,
            "temp_c": (data.get('main') or {}).get('temp'),
            "city": data.get('name'),
            "country": (data.get('sys') or {}).get('country'),
            "lat": lat,
            "lon": lon,
        }
    except Exception as e:
        return None, str(e)

@st.cache_resource
def train_model():
    """Train and cache the ML model"""
    customers_df, delivery_df, calendar_df = load_data()
    if customers_df is None:
        return None, None, None
    
    # Merge datasets
    merged_df = delivery_df.merge(customers_df, on='customer_id', how='left')
    
    # Feature engineering
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6])
    merged_df['hour'] = merged_df['time_window_start'].str.split(':').str[0].astype(int)
    
    # Select features
    feature_columns = [
        'day_of_week', 'month', 'hour', 'age', 'household_size',
        'delivery_attempt_number', 'package_value', 'prev_success_rate'
    ]
    
    categorical_features = ['profile_type', 'location_type', 'time_period', 'weather_condition']
    
    # Encode categorical features
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        merged_df[f'{feature}_encoded'] = le.fit_transform(merged_df[feature].astype(str))
        label_encoders[feature] = le
        feature_columns.append(f'{feature}_encoded')
    
    # Add boolean features
    boolean_features = ['has_pets', 'work_from_home', 'is_holiday', 'requires_signature', 'is_weekend']
    feature_columns.extend(boolean_features)
    
    # Prepare data
    X = merged_df[feature_columns].fillna(0)
    y = merged_df['was_home']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, label_encoders, feature_columns

def predict_availability(model, label_encoders, feature_columns, customer_data, day_of_week, hour, weather='sunny'):
    """Make availability prediction"""
    if model is None:
        return None
    
    # Create prediction data
    pred_data = {
        'day_of_week': day_of_week,
        'month': datetime.now().month,
        'hour': hour,
        'age': customer_data['age'],
        'household_size': customer_data['household_size'],
        'delivery_attempt_number': 1,
        'package_value': 50.0,
        'prev_success_rate': 0.7,
        'has_pets': customer_data['has_pets'],
        'work_from_home': customer_data['work_from_home'],
        'is_holiday': False,
        'requires_signature': False,
        'is_weekend': day_of_week >= 5,
    }
    
    # Determine time period
    if 6 <= hour < 9:
        time_period = 'morning'
    elif 9 <= hour < 12:
        time_period = 'late_morning'
    elif 12 <= hour < 15:
        time_period = 'afternoon'
    elif 15 <= hour < 18:
        time_period = 'late_afternoon'
    elif 18 <= hour < 21:
        time_period = 'evening'
    else:
        time_period = 'night'
    
    # Encode categorical features
    categorical_values = {
        'profile_type': customer_data['profile_type'],
        'location_type': customer_data['location_type'],
        'time_period': time_period,
        'weather_condition': weather
    }
    
    for feature, value in categorical_values.items():
        if feature in label_encoders:
            try:
                pred_data[f'{feature}_encoded'] = label_encoders[feature].transform([value])[0]
            except:
                pred_data[f'{feature}_encoded'] = 0
    
    # Create feature vector
    X_pred = []
    for feature in feature_columns:
        X_pred.append(pred_data.get(feature, 0))
    
    # Make prediction
    X_pred = np.array(X_pred).reshape(1, -1)
    probability = model.predict_proba(X_pred)[0, 1]
    prediction = model.predict(X_pred)[0]
    
    return {
        'probability': probability,
        'prediction': prediction,
        'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.15 else 'Low'
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">📦 AI Delivery Availability Predictor — Created by Braj Patel</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and model
    customers_df, delivery_df, calendar_df = load_data()
    model, label_encoders, feature_columns = train_model()
    
    if customers_df is None or model is None:
        st.error("Failed to load data or train model. Please check your files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏱️ Clock Settings")
    show_clock = st.sidebar.checkbox("Show Clock", value=True)
    tz_choice = st.sidebar.selectbox(
        "Timezone",
        [
            "Local (browser)",
            "Europe/London",
            "UTC",
            "US/Eastern",
            "US/Central",
            "US/Mountain",
            "US/Pacific",
            "Asia/Kolkata",
            "Asia/Dubai",
            "Asia/Tokyo",
            "Australia/Sydney",
        ],
        index=1,
    )
    render_clock(show_clock, "" if tz_choice.startswith("Local") else tz_choice)
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔮 Prediction Tool", "📊 Dataset Overview", "📈 Analytics Dashboard", "💡 Business Insights"]
    )
    
    if page == "🔮 Prediction Tool":
        prediction_page(model, label_encoders, feature_columns, customers_df)
    elif page == "📊 Dataset Overview":
        dataset_overview_page(customers_df, delivery_df, calendar_df)
    elif page == "📈 Analytics Dashboard":
        analytics_dashboard_page(delivery_df, customers_df)
    elif page == "💡 Business Insights":
        business_insights_page(delivery_df)

def prediction_page(model, label_encoders, feature_columns, customers_df):
    st.header("🔮 Customer Availability Prediction")
    st.markdown("Predict the likelihood of a customer being available for delivery at a specific time.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Customer Information")
        
        # Customer selection
        customer_option = st.radio(
            "Choose customer input method:",
            ["Select existing customer", "Create custom customer"]
        )
        
        customer_lat = None
        customer_lon = None
        if customer_option == "Select existing customer":
            customer_id = st.selectbox(
                "Select Customer ID:",
                customers_df['customer_id'].tolist()
            )
            selected_row = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
            customer_data = selected_row.to_dict()
            # capture coordinates if present
            customer_lat = float(selected_row.get('latitude')) if 'latitude' in selected_row else None
            customer_lon = float(selected_row.get('longitude')) if 'longitude' in selected_row else None
        else:
            customer_data = {
                'customer_id': 9999,
                'profile_type': st.selectbox("Profile Type:", 
                    ['working_professional', 'stay_at_home', 'student', 'retiree', 'shift_worker']),
                'age': st.slider("Age:", 18, 80, 35),
                'location_type': st.selectbox("Location Type:", ['urban', 'suburban', 'rural']),
                'household_size': st.slider("Household Size:", 1, 5, 2),
                'has_pets': st.checkbox("Has Pets"),
                'work_from_home': st.checkbox("Works from Home")
            }
        
        # Display customer info
        st.markdown("**Customer Profile:**")
        st.write(f"- **Profile:** {customer_data['profile_type'].replace('_', ' ').title()}")
        st.write(f"- **Age:** {customer_data['age']}")
        st.write(f"- **Location:** {customer_data.get('city', '')} {(', ' + customer_data.get('region')) if customer_data.get('region') else ''}")
        st.write(f"- **Area Type:** {customer_data['location_type'].title()}")
        st.write(f"- **Household Size:** {customer_data['household_size']}")
        st.write(f"- **Has Pets:** {'Yes' if customer_data['has_pets'] else 'No'}")
        st.write(f"- **Work from Home:** {'Yes' if customer_data['work_from_home'] else 'No'}")

        # Mini map of selected customer's location (if coordinates available)
        if customer_lat is not None and customer_lon is not None:
            mm_df = pd.DataFrame({
                'lat': [customer_lat],
                'lon': [customer_lon],
                'label': [customer_data.get('city', 'Customer Location')]
            })
            fig_loc = px.scatter_mapbox(mm_df, lat='lat', lon='lon', hover_name='label', zoom=9)
            fig_loc.update_layout(mapbox_style='open-street-map', height=240, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_loc, use_container_width=True)
    
    with col2:
        st.subheader("📅 Delivery Details")
        
        # Delivery time selection
        delivery_date = st.date_input(
            "Delivery Date:",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime.now().date()
        )
        
        delivery_time = st.time_input(
            "Delivery Time:",
            value=datetime.strptime("14:00", "%H:%M").time()
        )
        
        # Weather selection: allow live weather from OpenWeather if key provided
        use_live_weather = st.checkbox("Use live weather (OpenWeather/WeatherAPI)", value=True)
        openweather_key = _get_secret("OPENWEATHER_API_KEY")
        weatherapi_key = _get_secret("WEATHERAPI_KEY")
        try:
            ttl = int(_get_secret("WEATHER_CACHE_TTL", 600))
        except Exception:
            ttl = 600
        cache_bucket = int(time.time() // max(60, ttl))
        weather_details = None
        weather = None  # ensure defined
        if use_live_weather and (openweather_key or weatherapi_key):
            # If an existing customer has coordinates, offer to auto-use them
            use_customer_coords = False
            if customer_lat is not None and customer_lon is not None:
                use_customer_coords = st.checkbox(
                    "Use selected customer's location for weather",
                    value=True
                )
            if use_customer_coords:
                with st.spinner("Fetching current weather for customer's location..."):
                    mapped, info = (None, None)
                    if openweather_key:
                        mapped, info = fetch_live_weather_coords(customer_lat, customer_lon, openweather_key, cache_bucket)
                    if not mapped and weatherapi_key:
                        mapped, info = fetch_weatherapi_coords(customer_lat, customer_lon, weatherapi_key, cache_bucket)
                if mapped:
                    weather = mapped
                    weather_details = info
                    loc_txt = f"{customer_data.get('city', 'Customer Location')}, {customer_data.get('region', '')}".strip(', ')
                    st.info(f"Live weather ({info.get('source')}): {info['main']} | Category: {mapped} | Temp: {info.get('temp_c')}°C at {loc_txt}")
                else:
                    st.warning(f"Could not fetch live weather ({info}). Falling back to manual/other methods.")
            
            method = st.radio("Weather location source:", ["City", "Coordinates"], horizontal=True)
            if method == "City":
                city_query = st.text_input("Search city (e.g., London or London,UK)", value="London")
                selected_city = None
                if city_query.strip() and openweather_key:
                    with st.spinner("Searching cities..."):
                        candidates = fetch_city_candidates(city_query.strip(), openweather_key, cache_bucket)
                    if candidates:
                        labels = [
                            f"{c['name']}{', ' + c['state'] if c.get('state') else ''}, {c.get('country')} (lat {c.get('lat'):.2f}, lon {c.get('lon'):.2f})"
                            for c in candidates
                        ]
                        idx = st.selectbox("Select a city:", options=list(range(len(candidates))), format_func=lambda i: labels[i])
                        selected_city = candidates[idx]
                    else:
                        st.warning("No matching cities found. Try refining your search.")
                elif city_query.strip() and not openweather_key:
                    st.info("City search uses OpenWeather Geocoding. Add OPENWEATHER_API_KEY to use search. Falling back to manual entry.")

                if selected_city and (openweather_key or weatherapi_key):
                    with st.spinner("Fetching current weather..."):
                        mapped, info = (None, None)
                        # Prefer coordinate-based fetch for precision
                        if openweather_key:
                            mapped, info = fetch_live_weather_coords(selected_city['lat'], selected_city['lon'], openweather_key, cache_bucket)
                        if not mapped and weatherapi_key:
                            mapped, info = fetch_weatherapi_coords(selected_city['lat'], selected_city['lon'], weatherapi_key, cache_bucket)
                    if mapped:
                        weather = mapped
                        weather_details = info
                        loc_txt = f"{selected_city['name']}{', ' + selected_city['state'] if selected_city.get('state') else ''}, {selected_city.get('country')}"
                        st.info(f"Live weather ({info.get('source')}): {info['main']} | Category: {mapped} | Temp: {info.get('temp_c')}°C in {loc_txt}")
                    else:
                        st.warning(f"Could not fetch live weather ({info}). Falling back to manual selection.")
                        weather = st.selectbox(
                            "Weather Condition:",
                            ['sunny', 'cloudy', 'rainy', 'snowy']
                        )
                else:
                    # No city selected/chosen; provide manual selector
                    weather = st.selectbox(
                        "Weather Condition:",
                        ['sunny', 'cloudy', 'rainy', 'snowy']
                    )
            else:
                col_lat, col_lon = st.columns(2)
                with col_lat:
                    lat = st.number_input("Latitude", value=51.5072, format="%.6f")
                with col_lon:
                    lon = st.number_input("Longitude", value=-0.1276, format="%.6f")
                with st.spinner("Fetching current weather..."):
                    mapped, info = (None, None)
                    if openweather_key:
                        mapped, info = fetch_live_weather_coords(lat, lon, openweather_key, cache_bucket)
                    if not mapped and weatherapi_key:
                        mapped, info = fetch_weatherapi_coords(lat, lon, weatherapi_key, cache_bucket)
                if mapped:
                    weather = mapped
                    weather_details = info
                    loc_txt = f"{info.get('city')}, {info.get('country')}" if info.get('city') else f"{lat:.4f}, {lon:.4f}"
                    st.info(f"Live weather ({info.get('source')}): {info['main']} | Category: {mapped} | Temp: {info.get('temp_c')}°C at {loc_txt}")
                else:
                    st.warning(f"Could not fetch live weather ({info}). Falling back to manual selection.")
                    weather = st.selectbox(
                        "Weather Condition:",
                        ['sunny', 'cloudy', 'rainy', 'snowy']
                    )
        else:
            if use_live_weather and not (openweather_key or weatherapi_key):
                st.warning("No weather API key set (OPENWEATHER_API_KEY or WEATHERAPI_KEY) in Streamlit Secrets. Using manual weather selection.")
            weather = st.selectbox(
                "Weather Condition:",
                ['sunny', 'cloudy', 'rainy', 'snowy']
            )

        # Final safety net: if weather still not set (shouldn't happen), default to manual selector
        if weather is None:
            weather = st.selectbox(
                "Weather Condition:",
                ['sunny', 'cloudy', 'rainy', 'snowy']
            )
        
        # Convert to required format
        day_of_week = delivery_date.weekday()
        hour = delivery_time.hour
        
        # Make prediction
        if st.button("🎯 Predict Availability", type="primary"):
            result = predict_availability(
                model, label_encoders, feature_columns, 
                customer_data, day_of_week, hour, weather
            )
            
            if result:
                probability = result['probability']
                prediction = result['prediction']
                confidence = result['confidence']
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Availability Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Result interpretation
                if probability >= 0.8:
                    st.markdown(f"""
                    <div class="prediction-success">
                        <h4>✅ High Availability ({probability:.1%})</h4>
                        <p>Customer is very likely to be available for delivery at this time.</p>
                        <p><strong>Recommendation:</strong> Schedule the delivery - high success probability!</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif probability >= 0.6:
                    st.markdown(f"""
                    <div class="prediction-warning">
                        <h4>⚠️ Moderate Availability ({probability:.1%})</h4>
                        <p>Customer has moderate likelihood of being available.</p>
                        <p><strong>Recommendation:</strong> Consider alternative time slots or contact customer.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-danger">
                        <h4>❌ Low Availability ({probability:.1%})</h4>
                        <p>Customer is unlikely to be available at this time.</p>
                        <p><strong>Recommendation:</strong> Choose a different time slot for better success rate.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Probability", f"{probability:.1%}")
                with col2:
                    st.metric("Confidence", confidence)
                with col3:
                    day_name = delivery_date.strftime("%A")
                    st.metric("Day", day_name)

def dataset_overview_page(customers_df, delivery_df, calendar_df):
    st.header("📊 Dataset Overview")
    
    # Filters
    with st.expander("Filters", expanded=True):
        regions = sorted([r for r in customers_df.get('region', pd.Series()).dropna().unique()]) if 'region' in customers_df.columns else []
        cities = sorted([c for c in customers_df.get('city', pd.Series()).dropna().unique()]) if 'city' in customers_df.columns else []
        sel_regions = st.multiselect("Region(s)", regions, default=regions[:5] if regions else [])
        sel_cities = st.multiselect("City/Cities", cities, default=[])
    # Apply filters
    cust_f = customers_df.copy()
    if sel_regions:
        cust_f = cust_f[cust_f['region'].isin(sel_regions)]
    if sel_cities:
        cust_f = cust_f[cust_f['city'].isin(sel_cities)]
    # Filter deliveries and calendar by selected customers
    deliv_f = delivery_df[delivery_df['customer_id'].isin(cust_f['customer_id'])]
    cal_f = calendar_df[calendar_df['customer_id'].isin(cust_f['customer_id'])]

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(cust_f):,}")
    with col2:
        st.metric("Delivery Records", f"{len(deliv_f):,}")
    with col3:
        st.metric("Calendar Events", f"{len(cal_f):,}")
    with col4:
        st.metric("Success Rate", f"{deliv_f['was_home'].mean():.1%}")
    
    st.markdown("---")
    
    # Customer profiles distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Customer Profile Distribution")
        profile_counts = cust_f['profile_type'].value_counts()
        fig = px.pie(
            values=profile_counts.values,
            names=[name.replace('_', ' ').title() for name in profile_counts.index],
            title="Customer Profiles"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏠 Location Distribution")
        location_counts = cust_f['location_type'].value_counts()
        fig = px.bar(
            x=[name.title() for name in location_counts.index],
            y=location_counts.values,
            title="Customer Locations"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.subheader("📋 Sample Data")
    tab1, tab2, tab3 = st.tabs(["Customers", "Deliveries", "Calendar"])
    
    with tab1:
        st.dataframe(cust_f.head(10))
    
    with tab2:
        st.dataframe(deliv_f.head(10))
    
    with tab3:
        st.dataframe(cal_f.head(10))

def analytics_dashboard_page(delivery_df, customers_df):
    st.header("📈 Analytics Dashboard")
    # Filters
    with st.expander("Filters", expanded=True):
        regions = sorted([r for r in customers_df.get('region', pd.Series()).dropna().unique()]) if 'region' in customers_df.columns else []
        cities = sorted([c for c in customers_df.get('city', pd.Series()).dropna().unique()]) if 'city' in customers_df.columns else []
        sel_regions = st.multiselect("Region(s)", regions, default=regions[:5] if regions else [])
        sel_cities = st.multiselect("City/Cities", cities, default=[])
    cust_f = customers_df.copy()
    if sel_regions:
        cust_f = cust_f[cust_f['region'].isin(sel_regions)]
    if sel_cities:
        cust_f = cust_f[cust_f['city'].isin(sel_cities)]
    deliv_f = delivery_df.merge(cust_f[['customer_id']], on='customer_id', how='inner')
    
    # Success rates by time period
    st.subheader("⏰ Success Rates by Time Period")
    time_success = deliv_f.groupby('time_period')['was_home'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=[period.replace('_', ' ').title() for period in time_success.index],
        y=time_success.values,
        title="Delivery Success Rate by Time Period",
        labels={'y': 'Success Rate', 'x': 'Time Period'}
    )
    fig.update_traces(marker_color='lightblue')
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekend vs Weekday analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Weekend vs Weekday")
        deliv_f['is_weekend'] = deliv_f['day_of_week'].isin([5, 6])
        weekend_success = deliv_f.groupby('is_weekend')['was_home'].mean()
        
        fig = px.bar(
            x=['Weekday', 'Weekend'],
            y=[weekend_success[False], weekend_success[True]],
            title="Success Rate: Weekend vs Weekday"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌤️ Weather Impact")
        weather_success = deliv_f.groupby('weather_condition')['was_home'].mean()
        
        fig = px.bar(
            x=[weather.title() for weather in weather_success.index],
            y=weather_success.values,
            title="Success Rate by Weather"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap of success rates
    st.subheader("🔥 Success Rate Heatmap (Day vs Hour)")
    
    # Create hour column
    deliv_f['hour'] = deliv_f['time_window_start'].str.split(':').str[0].astype(int)
    
    # Create pivot table
    heatmap_data = deliv_f.groupby(['day_of_week', 'hour'])['was_home'].mean().unstack(fill_value=0)
    
    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        title="Success Rate Heatmap (Day of Week vs Hour)",
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

    # UK map of regional success (uses customers' coordinates and regions)
    st.subheader("🗺️ UK Regional Delivery Success Map")
    try:
        # Merge delivery success with customer location/region
        merged = deliv_f.merge(customers_df[['customer_id', 'region', 'latitude', 'longitude']], on='customer_id', how='left')
        region_stats = merged.groupby('region').agg(
            success_rate=('was_home', 'mean'),
            lat=('latitude', 'mean'),
            lon=('longitude', 'mean'),
            deliveries=('was_home', 'count')
        ).reset_index().dropna(subset=['lat', 'lon'])

        fig_map = px.scatter_mapbox(
            region_stats,
            lat='lat', lon='lon',
            size='deliveries',
            color='success_rate',
            color_continuous_scale='RdYlGn',
            size_max=30,
            zoom=4.5,
            hover_name='region',
            hover_data={'success_rate': ':.1%', 'deliveries': True, 'lat': False, 'lon': False},
            title="Regional Delivery Success (click points)"
        )
        fig_map.update_layout(mapbox_style='open-street-map', height=500)
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render UK map: {e}")

def business_insights_page(delivery_df):
    st.header("💡 Business Insights & Recommendations")
    
    # Key insights
    st.subheader("🎯 Key Findings")
    
    # Calculate insights
    time_success = delivery_df.groupby('time_period')['was_home'].mean().sort_values(ascending=False)
    delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6])
    weekend_success = delivery_df[delivery_df['is_weekend']]['was_home'].mean()
    weekday_success = delivery_df[~delivery_df['is_weekend']]['was_home'].mean()
    weather_success = delivery_df.groupby('weather_condition')['was_home'].mean().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Performance Metrics")
        st.metric("Best Time Period", time_success.index[0].replace('_', ' ').title(), f"{time_success.iloc[0]:.1%}")
        st.metric("Weekend Advantage", f"+{(weekend_success - weekday_success):.1%}", "vs Weekdays")
        st.metric("Best Weather", weather_success.index[0].title(), f"{weather_success.iloc[0]:.1%}")
    
    with col2:
        st.markdown("### 🎯 Optimization Opportunities")
        worst_time = time_success.index[-1]
        improvement_potential = time_success.iloc[0] - time_success.iloc[-1]
        st.metric("Improvement Potential", f"+{improvement_potential:.1%}", f"vs {worst_time.replace('_', ' ')}")
        
        total_deliveries = len(delivery_df)
        failed_deliveries = len(delivery_df[~delivery_df['was_home']])
        st.metric("Failed Deliveries", f"{failed_deliveries:,}", f"{(failed_deliveries/total_deliveries):.1%} of total")
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("🚀 Strategic Recommendations")
    
    recommendations = [
        {
            "title": "🕐 Optimize Delivery Time Slots",
            "description": f"Focus on {time_success.index[0].replace('_', ' ')} deliveries which have {time_success.iloc[0]:.1%} success rate",
            "impact": "High",
            "effort": "Low"
        },
        {
            "title": "📅 Weekend Delivery Program",
            "description": f"Expand weekend deliveries - they have {(weekend_success - weekday_success):.1%} higher success rate",
            "impact": "High",
            "effort": "Medium"
        },
        {
            "title": "🎯 Customer Profile Targeting",
            "description": "Personalize delivery times based on customer profiles (working professionals prefer evenings)",
            "impact": "Medium",
            "effort": "Medium"
        },
        {
            "title": "🌤️ Weather-Adaptive Scheduling",
            "description": "Adjust delivery schedules based on weather forecasts to optimize success rates",
            "impact": "Low",
            "effort": "Low"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"{i}. {rec['title']}"):
            st.write(rec['description'])
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Impact:** {rec['impact']}")
            with col2:
                st.write(f"**Effort:** {rec['effort']}")
    
    # ROI Calculator
    st.markdown("---")
    st.subheader("💰 ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Metrics:**")
        current_success_rate = delivery_df['was_home'].mean()
        total_deliveries_per_month = st.number_input("Monthly Deliveries:", value=10000, step=1000)
        cost_per_failed_delivery = st.number_input("Cost per Failed Delivery ($):", value=15.0, step=1.0)
    
    with col2:
        st.markdown("**Projected Improvement:**")
        improvement_percentage = st.slider("Expected Success Rate Improvement (%):", 1, 20, 10)
        
        # Calculate ROI
        current_failures = total_deliveries_per_month * (1 - current_success_rate)
        new_success_rate = min(1.0, current_success_rate + (improvement_percentage / 100))
        new_failures = total_deliveries_per_month * (1 - new_success_rate)
        
        monthly_savings = (current_failures - new_failures) * cost_per_failed_delivery
        annual_savings = monthly_savings * 12
        
        st.metric("Monthly Savings", f"${monthly_savings:,.0f}")
        st.metric("Annual Savings", f"${annual_savings:,.0f}")
        st.metric("New Success Rate", f"{new_success_rate:.1%}")

if __name__ == "__main__":
    main()
