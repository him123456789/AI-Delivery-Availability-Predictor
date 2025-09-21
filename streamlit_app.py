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
</style>
""", unsafe_allow_html=True)

@st.cache_data
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
def fetch_live_weather(city_query: str, api_key: str):
    """Fetch current weather from OpenWeather and return mapped category and raw payload.

    Returns tuple: (category:str, details:dict) or (None, error:str)
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
            "main": main,
            "category": category,
            "temp_c": (data.get('main') or {}).get('temp'),
            "city": data.get('name'),
            "country": (data.get('sys') or {}).get('country'),
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
        
        if customer_option == "Select existing customer":
            customer_id = st.selectbox(
                "Select Customer ID:",
                customers_df['customer_id'].tolist()
            )
            customer_data = customers_df[customers_df['customer_id'] == customer_id].iloc[0].to_dict()
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
        st.write(f"- **Location:** {customer_data['location_type'].title()}")
        st.write(f"- **Household Size:** {customer_data['household_size']}")
        st.write(f"- **Has Pets:** {'Yes' if customer_data['has_pets'] else 'No'}")
        st.write(f"- **Work from Home:** {'Yes' if customer_data['work_from_home'] else 'No'}")
    
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
        use_live_weather = st.checkbox("Use live weather (OpenWeather)", value=True)
        openweather_key = st.secrets.get("OPENWEATHER_API_KEY") if hasattr(st, 'secrets') else None
        weather_details = None
        if use_live_weather and openweather_key:
            city_query = st.text_input("City (e.g., London or London,UK)", value="London")
            if city_query.strip():
                with st.spinner("Fetching current weather..."):
                    mapped, info = fetch_live_weather(city_query.strip(), openweather_key)
                if mapped:
                    weather = mapped
                    weather_details = info
                    st.info(f"Live weather: {info['main']} | Category used: {mapped} | Temp: {info.get('temp_c')}°C in {info.get('city')}, {info.get('country')}")
                else:
                    st.warning(f"Could not fetch live weather ({info}). Falling back to manual selection.")
                    weather = st.selectbox(
                        "Weather Condition:",
                        ['sunny', 'cloudy', 'rainy', 'snowy']
                    )
            else:
                weather = st.selectbox(
                    "Weather Condition:",
                    ['sunny', 'cloudy', 'rainy', 'snowy']
                )
        else:
            if use_live_weather and not openweather_key:
                st.warning("OPENWEATHER_API_KEY is not set in Streamlit Secrets. Using manual weather selection.")
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(customers_df):,}")
    with col2:
        st.metric("Delivery Records", f"{len(delivery_df):,}")
    with col3:
        st.metric("Calendar Events", f"{len(calendar_df):,}")
    with col4:
        st.metric("Success Rate", f"{delivery_df['was_home'].mean():.1%}")
    
    st.markdown("---")
    
    # Customer profiles distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Customer Profile Distribution")
        profile_counts = customers_df['profile_type'].value_counts()
        fig = px.pie(
            values=profile_counts.values,
            names=[name.replace('_', ' ').title() for name in profile_counts.index],
            title="Customer Profiles"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏠 Location Distribution")
        location_counts = customers_df['location_type'].value_counts()
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
        st.dataframe(customers_df.head(10))
    
    with tab2:
        st.dataframe(delivery_df.head(10))
    
    with tab3:
        st.dataframe(calendar_df.head(10))

def analytics_dashboard_page(delivery_df, customers_df):
    st.header("📈 Analytics Dashboard")
    
    # Success rates by time period
    st.subheader("⏰ Success Rates by Time Period")
    time_success = delivery_df.groupby('time_period')['was_home'].mean().sort_values(ascending=False)
    
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
        delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6])
        weekend_success = delivery_df.groupby('is_weekend')['was_home'].mean()
        
        fig = px.bar(
            x=['Weekday', 'Weekend'],
            y=[weekend_success[False], weekend_success[True]],
            title="Success Rate: Weekend vs Weekday"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌤️ Weather Impact")
        weather_success = delivery_df.groupby('weather_condition')['was_home'].mean()
        
        fig = px.bar(
            x=[weather.title() for weather in weather_success.index],
            y=weather_success.values,
            title="Success Rate by Weather"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap of success rates
    st.subheader("🔥 Success Rate Heatmap (Day vs Hour)")
    
    # Create hour column
    delivery_df['hour'] = delivery_df['time_window_start'].str.split(':').str[0].astype(int)
    
    # Create pivot table
    heatmap_data = delivery_df.groupby(['day_of_week', 'hour'])['was_home'].mean().unstack(fill_value=0)
    
    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        title="Success Rate Heatmap (Day of Week vs Hour)",
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

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
