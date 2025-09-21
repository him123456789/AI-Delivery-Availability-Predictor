#!/usr/bin/env python3
"""
Simple Delivery Availability Prediction Demo
============================================

A working demonstration of the AI model for predicting customer availability.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def load_dataset_info():
    """Load and display dataset information"""
    print("=" * 60)
    print("DELIVERY AVAILABILITY PREDICTION SYSTEM")
    print("=" * 60)
    
    # Load datasets
    customers_df = pd.read_csv('customers.csv')
    delivery_df = pd.read_csv('delivery_history.csv')
    calendar_df = pd.read_csv('calendar_data.csv')
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   • Total Customers: {len(customers_df):,}")
    print(f"   • Historical Deliveries: {len(delivery_df):,}")
    print(f"   • Calendar Events: {len(calendar_df):,}")
    print(f"   • Overall Success Rate: {delivery_df['was_home'].mean():.1%}")
    
    print(f"\n👥 CUSTOMER PROFILES")
    profile_counts = customers_df['profile_type'].value_counts()
    for profile, count in profile_counts.items():
        percentage = (count / len(customers_df)) * 100
        print(f"   • {profile.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print(f"\n⏰ SUCCESS RATES BY TIME PERIOD")
    time_success = delivery_df.groupby('time_period')['was_home'].mean().sort_values(ascending=False)
    for period, rate in time_success.items():
        print(f"   • {period.replace('_', ' ').title()}: {rate:.1%}")
    
    return customers_df, delivery_df, calendar_df

def train_simple_model():
    """Train a simplified version of the model"""
    print(f"\n🤖 TRAINING SIMPLIFIED AI MODEL")
    print("=" * 40)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load data
    customers_df = pd.read_csv('customers.csv')
    delivery_df = pd.read_csv('delivery_history.csv')
    
    # Merge datasets
    merged_df = delivery_df.merge(customers_df, on='customer_id', how='left')
    
    # Simple feature engineering
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Test samples: {len(X_test):,}")
    print(f"   • Features used: {len(feature_columns)}")
    print(f"   • Accuracy: {accuracy:.1%}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 TOP 5 MOST IMPORTANT FEATURES:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"   • {row['feature']}: {row['importance']:.3f}")
    
    return model, label_encoders, feature_columns

def make_predictions(model, label_encoders, feature_columns):
    """Make sample predictions"""
    print(f"\n🔮 SAMPLE PREDICTIONS")
    print("=" * 40)
    
    # Example customers
    examples = [
        {
            'name': 'Working Professional (Urban)',
            'profile_type': 'working_professional',
            'age': 32,
            'location_type': 'urban',
            'household_size': 2,
            'has_pets': True,
            'work_from_home': False
        },
        {
            'name': 'Retiree (Suburban)',
            'profile_type': 'retiree',
            'age': 68,
            'location_type': 'suburban',
            'household_size': 2,
            'has_pets': False,
            'work_from_home': False
        },
        {
            'name': 'Stay-at-Home Parent',
            'profile_type': 'stay_at_home',
            'age': 35,
            'location_type': 'suburban',
            'household_size': 4,
            'has_pets': True,
            'work_from_home': False
        }
    ]
    
    # Time windows to test
    time_windows = [
        ('09:00', 'Morning'),
        ('14:00', 'Afternoon'),
        ('18:00', 'Evening')
    ]
    
    # Days to test
    days = [
        (1, 'Tuesday'),  # Weekday
        (5, 'Saturday')  # Weekend
    ]
    
    for example in examples:
        print(f"\n👤 {example['name']}:")
        
        for day_num, day_name in days:
            print(f"   📅 {day_name}:")
            
            for time_str, time_label in time_windows:
                # Create prediction data
                pred_data = {
                    'day_of_week': day_num,
                    'month': 9,  # September
                    'hour': int(time_str.split(':')[0]),
                    'age': example['age'],
                    'household_size': example['household_size'],
                    'delivery_attempt_number': 1,
                    'package_value': 50.0,
                    'prev_success_rate': 0.7,
                    'has_pets': example['has_pets'],
                    'work_from_home': example['work_from_home'],
                    'is_holiday': False,
                    'requires_signature': False,
                    'is_weekend': day_num >= 5,
                    'time_period': 'morning' if time_str == '09:00' else 'afternoon' if time_str == '14:00' else 'evening',
                    'weather_condition': 'sunny'
                }
                
                # Encode categorical features
                for feature in ['profile_type', 'location_type', 'time_period', 'weather_condition']:
                    if feature in label_encoders:
                        try:
                            if feature == 'profile_type':
                                value = example['profile_type']
                            elif feature == 'location_type':
                                value = example['location_type']
                            else:
                                value = pred_data[feature]
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
                
                status = "✅" if prediction else "❌"
                print(f"      {status} {time_label}: {probability:.1%}")

def show_business_insights():
    """Show key business insights"""
    print(f"\n💼 BUSINESS INSIGHTS")
    print("=" * 40)
    
    delivery_df = pd.read_csv('delivery_history.csv')
    
    # Best times analysis
    time_success = delivery_df.groupby('time_period')['was_home'].mean().sort_values(ascending=False)
    print(f"📈 BEST DELIVERY TIMES:")
    for i, (period, rate) in enumerate(time_success.items(), 1):
        print(f"   {i}. {period.replace('_', ' ').title()}: {rate:.1%}")
    
    # Weekend vs weekday
    delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6])
    weekend_rate = delivery_df[delivery_df['is_weekend']]['was_home'].mean()
    weekday_rate = delivery_df[~delivery_df['is_weekend']]['was_home'].mean()
    
    print(f"\n📅 WEEKEND VS WEEKDAY:")
    print(f"   • Weekend success: {weekend_rate:.1%}")
    print(f"   • Weekday success: {weekday_rate:.1%}")
    print(f"   • Weekend advantage: +{(weekend_rate - weekday_rate):.1%}")
    
    # Weather impact
    weather_success = delivery_df.groupby('weather_condition')['was_home'].mean().sort_values(ascending=False)
    print(f"\n🌤️ WEATHER IMPACT:")
    for weather, rate in weather_success.items():
        print(f"   • {weather.title()}: {rate:.1%}")
    
    print(f"\n💡 KEY RECOMMENDATIONS:")
    print(f"   1. Prioritize {time_success.index[0].replace('_', ' ')} deliveries")
    print(f"   2. Schedule more weekend deliveries when possible")
    print(f"   3. Adjust for weather conditions")
    print(f"   4. Use customer profiles for personalized scheduling")

def main():
    """Main demo function"""
    try:
        # Load dataset info
        customers_df, delivery_df, calendar_df = load_dataset_info()
        
        # Train model
        model, label_encoders, feature_columns = train_simple_model()
        
        # Make predictions
        make_predictions(model, label_encoders, feature_columns)
        
        # Show insights
        show_business_insights()
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The AI model successfully predicts customer availability")
        print("and can recommend optimal delivery times!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
