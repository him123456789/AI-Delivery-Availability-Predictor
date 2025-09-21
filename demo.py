#!/usr/bin/env python3
"""
Delivery Availability Prediction Demo
=====================================

This demo showcases the AI model that predicts customer availability for deliveries
and recommends optimal delivery time windows based on historical data and calendar information.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from delivery_availability_model import DeliveryAvailabilityPredictor
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_display_dataset_info():
    """Load and display information about the generated dataset"""
    print("=" * 60)
    print("DELIVERY AVAILABILITY PREDICTION SYSTEM")
    print("=" * 60)
    
    # Load datasets
    customers_df = pd.read_csv('customers.csv')
    delivery_df = pd.read_csv('delivery_history.csv')
    calendar_df = pd.read_csv('calendar_data.csv')
    
    with open('dataset_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   • Total Customers: {len(customers_df):,}")
    print(f"   • Historical Deliveries: {len(delivery_df):,}")
    print(f"   • Calendar Events: {len(calendar_df):,}")
    print(f"   • Data Period: {metadata['days_history']} days")
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
    
    print(f"\n📅 SUCCESS RATES BY DAY OF WEEK")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_success = delivery_df.groupby('day_of_week')['was_home'].mean()
    for day_num, rate in day_success.items():
        print(f"   • {day_names[day_num]}: {rate:.1%}")
    
    return customers_df, delivery_df, calendar_df

def train_and_evaluate_model():
    """Train the AI model and show evaluation results"""
    print(f"\n🤖 TRAINING AI MODEL")
    print("=" * 40)
    
    # Initialize predictor
    predictor = DeliveryAvailabilityPredictor()
    
    # Load and prepare data
    print("Loading and preparing training data...")
    df = predictor.load_and_prepare_data(
        'customers.csv',
        'delivery_history.csv', 
        'calendar_data.csv'
    )
    
    # Prepare features and target
    X = predictor.prepare_features(df)
    y = df['was_home'].values
    
    print(f"Training dataset: {len(df):,} records with {X.shape[1]} features")
    
    # Train model
    print("\nTraining multiple models and selecting the best...")
    results = predictor.train_model(X, y)
    
    print(f"\n✅ Model training completed!")
    print(f"Best model selected based on AUC score")
    
    # Save model
    predictor.save_model('delivery_availability_model.pkl')
    
    return predictor

def demonstrate_predictions(predictor):
    """Demonstrate the model's prediction capabilities"""
    print(f"\n🔮 PREDICTION DEMONSTRATIONS")
    print("=" * 40)
    
    # Load customer data for examples
    customers_df = pd.read_csv('customers.csv')
    
    # Example customers with different profiles
    example_customers = [
        {
            'name': 'Working Professional',
            'data': {
                'customer_id': 1,
                'profile_type': 'working_professional',
                'age': 32,
                'location_type': 'urban',
                'household_size': 2,
                'has_pets': True,
                'work_from_home': False
            }
        },
        {
            'name': 'Stay-at-Home Parent',
            'data': {
                'customer_id': 2,
                'profile_type': 'stay_at_home',
                'age': 35,
                'location_type': 'suburban',
                'household_size': 4,
                'has_pets': True,
                'work_from_home': False
            }
        },
        {
            'name': 'Retiree',
            'data': {
                'customer_id': 3,
                'profile_type': 'retiree',
                'age': 68,
                'location_type': 'suburban',
                'household_size': 2,
                'has_pets': False,
                'work_from_home': False
            }
        },
        {
            'name': 'Remote Worker',
            'data': {
                'customer_id': 4,
                'profile_type': 'working_professional',
                'age': 29,
                'location_type': 'urban',
                'household_size': 1,
                'has_pets': False,
                'work_from_home': True
            }
        }
    ]
    
    # Predict for different days and times
    target_dates = [
        datetime.now() + timedelta(days=1),  # Tomorrow
        datetime.now() + timedelta(days=2),  # Day after tomorrow
    ]
    
    for i, target_date in enumerate(target_dates):
        day_name = target_date.strftime('%A')
        date_str = target_date.strftime('%Y-%m-%d')
        
        print(f"\n📅 PREDICTIONS FOR {day_name.upper()} ({date_str})")
        print("-" * 50)
        
        for customer in example_customers:
            print(f"\n👤 {customer['name']}:")
            
            # Find optimal delivery time
            optimal_result = predictor.find_optimal_delivery_time(
                customer['data'], target_date
            )
            
            best_window = optimal_result['best_time_window']
            print(f"   🎯 Best Time: {best_window['time_window']} "
                  f"({best_window['availability_probability']:.1%} probability)")
            print(f"   📊 Confidence: {best_window['confidence'].title()}")
            
            # Show all time windows
            print(f"   📋 All Time Windows:")
            for pred in optimal_result['all_predictions']:
                status = "✅" if pred['is_likely_available'] else "❌"
                print(f"      {status} {pred['time_window']}: {pred['availability_probability']:.1%}")

def demonstrate_calendar_integration():
    """Show how calendar events affect predictions"""
    print(f"\n📅 CALENDAR INTEGRATION DEMO")
    print("=" * 40)
    
    # Load a trained model
    predictor = DeliveryAvailabilityPredictor()
    predictor.load_model('delivery_availability_model.pkl')
    
    # Example customer
    customer_data = {
        'customer_id': 100,
        'profile_type': 'working_professional',
        'age': 30,
        'location_type': 'urban',
        'household_size': 2,
        'has_pets': False,
        'work_from_home': False
    }
    
    target_date = datetime.now() + timedelta(days=1)
    time_window = ('14:00', '17:00')
    
    print(f"Customer: Working Professional, Age 30")
    print(f"Target: {target_date.strftime('%A, %Y-%m-%d')} at {time_window[0]}-{time_window[1]}")
    
    # Scenario 1: No calendar conflicts
    print(f"\n📊 Scenario 1: No Calendar Conflicts")
    pred1 = predictor.predict_availability(customer_data, target_date, time_window)
    print(f"   Availability: {pred1['availability_probability']:.1%}")
    print(f"   Likely Available: {'Yes' if pred1['is_likely_available'] else 'No'}")
    print(f"   Confidence: {pred1['confidence'].title()}")
    
    # Scenario 2: With calendar conflict
    print(f"\n📊 Scenario 2: With Important Meeting")
    customer_data_busy = customer_data.copy()
    customer_data_busy.update({
        'calendar_event': 'work_meeting',
        'has_calendar_conflict': True,
        'num_calendar_events': 2,
        'high_priority_events': 1
    })
    
    pred2 = predictor.predict_availability(customer_data_busy, target_date, time_window)
    print(f"   Availability: {pred2['availability_probability']:.1%}")
    print(f"   Likely Available: {'Yes' if pred2['is_likely_available'] else 'No'}")
    print(f"   Confidence: {pred2['confidence'].title()}")
    
    impact = pred1['availability_probability'] - pred2['availability_probability']
    print(f"   📉 Calendar Impact: -{impact:.1%} availability")

def show_business_insights():
    """Show business insights from the model"""
    print(f"\n💼 BUSINESS INSIGHTS")
    print("=" * 40)
    
    delivery_df = pd.read_csv('delivery_history.csv')
    
    print(f"📈 KEY FINDINGS:")
    
    # Best delivery times
    time_success = delivery_df.groupby('time_period')['was_home'].mean().sort_values(ascending=False)
    best_time = time_success.index[0]
    best_rate = time_success.iloc[0]
    worst_time = time_success.index[-1]
    worst_rate = time_success.iloc[-1]
    
    print(f"   • Best delivery period: {best_time.replace('_', ' ').title()} ({best_rate:.1%} success)")
    print(f"   • Worst delivery period: {worst_time.replace('_', ' ').title()} ({worst_rate:.1%} success)")
    
    # Weekend vs weekday
    delivery_df['is_weekend'] = delivery_df['day_of_week'].isin([5, 6])
    weekend_success = delivery_df[delivery_df['is_weekend']]['was_home'].mean()
    weekday_success = delivery_df[~delivery_df['is_weekend']]['was_home'].mean()
    
    print(f"   • Weekend success rate: {weekend_success:.1%}")
    print(f"   • Weekday success rate: {weekday_success:.1%}")
    
    # Weather impact
    weather_success = delivery_df.groupby('weather_condition')['was_home'].mean().sort_values(ascending=False)
    print(f"   • Weather impact:")
    for weather, rate in weather_success.items():
        print(f"     - {weather.title()}: {rate:.1%}")
    
    # Delivery attempt analysis
    attempt_success = delivery_df.groupby('delivery_attempt_number')['was_home'].mean()
    print(f"   • Success by attempt number:")
    for attempt, rate in attempt_success.items():
        print(f"     - Attempt {attempt}: {rate:.1%}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Schedule more deliveries during {best_time.replace('_', ' ')} hours")
    print(f"   • Consider weather conditions when planning routes")
    print(f"   • Implement customer preference learning for repeat deliveries")
    print(f"   • Use calendar integration to avoid conflicts")

def main():
    """Main demo function"""
    try:
        # Load and display dataset information
        customers_df, delivery_df, calendar_df = load_and_display_dataset_info()
        
        # Train the model
        predictor = train_and_evaluate_model()
        
        # Demonstrate predictions
        demonstrate_predictions(predictor)
        
        # Show calendar integration
        demonstrate_calendar_integration()
        
        # Show business insights
        show_business_insights()
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The AI model is now ready to predict customer availability")
        print("and recommend optimal delivery times for your logistics operations.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        print("Please ensure all required files are present and try again.")

if __name__ == "__main__":
    main()
