import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple
import json

class DeliveryAvailabilityDatasetGenerator:
    """
    Generates synthetic dataset for predicting customer availability for deliveries
    based on historical records and calendar patterns.
    """
    
    def __init__(self, num_customers: int = 1000, days_history: int = 365):
        self.num_customers = num_customers
        self.days_history = days_history
        self.start_date = datetime.now() - timedelta(days=days_history)
        
        # Define customer profiles
        self.customer_profiles = {
            'working_professional': {
                'home_weekday_morning': 0.2,  # 6-9 AM
                'home_weekday_afternoon': 0.1,  # 12-6 PM
                'home_weekday_evening': 0.8,   # 6-10 PM
                'home_weekend': 0.7,
                'calendar_busy_prob': 0.3
            },
            'stay_at_home': {
                'home_weekday_morning': 0.9,
                'home_weekday_afternoon': 0.8,
                'home_weekday_evening': 0.9,
                'home_weekend': 0.9,
                'calendar_busy_prob': 0.2
            },
            'student': {
                'home_weekday_morning': 0.6,
                'home_weekday_afternoon': 0.4,
                'home_weekday_evening': 0.7,
                'home_weekend': 0.8,
                'calendar_busy_prob': 0.4
            },
            'retiree': {
                'home_weekday_morning': 0.8,
                'home_weekday_afternoon': 0.9,
                'home_weekday_evening': 0.8,
                'home_weekend': 0.9,
                'calendar_busy_prob': 0.1
            },
            'shift_worker': {
                'home_weekday_morning': 0.5,
                'home_weekday_afternoon': 0.6,
                'home_weekday_evening': 0.4,
                'home_weekend': 0.6,
                'calendar_busy_prob': 0.2
            }
        }
        
        # Time windows for delivery
        self.time_windows = [
            ('06:00', '09:00', 'morning'),
            ('09:00', '12:00', 'late_morning'),
            ('12:00', '15:00', 'afternoon'),
            ('15:00', '18:00', 'late_afternoon'),
            ('18:00', '21:00', 'evening'),
            ('21:00', '23:00', 'night')
        ]
        
        # Calendar event types
        self.calendar_events = [
            'work_meeting', 'doctor_appointment', 'social_event', 
            'travel', 'family_event', 'personal_task', 'vacation', 
            'conference', 'training', 'none'
        ]
        
    def generate_customer_profiles(self) -> pd.DataFrame:
        """Generate customer profile data"""
        customers = []

        # Curated sample of UK locations with approximate coordinates and urban classification
        uk_locations = [
            {"city": "London", "region": "England", "lat": 51.5074, "lon": -0.1278, "type": "urban"},
            {"city": "Birmingham", "region": "England", "lat": 52.4862, "lon": -1.8904, "type": "urban"},
            {"city": "Manchester", "region": "England", "lat": 53.4808, "lon": -2.2426, "type": "urban"},
            {"city": "Leeds", "region": "England", "lat": 53.8008, "lon": -1.5491, "type": "urban"},
            {"city": "Liverpool", "region": "England", "lat": 53.4084, "lon": -2.9916, "type": "urban"},
            {"city": "Bristol", "region": "England", "lat": 51.4545, "lon": -2.5879, "type": "urban"},
            {"city": "Sheffield", "region": "England", "lat": 53.3811, "lon": -1.4701, "type": "urban"},
            {"city": "Newcastle upon Tyne", "region": "England", "lat": 54.9783, "lon": -1.6178, "type": "urban"},
            {"city": "Nottingham", "region": "England", "lat": 52.9548, "lon": -1.1581, "type": "urban"},
            {"city": "Leicester", "region": "England", "lat": 52.6369, "lon": -1.1398, "type": "urban"},
            {"city": "Cardiff", "region": "Wales", "lat": 51.4816, "lon": -3.1791, "type": "urban"},
            {"city": "Swansea", "region": "Wales", "lat": 51.6214, "lon": -3.9436, "type": "urban"},
            {"city": "Edinburgh", "region": "Scotland", "lat": 55.9533, "lon": -3.1883, "type": "urban"},
            {"city": "Glasgow", "region": "Scotland", "lat": 55.8642, "lon": -4.2518, "type": "urban"},
            {"city": "Aberdeen", "region": "Scotland", "lat": 57.1497, "lon": -2.0943, "type": "urban"},
            {"city": "Belfast", "region": "Northern Ireland", "lat": 54.5973, "lon": -5.9301, "type": "urban"},
            {"city": "Derry", "region": "Northern Ireland", "lat": 54.9979, "lon": -7.3090, "type": "urban"},
            {"city": "Cambridge", "region": "England", "lat": 52.2053, "lon": 0.1218, "type": "suburban"},
            {"city": "Oxford", "region": "England", "lat": 51.7520, "lon": -1.2577, "type": "suburban"},
            {"city": "Bath", "region": "England", "lat": 51.3811, "lon": -2.3590, "type": "suburban"},
            {"city": "York", "region": "England", "lat": 53.9590, "lon": -1.0815, "type": "suburban"},
            {"city": "Exeter", "region": "England", "lat": 50.7184, "lon": -3.5339, "type": "suburban"},
            {"city": "Inverness", "region": "Scotland", "lat": 57.4778, "lon": -4.2247, "type": "rural"},
            {"city": "Truro", "region": "England", "lat": 50.2632, "lon": -5.0510, "type": "rural"},
            {"city": "Bangor", "region": "Wales", "lat": 53.2270, "lon": -4.1294, "type": "rural"},
            {"city": "Stirling", "region": "Scotland", "lat": 56.1165, "lon": -3.9369, "type": "suburban"},
            {"city": "Reading", "region": "England", "lat": 51.4543, "lon": -0.9781, "type": "suburban"},
        ]
        
        for customer_id in range(1, self.num_customers + 1):
            profile_type = random.choice(list(self.customer_profiles.keys()))
            age = random.randint(18, 80)
            
            # Age-based adjustments
            if age < 25:
                profile_type = random.choice(['student', 'working_professional'])
            elif age > 65:
                profile_type = 'retiree'
            
            loc = random.choice(uk_locations)
            customers.append({
                'customer_id': customer_id,
                'profile_type': profile_type,
                'age': age,
                'location_type': loc['type'],
                'household_size': random.randint(1, 5),
                'has_pets': random.choice([True, False]),
                'work_from_home': random.choice([True, False]) if profile_type == 'working_professional' else False,
                'country': 'United Kingdom',
                'region': loc['region'],
                'city': loc['city'],
                'latitude': loc['lat'],
                'longitude': loc['lon'],
            })
        
        return pd.DataFrame(customers)
    
    def generate_historical_deliveries(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate historical delivery attempt data"""
        delivery_data = []
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            profile = self.customer_profiles[customer['profile_type']]
            
            # Generate 50-200 delivery attempts per customer over the period
            num_deliveries = random.randint(50, 200)
            
            for _ in range(num_deliveries):
                # Random date within the history period
                random_days = random.randint(0, self.days_history - 1)
                delivery_date = self.start_date + timedelta(days=random_days)
                
                # Random time window
                time_window = random.choice(self.time_windows)
                start_time, end_time, period = time_window
                
                # Determine if customer was home based on profile and various factors
                was_home = self._determine_availability(
                    customer, profile, delivery_date, period
                )
                
                # Generate calendar event
                calendar_event = self._generate_calendar_event(
                    profile, delivery_date, period
                )
                
                # Weather impact (simplified)
                weather_condition = random.choice(['sunny', 'rainy', 'snowy', 'cloudy'])
                weather_impact = -0.1 if weather_condition in ['rainy', 'snowy'] else 0
                
                # Previous delivery success rate (customer learning)
                prev_success_rate = random.uniform(0.3, 0.9)
                
                delivery_data.append({
                    'customer_id': customer_id,
                    'delivery_date': delivery_date.strftime('%Y-%m-%d'),
                    'day_of_week': delivery_date.weekday(),  # 0=Monday, 6=Sunday
                    'month': delivery_date.month,
                    'time_window_start': start_time,
                    'time_window_end': end_time,
                    'time_period': period,
                    'was_home': was_home,
                    'calendar_event': calendar_event,
                    'weather_condition': weather_condition,
                    'is_holiday': self._is_holiday(delivery_date),
                    'prev_success_rate': prev_success_rate,
                    'delivery_attempt_number': random.randint(1, 3),  # 1st, 2nd, or 3rd attempt
                    'package_value': random.uniform(10, 500),  # Package value in dollars
                    'requires_signature': random.choice([True, False]),
                    'special_instructions': random.choice([True, False])
                })
        
        return pd.DataFrame(delivery_data)
    
    def _determine_availability(self, customer: pd.Series, profile: Dict, 
                              delivery_date: datetime, period: str) -> bool:
        """Determine if customer is available based on various factors"""
        is_weekend = delivery_date.weekday() >= 5
        
        # Base probability from profile
        if is_weekend:
            base_prob = profile['home_weekend']
        else:
            if period == 'morning':
                base_prob = profile['home_weekday_morning']
            elif period in ['afternoon', 'late_afternoon']:
                base_prob = profile['home_weekday_afternoon']
            else:
                base_prob = profile['home_weekday_evening']
        
        # Adjustments
        if customer['work_from_home']:
            base_prob += 0.2
        
        if customer['has_pets']:
            base_prob += 0.1
        
        if customer['household_size'] > 3:
            base_prob += 0.1
        
        # Random variation
        final_prob = max(0, min(1, base_prob + random.uniform(-0.2, 0.2)))
        
        return random.random() < final_prob
    
    def _generate_calendar_event(self, profile: Dict, delivery_date: datetime, 
                               period: str) -> str:
        """Generate calendar event for the given time period"""
        if random.random() > profile['calendar_busy_prob']:
            return 'none'
        
        # Different events more likely at different times
        if period == 'morning':
            events = ['work_meeting', 'doctor_appointment', 'personal_task']
        elif period in ['afternoon', 'late_afternoon']:
            events = ['work_meeting', 'doctor_appointment', 'social_event', 'personal_task']
        else:
            events = ['social_event', 'family_event', 'personal_task']
        
        return random.choice(events)
    
    def _is_holiday(self, date: datetime) -> bool:
        """Simple holiday detection (can be expanded)"""
        # Major US holidays (simplified)
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (11, 24), # Thanksgiving (approximation)
        ]
        return (date.month, date.day) in holidays
    
    def generate_calendar_data(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed calendar/schedule data for customers"""
        calendar_data = []
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            
            # Generate calendar events for the next 30 days
            for days_ahead in range(30):
                event_date = datetime.now() + timedelta(days=days_ahead)
                
                # Generate 0-5 events per day
                num_events = random.choices([0, 1, 2, 3, 4, 5], 
                                          weights=[30, 35, 20, 10, 4, 1])[0]
                
                for _ in range(num_events):
                    start_hour = random.randint(6, 22)
                    duration = random.choice([1, 2, 3, 4, 8])  # hours
                    
                    calendar_data.append({
                        'customer_id': customer_id,
                        'event_date': event_date.strftime('%Y-%m-%d'),
                        'start_time': f"{start_hour:02d}:00",
                        'end_time': f"{min(23, start_hour + duration):02d}:00",
                        'event_type': random.choice(self.calendar_events[:-1]),  # exclude 'none'
                        'is_recurring': random.choice([True, False]),
                        'priority': random.choice(['low', 'medium', 'high']),
                        'location': random.choice(['home', 'office', 'external', 'online'])
                    })
        
        return pd.DataFrame(calendar_data)
    
    def generate_complete_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate the complete dataset"""
        print("Generating customer profiles...")
        customers_df = self.generate_customer_profiles()
        
        print("Generating historical delivery data...")
        delivery_df = self.generate_historical_deliveries(customers_df)
        
        print("Generating calendar data...")
        calendar_df = self.generate_calendar_data(customers_df)
        
        return customers_df, delivery_df, calendar_df
    
    def save_datasets(self, customers_df: pd.DataFrame, delivery_df: pd.DataFrame, 
                     calendar_df: pd.DataFrame, output_dir: str = "."):
        """Save datasets to CSV files"""
        customers_df.to_csv(f"{output_dir}/customers.csv", index=False)
        delivery_df.to_csv(f"{output_dir}/delivery_history.csv", index=False)
        calendar_df.to_csv(f"{output_dir}/calendar_data.csv", index=False)
        
        # Save metadata
        metadata = {
            'num_customers': self.num_customers,
            'days_history': self.days_history,
            'generation_date': datetime.now().isoformat(),
            'customer_profiles': self.customer_profiles,
            'time_windows': self.time_windows
        }
        
        with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Datasets saved to {output_dir}/")
        print(f"- customers.csv: {len(customers_df)} customers")
        print(f"- delivery_history.csv: {len(delivery_df)} delivery records")
        print(f"- calendar_data.csv: {len(calendar_df)} calendar events")


if __name__ == "__main__":
    # Generate dataset
    generator = DeliveryAvailabilityDatasetGenerator(num_customers=1000, days_history=365)
    customers_df, delivery_df, calendar_df = generator.generate_complete_dataset()
    
    # Save datasets
    generator.save_datasets(customers_df, delivery_df, calendar_df)
    
    # Display basic statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total customers: {len(customers_df)}")
    print(f"Total delivery attempts: {len(delivery_df)}")
    print(f"Success rate: {delivery_df['was_home'].mean():.2%}")
    print(f"Calendar events: {len(calendar_df)}")
    
    print("\n=== Customer Profile Distribution ===")
    print(customers_df['profile_type'].value_counts())
    
    print("\n=== Delivery Success by Time Period ===")
    success_by_period = delivery_df.groupby('time_period')['was_home'].mean().sort_values(ascending=False)
    for period, rate in success_by_period.items():
        print(f"{period}: {rate:.2%}")
