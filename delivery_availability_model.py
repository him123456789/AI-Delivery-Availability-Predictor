import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from datetime import datetime, timedelta
import joblib
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DeliveryAvailabilityPredictor:
    """
    AI Model to predict customer availability for deliveries and recommend
    optimal delivery time windows based on historical data and calendar information.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def load_and_prepare_data(self, customers_file: str, delivery_file: str, 
                            calendar_file: str) -> pd.DataFrame:
        """Load and merge all datasets for training"""
        
        # Load datasets
        customers_df = pd.read_csv(customers_file)
        delivery_df = pd.read_csv(delivery_file)
        calendar_df = pd.read_csv(calendar_file)
        
        print(f"Loaded {len(customers_df)} customers, {len(delivery_df)} deliveries, {len(calendar_df)} calendar events")
        
        # Merge customer data with delivery data
        merged_df = delivery_df.merge(customers_df, on='customer_id', how='left')
        
        # Add calendar features
        merged_df = self._add_calendar_features(merged_df, calendar_df)
        
        # Feature engineering
        merged_df = self._engineer_features(merged_df)
        
        return merged_df
    
    def _add_calendar_features(self, delivery_df: pd.DataFrame, 
                             calendar_df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features to delivery data"""
        
        # Convert dates
        delivery_df['delivery_date'] = pd.to_datetime(delivery_df['delivery_date'])
        calendar_df['event_date'] = pd.to_datetime(calendar_df['event_date'])
        
        # For each delivery, check if there are conflicting calendar events
        delivery_features = []
        
        for _, delivery in delivery_df.iterrows():
            customer_id = delivery['customer_id']
            delivery_date = delivery['delivery_date']
            delivery_start = delivery['time_window_start']
            delivery_end = delivery['time_window_end']
            
            # Get customer's calendar for that day
            customer_calendar = calendar_df[
                (calendar_df['customer_id'] == customer_id) & 
                (calendar_df['event_date'] == delivery_date)
            ]
            
            # Check for conflicts
            has_conflict = False
            num_events = len(customer_calendar)
            high_priority_events = 0
            
            for _, event in customer_calendar.iterrows():
                event_start = event['start_time']
                event_end = event['end_time']
                
                # Simple time overlap check
                if self._times_overlap(delivery_start, delivery_end, event_start, event_end):
                    has_conflict = True
                
                if event['priority'] == 'high':
                    high_priority_events += 1
            
            delivery_features.append({
                'has_calendar_conflict': has_conflict,
                'num_calendar_events': num_events,
                'high_priority_events': high_priority_events
            })
        
        # Add features to dataframe
        calendar_features_df = pd.DataFrame(delivery_features)
        result_df = pd.concat([delivery_df.reset_index(drop=True), 
                              calendar_features_df.reset_index(drop=True)], axis=1)
        
        return result_df
    
    def _times_overlap(self, start1: str, end1: str, start2: str, end2: str) -> bool:
        """Check if two time ranges overlap"""
        try:
            s1 = datetime.strptime(start1, '%H:%M').time()
            e1 = datetime.strptime(end1, '%H:%M').time()
            s2 = datetime.strptime(start2, '%H:%M').time()
            e2 = datetime.strptime(end2, '%H:%M').time()
            
            return s1 < e2 and s2 < e1
        except:
            return False
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for the model"""
        
        # Time-based features
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_monday'] = df['day_of_week'] == 0
        df['is_friday'] = df['day_of_week'] == 4
        
        # Season
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Time period encoding
        time_period_order = {'morning': 1, 'late_morning': 2, 'afternoon': 3, 
                           'late_afternoon': 4, 'evening': 5, 'night': 6}
        df['time_period_numeric'] = df['time_period'].map(time_period_order)
        
        # Customer behavior patterns
        customer_stats = df.groupby('customer_id').agg({
            'was_home': ['mean', 'count'],
            'delivery_attempt_number': 'mean'
        }).round(3)
        
        customer_stats.columns = ['customer_success_rate', 'total_deliveries', 'avg_attempts']
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Weather impact
        weather_impact = {'sunny': 1, 'cloudy': 0.8, 'rainy': 0.6, 'snowy': 0.4}
        df['weather_score'] = df['weather_condition'].map(weather_impact)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for training/prediction"""
        
        # Select features for the model
        categorical_features = [
            'profile_type', 'location_type', 'time_period', 'calendar_event',
            'weather_condition', 'season'
        ]
        
        numerical_features = [
            'day_of_week', 'month', 'age', 'household_size', 'delivery_attempt_number',
            'package_value', 'prev_success_rate', 'customer_success_rate',
            'total_deliveries', 'avg_attempts', 'time_period_numeric',
            'num_calendar_events', 'high_priority_events', 'weather_score'
        ]
        
        boolean_features = [
            'has_pets', 'work_from_home', 'is_holiday', 'requires_signature',
            'special_instructions', 'is_weekend', 'is_monday', 'is_friday',
            'has_calendar_conflict'
        ]
        
        # Encode categorical features
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
            else:
                # Handle unseen categories during prediction
                df[f'{feature}_encoded'] = df[feature].apply(
                    lambda x: self._safe_transform(self.label_encoders[feature], str(x))
                )
        
        # Combine all features
        encoded_categorical = [f'{f}_encoded' for f in categorical_features]
        self.feature_columns = numerical_features + boolean_features + encoded_categorical
        
        # Create feature matrix
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X.values
    
    def _safe_transform(self, encoder: LabelEncoder, value: str) -> int:
        """Safely transform a value, handling unseen categories"""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # Return the most common class for unseen categories
            return 0
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the availability prediction model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select the best
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for name, model in models.items():
            # Train model
            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = (y_pred == y_test).mean()
            
            model_results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'model': model
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
        
        self.model = best_model
        self.is_trained = True
        
        # Feature importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        return model_results
    
    def predict_availability(self, customer_data: Dict, 
                           delivery_datetime: datetime,
                           time_window: Tuple[str, str]) -> Dict:
        """Predict availability for a specific customer and time"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create prediction dataframe
        pred_data = {
            'customer_id': customer_data['customer_id'],
            'delivery_date': delivery_datetime.strftime('%Y-%m-%d'),
            'day_of_week': delivery_datetime.weekday(),
            'month': delivery_datetime.month,
            'time_window_start': time_window[0],
            'time_window_end': time_window[1],
            'time_period': self._get_time_period(time_window[0]),
            **customer_data
        }
        
        # Add default values for missing features
        defaults = {
            'calendar_event': 'none',
            'weather_condition': 'sunny',
            'is_holiday': False,
            'prev_success_rate': 0.7,
            'delivery_attempt_number': 1,
            'package_value': 50.0,
            'requires_signature': False,
            'special_instructions': False,
            'has_calendar_conflict': False,
            'num_calendar_events': 0,
            'high_priority_events': 0
        }
        
        for key, value in defaults.items():
            if key not in pred_data:
                pred_data[key] = value
        
        pred_df = pd.DataFrame([pred_data])
        pred_df = self._engineer_features(pred_df)
        
        # Prepare features
        X_pred = self.prepare_features(pred_df)
        
        # Scale if using logistic regression
        if isinstance(self.model, LogisticRegression):
            X_pred = self.scaler.transform(X_pred)
        
        # Make prediction
        availability_prob = self.model.predict_proba(X_pred)[0, 1]
        is_available = self.model.predict(X_pred)[0]
        
        return {
            'availability_probability': float(availability_prob),
            'is_likely_available': bool(is_available),
            'confidence': 'high' if abs(availability_prob - 0.5) > 0.3 else 'medium' if abs(availability_prob - 0.5) > 0.15 else 'low'
        }
    
    def find_optimal_delivery_time(self, customer_data: Dict, 
                                 target_date: datetime,
                                 time_windows: List[Tuple[str, str]] = None) -> Dict:
        """Find the best delivery time window for a customer"""
        
        if time_windows is None:
            time_windows = [
                ('09:00', '12:00'),
                ('12:00', '15:00'),
                ('15:00', '18:00'),
                ('18:00', '21:00')
            ]
        
        predictions = []
        
        for start_time, end_time in time_windows:
            pred_result = self.predict_availability(
                customer_data, target_date, (start_time, end_time)
            )
            
            predictions.append({
                'time_window': f"{start_time}-{end_time}",
                'start_time': start_time,
                'end_time': end_time,
                **pred_result
            })
        
        # Sort by availability probability
        predictions.sort(key=lambda x: x['availability_probability'], reverse=True)
        
        return {
            'best_time_window': predictions[0],
            'all_predictions': predictions,
            'recommendation': f"Best delivery time: {predictions[0]['time_window']} "
                           f"(probability: {predictions[0]['availability_probability']:.2%})"
        }
    
    def _get_time_period(self, time_str: str) -> str:
        """Convert time string to period"""
        hour = int(time_str.split(':')[0])
        
        if 6 <= hour < 9:
            return 'morning'
        elif 9 <= hour < 12:
            return 'late_morning'
        elif 12 <= hour < 15:
            return 'afternoon'
        elif 15 <= hour < 18:
            return 'late_afternoon'
        elif 18 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")


def main():
    """Main training and evaluation pipeline"""
    
    # Initialize predictor
    predictor = DeliveryAvailabilityPredictor()
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = predictor.load_and_prepare_data(
        'customers.csv',
        'delivery_history.csv', 
        'calendar_data.csv'
    )
    
    print(f"Prepared dataset with {len(df)} records")
    print(f"Success rate: {df['was_home'].mean():.2%}")
    
    # Prepare features and target
    X = predictor.prepare_features(df)
    y = df['was_home'].values
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Train model
    print("\nTraining models...")
    results = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model('delivery_availability_model.pkl')
    
    # Example prediction
    print("\n=== Example Prediction ===")
    example_customer = {
        'customer_id': 1,
        'profile_type': 'working_professional',
        'age': 35,
        'location_type': 'urban',
        'household_size': 2,
        'has_pets': True,
        'work_from_home': False
    }
    
    target_date = datetime.now() + timedelta(days=1)
    
    # Find optimal delivery time
    optimal_result = predictor.find_optimal_delivery_time(example_customer, target_date)
    
    print(f"Customer: {example_customer['profile_type']}, Age: {example_customer['age']}")
    print(f"Target date: {target_date.strftime('%Y-%m-%d %A')}")
    print(f"{optimal_result['recommendation']}")
    
    print("\nAll time window predictions:")
    for pred in optimal_result['all_predictions']:
        print(f"  {pred['time_window']}: {pred['availability_probability']:.2%} "
              f"({pred['confidence']} confidence)")


if __name__ == "__main__":
    main()
