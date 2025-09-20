import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def preprocess_odisha_data():
    """
    Preprocess the crop production data for Odisha state.
    Remove invalid/missing entries and create yield feature.
    """
    print("Loading crop production data...")
    df = pd.read_csv('crop_production.csv')
    
    # Filter for Odisha state only
    odisha_data = df[df['State_Name'] == 'Odisha'].copy()
    print(f"Odisha data shape: {odisha_data.shape}")
    
    # Remove records with missing production data
    odisha_data = odisha_data.dropna(subset=['Production'])
    print(f"After removing missing production: {odisha_data.shape}")
    
    # Remove records with zero or negative area/production
    odisha_data = odisha_data[(odisha_data['Area'] > 0) & (odisha_data['Production'] > 0)]
    print(f"After removing invalid area/production: {odisha_data.shape}")
    
    # Calculate yield (Production/Area)
    odisha_data['Yield'] = odisha_data['Production'] / odisha_data['Area']
    
    # Remove outliers (yield values beyond 3 standard deviations)
    yield_mean = odisha_data['Yield'].mean()
    yield_std = odisha_data['Yield'].std()
    odisha_data = odisha_data[
        (odisha_data['Yield'] >= yield_mean - 3*yield_std) & 
        (odisha_data['Yield'] <= yield_mean + 3*yield_std)
    ]
    print(f"After removing outliers: {odisha_data.shape}")
    
    # Clean season names (remove extra spaces)
    odisha_data['Season'] = odisha_data['Season'].str.strip()
    
    return odisha_data

def train_yield_prediction_model(data):
    """
    Train a Random Forest regression model to predict crop yield.
    """
    print("\nTraining yield prediction model...")
    
    # Features for prediction: District, Crop, Season, Year, Area
    features = ['District_Name', 'Crop', 'Season', 'Crop_Year', 'Area']
    target = 'Yield'
    
    # Create label encoders for categorical variables
    encoders = {}
    data_encoded = data.copy()
    
    for col in ['District_Name', 'Crop', 'Season']:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} categories")
    
    # Prepare features and target
    X = data_encoded[features]
    y = data_encoded[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    return rf_model, encoders

def save_model_and_encoders(model, encoders, data):
    """
    Save the trained model and label encoders.
    """
    print("\nSaving model and encoders...")
    
    # Save the model
    joblib.dump(model, 'crop_yield_model.pkl')
    
    # Save encoders
    joblib.dump(encoders, 'label_encoders.pkl')
    
    # Save unique values for reference
    unique_values = {
        'districts': sorted(data['District_Name'].unique().tolist()),
        'crops': sorted(data['Crop'].unique().tolist()),
        'seasons': sorted(data['Season'].unique().tolist()),
        'years': sorted([int(x) for x in data['Crop_Year'].unique().tolist()]),
        'area_range': {
            'min': float(data['Area'].min()),
            'max': float(data['Area'].max())
        }
    }
    
    import json
    with open('unique_values.json', 'w') as f:
        json.dump(unique_values, f, indent=2)
    
    print("Model and encoders saved successfully!")
    print(f"Districts: {len(unique_values['districts'])}")
    print(f"Crops: {len(unique_values['crops'])}")
    print(f"Seasons: {len(unique_values['seasons'])}")
    print(f"Years: {unique_values['years'][0]} - {unique_values['years'][-1]}")

def main():
    """
    Main function to run the preprocessing and model training pipeline.
    """
    print("=== Odisha Crop Yield Prediction Model Training ===")
    
    # Preprocess data
    odisha_data = preprocess_odisha_data()
    
    # Train model
    model, encoders = train_yield_prediction_model(odisha_data)
    
    # Save model and encoders
    save_model_and_encoders(model, encoders, odisha_data)
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()