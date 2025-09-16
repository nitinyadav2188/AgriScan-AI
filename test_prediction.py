#!/usr/bin/env python3
"""
Test script to verify the crop yield prediction functionality.
"""

import pandas as pd
import numpy as np
import joblib
import json

def test_prediction():
    """Test the prediction functionality without Streamlit."""
    print("Testing crop yield prediction...")
    
    try:
        # Load model and encoders
        model = joblib.load('crop_yield_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        
        with open('unique_values.json', 'r') as f:
            unique_values = json.load(f)
        
        print("✅ Model and data loaded successfully")
        
        # Test prediction with sample data
        test_cases = [
            {
                'district': 'CUTTACK',
                'crop': 'Paddy',
                'season': 'Kharif',
                'year': 2015,
                'area': 100.0
            },
            {
                'district': 'GANJAM',
                'crop': 'Groundnut',
                'season': 'Rabi',
                'year': 2014,
                'area': 50.0
            },
            {
                'district': 'KALAHANDI',
                'crop': 'Maize',
                'season': 'Kharif',
                'year': 2013,
                'area': 75.0
            }
        ]
        
        print("\n🧪 Running test predictions...")
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                # Encode categorical features
                district_encoded = encoders['District_Name'].transform([test_case['district']])[0]
                crop_encoded = encoders['Crop'].transform([test_case['crop']])[0]
                season_encoded = encoders['Season'].transform([test_case['season']])[0]
                
                # Create feature array
                features = np.array([[
                    district_encoded, 
                    crop_encoded, 
                    season_encoded, 
                    test_case['year'], 
                    test_case['area']
                ]])
                
                # Make prediction
                yield_prediction = model.predict(features)[0]
                total_production = yield_prediction * test_case['area']
                
                print(f"\nTest Case {i}:")
                print(f"  District: {test_case['district']}")
                print(f"  Crop: {test_case['crop']}")
                print(f"  Season: {test_case['season']}")
                print(f"  Year: {test_case['year']}")
                print(f"  Area: {test_case['area']} hectares")
                print(f"  Predicted Yield: {yield_prediction:.2f} tons/hectare")
                print(f"  Total Production: {total_production:.2f} tons")
                
            except ValueError as e:
                print(f"❌ Test Case {i} failed: {e}")
        
        print("\n📊 Model Information:")
        print(f"  Districts: {len(unique_values['districts'])}")
        print(f"  Crops: {len(unique_values['crops'])}")
        print(f"  Seasons: {len(unique_values['seasons'])}")
        print(f"  Year Range: {unique_values['years'][0]} - {unique_values['years'][-1]}")
        
        print("\n✅ All tests completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run 'python train_model.py' first to generate the model files.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_prediction()