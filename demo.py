#!/usr/bin/env python3
"""
Demo script showing the crop yield prediction functionality.
"""

import pandas as pd
import numpy as np
import joblib
import json

def main():
    print("🌾 AGRISCAN AI - CROP YIELD PREDICTION DEMO")
    print("=" * 50)
    
    try:
        # Load model components
        print("Loading trained model and encoders...")
        model = joblib.load('crop_yield_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        
        with open('unique_values.json', 'r') as f:
            unique_values = json.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model covers {len(unique_values['districts'])} districts, {len(unique_values['crops'])} crops")
        
        # Interactive demo
        print("\n🎯 PREDICTION DEMO")
        print("-" * 30)
        
        # Sample predictions for different scenarios
        scenarios = [
            {
                'name': 'Rice in Cuttack (Main Season)',
                'district': 'CUTTACK',
                'crop': 'Paddy',
                'season': 'Kharif',
                'year': 2015,
                'area': 100.0
            },
            {
                'name': 'Groundnut in Ganjam (Winter)',
                'district': 'GANJAM', 
                'crop': 'Groundnut',
                'season': 'Rabi',
                'year': 2014,
                'area': 50.0
            },
            {
                'name': 'Sugarcane in Bargarh (Year Round)',
                'district': 'BARGARH',
                'crop': 'Sugarcane',
                'season': 'Whole Year',
                'year': 2015,
                'area': 25.0
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📍 Scenario {i}: {scenario['name']}")
            
            # Encode features
            district_encoded = encoders['District_Name'].transform([scenario['district']])[0]
            crop_encoded = encoders['Crop'].transform([scenario['crop']])[0]
            season_encoded = encoders['Season'].transform([scenario['season']])[0]
            
            # Predict
            features = np.array([[
                district_encoded, crop_encoded, season_encoded, 
                scenario['year'], scenario['area']
            ]])
            
            yield_pred = model.predict(features)[0]
            total_production = yield_pred * scenario['area']
            
            print(f"   🏘️  District: {scenario['district']}")
            print(f"   🌱 Crop: {scenario['crop']}")
            print(f"   📅 Season: {scenario['season']} {scenario['year']}")
            print(f"   📐 Area: {scenario['area']} hectares")
            print(f"   📈 Predicted Yield: {yield_pred:.2f} tons/hectare")
            print(f"   🚚 Total Production: {total_production:.2f} tons")
            
            # Yield assessment
            if yield_pred < 1.0:
                status = "⚠️  Low yield - consider improvements"
            elif yield_pred < 3.0:
                status = "ℹ️  Moderate yield - good practices recommended"
            else:
                status = "🎉 High yield - excellent conditions"
            print(f"   {status}")
        
        print("\n📋 SYSTEM CAPABILITIES")
        print("-" * 30)
        print(f"✓ Districts: {len(unique_values['districts'])} (All major districts in Odisha)")
        print(f"✓ Crops: {len(unique_values['crops'])} (Including Rice, Wheat, Maize, Groundnut, etc.)")
        print(f"✓ Seasons: {len(unique_values['seasons'])} (Kharif, Rabi, Summer, etc.)")
        print(f"✓ Years: {unique_values['years'][0]}-{unique_values['years'][-1]} (Historical data coverage)")
        print(f"✓ Area Range: {unique_values['area_range']['min']}-{unique_values['area_range']['max']} hectares")
        
        print("\n🚀 TO RUN THE WEB APPLICATION:")
        print("-" * 30)
        print("1. Install Streamlit: pip install streamlit")
        print("2. Run the app: streamlit run app.py")
        print("3. Open your browser to the displayed URL")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()