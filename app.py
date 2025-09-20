import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Set page config
st.set_page_config(
    page_title="Odisha Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data
@st.cache_resource
def load_model_and_data():
    """Load the trained model, encoders, and unique values."""
    try:
        model = joblib.load('crop_yield_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        
        with open('unique_values.json', 'r') as f:
            unique_values = json.load(f)
        
        return model, encoders, unique_values
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run train_model.py first to train the model.")
        st.stop()

def predict_yield(model, encoders, district, crop, season, year, area):
    """Make yield prediction using the trained model."""
    try:
        # Encode categorical features
        district_encoded = encoders['District_Name'].transform([district])[0]
        crop_encoded = encoders['Crop'].transform([crop])[0]
        season_encoded = encoders['Season'].transform([season])[0]
        
        # Create feature array
        features = np.array([[district_encoded, crop_encoded, season_encoded, year, area]])
        
        # Make prediction
        yield_prediction = model.predict(features)[0]
        
        return yield_prediction
    
    except ValueError as e:
        return None

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🌾 Odisha Crop Yield Predictor")
    st.markdown("""
    This AI-powered application predicts crop yield in Odisha, India using machine learning.
    The model is trained on historical crop production data and uses Random Forest regression.
    """)
    
    # Load model and data
    model, encoders, unique_values = load_model_and_data()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🔧 Input Parameters")
        
        # District selection
        district = st.selectbox(
            "Select District",
            options=unique_values['districts'],
            help="Choose the district in Odisha where you want to predict crop yield."
        )
        
        # Crop selection
        crop = st.selectbox(
            "Select Crop",
            options=unique_values['crops'],
            help="Choose the type of crop for yield prediction."
        )
        
        # Season selection
        season = st.selectbox(
            "Select Season",
            options=unique_values['seasons'],
            help="Choose the growing season for the crop."
        )
        
        # Year selection
        year = st.slider(
            "Select Year",
            min_value=unique_values['years'][0],
            max_value=unique_values['years'][-1],
            value=unique_values['years'][-1],
            help="Choose the year for prediction."
        )
        
        # Area input
        area = st.number_input(
            "Area (in hectares)",
            min_value=0.1,
            max_value=unique_values['area_range']['max'],
            value=100.0,
            step=0.1,
            help="Enter the cultivated area in hectares."
        )
        
        # Prediction button
        predict_button = st.button("🔍 Predict Yield", type="primary")
    
    with col2:
        st.header("📊 Prediction Results")
        
        if predict_button:
            with st.spinner("Calculating yield prediction..."):
                yield_pred = predict_yield(model, encoders, district, crop, season, year, area)
                
                if yield_pred is not None:
                    # Display prediction
                    st.success("✅ Prediction Complete!")
                    
                    # Create metrics
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.metric(
                            label="Predicted Yield",
                            value=f"{yield_pred:.2f}",
                            help="Yield in tons per hectare"
                        )
                    
                    with col2_2:
                        total_production = yield_pred * area
                        st.metric(
                            label="Total Production",
                            value=f"{total_production:.2f} tons",
                            help="Total expected production for the given area"
                        )
                    
                    # Additional information
                    st.info(f"""
                    **Prediction Summary:**
                    - **District:** {district}
                    - **Crop:** {crop}
                    - **Season:** {season}
                    - **Year:** {year}
                    - **Area:** {area} hectares
                    - **Predicted Yield:** {yield_pred:.2f} tons/hectare
                    - **Total Production:** {total_production:.2f} tons
                    """)
                    
                    # Yield interpretation
                    if yield_pred < 1.0:
                        st.warning("⚠️ Low yield predicted. Consider crop management improvements.")
                    elif yield_pred < 3.0:
                        st.info("ℹ️ Moderate yield predicted. Good agricultural practices recommended.")
                    else:
                        st.success("🎉 High yield predicted! Excellent conditions for this crop.")
                
                else:
                    st.error("❌ Prediction failed. Please check your input values.")
        
        else:
            st.info("👆 Enter the parameters on the left and click 'Predict Yield' to get started.")
    
    # Footer with model information
    st.markdown("---")
    st.markdown("### 📈 Model Information")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Districts Covered", len(unique_values['districts']))
    
    with col4:
        st.metric("Crops Supported", len(unique_values['crops']))
    
    with col5:
        st.metric("Seasons Available", len(unique_values['seasons']))
    
    # Additional information
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        **About the Model:**
        - Uses Random Forest regression algorithm
        - Trained on historical crop production data from Odisha
        - Features: District, Crop, Season, Year, and Area
        - Model Performance: R² Score > 0.94
        
        **Data Source:**
        - Crop Production in India dataset (Kaggle)
        - Filtered for Odisha state
        - Years covered: 1997-2015
        
        **How to Use:**
        1. Select the district where you want to grow the crop
        2. Choose the type of crop you want to cultivate
        3. Select the appropriate growing season
        4. Enter the year and cultivated area
        5. Click 'Predict Yield' to get the prediction
        
        **Note:** This is a predictive model based on historical data. Actual yields may vary due to weather conditions, soil quality, farming practices, and other factors not captured in the model.
        """)

if __name__ == "__main__":
    main()