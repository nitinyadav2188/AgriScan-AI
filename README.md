# 🌾 AgriScan AI - Crop Yield Prediction for Odisha

AgriScan AI is an advanced agricultural application that combines plant health diagnostics with AI-powered crop yield prediction specifically for Odisha, India. This repository contains both the original plant disease detection functionality and a new crop yield prediction system.

## 🆕 New Feature: Crop Yield Prediction

The latest addition to AgriScan AI is an intelligent crop yield prediction system that helps farmers in Odisha make informed decisions about crop cultivation. The system uses a Random Forest regression model trained on historical crop production data to predict expected yields.

### 🎯 Features

- **AI-Powered Predictions**: Uses Random Forest machine learning algorithm
- **Odisha-Specific**: Trained exclusively on Odisha crop production data
- **Comprehensive Coverage**: Supports 30 districts, 39 crops, and 6 seasons
- **User-Friendly Interface**: Interactive Streamlit web application
- **Real-Time Predictions**: Instant yield calculations based on input parameters

### 📊 Model Performance

- **Algorithm**: Random Forest Regression
- **R² Score**: 0.9434 (94.34% accuracy)
- **RMSE**: 0.7959
- **Training Data**: 13,000+ cleaned records from 1997-2015

### 🗺️ Coverage

**Districts (30)**: All major districts in Odisha including:
- ANUGUL, BALANGIR, BALESHWAR, BARGARH, BHADRAK, BOUDH
- CUTTACK, DEOGARH, DHENKANAL, GAJAPATI, GANJAM, JAGATSINGHAPUR
- And 18 more districts...

**Crops (39)**: Major crops including:
- Rice (Paddy), Wheat, Maize, Jowar, Bajra
- Arhar/Tur, Gram, Groundnut, Sugarcane
- Cotton, Jute, Coconut, and many more...

**Seasons (6)**: 
- Kharif, Rabi, Autumn, Summer, Winter, Whole Year

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nitinyadav2188/AgriScan-AI.git
   cd AgriScan-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (one-time setup):
   ```bash
   python train_model.py
   ```
   This will:
   - Process the crop production data for Odisha
   - Train the Random Forest model
   - Save the model and encoders
   - Create reference files for the web app

4. **Run the web application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to the displayed URL (usually `http://localhost:8501`)

## 📁 Project Structure

```
AgriScan-AI/
├── crop_production.csv      # Raw dataset (Kaggle: crop-production-in-india)
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── index.html             # Original plant disease detection app
├── manifest.json          # PWA manifest
├── service-worker.js      # Service worker for PWA
└── Generated files (after training):
    ├── crop_yield_model.pkl     # Trained Random Forest model
    ├── label_encoders.pkl       # Label encoders for categorical data
    └── unique_values.json       # Reference data for the web app
```

## 🔧 Usage

### Web Application

1. **Select District**: Choose from 30 districts in Odisha
2. **Select Crop**: Pick from 39 supported crops
3. **Choose Season**: Select the appropriate growing season
4. **Set Year**: Use the slider to choose the year
5. **Enter Area**: Input the cultivated area in hectares
6. **Get Prediction**: Click "Predict Yield" for instant results

### Command Line (Advanced)

You can also use the model programmatically:

```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load('crop_yield_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# Prepare input (example)
district_encoded = encoders['District_Name'].transform(['CUTTACK'])[0]
crop_encoded = encoders['Crop'].transform(['Paddy'])[0]
season_encoded = encoders['Season'].transform(['Kharif'])[0]
year = 2015
area = 100.0

# Make prediction
features = np.array([[district_encoded, crop_encoded, season_encoded, year, area]])
yield_prediction = model.predict(features)[0]

print(f"Predicted yield: {yield_prediction:.2f} tons/hectare")
```

## 📈 Model Details

### Data Preprocessing

1. **Filtering**: Extracted Odisha-specific data from the national dataset
2. **Cleaning**: Removed records with missing or invalid production/area values
3. **Outlier Removal**: Eliminated yield values beyond 3 standard deviations
4. **Feature Engineering**: Calculated yield as Production/Area ratio

### Feature Importance

Based on model analysis:

1. **Crop Type** (75.8%): Most important factor
2. **Season** (12.6%): Second most important
3. **Area** (5.7%): Moderate importance
4. **Year** (3.1%): Minor importance
5. **District** (2.9%): Least important

### Model Configuration

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

## 📊 Data Source

The model is trained on the "Crop Production in India" dataset available on Kaggle:
- **Source**: [abhinand05/crop-production-in-india](https://www.kaggle.com/abhinand05/crop-production-in-india)
- **Coverage**: Agricultural data from 1997-2015
- **Scope**: Filtered for Odisha state only
- **Records**: 13,000+ processed records after cleaning

## 🛠️ Dependencies

```
streamlit        # Web application framework
pandas          # Data manipulation and analysis
scikit-learn    # Machine learning library
numpy           # Numerical computing
joblib          # Model serialization
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Thanks to the contributors of the Crop Production in India dataset on Kaggle
- **Government of Odisha**: For maintaining agricultural statistics
- **Open Source Community**: For the amazing tools and libraries used in this project

## 📞 Support

If you have any questions or issues:

1. Check the [Issues](https://github.com/nitinyadav2188/AgriScan-AI/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide as much detail as possible including error messages and system information

## 🔮 Future Enhancements

- [ ] Weather data integration for improved predictions
- [ ] Soil quality parameters inclusion
- [ ] Market price prediction
- [ ] Mobile app version
- [ ] Real-time satellite data integration
- [ ] Multi-language support (Hindi, Odia)

---

**Note**: This prediction model is based on historical data and should be used as a reference tool. Actual crop yields may vary due to various factors including weather conditions, soil quality, farming practices, and other environmental factors not captured in the historical dataset.