# eSwatini Wildfire Predictor 🔥

The **eSwatini Wildfire Predictor** is a data-driven intelligence platform designed to enhance disaster preparedness and fire management in the Kingdom of Eswatini. By shifting from reactive firefighting to **anticipatory prevention**, this platform leverages advanced Machine Learning to identify wildfire risks before they escalate.

## 🚀 Key Features

- **Wildfire Risk Prediction**: Uses a high-performance **XGBoost** model to calculate fire probability based on real time environmental data.
- **Deep Data Analysis**: Explore spatiotemporal drivers of fire, including meteorology (ERA5), vegetation indices (NASA FIRMS/GEE), and human activity.
- **Interactive Mapping**: Geographic distribution of risk factors and historical wildfire clusters using Folium and MarkerClusters.
- **Scenario Analysis**: "What if" checks to see how changes in temperature or precipitation impact regional risk.
- **Mobile Responsive**: Optimised for both desktop and mobile field use with a sleek, user-friendly interface.

## 🧠 Technology Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: XGBoost, Scikit-Learn
- **Visualization**: Plotly, Folium
- **Data Sources**: NASA FIRMS, Google Earth Engine, ERA5 (Copernicus)

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lunga-N/wildfire_app.git
   cd wildfire_app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🤝 Contact

- **Developer**: Lunga Ndzimandze
- **Email**: ndzimandzelunga@gmail.com

---
*Lunga Ndzimandze © 2025*
