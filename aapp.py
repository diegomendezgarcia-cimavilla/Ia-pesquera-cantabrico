# app.py - IA Pesquera Cantábrico
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- Dataset maestro simulado ---
np.random.seed(42)
dates = pd.date_range(start='2005-01-01', end='2024-12-31', freq='D')
n = len(dates)
temp_water = 14 + 4 * np.sin((dates.dayofyear-80)*2*np.pi/365) + np.random.normal(0,0.8,n)
wave_height = np.clip(np.random.normal(1.2,0.7,n),0,None)
wind_speed = np.clip(np.random.normal(15,6,n),0,None)
rain_mm = np.clip(np.random.exponential(2,n)-0.5,0,None)
pressure = np.random.normal(1015,7,n)
pulpo_prob = ((temp_water>16)&(temp_water<20)&(wave_height<1.2)).astype(int)
lubina_prob = ((temp_water>14)&(wave_height>1)&(wave_height<2.5)).astype(int)
percebe_prob = ((wave_height>2)).astype(int)
pulpo_catch = np.where(pulpo_prob, np.random.gamma(2,8,n), np.random.gamma(0.5,2,n))
lubina_catch = np.where(lubina_prob, np.random.gamma(2,5,n), np.random.gamma(0.5,1.5,n))
percebe_catch = np.where(percebe_prob, np.random.gamma(2,4,n), np.random.gamma(0.5,1,n))

df = pd.DataFrame({
    'date': dates,
    'water_temp_C': temp_water.round(2),
    'wave_height_m': wave_height.round(2),
    'wind_speed_kmh': wind_speed.round(1),
    'rain_mm': rain_mm.round(2),
    'pressure_hpa': pressure.round(1),
    'pulpo_catch_kg': pulpo_catch.round(2),
    'lubina_catch_kg': lubina_catch.round(2),
    'percebe_catch_kg': percebe_catch.round(2)
})
df['species'] = df[['pulpo_catch_kg','lubina_catch_kg','percebe_catch_kg']].idxmax(axis=1).str.replace('_catch_kg','')

# --- Entrenamiento IA ---
features = ['water_temp_C','wave_height_m','wind_speed_kmh','rain_mm','pressure_hpa']
X = df[features]
y = df['species']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X,y)

# --- Streamlit App ---
st.set_page_config(page_title="IA Pesquera Cantábrico", layout="centered")
st.title("🌊 IA Pesquera del Cantábrico")
st.write("Predice Pulpo, Lubina o Percebe según condiciones del mar")

temp = st.slider("Temperatura del agua (°C)", 10.0, 24.0, 18.0)
wave = st.slider("Altura de ola (m)", 0.0, 4.0, 1.0)
wind = st.slider("Velocidad del viento (km/h)", 0.0, 60.0, 15.0)
rain = st.slider("Lluvia (mm)", 0.0, 20.0, 0.0)
pressure = st.slider("Presión atmosférica (hPa)", 990.0, 1040.0, 1015.0)

if st.button("Predecir pesca favorable"):
    pred = model.predict([[temp, wave, wind, rain, pressure]])[0]
    st.subheader(f"🎯 Recomendación: {pred.upper()}")
    st.write("Basado en patrones históricos del dataset simulado del Cantábrico.")

st.header("Explorar capturas históricas")
species_select = st.selectbox("Selecciona especie", ["pulpo","lubina","percebe"])
st.line_chart(df[f"{species_select}_catch_kg"])
