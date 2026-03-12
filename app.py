import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🌊 IA Pesquera del Cantábrico")

# dataset pequeño
X = [
    [18,1.0],
    [19,0.8],
    [14,2.0],
    [13,2.5],
    [17,1.5],
    [16,0.7]
]

y = ["pulpo","pulpo","percebe","percebe","lubina","lubina"]

model = RandomForestClassifier()
model.fit(X,y)

st.write("Introduce condiciones del mar")

temp = st.slider("Temperatura agua",10,24,18)
ola = st.slider("Altura ola",0.0,4.0,1.0)

if st.button("Predecir pesca"):
    pred = model.predict([[temp,ola]])[0]
    st.success(f"Pesca favorable: {pred}")
