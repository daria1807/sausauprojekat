"""
ui.py
Streamlit aplikacija za predikciju cijene kuce.

Princip (kao na vjezbi 6): aplikacija ucitava JEDAN sacuvani Pipeline
(preprocesiranje + model) i salje mu SIROVE atribute - enkodiranje i
skaliranje radi pipeline interno.

Pokretanje iz glavnog foldera projekta:
  streamlit run app/ui.py
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.json"


@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return pipeline, meta


pipeline, meta = load_model()

# Gradovi i ZIP kodovi koje je model vidio u trening skupu
pre = pipeline.named_steps["preprocessor"]
ohe = pre.named_transformers_["cat"]
cat_cols = pre.transformers_[1][2]
categories = {col: sorted(cats) for col, cats in zip(cat_cols, ohe.categories_)}

st.title("House Price Prediction")
st.write(
    "Aplikacija procjenjuje cijenu kuce (King County, WA) na osnovu "
    f"njenih karakteristika. Model: **{meta['best_model']}**"
)

st.subheader("Karakteristike kuce")

col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Broj spavacih soba", 1, 10, 3)
    sqft_living = st.number_input("Stambena povrsina (ft²)", 200, 15000, 1800)
    floors = st.selectbox("Broj spratova", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)
    condition = st.slider("Stanje kuce (1-5)", 1, 5, 3)
    house_age = st.number_input("Starost kuce (godina)", 0, 150, 30)
    city = st.selectbox("Grad", categories["city"],
                        index=categories["city"].index("Seattle")
                        if "Seattle" in categories["city"] else 0)
with col2:
    bathrooms = st.number_input("Broj kupatila", 0.5, 8.0, 2.0, step=0.25)
    sqft_lot = st.number_input("Povrsina parcele (ft²)", 500, 200000, 7500)
    waterfront = st.selectbox("Pogled na vodu", [0, 1],
                              format_func=lambda x: "Da" if x else "Ne")
    view = st.slider("Ocjena pogleda (0-4)", 0, 4, 0)
    is_renovated = st.selectbox("Renovirana", [0, 1],
                                format_func=lambda x: "Da" if x else "Ne")
    zipcode = st.selectbox("ZIP kod", categories["zipcode"])

sqft_above = st.number_input("Povrsina iznad zemlje (ft²)", 200, 15000,
                             int(sqft_living))
sqft_basement = st.number_input("Povrsina podruma (ft²)", 0, 5000, 0)
years_since_ren = st.number_input("Godina od renovacije", 0, 120, 0,
                                  disabled=(is_renovated == 0))

if st.button("Procijeni cijenu"):
    input_data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "house_age": house_age,
        "is_renovated": is_renovated,
        "years_since_ren": years_since_ren if is_renovated else 0,
        "sqft_ratio": sqft_living / (sqft_lot + 1),
        "city": city,
        "zipcode": zipcode,
    }])

    # Poredaj kolone onako kako ih je pipeline vidio pri treniranju
    input_data = input_data[meta["raw_input_columns"]]

    if sqft_above > sqft_living:
        st.warning("Povrsina iznad zemlje ne moze biti veca od ukupne "
                   "stambene povrsine - provjeri unos.")

    pred_log = pipeline.predict(input_data)[0]
    price = float(np.expm1(pred_log))

    st.subheader("Rezultat")
    st.write("Ulazni podaci")
    st.dataframe(input_data)
    st.success(f"Procijenjena cijena: **${price:,.0f}**")

# Pokretanje iz glavnog foldera projekta:
# streamlit run app/ui.pywhere python

