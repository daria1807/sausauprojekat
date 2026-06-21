"""
api.py
FastAPI servis za predikciju cijene kuce.

Princip (kao na vjezbi 6): ucitava se JEDAN sacuvani Pipeline i salju mu
se SIROVI atributi - enkodiranje i skaliranje radi pipeline interno.
Pydantic model validira ulaz (tipovi i opsezi).

Pokretanje iz glavnog foldera projekta:
  uvicorn app.api:app --reload
Dokumentacija: http://127.0.0.1:8000/docs
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="House Price API",
    description="API za predikciju cijene kuce (King County) "
                "koriscenjem sacuvanog ML pipeline-a.",
    version="1.0",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"
META_PATH = BASE_DIR / "models" / "model_meta.json"

pipeline = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)


class HouseInput(BaseModel):
    """Ulazni podaci - opsezi validirani Pydantic-om."""
    bedrooms: int = Field(ge=1, le=10, examples=[3])
    bathrooms: float = Field(ge=0.5, le=8.0, examples=[2.0])
    sqft_living: int = Field(ge=200, le=15000, examples=[1800])
    sqft_lot: int = Field(ge=500, le=200000, examples=[7500])
    floors: float = Field(ge=1.0, le=4.0, examples=[2.0])
    waterfront: int = Field(ge=0, le=1, examples=[0])
    view: int = Field(ge=0, le=4, examples=[0])
    condition: int = Field(ge=1, le=5, examples=[3])
    sqft_above: int = Field(ge=200, le=15000, examples=[1800])
    sqft_basement: int = Field(ge=0, le=5000, examples=[0])
    house_age: int = Field(ge=0, le=150, examples=[30])
    is_renovated: int = Field(ge=0, le=1, examples=[0])
    years_since_ren: int = Field(ge=0, le=120, examples=[0])
    city: str = Field(examples=["Seattle"])
    zipcode: str = Field(examples=["98103"])


@app.get("/")
def home():
    return {
        "message": "House Price API is running.",
        "model": meta["best_model"],
    }


@app.post("/predict")
def predict(house: HouseInput):
    data = house.model_dump()
    data["sqft_ratio"] = data["sqft_living"] / (data["sqft_lot"] + 1)

    input_data = pd.DataFrame([data])[meta["raw_input_columns"]]
    pred_log = pipeline.predict(input_data)[0]
    price = round(float(np.expm1(pred_log)), 2)

    return {
        "predicted_price_usd": price,
        "model": meta["best_model"],
    }
