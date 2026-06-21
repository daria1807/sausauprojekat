"""
predict.py
Ucitava finalni sacuvani Pipeline (preprocesiranje + model u jednom
objektu) i vrsi predikciju za nove primjere.

Posto je preprocesiranje (skaliranje + One-Hot enkodiranje) dio
Pipeline-a, ulaz su SIROVI atributi - ukljucujuci city i zipcode kao
stringove. Nije potrebno rucno enkodiranje niti rekonstrukcija kolona.

Model je treniran na log(price), pa se predikcija vraca kroz expm1.

Primjer poziva:
  python src/predict.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
MODELS_DIR = os.path.join(_ROOT_DIR, "models")

# Granice validacije ulaznih atributa (min, max) - upozorenja, ne greske
_VALID_RANGES = {
    "bedrooms":        (1,    10),
    "bathrooms":       (0.5,  8.0),
    "sqft_living":     (200,  15000),
    "sqft_lot":        (500,  200000),
    "floors":          (1.0,  4.0),
    "waterfront":      (0,    1),
    "view":            (0,    4),
    "condition":       (1,    5),
    "sqft_above":      (200,  15000),
    "sqft_basement":   (0,    5000),
    "house_age":       (0,    150),
    "is_renovated":    (0,    1),
    "years_since_ren": (0,    120),
}


def load_model(models_dir: str = MODELS_DIR):
    """
    Ucitava finalni pipeline i meta podatke.

    Returns:
        (pipeline, meta: dict) - meta sadrzi listu ulaznih kolona,
        naziv najboljeg modela i CV rezultate.
    """
    model_path = os.path.join(models_dir, "final_model.joblib")
    meta_path = os.path.join(models_dir, "model_meta.json")
    for path in (model_path, meta_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fajl nije pronadjen: {path}")

    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"[INFO] Ucitan finalni model : {meta['best_model']}")
    
    # Sada koristimo MAE umjesto R2 (u skladu sa train.py)
    if "cv_mae_mean" in meta:
        print(f"[INFO] CV MAE               : ${meta['cv_mae_mean']:,.0f} "
              f"+/- ${meta['cv_mae_std']:,.0f}")
    else:
        # Fallback za stare verzije (ako neko pokrene stari model)
        print(f"[INFO] CV R2(log)           : {meta.get('cv_r2_log_mean', 'N/A')} "
              f"+/- {meta.get('cv_r2_log_std', 'N/A')}")
    
    print(f"[INFO] Ulazne kolone ({len(meta['raw_input_columns'])}): "
          f"{meta['raw_input_columns']}")
    return pipeline, meta


def known_categories(pipeline) -> dict:
    """
    Vraca kategorije (gradove, ZIP kodove) koje je OneHotEncoder vidio
    u trening skupu.
    """
    pre = pipeline.named_steps["preprocessor"]
    ohe = pre.named_transformers_["cat"]
    cat_cols = pre.transformers_[1][2]

    return {
        col: list(cats)
        for col, cats in zip(cat_cols, ohe.categories_)
    }


def _validate_sample(sample: dict, categories: dict) -> list:
    """Vraca listu upozorenja (prazna lista = sve OK)."""
    warnings_out = []

    for attr, (lo, hi) in _VALID_RANGES.items():
        if attr in sample and not (lo <= sample[attr] <= hi):
            warnings_out.append(
                f"  [UPOZORENJE] '{attr}' = {sample[attr]} je van "
                f"ocekivanog opsega [{lo}, {hi}]")

    if sample.get("sqft_above", 0) > sample.get("sqft_living", float("inf")):
        warnings_out.append(
            "  [UPOZORENJE] 'sqft_above' ne moze biti vece od 'sqft_living'")

    for col, cats in categories.items():
        val = sample.get(col)
        if val is not None and val not in cats:
            warnings_out.append(
                f"  [UPOZORENJE] '{col}' = '{val}' nije vidjen u trening "
                f"skupu - tretira se kao nepoznata kategorija (sve nule).")
    return warnings_out


def predict(sample: dict, pipeline, meta: dict, validate: bool = True) -> float:
    """
    Predvidja cijenu kuce na osnovu sirovih atributa.

    Args:
        sample  : rjecnik atributa, npr.
                  {"bedrooms": 3, ..., "city": "Seattle", "zipcode": "98103"}
                  sqft_ratio se automatski racuna ako nije naveden.
        pipeline: finalni Pipeline (iz load_model).
        meta    : meta podaci (iz load_model).
        validate: ako True, ispisuje upozorenja za sumnjive vrijednosti.

    Returns:
        Predvidjena cijena u USD (float).
    """
    sample = dict(sample)

    # Automatski izracunaj sqft_ratio ako nije naveden
    if "sqft_ratio" not in sample and \
            "sqft_living" in sample and "sqft_lot" in sample:
        sample["sqft_ratio"] = sample["sqft_living"] / (sample["sqft_lot"] + 1)

    if validate:
        issues = _validate_sample(sample, known_categories(pipeline))
        if issues:
            print("[VALIDACIJA]")
            for w in issues:
                print(w)

    # DataFrame sa svim ocekivanim sirovim kolonama
    row = {col: sample.get(col, np.nan) for col in meta["raw_input_columns"]}
    missing = [c for c, v in row.items() if pd.isna(v)]
    if missing:
        raise ValueError(f"Nedostaju atributi: {missing}")

    X_new = pd.DataFrame([row])
    pred_log = pipeline.predict(X_new)[0]
    return round(float(np.expm1(pred_log)), 2)


def demo():
    print("=" * 65)
    print("  PREDIKCIJA CIJENE KUCE - demo (finalni model)")
    print("=" * 65)

    pipeline, meta = load_model()
    cats = known_categories(pipeline)
    print(f"\n[INFO] Primjer gradova: {sorted(cats['city'])[:8]} ...")

    examples = [
        {
            "label": "Manja, starija kuca u Shoreline-u",
            "bedrooms": 2, "bathrooms": 1.0,
            "sqft_living": 900, "sqft_lot": 5000, "floors": 1.0,
            "waterfront": 0, "view": 0, "condition": 3,
            "sqft_above": 900, "sqft_basement": 0,
            "house_age": 60, "is_renovated": 0, "years_since_ren": 0,
            "city": "Shoreline", "zipcode": "98133",
        },
        {
            "label": "Prosjecna kuca u Seattleu",
            "bedrooms": 3, "bathrooms": 2.0,
            "sqft_living": 1800, "sqft_lot": 7500, "floors": 2.0,
            "waterfront": 0, "view": 0, "condition": 3,
            "sqft_above": 1800, "sqft_basement": 0,
            "house_age": 30, "is_renovated": 0, "years_since_ren": 0,
            "city": "Seattle", "zipcode": "98103",
        },
        {
            "label": "Luksuzna kuca u Bellevueu (renovirana, pogled na vodu)",
            "bedrooms": 5, "bathrooms": 3.5,
            "sqft_living": 4000, "sqft_lot": 12000, "floors": 2.0,
            "waterfront": 1, "view": 4, "condition": 5,
            "sqft_above": 3000, "sqft_basement": 1000,
            "house_age": 10, "is_renovated": 1, "years_since_ren": 5,
            "city": "Bellevue", "zipcode": "98004",
        },
        {
            "label": "Test validacije: neispravne vrijednosti",
            "bedrooms": -1,
            "bathrooms": 2.0,
            "sqft_living": 1500, "sqft_lot": 6000, "floors": 1.0,
            "waterfront": 0, "view": 0, "condition": 3,
            "sqft_above": 2000,
            "sqft_basement": 0,
            "house_age": 20, "is_renovated": 0, "years_since_ren": 0,
            "city": "Seatle",
            "zipcode": "98103",
        },
    ]

    print()
    for ex in examples:
        label = ex.pop("label")
        print(f"  [{label}]")
        price = predict(ex, pipeline, meta, validate=True)
        print(f"    sqft_living={ex['sqft_living']}, bedrooms={ex['bedrooms']}, "
              f"waterfront={ex['waterfront']}, grad={ex['city']}")
        print(f"    => Procijenjena cijena: ${price:,.0f}\n")


if __name__ == "__main__":
    demo()