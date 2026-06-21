"""
train.py
Treniranje, podesavanje hiperparametara i poredjenje modela
za predikciju cijene kuce.

Modeli: Baseline, Linearna regresija, Ridge, Lasso, KNN,
        Stablo odlucivanja, Random Forest, Gradient Boosting.

Evaluacioni protokol (u skladu sa gradivom):
  1. podjela podataka 70/15/15 (train / validation / test) PRIJE
     bilo kakvog preprocesiranja koje uci iz podataka,
  2. IQR uklanjanje outliera - pragovi se racunaju SAMO na train skupu
     i primjenjuju SAMO na train skup (val/test ostaju netaknuti,
     da bi procjena bila realna),
  3. skaliranje i One-Hot enkodiranje unutar sklearn Pipeline-a
     (fit iskljucivo na train skupu),
  4. hiperparametri se biraju na VALIDACIONOM skupu po MAE,
  5. unakrsna validacija (5-fold) za provjeru stabilnosti - MAE u dolarima,
  6. TEST skup se koristi JEDNOM, tek nakon izbora najboljeg modela.

Target: log(price) - log transformacija stabilizuje raspodjelu cijene;
predikcije se vracaju u dolare pomocu expm1.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

from data_preparation import CATEGORICAL_FEATURES

RANDOM_STATE = 42


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> dict:
    X = df.drop(columns=["price"])
    y_log = np.log1p(df["price"])
    y_orig = df["price"]

    X_train, X_temp, ylog_train, ylog_temp, yorig_train, yorig_temp = train_test_split(
        X, y_log, y_orig, test_size=0.30, random_state=random_state
    )
    X_val, X_test, ylog_val, ylog_test, yorig_val, yorig_test = train_test_split(
        X_temp, ylog_temp, yorig_temp, test_size=0.50, random_state=random_state
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "ylog_train": ylog_train, "ylog_val": ylog_val, "ylog_test": ylog_test,
        "yorig_train": yorig_train, "yorig_val": yorig_val, "yorig_test": yorig_test,
    }


def iqr_filter_train(X_train, ylog_train, yorig_train, iqr_factor: float = 3.0):
    df_tr = X_train.copy()
    df_tr["price"] = yorig_train.values

    mask = pd.Series(True, index=df_tr.index)
    mask &= df_tr["bedrooms"] <= 8

    for col in ["price", "sqft_living", "sqft_lot"]:
        Q1 = df_tr[col].quantile(0.25)
        Q3 = df_tr[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        mask &= (df_tr[col] >= lower_bound) & (df_tr[col] <= upper_bound)

    return X_train[mask], ylog_train[mask], yorig_train[mask]


def outlier_analysis(X_train, ylog_train, yorig_train, X_val, ylog_val, yorig_val):
    print("\n[ANALIZA] Uticaj IQR uklanjanja outliera (validacioni skup):")
    Xf, yf, _ = iqr_filter_train(X_train, ylog_train, yorig_train)
    print(f"  IQR bi uklonio {len(X_train) - len(Xf)} od {len(X_train)} "
          f"trening redova ({(len(X_train) - len(Xf)) / len(X_train):.1%})")

    probes = [("Ridge(alpha=1)", lambda: Ridge(alpha=1.0)),
              ("GradBoost(lr=0.1,n=200)",
               lambda: GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, random_state=RANDOM_STATE))]

    for label, (Xt, yt) in [("svi podaci", (X_train, ylog_train)), ("IQR filtrirano", (Xf, yf))]:
        for pname, factory in probes:
            pipe = make_pipeline(factory(), Xt)
            pipe.fit(Xt, yt)
            pred_log = pipe.predict(X_val)
            r2l = r2_score(ylog_val, pred_log)
            mae = mean_absolute_error(yorig_val, np.expm1(pred_log))
            print(f"    {label:15s} | {pname:24s} | R2(log)={r2l:.4f} | MAE=${mae:,.0f}")
    print("  => Zakljucak: outlieri su legitimne skupe kuce; treniranje "
          "se nastavlja na SVIM trening podacima.")


def make_pipeline(model, X: pd.DataFrame) -> Pipeline:
    """
    Kreira Pipeline: preprocesiranje + model.
    Automatski detektuje koje kategorijske kolone stvarno postoje u X.
    """
    actual_cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    numeric_features = [c for c in X.columns if c not in actual_cat]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), actual_cat),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def compute_metrics(name, y_true_orig, y_pred_orig, y_true_log, y_pred_log, split="VAL") -> dict:
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    r2_log = r2_score(y_true_log, y_pred_log)

    print(f"[{split}] {name} | MAE: {mae:.0f} | RMSE: {rmse:.0f} | R2: {r2:.4f} | R2(log): {r2_log:.4f}")

    return {
        "model": name,
        "split": split,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "R2_log": round(r2_log, 4)
    }


def eval_pipeline(name, pipe, X, ylog, yorig, split="VAL") -> dict:
    pred_log = pipe.predict(X)
    pred_orig = np.expm1(pred_log)
    return compute_metrics(name, yorig, pred_orig, ylog, pred_log, split)


def train_all(processed_path: str, models_dir: str, metrics_dir: str) -> None:
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    df = pd.read_csv(processed_path, dtype={"zipcode": str})
    print(f"[INFO] Ucitano: {df.shape[0]} uzoraka, {df.shape[1] - 1} atributa")

    s = split_data(df)
    print(f"[INFO] Train: {len(s['X_train'])} | Val: {len(s['X_val'])} | Test: {len(s['X_test'])}")

    outlier_analysis(s["X_train"], s["ylog_train"], s["yorig_train"],
                     s["X_val"], s["ylog_val"], s["yorig_val"])

    X_train, ylog_train, yorig_train = s["X_train"], s["ylog_train"], s["yorig_train"]
    X_val, ylog_val, yorig_val = s["X_val"], s["ylog_val"], s["yorig_val"]
    X_test, ylog_test, yorig_test = s["X_test"], s["ylog_test"], s["yorig_test"]

    all_metrics = []
    candidates = {}

    print("\n" + "=" * 95)
    print("TRENIRANJE I IZBOR HIPERPARAMETARA - REZULTATI NA VALIDACIONOM SKUPU")
    print("=" * 95)

    # Baseline
    base_pred_log = np.full(len(ylog_val), ylog_train.mean())
    m = compute_metrics("Baseline (prosjek train skupa)", yorig_val,
                        np.expm1(base_pred_log), ylog_val, base_pred_log)
    all_metrics.append(m)

    # Linearna regresija
    lr = make_pipeline(LinearRegression(), X_train)
    lr.fit(X_train, ylog_train)
    m = eval_pipeline("Linearna regresija", lr, X_val, ylog_val, yorig_val)
    all_metrics.append(m)
    candidates["Linearna regresija"] = (lr, m["MAE"])

    # Ridge
    print("\n  Ridge - pretraga alpha:")
    best_pipe, best_mae, best_alpha = None, np.inf, None
    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        pipe = make_pipeline(Ridge(alpha=alpha), X_train)
        pipe.fit(X_train, ylog_train)
        mae = mean_absolute_error(yorig_val, np.expm1(pipe.predict(X_val)))
        print(f"    alpha={alpha:8.1f}  MAE=${mae:,.0f}")
        if mae < best_mae:
            best_pipe, best_mae, best_alpha = pipe, mae, alpha
    name = f"Ridge (alpha={best_alpha})"
    print(f"  => Najbolji alpha: {best_alpha}")
    m = eval_pipeline(name, best_pipe, X_val, ylog_val, yorig_val)
    all_metrics.append(m)
    candidates[name] = (best_pipe, m["MAE"])

    # Lasso
    print("\n  Lasso - pretraga alpha:")
    best_pipe, best_mae, best_alpha = None, np.inf, None
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        pipe = make_pipeline(Lasso(alpha=alpha, max_iter=20000), X_train)
        pipe.fit(X_train, ylog_train)
        mae = mean_absolute_error(yorig_val, np.expm1(pipe.predict(X_val)))
        print(f"    alpha={alpha:8.4f}  MAE=${mae:,.0f}")
        if mae < best_mae:
            best_pipe, best_mae, best_alpha = pipe, mae, alpha
    name = f"Lasso (alpha={best_alpha})"
    print(f"  => Najbolji alpha: {best_alpha}")
    m = eval_pipeline(name, best_pipe, X_val, ylog_val, yorig_val)
    all_metrics.append(m)
    candidates[name] = (best_pipe, m["MAE"])

    # KNN
    print("\n  KNN - pretraga broja suseda K:")
    best_pipe, best_mae, best_k = None, np.inf, None
    knn_results = {}
    for k in range(1, 21):
        pipe = make_pipeline(KNeighborsRegressor(n_neighbors=k), X_train)
        pipe.fit(X_train, ylog_train)
        mae = mean_absolute_error(yorig_val, np.expm1(pipe.predict(X_val)))
        knn_results[k] = round(mae, 2)
        if mae < best_mae:
            best_pipe, best_mae, best_k = pipe, mae, k
    print(f"    K rezultati: {knn_results}")
    name = f"KNN (K={best_k})"
    print(f"  => Najbolji K (min validacioni MAE): {best_k}")
    m = eval_pipeline(name, best_pipe, X_val, ylog_val, yorig_val)
    all_metrics.append(m)
    candidates[name] = (best_pipe, m["MAE"])
    with open(os.path.join(metrics_dir, "knn_elbow.json"), "w") as f:
        json.dump(knn_results, f)

    # Decision Tree
    print("\n  Stablo odlucivanja - pretraga max_depth / min_samples_leaf:")
    best_pipe, best_mae, best_params = None, np.inf, None
    for depth in [5, 7, 10, 15, None]:
        for leaf in [1, 5, 10, 20]:
            pipe = make_pipeline(
                DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=RANDOM_STATE), X_train)
            pipe.fit(X_train, ylog_train)
            mae = mean_absolute_error(yorig_val, np.expm1(pipe.predict(X_val)))
            if mae < best_mae:
                best_pipe, best_mae, best_params = pipe, mae, (depth, leaf)
    name = f"Decision Tree (depth={best_params[0]}, leaf={best_params[1]})"
    print(f"  => Najbolji: max_depth={best_params[0]}, min_samples_leaf={best_params[1]}  (MAE=${best_mae:,.0f})")
    m = eval_pipeline(name, best_pipe, X_val, ylog_val, yorig_val)
    all_metrics.append(m)
    candidates[name] = (best_pipe, m["MAE"])

    # Random Forest
    print("\n  Random Forest - pretraga n_estimators / max_features:")
    best_pipe, best_mae, best_params = None, np.inf, None
    for n_est in [100, 200]:
        for max_feat in ["sqrt", 0.5]:
            pipe = make_pipeline(
                RandomForestRegressor(n_estimators=n_est, max_features=max_feat, random_state=RANDOM_STATE, n_jobs=-1), X_train)
            pipe.fit(X_train, ylog_train)
            mae = mean_absolute_error(yorig_val, np.expm1(pipe.predict(X_val)))
            print(f"    n_estimators={n_est:4d}, max_features={str(max_feat):5s}  MAE=${mae:,.0f}")
            if mae < best_mae:
                best_pipe, best_mae, best_params = pipe, mae, (n_est, max_feat)
    name = f"Random Forest (n={best_params[0]}, max_feat={best_params[1]})"
    print(f"  => Najbolji: n_estimators={best_params[0]}, max_features={best_params[1]}")
    m = eval_pipeline(name, best_pipe, X_val, ylog_val, yorig_val)
    all_metrics.append(m)
    candidates[name] = (best_pipe, m["MAE"])

    # Gradient Boosting
    print("\n  Gradient Boosting - pretraga learning_rate / n_estimators:")
    best_pipe, best_mae, best_params = None, np.inf, None
    for lrate in [0.05, 0.1]:
        for n_est in [200, 400]:
            pipe = make_pipeline(
                GradientBoostingRegressor(learning_rate=lrate, n_estimators=n_est, random_state=RANDOM_STATE), X_train)
            pipe.fit(X_train, ylog_train)
            mae = mean_absolute_error(yorig_val, np.expm1(pipe.predict(X_val)))
            print(f"    learning_rate={lrate:.2f}, n_estimators={n_est:4d}  MAE=${mae:,.0f}")
            if mae < best_mae:
                best_pipe, best_mae, best_params = pipe, mae, (lrate, n_est)
    name = f"Gradient Boosting (lr={best_params[0]}, n={best_params[1]})"
    print(f"  => Najbolji: learning_rate={best_params[0]}, n_estimators={best_params[1]}")
    m = eval_pipeline(name, best_pipe, X_val, ylog_val, yorig_val)
    all_metrics.append(m)
    candidates[name] = (best_pipe, m["MAE"])

    # Izbor najboljeg modela na osnovu VALIDACIONOG MAE
    best_name = min(candidates, key=lambda k: candidates[k][1])
    best_pipeline = candidates[best_name][0]
    print("\n" + "=" * 95)
    print(f"IZABRAN NAJBOLJI MODEL (po validacionom MAE): {best_name}")
    print("=" * 95)

    # ✅ Unakrsna validacija - custom scorer koji vraca MAE u dolarima
    # Pipeline predvidja log(price), pa moramo expm1 da dobijemo dolare.
    # Koristimo make_scorer sa funkcijom koja radi tu konverziju.
    def mae_dollars(y_true_log, y_pred_log):
        return mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))

    dollar_scorer = make_scorer(mae_dollars, greater_is_better=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # ISPRAVKA: saljemo ylog_train (ne yorig_train) jer pipeline predvidja log(price)
    cv_scores = cross_val_score(best_pipeline, X_train, ylog_train,
                                cv=kf, scoring=dollar_scorer)
    cv_mae = -cv_scores  # vrati u pozitivne vrijednosti (dolari)
    print(f"  CV MAE, 5-fold na train skupu: "
          f"${cv_mae.mean():,.0f} +/- ${cv_mae.std():,.0f}")

    # Provjera overfittinga (R2 na log skali - dijagnosticki, nije odlucna metrika)
    train_pred = best_pipeline.predict(X_train)
    val_pred = best_pipeline.predict(X_val)
    train_r2 = r2_score(ylog_train, train_pred)
    val_r2 = r2_score(ylog_val, val_pred)
    print("\nProvjera overfittinga:")
    print(f"Train R2(log): {train_r2:.4f}")
    print(f"Validation R2(log): {val_r2:.4f}")
    print(f"Razlika: {train_r2 - val_r2:.4f}")

    # Finalna evaluacija na TEST skupu
    print("\n" + "=" * 95)
    print("FINALNA EVALUACIJA NA TEST SKUPU (samo izabrani model)")
    print("=" * 95)
    m_test = eval_pipeline(best_name, best_pipeline, X_test, ylog_test, yorig_test, split="TEST")
    all_metrics.append(m_test)

    # Cuvanje modela
    for name, (pipe, _) in candidates.items():
        fname = (name.split(" (")[0].lower().replace(" ", "_") + ".joblib")
        joblib.dump(pipe, os.path.join(models_dir, fname))
    joblib.dump(best_pipeline, os.path.join(models_dir, "final_model.joblib"))

    # Meta podaci - CV MAE u dolarima
    meta = {
        "best_model": best_name,
        "cv_mae_mean": round(cv_mae.mean(), 2),
        "cv_mae_std": round(cv_mae.std(), 2),
        "metric_for_selection": "MAE",
        "raw_input_columns": list(X_train.columns),
        "categorical_features": CATEGORICAL_FEATURES,
        "target": "log1p(price), predikcija se vraca sa expm1",
    }
    with open(os.path.join(models_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[INFO] Modeli sacuvani u: {models_dir}")
    print(f"[INFO] Finalni model: models/final_model.joblib ({best_name})")
    print(f"[INFO] Metrike sacuvane u: {metrics_dir}/metrics.json")


if __name__ == "__main__":
    train_all(
        processed_path="data/processed/data_clean.csv",
        models_dir="models/",
        metrics_dir="results/metrics/",
    )
