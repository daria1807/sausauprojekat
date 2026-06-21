"""
evaluate.py
EDA vizualizacije, poredjenje modela i analiza znacajnosti atributa.

Principi (u skladu sa gradivom):
  - koristi ISTU podjelu podataka kao train.py (split_data, random_state=42),
  - EDA se radi na TRENING skupu - odluke o obradi podataka ne smiju
    biti donesene na osnovu test podataka,
  - znacajnost atributa (permutation importance) se racuna na
    VALIDACIONOM skupu,
  - poredjenje "svi vs top atributi" se radi na VALIDACIONOM skupu,
    jer je izbor podskupa atributa odluka u razvoju modela
    (test skup sluzi samo za finalnu procjenu u train.py).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.inspection import permutation_importance

from data_preparation import CATEGORICAL_FEATURES
from train import split_data, make_pipeline

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

"""A KAKO SU NEGATIVNI BEDROOMS I FLOORS1:58 AMClaude responded: Odlično pitanje — to treba znati objasniti na odbrani!Odlično pitanje — to treba znati objasniti na odbrani!

Zašto su bedrooms i floors negativni kod GB?
Negativna vrijednost u permutation importance znači: kada izmješamo (permutujemo) taj atribut, MAE se smanjuje — što znači model bez tog atributa radi bolje.
Razlog je multikolinearnost i redundantnost:

bedrooms je visoko korelisan sa sqft_living — veća kuća ima više spavaćih soba. GB već hvata tu informaciju kroz sqft_living, pa kad mu damo i bedrooms, taj atribut unosi šum više nego korisnu informaciju
floors je slično — broj spratova je implicitno sadržan u sqft_living i sqft_above

GB kao model sam uči da ignoriše te atribute, ali kada su prisutni u podacima mogu blago narušiti odluke u nekim stablima. Kada ih permutujemo, slučajno uklonimo taj šum i MAE se malo poboljša."""
def save_fig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ─── 1. EDA (na trening skupu) ────────────────────────────────────────────────

def eda(df_train: pd.DataFrame):
    print("\n[EDA] Generisanje grafika (trening skup)...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_train["price"], bins=60, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribucija cijene (originalna)")
    axes[0].set_xlabel("Cijena ($)")
    axes[0].set_ylabel("Broj kuca")
    axes[1].hist(np.log1p(df_train["price"]), bins=60,
                 color="darkorange", edgecolor="white")
    axes[1].set_title("Distribucija log(cijena)")
    axes[1].set_xlabel("log(Cijena)")
    save_fig("01_price_distribution.png")

    num_cols = df_train.select_dtypes(include=np.number).columns
    corr = df_train[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Korelaciona matrica atributa (trening skup)")
    save_fig("02_correlation_matrix.png")

    corr_price = corr["price"].drop("price").abs().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c" if v > 0.5 else "#3498db" for v in corr_price]
    corr_price.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Korelacija atributa sa cijenom (apsolutna vrijednost)")
    ax.set_xlabel("|r|")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="r=0.5")
    ax.legend()
    save_fig("03_price_correlations.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(df_train["sqft_living"], df_train["price"],
                    alpha=0.3, s=8, color="steelblue")
    axes[0].set_xlabel("sqft_living (stambena povrsina)")
    axes[0].set_ylabel("Cijena ($)")
    axes[0].set_title("Stambena povrsina vs Cijena")
    axes[1].scatter(df_train["house_age"], df_train["price"],
                    alpha=0.3, s=8, color="darkorange")
    axes[1].set_xlabel("house_age (starost kuce)")
    axes[1].set_ylabel("Cijena ($)")
    axes[1].set_title("Starost kuce vs Cijena")
    save_fig("04_scatter_plots.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    df_train.boxplot(column="price", ax=ax,
                     boxprops=dict(color="steelblue"),
                     medianprops=dict(color="red", linewidth=2))
    ax.set_title("Boxplot cijene - pregled outliera (trening skup)")
    ax.set_ylabel("Cijena ($)")
    save_fig("05_price_boxplot.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    df_train.boxplot(column="price", by="waterfront", ax=ax,
                     boxprops=dict(color="steelblue"),
                     medianprops=dict(color="red", linewidth=2))
    ax.set_title("Cijena po waterfront statusu")
    ax.set_xlabel("waterfront (0=ne, 1=da)")
    ax.set_ylabel("Cijena ($)")
    plt.suptitle("")
    save_fig("06_waterfront_price.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    df_train.boxplot(column="price", by="view", ax=ax,
                     boxprops=dict(color="steelblue"),
                     medianprops=dict(color="red", linewidth=2))
    ax.set_title("Cijena po ocjeni pogleda")
    ax.set_xlabel("view (0-4)")
    ax.set_ylabel("Cijena ($)")
    plt.suptitle("")
    save_fig("07_view_price.png")

    print("[EDA] zavrseno")


# ─── 2. KNN kriva ─────────────────────────────────────────────────────────────

def plot_knn_curve(metrics_dir: str):
    path = os.path.join(metrics_dir, "knn_elbow.json")
    if not os.path.exists(path):
        print("[WARN] knn_elbow.json nije pronadjen.")
        return
    with open(path) as f:
        data = json.load(f)
    ks = [int(k) for k in data.keys()]
    maes = list(data.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, maes, "bo-", linewidth=2, markersize=6)
    ax.set_xlabel("Broj suseda K")
    ax.set_ylabel("MAE ($) - validacioni skup")
    ax.set_title("KNN - validacioni MAE u zavisnosti od K")
    ax.grid(True, alpha=0.3)
    best_k = ks[int(np.argmin(maes))]
    ax.axvline(best_k, color="red", linestyle="--",
               label=f"Izabrano K={best_k} (min validacioni MAE)")
    ax.legend()
    save_fig("08_knn_k_curve.png")


# ─── 3. Poredjenje modela ─────────────────────────────────────────────────────

def plot_metrics_comparison(metrics_dir: str):
    path = os.path.join(metrics_dir, "metrics.json")
    if not os.path.exists(path):
        print("[WARN] metrics.json nije pronadjen.")
        return
    with open(path) as f:
        metrics = json.load(f)
    df_m = pd.DataFrame([m for m in metrics if m["split"] == "VAL"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e67e22",
              "#9b59b6", "#e74c3c", "#1abc9c", "#f39c12"]
    for i, (col, label) in enumerate(
            [("MAE", "MAE ($)"), ("RMSE", "RMSE ($)"), ("R2", "R2")]):
        bars = axes[i].bar(df_m["model"], df_m[col], color=colors[:len(df_m)])
        axes[i].set_title(label)
        axes[i].set_xticklabels(df_m["model"], rotation=35, ha="right", fontsize=8)
        for bar in bars:
            h = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., h * 1.01,
                         f"{h:,.0f}" if col != "R2" else f"{h:.3f}",
                         ha="center", va="bottom", fontsize=8)
    fig.suptitle("Poredjenje modela - validacioni skup", fontsize=14,
                 fontweight="bold")
    save_fig("09_model_comparison.png")


# ─── 4. Znacajnost atributa ───────────────────────────────────────────────────

def feature_importance(X_train, ylog_train, X_val, ylog_val, yorig_val, models_dir):
    """
    Tri pogleda na znacajnost atributa (sve konzistentno sa MAE metrikom):
      4a. koeficijenti linearne regresije (na enkodiranim atributima),
      4b. permutation importance - MAE (Ridge) - custom scorer,
      4c. ugradjena feature importance Random Forest-a,
      4d. permutation importance - MAE (Gradient Boosting) - custom scorer.

    Custom scorer objasnjenje:
      pipeline.predict() vraca log(price).
      Scorer prima (y_true=yorig_val, y_pred=log_price) i racuna:
        MAE(y_true, expm1(y_pred))
      permutation_importance vraca importances_mean direktno kao
      pad performanse - veci broj znaci vazniji atribut.
      NE negiramo importances_mean jer scorer vec vraca pozitivne
      vrijednosti za pad MAE.

    Ranking za top K atributa se odredjuje po GB permutation importance
    jer je GB izabrani najbolji model projekta.
    """

    # Custom scorer: y_true su originalne cijene, y_pred su log cijene iz pipeline
    # greater_is_better=False jer je MAE greska (manje = bolje)
    def mae_dollars_from_log(y_true_orig, y_pred_log):
        return mean_absolute_error(y_true_orig, np.expm1(y_pred_log))

    log_mae_scorer = make_scorer(mae_dollars_from_log, greater_is_better=False)

    # 4a. Koeficijenti linearne regresije (top 20 po |w|)
    lr_pipe = joblib.load(os.path.join(models_dir, "linearna_regresija.joblib"))
    feat_names = lr_pipe.named_steps["preprocessor"].get_feature_names_out()
    coefs = pd.Series(lr_pipe.named_steps["model"].coef_, index=feat_names)
    top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:20]
    top_coefs = top_coefs.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors_c = ["#e74c3c" if v > 0 else "#3498db" for v in top_coefs]
    top_coefs.plot(kind="barh", ax=ax, color=colors_c)
    ax.set_title("Linearna regresija - top 20 koeficijenata (log cijena)")
    ax.set_xlabel("Vrijednost koeficijenta (skalirani atributi)")
    ax.axvline(0, color="black", linewidth=0.8)
    save_fig("10_linear_coefficients.png")

    # 4b. Permutation importance - MAE (Ridge)
    # importances_mean > 0 znaci: permutacija povecava MAE = atribut je vazan
    ridge_pipe = joblib.load(os.path.join(models_dir, "ridge.joblib"))
    result_ridge = permutation_importance(
        ridge_pipe, X_val, yorig_val,
        n_repeats=10, random_state=42,
        scoring=log_mae_scorer
    )
    # ISPRAVNO: koristimo importances_mean direktno, bez negiranja
    perm_imp_ridge = pd.Series(
        result_ridge.importances_mean, index=X_val.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors_r = ["#e74c3c" if v > 0 else "#3498db" for v in perm_imp_ridge]
    perm_imp_ridge.plot(kind="barh", ax=ax, color=colors_r)
    ax.set_title("Permutation Importance (Ridge) - MAE, validacioni skup")
    ax.set_xlabel("Pogoršanje MAE ($) pri permutaciji atributa")
    ax.axvline(0, color="black", linewidth=0.8)
    save_fig("11_permutation_importance_ridge_mae.png")

    # 4d. Permutation importance - MAE (Gradient Boosting)
    gb_pipe = joblib.load(os.path.join(models_dir, "gradient_boosting.joblib"))
    result_gb = permutation_importance(
        gb_pipe, X_val, yorig_val,
        n_repeats=10, random_state=42,
        scoring=log_mae_scorer
    )
    # ISPRAVNO: koristimo importances_mean direktno, bez negiranja
    perm_imp_gb = pd.Series(
        result_gb.importances_mean, index=X_val.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors_g = ["#e74c3c" if v > 0 else "#3498db" for v in perm_imp_gb]
    perm_imp_gb.plot(kind="barh", ax=ax, color=colors_g)
    ax.set_title("Permutation Importance (Gradient Boosting) - MAE, validacioni skup")
    ax.set_xlabel("Pogoršanje MAE ($) pri permutaciji atributa")
    ax.axvline(0, color="black", linewidth=0.8)
    save_fig("11_permutation_importance_gb_mae.png")

    # 4c. Random Forest feature importance
    rf_pipe = joblib.load(os.path.join(models_dir, "random_forest.joblib"))
    rf_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    rf_imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_,
                       index=rf_names).sort_values(ascending=False)[:20].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    rf_imp.plot(kind="barh", ax=ax, color="#27ae60")
    ax.set_title("Random Forest - top 20 Feature Importance")
    ax.set_xlabel("Importance")
    save_fig("13_rf_feature_importance.png")

    # Ranking po GB permutation importance (MAE, konzistentno sa odlucnom metrikom)
    ranked_gb = perm_imp_gb.sort_values(ascending=False)
    print("\n  Atributi po znacajnosti (Permutation Importance GB - MAE, validacija):")
    for feat, val in ranked_gb.items():
        print(f"    {feat:18s}: ${val:,.1f}")
    return ranked_gb.index.tolist()


# ─── 5. Svi atributi vs top K atributa ────────────────────────────────────────

def compare_all_vs_top(X_train, ylog_train, yorig_train, X_val, ylog_val, yorig_val,
                       ranked_features, models_dir, top_k=6):
    """
    Poredi performanse modela sa SVIM atributima i sa TOP K atributa.
    Poredjenje se radi na VALIDACIONOM skupu po MAE metrici.
    """
    top_feats = ranked_features[:top_k]
    print(f"\n  Top {top_k} atributa: {top_feats}")

    model_files = {
        "LR": "linearna_regresija.joblib",
        "Ridge": "ridge.joblib",
        "KNN": "knn.joblib",
        "DT": "decision_tree.joblib",
        "RF": "random_forest.joblib",
        "GB": "gradient_boosting.joblib",
    }
    results = {}
    for label, fname in model_files.items():
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            continue
        pipe_all = joblib.load(path)

        # Svi atributi - MAE na originalnim cijenama
        pred_all = np.expm1(pipe_all.predict(X_val))
        mae_all = mean_absolute_error(yorig_val, pred_all)
        results[f"{label} (svi)"] = mae_all

        # Top K atributa - kreiramo novi pipeline sa samo top feats
        model_clone = clone(pipe_all.named_steps["model"])
        pipe_top = make_pipeline(model_clone, X_train[top_feats])
        pipe_top.fit(X_train[top_feats], ylog_train)
        pred_top = np.expm1(pipe_top.predict(X_val[top_feats]))
        mae_top = mean_absolute_error(yorig_val, pred_top)
        results[f"{label} (top {top_k})"] = mae_top

    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bar = ["#3498db" if "svi" in k else "#2ecc71" for k in results]
    bars = ax.bar(list(results.keys()), list(results.values()), color=colors_bar)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h * 1.01,
                f"${h:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Svi atributi vs top {top_k} - MAE na validacionom skupu")
    ax.set_ylabel("MAE ($)")
    ax.set_ylim(0, max(results.values()) * 1.15)
    ax.set_xticklabels(list(results.keys()), rotation=35, ha="right")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#3498db", label="Svi atributi"),
                       Patch(color="#2ecc71", label=f"Top {top_k}")],
              loc="upper left")
    save_fig("12_all_vs_top_features_mae.png")

    print("\n  Rezultati poredjenja (validacioni MAE):")
    for k, v in results.items():
        print(f"    {k:18s}: MAE=${v:,.0f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("data/processed/data_clean.csv", dtype={"zipcode": str})

    s = split_data(df)
    X_train, ylog_train, yorig_train = (
        s["X_train"], s["ylog_train"], s["yorig_train"])
    X_val, ylog_val = s["X_val"], s["ylog_val"]
    yorig_val = s["yorig_val"]

    df_train = X_train.copy()
    df_train["price"] = yorig_train.values

    eda(df_train)
    plot_knn_curve("results/metrics")
    plot_metrics_comparison("results/metrics")
    ranked = feature_importance(X_train, ylog_train, X_val, ylog_val, yorig_val, "models")
    compare_all_vs_top(X_train, ylog_train, yorig_train, X_val, ylog_val, yorig_val,
                       ranked, "models")


if __name__ == "__main__":
    main()