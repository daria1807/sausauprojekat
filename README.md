# Predikcija cijene kuće — King County Housing Dataset

Projekat urađen u okviru predmeta **Softverski algoritmi u sistemima automatskog upravljanja (SAUSAU)**
Fakultet tehničkih nauka, Novi Sad

---

## O projektu

Regresioni ML model za predikciju cijene nekretnina na osnovu karakteristika kuće.
Dataset: King County Housing (Seattle, WA) — 4600 uzoraka, 18 atributa.

**Cilj:** Predvidjeti kontinualnu vrijednost (`price`) na osnovu ulaznih atributa.
**Target u modelu:** `log1p(price)` — log transformacija stabilizuje jako asimetričnu
raspodjelu cijena; predikcije se vraćaju u dolare pomoću `expm1`.

---

## Struktura projekta

```
ml-project/
├── data/
│   ├── raw/                 ← originalni sirovi dataset
│   └── processed/           ← pripremljeni dataset (output data_preparation.py)
├── src/
│   ├── data_preparation.py  ← čišćenje nevalidnih zapisa, feature engineering
│   ├── train.py             ← podjela, analiza outliera, treniranje, tuning, finalni test
│   ├── evaluate.py          ← EDA grafici, poređenje modela, značajnost atributa
│   └── predict.py           ← predikcija za nove primjere (finalni pipeline)
├── models/                  ← sačuvani Pipeline-i (.joblib) + model_meta.json
├── results/
│   ├── figures/             ← svi grafici (.png)
│   └── metrics/             ← metrike modela (.json)
├── app/
│   └── ui.py                ← Streamlit aplikacija
├── README.md
└── requirements.txt
```

---

## Evaluacioni protokol (sprječavanje curenja podataka)

Redoslijed koraka prati gradivo: *"Podeliti podatke pa tek onda uraditi
preprocessing na train skupu i primeniti ga na validation/test skup."*

1. **data_preparation.py** radi samo operacije koje ne zavise od statistike
   skupa: uklanjanje očigledno nevalidnih zapisa (`price == 0`, `bedrooms == 0`,
   duplikati) i transformacije na nivou reda (`house_age`, `is_renovated`,
   `years_since_ren`, `sqft_ratio`, izdvajanje `zipcode`). Kategorički atributi
   (`city`, `zipcode`) ostaju **sirovi stringovi**.
2. **Podjela 70/15/15** (train / validation / test, `random_state=42`) prije
   bilo kakvog učenja iz podataka. Ista funkcija podjele koristi se i u
   `evaluate.py`, pa su sve analize konzistentne.
3. **Preprocesiranje je dio sklearn Pipeline-a**:
   `ColumnTransformer(StandardScaler za numeričke, OneHotEncoder za city/zipcode)`.
   Fit se dešava isključivo na podacima koje pipeline dobije u `.fit()` —
   dakle samo na trening skupu, i unutar svakog CV folda posebno.
   `handle_unknown='ignore'` — grad/ZIP koji nije viđen u treningu ne ruši predikciju.
4. **Hiperparametri** (alpha za Ridge/Lasso; K za KNN; max_depth i
   min_samples_leaf za stablo; n_estimators i max_features za Random Forest;
   learning_rate i n_estimators za Gradient Boosting) biraju se na
   **validacionom skupu**.
5. **Unakrsna validacija** (5-fold) provjerava stabilnost najboljeg modela.
6. **Test skup se koristi jednom**, tek nakon izbora najboljeg modela —
   isključivo za finalnu procjenu.

### Analiza outliera — zašto NISU uklonjeni

EDA pokazuje ekstremne cijene (do ~$3.8M u validacionom skupu). Analiza u
`train.py` (`outlier_analysis`) poredi treniranje sa svim podacima i nakon
IQR filtriranja (pragovi računati samo na train skupu, val/test netaknuti):

- IQR filtriranje pogoršava rezultate svih modela — model nikad ne vidi
  skupe kuće, a procjenjuje se na realnoj raspodjeli koja ih sadrži;
- ekstremne cijene su **legitimne luksuzne kuće** (npr. 7050 ft² u Clyde
  Hill-u), a ne greške u podacima — gradivo: *"Da li je svaka udaljena
  tačka nužno anomalija?"*;
- log-transformacija targeta već ublažava asimetriju raspodjele.

Zato se trening radi na svim trening podacima, a odluka je dokumentovana
ispisom analize.

---

## Rezultati

| Model | Val R²(log) | Val MAE ($) |
|---|---|---|
| Baseline (prosjek train skupa) | ~0.00 | 228,351 |
| Linearna regresija | 0.798 | 116,196 |
| Ridge (alpha=1.0) | 0.798 | 116,215 |
| Lasso (alpha=0.0001) | 0.798 | 116,739 |
| KNN (K=5) | 0.694 | 123,685 |
| Decision Tree (depth=10, leaf=5) | 0.668 | 129,566 |
| Random Forest (n=200, sqrt) | 0.780 | 104,241 |
| **Gradient Boosting (lr=0.1, n=400)** | **0.815** | **90,984** |

**Finalni model: Gradient Boosting** — test skup (korišćen jednom):
MAE ≈ **$99.8k**, R²(log) = **0.744**, R² ($) = **0.700**;
CV R²(log) = 0.766 ± 0.038 (stabilan).

### Napomena o izboru i tumačenju metrika

- **MAE u dolarima** — primarna interpretabilna metrika (prosječno
  odstupanje procjene u $).
- **R² u log prostoru** — kriterijum izbora modela, jer je target log(price).
- **R² u dolarima** kod linearnih modela može biti i negativan iako je
  R²(log) visok: nekoliko ekstremno skupih kuća dominira kvadratnom greškom
  nakon `expm1` (linearna ekstrapolacija u log prostoru → ogromne vrijednosti
  u $). To je očekivano ponašanje metrike osjetljive na velika odstupanja
  (gradivo: *"kvadratne greške posebno naglašavaju velika odstupanja"*),
  a ne greška u kodu — i jedan je od razloga zašto je ansambl (GB) izabran.

### Značajnost atributa

Permutation importance (Ridge, validacioni skup) nad **originalnim**
atributima (city/zipcode se permutuju kao cjelina):
lokacija (`zipcode`, `city`) i površina (`sqft_above`, `sqft_living`)
dominiraju; slijede `view`, `bathrooms`, `condition`.
Poređenje "svi vs top 6 atributa" (validacioni skup): linearni modeli i GB
gube ~0.01 R²(log), a KNN se čak popravlja (manja dimenzionalnost —
gradivo: KNN je *"slabiji u visokoj dimenzionalnosti"*).

---

## Pokretanje

```bash
pip install -r requirements.txt

# 1. Priprema podataka
python src/data_preparation.py

# 2. Treniranje + izbor modela + finalni test
python src/train.py

# 3. EDA grafici i analize
python src/evaluate.py

# 4. Predikcija (demo)
python src/predict.py

# 5. Streamlit aplikacija
streamlit run app/ui.py
```

Napomena: skripte se pokreću iz glavnog foldera projekta (`ml-project/`).

---

## Atributi

| Atribut | Opis |
|---|---|
| price | **Target** — cijena kuće ($), modelovan kao log1p(price) |
| bedrooms, bathrooms | broj soba / kupatila |
| sqft_living, sqft_lot | stambena površina / parcela (ft²) |
| floors | broj spratova |
| waterfront | pogled na vodu (0/1) |
| view | ocjena pogleda (0–4) |
| condition | stanje kuće (1–5) |
| sqft_above, sqft_basement | površina iznad zemlje / podrum (ft²) |
| house_age | izvedeno: 2015 − yr_built |
| is_renovated, years_since_ren | izvedeno iz yr_renovated |
| sqft_ratio | izvedeno: sqft_living / (sqft_lot + 1) |
| city, zipcode | kategorički (One-Hot u Pipeline-u) |

Uklonjeni atributi: `date`, `street` (jedinstvena adresa), `country`
(konstanta), `statezip` (zamijenjen sa `zipcode`), `yr_built`/`yr_renovated`
(zamijenjeni izvedenim atributima).