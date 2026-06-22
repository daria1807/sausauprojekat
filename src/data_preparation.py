"""
data_preparation.py
Ucitavanje, ciscenje i feature engineering za predikciju cijene kuce.

Sprjecavanje curenja podataka (data leakage):
U ovoj fazi radim samo operacije koje ne zavise od statistike cijelog
skupa podataka:
  - uklanjanje ocigledno nevalidnih zapisa (npr. price == 0),
  - transformacije na nivou jednog reda (house_age, is_renovated, ...).

"""

import pandas as pd
import numpy as np  
import os 

# Dataset je prikupljen 2014/2015 - referentna godina za starost kuce
REFERENTNA_GODINA = 2015

# Kategoricke kolone ne enkodiram ovdje - ostaju kao tekst,
# enkodira ih OneHotEncoder unutar Pipeline-a (samo na train skupu)
CATEGORICAL_FEATURES = ["city", "zipcode"]


def ucitaj_podatke(path):
    """Ucitavam dataset iz CSV fajla"""
    df = pd.read_csv(path)
    print(f"Ucitano podataka: {df.shape[0]} redova i {df.shape[1]} kolona")
    return df


def provjera_nedostajucih(df):
    """
    Provjeravam nedostajuce vrijednosti po kolonama.
    Da ih je bilo, popunjavanje bih radila u train.py na trening skupu.
    """
    nedostajuce = df.isnull().sum()
    print("Provjera nedostajucih vrijednosti:")
    if nedostajuce.sum() == 0:
        print("  nema nedostajucih vrijednosti")
    else:
        print(nedostajuce[nedostajuce > 0])


def provjera_duplikata(df):
    """Provjeravam duplikate"""
    broj_duplikata = df.duplicated().sum()
    if broj_duplikata > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Pronadjeno i uklonjeno duplikata: {broj_duplikata}")
    else:
        print("Skup ne sadrzi duplikate")
    return df


def ukloni_nevalidne(df):
    """
    Uklanjam ocigledno nevalidne zapise (greske u podacima):
      - price == 0     kuca bez cijene ne nosi informaciju o targetu
      - bedrooms == 0  
      ekstremne u train

    
    """
    prije = len(df)
    df = df[df["price"] > 0]
    df = df[df["bedrooms"] > 0]
    poslije = len(df)
    print(f"Uklonjeno nevalidnih zapisa: {prije - poslije}. Ostalo: {poslije}")
    return df.reset_index(drop=True)


def feature_engineering(df):
    """
    Pravim nove i transformisem postojece atribute (sve na nivou jednog reda):
      - house_age       starost kuce (REFERENTNA_GODINA - yr_built)
      - is_renovated    1 ako je kuca renovirana, inace 0
      - years_since_ren godine od renovacije (0 ako nije renovirana)
      - sqft_ratio      omjer stambene povrsine i parcele
      - zipcode         izvucen iz statezip (npr. 'WA 98103' -> '98103')

    Uklanjam atribute koji ocigledno ne uticu na izlaz ili su redundantni:
    date, street (jedinstvena adresa), country (uvijek USA), statezip
    (zamijenjen sa zipcode), yr_built i yr_renovated (zamijenjeni izvedenim
    atributima). city i zipcode ostaju kao kategorije - enkodira ih Pipeline.
    

     yr built brisem jer ga pretavram u bolju informaciju,ovo sa oduzimanjem
    yr renovated brisem i radim provjeru jel renoviran ail ne ,TRUE False
    KOLIKO JE PROSLO OD RENOVACIJE 2015-yr renovated

  
    """

    df = df.copy()
    df["house_age"] = REFERENTNA_GODINA - df["yr_built"]
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["years_since_ren"] = np.where(
        df["yr_renovated"] > 0, REFERENTNA_GODINA - df["yr_renovated"], 0
    )
    df["sqft_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)

    # Iz statezip izvlacim samo ZIP broj (npr. "WA 98103" -> "98103")
    df["zipcode"] = df["statezip"].str.split().str[1]

    kolone_za_brisanje = ["date", "street", "statezip", "country",
                          "yr_built", "yr_renovated"]
    df = df.drop(columns=kolone_za_brisanje)

    print(f"Feature engineering zavrsen. Broj kolona: {len(df.columns)}")
    print(f"Kategoricki atributi (enkodira ih Pipeline na train skupu): "
          f"{CATEGORICAL_FEATURES}")
    print(f"   city: {df['city'].nunique()} razlicitih gradova")
    print(f"   zipcode: {df['zipcode'].nunique()} razlicitih ZIP kodova")
    return df


def pripremi_i_sacuvaj(raw_path, out_path):
    """Glavni tok pripreme podataka (bez operacija koje uce iz podataka)."""
    df = ucitaj_podatke(raw_path)
    provjera_nedostajucih(df)
    df = provjera_duplikata(df)
    df = ukloni_nevalidne(df)
    df = feature_engineering(df)

    
    df.to_csv(out_path, index=False)


    
    return df


if __name__ == "__main__":
    RAW_PATH = "data/raw/data.csv"
    OUT_PATH = "data/processed/data_clean.csv"
    df = pripremi_i_sacuvaj(RAW_PATH, OUT_PATH)
    print(df.head())
