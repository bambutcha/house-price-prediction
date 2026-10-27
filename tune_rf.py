"""
Перебор гиперпараметров RandomForest на validation RMSLE.

Запуск:
    source .venv/bin/activate
    python tune_rf.py

Меняй GRID и FEATURES под новые опыты.
Результат: tune_results.csv + печать топа.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Признаки из разведки ошибок / экспериментов
FEATURES = ("Qual_x_SF", "KitchenQualNum")

GRID = {
    "n_estimators": [300, 400, 500, 700],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 5],
    "max_features": [0.35, 0.4, 0.45],
    "max_depth": [None, 20, 30],
}

KQ = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}


def prepare_xy(features=FEATURES):
    features = set(features)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    train = train[
        ~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))
    ].reset_index(drop=True)

    y = train["SalePrice"].copy()
    X = train.drop(columns=["Id", "SalePrice"])
    X_test = test.drop(columns=["Id"])

    none_cols = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]
    for col in none_cols:
        X[col] = X[col].fillna("None")
        X_test[col] = X_test[col].fillna("None")

    for col in ["GarageYrBlt", "MasVnrArea"]:
        X[col] = X[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

    for df in (X, X_test):
        df["TotalSF"] = (
            df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]
        )
        df["TotalBath"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )
        df["Age"] = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

        if "Qual_x_SF" in features:
            df["Qual_x_SF"] = df["OverallQual"] * df["TotalSF"]
        if "Has2ndFlr" in features:
            df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
        if "KitchenQualNum" in features:
            df["KitchenQualNum"] = df["KitchenQual"].map(KQ).fillna(3)
        if "OldLowQual" in features:
            df["OldLowQual"] = (
                (df["Age"] >= 40) & (df["OverallQual"] <= 5)
            ).astype(int)
        if "IsAbnormal" in features:
            df["IsAbnormal"] = (df["SaleCondition"] == "Abnorml").astype(int)
        if "TotalPorch" in features:
            df["TotalPorch"] = (
                df["OpenPorchSF"]
                + df["EnclosedPorch"]
                + df["3SsnPorch"]
                + df["ScreenPorch"]
                + df["WoodDeckSF"]
            )

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        median = X[col].median()
        X[col] = X[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("Missing").replace("nan", "Missing")
        X_test[col] = (
            X_test[col].astype(str).fillna("Missing").replace("nan", "Missing")
        )

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X[col], X_test[col]]))
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])

    return X, y


def main():
    X, y = prepare_xy()
    y_log = np.log1p(y)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_val_price = np.expm1(y_val)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Features: {FEATURES}")
    print(f"Комбинаций параметров: {len(combos)}")

    rows = []
    for i, vals in enumerate(combos, start=1):
        params = dict(zip(keys, vals))
        model = RandomForestRegressor(
            **params, random_state=42, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        pred = np.clip(np.expm1(model.predict(X_val)), 0, None)
        rmsle = float(np.sqrt(mean_squared_log_error(y_val_price, pred)))
        row = {"rmsle": rmsle, **params}
        rows.append(row)
        print(f"[{i}/{len(combos)}] {params} -> {rmsle:.5f}")

    results = pd.DataFrame(rows).sort_values("rmsle").reset_index(drop=True)
    out = ROOT / "tune_results.csv"
    results.to_csv(out, index=False)

    print("\n=== TOP 10 ===")
    print(results.head(10).to_string(index=False))
    print(f"\nЛучшее: {results.iloc[0].to_dict()}")
    print(f"Таблица: {out}")


if __name__ == "__main__":
    main()
