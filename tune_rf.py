"""
Перебор гиперпараметров RandomForest на validation RMSLE.

Запуск из папки проекта:
    source .venv/bin/activate
    python tune_rf.py

Скрипт:
1) готовит данные так же, как ноутбук;
2) перебирает комбинации параметров;
3) печатает топ результатов и лучшие параметры;
4) пишет полную таблицу в tune_results.csv.
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

# Сетку можно менять под новые эксперименты
GRID = {
    "n_estimators": [400, 600, 800, 1000],
    "min_samples_leaf": [1, 2, 3],
    "max_features": [0.3, 0.4, 0.5, "sqrt"],
}


def prepare_xy():
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

    rows = []
    combos = list(
        itertools.product(
            GRID["n_estimators"],
            GRID["min_samples_leaf"],
            GRID["max_features"],
        )
    )
    print(f"Комбинаций: {len(combos)}")

    for i, (n, leaf, mf) in enumerate(combos, start=1):
        model = RandomForestRegressor(
            n_estimators=n,
            min_samples_leaf=leaf,
            max_features=mf,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        pred = np.clip(np.expm1(model.predict(X_val)), 0, None)
        rmsle = float(np.sqrt(mean_squared_log_error(y_val_price, pred)))
        rows.append(
            {
                "rmsle": rmsle,
                "n_estimators": n,
                "min_samples_leaf": leaf,
                "max_features": mf,
            }
        )
        print(f"[{i}/{len(combos)}] n={n} leaf={leaf} mf={mf} -> {rmsle:.5f}")

    results = pd.DataFrame(rows).sort_values("rmsle").reset_index(drop=True)
    out = ROOT / "tune_results.csv"
    results.to_csv(out, index=False)

    best = results.iloc[0]
    print("\n=== TOP 10 ===")
    print(results.head(10).to_string(index=False))
    print(f"\nЛучшее: {best.to_dict()}")
    print(f"Таблица сохранена: {out}")
    print(
        "Подставь лучшие числа в RandomForestRegressor в house_prices_lab.ipynb "
        "и пересобери submission."
    )


if __name__ == "__main__":
    main()
