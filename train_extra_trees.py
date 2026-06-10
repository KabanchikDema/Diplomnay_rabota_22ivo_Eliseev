# -*- coding: utf-8 -*-
"""
Обучение финальной модели Extra Trees для Streamlit-приложения.
Подходит для Google Colab и локального запуска.
"""

import re
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================
# НАСТРОЙКИ
# ==============================

DATA_PATH = Path("cleaned_data.xlsx")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUT_DIR / "extra_trees_final.joblib"
METRICS_PATH = OUT_DIR / "extra_trees_final_metrics.json"

# ВАЖНО:
# False — как в общей таблице моделей: влажность, плотность, широта, долгота.
# True  — добавить месяц как признак. В Streamlit тогда в файле должен быть месяц.
USE_MONTH_IN_FINAL_MODEL = False

MONTH_ORDER = {
    "Январь": 1,
    "Февраль": 2,
    "Март": 3,
    "Апрель": 4,
    "Май": 5,
    "Июнь": 6,
    "Июль": 7,
    "Август": 8,
    "Сентябрь": 9,
    "Октябрь": 10,
    "Ноябрь": 11,
    "Декабрь": 12,
}

# ==============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def normalize_column_name(col):
    col = str(col).replace("\n", " ").replace("\r", " ")
    col = re.sub(r"\s+", " ", col)
    return col.strip()


def find_column(columns, keywords, required=True):
    for col in columns:
        low = str(col).lower()
        for key in keywords:
            if str(key).lower() in low:
                return col

    if required:
        raise ValueError(f"Не найден столбец по ключам: {keywords}")
    return None


def to_numeric_series(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .replace(["nan", "None", ""], np.nan),
        errors="coerce",
    )


def normalize_humidity(series):
    """
    Приводит влажность к процентам.
    0.165 -> 16.5
    16.5  -> 16.5
    """
    s = series.astype(float).copy()
    mask_fraction = (s > 0) & (s <= 1)
    s.loc[mask_fraction] = s.loc[mask_fraction] * 100
    return s


def normalize_density(series):
    """
    Приводит плотность к г/см^3.
    1.11 -> 1.11
    11.1 -> 1.11
    1110 -> 1.11
    """
    s = series.astype(float).copy()

    mask_x10 = (s > 5) & (s < 30)
    s.loc[mask_x10] = s.loc[mask_x10] / 10

    mask_kg_m3 = (s >= 500) & (s <= 3000)
    s.loc[mask_kg_m3] = s.loc[mask_kg_m3] / 1000

    return s


def calculate_metrics(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = n_features

    if n > p + 1:
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    else:
        adj_r2 = np.nan

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "R2": float(r2),
        "Adj_R2": float(adj_r2) if not np.isnan(adj_r2) else None,
        "MAE": float(mae),
        "RMSE": float(rmse),
    }

# ==============================
# ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
# ==============================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Не найден файл {DATA_PATH}. Положи cleaned_data.xlsx рядом со скриптом."
    )

df = pd.read_excel(DATA_PATH)
df.columns = [normalize_column_name(c) for c in df.columns]

lat_col = find_column(df.columns, ["широта", "latitude", "lat"])
lon_col = find_column(df.columns, ["долгота", "longitude", "lon"])
humidity_col = find_column(df.columns, ["влажность", "humidity"])
density_col = find_column(df.columns, ["плотность", "density"])
target_col = find_column(df.columns, ["сопротивление пенетрации", "сопротивление", "ew"])
month_col = find_column(df.columns, ["месяц", "month"], required=False)

for col in [lat_col, lon_col, humidity_col, density_col, target_col]:
    df[col] = to_numeric_series(df[col])

df["Влажность_%"] = normalize_humidity(df[humidity_col])
df["Плотность_г_см3"] = normalize_density(df[density_col])

feature_cols = [
    "Влажность_%",
    "Плотность_г_см3",
    lat_col,
    lon_col,
]

if USE_MONTH_IN_FINAL_MODEL:
    if month_col is None:
        raise ValueError("USE_MONTH_IN_FINAL_MODEL=True, но столбец месяца не найден.")

    df["Месяц_число"] = df[month_col].map(MONTH_ORDER)
    df["Месяц_число"] = df["Месяц_число"].fillna(
        pd.to_numeric(df[month_col], errors="coerce")
    )
    feature_cols.append("Месяц_число")

base_cols = feature_cols + [target_col]
data = df[base_cols].copy()

data = data[
    data["Влажность_%"].between(0, 100)
    & data["Плотность_г_см3"].between(0.5, 2.5)
    & data[target_col].ge(0)
    & data[lat_col].between(-90, 90)
    & data[lon_col].between(-180, 180)
].dropna()

X = data[feature_cols].copy()
y = data[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# ==============================
# ОБУЧЕНИЕ EXTRA TREES
# ==============================

model = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
pred = np.maximum(pred, 0)

metrics = calculate_metrics(y_test, pred, X.shape[1])
metrics.update(
    {
        "model": "ExtraTreesRegressor",
        "rows_total_after_cleaning": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_cols,
        "target": target_col,
        "use_month_in_final_model": USE_MONTH_IN_FINAL_MODEL,
    }
)

artifact = {
    "model": model,
    "model_name": "ExtraTreesRegressor",
    "feature_cols": feature_cols,
    "target_col": target_col,
    "month_order": MONTH_ORDER,
    "use_month_in_final_model": USE_MONTH_IN_FINAL_MODEL,
    "clip_negative_predictions": True,
    "metrics": metrics,
}

joblib.dump(artifact, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("Модель сохранена:", MODEL_PATH)
print("Метрики сохранены:", METRICS_PATH)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
