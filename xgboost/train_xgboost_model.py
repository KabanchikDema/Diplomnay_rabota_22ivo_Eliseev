from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

DATA_PATH = Path(__file__).with_name('cleaned_data.xlsx')
MODEL_PATH = Path(__file__).with_name('xgboost_final_fixed.joblib')
METRICS_PATH = Path(__file__).with_name('xgboost_metrics.json')

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str)
                  .str.replace('\n', ' ', regex=False)
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.strip())
    return df

def main():
    df = normalize_columns(pd.read_excel(DATA_PATH))
    target_col = 'Сопротивление пенетрации, Ew'
    feature_cols = ['Влажность, %', 'Плотность, г/см^3', 'Широта', 'Долгота']
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f'Не найдены столбцы: {missing}')

    train_df = df[feature_cols + [target_col]].apply(pd.to_numeric, errors='coerce').dropna()
    X = train_df[feature_cols]
    y = train_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('xgb', XGBRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1,
            tree_method='hist'
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        'model_type': 'XGBRegressor',
        'rows_total': int(len(df)),
        'rows_used': int(len(train_df)),
        'features': feature_cols,
        'target': target_col,
        'test_size': 0.2,
        'random_state': 42,
        'r2': float(r2_score(y_test, pred)),
        'mae': float(mean_absolute_error(y_test, pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, pred)))
    }

    artifact = {
        'model': model,
        'model_type': 'XGBRegressor',
        'feature_cols': feature_cols,
        'target_col': target_col,
        'metrics': metrics
    }
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
