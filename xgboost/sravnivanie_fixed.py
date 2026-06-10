from pathlib import Path
import io
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "xgboost_final_fixed.joblib"

st.set_page_config(
    page_title="XGBoost-анализ сопротивления пенетрации грунта",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Служебные функции
# -----------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит названия столбцов к единому виду: убирает переносы и лишние пробелы."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Безопасно переводит числа, включая строки с десятичной запятой."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )


@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str):
    return joblib.load(path)


def load_model_from_upload(uploaded_model):
    return joblib.load(uploaded_model)


def unpack_model(artifact):
    """Поддерживает новый формат {model, feature_cols, target_col} и старый raw-estimator."""
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        feature_cols = artifact.get("feature_cols", [])
        target_col = artifact.get("target_col", "Сопротивление пенетрации, Ew")
        metrics = artifact.get("metrics", {})
        model_type = artifact.get("model_type", type(model).__name__)
    else:
        model = artifact
        feature_cols = ["Влажность, %", "Плотность, г/см^3", "Широта", "Долгота"]
        target_col = "Сопротивление пенетрации, Ew"
        metrics = {}
        model_type = type(model).__name__
    return model, feature_cols, target_col, metrics, model_type


def is_xgboost_model(model) -> bool:
    name = type(model).__name__.lower()
    module = type(model).__module__.lower()
    if "xgb" in name or "xgboost" in module:
        return True
    if hasattr(model, "named_steps"):
        return any(is_xgboost_model(step) for step in model.named_steps.values())
    return False


def read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = uploaded_file.getvalue()
        # Пробуем популярные кодировки и разделители.
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            for sep in (",", ";", "\t"):
                try:
                    df_try = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=sep)
                    if df_try.shape[1] > 1:
                        return df_try
                except Exception as exc:
                    last_error = exc
        raise ValueError(f"Не удалось прочитать CSV: {last_error}")
    return pd.read_excel(uploaded_file)


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError("Не найдены нужные столбцы: " + ", ".join(missing))
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = to_numeric_safe(X[col])
    return X


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_download_excel(data: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="results")
    return buffer.getvalue()

# -----------------------------------------------------------------------------
# Интерфейс
# -----------------------------------------------------------------------------

st.title("🌱 XGBoost-анализ сопротивления пенетрации грунта")
st.caption("Приложение загружает очищенный набор данных, строит прогноз Ew, оценивает качество модели и показывает пространственную аналитику.")

with st.sidebar:
    st.header("1. Модель")
    uploaded_model = st.file_uploader("При необходимости загрузите .joblib", type=["joblib", "pkl"])
    strict_xgb = st.checkbox("Требовать именно XGBoost", value=True)

try:
    if uploaded_model is not None:
        artifact = load_model_from_upload(uploaded_model)
        model_source = uploaded_model.name
    else:
        artifact = load_model_from_path(str(DEFAULT_MODEL_PATH))
        model_source = DEFAULT_MODEL_PATH.name

    model, feature_cols, target_col, saved_metrics, model_type = unpack_model(artifact)

    if strict_xgb and not is_xgboost_model(model):
        st.error(f"Загружена не XGBoost-модель: {type(model).__module__}.{type(model).__name__}. Загрузите корректный файл XGBoost или отключите строгую проверку.")
        st.stop()

    st.sidebar.success(f"Модель загружена: {model_source}")
    st.sidebar.write(f"Тип: `{model_type}`")
    st.sidebar.write("Признаки:")
    st.sidebar.code("\n".join(feature_cols), language="text")
except Exception as exc:
    st.error(f"Ошибка загрузки модели: {exc}")
    st.stop()

with st.sidebar:
    st.header("2. Данные")
    uploaded_file = st.file_uploader("Загрузите Excel/CSV с данными", type=["xlsx", "xls", "csv"])

if uploaded_file is None:
    st.info("Загрузите файл `cleaned_data.xlsx` или другой файл с такими же столбцами.")
    st.stop()

try:
    df = normalize_columns(read_table(uploaded_file))
except Exception as exc:
    st.error(f"Ошибка чтения файла: {exc}")
    st.stop()

try:
    X = prepare_features(df, feature_cols)
except KeyError as exc:
    st.error(str(exc))
    st.write("Столбцы в загруженном файле:")
    st.write(df.columns.tolist())
    st.stop()

bad_rows = X.isna().any(axis=1)
if bad_rows.any():
    st.warning(f"В {int(bad_rows.sum())} строках есть некорректные признаки. Для прогноза они будут заполнены медианами внутри модели, если это предусмотрено pipeline.")

try:
    df["Прогноз_Ew"] = model.predict(X)

    # защита от отрицательных значений
    df["Прогноз_Ew"] = np.maximum(df["Прогноз_Ew"], 0)

except Exception as e:
    st.error(f"Ошибка прогнозирования: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# KPI
# -----------------------------------------------------------------------------

st.subheader("Ключевые показатели")
cols = st.columns(6)
cols[0].metric("Измерений", f"{len(df):,}".replace(",", " "))
cols[1].metric("Признаков модели", len(feature_cols))
cols[2].metric("Средний прогноз Ew", f"{df['Прогноз_Ew'].mean():.2f}")
cols[3].metric("Минимум Ew", f"{df['Прогноз_Ew'].min():.2f}")
cols[4].metric("Максимум Ew", f"{df['Прогноз_Ew'].max():.2f}")
if "Прямоугольная зона №" in df.columns:
    cols[5].metric("Зон", int(df["Прямоугольная зона №"].nunique()))
else:
    cols[5].metric("Зон", "—")

# -----------------------------------------------------------------------------
# Вкладки
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Прогноз",
    " Качество",
    " Карта",
    " Месяцы",
    " Факторы",
    " Рекомендации"
])

with tab1:
    st.subheader("Результаты прогнозирования")
    st.dataframe(df, use_container_width=True, height=420)

    fig = px.histogram(df, x="Прогноз_Ew", nbins=30, title="Распределение прогнозных значений Ew")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Проверка качества")
    if target_col in df.columns:
        y_true = to_numeric_safe(df[target_col])
        y_pred = pd.Series(df["Прогноз_Ew"], index=df.index)
        mask = y_true.notna() & y_pred.notna()

        if mask.sum() < 2:
            st.warning("Недостаточно валидных фактических значений для расчёта метрик.")
        else:
            r2 = r2_score(y_true[mask], y_pred[mask])
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            rmse = rmse_score(y_true[mask], y_pred[mask])

            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{r2:.3f}")
            m2.metric("MAE", f"{mae:.3f}")
            m3.metric("RMSE", f"{rmse:.3f}")

            compare_df = pd.DataFrame({"Факт": y_true[mask], "Прогноз": y_pred[mask]})
            fig = px.scatter(compare_df, x="Факт", y="Прогноз", title="Фактические и прогнозные значения")
            min_v = float(min(compare_df["Факт"].min(), compare_df["Прогноз"].min()))
            max_v = float(max(compare_df["Факт"].max(), compare_df["Прогноз"].max()))
            fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Идеальный прогноз"))
            st.plotly_chart(fig, use_container_width=True)

            df["Ошибка"] = y_true - y_pred
            fig_err = px.histogram(df.loc[mask], x="Ошибка", nbins=30, title="Распределение ошибок: факт − прогноз")
            st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.info(f"В данных нет целевого столбца `{target_col}`, поэтому метрики качества не рассчитаны.")

    if saved_metrics:
        with st.expander("Метрики, сохранённые при обучении модели"):
            st.json(saved_metrics)

with tab3:
    st.subheader("Пространственное распределение прогноза")
    if {"Широта", "Долгота"}.issubset(df.columns):
        map_df = df.copy()
        map_df["Широта"] = to_numeric_safe(map_df["Широта"])
        map_df["Долгота"] = to_numeric_safe(map_df["Долгота"])
        map_df = map_df.dropna(subset=["Широта", "Долгота", "Прогноз_Ew"])

        df["Размер_точки"] = df["Прогноз_Ew"] - df["Прогноз_Ew"].min() + 1

        fig = px.scatter_mapbox(
            df,
            lat="Широта",
            lon="Долгота",
            color="Прогноз_Ew",
            size="Размер_точки",
            zoom=12,
            height=700,
            title="Пространственное распределение Ew",
            hover_data=["Прогноз_Ew"]
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 40, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Для карты нужны столбцы `Широта` и `Долгота`.")

with tab4:
    st.subheader("Динамика по месяцам")
    if "Месяц" in df.columns:
        month_stats = df.groupby("Месяц", dropna=False)["Прогноз_Ew"].agg(["count", "mean", "std", "min", "max"]).reset_index()
        st.dataframe(month_stats, use_container_width=True)
        fig = px.line(month_stats, x="Месяц", y="mean", markers=True, title="Среднее прогнозное Ew по месяцам")
        st.plotly_chart(fig, use_container_width=True)
    elif "Дата, Время" in df.columns:
        date_series = pd.to_datetime(df["Дата, Время"], errors="coerce")
        tmp = df.assign(Месяц_даты=date_series.dt.to_period("M").astype(str))
        month_stats = tmp.groupby("Месяц_даты")["Прогноз_Ew"].agg(["count", "mean", "std", "min", "max"]).reset_index()
        st.dataframe(month_stats, use_container_width=True)
        fig = px.line(month_stats, x="Месяц_даты", y="mean", markers=True, title="Среднее прогнозное Ew по месяцам")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("В данных нет столбца `Месяц` или даты.")

with tab5:
    st.subheader("Факторный анализ")
    numeric_candidates = feature_cols + [target_col, "Прогноз_Ew"]
    corr_cols = []
    corr_df = pd.DataFrame(index=df.index)
    for col in numeric_candidates:
        if col in df.columns and col not in corr_cols:
            corr_df[col] = to_numeric_safe(df[col])
            corr_cols.append(col)

    if len(corr_cols) >= 2:
        corr = corr_df[corr_cols].corr()
        st.dataframe(corr, use_container_width=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Матрица корреляций")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Недостаточно числовых столбцов для корреляционного анализа.")

    # Важность признаков для XGBoost внутри pipeline.
    xgb_step = None
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if is_xgboost_model(step) and hasattr(step, "feature_importances_"):
                xgb_step = step
                break
    elif hasattr(model, "feature_importances_"):
        xgb_step = model

    if xgb_step is not None:
        imp = pd.DataFrame({"Признак": feature_cols, "Важность": xgb_step.feature_importances_})
        imp = imp.sort_values("Важность", ascending=False)
        fig = px.bar(imp, x="Признак", y="Важность", title="Важность признаков XGBoost")
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Рекомендации по зонам")
    if "Прямоугольная зона №" in df.columns:
        zone_stats = (
            df.groupby("Прямоугольная зона №", dropna=False)["Прогноз_Ew"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .rename(columns={"mean": "Средний Ew", "std": "Стандартное отклонение", "count": "Измерений"})
        )
        mean_ew = zone_stats["Средний Ew"].mean()
        std_ew = zone_stats["Средний Ew"].std(ddof=0)

        def rec_text(value):
            if value >= mean_ew + std_ew:
                return "Высокое уплотнение: рекомендуется глубокое рыхление и ограничение проходов техники."
            if value >= mean_ew:
                return "Среднее уплотнение: требуется контроль и точечное рыхление при необходимости."
            return "Состояние удовлетворительное: достаточно планового мониторинга."

        zone_stats["Рекомендация"] = zone_stats["Средний Ew"].apply(rec_text)
        st.dataframe(zone_stats, use_container_width=True)

        fig = px.bar(zone_stats, x="Прямоугольная зона №", y="Средний Ew", title="Средний прогноз Ew по зонам")
        fig.add_hline(y=mean_ew, line_dash="dash", annotation_text="Среднее по зонам")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("В данных нет столбца `Прямоугольная зона №`, поэтому рекомендации по зонам не сформированы.")

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Скачать результаты CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="results_xgboost.csv",
        mime="text/csv"
    )
with col_b:
    st.download_button(
        "Скачать результаты Excel",
        data=make_download_excel(df),
        file_name="results_xgboost.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
