import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import plotly.express as px

# ==========================================
# Настройка страницы
# ==========================================

st.set_page_config(
    page_title="Прогнозирование сопротивления пенетрации",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Прогнозирование сопротивления пенетрации грунта")

# ==========================================
# Загрузка модели
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("xgboost_final.joblib")

try:
    model = load_model()

    st.sidebar.success("Модель загружена")
    st.sidebar.write(type(model))

except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# ==========================================
# Загрузка данных
# ==========================================

uploaded_file = st.file_uploader(
    "Загрузите Excel-файл",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is None:
    st.info("Загрузите файл для анализа")
    st.stop()

# ==========================================
# Чтение данных
# ==========================================

try:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

except Exception as e:
    st.error(e)
    st.stop()

# ==========================================
# Очистка названий столбцов
# ==========================================

df.columns = (
    df.columns
    .astype(str)
    .str.replace("\n", " ", regex=False)
    .str.replace("  ", " ", regex=False)
    .str.strip()
)

# ==========================================
# Признаки модели
# ==========================================

feature_cols = [
    "Влажность, %",
    "Плотность, г/см^3",
    "Широта",
    "Долгота"
]

for col in feature_cols:
    if col not in df.columns:
        st.error(f"Не найден столбец: {col}")
        st.write("Найденные столбцы:")
        st.write(df.columns.tolist())
        st.stop()

X = df[feature_cols]

# ==========================================
# Прогноз
# ==========================================

try:
    df["Прогноз_Ew"] = model.predict(X)

except Exception as e:
    st.error(f"Ошибка прогнозирования: {e}")
    st.stop()

# ==========================================
# KPI
# ==========================================

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Измерений", len(df))

if "Прямоугольная зона №" in df.columns:
    c2.metric(
        "Зон",
        df["Прямоугольная зона №"].nunique()
    )

if "Месяц" in df.columns:
    c3.metric(
        "Месяцев",
        df["Месяц"].nunique()
    )

c4.metric(
    "Средний Ew",
    round(df["Прогноз_Ew"].mean(), 2)
)

c5.metric(
    "Макс Ew",
    round(df["Прогноз_Ew"].max(), 2)
)

c6.metric(
    "Мин Ew",
    round(df["Прогноз_Ew"].min(), 2)
)

# ==========================================
# Вкладки
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Прогноз",
    "Качество",
    "Карта",
    "Месяцы",
    "Факторы",
    "Рекомендации"
])

# ==========================================
# ПРОГНОЗ
# ==========================================

with tab1:

    st.subheader("Результаты прогнозирования")

    st.dataframe(df)

    fig = px.histogram(
        df,
        x="Прогноз_Ew",
        nbins=30,
        title="Распределение прогнозных значений Ew"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# КАЧЕСТВО
# ==========================================

with tab2:

    target_col = "Сопротивление пенетрации, Ew"

    if target_col in df.columns:

        y_true = df[target_col]
        y_pred = df["Прогноз_Ew"]

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        k1, k2, k3 = st.columns(3)

        k1.metric("R²", round(r2, 3))
        k2.metric("MAE", round(mae, 3))
        k3.metric("RMSE", round(rmse, 3))

        compare_df = pd.DataFrame({
            "Факт": y_true,
            "Прогноз": y_pred
        })

        fig = px.scatter(
            compare_df,
            x="Факт",
            y="Прогноз",
            title="Фактические и прогнозные значения"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        df["Ошибка"] = y_true - y_pred

# ==========================================
# КАРТА
# ==========================================

with tab3:

    if (
        "Широта" in df.columns
        and
        "Долгота" in df.columns
    ):

        fig = px.scatter_mapbox(
            df,
            lat="Широта",
            lon="Долгота",
            color="Прогноз_Ew",
            size="Прогноз_Ew",
            zoom=12,
            height=700,
            title="Пространственное распределение Ew"
        )

        fig.update_layout(
            mapbox_style="open-street-map"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==========================================
# АНАЛИЗ ПО МЕСЯЦАМ
# ==========================================

with tab4:

    if "Месяц" in df.columns:

        month_stats = (
            df.groupby("Месяц")["Прогноз_Ew"]
            .agg(["mean", "std"])
            .reset_index()
        )

        st.dataframe(month_stats)

        fig = px.line(
            month_stats,
            x="Месяц",
            y="mean",
            markers=True,
            title="Среднее значение Ew по месяцам"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==========================================
# ФАКТОРЫ
# ==========================================

with tab5:

    st.subheader("Корреляции")

    corr_cols = [
        c for c in [
            "Влажность, %",
            "Плотность, г/см^3",
            "Сопротивление пенетрации, Ew"
        ]
        if c in df.columns
    ]

    if len(corr_cols) >= 2:

        corr = df[corr_cols].corr()

        st.dataframe(corr)

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Матрица корреляций"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==========================================
# РЕКОМЕНДАЦИИ
# ==========================================

with tab6:

    if "Прямоугольная зона №" in df.columns:

        zone_stats = (
            df.groupby(
                "Прямоугольная зона №"
            )["Прогноз_Ew"]
            .mean()
            .reset_index()
        )

        mean_ew = zone_stats["Прогноз_Ew"].mean()
        std_ew = zone_stats["Прогноз_Ew"].std()

        recommendations = []

        for _, row in zone_stats.iterrows():

            zone = row["Прямоугольная зона №"]
            ew = row["Прогноз_Ew"]

            if ew > mean_ew + std_ew:
                rec = "Высокое уплотнение. Рекомендуется глубокое рыхление."

            elif ew > mean_ew:
                rec = "Среднее уплотнение. Требуется контроль."

            else:
                rec = "Состояние удовлетворительное."

            recommendations.append(
                [zone, round(ew, 2), rec]
            )

        rec_df = pd.DataFrame(
            recommendations,
            columns=[
                "Зона",
                "Средний Ew",
                "Рекомендация"
            ]
        )

        st.dataframe(rec_df)

# ==========================================
# Выгрузка результатов
# ==========================================

st.download_button(
    "Скачать результаты CSV",
    data=df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="results.csv",
    mime="text/csv"
)