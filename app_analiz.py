
from pathlib import Path
import re
import warnings
import folium
from streamlit_folium import st_folium
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# =====================================================
# НАСТРОЙКА СТРАНИЦЫ
# =====================================================

st.set_page_config(
    page_title="Extra Trees-анализ Ew",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Прогнозирование сопротивления пенетрации грунта")
st.caption("Финальная модель: Extra Trees Regressor")

# =====================================================
# КОНСТАНТЫ
# =====================================================

MODEL_PATHS = [
    Path("models/extra_trees_final.joblib"),
    Path("extra_trees_final.joblib"),
    Path(r"C:\учеба\Диплом\Extra trees\extra_trees_final.joblib"),
]

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

# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================


def normalize_column_name(col: str) -> str:
    col = str(col)
    col = col.replace("\n", " ").replace("\r", " ")
    col = col.replace("³", "^3")
    col = re.sub(r"\s+", " ", col)
    return col.strip()


def find_column(columns, variants, required=False):
    normalized = {normalize_column_name(c).lower(): c for c in columns}

    # 1. точное совпадение после нормализации
    for variant in variants:
        key = normalize_column_name(variant).lower()
        if key in normalized:
            return normalized[key]

    # 2. поиск по вхождению
    for col in columns:
        low = normalize_column_name(col).lower()
        for variant in variants:
            variant_low = normalize_column_name(variant).lower()
            if variant_low in low:
                return col

    # 3. поиск по словам
    for col in columns:
        low = normalize_column_name(col).lower()
        for variant in variants:
            words = normalize_column_name(variant).lower().split()
            if all(word in low for word in words if len(word) > 1):
                return col

    if required:
        raise ValueError(f"Не найден столбец: {variants}")
    return None


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .replace(["nan", "None", "", "-"], np.nan),
        errors="coerce",
    )


def normalize_humidity(series: pd.Series) -> pd.Series:
    s = series.astype(float).copy()
    mask_fraction = (s > 0) & (s <= 1)
    s.loc[mask_fraction] = s.loc[mask_fraction] * 100
    return s


def normalize_density(series: pd.Series) -> pd.Series:
    s = series.astype(float).copy()

    # 11.1 -> 1.11
    mask_x10 = (s > 5) & (s < 30)
    s.loc[mask_x10] = s.loc[mask_x10] / 10

    # 1110 -> 1.11
    mask_kg_m3 = (s >= 500) & (s <= 3000)
    s.loc[mask_kg_m3] = s.loc[mask_kg_m3] / 1000

    return s


def load_model():
    existing_path = None
    for path in MODEL_PATHS:
        if path.exists():
            existing_path = path
            break

    if existing_path is None:
        raise FileNotFoundError(
            "Не найден extra_trees_final.joblib. Положите модель рядом с приложением "
            "или в папку models/."
        )

    artifact = joblib.load(existing_path)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        model_name = artifact.get("model_name", type(model).__name__)
        feature_cols = artifact.get(
            "feature_cols",
            ["Влажность_%", "Плотность_г_см3", "Широта", "Долгота"],
        )
        metrics = artifact.get("metrics", {})
    else:
        model = artifact
        model_name = type(model).__name__
        feature_cols = ["Влажность_%", "Плотность_г_см3", "Широта", "Долгота"]
        metrics = {}

    return model, model_name, feature_cols, metrics, existing_path


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def prepare_data(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    lat_source = find_column(df.columns, ["Широта", "latitude", "lat"], required=True)
    lon_source = find_column(df.columns, ["Долгота", "longitude", "lon"], required=True)
    humidity_source = find_column(df.columns, ["Влажность, %", "Влажность", "humidity"], required=True)
    density_source = find_column(df.columns, ["Плотность, г/см^3", "Плотность", "density"], required=True)

    target_source = find_column(
        df.columns,
        ["Сопротивление пенетрации, Ew", "Сопротивление пенетрации", "Ew"],
        required=False,
    )
    month_source = find_column(df.columns, ["Месяц", "month"], required=False)
    zone_source = find_column(df.columns, ["Прямоугольная зона №", "Зона", "zone"], required=False)

    # Единые технические колонки
    df["Широта"] = to_numeric_series(df[lat_source])
    df["Долгота"] = to_numeric_series(df[lon_source])
    df["Влажность_%"] = normalize_humidity(to_numeric_series(df[humidity_source]))
    df["Плотность_г_см3"] = normalize_density(to_numeric_series(df[density_source]))

    if target_source is not None:
        df[target_source] = to_numeric_series(df[target_source])

    if month_source is not None:
        df["Месяц_число"] = df[month_source].map(MONTH_ORDER)
        df["Месяц_число"] = df["Месяц_число"].fillna(
            pd.to_numeric(df[month_source], errors="coerce")
        )
    elif "Дата, Время" in df.columns:
        df["Месяц_число"] = pd.to_datetime(df["Дата, Время"], errors="coerce").dt.month
    else:
        df["Месяц_число"] = np.nan

    cols = {
        "target_col": target_source,
        "month_col": month_source,
        "zone_col": zone_source,
    }

    return df, cols


def make_predictions(model, df: pd.DataFrame, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Не хватает признаков модели: {missing}")

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    valid_mask = X.notna().all(axis=1)
    df["Прогноз_Ew"] = np.nan

    if valid_mask.sum() == 0:
        raise ValueError("Нет строк без пропусков в признаках модели.")

    pred = model.predict(X.loc[valid_mask])
    df.loc[valid_mask, "Прогноз_Ew"] = np.maximum(pred, 0)
    df["Прогноз_Ew"] = pd.to_numeric(df["Прогноз_Ew"], errors="coerce")

    return df, valid_mask


def show_matplotlib(fig):
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


# =====================================================
# ЗАГРУЗКА МОДЕЛИ
# =====================================================

try:
    model, model_name, feature_cols, saved_metrics, model_path = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

st.sidebar.success("Модель загружена")
st.sidebar.write(f"Файл: `{model_path}`")
st.sidebar.write(f"Модель: `{model_name}`")
st.sidebar.write("Признаки модели:")
st.sidebar.write(feature_cols)

if saved_metrics:
    st.sidebar.write("Метрики при обучении:")
    st.sidebar.json(saved_metrics)

# =====================================================
# ЗАГРУЗКА ДАННЫХ
# =====================================================

uploaded_file = st.file_uploader(
    "Загрузите Excel/CSV-файл с данными",
    type=["xlsx", "xls", "csv"],
)

if uploaded_file is None:
    st.info("Загрузите cleaned_data.xlsx или CSV с такими же столбцами.")
    st.stop()

try:
    df_raw = read_uploaded_file(uploaded_file)
    df, cols = prepare_data(df_raw)
    df, valid_mask = make_predictions(model, df, feature_cols)
except Exception as e:
    st.error(f"Ошибка обработки: {e}")
    if "df_raw" in locals():
        st.write("Колонки загруженного файла:")
        st.write([normalize_column_name(c) for c in df_raw.columns])
    st.stop()

target_col = cols["target_col"]
month_col = cols["month_col"]
zone_col = cols["zone_col"]

plot_df = df.dropna(subset=["Прогноз_Ew"]).copy()

# =====================================================
# KPI
# =====================================================

st.subheader("Краткая сводка")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Строк", len(df))
k2.metric("Прогнозов", len(plot_df))
k3.metric("Зон", df[zone_col].nunique() if zone_col is not None else "—")
k4.metric("Месяцев", df[month_col].nunique() if month_col is not None else "—")
k5.metric("Средний прогноз Ew", round(plot_df["Прогноз_Ew"].mean(), 2) if len(plot_df) else "—")
k6.metric("Максимум Ew", round(plot_df["Прогноз_Ew"].max(), 2) if len(plot_df) else "—")

with st.expander("Диагностика"):
    st.write("Размер исходного файла:", df_raw.shape)
    st.write("Размер после прогноза:", df.shape)
    st.write("Количество прогнозов:", len(plot_df))
    st.write("Колонки:", df.columns.tolist())
    st.write("Прогноз Ew:")
    st.write(df["Прогноз_Ew"].describe())
    st.write("Координаты:")
    st.write(df[["Широта", "Долгота"]].describe())
    if target_col is not None:
        st.write("Фактический Ew:")
        st.write(df[target_col].describe())

# =====================================================
# ВКЛАДКИ
# =====================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Прогноз", "Качество", "Карта", "Месяцы", "Факторы", "Рекомендации"]
)

# =====================================================
# TAB 1 — ПРОГНОЗ
# =====================================================

with tab1:
    st.subheader("Результаты прогнозирования")
    st.dataframe(df, use_container_width=True)

    st.write("Точек для распределения:", len(plot_df))

    if len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(plot_df["Прогноз_Ew"], bins=30, edgecolor="black")
        ax.set_title("Распределение прогнозных значений Ew")
        ax.set_xlabel("Прогноз Ew")
        ax.set_ylabel("Количество")
        ax.grid(axis="y", alpha=0.3)
        show_matplotlib(fig)
    else:
        st.warning("Нет прогнозных значений для построения распределения.")

# =====================================================
# TAB 2 — КАЧЕСТВО
# =====================================================

with tab2:
    st.subheader("Качество прогноза")

    if target_col is None or target_col not in df.columns:
        st.info("В файле нет фактического столбца Ew, поэтому качество рассчитать нельзя.")
    else:
        compare_df = df[[target_col, "Прогноз_Ew"]].dropna().copy()
        st.write("Точек для графика факт-прогноз:", len(compare_df))

        if len(compare_df) > 1:
            y_true = compare_df[target_col]
            y_pred = compare_df["Прогноз_Ew"]

            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            c1, c2, c3 = st.columns(3)
            c1.metric("R²", round(r2, 3))
            c2.metric("MAE", round(mae, 3))
            c3.metric("RMSE", round(rmse, 3))

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_true, y_pred, alpha=0.7)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            ax.set_title("Фактические и прогнозные значения Ew")
            ax.set_xlabel("Факт")
            ax.set_ylabel("Прогноз")
            ax.grid(alpha=0.3)
            show_matplotlib(fig)
        else:
            st.warning("Недостаточно данных для расчёта качества.")

# =====================================================
# TAB 3 — КАРТА
# =====================================================
lat_col = "Широта"
lon_col = "Долгота"
zone_col = "Прямоугольная зона №"
with tab3:
    st.subheader("Яндекс-карта прогнозного сопротивления пенетрации")

    map_df = df[
        ["Широта", "Долгота", "Прогноз_Ew", "Прямоугольная зона №"]
    ].dropna().copy()

    points = []

    for _, row in map_df.iterrows():
        points.append({
            "lat": float(row["Широта"]),
            "lon": float(row["Долгота"]),
            "ew": float(row["Прогноз_Ew"]),
            "zone": str(row["Прямоугольная зона №"])
        })

    points_json = json.dumps(points, ensure_ascii=False)

    center_lat = float(map_df["Широта"].mean())
    center_lon = float(map_df["Долгота"].mean())

    html = f"""
<div id="map" style="width: 100%; height: 700px;"></div>

<script src="https://api-maps.yandex.ru/2.1/?lang=ru_RU"></script>

<script>
    ymaps.ready(init);

    function getColor(value) {{
        if (value < 25) return "#4C78A8";      // мягкий синий
        if (value < 35) return "#72B7B2";      // бирюзовый
        if (value < 45) return "#A0CBE8";      // светло-голубой
        if (value < 55) return "#F2CF5B";      // мягкий жёлтый
        return "#E45756";                      // спокойный красный
    }}

    function init() {{
        var map = new ymaps.Map("map", {{
            center: [{center_lat}, {center_lon}],
            zoom: 14,
            type: "yandex#hybrid",
            controls: ["zoomControl", "typeSelector", "fullscreenControl"]
        }});

        var points = {points_json};

        // группировка точек по зонам
        var zones = {{}};

        points.forEach(function(p) {{
            if (!zones[p.zone]) {{
                zones[p.zone] = [];
            }}

            zones[p.zone].push(p);
        }});

        // линии внутри каждой зоны
        Object.keys(zones).forEach(function(zone) {{
            var zonePoints = zones[zone];

            zonePoints.sort(function(a, b) {{
                return a.lat - b.lat || a.lon - b.lon;
            }});

            var coords = zonePoints.map(function(p) {{
                return [p.lat, p.lon];
            }});

            if (coords.length >= 2) {{
                var line = new ymaps.Polyline(
                    coords,
                    {{
                        hintContent: "Зона " + zone
                    }},
                    {{
                        strokeColor: "#2F4F4F",
                        strokeWidth: 2,
                        strokeOpacity: 0.45
                    }}
                );

                map.geoObjects.add(line);
            }}
        }});

        // точки
        points.forEach(function(p) {{
            var placemark = new ymaps.Circle(
                [[p.lat, p.lon], 8],
                {{
                    balloonContent:
                        "<b>Зона:</b> " + p.zone +
                        "<br><b>Прогноз Ew:</b> " + p.ew.toFixed(2),
                    hintContent: "Зона " + p.zone + ", Ew: " + p.ew.toFixed(2)
                }},
                {{
                    fillColor: getColor(p.ew),
                    fillOpacity: 0.75,
                    strokeColor: "#1F2933",
                    strokeOpacity: 0.8,
                    strokeWidth: 1
                }}
            );

            map.geoObjects.add(placemark);
        }});
    }}
    </script>
    """

    components.html(html, height=720)

# =====================================================
# TAB 4 — МЕСЯЦЫ
# =====================================================

with tab4:
    st.subheader("Средний прогноз Ew по месяцам")

    if month_col is None or month_col not in df.columns:
        st.info("В данных нет столбца месяца.")
    else:
        month_stats = (
            plot_df.groupby(month_col, dropna=False)["Прогноз_Ew"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        month_stats["Порядок"] = month_stats[month_col].map(MONTH_ORDER).fillna(999)
        month_stats = month_stats.sort_values("Порядок").drop(columns="Порядок")

        st.write("Месяцев для графика:", len(month_stats))
        st.dataframe(month_stats, use_container_width=True)

        if len(month_stats) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(month_stats[month_col].astype(str), month_stats["mean"], marker="o")
            ax.set_title("Средний прогноз Ew по месяцам")
            ax.set_xlabel("Месяц")
            ax.set_ylabel("Средний прогноз Ew")
            ax.grid(alpha=0.3)
            plt.xticks(rotation=30)
            show_matplotlib(fig)
        else:
            st.warning("Нет данных для графика по месяцам.")

# =====================================================
# TAB 5 — ФАКТОРЫ
# =====================================================

with tab5:
    st.subheader("Матрица корреляций")

    corr_candidates = ["Влажность_%", "Плотность_г_см3", target_col, "Прогноз_Ew"]
    corr_cols = [c for c in corr_candidates if c is not None and c in df.columns]

    corr_df = df[corr_cols].copy()
    for col in corr_cols:
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")
    corr_df = corr_df.dropna()

    st.write("Строк для корреляции:", len(corr_df))
    st.write("Колонки корреляции:", corr_cols)

    if len(corr_df) > 2 and len(corr_cols) >= 2:
        corr = corr_df.corr()
        st.dataframe(corr, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr.values)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

        ax.set_title("Матрица корреляций")
        fig.colorbar(im, ax=ax)
        show_matplotlib(fig)
    else:
        st.warning("Недостаточно числовых данных для корреляционной матрицы.")

# =====================================================
# TAB 6 — РЕКОМЕНДАЦИИ
# =====================================================

with tab6:
    st.subheader("Средний прогноз Ew по зонам и рекомендации")

    if zone_col is None or zone_col not in df.columns:
        st.info("В данных нет столбца зоны.")
    else:
        zone_stats = (
            plot_df.groupby(zone_col)["Прогноз_Ew"]
            .mean()
            .reset_index()
            .rename(columns={zone_col: "Зона", "Прогноз_Ew": "Средний Ew"})
        )

        st.write("Зон для графика:", len(zone_stats))

        if len(zone_stats) > 0:
            mean_ew = zone_stats["Средний Ew"].mean()
            std_ew = zone_stats["Средний Ew"].std()

            def make_recommendation(value):
                if value > mean_ew + std_ew:
                    return "Высокое уплотнение. Рекомендуется глубокое рыхление."
                if value > mean_ew:
                    return "Среднее уплотнение. Требуется контроль состояния почвы."
                return "Состояние удовлетворительное. Специальные меры не требуются."

            zone_stats["Рекомендация"] = zone_stats["Средний Ew"].apply(make_recommendation)
            zone_stats["Средний Ew"] = zone_stats["Средний Ew"].round(2)

            st.dataframe(zone_stats, use_container_width=True)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(zone_stats["Зона"].astype(str), zone_stats["Средний Ew"])
            ax.set_title("Средний прогноз Ew по зонам")
            ax.set_xlabel("Зона")
            ax.set_ylabel("Средний прогноз Ew")
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=45)
            show_matplotlib(fig)
        else:
            st.warning("Нет данных для рекомендаций по зонам.")

# =====================================================
# ВЫГРУЗКА
# =====================================================

st.download_button(
    "Скачать результаты CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="extra_trees_results.csv",
    mime="text/csv",
)
