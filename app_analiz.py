# © 2026 Елисеев Д.А.
# Программное средство для прогнозирования сопротивления пенетрации грунта.
#Все права защищены.
from pathlib import Path
import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import streamlit as st
import streamlit.components.v1 as components
import json
from urllib.parse import quote
import uuid

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    st_folium = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    OrdinaryKriging = None
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
st.markdown("© 2026 Елисеев Д.А. Программное средство для анализа физико-механических свойств грунта.")
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


def get_yandex_maps_api_key() -> tuple[str, str]:
    """
    Возвращает ключ Яндекс.Карт и источник ключа.

    Для деплоя на Streamlit Community Cloud ключ нужно добавить в Secrets:
    YANDEX_MAPS_API_KEY = "..."

    Root-level secrets Streamlit также делает доступными как переменные окружения,
    поэтому сначала проверяем os.environ — это не вызывает красных ошибок, если
    локального файла .streamlit/secrets.toml нет.
    """
    YANDEX_MAPS_API_KEY = "06b4b494-5512-4418-8d95-ca7ffff6404c"
    for name in ("YANDEX_MAPS_API_KEY", "YANDEX_API_KEY"):
        value = os.environ.get(name, "")
        if value:
            return value.strip(), f"секрет/переменная окружения {name}"

    # Локально st.secrets без файла secrets.toml может бросать FileNotFoundError.
    # Поэтому обращаемся к нему только если файл существует.
    possible_secret_files = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    if any(path.exists() for path in possible_secret_files):
        for name in ("YANDEX_MAPS_API_KEY", "YANDEX_API_KEY"):
            try:
                value = st.secrets.get(name, "")
                if value:
                    return str(value).strip(), f"st.secrets[{name}]"
            except Exception:
                pass

    return "", "ключ не найден"


# =====================================================
# ФУНКЦИИ ДЛЯ КАРТЫ КРИГИНГА И РЕКОМЕНДАЦИЙ
# =====================================================


def ew_color(value: float) -> str:
    """Цветовая шкала для Ew: от низкого к высокому уплотнению."""
    if value < 25:
        return "#4C78A8"  # низкое
    if value < 35:
        return "#72B7B2"  # умеренное
    if value < 55:
        return "#F2CF5B"  # повышенное
    return "#E45756"      # высокое


def continuous_ew_color(value: float, vmin: float, vmax: float) -> str:
    """Плавная цветовая шкала для поверхности кригинга, как на heatmap."""
    if not np.isfinite(value):
        return "#808080"
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        ratio = 0.5
    else:
        ratio = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
        ratio = min(1.0, max(0.0, ratio))
    r, g, b, _ = plt.get_cmap("jet")(ratio)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def ew_class(value: float) -> str:
    """Класс сопротивления пенетрации для интерпретации проходимости техники."""
    if value < 25:
        return "низкое сопротивление"
    if value < 35:
        return "рабочее сопротивление"
    if value < 55:
        return "повышенное сопротивление"
    return "высокое сопротивление"


def trafficability_class(value: float) -> str:
    """Практическая оценка влияния Ew на проходимость техники."""
    if value < 25:
        return "осторожный заезд: возможна слабая несущая способность"
    if value < 35:
        return "проходимость в целом нормальная"
    if value < 55:
        return "проходимость допустима, но движение лучше ограничить маршрутами"
    return "проходимость технически возможна, но есть риск усиления переуплотнения"


def ew_zone_action(value: float) -> str:
    """Рекомендация по движению техники и дальнейшим действиям с участком."""
    if value < 25:
        return (
            "Перед заездом техники проверить влажность и фактическую несущую способность. "
            "Использовать лёгкую технику, снижать давление на почву, избегать движения после осадков."
        )
    if value < 35:
        return (
            "Участок можно использовать для стандартного прохода техники. "
            "Желательно сохранять движение по технологическим колеям и не выполнять лишние проходы."
        )
    if value < 55:
        return (
            "Проход техники допустим, но участок уже работает как зона риска. "
            "Рекомендуется ограничить тяжёлую технику, двигаться по постоянным маршрутам, после работ проверить колеи и уплотнение."
        )
    return (
        "Проблемная зона для эксплуатации: повторные проходы тяжёлой техники могут усилить переуплотнение. "
        "Лучше исключить лишние заезды, работать только при необходимости, затем запланировать глубокое рыхление/щелевание и повторный контроль Ew."
    )


def overall_field_recommendation(mean_ew: float, high_share: float, elevated_share: float, low_share: float) -> tuple[str, str]:
    """Возвращает общий статус поля и текст рекомендации по проходимости техники."""
    if mean_ew >= 55 or high_share >= 25:
        return (
            "Проходимость ограничена зонами высокого сопротивления",
            "По карте есть выраженные участки высокого Ew. Техника, скорее всего, сможет пройти по таким зонам, "
            "но каждый дополнительный проход будет усиливать переуплотнение и ухудшать состояние почвы. "
            "Рекомендуется сократить число проходов, закрепить постоянные технологические маршруты, исключить тяжёлую технику на проблемных участках без необходимости, "
            "а после сезона провести глубокое рыхление/щелевание и повторные измерения пенетрации."
        )
    if mean_ew >= 35 or elevated_share >= 35:
        return (
            "Проходимость допустима, но требует управления движением техники",
            "Поле можно использовать для движения техники, однако повышенное сопротивление на части площади показывает риск дальнейшего уплотнения. "
            "Лучше не распределять движение хаотично по всему полю: использовать технологические колеи, ограничить тяжёлые агрегаты, "
            "не выполнять лишние проходы и отдельно проверить зоны с максимальным Ew."
        )
    if low_share >= 35:
        return (
            "Нужна проверка несущей способности перед заездом",
            "На заметной части поля сопротивление низкое. Это может означать более мягкое или переувлажнённое состояние, "
            "поэтому перед массовым заездом техники лучше проверить влажность, риск буксования и образования колеи. "
            "Рекомендуется использовать более лёгкую технику, снижать давление в шинах и переносить работы после осадков."
        )
    if mean_ew >= 25:
        return (
            "Проходимость поля в целом рабочая",
            "Средний уровень Ew находится в рабочем диапазоне. Технику можно запускать в обычном режиме, "
            "но лучше сохранять движение по постоянным маршрутам, избегать лишних проходов и не работать по переувлажнённой почве."
        )
    return (
        "Проходимость требует осторожной проверки",
        "Среднее сопротивление низкое. Перед заездом тяжёлой техники нужно убедиться, что почва держит нагрузку и не будет глубокого колееобразования. "
        "Оптимально начинать с лёгкой техники или тестового прохода, затем принимать решение по всему полю."
    )


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Монотонная цепь Эндрю. Точки передаются как (lon, lat)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def make_measurement_points_payload(map_df: pd.DataFrame) -> list[dict]:
    points = []
    for _, row in map_df.iterrows():
        value = float(row["Прогноз_Ew"])
        points.append(
            {
                "lat": float(row["Широта"]),
                "lon": float(row["Долгота"]),
                "ew": round(value, 2),
                "color": ew_color(value),
                "className": ew_class(value),
            }
        )
    return points


def build_kriging_surface(
    map_df: pd.DataFrame,
    grid_size: int = 120,
    mask_to_hull: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит ordinary kriging и возвращает поверхность для интерактивной Plotly-карты.
    Возвращает: уникальные точки, сетка долгот, сетка широт, матрица значений Ew.
    """
    if OrdinaryKriging is None:
        raise ImportError(
            "Библиотека pykrige не установлена. Установите её командой: pip install pykrige"
        )

    clean = map_df[["Широта", "Долгота", "Прогноз_Ew"]].dropna().copy()

    # Убираем дубли координат: для кригинга в одной точке должно быть одно значение.
    clean = (
        clean.groupby(["Широта", "Долгота"], as_index=False)["Прогноз_Ew"]
        .mean()
        .sort_values(["Широта", "Долгота"])
    )

    if len(clean) < 5:
        raise ValueError("Для кригинга нужно минимум 5 уникальных точек измерения.")

    x = clean["Долгота"].to_numpy(dtype=float)
    y = clean["Широта"].to_numpy(dtype=float)
    z = clean["Прогноз_Ew"].to_numpy(dtype=float)

    if np.isclose(x.min(), x.max()) or np.isclose(y.min(), y.max()):
        raise ValueError("Точки должны покрывать площадь поля, а не лежать на одной линии.")

    grid_lon = np.linspace(x.min(), x.max(), grid_size)
    grid_lat = np.linspace(y.min(), y.max(), grid_size)

    ok = OrdinaryKriging(
        x,
        y,
        z,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
    )
    zgrid, _ = ok.execute("grid", grid_lon, grid_lat)

    if hasattr(zgrid, "filled"):
        zgrid = zgrid.filled(np.nan)
    zgrid = np.asarray(zgrid, dtype=float)

    if mask_to_hull:
        hull = convex_hull(list(zip(x, y)))
        if len(hull) >= 3:
            hull_path = MplPath(hull)
            xx, yy = np.meshgrid(grid_lon, grid_lat)
            mask = hull_path.contains_points(
                np.column_stack([xx.ravel(), yy.ravel()])
            ).reshape(xx.shape)
            zgrid[~mask] = np.nan

    return clean, grid_lon, grid_lat, zgrid


def build_kriging_cells(map_df: pd.DataFrame, grid_size: int = 45, mask_to_hull: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """
    Строит ordinary kriging по точкам и возвращает ячейки для отрисовки на Яндекс.Карте.
    Ячейки ограничиваются выпуклой оболочкой точек, чтобы не закрашивать прямоугольник за пределами поля.
    """
    if OrdinaryKriging is None:
        raise ImportError(
            "Библиотека pykrige не установлена. Установите её командой: pip install pykrige"
        )

    clean = map_df[["Широта", "Долгота", "Прогноз_Ew"]].dropna().copy()

    # Если в одной координате несколько измерений, усредняем их — PyKrige плохо работает с дублями координат.
    clean = (
        clean.groupby(["Широта", "Долгота"], as_index=False)["Прогноз_Ew"]
        .mean()
        .sort_values(["Широта", "Долгота"])
    )

    if len(clean) < 5:
        raise ValueError("Для кригинга нужно минимум 5 уникальных точек измерения.")

    x = clean["Долгота"].to_numpy(dtype=float)
    y = clean["Широта"].to_numpy(dtype=float)
    z = clean["Прогноз_Ew"].to_numpy(dtype=float)

    if np.isclose(x.min(), x.max()) or np.isclose(y.min(), y.max()):
        raise ValueError("Точки должны покрывать площадь поля, а не лежать на одной линии.")

    grid_lon = np.linspace(x.min(), x.max(), grid_size)
    grid_lat = np.linspace(y.min(), y.max(), grid_size)

    ok = OrdinaryKriging(
        x,
        y,
        z,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
    )
    zgrid, _ = ok.execute("grid", grid_lon, grid_lat)

    if hasattr(zgrid, "filled"):
        zgrid = zgrid.filled(np.nan)
    zgrid = np.asarray(zgrid, dtype=float)

    hull_path = None
    if mask_to_hull:
        hull = convex_hull(list(zip(x, y)))
        hull_path = MplPath(hull) if len(hull) >= 3 else None

    zmin = float(np.nanmin(zgrid))
    zmax = float(np.nanmax(zgrid))

    cells = []
    for i in range(len(grid_lat) - 1):
        lat0 = float(grid_lat[i])
        lat1 = float(grid_lat[i + 1])
        center_lat = (lat0 + lat1) / 2

        for j in range(len(grid_lon) - 1):
            lon0 = float(grid_lon[j])
            lon1 = float(grid_lon[j + 1])
            center_lon = (lon0 + lon1) / 2

            if mask_to_hull and hull_path is not None and not hull_path.contains_point((center_lon, center_lat)):
                continue

            value = float(
                np.nanmean(
                    [
                        zgrid[i, j],
                        zgrid[i + 1, j],
                        zgrid[i, j + 1],
                        zgrid[i + 1, j + 1],
                    ]
                )
            )

            if np.isnan(value):
                continue

            cells.append(
                {
                    "coords": [
                        [lat0, lon0],
                        [lat0, lon1],
                        [lat1, lon1],
                        [lat1, lon0],
                    ],
                    "value": round(value, 2),
                    "color": continuous_ew_color(value, zmin, zmax),
                    "className": ew_class(value),
                }
            )

    if not cells:
        raise ValueError("Не удалось построить ячейки кригинга по текущим координатам.")

    return clean, cells


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

with tab3:
    st.subheader("Интерактивная Яндекс.Карта кригинга по точкам пенетрации")

    st.markdown(
        """
Здесь строится именно **интерактивная Яндекс.Карта**: спутниковая подложка + цветная поверхность кригинга + точки измерений.  
Ключ не нужно вводить каждый раз: приложение автоматически берёт его из `YANDEX_MAPS_API_KEY` в Streamlit Secrets или из переменных окружения.
"""
    )

    default_yandex_key, key_source = get_yandex_maps_api_key()

    with st.expander("Настройка API-ключа", expanded=not bool(default_yandex_key)):
        if default_yandex_key:
            st.success(f"API-ключ найден автоматически: {key_source}. Вводить его вручную не нужно.")
            override_key = st.text_input(
                "Временно заменить ключ вручную",
                value="",
                type="password",
                help="Оставьте поле пустым, чтобы использовать ключ из Secrets. Это нужно только для проверки другого ключа.",
            ).strip()
            yandex_api_key = override_key or default_yandex_key
        else:
            st.warning(
                "API-ключ не найден. Для локальной проверки можно временно вставить его ниже, "
                "но для GitHub/Streamlit Cloud ключ нужно хранить в Secrets, а не в коде."
            )
            yandex_api_key = st.text_input(
                "API-ключ Яндекс.Карт",
                value="",
                type="password",
                help="Ключ JavaScript API Яндекс.Карт. Для постоянного запуска добавьте его в Streamlit Secrets как YANDEX_MAPS_API_KEY.",
            ).strip()
            st.code('YANDEX_MAPS_API_KEY = "ваш_ключ_яндекс_карт"', language="toml")

        st.caption(
            "Для локального запуска в ограничениях ключа Яндекса используйте HTTP Referer: localhost. "
            "Для онлайн-деплоя добавьте домен приложения Streamlit без https:// и без пути."
        )

    map_df = plot_df[["Широта", "Долгота", "Прогноз_Ew"]].dropna().copy()
    map_df = map_df[
        map_df["Широта"].between(-90, 90)
        & map_df["Долгота"].between(-180, 180)
    ]

    if not yandex_api_key:
        st.warning(
            "Карта не будет строиться, пока не найден ключ. Добавьте `YANDEX_MAPS_API_KEY` в Streamlit Secrets "
            "или временно вставьте ключ в поле выше."
        )
    elif len(map_df) < 5:
        st.warning("Для построения кригинга требуется минимум 5 точек с координатами и прогнозом Ew.")
    else:
        try:
            # Слишком большое число полигонов может тормозить карту в браузере.
            # 34 даёт 1089 ячеек — достаточно детально и обычно стабильно для Streamlit iframe.
            unique_points, kriging_cells = build_kriging_cells(
                map_df,
                grid_size=34,
                mask_to_hull=False,
            )

            center_lat = float(unique_points["Широта"].mean())
            center_lon = float(unique_points["Долгота"].mean())
            min_lat = float(unique_points["Широта"].min())
            max_lat = float(unique_points["Широта"].max())
            min_lon = float(unique_points["Долгота"].min())
            max_lon = float(unique_points["Долгота"].max())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Точек измерения", len(unique_points))
            c2.metric("Ячеек кригинга", len(kriging_cells))
            c3.metric("Средний Ew", round(unique_points["Прогноз_Ew"].mean(), 2))
            c4.metric("Максимальный Ew", round(unique_points["Прогноз_Ew"].max(), 2))

            points_json = json.dumps(
                [
                    {
                        "lat": float(row["Широта"]),
                        "lon": float(row["Долгота"]),
                        "ew": round(float(row["Прогноз_Ew"]), 2),
                        "className": ew_class(float(row["Прогноз_Ew"])),
                    }
                    for _, row in unique_points.iterrows()
                ],
                ensure_ascii=False,
            )
            cells_json = json.dumps(kriging_cells, ensure_ascii=False)
            yandex_script_url = (
                "https://api-maps.yandex.ru/2.1/?"
                + "apikey=" + quote(yandex_api_key, safe="")
                + "&lang=ru_RU"
                + "&load=package.full"
            )

            map_id = "yandex_kriging_map_" + uuid.uuid4().hex
            html = f"""
<div style="width:100%;font-family:Arial, sans-serif;">
    <div id="map_status" style="padding:10px 12px;margin-bottom:8px;font-size:14px;color:#444;background:#f7f7f7;border:1px solid #ddd;border-radius:8px;">
        Загружаю JavaScript API Яндекс.Карт...
    </div>
    <div id="{map_id}" style="width:100%;height:720px;border-radius:12px;overflow:hidden;background:#eeeeee;border:1px solid #d0d0d0;"></div>
    <div id="map_debug" style="margin-top:8px;font-size:12px;color:#666;line-height:1.45;"></div>
</div>

<script>
(function() {{
    const mapId = "{map_id}";
    const statusEl = document.getElementById("map_status");
    const debugEl = document.getElementById("map_debug");
    const center = [{center_lat}, {center_lon}];
    const bounds = [[{min_lat}, {min_lon}], [{max_lat}, {max_lon}]];
    const cells = {cells_json};
    const points = {points_json};
    let mapWasCreated = false;

    function setStatus(text, color, bg) {{
        if (statusEl) {{
            statusEl.innerHTML = text;
            statusEl.style.color = color || "#444";
            if (bg) statusEl.style.background = bg;
        }}
    }}

    function setDebug(extra) {{
        if (debugEl) {{
            debugEl.innerHTML =
                "<b>Диагностика iframe:</b><br>" +
                "document.referrer: " + (document.referrer || "пусто") + "<br>" +
                "window.location.href: " + window.location.href + "<br>" +
                (extra || "");
        }}
    }}

    function createMap() {{
        try {{
            if (!window.ymaps) {{
                setStatus("ymaps не найден: API Яндекс.Карт не загрузился.", "#b00020", "#fff0f0");
                setDebug("Проверьте ключ и ограничения HTTP Referer в кабинете Яндекса.");
                return;
            }}

            window.ymaps.ready(function() {{
                try {{
                    const map = new window.ymaps.Map(mapId, {{
                        center: center,
                        zoom: 16,
                        type: "yandex#satellite",
                        controls: ["zoomControl", "typeSelector", "fullscreenControl", "rulerControl"]
                    }});

                    map.setBounds(bounds, {{checkZoomRange: true, zoomMargin: 40}});

                    const krigingCollection = new window.ymaps.GeoObjectCollection();
                    cells.forEach(function(cell) {{
                        krigingCollection.add(new window.ymaps.Polygon(
                            [cell.coords],
                            {{
                                hintContent: "Кригинг Ew: " + Number(cell.value).toFixed(2),
                                balloonContent:
                                    "<b>Кригинг Ew:</b> " + Number(cell.value).toFixed(2) +
                                    "<br><b>Класс:</b> " + cell.className
                            }},
                            {{
                                fillColor: cell.color,
                                fillOpacity: 0.56,
                                strokeColor: cell.color,
                                strokeOpacity: 0.10,
                                strokeWidth: 1
                            }}
                        ));
                    }});
                    map.geoObjects.add(krigingCollection);

                    const pointsCollection = new window.ymaps.GeoObjectCollection();
                    points.forEach(function(p) {{
                        pointsCollection.add(new window.ymaps.Circle(
                            [[p.lat, p.lon], 5],
                            {{
                                hintContent: "Точка измерения: Ew " + Number(p.ew).toFixed(2),
                                balloonContent:
                                    "<b>Точка измерения пенетрации</b><br>" +
                                    "Ew: " + Number(p.ew).toFixed(2) + "<br>" +
                                    "Класс: " + p.className + "<br>" +
                                    "Широта: " + Number(p.lat).toFixed(6) + "<br>" +
                                    "Долгота: " + Number(p.lon).toFixed(6)
                            }},
                            {{
                                fillColor: "#ff0000",
                                fillOpacity: 0.96,
                                strokeColor: "#000000",
                                strokeWidth: 3,
                                zIndex: 10000
                            }}
                        ));
                    }});
                    map.geoObjects.add(pointsCollection);

                    mapWasCreated = true;
                    setStatus(
                        "Яндекс.Карта загружена. Можно приближать, двигать карту и нажимать на точки/ячейки.",
                        "#1b5e20",
                        "#eef8ee"
                    );
                    setDebug("Карта создана успешно. Полигонов кригинга: " + cells.length + "; точек: " + points.length + ".");
                }} catch (err) {{
                    console.error(err);
                    setStatus("Ошибка при создании карты: " + err.message, "#b00020", "#fff0f0");
                    setDebug("Чаще всего причина — ограничения ключа по HTTP Referer или неактивированный ключ.");
                }}
            }});
        }} catch (err) {{
            console.error(err);
            setStatus("Ошибка загрузки карты: " + err.message, "#b00020", "#fff0f0");
            setDebug("Проверьте консоль браузера и ограничения ключа.");
        }}
    }}

    window.addEventListener("error", function(event) {{
        if (!mapWasCreated) {{
            setStatus("JavaScript-ошибка: " + event.message, "#b00020", "#fff0f0");
            setDebug("Проверьте ограничения ключа Яндекса и домен запуска приложения.");
        }}
    }});

    const script = document.createElement("script");
    script.src = "{yandex_script_url}";
    script.type = "text/javascript";
    script.async = true;
    script.onload = function() {{
        setStatus("API Яндекс.Карт загружен, создаю карту...", "#555", "#f7f7f7");
        createMap();
    }};
    script.onerror = function() {{
        setStatus(
            "Не удалось загрузить JavaScript API Яндекс.Карт. Проверьте ключ, интернет, блокировщики и ограничения HTTP Referer.",
            "#b00020",
            "#fff0f0"
        );
        setDebug("Для локального запуска добавьте Referer: localhost. Для Streamlit Cloud — домен приложения без https://.");
    }};
    document.head.appendChild(script);

    setDebug("Ожидание загрузки API...");
    setTimeout(function() {{
        if (!mapWasCreated) {{
            setStatus(
                "Карта пока не появилась. Если блок остаётся пустым, проверьте HTTP Referer в ключе Яндекса. " +
                "Для локального запуска нужен localhost, для онлайн-деплоя — домен вашего Streamlit-приложения.",
                "#b36b00",
                "#fff8e6"
            );
        }}
    }}, 12000);
}})();
</script>
"""
            components.html(html, height=820, scrolling=True)

            st.markdown(
                """
**Легенда Ew**  
🔵 низкие значения Ew  
🟢 средние значения Ew  
🟡 повышенные значения Ew  
🔴 высокие значения Ew  

Если карта не появилась, но метрики выше есть, значит кригинг посчитан, а проблема только в загрузке Яндекс API/ключа.
"""
            )

        except Exception as e:
            st.error(f"Не удалось построить Яндекс.Карту кригинга: {e}")
            st.info(
                "Проверьте, что есть минимум 5 разных точек, координаты не лежат на одной линии, "
                "pykrige установлен, а API-ключ Яндекс.Карт действителен."
            )

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
    st.subheader("Выводы о проходимости техники по полю")
    st.info(
        "Рекомендации теперь формируются с учётом того, что сопротивление пенетрации Ew влияет на проходимость техники: "
        "низкие значения могут указывать на мягкие/переувлажнённые участки с риском колеи, "
        "а высокие значения — на плотные участки, где повторные проходы техники усиливают переуплотнение."
    )

    if len(plot_df) == 0:
        st.warning("Нет прогнозных значений Ew для формирования выводов о проходимости техники.")
    else:
        ew_values = plot_df["Прогноз_Ew"].dropna()
        mean_ew = float(ew_values.mean())
        median_ew = float(ew_values.median())
        min_ew = float(ew_values.min())
        max_ew = float(ew_values.max())
        low_share = float((ew_values < 25).mean() * 100)
        elevated_share = float((ew_values >= 35).mean() * 100)
        high_share = float((ew_values >= 55).mean() * 100)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Средний Ew", round(mean_ew, 2))
        m2.metric("Медианный Ew", round(median_ew, 2))
        m3.metric("Минимальный Ew", round(min_ew, 2))
        m4.metric("Максимальный Ew", round(max_ew, 2))
        m5.metric("Доля Ew < 25", f"{low_share:.1f}%")
        m6.metric("Доля Ew ≥ 35", f"{elevated_share:.1f}%")

        status, recommendation = overall_field_recommendation(mean_ew, high_share, elevated_share, low_share)

        if "ограничена" in status or "проверки" in status:
            st.warning(f"**Итог по проходимости:** {status}")
        elif "допустима" in status:
            st.info(f"**Итог по проходимости:** {status}")
        else:
            st.success(f"**Итог по проходимости:** {status}")

        st.markdown(f"**Рекомендация по движению техники:** {recommendation}")

        st.markdown("### Распределение точек по влиянию Ew на проходимость")
        class_stats = (
            ew_values.apply(ew_class)
            .value_counts()
            .rename_axis("Класс сопротивления")
            .reset_index(name="Количество точек")
        )
        class_stats["Доля, %"] = (class_stats["Количество точек"] / len(ew_values) * 100).round(1)
        class_stats["Интерпретация для техники"] = class_stats["Класс сопротивления"].map(
            {
                "низкое сопротивление": "проверить риск колеи и буксования",
                "рабочее сопротивление": "обычная проходимость при нормальной влажности",
                "повышенное сопротивление": "движение по маршрутам, без лишних проходов",
                "высокое сопротивление": "исключить повторные проходы тяжёлой техники",
            }
        )
        st.dataframe(class_stats, use_container_width=True)

        st.markdown("### Краткий вывод")
        if high_share > 0:
            st.markdown(
                f"На поле есть зоны высокого сопротивления Ew: **{high_share:.1f}%** точек имеют Ew ≥ 55. "
                "Эти участки не стоит считать просто удобными для проезда: техника может пройти, "
                "но повторные проходы будут закреплять переуплотнение."
            )
        if low_share > 0:
            st.markdown(
                f"Также есть зоны низкого сопротивления Ew: **{low_share:.1f}%** точек имеют Ew < 25. "
                "Перед заездом тяжёлой техники их нужно проверять по влажности, потому что именно там выше риск колеи и буксования."
            )
        if high_share == 0 and low_share == 0:
            st.markdown(
                "Крайних зон по Ew не выявлено. Основная рекомендация — сохранять движение техники по постоянным маршрутам "
                "и не выполнять лишние проходы по полю."
            )

        if zone_col is not None and zone_col in plot_df.columns:
            st.markdown("### Локальные рекомендации по зонам для движения техники")

            zone_stats = (
                plot_df.groupby(zone_col)["Прогноз_Ew"]
                .agg(
                    Количество_точек="count",
                    Средний_Ew="mean",
                    Минимальный_Ew="min",
                    Максимальный_Ew="max",
                )
                .reset_index()
                .rename(columns={zone_col: "Зона"})
            )

            low_by_zone = (
                plot_df.assign(Низкое=plot_df["Прогноз_Ew"] < 25)
                .groupby(zone_col)["Низкое"]
                .mean()
                .reset_index(drop=True)
                * 100
            )
            high_by_zone = (
                plot_df.assign(Высокое=plot_df["Прогноз_Ew"] >= 55)
                .groupby(zone_col)["Высокое"]
                .mean()
                .reset_index(drop=True)
                * 100
            )
            elevated_by_zone = (
                plot_df.assign(Повышенное=plot_df["Прогноз_Ew"] >= 35)
                .groupby(zone_col)["Повышенное"]
                .mean()
                .reset_index(drop=True)
                * 100
            )

            zone_stats["Доля Ew < 25, %"] = low_by_zone.round(1)
            zone_stats["Доля Ew ≥ 35, %"] = elevated_by_zone.round(1)
            zone_stats["Доля Ew ≥ 55, %"] = high_by_zone.round(1)
            zone_stats["Класс сопротивления"] = zone_stats["Средний_Ew"].apply(ew_class)
            zone_stats["Оценка проходимости"] = zone_stats["Средний_Ew"].apply(trafficability_class)
            zone_stats["Что делать технике"] = zone_stats["Средний_Ew"].apply(ew_zone_action)
            zone_stats["Средний_Ew"] = zone_stats["Средний_Ew"].round(2)
            zone_stats["Минимальный_Ew"] = zone_stats["Минимальный_Ew"].round(2)
            zone_stats["Максимальный_Ew"] = zone_stats["Максимальный_Ew"].round(2)

            st.dataframe(zone_stats, use_container_width=True)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(zone_stats["Зона"].astype(str), zone_stats["Средний_Ew"])
            ax.axhline(25, linestyle="--", linewidth=1, label="Низкое сопротивление: проверить несущую способность")
            ax.axhline(35, linestyle="--", linewidth=1, label="Повышенное сопротивление: ограничить лишние проходы")
            ax.axhline(55, linestyle="--", linewidth=1, label="Высокое сопротивление: риск переуплотнения")
            ax.set_title("Средний прогноз Ew по зонам и проходимость техники")
            ax.set_xlabel("Зона")
            ax.set_ylabel("Средний прогноз Ew")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            plt.xticks(rotation=45)
            show_matplotlib(fig)
        else:
            st.markdown(
                "### Локальные рекомендации по зонам\n"
                "В данных нет столбца зоны, поэтому локальные выводы по проходимости лучше делать по карте кригинга во вкладке **Карта**."
            )

        st.markdown(
            """
### Как интерпретировать выводы для техники
- **Ew < 25** — участок может быть мягким или переувлажнённым: перед заездом проверить риск колеи и буксования.
- **Ew 25–35** — рабочий диапазон: техника обычно может проходить при нормальной влажности.
- **Ew 35–55** — повышенное сопротивление: технику лучше вести по постоянным маршрутам и не делать лишних проходов.
- **Ew ≥ 55** — зона высокого сопротивления: повторные проходы тяжёлой техники могут усиливать переуплотнение; после работ желательно планировать рыхление/щелевание и повторный контроль.
"""
        )

# =====================================================
# ВЫГРУЗКА
# =====================================================

st.download_button(
    "Скачать результаты CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="extra_trees_results.csv",
    mime="text/csv",
)
