import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
import folium
import branca.colormap as cm
import os
import warnings

warnings.filterwarnings("ignore")

# ===================== НАСТРОЙКИ =====================
file_path = 'C:\учеба\Диплом\cleaned_data.xlsx'
base_dir = 'C:\учеба\Диплом\kriging'

os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/maps', exist_ok=True)
os.makedirs(f'{base_dir}/monthly_maps', exist_ok=True)
# ===================== ЗАГРУЗКА =====================
print("Загружаем данные...")
df = pd.read_excel(file_path, sheet_name='Sheet1')
df.columns = df.columns.str.strip()

# ←←← ИСПРАВЛЕНИЕ ДАТЫ
if 'Дата, Время' in df.columns:
    df = df.rename(columns={'Дата, Время': 'Дата'})

df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
df['Месяц'] = df['Дата'].dt.strftime('%B')

print("Месяцы:", sorted(df['Месяц'].unique()))

# ===================== ПЕРЕМЕННЫЕ =====================
variables = [
    'Сопротивление пенетрации, Ew',
    'Значение\nиндикатора\nпенетрометра,\nмкм',
    'Влажность,  %',
    'Плотность, г/см^3',
    'Усилие пенетрации, P',
    'Удельное сцепление грунта',          
    'Модуль упругости, Eу',
    'Модуль упругости, E (Мпа)',
    'Модуль деформации, E',            
    'Индекс конуса, CI (кПа)'
]

# ===================== ФУНКЦИЯ KRIGING =====================
def create_kriging_map(data, var, title, save_path, is_overall=False):
    if len(data) < 10:
        print(f"⚠ Мало данных для {title}")
        return
    
    x = data['Долгота'].values
    y = data['Широта'].values
    z = data[var].values

    # Улучшенные параметры Kriging
    ok = OrdinaryKriging(
        x, y, z,
        variogram_model='spherical',
        variogram_parameters={'sill': np.var(z)*0.8, 'range': 0.012, 'nugget': np.var(z)*0.15},
        nlags=25,
        verbose=False,
        enable_plotting=False
    )

    gridx = np.linspace(x.min() - 0.0008, x.max() + 0.0008, 100)
    gridy = np.linspace(y.min() - 0.0008, y.max() + 0.0008, 100)
    
    zgrid, ss = ok.execute('grid', gridx, gridy)

    m = folium.Map(location=[y.mean(), x.mean()], zoom_start=14.5,
                   tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google')

    # Улучшенная цветовая шкала
    if is_overall:
        vmin = np.nanpercentile(zgrid, 5)
        vmax = np.nanpercentile(zgrid, 95)
        title += " (5-95%)"
    else:
        vmin, vmax = np.nanpercentile(zgrid, [2, 98])

    colormap = cm.LinearColormap(
        colors=['#0000ff', '#00aaff', '#00ff88', '#ffff00', '#ff8800', '#ff0000'],
        vmin=vmin, vmax=vmax, caption=f"{var}\n({title})"
    )
    colormap.add_to(m)

    # Растровая поверхность
    for i in range(len(gridy)):
        for j in range(len(gridx)):
            val = zgrid[i, j]
            if not np.isnan(val):
                folium.Rectangle(
                    bounds=[[gridy[i]-0.00035, gridx[j]-0.00035], 
                           [gridy[i]+0.00035, gridx[j]+0.00035]],
                    fill=True,
                    fill_color=colormap(val),
                    fill_opacity=0.65,
                    stroke=False
                ).add_to(m)

    # Точки измерений
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Широта'], row['Долгота']],
            radius=4.5,
            popup=f"<b>{var}:</b> {row[var]:.2f}<br>"
                  f"Зона: {row.get('Прямоугольная зона №', '—')}<br>"
                  f"Месяц: {row.get('Месяц', '—')}",
            tooltip=f"{row[var]:.1f}",
            color='black',
            fill_color='red',
            fill_opacity=0.9
        ).add_to(m)

    m.save(save_path)
    print(f"✅ {title} — {len(data)} точек")


# ===================== ЗАПУСК =====================
print("\n=== Создаём Kriging карты по всем переменным ===")

for var in variables:
    print(f"\n🔄 Обработка переменной: {var}")
    safe_name = "".join(c if c.isalnum() else "_" for c in var[:25])
    
    # Общая карта
    # Общая карта
    create_kriging_map(df, var, f'Общая Kriging — {var}', 
                  f'{base_dir}/maps/kriging_overall_{safe_name}.html', 
                  is_overall=True)
    
    # По всем месяцам
    for month in sorted(df['Месяц'].unique()):
        month_data = df[df['Месяц'] == month].copy()
        if len(month_data) < 10:
            print(f"   Пропущен {month} — мало данных")
            continue
            
        month_dir = f'{base_dir}/monthly_maps/{month}'
        os.makedirs(month_dir, exist_ok=True)
        
        create_kriging_map(month_data, var, f'Kriging {var} — {month}',
                          f'{month_dir}/kriging_{safe_name}_{month}.html')

print(f"\n ВСЁ ГОТОВО!")