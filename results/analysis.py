import pandas as pd
import folium
import branca.colormap as cm
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== НАСТРОЙКИ =====================
file_path = r'C:\учеба\Диплом\cleaned_data.xlsx'
base_dir = r'C:\учеба\Диплом\results'

os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/maps', exist_ok=True)
os.makedirs(f'{base_dir}/monthly_maps', exist_ok=True)
os.makedirs(f'{base_dir}/correlation_matrices', exist_ok=True)

# ===================== ЗАГРУЗКА =====================
print("Загружаем данные...")
df = pd.read_excel(file_path, sheet_name='Sheet1')
df.columns = df.columns.str.strip()

if 'Дата, Время' in df.columns:
    df = df.rename(columns={'Дата, Время': 'Дата'})

df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
df['Месяц'] = df['Дата'].dt.strftime('%B')

print("Месяцы:", sorted(df['Месяц'].unique()))

# ===================== ПЕРЕМЕННЫЕ =====================
variables = [
    'Значение\nиндикатора\nпенетрометра,\nмкм',
    'Влажность,  %',              
    'Плотность, г/см^3',
    'Усилие пенетрации, P',
    'Сопротивление пенетрации, Ew',
    'Удельное сцепление грунта',
    'Модуль упругости, Eу',
    'Модуль упругости, E (Мпа)',
    'Модуль деформации, E',
    'Индекс конуса, CI (кПа)'
]

valid_vars = [v for v in variables if v in df.columns]
print(f"\nИспользуемые переменные ({len(valid_vars)}):")
print(valid_vars)

# ===================== МАТРИЦЫ КОРРЕЛЯЦИЙ =====================
print("\nСоздаём матрицы корреляций...")

# Общая матрица
corr_matrix = df[valid_vars].corr()
corr_matrix.to_csv(f'{base_dir}/correlation_matrices/overall_correlation.csv')

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('Матрица корреляций (вся выборка)')
plt.tight_layout()
plt.savefig(f'{base_dir}/correlation_matrices/overall_correlation_heatmap.png', dpi=300)
plt.close()

print("✓ Общая матрица корреляций сохранена")

# По месяцам
print("\nСоздаём матрицы корреляций по месяцам...")
for month in sorted(df['Месяц'].unique()):
    month_data = df[df['Месяц'] == month]
    if len(month_data) < 5:
        print(f"⚠ Пропущен {month} — мало данных ({len(month_data)})")
        continue
        
    month_corr = month_data[valid_vars].corr()
    
    # Сохраняем CSV
    month_corr.to_csv(f'{base_dir}/correlation_matrices/correlation_{month}.csv')
    
    # Создаём heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(month_corr, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5, square=True)
    plt.title(f'Матрица корреляций — {month} ({len(month_data)} наблюдений)')
    plt.tight_layout()
    plt.savefig(f'{base_dir}/correlation_matrices/correlation_{month}_heatmap.png', dpi=300)
    plt.close()
    
    print(f"✓ {month} — {len(month_data)} точек")

# ===================== КАРТЫ =====================
def create_google_map(data, var, title, save_path):
    if len(data) == 0:
        return
    center_lat = data['Широта'].mean()
    center_lon = data['Долгота'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14.5,
                   tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google')

    vmin = data[var].quantile(0.05)
    vmax = data[var].quantile(0.95)
    
    colormap = cm.LinearColormap(
        colors=['red', 'orange', 'yellow', 'lightgreen', 'blue'],
        vmin=vmin, vmax=vmax, caption=var
    )
    colormap.add_to(m)

    for _, row in data.iterrows():
        value = row[var]
        color = colormap(value)
        
        popup_text = f"""
        <h4>Зона №{row.get('Прямоугольная зона №', '—')}</h4>
        <b>{var}:</b> {value:.2f}<br>
        <b>Влажность:</b> {row.get('Влажность,  %', '—')} %<br>
        <b>Плотность:</b> {row.get('Плотность, г/см^3', '—')} г/см³<br>
        <b>E (МПа):</b> {row.get('Модуль упругости, E (Мпа)', '—')}<br>
        """

        folium.CircleMarker(
            location=[row['Широта'], row['Долгота']],
            radius=7,
            popup=folium.Popup(popup_text, max_width=320),
            tooltip=f"{value:.1f}",
            color='black',
            fill_color=color,
            fillOpacity=0.85
        ).add_to(m)

    # Зоны
    if 'Прямоугольная зона №' in data.columns:
        for zone in sorted(data['Прямоугольная зона №'].unique()):
            zone_data = data[data['Прямоугольная зона №'] == zone]
            if len(zone_data) > 2:
                points = list(zip(zone_data['Широта'], zone_data['Долгота']))
                folium.Polygon(locations=points, color='blue', weight=2.5, 
                             fill=True, fillOpacity=0.08, popup=f'Зона №{zone}').add_to(m)

    m.save(save_path)
    print(f"✓ {title}")

# ===================== ЗАПУСК КАРТ =====================
print("\n=== Общие карты ===")
for var in valid_vars:
    safe_name = "".join(c if c.isalnum() else "_" for c in var[:40])
    create_google_map(df, var, f'Общая — {var}', f'{base_dir}/maps/overall_{safe_name}.html')

print("\n=== Карты по месяцам ===")
for month in sorted(df['Месяц'].unique()):
    month_data = df[df['Месяц'] == month].copy()
    month_dir = f'{base_dir}/monthly_maps/{month}'
    os.makedirs(month_dir, exist_ok=True)
    print(f"→ {month} ({len(month_data)} точек)")
    for var in valid_vars:
        safe_name = "".join(c if c.isalnum() else "_" for c in var[:40])
        create_google_map(month_data, var, f'{var} — {month}', 
                         f'{month_dir}/{safe_name}.html')

print("\n Всё готово!")