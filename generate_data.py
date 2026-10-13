
import os
import numpy as np
import pandas as pd

# Создаём папку для данных
os.makedirs("data", exist_ok=True)

# Генерируем воспроизводимые демонстрационные данные
rng = np.random.default_rng(42)
n = 2000

# Температура за предыдущие три дня и сегодня
temp_3_days_ago = rng.uniform(-15, 30, n)
temp_2_days_ago = temp_3_days_ago + rng.normal(0, 3, n)
temp_yesterday = temp_2_days_ago + rng.normal(0, 3, n)
temp_today = temp_yesterday + rng.normal(0, 3, n)

# Метеорологические показатели
pressure = rng.normal(1013, 12, n).clip(970, 1050)
humidity = rng.uniform(20, 100, n)
wind_speed = rng.uniform(0, 25, n)

# Демонстрационная температура следующего дня
temp_tomorrow = (
    0.48 * temp_today
    + 0.25 * temp_yesterday
    + 0.12 * temp_2_days_ago
    + 0.08 * temp_3_days_ago
    + 0.025 * (pressure - 1013)
    - 0.025 * (humidity - 60)
    - 0.06 * wind_speed
    + rng.normal(0, 2, n)
)

# Формируем таблицу
df = pd.DataFrame({
    "pressure": pressure,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "temp_today": temp_today,
    "temp_yesterday": temp_yesterday,
    "temp_2_days_ago": temp_2_days_ago,
    "temp_3_days_ago": temp_3_days_ago,
    "temp_tomorrow": temp_tomorrow
})

# Сохраняем данные в CSV
df.to_csv("data/weather_data.csv", index=False)

print("Данные успешно созданы!")
print("Количество записей:", len(df))
print("\nПервые 5 записей:")
print(df.head())
print("\nНазвания столбцов:")
print(df.columns.tolist())