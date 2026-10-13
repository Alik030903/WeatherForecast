
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Создаём папку для сохранения модели
os.makedirs("models", exist_ok=True)

# Загружаем данные
df = pd.read_csv("data/weather_data.csv")

# Выбираем признаки для прогноза
features = [
    "pressure",
    "humidity",
    "wind_speed",
    "temp_today",
    "temp_yesterday",
    "temp_2_days_ago",
    "temp_3_days_ago"
]

X = df[features]
y = df["temp_tomorrow"]

# Разделяем данные: 80% обучение, 20% тестирование
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Создаём модель машинного обучения
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# Обучаем модель
print("Обучение модели...")
model.fit(X_train, y_train)

# Проверяем качество прогноза
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# Сохраняем обученную модель
joblib.dump(model, "models/weather_model.pkl")

print("\nМодель успешно обучена!")
print(f"Обучающих записей: {len(X_train)}")
print(f"Тестовых записей: {len(X_test)}")
print(f"MAE:  {mae:.2f} °C")
print(f"RMSE: {rmse:.2f} °C")
print(f"R²:   {r2:.3f}")
print("\nМодель сохранена: models/weather_model.pkl")