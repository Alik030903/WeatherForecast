
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загружаем данные и обученную модель
df = pd.read_csv("data/weather_data.csv")
model = joblib.load("models/weather_model.pkl")

features = [
    "pressure",
    "humidity",
    "wind_speed",
    "temp_today",
    "temp_yesterday",
    "temp_2_days_ago",
    "temp_3_days_ago"
]

# Воспроизводим то же тестовое разделение
from sklearn.model_selection import train_test_split

X = df[features]
y = df["temp_tomorrow"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Прогноз сохранённой моделью на тестовой выборке
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# Сравнение с простым прогнозом средним значением
baseline = [y_train.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline)

print("=== ПРОВЕРКА WEATHERFORECAST ===")
print(f"Всего записей: {len(df)}")
print(f"Обучающих записей: {len(X_train)}")
print(f"Тестовых записей: {len(X_test)}")
print(f"MAE модели: {mae:.2f} °C")
print(f"RMSE модели: {rmse:.2f} °C")
print(f"R² модели: {r2:.3f}")
print(f"MAE простого прогноза: {baseline_mae:.2f} °C")

print("\nПримеры прогнозов:")
for i in range(5):
    print(
        f"Факт: {y_test.iloc[i]:.1f} °C | "
        f"Прогноз: {predictions[i]:.1f} °C"
    )

if mae < baseline_mae:
    print("\nПроверка: модель лучше простого прогноза по MAE.")
else:
    print("\nПроверка: модель не улучшила MAE простого прогноза.")
