
import os
import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "models/weather_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Модель не найдена. Сначала запустите train_model.py"
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "pressure",
    "humidity",
    "wind_speed",
    "temp_today",
    "temp_yesterday",
    "temp_2_days_ago",
    "temp_3_days_ago"
]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    values = {
        "pressure": "1013",
        "humidity": "60",
        "wind_speed": "5",
        "temp_today": "15",
        "temp_yesterday": "14",
        "temp_2_days_ago": "13",
        "temp_3_days_ago": "12"
    }

    if request.method == "POST":
        try:
            for key in FEATURES:
                values[key] = request.form.get(key, "").strip()

                if not values[key]:
                    raise ValueError("Заполните все поля.")

            data = {key: float(values[key]) for key in FEATURES}

            if not all(pd.notna(v) and abs(v) != float("inf")
                       for v in data.values()):
                raise ValueError("Введите корректные числа.")

            if not 800 <= data["pressure"] <= 1100:
                raise ValueError(
                    "Давление должно быть от 800 до 1100 гПа."
                )

            if not 0 <= data["humidity"] <= 100:
                raise ValueError(
                    "Влажность должна быть от 0 до 100%."
                )

            if not 0 <= data["wind_speed"] <= 150:
                raise ValueError(
                    "Скорость ветра должна быть от 0 до 150 км/ч."
                )

            temperature_fields = [
                "temp_today",
                "temp_yesterday",
                "temp_2_days_ago",
                "temp_3_days_ago"
            ]

            for key in temperature_fields:
                if not -90 <= data[key] <= 60:
                    raise ValueError(
                        "Температура должна быть от -90 до 60 °C."
                    )

            input_data = pd.DataFrame([data], columns=FEATURES)
            prediction = float(model.predict(input_data)[0])
            result = round(prediction, 1)

        except (ValueError, TypeError, OverflowError) as exc:
            error = str(exc) or "Проверьте введённые данные."

    return render_template(
        "index.html",
        result=result,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)