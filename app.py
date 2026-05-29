import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from datasets import load_dataset

st.set_page_config(page_title="Electricity Forecast App", layout="wide")

st.title("📊 Predicción de Demanda Eléctrica con Machine Learning")
st.write("Modelo MLPRegressor para forecasting de consumo eléctrico horario en Brasil")

# =========================
# CARGA DE DATOS
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if st.button("📥 Cargar dataset"):
    dataset = load_dataset(
        "SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset",
        split="train"
    )
    df = pd.DataFrame(dataset)
    st.session_state.df = df
    st.success("Dataset cargado correctamente")
    st.write(df.head())

# =========================
# FILTRO
# =========================
if st.session_state.df is not None:
    df = st.session_state.df

    region = st.selectbox("Selecciona región", df["nom_subsistema"].unique())

    df = df[df["nom_subsistema"] == region]

    st.write(f"Registros en {region}: {len(df)}")

    # =========================
    # PREPROCESAMIENTO
    # =========================
    if st.button("🤖 Entrenar modelo"):

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["din_instante"])
        df = df.sort_values("timestamp")

        data = df[["timestamp", "val_cargaenergiahomwmed"]].copy()
        data.rename(columns={"val_cargaenergiahomwmed": "demanda"}, inplace=True)

        scaler = MinMaxScaler()
        data["demanda_norm"] = scaler.fit_transform(data[["demanda"]])

        # Ventanas
        time_step = 24
        X, y, X_dates = [], [], []

        for i in range(len(data) - time_step):
            X.append(data["demanda_norm"].values[i:i+time_step])
            y.append(data["demanda_norm"].values[i+time_step])
            X_dates.append(data["timestamp"].iloc[i+time_step])

        X = np.array(X)
        y = np.array(y)
        X_dates = np.array(X_dates)

        # Split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_test = X_dates[train_size:]

        # Modelo
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Predicción test
        pred_test = model.predict(X_test)

        pred_test_actual = scaler.inverse_transform(pred_test.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_test_actual, pred_test_actual))

        errors = y_test_actual.flatten() - pred_test_actual.flatten()
        std_dev = np.std(errors)
        normalized_std_dev = std_dev / np.mean(y_test_actual)

        # =========================
        # MÉTRICAS
        # =========================
        st.subheader("📊 Métricas del modelo")

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("Std Error", f"{std_dev:.2f}")
        col3.metric("Error normalizado", f"{normalized_std_dev:.4f}")

        # =========================
        # GRÁFICA TEST
        # =========================
        st.subheader("📈 Predicción vs Real (Test)")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(dates_test[:200], y_test_actual[:200], label="Real")
        ax.plot(dates_test[:200], pred_test_actual[:200], label="Predicción")
        ax.legend()
        ax.set_title("Test Set")
        plt.xticks(rotation=45)

        st.pyplot(fig)

        # =========================
        # FORECAST FUTURO
        # =========================
        st.subheader("🔮 Predicción futura (48 horas)")

        future_steps = 48
        future_predictions = []
        last_window = X[-1]
        last_timestamp = data["timestamp"].iloc[-1]
        future_dates = []

        for i in range(future_steps):
            pred_next = model.predict(last_window.reshape(1, -1))[0]
            future_predictions.append(pred_next)

            last_window = np.roll(last_window, -1)
            last_window[-1] = pred_next

            future_dates.append(last_timestamp + pd.Timedelta(hours=i+1))

        future_actual = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )

        future_df = pd.DataFrame({
            "timestamp": future_dates,
            "prediccion": future_actual.flatten()
        })

        st.write(future_df)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(future_dates, future_actual, marker="o")
        ax2.set_title("Forecast 48 horas")
        plt.xticks(rotation=45)

        st.pyplot(fig2)

        st.success("Modelo entrenado y predicciones generadas")