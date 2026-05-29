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
        # TABLA COMPARATIVA TEST
        # =========================
        st.subheader("📋 Tabla comparativa (Test)")

        comp_df = pd.DataFrame({
            "timestamp": dates_test[:50],
            "real": y_test_actual[:50].flatten(),
            "prediccion": pred_test_actual[:50].flatten()
        })

        comp_df["% variacion"] = ((comp_df["prediccion"] - comp_df["real"]) / comp_df["real"]) * 100

        st.dataframe(comp_df.style.format({
            "real": "{:.2f}",
            "prediccion": "{:.2f}",
            "% variacion": "{:.2f}%"
        }))

        # =========================
        # MÉTRICAS
        # =========================
        st.subheader("📊 Métricas del modelo")

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("Std Error", f"{std_dev:.2f}")
        col3.metric("Error normalizado", f"{normalized_std_dev:.4f}")

        # =========================
        # INTERPRETACIÓN DE RESULTADOS
        # =========================
        mean_demand = np.mean(y_test_actual)
        min_demand = np.min(y_test_actual)
        max_demand = np.max(y_test_actual)
        demand_range = max_demand - min_demand

        # RMSE
        rmse_ratio = rmse / mean_demand
        if rmse_ratio < 0.05:
            rmse_interp = "EXCELLENT (<5% of mean demand) — industrial-grade forecasting accuracy"
            rmse_range = "< 5%"
        elif rmse_ratio < 0.10:
            rmse_interp = "GOOD (5%–10% of mean demand) — reliable forecasting for operational use"
            rmse_range = "5%–10%"
        elif rmse_ratio < 0.20:
            rmse_interp = "MODERATE (10%–20%) — captures general trend but with noticeable deviations"
            rmse_range = "10%–20%"
        else:
            rmse_interp = "HIGH ERROR (>20%) — model struggles to capture demand dynamics"
            rmse_range = "> 20%"
        rmse_value_pct = rmse_ratio * 100

        # STD ERROR
        std_ratio = std_dev / mean_demand
        if std_ratio < 0.05:
            std_interp = "VERY STABLE (<5%) — low dispersion, consistent predictions"
            std_range = "< 5%"
        elif std_ratio < 0.10:
            std_interp = "STABLE (5%–10%) — acceptable variance in predictions"
            std_range = "5%–10%"
        elif std_ratio < 0.20:
            std_interp = "MODERATE VARIABILITY (10%–20%) — noticeable fluctuation in errors"
            std_range = "10%–20%"
        else:
            std_interp = "HIGH VOLATILITY (>20%) — unstable error distribution"
            std_range = "> 20%"
        std_value_pct = std_ratio * 100

        # NORMALIZED ERROR
        if normalized_std_dev < 0.05:
            norm_interp = "EXCELLENT (<5%) — industrial-grade relative accuracy"
            norm_range = "< 5%"
        elif normalized_std_dev < 0.10:
            norm_interp = "GOOD (5%–10%) — reliable predictive performance"
            norm_range = "5%–10%"
        elif normalized_std_dev < 0.20:
            norm_interp = "ACCEPTABLE (10%–20%) — usable but improvable model"
            norm_range = "10%–20%"
        else:
            norm_interp = "HIGH ERROR (>20%) — weak relative performance"
            norm_range = "> 20%"
        norm_value_pct = normalized_std_dev * 100

        st.subheader("🧠 Interpretación de resultados")
        st.write(f"""
📊 **Contexto del sistema ({region} / Brazil):**
- Mean demand: {mean_demand:.2f} MWmed  
- Min demand: {min_demand:.2f} MWmed  
- Max demand: {max_demand:.2f} MWmed  
- Demand range: {demand_range:.2f} MWmed  

---

## 📉 RMSE Interpretation
- Value: **{rmse:.2f} MWmed**
- Relative error: **{rmse_value_pct:.2f}%**
- Benchmark ranges:
  - <5% → Excellent  
  - 5–10% → Good  
  - 10–20% → Moderate  
  - >20% → High error  

👉 This model falls in: **{rmse_range}**  
{rmse_interp}

---

## 📊 Error Dispersion (Std Error)
- Value: **{std_dev:.2f} MWmed**
- Relative error: **{std_value_pct:.2f}%**
- Benchmark ranges:
  - <5% → Very stable  
  - 5–10% → Stable  
  - 10–20% → Moderate variability  
  - >20% → High volatility  

👉 This model falls in: **{std_range}**  
{std_interp}

---

## 📌 Normalized Error
- Value: **{normalized_std_dev:.4f}**
- Percentage: **{norm_value_pct:.2f}%**
- Benchmark ranges:
  - <5% → Excellent  
  - 5–10% → Good  
  - 10–20% → Acceptable  
  - >20% → High error  

👉 This model falls in: **{norm_range}**  
{norm_interp}
        """)

       