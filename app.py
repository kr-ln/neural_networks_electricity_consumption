import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from datasets import load_dataset

st.set_page_config(page_title="Electricity Forecast App", layout="wide")

st.title("📊 Electricity Demand Prediction with Machine Learning")
st.write("MLPRegressor model for hourly electricity consumption forecasting in Brazil")

# =========================
# CARGA DE DATOS
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if st.button("📥 Load dataset"):
    dataset = load_dataset(
        "SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset",
        split="train"
    )
    df = pd.DataFrame(dataset)
    st.session_state.df = df
    st.success("Dataset loaded correctly")

# =========================
# 🟦 INTERACTIVE TABLE (ONLY VISUAL / EXPLORATION)
# =========================

# =========================
# FILTRO
# =========================
if st.session_state.df is not None:
    df = st.session_state.df

    st.subheader("📋 Interactive Dataset (Excel-like)")

    # Configurar tabla interactiva
    gb = GridOptionsBuilder.from_dataframe(df)  # df = datos originales
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    grid_options = gb.build()

    # Mostrar tabla y capturar filtrado
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED
    )

    # Obtener el dataframe después de filtrar/ordenar
    filtered_df = grid_response["data"]

    # Mostrar cantidad de registros después de los filtros
    st.write(f"📊 Total records after filtering: {len(filtered_df)}")

    st.markdown("---")

    region = st.selectbox("Select region", df["nom_subsistema"].unique())

    df_region = df[df["nom_subsistema"] == region]

    st.write(f"Records in {region}: {len(df)}")

    # =========================
    # PREPROCESAMIENTO
    # =========================
    if st.button("🤖 Train model"):

        df_region = df_region.copy()
        df_region["timestamp"] = pd.to_datetime(df_region["din_instante"])
        df_region = df_region.sort_values("timestamp")

        data = df_region[["timestamp", "val_cargaenergiahomwmed"]].copy()
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
        st.subheader("📋 Comparative Table (Test) - Interactive (20% test set)")

        comp_df = pd.DataFrame({
            "timestamp": dates_test,
            "actual": y_test_actual.flatten(),
            "prediction": pred_test_actual.flatten()
        })

        comp_df["% variation"] = ((comp_df["prediction"] - comp_df["actual"]) / comp_df["actual"]) * 100

        # Configurar AgGrid interactiva
        gb = GridOptionsBuilder.from_dataframe(comp_df)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        grid_options = gb.build()

        grid_response = AgGrid(
            comp_df,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED
        )

        # Obtener dataframe filtrado/ordenado por el usuario
        comp_df_filtered = grid_response["data"]

        # Mostrar total de registros después de los filtros
        st.write(f"📊 Total records after filtering: {len(comp_df_filtered)}")

        # =========================
        # MÉTRICAS
        # =========================
        st.subheader("📊 Model Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("Std Error", f"{std_dev:.2f}")
        col3.metric("Normalized Error", f"{normalized_std_dev:.4f}")

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

        st.subheader("🧠 Results Interpretation")
        st.write(f"""
📊 **System Context ({region} / Brazil):**
- Mean demand: {mean_demand:.2f} MWmed  
- Min demand: {min_demand:.2f} MWmed  
- Max demand: {max_demand:.2f} MWmed  
- Demand range: {demand_range:.2f} MWmed  

---

### 📉 RMSE Interpretation
- Value: **{rmse:.2f} MWmed**
- Relative error: **{rmse_value_pct:.2f}%**
- Benchmark ranges:
  - <5% → Excellent  
  - 5–10% → Good  
  - 10–20% → Moderate  
  - \>20% → High error  

👉 This model falls in: **{rmse_range}**  
{rmse_interp}

---

### 📊 Error Dispersion (Std Error)
- Value: **{std_dev:.2f} MWmed**
- Relative error: **{std_value_pct:.2f}%**
- Benchmark ranges:
  - <5% → Very stable  
  - 5–10% → Stable  
  - 10–20% → Moderate variability  
  - \>20% → High volatility  

👉 This model falls in: **{std_range}**  
{std_interp}

---

### 📌 Normalized Error
- Value: **{normalized_std_dev:.4f}**
- Percentage: **{norm_value_pct:.2f}%**
- Benchmark ranges:
  - <5% → Excellent  
  - 5–10% → Good  
  - 10–20% → Acceptable  
  - \>20% → High error  

👉 This model falls in: **{norm_range}**  
{norm_interp}

---
        """)

        # =========================
        # GRÁFICA TEST
        # =========================
        st.subheader("📈 Prediction vs Actual (Test)")

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            dates_test[:200],
            y_test_actual[:200],
            label="Actual"
        )

        ax.plot(
            dates_test[:200],
            pred_test_actual[:200],
            label="Prediction"
        )

        ax.set_title(f"Prediction vs Actual - Region {region}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Electric Demand (MWmed)")
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        # =========================
        # FORECAST FUTURO
        # =========================
        st.subheader("🔮 Future Prediction (48 hours)")

        future_steps = 48
        future_predictions = []

        last_window = X[-1]
        last_timestamp = data["timestamp"].iloc[-1]

        future_dates = []

        for i in range(future_steps):

            pred_next = model.predict(
                last_window.reshape(1, -1)
            )[0]

            future_predictions.append(pred_next)

            last_window = np.roll(last_window, -1)
            last_window[-1] = pred_next

            future_dates.append(
                last_timestamp + pd.Timedelta(hours=i+1)
            )

        future_actual = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )

        future_df = pd.DataFrame({
            "timestamp": future_dates,
            "prediction": future_actual.flatten()
        })

        st.write("#### Future Predictions Table")
        st.dataframe(
            future_df.style.format({
                "prediction": "{:.2f}"
            })
        )

        fig2, ax2 = plt.subplots(figsize=(12, 4))

        ax2.plot(
            future_dates,
            future_actual,
            marker="o",
            label="Forecast"
        )

        ax2.set_title(
            f"Future Prediction - Next 48 Hours ({region})"
        )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Electric Demand (MWmed)")
        ax2.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig2)

        st.success(
            "Model trained and predictions generated correctly."
        )