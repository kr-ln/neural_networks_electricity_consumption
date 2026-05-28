import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
import warnings
import os

# Limpiamos advertencias y la consola
warnings.filterwarnings("ignore")
os.system('cls' if os.name == 'nt' else 'clear')

print("📌 Paso 1: Cargando el dataset de consumo eléctrico desde Hugging Face...")
dataset = load_dataset("SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset", split="train")
df = pd.DataFrame(dataset)
print("✅ Dataset cargado correctamente. Primeras filas:")
print(df.head())
print("\nLas columnas son:", df.columns)
print("Esto nos muestra la información que tenemos: fecha, región y consumo eléctrico.")

# Filtramos la región "NORTE"
print("\n📌 Paso 2: Filtrando datos de la región NORTE...")
df = df[df['nom_subsistema'] == 'NORTE']
print(f"✅ Filtradas las filas de NORTE. Total de registros: {len(df)}")

# Convertimos la columna de fecha/hora y ordenamos
print("\n📌 Paso 3: Ordenando los datos por fecha y hora...")
df['timestamp'] = pd.to_datetime(df['din_instante'])
df = df.sort_values('timestamp')
print("✅ Datos ordenados. Primeras fechas:")
print(df['timestamp'].head())

# Seleccionamos solo la columna de demanda eléctrica
print("\n📌 Paso 4: Seleccionando la columna de demanda eléctrica y renombrando...")
data = df[['timestamp','val_cargaenergiahomwmed']].copy()
data.rename(columns={'val_cargaenergiahomwmed':'demanda'}, inplace=True)
print("✅ Columna de demanda lista. Primeros valores:")
print(data.head())

# Normalizamos la demanda
print("\n📌 Paso 5: Normalizando los datos de demanda...")
scaler = MinMaxScaler()
data['demanda_norm'] = scaler.fit_transform(data[['demanda']])
print("✅ Datos normalizados entre 0 y 1. Esto ayuda a que la red aprenda más rápido y mejor.")

# Creamos ventanas de 24 horas para predecir la siguiente hora
print("\n📌 Paso 6: Creando ventanas de 24 horas para entrenar la red...")
time_step = 24
X, y, X_dates = [], [], []
for i in range(len(data) - time_step):
    X.append(data['demanda_norm'].values[i:i+time_step])
    y.append(data['demanda_norm'].values[i+time_step])
    X_dates.append(data['timestamp'].iloc[i+time_step])
X = np.array(X)
y = np.array(y)
X_dates = np.array(X_dates)
print(f"✅ Se crearon {len(X)} ventanas de 24 horas cada una. Cada ventana ayudará a predecir la siguiente hora.")

# Dividimos en entrenamiento y prueba
print("\n📌 Paso 7: Dividiendo los datos en entrenamiento (80%) y prueba (20%)...")
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
dates_test = X_dates[train_size:]
print(f"✅ Tamaño de entrenamiento: {len(X_train)} | Tamaño de prueba: {len(X_test)}")

# Entrenamos la red neuronal MLP
print("\n📌 Paso 8: Entrenando la red neuronal MLP...")
model = MLPRegressor(hidden_layer_sizes=(100,50), activation='relu',
                     solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)
print("✅ Entrenamiento completado. La red ahora puede aprender patrones de consumo horario.")

# Predicción sobre el conjunto de prueba
print("\n📌 Paso 9: Realizando predicciones sobre el conjunto de prueba...")
pred_test = model.predict(X_test)
pred_test_actual = scaler.inverse_transform(pred_test.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
rmse = np.sqrt(mean_squared_error(y_test_actual, pred_test_actual))
print(f"📊 RMSE en test: {rmse:.2f}")
print("➡️ Esto nos dice, en promedio, cuánto se desvía la predicción de los valores reales.")

# Calculamos la desviación estándar de los errores
errors = y_test_actual.flatten() - pred_test_actual.flatten()
std_dev = np.std(errors)
print(f"📊 Desviación estándar de los errores: {std_dev:.2f}")
print("➡️ Esto nos da una idea de cuán dispersos están los errores de predicción. Cuanto menor sea la desviación estándar, más consistentes serán las predicciones.")

# Normalizamos la desviación estándar
mean_real = np.mean(y_test_actual)  # Calculamos la media de los valores reales
normalized_std_dev = std_dev / mean_real  # Normalizamos la desviación estándar
print(f"📊 Desviación estándar normalizada: {normalized_std_dev:.4f}")
print("➡️ Esto nos indica qué tan grande es el error relativo comparado con el valor promedio de la demanda.")

# Graficamos Predicción vs Real con fechas reales
print("\n📌 Paso 10: Graficando Predicción vs Real (primeras 200 horas)...")
plt.figure(figsize=(12,4))
plt.plot(dates_test[:200], y_test_actual[:200], label="Real")
plt.plot(dates_test[:200], pred_test_actual[:200], label="Predicción")
plt.title("Predicción vs Real (Test) – Fechas reales")
plt.xlabel("Fecha y hora")
plt.ylabel("Demanda eléctrica")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
print("✅ Gráfica completada. Se puede ver cómo la red sigue la estacionalidad y los picos horarios.")

# Predicción futura de 48 horas
print("\n📌 Paso 11: Prediciendo las próximas 48 horas de demanda eléctrica...")
future_steps = 48
future_predictions = []
last_window = X[-1]
last_timestamp = data['timestamp'].iloc[-1]
future_dates = []

for i in range(future_steps):
    pred_next = model.predict(last_window.reshape(1,-1))[0]
    future_predictions.append(pred_next)
    last_window = np.roll(last_window, -1)
    last_window[-1] = pred_next
    future_dates.append(last_timestamp + pd.Timedelta(hours=i+1))

future_actual = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

# Imprimir las predicciones futuras en consola
print("\n🔮 Predicciones de las próximas 48 horas (valores numéricos):")
for dt, val in zip(future_dates, future_actual.flatten()):
    print(f"{dt}: {val:.2f} kWh")

# Graficamos Predicción futura con etiquetas sobre cada punto
print("\n📌 Paso 12: Graficando predicción futura de 48 horas con etiquetas...")
plt.figure(figsize=(12,4))
plt.plot(future_dates, future_actual, label="Demanda futura", marker='o')

# Añadir etiquetas sobre cada punto
for dt, val in zip(future_dates, future_actual.flatten()):
    plt.text(dt, val + 100, f"{val:.0f}", ha='center', va='bottom', fontsize=8, rotation=90)
plt.title("Predicción demanda eléctrica próximas 48 horas (fechas reales con etiquetas)")
plt.xlabel("Fecha y hora")
plt.ylabel("Demanda eléctrica")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
print("✅ Gráfica completada. Esto muestra cómo la red espera que varíe la demanda y los valores exactos hora por hora.")

print("\n🔮 Predicciones futuras completadas y todas las explicaciones incluidas.")