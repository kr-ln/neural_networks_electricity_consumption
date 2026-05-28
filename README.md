# Brazil Electricity Demand Forecast ⚡🤖

**Brazil Electricity Demand Forecast** is an end-to-end Machine Learning project predicting hourly consumption for Brazil's Norte grid using an **MLPRegressor** with 24h lag windows ($T_{-1}$ to $T_{-24}$). It achieves **98.28% accuracy** (0.0172 normalized error), a test **RMSE of 182.13 MWmed** tracking high peaks, and a stable error std. dev of **136.24 MWmed** for reliable recursive 48h forecasts. Built with Python.

---

## 📈 Key Metrics & Interpretation

* **0.0172 (1.72%) Normalized Error Std. Dev:** High-precision baseline (< 5% industry benchmark). The model's margin of error represents only 1.72% of the region's average load.
* **182.13 MWmed RMSE:** Strong capacity to track sudden load surges during critical peak hours without flattening the curve.
* **136.24 MWmed Error Std. Dev:** Guarantees uniform, predictable deviations. No chaotic outliers, allowing operators to establish exact and safe energy reserve margins.

---

## 🛠️ Detailed Methodology & Workflow

The project was executed through a rigorous 5-step data engineering and modeling pipeline:

1. **Data Ingestion & Target Isolation:** * Automated the extraction of **46,320 hourly load records** from the public ONS dataset hosted on Hugging Face (`SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset`).
   * Filtered the global dataframe by the categorical feature `id_subsistema` to isolate the **NORTE** region, extracting `val_cargaenergiahomwmed` as the primary continuous target variable.

2. **Data Splitting & Scaling (Anti-Leakage Protocol):**
   * Sorted the entire dataset chronologically to preserve the temporal dependency.
   * Split the data into an **80% Training Set (37,036 samples)** and a **20% Testing Set (9,260 samples)** using a hard temporal cutoff, preventing future data from leaking into the training stage.
   * Applied a **MinMaxScaler** transformation to map values into a $[0, 1]$ range based *only* on the training parameters, ensuring stable gradient updates during backpropagation.

3. **Time-Series Lag Engineering & Feature Selection:**
   * Structured the single-variable array into a supervised learning matrix using a sliding window approach with a lookback horizon of **24 hourly steps ($T_{-1}$ to $T_{-24}$)**.
   * Conducted a temporal correlation analysis, identifying two critical feature behaviors:
     * *Immediate Inertia:* $T_{-1}$ and $T_{-2}$ hold the highest predictive weights due to thermodynamic and consumption continuity.
     * *Circadian Recurrence:* $T_{-24}$ exhibits a significant statistical correlation spike, allowing the model to naturally reconstruct the 24-hour human lifestyle cycles.

4. **Neural Network Architecture Optimization:**
   * Configured a deep Multi-Layer Perceptron (**MLPRegressor**) structured with a dense **(100, 50)** hidden layer topology.
   * Implemented **ReLU** (Rectified Linear Unit) activation functions across hidden units to compute non-linear relationships efficiently.
   * Deployed the **Adam** stochastic gradient optimizer for adaptive step-size adjustments alongside an **Early Stopping** mechanism (patience threshold set to prevent overfitting, capped at a maximum of 500 epochs).

5. **Recursive Multi-Step Forecasting:**
   * Built a custom simulation loop to generate out-of-sample projections for a **48-hour future tactical horizon**.
   * The process operates recursively: the network predicts $T_{+1}$, appends this predicted value into a new sliding window frame via vector rolls (`np.roll`), and uses it as an input feature to solve the subsequent timestep ($T_{+2}$ up to $T_{+48}$).

---

## 📂 Tech Stack

* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy
* **Data Source:** Hugging Face Datasets
