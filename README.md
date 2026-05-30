# 📊 Electricity Demand Forecasting in Brazil (Machine Learning Project with Multi-Layer Perceptron Neural Networks)

## 🧠 Project Overview

This project focuses on forecasting hourly electricity demand in **Brazil** using Machine Learning techniques.

The model was originally developed using data from the **NORTE region of Brazil**, which served as the baseline experimental dataset for initial analysis, preprocessing, and model development. The current Streamlit application extends this approach by enabling interactive multi-region forecasting across Brazil.

To extend usability beyond a fixed regional scope, an interactive **Streamlit web application** was developed, allowing users to explore, train, and evaluate the model dynamically across different regions of Brazil.

---

## 🎯 Objective

The objective of this project is to model and forecast short-term electricity demand in **Brazil**, using historical consumption data to understand regional load behavior and improve predictive accuracy in time series forecasting.

This is formulated as a **supervised time series regression problem** using a neural network-based architecture.

---

## 📦 Dataset

The dataset is sourced from Hugging Face:

* 📊 Source: `SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset`
* ⏱️ Frequency: Hourly electricity demand data across Brazil
* 🌍 Features: Region (Brazil), timestamp, electricity load (MWmed)

### 📌 Modeling Scope

* The **original model training and evaluation** were performed using the **NORTE region of Brazil** as the baseline dataset
* The Streamlit application enables **interactive training and forecasting across multiple regions in Brazil**
* Model evaluation uses an **80% training set and 20% test set (out-of-sample evaluation)**
* The **test split is used exclusively for performance reporting, Comparative Table (Test), and metric computation**

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Filtering dataset by selected region
* Timestamp conversion and temporal ordering
* Feature selection (electric demand in MWmed)
* Scaling using `MinMaxScaler`

### 2. Time Series Transformation

* Sliding window approach (24-hour lag structure)
* Each input sequence of 24 hours predicts the next hourly demand value

### 3. Model Architecture

* Neural Network: `MLPRegressor`
* Hidden layers: (100, 50)
* Activation function: ReLU
* Optimizer: Adam
* Max iterations: 500

### 4. Train/Test Strategy

* 80% training set
* 20% test set (strictly out-of-sample)
* Test set is used for:

  * RMSE calculation
  * Error analysis
  * Interactive comparative table

---

## 📊 Experimental Results (Example: NORTE Region)

Diagnostic logs from the out-of-sample **Test Set (20% split of selected region)** yield:

* **0.0172 (1.72%) Normalized Error Standard Deviation:**
  Indicates very low relative dispersion of prediction errors.

* **182.13 MWmed RMSE (Root Mean Squared Error):**
  Shows average deviation between predicted and actual demand in real units.

* **136.24 MWmed Error Standard Deviation:**
  Indicates stable and consistent prediction errors without high volatility.

---

## 📊 Visualizations & Forecasting

### 📈 Test Performance (Actual vs Predicted)

The model evaluates performance exclusively on the **20% test set**, ensuring out-of-sample validation and avoiding training leakage.

### 🔮 48-Hour Forecast

The model generates recursive future predictions based on the latest observed window, producing short-term demand forecasts per selected region.

---

## 🌐 Streamlit Web Application

An interactive Streamlit application was developed to operationalize the model beyond static execution.

### Key Features:

* 📥 Dynamic dataset loading (Brazil electricity demand)
* 🌍 Region selector for multi-region analysis
* 📊 Interactive dataset exploration (Excel-like table)
* 📋 Interactive **test set (20%) comparative table**
* 🔎 Filtering, sorting, and search capabilities in both datasets
* 🤖 Model training on selected region
* 📈 Real-time performance metrics (RMSE, Std Error, Normalized Error)
* 📉 Actual vs predicted comparison (test set only)
* 🔮 48-hour forecasting module

### Key Design Principle:

* The **dataset viewer table** is independent and used for exploration only
* The **Comparative Table (Test)** is strictly based on the **hold-out 20% test set**
* Metrics and evaluation are computed only on **out-of-sample data**

---

## 🚀 How to Run the Project

### ▶️ 1. Run Streamlit App

```bash
streamlit run app.py
```

To stop the app:

```
Ctrl + C
```

---

### ▶️ 2. Run Original Script (NORTE Baseline Version)

```bash
python main.py
```

This version runs a fixed pipeline using the **NORTE region of Brazil** and outputs results in the terminal.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit multi-region interactive app
├── main.py                 # Original NORTE-only baseline model
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
└── images/                 # Optional visual outputs
```

---

## 💡 Key Learnings

* Time series forecasting using neural networks (MLPRegressor)
* Electricity demand modeling in Brazil (regional behavior analysis)
* Importance of train/test separation in time series problems
* Feature engineering using sliding window approaches
* Deployment of ML systems using Streamlit
* Interactive evaluation improves model interpretability
* Separation between exploration data and evaluation data is critical

---

## 🌱 Future Improvements

* Implementation of LSTM / GRU models
* Integration of external variables (weather, holidays)
* Hyperparameter optimization (GridSearch / Optuna)
* Cloud deployment of Streamlit app
* Advanced anomaly detection for demand spikes
* Enhanced dashboard linking filters directly to model retraining
* Probabilistic forecasting instead of point estimates
