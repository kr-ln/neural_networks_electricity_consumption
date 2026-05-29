# 📊 Electricity Demand Forecasting in Brazil (Machine Learning Project)

## 🧠 Project Overview

This project focuses on forecasting hourly electricity demand in **Brazil** using Machine Learning techniques.

The model was originally developed and trained using data from the **NORTH (NORTE) region of Brazil**, which serves as the base experimental dataset for analysis, preprocessing, model development, and evaluation of predictive performance.

To extend usability beyond a fixed regional scope, an interactive **Streamlit web application** was developed, allowing users to explore and train the model dynamically across different regions of Brazil.

---

## 🎯 Objective

The objective of this project is to model and forecast short-term electricity demand in **Brazil**, using historical consumption data to understand regional load behavior and improve predictive accuracy in time series forecasting.

This is formulated as a **supervised time series regression problem** using a neural network-based architecture.

---

## 📦 Dataset

The dataset is sourced from Hugging Face:

- 📊 Source: `SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset`
- ⏱️ Frequency: Hourly electricity demand data across Brazil
- 🌍 Features: Region (Brazil), timestamp, electricity load (MWmed)

### 📌 Modeling Scope

- The **original model training, validation, and testing** were performed exclusively on the **NORTE region of Brazil**
- This region serves as the **baseline experimental environment**
- The Streamlit application enables **multi-region exploration across Brazil**

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Filtering dataset by region (initial focus: NORTE, Brazil)
- Timestamp conversion and temporal ordering
- Feature selection (electric demand in MWmed)
- Scaling using `MinMaxScaler`

### 2. Time Series Transformation
- Sliding window approach (24-hour lag structure)
- Each input sequence of 24 hours predicts the next hourly demand value

### 3. Model Architecture
- Neural Network: `MLPRegressor`
- Hidden layers: (100, 50)
- Activation function: ReLU
- Optimizer: Adam
- Max iterations: 500

### 4. Evaluation Metrics
- RMSE (Root Mean Squared Error)
- Standard deviation of residual errors
- Normalized error standard deviation

---

## 📊 Experimental Results & Operational Interpretation (NORTE Region)

Diagnostic logs from the out-of-sample **Testing Set (NORTE region of Brazil)** yield the following reliability metrics:

* **0.0172 (1.72%) Normalized Error Standard Deviation:**  
  This represents the baseline accuracy of the model in the NORTE region. A relative error under 2% is considered an elite benchmark in power systems forecasting, where industry tolerance typically sits below 5%.

* **182.13 MWmed RMSE (Root Mean Squared Error):**  
  Since RMSE heavily penalizes large deviations, this low value demonstrates that the model successfully tracks peak demand behavior in the NORTE region without significant desynchronization or curve flattening.

* **136.24 MWmed Error Standard Deviation:**  
  This confirms that residual errors are tightly clustered around zero, indicating high stability and the absence of systemic drift or extreme outliers in the NORTE region forecasts.

---

## 📊 Visualizations & Projections (NORTE Region)

### 📈 Historical Test Performance (Actual vs Predicted - NORTE)
The chart below illustrates the model's high fidelity in capturing daily demand dynamics and peak load behavior in the **NORTE region of Brazil**.

![Test Forecast](images/prediction_vs_real.png)

---

### 🔮 48-Hour Out-of-Sample Forecast (NORTE)
The recursive forecasting mechanism generates stable short-term predictions for the **NORTE region of Brazil**.

![Future Forecast](images/prediction_next_48_hours.png)

---

## 🌐 Streamlit Web Application

An interactive Streamlit application was developed to operationalize the model beyond static execution.

### Key Features:
- 📥 Dynamic loading of electricity demand dataset (Brazil)
- 🌍 Region selector (multi-region analysis across Brazil)
- 🤖 Interactive model training
- 📊 Real-time performance metrics
- 📈 Visual comparison (actual vs predicted)
- 🔮 48-hour forecasting module

### Key Design Difference:

- **Original system:** fixed analytical pipeline on the **NORTE region of Brazil**
- **Streamlit system:** generalized framework for **multi-region exploration across Brazil**

---

## 🚀 How to Run the Project

### ▶️ 1. Run Streamlit App

```python
streamlit run app.py
```

📌 To stop the Streamlit application while running in the terminal, press:

```
Ctrl + C
```

This will immediately terminate the local server.

### ▶️ 2. Run Original Script (Optional / Research Version)

If you want to execute the original Python script (non-interactive version):

```python
main.py
```

This version runs the full pipeline using the NORTE region of Brazil as the fixed dataset and prints results directly in the terminal.

#### 📁 Project Structure

```
├── app.py                  # Streamlit application (multi-region Brazil)

├── original_script.py     # Original NORTE region model

├── README.md               # Project documentation

├── requirements.txt        # Dependencies

└── images/                 # Visual outputs (NORTE region results)
```

#### 💡 Key Learnings

Time series forecasting using neural networks

Electricity demand modeling in Brazil (NORTE baseline analysis)

Importance of region-specific training for load forecasting

Feature engineering for sequential data

Deployment of ML systems using Streamlit

Translation of research-grade code into interactive applications

#### 🌱 Future Improvements

- Implementation of LSTM / GRU architectures

- Integration of exogenous variables (weather, holidays in Brazil)

- Hyperparameter optimization

- Cloud deployment of Streamlit application

- Anomaly detection for demand spikes
