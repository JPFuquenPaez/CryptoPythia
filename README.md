# Bitcoin Price Forecasting with Machine Learning

## Table of Contents

1. [Introduction](#introduction)
2. [System Design](#system-design)
3. [Data Fetching](#data-fetching)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [MLflow Integration](#mlflow-integration)
7. [Model Training](#model-training)
8. [Forecasting](#forecasting)
9. [Evaluation Metrics](#evaluation-metrics)
10. [References](#references)

## Introduction

This project focuses on time series forecasting using a combination of Long Short-Term Memory (LSTM) and Convolutional 1D (Conv1D) layers. Time series forecasting is a crucial task in various domains such as finance, weather prediction, and demand forecasting. The goal is to predict future values based on historical data.

## System Design

### Architecture Overview

The system is designed to fetch, preprocess, and engineer features from historical Bitcoin price data. It then trains a deep learning model using LSTM and Conv1D layers to forecast future prices. The model is registered and versioned using MLflow for easy deployment and monitoring.

### Components

1. **Data Fetching**: Fetches historical Bitcoin price data from the Binance API.
2. **Data Preprocessing**: Cleans and normalizes the data.
3. **Feature Engineering**: Calculates technical indicators, cyclical features, Fourier transforms, rolling means, and Exponential Moving Averages (EMA).
4. **MLflow Integration**: Tracks experiments, manages models, and deploys models.
5. **Model Training**: Trains an LSTM-Conv1D model on the preprocessed data.
6. **Forecasting**: Uses the trained model to make predictions on new data.
7. **Evaluation**: Evaluates the model using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).
8. **Model Management**: Registers and versions the model using MLflow.

### Data Flow

1. **Data Ingestion**: Historical data is fetched from the Binance API and saved as raw data (`bronze.csv`).
2. **Preprocessing**: The raw data is preprocessed and saved as `silver.csv`.
3. **Feature Engineering**: Features are engineered and the resulting data is saved as `gold.csv`.
4. **Model Training**: The model is trained on the engineered data and registered in MLflow.
5. **Inference**: The trained model is used to make predictions on new data.

### System Design with MLflow

1. **Experiment Tracking**: Use MLflow to track experiments, log parameters, metrics, and artifacts.
2. **Model Registration**: Register models in the MLflow Model Registry to manage model versions and stages.
3. **Model Serving**: Deploy models for inference using MLflow's serving capabilities.
4. **Artifact Storage**: Store and manage artifacts such as model files, data files, and plots.
5. **Collaboration**: Share experiments and models with team members using MLflow's collaboration features.

BEFORE EXPERIMENTING:

cd cryptopythia
source mlflowtensor/bin/activate #Activate your custom env

Step 1: Start the MLflow Server
Ensure that the MLflow server is running. You can start the MLflow server using the following command:

mlflow server --backend-store-uri file:/INSERT YOUR PATH HERE --default-artifact-root file:/ INSERT PATH HERE /mlruns/artifacts --host 0.0.0.0 --port 8000

#### 1. Experiment Tracking

**Functionality**: Track experiments to log parameters, metrics, and artifacts.

**Code**:
```python
import mlflow
import mlflow.tensorflow
import mlflow.sklearn

# Set the tracking URI
mlflow.set_tracking_uri('http://localhost:8000')

# Set the experiment name
experiment_name = "Bitcoin_Forecasting"
mlflow.set_experiment(experiment_name)

# Start a new run
with mlflow.start_run(run_name="LSTM_Conv1D_35") as run:
    # Log parameters
    mlflow.log_param("n_timesteps", n_timesteps)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    # Log metrics
    mlflow.log_metric("mse", mse_value)
    mlflow.log_metric("r2_score", r2_value)

    # Log artifacts
    mlflow.log_artifact("model_loss_by_epoch.png")
```

- **Methodology**: Use MLflow's tracking capabilities to log parameters, metrics, and artifacts. This helps in reproducing experiments and comparing results.
- **Best Practices**: Ensure that all relevant parameters, metrics, and artifacts are logged for each experiment.

#### 2. Model Registration

**Functionality**: Register models in the MLflow Model Registry to manage model versions and stages.

**Code**:
```python
# Register the scaler in the MLflow Model Registry
scaler_uri = f"runs:/{mlflow.active_run().info.run_id}/{scaler_name}"
scaler_result = mlflow.register_model(scaler_uri, scaler_name)
logging.info(f"Scaler registered: {scaler_result}")

# Register the model in the MLflow Model Registry
model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
model_result = mlflow.register_model(model_uri, model_name)
logging.info(f"Model registered: {model_result}")

# Transition the model version to "Staging"
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(name=model_name, version=model_result.version, stage="Staging")
logging.info(f"Model transitioned to Staging: {model_result.version}")

# Transition the scaler version to "Staging"
client.transition_model_version_stage(name=scaler_name, version=scaler_result.version, stage="Staging")
logging.info(f"Scaler transitioned to Staging: {scaler_result.version}")
```

- **Methodology**: Use MLflow's Model Registry to manage model versions and stages. This helps in deploying the best model version to production.
- **Best Practices**: Ensure that models are registered and transitioned through different stages (e.g., Staging, Production) based on their performance.

#### 3. Model Serving

**Functionality**: Deploy models for inference using MLflow's serving capabilities.

**Code**:
```python
# Load the best model and scaler for inference
model = mlflow.tensorflow.load_model(model_uri)
scaler = mlflow.sklearn.load_model(scaler_uri)

# Forecast future values until the specified target date
forecasted_values = forecast_future_values(model, scaler, data, n_timesteps, target_date, date_column)
```

- **Methodology**: Use MLflow to load registered models and scalers for inference. This ensures that the correct model version is used for predictions.
- **Best Practices**: Ensure that the model and scaler are correctly loaded and used for inference.

#### 4. Artifact Storage

**Functionality**: Store and manage artifacts such as model files, data files, and plots.

**Code**:
```python
# Log the plot as an artifact
mlflow.log_artifact("model_loss_by_epoch.png")
```

- **Methodology**: Use MLflow to log and manage artifacts. This helps in keeping track of important files and visualizations.
- **Best Practices**: Ensure that all relevant artifacts are logged and managed using MLflow.

#### 5. Collaboration

**Functionality**: Share experiments and models with team members using MLflow's collaboration features.

**Code**:
```python
# Share the experiment with team members
mlflow.set_experiment(experiment_name)
```

- **Methodology**: Use MLflow's collaboration features to share experiments and models with team members. This helps in collaborative development and review.
- **Best Practices**: Ensure that experiments and models are shared with relevant team members for collaborative development.

## Data Fetching

### Purpose
The first step in any machine learning project is to fetch the data. This involves collecting the raw data that will be used for training and testing the model.

### Steps
1. **Identify Data Sources**: Determine where the data will come from. This could be databases, APIs, or publicly available datasets.
2. **Download Data**: Use appropriate tools or scripts to download the data. Ensure that the data is in a format that can be easily processed (e.g., CSV, JSON).

### Example
```python
import pandas as pd

# Example: Load data from a CSV file
data = pd.read_csv('path/to/your/data.csv')
```

## Data Preprocessing

### Purpose
Data preprocessing is essential for cleaning and transforming the raw data into a format suitable for model training. This step ensures that the data is consistent, free of errors, and in the correct format.

### Steps
1. **Handle Missing Values**: Identify and handle missing values in the dataset. This can be done by imputing missing values or removing rows/columns with missing data.
2. **Normalize Data**: Scale the data to a standard range, typically between 0 and 1. This helps in improving the convergence of the model during training.
3. **Split Data**: Divide the data into training, validation, and test sets. This ensures that the model can be evaluated on unseen data.

### Example
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Handle missing values
data = data.fillna(method='ffill')

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data_scaled[:, :-1], data_scaled[:, -1], test_size=0.2, random_state=42)
```

## Feature Engineering

### Purpose
Feature engineering involves creating new features from the existing data to improve the model's performance. This step is crucial for capturing the underlying patterns in the data.

### Steps
1. **Create Lag Features**: Generate lag features to capture the temporal dependencies in the time series data.
2. **Rolling Statistics**: Calculate rolling statistics such as mean, standard deviation, and moving averages to capture trends and seasonality.
3. **Fourier Transform**: Use Fourier transform to capture periodic patterns in the data.

### Example
```python
import numpy as np

# Create lag features
def create_lag_features(data, lags):
    for lag in lags:
        data[f'lag_{lag}'] = data['value'].shift(lag)
    return data

# Calculate rolling statistics
def calculate_rolling_statistics(data, window):
    data['rolling_mean'] = data['value'].rolling(window=window).mean()
    data['rolling_std'] = data['value'].rolling(window=window).std()
    return data

# Apply feature engineering
data = create_lag_features(data, lags=[1, 2, 3])
data = calculate_rolling_statistics(data, window=7)
```

## Model Training

### Purpose
Model training involves training the neural network on the preprocessed data. This step includes defining the model architecture, compiling the model, and fitting it to the training data.

### Steps
1. **Define Model Architecture**: Use a combination of Conv1D and LSTM layers to capture both spatial and temporal patterns in the data.
2. **Compile the Model**: Specify the loss function, optimizer, and evaluation metrics.
3. **Train the Model**: Fit the model to the training data and validate its performance on the validation set.

### Example
```python
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense

# Define the neural network architecture
n_timesteps = 100  # Example value for n_timesteps
X_train_shape_2 = 10  # Example value for X_train.shape[2]

lstm_model = tf.keras.Sequential([
    Conv1D(filters=128, kernel_size=3, activation='linear', input_shape=(n_timesteps, X_train_shape_2)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=256, kernel_size=3, activation='linear'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=200, kernel_size=3, activation='linear'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=150, kernel_size=3, activation='linear'),
    MaxPooling1D(pool_size=2),
    LSTM(100, return_sequences=True),
    LSTM(50),
    Dense(30, activation='linear'),
    Dense(15, activation='linear'),
    Dense(1)
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = lstm_model.fit(X_train, y_train,
                        epochs=20,
                        batch_size=64,
                        validation_data=(X_test, y_test))
```

### Theoretical Background
- **Conv1D Layer**: The Conv1D layer is used to capture spatial patterns in the time series data. It applies convolutional filters to the input data to extract features.
- **LSTM Layer**: The LSTM layer is used to capture temporal dependencies in the data. It is a type of recurrent neural network (RNN) that can learn long-term dependencies.

## Forecasting

### Purpose
Forecasting involves using the trained model to make predictions on new, unseen data. This step is crucial for evaluating the model's performance in real-world scenarios.

### Steps
1. **Preprocess New Data**: Apply the same preprocessing steps to the new data as were applied to the training data.
2. **Make Predictions**: Use the trained model to make predictions on the new data.
3. **Post-process Predictions**: Convert the predictions back to the original scale if necessary.

### Example
```python
# Preprocess new data
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = lstm_model.predict(new_data_scaled)

# Post-process predictions
predictions = scaler.inverse_transform(predictions)
```

## Evaluation Metrics

### Purpose
Evaluation metrics are used to assess the performance of the model. They provide a quantitative measure of how well the model is performing.

### Metrics
1. **Mean Squared Error (MSE)**: Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
2. **Root Mean Squared Error (RMSE)**: The square root of the average of squared errors. It is more interpretable than MSE because it is in the same units as the target variable.
3. **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.

### Example
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
```

## References

1. **Long Short-Term Memory (LSTM)**:
   - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
   - [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069)

2. **Convolutional Neural Networks (CNN)**:
   - [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285)
   - [Deep Learning Book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)

3. **Time Series Forecasting**:
   - [Time Series Forecasting: Principles and Practice](https://otexts.com/fpp2/)
   - [Forecasting: principles and practice](https://www.otexts.org/fpp2)

4. **ML System Design**:
   - [How to Develop LSTM Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)
   - [Deep-Learning for Time Series Forecasting: LSTM and CNN Neural Network](https://medium.com/@sandha.iitr/deep-learning-for-time-series-forecasting-lstm-and-cnn-neur-4c934cb16707)
   - [Time Series Forecasting | TensorFlow Core](https://www.tensorflow.org/tutorials/structured_data/time_series)
   - [Time-series Forecasting using Conv1D-LSTM: Multiple timesteps into future](https://shivapriya-katta.medium.com/time-series-forecasting-using-conv1d-lstm-multiple-timesteps-into-future-acc684dcaaa)

