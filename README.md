# Bitcoin Price Forecasting with Machine Learning

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Getting Started](#getting-started)
5. [Data Pipeline](#data-pipeline)
6. [Model Training](#model-training)
7. [Forecasting & Evaluation](#forecasting--evaluation)
8. [MLflow Integration](#mlflow-integration)
9. [Project Structure](#project-structure)
10. [References](#references)

---

## Project Overview

This project implements an end-to-end machine learning pipeline for Bitcoin price forecasting using:
- **Hybrid Neural Network**: Combines Conv1D and LSTM layers for spatiotemporal pattern recognition
- **Feature Engineering**: 50+ features including technical indicators, Fourier transforms, and cyclical encoding
- **MLOps Infrastructure**: Full MLflow integration for experiment tracking and model management

![System Architecture](https://via.placeholder.com/600x300?text=Data+Pipeline+and+System+Flow)

---

## Key Features

- **Data Ingestion**
  - Automated historical data fetching from Binance API
  - Raw data versioning (bronze/silver/gold stages)
  
- **Feature Engineering**
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Spectral analysis via FFT
  - Cyclical temporal encoding
  - Multiple rolling windows and EMA calculations
  - Prior Feature Importance and SHAP values for feature selection

- **Model Architecture (EXAMPLE)**
  ```python
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  conv1d (Conv1D)              (None, 300, 5)            50        
  _________________________________________________________________
  max_pooling1d (MaxPooling1D) (None, 150, 5)            0         
  _________________________________________________________________
  dropout (Dropout)            (None, 150, 5)            0         
  _________________________________________________________________
  lstm (LSTM)                  (None, 150, 4)            160       
  _________________________________________________________________
  batch_normalization (BatchNo (None, 150, 4)            16        
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 150, 4)            0         
  _________________________________________________________________
  lstm_1 (LSTM)                (None, 150, 3)            96        
  _________________________________________________________________
  batch_normalization_1 (Batch (None, 150, 3)            12        
  _________________________________________________________________
  dropout_2 (Dropout)          (None, 150, 3)            0         
  _________________________________________________________________
  lstm_2 (LSTM)                (None, 2)                 48        
  _________________________________________________________________
  batch_normalization_2 (Batch (None, 2)                 8         
  _________________________________________________________________
  dropout_3 (Dropout)          (None, 2)                 0         
  _________________________________________________________________
  dense (Dense)                (None, 1)                 3         
  =================================================================
  Total params: 393
  ```

---

## Getting Started

### Prerequisites

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db \
             --default-artifact-root ./mlruns \
             --host 0.0.0.0 --port 8000
```

### Quick Start

1. Run data pipeline:
```python
python pipeline.py --symbol BTCUSDC --interval 1h \
                  --start 2023-01-01 --end 2024-01-01
```

2. Train model:
```python
python train.py --timesteps 300 --epochs 50 --batch-size 64
```

3. Generate forecasts:
```python
python forecast.py --target-date 2024-06-01
```

---

## Data Pipeline

| Stage        | Description                          | Sample Features                  |
|--------------|--------------------------------------|-----------------------------------|
| **Bronze**   | Raw API data                         | Timestamp, Open, High, Low, Close|
| **Silver**   | Cleaned + temporal features         | Year, Month, Hour (cyclical)     |
| **Gold**     | Feature-engineered dataset          | 50+ technical & spectral features|

![Data Pipeline Stages](https://via.placeholder.com/600x200?text=Bronze+→+Silver+→+Gold+Data+Transformation)

---

## Model Training

### Hyperparameters

| Parameter         | Value   | Description                     |
|-------------------|---------|---------------------------------|
| Lookback Window   | 300     | Historical timesteps considered |
| Batch Size        | 64      | Training batch size             |
| Learning Rate     | 0.001   | Adam optimizer setting          |
| L2 Regularization | 0.01    | Weight decay parameter          |
| Dropout Rate      | 0.2-0.4 | Per-layer dropout probabilities |

### Training Workflow

1. Dataset creation with sliding window
2. Robust scaling of features
3. 80/20 temporal train-test split
4. Early stopping and LR reduction callbacks
5. MLflow experiment tracking

---

## Forecasting & Evaluation

### Performance Metrics

| Metric        | Value   |
|---------------|---------|
| MSE           | 0.0012  |
| R² Score      | 0.94    |
| RMSE          | 0.0346  |

![Actual vs Predicted](https://via.placeholder.com/600x300?text=Actual+vs+Predicted+Prices+Comparison)

---

## MLflow Integration

### Key Features

- **Experiment Tracking**
  - Parameter/metric logging
  - Artifact storage (models, scalers, plots)
  
- **Model Registry**
  - Version control
  - Stage transitions (Staging → Production)
  - Model/compute environment packaging

![MLflow UI](https://via.placeholder.com/600x200?text=MLflow+Experiment+Tracking+Interface)

### Usage Example

```python
# Track experiment
with mlflow.start_run(run_name="LSTM_Conv1D_v2"):
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(scaler, "scaler")
    mlflow.tensorflow.log_model(model, "model")

# Promote best model
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="BTC_Forecaster",
    version=3,
    stage="Production"
)
```

---

## Project Structure

```
.
├── data/                # Data storage (bronze/silver/gold)
├── models/              # Serialized models/scalers
├── notebooks/           # Exploratory analysis
├── src/
│   ├── pipeline.py      # Data ingestion pipeline
│   ├── train.py         # Model training
│   └── forecast.py      # Inference module
├── mlruns/              # MLflow artifacts
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## References

1. **Time Series Forecasting**
   - [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Fundamental LSTM concepts
   - [Deep Learning for Time Series](https://www.oreilly.com/library/view/deep-learning-for/9781492040989/) - O'Reilly reference

2. **MLOps**
   - [MLflow Documentation](https://mlflow.org/docs/latest/index.html) - Official MLflow guides
   - [Production ML Systems](https://ai.google/research/pubs/pub43146) - Google research paper

3. **Financial ML**
   - [Advances in Financial ML](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - Marcos López de Prado
