import os
import datetime

# Directory for data storage
DATA_DIR = 'data'

# Binance API parameters
SYMBOL = 'BTCUSDC'
INTERVAL = '1h'
START_TIME = int(datetime.datetime(2023, 1, 1).timestamp() * 1000)
END_TIME = int(datetime.datetime.now().timestamp() * 1000)

# Model parameters
N_TIMESTEPS = 10
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# MLflow parameters
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Change this to your MLflow tracking server URI
MLFLOW_EXPERIMENT_NAME = "cryptocurrency_price_prediction"
