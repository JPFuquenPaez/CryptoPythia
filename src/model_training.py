import numpy as np
import pandas as pd
import logging
import uuid
import os
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from config import N_TIMESTEPS, EPOCHS, BATCH_SIZE, LEARNING_RATE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
# Set random seeds for reproducibility
np.random.seed(69)
tf.random.set_seed(69)
random.seed(69)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_training_dataset(dataset: np.ndarray, n_timesteps: int) -> (np.ndarray, np.ndarray):
    """
    Create a dataset for training and testing.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - n_timesteps):
        x = dataset[i:i + n_timesteps, :]
        y = dataset[i + n_timesteps, 0]
        dataX.append(x)
        dataY.append(y)

    logging.info("Training dataset created successfully!")
    return np.array(dataX), np.array(dataY)

def train_model(data_final: pd.DataFrame, n_timesteps: int = N_TIMESTEPS, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, learning_rate: float = LEARNING_RATE) -> (RobustScaler, tf.keras.Model):
    """
    Train the LSTM model.
    """
    selected_features = [
        'close', 'volume', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower',
        'Year_sin', 'Year_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos', 'Hour_sin', 'Hour_cos',
        'low_freq_energy', 'high_freq_energy', 'hourly_freq_energy', 'mean_amplitude', 'variance_amplitude',
        'skewness_amplitude', 'kurtosis_amplitude', 'rolling_mean_close_6', 'rolling_mean_close_12',
        'rolling_mean_close_24', 'rolling_mean_close_48', 'rolling_mean_close_72', 'EMA_6', 'EMA_12',
        'EMA_24', 'EMA_48', 'EMA_72'
    ]

    data_final = data_final[selected_features]
    df_train, df_test = train_test_split(data_final, test_size=0.2, shuffle=False)

    scaler = RobustScaler()
    df_train_scaled = scaler.fit_transform(df_train)
    df_test_scaled = scaler.transform(df_test)

    X_train, y_train = create_training_dataset(df_train_scaled, n_timesteps)
    X_test, y_test = create_training_dataset(df_test_scaled, n_timesteps)

    logging.info(f"n_timesteps: {n_timesteps}")
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"y_test shape: {y_test.shape}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        lstm_model = Sequential([
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=(n_timesteps, X_train.shape[2]), kernel_regularizer=l2(0.01)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Bidirectional(LSTM(100, return_sequences=True, activation='sigmoid')),
            Dropout(0.2),
            Bidirectional(LSTM(50, activation='sigmoid')),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = "mean_squared_error"

        mlflow.log_param("n_timesteps", n_timesteps)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("loss", loss)

        lstm_model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
        lr_plateau = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.8, verbose=2, mode='min')

        history = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_plateau, early_stopping], validation_split=0.2, validation_data=(X_test, y_test))

        y_pred = lstm_model.predict(X_test)

        logging.info(f"Shape of y_pred: {y_pred.shape}")
        logging.info(f"Shape of y_test: {y_test.shape}")

        scaler_name = f"lstm_scaler_{uuid.uuid4()}"
        model_name = f"lstm_model_{uuid.uuid4()}"

        scaler_path = os.path.join(os.getcwd(), scaler_name)
        joblib.dump(scaler, scaler_path)

        model_path = os.path.join(os.getcwd(), model_name)
        lstm_model.save(model_path)

        mlflow.sklearn.log_model(scaler, scaler_name, registered_model_name=scaler_name)
        mlflow.tensorflow.log_model(lstm_model, model_name, registered_model_name=model_name)

        scaler_uri = mlflow.get_artifact_uri(scaler_name)
        scaler_result = mlflow.register_model(scaler_uri, scaler_name)

        model_uri = mlflow.get_artifact_uri(model_name)
        result = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=model_name, version=result.version, stage="Staging")
        client.transition_model_version_stage(name=scaler_name, version=scaler_result.version, stage="Staging")

        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss by Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        loss_plot_path = os.path.join(os.getcwd(), f"loss_plot_{uuid.uuid4()}.png")
        plt.savefig(loss_plot_path)
        mlflow.log_artifact(loss_plot_path)
        plt.show()

        mse = tf.keras.losses.MeanSquaredError()
        mse_value = mse(y_test, y_pred).numpy()
        r2_value = r2_score(y_test, y_pred)

        mlflow.log_metric("mse", mse_value)
        mlflow.log_metric("r2_score", r2_value)

        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model MSE by epoch')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

        mse_plot_path = os.path.join(os.getcwd(), f"mse_plot_{uuid.uuid4()}.png")
        plt.savefig(mse_plot_path)
        mlflow.log_artifact(mse_plot_path)
        plt.show()

        logging.info("Training process completed successfully.")
        logging.info(f"MSE: {mse_value}, R2 Score: {r2_value}")
        return scaler, lstm_model
