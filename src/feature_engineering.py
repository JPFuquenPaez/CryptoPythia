import pandas as pd
import numpy as np
import ta
from numpy.fft import rfft, rfftfreq
import logging
# Set random seeds for reproducibility
np.random.seed(69)
tf.random.set_seed(69)
random.seed(69)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators.
    """
    data['RSI'] = ta.momentum.rsi(data['close'], window=14)
    data['MACD'] = ta.trend.macd(data['close'])
    bollinger = ta.volatility.BollingerBands(data['close'])
    data['Bollinger_Upper'], data['Bollinger_Middle'], data['Bollinger_Lower'] = bollinger.bollinger_hband(), bollinger.bollinger_mavg(), bollinger.bollinger_lband()

    logging.info('Technical indicators calculated successfully.')
    return data

def calculate_cyclical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cyclical features.
    """
    data['Year_sin'] = np.sin(2 * np.pi * data['Year'] / 2024.0)
    data['Year_cos'] = np.cos(2 * np.pi * data['Year'] / 2024.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)
    data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31.0)
    data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31.0)
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)

    data.drop(['Hour', 'Month', 'Year', 'Day'], axis=1, inplace=True)

    logging.info('Cyclical features calculated successfully!')
    return data

def calculate_fourier_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Fourier transform and frequencies.
    """
    N = len(data['close'])
    T = 60

    yf = rfft(data['close'].values)
    xf = rfftfreq(N, T)

    low_freq_energy = np.sum(np.abs(yf[(xf >= 0) & (xf < 1 / (24 * 3600))]) ** 2)
    high_freq_energy = np.sum(np.abs(yf[(xf >= 1 / (24 * 3600)) & (xf < 1 / 3600)]) ** 2)
    hourly_freq_energy = np.sum(np.abs(yf[(xf >= 1 / 3600) & (xf < 1 / 60)]) ** 2)

    mean_amplitude = np.mean(np.abs(yf))
    variance_amplitude = np.var(np.abs(yf))
    skewness_amplitude = np.mean((np.abs(yf) - mean_amplitude) ** 3) / (variance_amplitude ** 1.5)
    kurtosis_amplitude = np.mean((np.abs(yf) - mean_amplitude) ** 4) / (variance_amplitude ** 2)

    data['low_freq_energy'] = low_freq_energy
    data['high_freq_energy'] = high_freq_energy
    data['hourly_freq_energy'] = hourly_freq_energy
    data['mean_amplitude'] = mean_amplitude
    data['variance_amplitude'] = variance_amplitude
    data['skewness_amplitude'] = skewness_amplitude
    data['kurtosis_amplitude'] = kurtosis_amplitude

    logging.info('Fourier transform and frequencies calculated successfully!')
    return data

def calculate_rolling_means(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling means for several windows.
    """
    data['rolling_mean_close_6'] = data['close'].rolling(window=6).mean()
    data['rolling_mean_close_12'] = data['close'].rolling(window=12).mean()
    data['rolling_mean_close_24'] = data['close'].rolling(window=24).mean()
    data['rolling_mean_close_48'] = data['close'].rolling(window=48).mean()
    data['rolling_mean_close_72'] = data['close'].rolling(window=72).mean()

    logging.info('Rolling means calculated successfully!')
    return data

def calculate_ema(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Exponential Moving Average (EMA) for several windows.
    """
    data['EMA_6'] = data['close'].ewm(span=6, adjust=False).mean()
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_24'] = data['close'].ewm(span=24, adjust=False).mean()
    data['EMA_48'] = data['close'].ewm(span=48, adjust=False).mean()
    data['EMA_72'] = data['close'].ewm(span=72, adjust=False).mean()

    logging.info('EMA calculated successfully!')
    return data

def drop_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows with missing values.
    """
    data.dropna(inplace=True)

    logging.info('Rows with missing values dropped successfully!')
    return data
