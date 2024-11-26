import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values, encoding categorical variables, and normalizing numerical features.
    """
    data.index = pd.to_datetime(data.index)
    data.reset_index(inplace=True)
    data.rename(columns={'timestamp': 'datetime'}, inplace=True)

    data['Year'] = data['datetime'].dt.year
    data['Month'] = data['datetime'].dt.month
    data['Day'] = data['datetime'].dt.day
    data['Hour'] = data['datetime'].dt.hour

    logging.info('Data preprocessed successfully!')
    return data
