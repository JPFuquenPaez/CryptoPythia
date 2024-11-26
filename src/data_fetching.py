import requests
import pandas as pd
import logging
from config import SYMBOL, INTERVAL, START_TIME, END_TIME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_historical_data(symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
    """
    Fetch historical data from Binance API in chunks.
    """
    df_list = []
    current_time = start_time

    while current_time < end_time:
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={current_time}&endTime={end_time}'

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df_list.append(df[['close', 'volume']])
            current_time = int(df.index[-1].timestamp() * 1000) + 1

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            break

    if df_list:
        final_df = pd.concat(df_list)
    else:
        final_df = pd.DataFrame(columns=['close', 'volume'])

    logging.info('Data fetched successfully!')
    return final_df
