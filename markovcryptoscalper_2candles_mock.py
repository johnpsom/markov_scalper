# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 00:00:00 2024

@author: ioann
"""

import os
import time
from datetime import datetime
from collections import defaultdict
import warnings
import pandas as pd


from binance.exceptions import BinanceAPIException
from binance.client import Client


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
api_key = os.environ.get('binance_api')
api_secret = os.environ.get('binance_secret')
# Initialize Binance client
client = Client(api_key, api_secret)


# Function to initially download all historical price data
def get_price_data(symbol, interval, duration):
    df = pd.DataFrame(client.futures_historical_klines(
        symbol, interval, duration)).iloc[:-1, :7]
    df.columns = ["datetime_open", "open", "high",
                  "low", "close", "volume", "datetime_close"]
    df['datetime_open'] = pd.to_datetime(df['datetime_open']//1000, unit='s')
    df['datetime_close'] = pd.to_datetime(df['datetime_close']//1000, unit='s')
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]].astype(float)
    # Calculate the direction based on comparison of 'open' and 'close' values from the same row
    df['direction'] = df.apply(lambda row: 1 if row['close'] > row['open']
                               else (-1 if row['close'] < row['open'] else 0), axis=1)
    # Calculate the return for each row
    df['return'] = ((df['close'] - df['open']) / df['open']) * 100
    # Set datetime_open as index
    df.set_index('datetime_open', inplace=True)

    # Get percentile thresholds for volatility
    q25 = df['return'].quantile(0.15)
    q75 = df['return'].quantile(0.84)

    def classify(row):
        # Determine direction
        direction_class = 'U' if row['direction'] == 1 else 'D' if row['direction'] == -1 else 'U'

        # Determine volatility based on return and thresholds
        if row['direction'] == 1:  # Price went up
            volatility_class = 'V' if row['return'] >= q75 else 'v'
        elif row['direction'] == -1:  # Price went down
            volatility_class = 'V' if row['return'] <= q25 else 'v'
        else:  # Price unchanged, classify as 'Uv' by default
            volatility_class = 'v'

        # Combine the direction and volatility classification
        return direction_class + volatility_class

    # Apply the classification
    df['volatility_class'] = df.apply(classify, axis=1)
    return df


# Function to update our price data with last candle
def update_data_with_last_candle(symbol, interval, df):
    """
    Downloads the latest candle and appends it to the existing dataframe.
    Ensures the dataframe grows over time dynamically.
    """
    last_candle_df = pd.DataFrame(client.futures_klines(
        symbol=symbol, interval=interval, limit=2))
    last_candle_df.columns = ["datetime_open", "open", "high", "low", "close", "volume", "close_time",
                              "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    last_candle_df['datetime_open'] = pd.to_datetime(
        last_candle_df['datetime_open'] // 1000, unit='s')
    last_candle_df['datetime_close'] = pd.to_datetime(
        last_candle_df['close_time'] // 1000, unit='s')
    last_candle_df = last_candle_df.loc[:, ["datetime_open", "open", "high",
                                            "low", "close", "volume", 'datetime_close']]  # Keep relevant columns
    last_candle_df[["open", "high", "low", "close", "volume"]] = last_candle_df[
        ["open", "high", "low", "close", "volume"]].astype(float)
    last_candle_df.set_index('datetime_open', inplace=True)
    # Append only if the last candle isn't already in the dataframe
    if last_candle_df.index[-2] > df.index[-1]:
        df = pd.concat([df, last_candle_df.iloc[0:1]], ignore_index=False)

    # Recalculate the features like direction and return for the new candle
    df['direction'] = df.apply(lambda row: 1 if row['close'] > row['open']
                               else (-1 if row['close'] < row['open'] else 0), axis=1)
    df['return'] = ((df['close'] - df['open']) / df['open']) * 100

    # Recalculate percentiles dynamically as the dataframe grows
    q25 = df['return'].quantile(0.22)
    q75 = df['return'].quantile(0.77)

    def classify(row):
        # Determine direction
        direction_class = 'U' if row['direction'] == 1 else 'D' if row['direction'] == -1 else 'U'
        # Determine volatility based on return and thresholds
        if row['direction'] == 1:  # Price went up
            volatility_class = 'V' if row['return'] >= q75 else 'v'
        elif row['direction'] == -1:  # Price went down
            volatility_class = 'V' if row['return'] <= q25 else 'v'
        else:  # Price unchanged, classify as 'Uv' by default
            volatility_class = 'v'
        return direction_class + volatility_class

    df['volatility_class'] = df.apply(classify, axis=1)

    return df


def create_markov_chain_two_candles(df, state_size=4):
    """
    Creates a Markov Chain transition matrix based on the volatility class, considering transitions to the next two candles.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'volatility_class' column.
        state_size (int): The number of rows to use for the present state.

    Returns:
        dict: A transition probability matrix for predicting two subsequent candles.
    """
    transitions = defaultdict(lambda: defaultdict(int))

    for i in range(len(df) - state_size - 1):  # Need to look two steps ahead, hence -1
        # Extract the present state (last 'state_size' rows of volatility_class)
        present_state = tuple(df['volatility_class'].iloc[i:i + state_size])

        # The next two states (rows after the present state sequence)
        next_two_states = tuple(
            df['volatility_class'].iloc[i + state_size:i + state_size + 2])

        # Count the transition from present state to the next two states
        transitions[present_state][next_two_states] += 1

    # Convert counts to probabilities (transition probability matrix)
    transition_matrix = {}

    for present_state, next_state_counts in transitions.items():
        total_transitions = sum(next_state_counts.values())

        # Convert counts to probabilities
        transition_matrix[present_state] = {next_state: count / total_transitions
                                            for next_state, count in next_state_counts.items()}

    return transition_matrix


def get_two_candle_prediction(df, transition_matrices, state_sizes):
    """
    Predict the next two volatility classes using the top prediction from each transition matrix,
    excluding the current candle which is still being formed.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the historical data with 'volatility_class'.
        transition_matrices (list): A list of transition matrices (e.g., [matrix4, matrix5, matrix6]).
        state_sizes (list): A list of state sizes corresponding to each transition matrix (e.g., [4, 5, 6]).

    Returns:
        tuple: (predicted next two candles, associated probabilities)
    """
    top_predictions = []

    for matrix, state_size in zip(transition_matrices, state_sizes):
        # Get the present state (the last 'state_size' volatility classes, excluding the current forming candle)
        present_state = tuple(df['volatility_class'].iloc[-(state_size + 1):-1])

        # Get the top prediction for the next two candles
        if present_state in matrix:
            next_state_probs = matrix[present_state]
            top_prediction = max(next_state_probs, key=next_state_probs.get)
            top_predictions.append((top_prediction))
        else:
            top_predictions.append((None, None))

    if len(top_predictions) < 3:
        return (None, None)  # No valid predictions
    else:
        # Count the occurrences of each prediction
        prediction_counts = pd.Series(top_predictions).value_counts()

        # If there's a majority prediction, return it
        if prediction_counts.iloc[0] > 1:
            return prediction_counts.index[0]
        else:
            return (None, None)  # No clear majority


# Function to get the leverage for a given symbol (BTCUSDC)
def get_leverage(symbol):

    account_info = client.futures_position_information(symbol=symbol)
    for position in account_info:
        if position['symbol'] == symbol:
            return int(position['leverage'])  # Return the leverage level for this symbol
    return 1  # Default leverage if not found


# Function to calculate the amount of BTC to trade using leverage with retry logic
def calculate_amount_with_leverage(symbol, balance, price, percentage=0.25):

    leverage = get_leverage(symbol)
    # 25% of the balance
    # Leverage allows controlling more with less capital
    amount_to_trade_usd = balance * percentage * leverage
    # Calculate how much BTC we can trade with that amount (leverage adjusted)
    amount_to_trade = amount_to_trade_usd / price
    return round(amount_to_trade, 3)


# Function to get the current price of the symbol (BTCUSDC)
def get_current_price(symbol):

    ticker = client.futures_symbol_ticker(symbol=symbol)
    return float(ticker['price'])


# Function to get the latest balance
def get_balance():
    account_info = client.futures_account()
    for asset in account_info['assets']:
        if asset['asset'] == 'BNFCR':
            return float(asset['availableBalance'])  # Available USDT balance
    return 0


# Function to open a position (long or short) with retry logic
def open_position(symbol, position_type, amount, max_retries=6):
    """
    Open a futures position with retry logic.

    Parameters:
        symbol (str): Trading pair (e.g., 'BTCUSDC')
        position_type (str): 'long' for buying, 'short' for selling
        amount (float): Quantity of the asset to trade
        max_retries (int): Number of retries in case of failure

    Returns:
        dict: Response from the Binance API
    """
    for _ in range(max_retries):
        try:
            if position_type == 'long':
                return client.futures_create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=amount
                )
            if position_type == 'short':
                return client.futures_create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=amount
                )
        except BinanceAPIException as err:
            if err.code == -1021:  # Timestamp issue
                print("Timestamp error, synchronizing time with Binance and retrying...")
                # Synchronize system time with Binance server time
                server_time = client.get_server_time()
                time_offset = server_time['serverTime'] - int(time.time() * 1000)
                client.time_offset = time_offset  # Adjust the client's time offset
                time.sleep(1)  # Small delay before retrying
            else:
                print(f"Error opening position: {err}")
                time.sleep(1)  # Add delay before retrying for any other errors
    raise Exception(f"Failed to open position after {max_retries} retries")


# Function to close a position (long or short) with retry logic
def close_position(symbol, position_type, amount, max_retries=6):
    """
    Close a futures position with retry logic.

    Parameters:
        symbol (str): Trading pair (e.g., 'BTCUSDC')
        position_type (str): 'long' for closing a long position,
                             'short' for closing a short position
        amount (float): Quantity of the asset to trade
        max_retries (int): Number of retries in case of failure

    Returns:
        dict: Response from the Binance API
    """
    for _ in range(max_retries):
        try:
            if position_type == 'long':
                return client.futures_create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=amount
                )
            if position_type == 'short':
                return client.futures_create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=amount
                )
        except BinanceAPIException as err:
            if err.code == -1021:  # Timestamp issue
                print("Timestamp error, synchronizing time with Binance and retrying...")
                # Synchronize system time with Binance server time
                server_time = client.get_server_time()
                time_offset = server_time['serverTime'] - int(time.time() * 1000)
                client.time_offset = time_offset  # Adjust the client's time offset
                time.sleep(1)  # Small delay before retrying
            else:
                print(f"Error closing position: {err}")
                time.sleep(1)  # Add delay before retrying for any other errors
    raise Exception(f"Failed to close position after {max_retries} retries")


# Function to handle retrying on API timestamp issue
def execute_with_retry(function, *args, max_retries=5, **kwargs):
    """
    Executes a function with retry logic in case of a timestamp error (-1021).
    """
    for _ in range(max_retries):
        try:
            return function(*args, **kwargs)
        except BinanceAPIException as err:
            if err.code == -1021:  # Timestamp issue
                print("Timestamp error, synchronizing time with Binance and retrying...")
                # Synchronize system time with Binance server time
                server_time = client.get_server_time()
                time_offset = server_time['serverTime'] - int(time.time() * 1000)
                client.time_offset = time_offset  # Adjust the client's time offset
                time.sleep(1)  # Small delay before retrying
            else:
                raise err  # Raise other API errors
    raise Exception("Max retries reached for function execution")


# Get the current server time from Binance (in milliseconds)
def get_binance_time():
    server_time = client.get_server_time()
    return server_time['serverTime']


# Function to calculate the time remaining until the next 15-minute candle
def get_time_until_next_candle():
    current_time = datetime.utcnow()
    minutes = current_time.minute
    seconds = current_time.second
    # Calculate minutes remaining until the next 15m candle (00, 15, 30, 45 minutes)
    remaining_minutes = 60 - (minutes % 60)
    remaining_seconds = (remaining_minutes * 60) - seconds
    return remaining_seconds


# Function to sleep until the next candle
def wait_for_next_candle():
    seconds_to_sleep = get_time_until_next_candle()
    print(f"Waiting for {seconds_to_sleep} seconds until the next 15-minute candle...")
    time.sleep(seconds_to_sleep)


# Main trading loop with synchronization to candle creation
def scalping_strategy(symbol, interval, duration):
    df = get_price_data(symbol, interval, duration)
    print(df)
    trades_df = pd.DataFrame()
    trades = 0
    profit_loss = 0
    # Create transition matrices
    state_sizes = [3, 4, 5]

    transition_matrix4 = create_markov_chain_two_candles(df, state_size=state_sizes[0])
    transition_matrix5 = create_markov_chain_two_candles(df, state_size=state_sizes[1])
    transition_matrix6 = create_markov_chain_two_candles(df, state_size=state_sizes[2])

    transition_matrices = [transition_matrix4, transition_matrix5, transition_matrix6]

    # Predict next volatility class at the close of the current candle
    predicted_class = get_two_candle_prediction(df, transition_matrices, state_sizes)
    print(df.tail(8))
    print(predicted_class)
    # Get current balance and price with retry logic
    balance = 400  # execute_with_retry(get_balance)
    current_price = execute_with_retry(get_current_price, symbol)

    # Calculate how much BTC to trade with leverage (25% of balance)
    amount_to_trade = execute_with_retry(
        calculate_amount_with_leverage, symbol, balance, current_price)

    in_trade = False
    entry_price = 0
    commission_fee = 1  # Assume a fixed commission fee of 1 unit per trade

    # Enter trade based on the prediction
    if predicted_class[0] == "UV" and predicted_class[1] in ['UV', 'Uv']:
        entry_price = current_price
        print(
            f"Mock trade - Entering LONG position at {entry_price} with amount {amount_to_trade}")
        in_trade = True
    elif predicted_class[0] == "Uv" and predicted_class[1] in ['UV']:
        entry_price = current_price
        print(
            f"Mock trade - Entering LONG position at {entry_price} with amount {amount_to_trade}")
        in_trade = True
    elif predicted_class[0] == "DV" and predicted_class[1] in ['DV', 'Dv']:
        entry_price = current_price
        print(
            f"Mock trade - Entering SHORT position at {entry_price} with amount {amount_to_trade}")
        in_trade = True
    elif predicted_class[0] == "Dv" and predicted_class[1] in ['DV']:
        entry_price = current_price
        print(
            f"Mock trade - Entering SHORT position at {entry_price} with amount {amount_to_trade}")
        in_trade = True
    else:
        print("No trade, waiting for next prediction")

    # Wait for the first candle to close
    wait_for_next_candle()

    while True:
        time.sleep(5)
        if in_trade:
            trades += 1
            # Get the next candle's close price
            next_close_price = execute_with_retry(get_current_price, symbol)

            # Close the position and calculate profit/loss
            if predicted_class[0] in ['UV', 'Uv']:
                profit_loss = (next_close_price - entry_price) * \
                    amount_to_trade - commission_fee
                print(f"Mock trade - Closing LONG position at {next_close_price}")
            elif predicted_class[0] in ['DV', 'Dv']:
                profit_loss = (entry_price - next_close_price) * \
                    amount_to_trade - commission_fee
                print(f"Mock trade - Closing SHORT position at {next_close_price}")

            print(
                f"Entry price: {entry_price}, Exit price: {next_close_price}, Profit/Loss: {profit_loss}, Commission: {commission_fee}")
            balance += profit_loss
            print(f"New Balance: {balance}\n")
            in_trade = False
            current_time = datetime.utcnow()
            cur_time = datetime.strftime(current_time, '%H:%M')
            stats_dict = {'n': trades, 'Time': cur_time, 'Prediction': predicted_class,
                          'entry': entry_price, 'next price': next_close_price, 'PnL': profit_loss}
            trades_df = pd.concat(
                [trades_df, pd.DataFrame([stats_dict])], ignore_index=True)
            trades_df.to_csv('trades.csv')

        # Update the dataframe with the most recent candle
        df = update_data_with_last_candle(symbol, interval, df)

        # Update transition matrices
        transition_matrix4 = create_markov_chain_two_candles(
            df, state_size=state_sizes[0])
        transition_matrix5 = create_markov_chain_two_candles(
            df, state_size=state_sizes[1])
        transition_matrix6 = create_markov_chain_two_candles(
            df, state_size=state_sizes[2])
        transition_matrices = [transition_matrix4, transition_matrix5, transition_matrix6]

        # Predict next volatility class at the close of the current candle
        predicted_class = get_two_candle_prediction(df, transition_matrices, state_sizes)
        print(df.tail(8))
        print(predicted_class, balance)

        # Get current price with retry logic
        current_price = execute_with_retry(get_current_price, symbol)

        # Calculate how much BTC to trade with leverage (25% of balance)
        amount_to_trade = execute_with_retry(
            calculate_amount_with_leverage, symbol, balance, current_price)

        # Enter trade based on the prediction
        if predicted_class[0] == "UV" and predicted_class[1] in ['UV', 'Uv']:
            entry_price = current_price
            print(
                f"Mock trade - Entering LONG position at {entry_price} with amount {amount_to_trade}")
            in_trade = True
        elif predicted_class[0] == "Uv" and predicted_class[1] in ['UV']:
            entry_price = current_price
            print(
                f"Mock trade - Entering LONG position at {entry_price} with amount {amount_to_trade}")
            in_trade = True
        elif predicted_class[0] == "DV" and predicted_class[1] in ['DV', 'Dv']:
            entry_price = current_price
            print(
                f"Mock trade - Entering SHORT position at {entry_price} with amount {amount_to_trade}")
            in_trade = True
        elif predicted_class[0] == "Dv" and predicted_class[1] in ['DV']:
            entry_price = current_price
            print(
                f"Mock trade - Entering SHORT position at {entry_price} with amount {amount_to_trade}")
            in_trade = True
        else:
            print("No trade, waiting for next prediction")

        # Wait for the next candle to close before closing the position
        wait_for_next_candle()


# Example usage
scalping_strategy(symbol='ADAUSDT', interval='1h', duration='3y')
