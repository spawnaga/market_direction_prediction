import pandas as pd

# Load the preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Initialize columns for trade simulation results and position tracking
df['trade_result'] = 0.0
df['position'] = None  # 'long', 'short', or None

# Variables to track additional details
multiplier = 20  # NQ futures multiplier
profitable_trades = 0
loss_trades = 0
initial_account_value = 100000  # Initial account value, for example, $100,000

# Variable to hold the price at which the last contract was bought or shorted
last_trade_price = 0.0

# Iterate through the DataFrame
for i in range(1, len(df)):
    current_direction = df.loc[i, 'direction']
    current_position = df.loc[i - 1, 'position']

    if current_direction == 1:
        if current_position == 'short':
            trade_result = (last_trade_price - df.loc[i, 'mid_price']) * multiplier
            df.loc[i, 'trade_result'] = trade_result
            df.loc[i, 'position'] = None  # Close position
            if trade_result > 0:
                profitable_trades += 1
            else:
                loss_trades += 1
        else:
            df.loc[i, 'position'] = 'long'
        last_trade_price = df.loc[i, 'mid_price']

    elif current_direction == -1:
        if current_position == 'long':
            trade_result = (df.loc[i, 'mid_price'] - last_trade_price) * multiplier
            df.loc[i, 'trade_result'] = trade_result
            df.loc[i, 'position'] = None  # Close position
            if trade_result > 0:
                profitable_trades += 1
            else:
                loss_trades += 1
        else:
            df.loc[i, 'position'] = 'short'
        last_trade_price = df.loc[i, 'mid_price']
    else:
        df.loc[i, 'position'] = current_position

df['cumulative_profit'] = df['trade_result'].cumsum()

# Calculate the ending value of the account
ending_account_value = initial_account_value + df['cumulative_profit'].iloc[-1]

# Print detailed results
print(f"Total trades: {len(df[df['trade_result'] != 0])}")
print(f"Profitable trades: {profitable_trades}")
print(f"Loss trades: {loss_trades}")
print(f"Total profit/loss: {df['cumulative_profit'].iloc[-1]}")
print(f"Ending account value: {ending_account_value}")

# Save the DataFrame with trade simulation results
df.to_csv("trade_simulation_results_detailed.csv", index=False)
