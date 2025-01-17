import pandas as pd
from datetime import datetime

# Load the data from the Webull CSV file
df = pd.read_csv(r'C:\Users\chadc\Downloads\Webull_Orders_Records_Options (76).csv')

# Strip timezone info and any trailing spaces or extra characters in the 'Placed Time' and 'Filled Time' columns
df['Placed Time'] = df['Placed Time'].str.replace(r'\s+EDT|\s+EST', '', regex=True).str.strip()
df['Filled Time'] = df['Filled Time'].str.replace(r'\s+EDT|\s+EST', '', regex=True).str.strip()

# Now try converting the timestamps to datetime format with error handling
df['Placed Time'] = pd.to_datetime(df['Placed Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
df['Filled Time'] = pd.to_datetime(df['Filled Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')

# Check if there are any rows with invalid/missing timestamps
invalid_times = df[df['Placed Time'].isnull() | df['Filled Time'].isnull()]
if not invalid_times.empty:
    print(f"Found invalid timestamps in the following rows:\n{invalid_times}")

# Add a column for P&L calculation. Assume 'Sell' gives profit and 'Buy' takes a cost.
df['Price'] = df['Price'].str.replace('@', '').astype(float)
df['pnl'] = df.apply(lambda row: row['Filled'] * row['Price'] if row['Side'] == 'Sell' else -row['Filled'] * row['Price'], axis=1)

# Initial statistics calculation variables
total_pnl = df['pnl'].sum()
total_trades = len(df)
winning_trades = df[df['pnl'] > 0].shape[0]
losing_trades = df[df['pnl'] < 0].shape[0]
break_even_trades = df[df['pnl'] == 0].shape[0]
largest_profit = df['pnl'].max()
largest_loss = df['pnl'].min()
average_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0

# Time and hold time calculations (only for valid dates)
valid_dates_df = df.dropna(subset=['Placed Time', 'Filled Time'])
valid_dates_df['Hold Time'] = (valid_dates_df['Filled Time'] - valid_dates_df['Placed Time']).dt.total_seconds() / 60  # in minutes
average_hold_time_all_trades = valid_dates_df['Hold Time'].mean()
average_hold_time_winning_trades = valid_dates_df[valid_dates_df['pnl'] > 0]['Hold Time'].mean()
average_hold_time_losing_trades = valid_dates_df[valid_dates_df['pnl'] < 0]['Hold Time'].mean()
average_hold_time_break_even_trades = valid_dates_df[valid_dates_df['pnl'] == 0]['Hold Time'].mean()

# Max consecutive wins/losses calculation
valid_dates_df['win'] = valid_dates_df['pnl'] > 0
valid_dates_df['loss'] = valid_dates_df['pnl'] < 0
valid_dates_df['consecutive_win'] = (valid_dates_df['win'] != valid_dates_df['win'].shift()).cumsum()
valid_dates_df['consecutive_loss'] = (valid_dates_df['loss'] != valid_dates_df['loss'].shift()).cumsum()
max_consecutive_wins = valid_dates_df[valid_dates_df['win']].groupby('consecutive_win').size().max()
max_consecutive_losses = valid_dates_df[valid_dates_df['loss']].groupby('consecutive_loss').size().max()

# Calculate average daily volume and total trading days
valid_dates_df['Trading Day'] = valid_dates_df['Placed Time'].dt.date
average_daily_volume = valid_dates_df.groupby('Trading Day')['Filled'].sum().mean()
total_trading_days = valid_dates_df['Trading Day'].nunique()

# Display statistics
print(f"Total P&L: ${total_pnl:.2f}")
print(f"Average Daily Volume: {average_daily_volume:.2f}")
print(f"Average Winning Trade: ${valid_dates_df[valid_dates_df['pnl'] > 0]['pnl'].mean():.2f}")
print(f"Average Losing Trade: ${valid_dates_df[valid_dates_df['pnl'] < 0]['pnl'].mean():.2f}")
print(f"Total Number of Trades: {total_trades}")
print(f"Number of Winning Trades: {winning_trades}")
print(f"Number of Losing Trades: {losing_trades}")
print(f"Number of Break Even Trades: {break_even_trades}")
print(f"Max Consecutive Wins: {max_consecutive_wins}")
print(f"Max Consecutive Losses: {max_consecutive_losses}")
print(f"Largest Profit: ${largest_profit:.2f}")
print(f"Largest Loss: ${largest_loss:.2f}")
print(f"Average Hold Time (All Trades): {average_hold_time_all_trades:.2f} minutes")
print(f"Average Hold Time (Winning Trades): {average_hold_time_winning_trades:.2f} minutes")
print(f"Average Hold Time (Losing Trades): {average_hold_time_losing_trades:.2f} minutes")
print(f"Average Hold Time (Break Even Trades): {average_hold_time_break_even_trades:.2f} minutes")
print(f"Average Trade P&L: ${average_trade_pnl:.2f}")
