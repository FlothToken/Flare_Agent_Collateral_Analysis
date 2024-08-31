import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from main import fetch_historical_data, calculate_collateral_ratio, MINIMAL_CR, SAFETY_CR, TOP_UP_CR
from matplotlib.lines import Line2D

# Fixed FLR price for pool collateral
FIXED_FLR_PRICE = 0.05


def backtest_strategy(vault_collateral, asset, start_date, end_date, fixed_flr_price=FIXED_FLR_PRICE, initial_cr=1.5,
                      initial_vault_collateral=100000, initial_pool_collateral=100000):
    """Backtest the strategy for given collaterals and asset using fixed FLR price."""
    # Fetch historical data for vault collateral and asset
    vault_collateral_data = fetch_historical_data(vault_collateral, start_date, end_date)
    asset_data = fetch_historical_data(asset, start_date, end_date)

    # Use the fixed FLR price for pool collateral
    pool_collateral_data = pd.Series(fixed_flr_price, index=vault_collateral_data.index)

    # Align data to common dates
    common_dates = vault_collateral_data.index.intersection(asset_data.index)
    vault_collateral_data = vault_collateral_data[common_dates]
    pool_collateral_data = pool_collateral_data[common_dates]
    asset_data = asset_data[common_dates]

    # Initialize variables
    initial_asset_value = initial_vault_collateral / initial_cr
    initial_vault_units = initial_vault_collateral / vault_collateral_data.iloc[0]
    initial_pool_units = initial_pool_collateral / fixed_flr_price

    vault_cr_series, pool_cr_series = [], []
    dates = []
    liquidation_events = []
    total_additional_vault_collateral = 0
    total_additional_pool_collateral = 0
    current_vault_units = initial_vault_units
    current_pool_units = initial_pool_units
    current_fassets = initial_asset_value

    # Simulate strategy over time
    for date, vault_price, asset_price in zip(common_dates, vault_collateral_data, asset_data):
        pool_price = fixed_flr_price
        vault_collateral_value = current_vault_units * vault_price
        pool_collateral_value = current_pool_units * pool_price
        asset_value = current_fassets * asset_price / asset_data.iloc[0]

        # Calculate collateral ratios
        vault_cr = calculate_collateral_ratio(vault_collateral_value, asset_value)
        pool_cr = calculate_collateral_ratio(pool_collateral_value, asset_value)

        vault_cr_series.append(vault_cr)
        pool_cr_series.append(pool_cr)

        # Initialize variables for this iteration
        additional_vault_collateral_needed = 0
        additional_pool_collateral_needed = 0

        # Check if we need to add collateral to avoid liquidation
        if vault_cr < MINIMAL_CR:
            # Add vault collateral to reach SAFETY_CR
            additional_vault_collateral_needed = max(0, (SAFETY_CR * asset_value - vault_collateral_value))
            current_vault_units += additional_vault_collateral_needed / vault_price
        elif pool_cr < MINIMAL_CR:
            # Add pool collateral to reach SAFETY_CR
            additional_pool_collateral_needed = max(0, (SAFETY_CR * asset_value - pool_collateral_value))
            current_pool_units += additional_pool_collateral_needed / pool_price
        elif pool_cr < TOP_UP_CR:
            # Top-up pool collateral
            additional_pool_collateral_needed = max(0, (SAFETY_CR * asset_value - pool_collateral_value))
            current_pool_units += additional_pool_collateral_needed / pool_price

        if additional_vault_collateral_needed > 0 or additional_pool_collateral_needed > 0:
            liquidation_events.append(
                (date, additional_vault_collateral_needed, additional_pool_collateral_needed, vault_cr, pool_cr))

        # Update total additional collateral
        total_additional_vault_collateral += additional_vault_collateral_needed
        total_additional_pool_collateral += additional_pool_collateral_needed

        dates.append(date)

    return {
        'initial_vault_cr': initial_cr,
        'initial_pool_cr': initial_cr,
        'initial_vault_collateral': initial_vault_collateral,
        'initial_pool_collateral': initial_pool_collateral,
        'liquidation_events': liquidation_events,
        'vault_cr_series': vault_cr_series,
        'pool_cr_series': pool_cr_series,
        'dates': dates,
        'total_additional_vault_collateral': total_additional_vault_collateral,
        'total_additional_pool_collateral': total_additional_pool_collateral
    }


def compare_strategies_over_time(vault_collaterals, assets, start_date, end_date, comparison_window=90):
    """Compare strategies for each week over the given time period."""
    results_over_time = {}
    date_ranges = pd.date_range(start=start_date, end=end_date - timedelta(days=comparison_window), freq='W-MON')

    for current_date in date_ranges:
        print(f"Processing date: {current_date.date()}")
        results = {}
        for vault_collateral in vault_collaterals:
            for asset in assets:
                key = f"{vault_collateral}/{asset}"
                result = backtest_strategy(vault_collateral, asset, current_date,
                                           current_date + timedelta(days=comparison_window))
                results[key] = result
        results_over_time[current_date] = results
    return results_over_time


def extended_plot(results_over_time, assets, vault_collaterals, end_date):
    """Create the final plot showing which strategy would have been better."""
    for asset in assets:
        fig, ax = plt.subplots(figsize=(15, 8))

        # Fetch full asset data for the entire period
        start_date = min(results_over_time.keys())
        full_asset_data = fetch_historical_data(asset, start_date, end_date)
        full_asset_data.index = full_asset_data.index.tz_localize(None)  # Making sure it's timezone naive

        # Plot the full asset price data in black
        ax.plot(full_asset_data.index, full_asset_data.values, color='black', linewidth=1, alpha=0.3)

        for date, results in results_over_time.items():
            eth_key = f"{vault_collaterals[0]}/{asset}"
            usdc_key = f"{vault_collaterals[1]}/{asset}"

            if eth_key in results and usdc_key in results:
                eth_result = results[eth_key]
                usdc_result = results[usdc_key]

                if eth_result is not None and usdc_result is not None:
                    end_date_segment = date + timedelta(days=90)
                    segment_data = full_asset_data.loc[date:end_date_segment]

                    if date <= end_date - timedelta(days=90):
                        color = 'green' if eth_result['total_additional_vault_collateral'] < usdc_result['total_additional_vault_collateral'] else 'blue'
                    else:
                        color = 'black'

                    ax.plot(segment_data.index, segment_data.values, color=color, linewidth=2)

        ax.set_title(f"Price and Better Initialization Strategy for {asset}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.grid(True)

        # Create a custom legend
        custom_lines = [Line2D([0], [0], color='green', lw=2),
                        Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='black', lw=2)]
        ax.legend(custom_lines, ['ETH better', 'USDC better', 'Last 3 months / Full price'])

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'price_strategy_{asset}.png')
        plt.close()


def output_summary(results_over_time, assets, vault_collaterals, output_file='summary.csv'):
    """Output a summary of the backtest results to a CSV file."""
    summary_data = []

    for date, results in results_over_time.items():
        for asset in assets:
            eth_key = f"{vault_collaterals[0]}/{asset}"
            usdc_key = f"{vault_collaterals[1]}/{asset}"

            if eth_key in results and usdc_key in results:
                eth_result = results[eth_key]
                usdc_result = results[usdc_key]

                if eth_result is not None and usdc_result is not None:
                    better_strategy = 'ETH' if eth_result['total_additional_vault_collateral'] < usdc_result[
                        'total_additional_vault_collateral'] else 'USDC'
                    summary_data.append({
                        'Date': date,
                        'Asset': asset,
                        'Better Strategy': better_strategy,
                        'ETH Final Price': eth_result['vault_cr_series'][-1],
                        'USDC Final Price': usdc_result['vault_cr_series'][-1],
                        'ETH Additional Collateral': eth_result['total_additional_vault_collateral'],
                        'USDC Additional Collateral': usdc_result['total_additional_vault_collateral']
                    })

    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")


# Define parameters for the extended backtest
vault_collaterals = ['ETH-USD', 'USDC-USD']
assets = ['XRP-USD', 'BTC-USD', 'DOGE-USD']
end_date = datetime.now().replace(tzinfo=None)
start_date = end_date - timedelta(days=3 * 365)  # 3 years ago

# Run the extended historical backtest
results_over_time = compare_strategies_over_time(vault_collaterals, assets, start_date, end_date)

# Plot and save the results
extended_plot(results_over_time, assets, vault_collaterals, end_date)

# Output a summary to a CSV file
output_summary(results_over_time, assets, vault_collaterals)

print("Analysis complete. Check the generated PNG files and summary.csv for results.")
