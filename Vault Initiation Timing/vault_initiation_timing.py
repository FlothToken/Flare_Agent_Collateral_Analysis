import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from main import fetch_historical_data, calculate_collateral_ratio, MINIMAL_CR, SAFETY_CR, TOP_UP_CR, CCB_CR
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


def backtest_strategy(vault_collateral, asset, start_date, end_date, initial_cr=1.5,
                      initial_vault_collateral=100000, initial_pool_collateral=100000):
    """Backtest the strategy for given collaterals and asset using actual FLR price."""
    # Fetch historical data for vault collateral, asset, and FLR
    vault_collateral_data = fetch_historical_data(vault_collateral, start_date, end_date)
    asset_data = fetch_historical_data(asset, start_date, end_date)
    pool_collateral_data = fetch_historical_data('FLR-USD', start_date, end_date)

    # Check if any of the data is empty
    if vault_collateral_data.empty or asset_data.empty or pool_collateral_data.empty:
        print(f"Insufficient data for {vault_collateral}/{asset} from {start_date} to {end_date}")
        return None

    # Align data to common dates
    common_dates = vault_collateral_data.index.intersection(asset_data.index).intersection(pool_collateral_data.index)
    vault_collateral_data = vault_collateral_data[common_dates]
    pool_collateral_data = pool_collateral_data[common_dates]
    asset_data = asset_data[common_dates]

    # Initialize variables
    initial_asset_value = initial_vault_collateral / initial_cr
    initial_vault_units = initial_vault_collateral / vault_collateral_data.iloc[0]
    initial_pool_units = initial_pool_collateral / pool_collateral_data.iloc[0]

    vault_cr_series, pool_cr_series = [], []
    lowest_vault_cr_series, lowest_pool_cr_series = [], []
    dates = []
    liquidation_events = []
    total_additional_vault_collateral = 0
    total_additional_pool_collateral = 0
    current_vault_units = initial_vault_units
    current_pool_units = initial_pool_units
    current_fassets = initial_asset_value

    # Simulate strategy over time
    for date, vault_price, asset_price, pool_price in zip(common_dates, vault_collateral_data, asset_data,
                                                          pool_collateral_data):
        vault_collateral_value = current_vault_units * vault_price
        pool_collateral_value = current_pool_units * pool_price
        asset_value = current_fassets * asset_price / asset_data.iloc[0]

        # Calculate collateral ratios
        vault_cr = calculate_collateral_ratio(vault_collateral_value, asset_value)
        pool_cr = calculate_collateral_ratio(pool_collateral_value, asset_value)

        # Store the lowest CR before adding collateral
        lowest_vault_cr = vault_cr
        lowest_pool_cr = pool_cr

        # Initialize variables for this iteration
        additional_vault_collateral_needed = 0
        additional_pool_collateral_needed = 0

        # Check if we need to add collateral to avoid liquidation
        if vault_cr < CCB_CR or pool_cr < CCB_CR:
            # Immediate liquidation
            additional_vault_collateral_needed = max(0, (SAFETY_CR * asset_value - vault_collateral_value))
            additional_pool_collateral_needed = max(0, (SAFETY_CR * asset_value - pool_collateral_value))
            current_vault_units += additional_vault_collateral_needed / vault_price
            current_pool_units += additional_pool_collateral_needed / pool_price
        elif vault_cr < MINIMAL_CR:
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

        # Recalculate CR after adding collateral
        vault_cr = calculate_collateral_ratio(current_vault_units * vault_price, asset_value)
        pool_cr = calculate_collateral_ratio(current_pool_units * pool_price, asset_value)

        vault_cr_series.append(vault_cr)
        pool_cr_series.append(pool_cr)
        lowest_vault_cr_series.append(lowest_vault_cr)
        lowest_pool_cr_series.append(lowest_pool_cr)

        if additional_vault_collateral_needed > 0 or additional_pool_collateral_needed > 0:
            liquidation_events.append(
                (date, additional_vault_collateral_needed, additional_pool_collateral_needed, lowest_vault_cr,
                 lowest_pool_cr))

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
        'lowest_vault_cr_series': lowest_vault_cr_series,
        'lowest_pool_cr_series': lowest_pool_cr_series,
        'dates': dates,
        'total_additional_vault_collateral': total_additional_vault_collateral,
        'total_additional_pool_collateral': total_additional_pool_collateral
    }


def compare_strategies_over_time(vault_collaterals, assets, start_date, end_date, comparison_window=90):
    """Compare strategies for each week over the given time period."""
    results_over_time = {}

    # Fetch FLR data to determine the actual start date
    flr_data = fetch_historical_data('FLR-USD', start_date, end_date)
    if flr_data.empty:
        print("No FLR data available for the specified date range.")
        return results_over_time

    actual_start_date = flr_data.index[0].tz_localize(None)  # Make timezone-naive

    # Ensure end_date is timezone-naive
    end_date = end_date.replace(tzinfo=None)

    date_ranges = pd.date_range(start=actual_start_date, end=end_date - timedelta(days=comparison_window), freq='W-MON')

    for current_date in date_ranges:
        print(f"Processing date: {current_date.date()}")
        results = {}
        for vault_collateral in vault_collaterals:
            for asset in assets:
                key = f"{vault_collateral}/{asset}"
                result = backtest_strategy(vault_collateral, asset, current_date,
                                           current_date + timedelta(days=comparison_window))
                if result is not None:
                    results[key] = result
        if results:  # Only add to results_over_time if there are valid results
            results_over_time[current_date] = results
    return results_over_time


def extended_plot(results_over_time, assets, vault_collaterals, end_date):
    """Create the final plot showing which strategy would have been better."""
    for asset in assets:
        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()  # Create a secondary y-axis

        # Remove "-USD" suffix from asset name
        clean_asset_name = asset.replace("-USD", "")

        # Fetch full asset data for the entire period
        start_date = min(results_over_time.keys())
        full_asset_data = fetch_historical_data(asset, start_date, end_date)
        full_asset_data.index = full_asset_data.index.tz_localize(None)  # Making sure it's timezone naive

        # Plot the full asset price data in black
        ax1.plot(full_asset_data.index, full_asset_data.values, color='black', linewidth=1, alpha=0.3)

        additional_collateral_data = []

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
                        eth_additional = eth_result['total_additional_vault_collateral']
                        usdc_additional = usdc_result['total_additional_vault_collateral']

                        # Update color selection logic
                        if eth_additional == 0 and usdc_additional == 0:
                            color = 'green' if eth_result['vault_cr_series'][-1] > usdc_result['vault_cr_series'][
                                -1] else 'blue'
                        else:
                            color = 'green' if eth_additional < usdc_additional else 'blue'

                        additional_collateral = min(eth_additional, usdc_additional)
                    else:
                        color = 'black'
                        additional_collateral = 0

                    ax1.plot(segment_data.index, segment_data.values, color=color, linewidth=2)
                    additional_collateral_data.append((date, additional_collateral))

        # Plot additional collateral requirements
        dates, collateral_values = zip(*additional_collateral_data)
        ax2.plot(dates, collateral_values, color='red', linestyle='--', alpha=0.5)

        ax1.set_title(f"{clean_asset_name} Price and Better Vault Initialization Strategy")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax2.set_ylabel("Additional Collateral Required (Best Strategy)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())  # Show every month
        ax1.grid(True)

        # Rotate and align the tick labels so they look better
        fig.autofmt_xdate()

        # Create a custom legend
        custom_lines = [Line2D([0], [0], color='green', lw=2),
                        Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='red', linestyle='--', alpha=0.5)]
        ax1.legend(custom_lines, ['ETH better', 'USDC better', 'Additional Collateral (Best Strategy)'])

        plt.tight_layout()
        plt.savefig(f'price_strategy_{clean_asset_name}.png')
        plt.close()


def interactive_plot(results_over_time, assets, vault_collaterals, end_date):
    """Create an interactive plot with toggle options for CR and additional collateral."""
    for asset in assets:
        clean_asset_name = asset.replace("-USD", "")

        start_date = min(results_over_time.keys())
        full_asset_data = fetch_historical_data(asset, start_date, end_date)
        full_asset_data.index = full_asset_data.index.tz_localize(None)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add full asset price data
        fig.add_trace(
            go.Scatter(x=full_asset_data.index, y=full_asset_data.values, name=f"{clean_asset_name} Price",
                       line=dict(color='black', width=1)),
            secondary_y=False,
        )

        best_cr_data = []
        additional_collateral_data = []

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
                        eth_additional = eth_result['total_additional_vault_collateral']
                        usdc_additional = usdc_result['total_additional_vault_collateral']

                        if eth_additional == 0 and usdc_additional == 0:
                            color = 'green' if eth_result['vault_cr_series'][-1] > usdc_result['vault_cr_series'][
                                -1] else 'blue'
                        else:
                            color = 'green' if eth_additional < usdc_additional else 'blue'

                        best_strategy = 'ETH' if color == 'green' else 'USDC'
                        best_cr = eth_result['vault_cr_series'][-1] if best_strategy == 'ETH' else \
                        usdc_result['vault_cr_series'][-1]
                        best_additional_collateral = min(eth_additional, usdc_additional)

                        # Add segment to plot (part of default view)
                        fig.add_trace(
                            go.Scatter(x=segment_data.index, y=segment_data.values, name=f"Segment {date}",
                                       line=dict(color=color, width=2), showlegend=False),
                            secondary_y=False,
                        )

                        best_cr_data.append((date, best_cr))
                        additional_collateral_data.append((date, best_additional_collateral))

        # Add CR traces (all in one group)
        cr_dates, cr_values = zip(*best_cr_data)
        fig.add_trace(
            go.Scatter(x=cr_dates, y=cr_values, name="Best Strategy CR", line=dict(color='purple', width=1),
                       visible='legendonly', legendgroup='CR'),
            secondary_y=True,
        )

        # Add CR threshold lines (all in one group)
        cr_thresholds = [
            (CCB_CR, "CCB CR", 'red'),
            (MINIMAL_CR, "Minimal CR", 'orange'),
            (SAFETY_CR, "Safety CR", 'blue'),
            (TOP_UP_CR, "Top-up CR", 'green')
        ]
        for cr, name, color in cr_thresholds:
            fig.add_trace(
                go.Scatter(x=[start_date, end_date], y=[cr, cr], name=name,
                           line=dict(color=color, width=1, dash='dash'), visible='legendonly', legendgroup='CR'),
                secondary_y=True,
            )

        # Add additional collateral trace
        collateral_dates, collateral_values = zip(*additional_collateral_data)
        fig.add_trace(
            go.Scatter(x=collateral_dates, y=collateral_values, name="Additional Collateral",
                       line=dict(color='red', width=1, dash='dash'), visible='legendonly'),
            secondary_y=False,
        )

        # Update layout
        fig.update_layout(
            title=f"{clean_asset_name} Price, Best Strategy CR, and Additional Collateral",
            xaxis_title="Date",
            legend_title="Legend",
            hovermode="x unified"
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="Price / Additional Collateral", secondary_y=False)
        fig.update_yaxes(title_text="Collateral Ratio (CR)", secondary_y=True)

        # Save the plot as an HTML file
        fig.write_html(f'interactive_plot_{clean_asset_name}.html')


def save_results(results, filename='backtest_results.json'):
    def convert_timestamps(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return str(obj)

    # Convert Timestamp keys to strings
    converted_results = {k.isoformat() if isinstance(k, pd.Timestamp) else k: v for k, v in results.items()}

    with open(filename, 'w') as f:
        json.dump(converted_results, f, default=convert_timestamps)
    print(f"Results saved to {filename}")


def load_results(filename='backtest_results.json'):
    with open(filename, 'r') as f:
        data = json.load(f)

    # Convert string keys back to Timestamps
    return {pd.Timestamp(k) if k.startswith('20') else k: v for k, v in data.items()}


# Main execution
if __name__ == "__main__":
    # Define parameters for the extended backtest
    vault_collaterals = ['ETH-USD', 'USDC-USD']
    assets = ['XRP-USD', 'BTC-USD', 'DOGE-USD']
    end_date = datetime.now().replace(tzinfo=None)
    start_date = end_date - timedelta(days=3 * 365)  # 3 years ago, but will be adjusted based on FLR data availability

    # Toggle for calculation
    run_calculation = True  # Set as False to skip calculation and use saved results

    if run_calculation:
        # Run the extended historical backtest
        results_over_time = compare_strategies_over_time(vault_collaterals, assets, start_date, end_date)
        # Save the results as JSON
        save_results(results_over_time)
    else:
        # Load the results from JSON
        results_over_time = load_results()

    # Generate both PNG and interactive HTML plots
    extended_plot(results_over_time, assets, vault_collaterals, end_date)
    interactive_plot(results_over_time, assets, vault_collaterals, end_date)

    print("Analysis complete. Check the generated PNG files and HTML files for results.")
