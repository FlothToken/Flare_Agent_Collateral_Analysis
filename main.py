import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants for collateral ratios
MINIMAL_CR = 1.4
CCB_CR = 1.3
SAFETY_CR = 1.5
MINTING_CR = 1.6
EXIT_CR = 1.45
TOP_UP_CR = 1.55
AGENT_POOL_STAKE_RATIO = 0.2
INITIAL_POOL_CR = 1.5

def fetch_historical_data(symbol, start_date, end_date):
    """Fetch historical price data for a given symbol."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    return data['Close']

def calculate_collateral_ratio(collateral_value, asset_value):
    """Calculate the collateral ratio."""
    return collateral_value / asset_value if asset_value != 0 else float('inf')

def backtest_strategy(vault_collateral, pool_collateral, asset, start_date, end_date, initial_cr=1.5,
                      initial_vault_collateral=100000, initial_pool_collateral=100000):
    """Backtest the strategy for given collaterals and asset."""
    # Fetch historical data
    vault_collateral_data = fetch_historical_data(vault_collateral, start_date, end_date)
    pool_collateral_data = fetch_historical_data(pool_collateral, start_date, end_date)
    asset_data = fetch_historical_data(asset, start_date, end_date)

    # Align data to common dates
    common_dates = vault_collateral_data.index.intersection(pool_collateral_data.index).intersection(asset_data.index)
    vault_collateral_data = vault_collateral_data[common_dates]
    pool_collateral_data = pool_collateral_data[common_dates]
    asset_data = asset_data[common_dates]

    # Initialize variables
    initial_asset_value = initial_vault_collateral / initial_cr
    initial_vault_units = initial_vault_collateral / vault_collateral_data.iloc[0]
    initial_pool_units = initial_pool_collateral / pool_collateral_data.iloc[0]

    vault_cr_series, pool_cr_series = [], []
    dates = []
    liquidation_events = []
    full_liquidations = 0
    below_minimum_events = 0
    healthy_days = 0
    current_vault_units = initial_vault_units
    current_pool_units = initial_pool_units
    total_additional_vault_collateral = 0
    total_additional_pool_collateral = 0
    current_fassets = initial_asset_value

    # Simulate strategy over time
    for date, vault_price, pool_price, asset_price in zip(common_dates, vault_collateral_data,
                                                          pool_collateral_data,
                                                          asset_data):
        vault_collateral_value = current_vault_units * vault_price
        pool_collateral_value = current_pool_units * pool_price
        asset_value = current_fassets * asset_price / asset_data.iloc[0]

        # Calculate collateral ratios
        vault_cr = calculate_collateral_ratio(vault_collateral_value, asset_value)
        pool_cr = calculate_collateral_ratio(pool_collateral_value, asset_value)

        vault_cr_series.append(vault_cr)
        pool_cr_series.append(pool_cr)

        if vault_cr >= SAFETY_CR and pool_cr >= SAFETY_CR:
            healthy_days += 1

        # Initialize variables for this iteration
        additional_vault_collateral_needed = 0
        additional_pool_collateral_needed = 0

        # Check if we need to add collateral to avoid liquidation
        if vault_cr < MINIMAL_CR:
            # Add vault collateral to reach SAFETY_CR
            additional_vault_collateral_needed = max(0, (SAFETY_CR * asset_value - vault_collateral_value))
            current_vault_units += additional_vault_collateral_needed / vault_price
            below_minimum_events += 1
        elif pool_cr < MINIMAL_CR:
            # Add pool collateral to reach SAFETY_CR
            additional_pool_collateral_needed = max(0, (SAFETY_CR * asset_value - pool_collateral_value))
            current_pool_units += additional_pool_collateral_needed / pool_price
            below_minimum_events += 1
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
        'full_liquidations': full_liquidations,
        'below_minimum_events': below_minimum_events,
        'healthy_days': healthy_days,
        'total_days': len(vault_cr_series),
        'avg_vault_cr': np.mean(vault_cr_series),
        'avg_pool_cr': np.mean(pool_cr_series),
        'min_vault_cr': min(vault_cr_series),
        'min_pool_cr': min(pool_cr_series),
        'max_vault_cr': max(vault_cr_series),
        'max_pool_cr': max(pool_cr_series),
        'vault_cr_series': vault_cr_series,
        'pool_cr_series': pool_cr_series,
        'dates': dates,
        'total_additional_vault_collateral': total_additional_vault_collateral,
        'total_additional_pool_collateral': total_additional_pool_collateral
    }

def compare_strategies(vault_collaterals, pool_collateral, assets, start_date, end_date):
    """Compare different strategies for given collaterals and assets."""
    results = {}
    for vault_collateral in vault_collaterals:
        for asset in assets:
            key = f"{vault_collateral}/{asset}"
            results[key] = backtest_strategy(vault_collateral, pool_collateral, asset, start_date, end_date,
                                             initial_vault_collateral=100000, initial_pool_collateral=100000)
    return results

def plot_results(results):
    """Plot collateral ratio dynamics for different strategies."""
    colors = {'XRP': 'blue', 'BTC': 'orange', 'DOGE': 'green'}
    for vault_collateral in ['USDC', 'ETH']:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), sharex=True)

        max_additional_vault = 0
        max_additional_pool = 0

        for asset in ['XRP', 'BTC', 'DOGE']:
            key = f"{vault_collateral}-USD/{asset}-USD"
            vault_cr_series = results[key]['vault_cr_series']
            pool_cr_series = results[key]['pool_cr_series']
            dates = results[key]['dates']

            ax1.plot(dates, vault_cr_series, label=f"{asset} (Vault)", color=colors[asset])
            ax2.plot(dates, pool_cr_series, label=f"{asset} (Pool)", color=colors[asset])

            max_additional_vault = max(max_additional_vault,
                                       max((event[1] for event in results[key]['liquidation_events']), default=0))
            max_additional_pool = max(max_additional_pool,
                                      max((event[2] for event in results[key]['liquidation_events']), default=0))

        # Round up max_additional to the nearest 1000
        max_additional_vault = np.ceil(max_additional_vault / 1000) * 1000
        max_additional_pool = np.ceil(max_additional_pool / 1000) * 1000

        for ax, max_additional, is_vault in [(ax1, max_additional_vault, True), (ax2, max_additional_pool, False)]:
            ax.axhline(y=MINIMAL_CR, color='r', linestyle='--', label='Minimal CR')
            ax.axhline(y=SAFETY_CR, color='g', linestyle='--', label='Safety CR')
            ax.legend(loc='upper left')

            if is_vault:
                ax.set_ylim(1, min(5, max(ax.get_ylim()[1], SAFETY_CR * 1.1)))
                ax.set_ylabel('Vault Collateral Ratio')
                ax_twin = ax.twinx()
                ax_twin.set_ylabel('Additional Vault Collateral Required ($)')
                ax_twin.set_ylim(0, max_additional)
            else:
                ax.set_ylim(1, min(10, max(ax.get_ylim()[1], SAFETY_CR * 1.1)))
                ax.set_ylabel('Pool Collateral Ratio')
                ax_twin = ax.twinx()
                ax_twin.set_ylabel('Additional Pool Collateral Required ($)')
                ax_twin.set_ylim(0, max_additional)

            # Plot liquidation events
            for asset in ['XRP', 'BTC', 'DOGE']:
                key = f"{vault_collateral}-USD/{asset}-USD"
                for event_date, additional_vault, additional_pool, vault_cr, pool_cr in results[key]['liquidation_events']:
                    if is_vault and additional_vault > 0:
                        ax.vlines(x=event_date, ymin=0, ymax=vault_cr, color=colors[asset], linestyle='--', alpha=0.5)
                        ax_twin.vlines(x=event_date, ymin=0, ymax=additional_vault, color=colors[asset], linestyle='--', alpha=0.5)
                        ax_twin.text(event_date, additional_vault, f'+${additional_vault:.0f}',
                                     rotation=90, verticalalignment='bottom', horizontalalignment='right',
                                     color=colors[asset], fontsize=8)
                    elif not is_vault and additional_pool > 0:
                        ax.vlines(x=event_date, ymin=0, ymax=pool_cr, color=colors[asset], linestyle='--', alpha=0.5)
                        ax_twin.vlines(x=event_date, ymin=0, ymax=additional_pool, color=colors[asset], linestyle='--', alpha=0.5)
                        ax_twin.text(event_date, additional_pool, f'+${additional_pool:.0f}',
                                     rotation=90, verticalalignment='bottom', horizontalalignment='right',
                                     color=colors[asset], fontsize=8)

        ax1.set_title(f'Vault Collateral Ratio Dynamics: {vault_collateral} vs XRP, BTC, and DOGE')
        ax2.set_title(f'Pool Collateral Ratio Dynamics: FLR vs XRP, BTC, and DOGE')
        ax2.set_xlabel('Date')

        plt.tight_layout()
        plt.savefig(f'collateral_ratio_{vault_collateral}_vault_and_pool.png')
        plt.close()

def print_results(results):
    """Print results for different strategies."""
    for strategy, metrics in results.items():
        print(f"Strategy: {strategy}")
        print(f"  Initial Vault CR: {metrics['initial_vault_cr']:.2f}")
        print(f"  Initial Pool CR: {metrics['initial_pool_cr']:.2f}")
        print(f"  Initial Vault Collateral: ${metrics['initial_vault_collateral']:.2f}")
        print(f"  Initial Pool Collateral: ${metrics['initial_pool_collateral']:.2f}")
        print(f"  Total liquidation events: {len(metrics['liquidation_events'])}")
        print(f"  Full liquidations: {metrics['full_liquidations']}")
        print(f"  Below minimum events: {metrics['below_minimum_events']}")
        print(f"  Healthy days: {metrics['healthy_days']}")
        print(f"  Total days: {metrics['total_days']}")
        print(f"  Average Vault CR: {metrics['avg_vault_cr']:.2f}")
        print(f"  Average Pool CR: {metrics['avg_pool_cr']:.2f}")
        print(f"  Min Vault CR: {metrics['min_vault_cr']:.2f}")
        print(f"  Min Pool CR: {metrics['min_pool_cr']:.2f}")
        print(f"  Max Vault CR: {metrics['max_vault_cr']:.2f}")
        print(f"  Max Pool CR: {metrics['max_pool_cr']:.2f}")
        print(f"  Total additional vault collateral: ${metrics['total_additional_vault_collateral']:.2f}")
        print(f"  Total additional pool collateral: ${metrics['total_additional_pool_collateral']:.2f}")
        print()

if __name__ == "__main__":
    # Define parameters
    vault_collaterals = ['USDC-USD', 'ETH-USD']
    pool_collateral = 'FLR-USD'
    assets = ['XRP-USD', 'BTC-USD', 'DOGE-USD']
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Fetch FLR data to determine the start date
    flr_data = fetch_historical_data(pool_collateral, '2023-01-01', end_date)
    start_date = flr_data.index[0].strftime('%Y-%m-%d')

    # Run comparison and plot results
    results = compare_strategies(vault_collaterals, pool_collateral, assets, start_date, end_date)
    plot_results(results)
    print_results(results)

    # Determine best strategy
    best_strategy = min(results, key=lambda x: len(results[x]['liquidation_events']))
    print(f"The strategy least prone to liquidations is: {best_strategy}")