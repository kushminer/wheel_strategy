# 1. Fetch historical close prices
# 2. Generate daily technical indicators
# 3. Filter daily tickers by technical indicators (booleans)
# 4. Extract entry dates (YYYY-MM-DD strings where value = True)

import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_close_prices(
    tickers: list,
    start_date: datetime,
    end_date: datetime = None,
    progress: bool = True
) -> pd.DataFrame:
    """
    Fetch historical close prices for given tickers from yfinance.
    
    Args:
        tickers: List of ticker symbols (e.g., ['SPY', 'QQQ'])
        start_date: Start date for data retrieval (datetime object)
        end_date: End date for data retrieval (datetime object). 
                 If None, uses current date
        progress: Whether to show download progress bar (default: True)
    
    Returns:
        DataFrame with close prices, indexed by date, with columns for each ticker
    """
    if end_date is None:
        end_date = datetime.now()
    
    print(f"Fetching data for {tickers} from {start_date.date()} to {end_date.date()}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=progress)
    
    return data['Close']

def generate_dict_technical_indicators(
    historic_close_data: pd.DataFrame,
    configs_technical_indicators: dict = None,
    window: int = 20,
    std_dev: float = 2.0,
    dropna: bool = True,
    print_sample: bool = False
) -> dict:
    """
    Calculate technical indicators for each symbol with multiple configurations.
    
    Args:
        historic_close_data: DataFrame with price data, each column represents the close price of a symbol
        configs_technical_indicators: Dictionary of configurations. If provided, returns nested dict structure.
                                      Format: {'config_name': {'window': int, 'std_dev': float}, ...}
                                      If None, uses single config with window/std_dev parameters.
        window: Rolling window size for SMA and standard deviation (default: 20, used if configs_technical_indicators is None)
        std_dev: Number of standard deviations for Bollinger Bands (default: 2.0, used if configs_technical_indicators is None)
        dropna: Whether to drop rows with NaN values (default: True)
        print_sample: Whether to print sample results (default: False)
    
    Returns:
        If configs_technical_indicators is provided:
            Nested dictionary: {config_name: {symbol: DataFrame}}
        Otherwise:
            Dictionary: {symbol: DataFrame}
            
        Each DataFrame contains:
          - sma_{window}: Simple Moving Average
          - std_{window}: Standard Deviation
          - bb_upper: Upper Bollinger Band
          - bb_lower: Lower Bollinger Band
    """
    def _calculate_single_config(close_data, window, std_dev, dropna):
        """Helper function to calculate indicators for a single configuration."""
        result_dict = {}
        
        # Loop through each column (symbol)
        for symbol in close_data.columns:
            close_series = close_data[symbol]
            
            # Create DataFrame for this symbol's technical indicators
            tech_df = pd.DataFrame(index=close_data.index)
            
            # Calculate rolling statistics
            roll = close_series.rolling(window=window, min_periods=window)
            tech_df[f'sma_{window}'] = roll.mean()
            tech_df[f'std_{window}'] = roll.std(ddof=0)
            
            # Calculate Bollinger Bands
            sma_col = f'sma_{window}'
            std_col = f'std_{window}'
            tech_df['bb_upper'] = tech_df[sma_col] + std_dev * tech_df[std_col]
            tech_df['bb_lower'] = tech_df[sma_col] - std_dev * tech_df[std_col]
                    
            # Drop NaN values if requested
            if dropna:
                tech_df = tech_df.dropna()
            
            # Store in dictionary
            result_dict[symbol] = tech_df
        
        return result_dict
    
    # If configs provided, calculate for each config and return nested structure
    if configs_technical_indicators is not None:
        daily_tech_indicators_dict = {}
        for config_name, params in configs_technical_indicators.items():
            daily_tech_indicators_dict[config_name] = _calculate_single_config(
                historic_close_data,
                window=params['window'],
                std_dev=params['std_dev'],
                dropna=dropna
            )
        
        # Print sample if requested
        if print_sample:
            print("Technical indicators dictionary (nested structure):")
            print(f"Configs in dictionary: {list(daily_tech_indicators_dict.keys())}")
            for config_name, symbols_dict in daily_tech_indicators_dict.items():
                print(f"\n{config_name}:")
                print(f"  Symbols: {list(symbols_dict.keys())}")
                for symbol, tech_df in symbols_dict.items():
                    print(f"\n  '{symbol}' technical indicators:")
                    print(tech_df.head(10))
                    print(f"  Columns: {list(tech_df.columns)}")
        
        return daily_tech_indicators_dict
    else:
        # Single config mode (backward compatibility)
        result_dict = _calculate_single_config(historic_close_data, window, std_dev, dropna)
        return result_dict

def generate_dict_technical_indicator_filter(
    daily_tech_indicators_dict: dict,
    historic_close_data: pd.DataFrame,
    configs_technical_indicator_filter: dict = None
) -> dict:
    """
    Filter daily tickers based on technical indicator conditions for multiple configurations.
    Applies each filter config to each technical indicator config.
    
    Args:
        daily_tech_indicators_dict: Nested dictionary of technical indicators 
                                    (key=tech_config_name, value={symbol: DataFrame})
        historic_close_data: DataFrame with close prices for each symbol
        configs_technical_indicator_filter: Dictionary mapping filter config names to filter condition functions.
                                            If None, uses default configs:
                       - config_1: close <= bb_upper
                       - config_2: close <= bb_lower
                       - config_3: (close.shift(-1) <= bb_lower.shift(-1)) & (close >= bb_lower) - bounce from lower band
    
    Returns:
        Nested dictionary with three levels:
        - Top level keys: technical indicator config names
        - Second level keys: filter config names
        - Third level keys: symbol names
        - Values: filtered DataFrames
    """
    # Default filter configurations
    if configs_technical_indicator_filter is None:
        configs_technical_indicator_filter = {
            'config_1': lambda df: df['close'] <= df['bb_upper'],
            'config_2': lambda df: df['close'] <= df['bb_lower'],
            'config_3': lambda df: (df['close'].shift(-1) <= df['bb_lower'].shift(-1)) & (df['close'] >= df['bb_lower']),
        }
    
    filtered_daily_tickers = {}
    
    # Loop through each technical indicator configuration
    for tech_config_name, symbols_dict in daily_tech_indicators_dict.items():
        filtered_daily_tickers[tech_config_name] = {}
        
        # Apply each filter config to this technical indicator config
        for filter_config_name, filter_func in configs_technical_indicator_filter.items():
            filtered_daily_tickers[tech_config_name][filter_config_name] = {}
            
            # Loop through each symbol in this technical indicator configuration
            for symbol, tech_df in symbols_dict.items():
                # Get close prices for this symbol
                close_series = historic_close_data[symbol]
                
                # Get the window size from the tech_df columns (e.g., 'sma_20' -> 20)
                sma_cols = [col for col in tech_df.columns if col.startswith('sma_')]
                if sma_cols:
                    window = int(sma_cols[0].split('_')[1])
                    sma_col = f'sma_{window}'
                else:
                    sma_col = 'sma_20'  # fallback
                
                # Combine close prices with technical indicators (align on index)
                df_equity_entry = pd.DataFrame({
                    'close': close_series,
                    sma_col: tech_df[sma_col],
                    'bb_upper': tech_df['bb_upper'],
                    'bb_lower': tech_df['bb_lower']
                })
                
                # Apply the filter condition for this filter config
                df_equity_entry['filter_entry'] = filter_func(df_equity_entry)
                
                # Filter to only include rows where filter_entry is True
                df_filtered = df_equity_entry[df_equity_entry['filter_entry'] == True].copy()
                
                # Drop NaN values
                df_filtered = df_filtered.dropna()
                
                # Store in dictionary
                filtered_daily_tickers[tech_config_name][filter_config_name][symbol] = df_filtered
                
                # Optional: Print summary
                print(f"{tech_config_name} | {filter_config_name} | {symbol}: {len(df_filtered)} entries pass filter (out of {len(df_equity_entry.dropna())} total)")
    
    return filtered_daily_tickers

def extract_entry_dates(filtered_daily_tickers_dict: dict, as_string: bool = False) -> dict:
    """
    Extract entry dates (next market day after filter dates) for each symbol and configuration.
    
    Args:
        filtered_daily_tickers_dict: Nested dictionary with three levels:
                                     - Top level keys: technical indicator config names
                                     - Second level keys: filter config names
                                     - Third level keys: symbol names
                                     - Values: filtered DataFrames (already filtered to filter_entry == True)
        as_string: If True, return dates as strings (YYYY-MM-DD format)
                   If False, return dates as pandas Timestamps
    
    Returns:
        Nested dictionary with three levels:
        - Top level keys: technical indicator config names
        - Second level keys: filter config names
        - Third level keys: symbol names
        - Values: lists of entry dates (next market day after filter dates)
    """
    entry_dates_dict = {}
    
    # Loop through each technical indicator configuration
    for tech_config_name, filter_configs_dict in filtered_daily_tickers_dict.items():
        entry_dates_dict[tech_config_name] = {}
        
        # Loop through each filter configuration
        for filter_config_name, symbols_dict in filter_configs_dict.items():
            entry_dates_dict[tech_config_name][filter_config_name] = {}
            
            # Loop through each symbol in this configuration
            for symbol, df_filtered in symbols_dict.items():
                # Extract dates from the index and shift forward by 1 business day
                # Entry date is the market day AFTER the filter date
                dates_series = pd.to_datetime(df_filtered.index)
                entry_dates = dates_series + pd.offsets.BDay(1)
                dates_list = entry_dates.tolist()
                
                # Convert to date strings if requested
                if as_string:
                    dates_list = [pd.Timestamp(date).strftime('%Y-%m-%d') for date in dates_list]
                
                entry_dates_dict[tech_config_name][filter_config_name][symbol] = dates_list
    
    return entry_dates_dict