# Module Extraction Plan

## Module Structure

### Core Modules
- `csp/config.py` - StrategyConfig, get_sp500_tickers()
- `csp/clients.py` - AlpacaClientManager

### Data Layer (`csp/data/`)
- `csp/data/vix.py` - VixDataFetcher
- `csp/data/equity.py` - EquityDataFetcher
- `csp/data/options.py` - OptionsDataFetcher, GreeksCalculator, OptionContract
- `csp/data/models.py` - MarketSnapshot
- `csp/data/manager.py` - DataManager

### Signals Layer (`csp/signals/`)
- `csp/signals/indicators.py` - TechnicalIndicators, EarningsCalendar, DividendCalendar, FomcCalendar
- `csp/signals/equity_filter.py` - EquityFilter, EquityFilterResult
- `csp/signals/options_filter.py` - OptionsFilter, OptionsFilterResult
- `csp/signals/scanner.py` - StrategyScanner, ScanResult

### Trading Layer (`csp/trading/`)
- `csp/trading/models.py` - PositionStatus, ExitReason, ActivePosition, OrderResult, RiskCheckResult
- `csp/trading/portfolio.py` - PortfolioManager, StrategyMetadataStore
- `csp/trading/risk.py` - RiskManager
- `csp/trading/execution.py` - ExecutionEngine
- `csp/trading/loop.py` - TradingLoop

### Main Entry Points
- `csp/main.py` - CLI entry point
- `csp/test_orders.py` - Test functions (replenish_buying_power, test_all_order_types)

## Extraction Order
1. Core modules (config, clients)
2. Data layer (vix, equity, options, models, manager)
3. Signals layer (indicators, equity_filter, options_filter, scanner)
4. Trading layer (models, portfolio, risk, execution, loop)
5. Entry points (main, test_orders)
