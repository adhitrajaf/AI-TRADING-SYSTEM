# AI Trading System for Indonesian Stock Market

An advanced LSTM-based trading system with integrated risk management for Indonesian stocks (IDX). This system provides real-time trading signals, comprehensive market analysis, and risk assessment tools through a professional web interface.

## Key Features

- **Enhanced LSTM Model**: 18+ technical indicators with advanced feature engineering
- **Real-time Trading Signals**: Buy/Sell/Hold recommendations with confidence levels
- **Risk Management**: Dynamic position sizing, stop-loss, and take-profit calculations
- **Professional UI**: Multi-page Streamlit dashboard with interactive charts
- **REST API**: FastAPI backend with comprehensive endpoints
- **Indonesian Market Focus**: Optimized for IDX stocks (JKSE, BBCA, BBRI, TLKM, ASII, UNVR)

## Performance Highlights

- **Enhanced Model MAPE**: 3.43% (vs 5.22% basic model)
- **Win Rate**: 64.5% with integrated risk management
- **Directional Accuracy**: 70%+ across tested stocks
- **Sharpe Ratio**: 1.2+ for risk-adjusted returns

## Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **Frontend**: Streamlit with Plotly visualizations
- **ML Framework**: TensorFlow/Keras, scikit-learn
- **Data**: Yahoo Finance API for real-time data
- **Architecture**: LSTM Neural Networks with advanced technical analysis

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1. Start the API Backend

```bash
cd src/api
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Launch the Dashboard

```bash
cd src/app
streamlit run streamlit.py
```

The dashboard will open at `http://localhost:8501`

## API Endpoints

### Market Data
- `GET /` - Health check and system status
- `GET /stocks` - Available stocks list
- `GET /market-overview` - Real-time market overview

### Predictions & Analysis
- `POST /predict` - Generate AI price predictions
- `GET /stock/{symbol}/analysis` - Comprehensive stock analysis
- `POST /trading-signals` - Generate trading signals with risk management

### Example Usage

```python
import requests

# Get market overview
response = requests.get("http://localhost:8000/market-overview")
market_data = response.json()

# Generate trading signals
payload = {
    "symbol": "BBCA",
    "analysis_type": "enhanced",
    "days": 30
}
signals = requests.post("http://localhost:8000/trading-signals", json=payload)
```

## Dashboard Features

### 1. Market Overview
- Real-time prices for all tracked stocks
- AI signal summaries with confidence levels
- Interactive price charts

### 2. Stock Analysis
- Individual stock deep-dive analysis
- Technical indicators (RSI, MACD, Bollinger Bands)
- AI predictions with confidence intervals

### 3. AI Predictions
- Multi-day price forecasting
- Trend analysis and pattern recognition
- Prediction accuracy metrics

### 4. Trading Signals
- Advanced buy/sell/hold recommendations
- Risk scoring (0-10 scale)
- Position sizing suggestions
- Stop-loss and take-profit levels

## Model Architecture

### Enhanced LSTM Features
- **Basic Indicators**: Close, Volume, SMA, RSI, Volatility
- **Advanced Features**: Bollinger Bands, MACD, Price Momentum
- **Risk Metrics**: Volatility analysis, trend scoring

### Model Performance
```
Stock Performance (MAPE):
- Astra International (ASII): 5.22%
- Telkom Indonesia (TLKM): 7.21%
- Bank Rakyat Indonesia (BBRI): 8.19%
- Enhanced Model Average: 3.43%
```

## Development Structure

```
AI-TRADING-SYSTEM/
├── src/
│   ├── api/                 # FastAPI backend
│   │   └── main.py         # API server
│   └── app/                # Streamlit frontend
│       └── streamlit.py    # Dashboard app
├── models/                 # ML models
│   ├── lstm_model.py      # Core LSTM implementation
│   └── enhanced_trading_advisor.py  # Enhanced model
├── notebooks/             # Jupyter notebooks
│   └── ModelTraining.ipynb  # Training pipeline
├── data/                  # Data storage
├── reports/               # Analysis reports
└── requirements.txt       # Dependencies
```

## Configuration

### Environment Variables
Create a `.env` file for configuration:
```
API_HOST=127.0.0.1
API_PORT=8000
STREAMLIT_PORT=8501
DATA_UPDATE_INTERVAL=300
```

### Supported Stocks
- **JKSE**: IDX Composite
- **BBCA**: Bank Central Asia
- **BBRI**: Bank Rakyat Indonesia
- **TLKM**: Telkom Indonesia
- **ASII**: Astra International
- **UNVR**: Unilever Indonesia

## Risk Management Features

### Position Sizing
- Dynamic position sizing based on volatility
- Risk score integration (0-10 scale)
- Confidence-adjusted recommendations

### Stop Loss & Take Profit
- Volatility-based stop loss calculation
- Risk-reward ratio optimization
- Market condition adaptation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Always conduct your own research and consider consulting with financial advisors before making investment decisions.

## Contact

- Repository: [AI-TRADING-SYSTEM](https://github.com/adhitrajaf/AI-TRADING-SYSTEM)
- Issues: [GitHub Issues](https://github.com/adhitrajaf/AI-TRADING-SYSTEM/issues)

---

**Built with Python, TensorFlow, FastAPI, and Streamlit**
