import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Technical indicators for stock market analysis"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> tuple:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)

class FeatureEngineer:
    """Main feature engineering class"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = self.indicators.sma(df['Close'], window)
            df[f'EMA_{window}'] = self.indicators.ema(df['Close'], window)
        
        # Price relative to moving averages
        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
        df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['Volatility_MA'] = df['Volatility'].rolling(10).mean()
        
        # Volume features
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def add_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators"""
        df = data.copy()
        
        # RSI
        df['RSI'] = self.indicators.rsi(df['Close'])
        df['RSI_14'] = self.indicators.rsi(df['Close'], 14)
        df['RSI_21'] = self.indicators.rsi(df['Close'], 21)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = bb_upper - bb_lower
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        macd, macd_signal, macd_hist = self.indicators.macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_hist
        
        # Stochastic
        if all(col in df.columns for col in ['High', 'Low']):
            stoch_k, stoch_d = self.indicators.stochastic(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            # Williams %R
            df['Williams_R'] = self.indicators.williams_r(df['High'], df['Low'], df['Close'])
        
        return df
    
    def add_price_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features"""
        df = data.copy()
        
        # Price momentum
        for period in [1, 3, 5, 10]:
            df[f'Price_Change_{period}d'] = df['Close'].pct_change(period)
            df[f'High_Low_Ratio_{period}d'] = df['High'].rolling(period).max() / df['Low'].rolling(period).min()
        
        # Gap analysis
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = df['Gap'] / df['Close'].shift(1)
        
        # Daily range
        df['Daily_Range'] = df['High'] - df['Low']
        df['Daily_Range_Pct'] = df['Daily_Range'] / df['Close']
        
        # Body and wick analysis (candlestick)
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Wick'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Wick'] = np.minimum(df['Open'], df['Close']) - df['Low']
        
        df['Body_Pct'] = df['Body'] / df['Close']
        df['Upper_Wick_Pct'] = df['Upper_Wick'] / df['Close']
        df['Lower_Wick_Pct'] = df['Lower_Wick'] / df['Close']
        
        return df
    
    def add_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators"""
        df = data.copy()
        
        # Trend indicators
        df['Trend_20'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
        df['Trend_50'] = np.where(df['Close'] > df['SMA_50'], 1, -1)
        df['MA_Cross'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        
        # Volatility regime
        vol_ma = df['Volatility'].rolling(50).mean()
        df['Vol_Regime'] = np.where(df['Volatility'] > vol_ma * 1.5, 1, 
                                   np.where(df['Volatility'] < vol_ma * 0.5, -1, 0))
        
        # Price momentum regime
        returns_ma = df['Returns'].rolling(20).mean()
        df['Momentum_Regime'] = np.where(returns_ma > 0.001, 1,
                                        np.where(returns_ma < -0.001, -1, 0))
        
        return df
    
    def add_lag_features(self, data: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features"""
        df = data.copy()
        
        key_features = ['Close', 'Volume', 'RSI', 'Returns']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def create_all_features(self, data: pd.DataFrame, include_lags: bool = False) -> pd.DataFrame:
        """Create all features at once"""
        print("Adding basic features...")
        df = self.add_basic_features(data)
        
        print("Adding advanced features...")
        df = self.add_advanced_features(df)
        
        print("Adding price patterns...")
        df = self.add_price_patterns(df)
        
        print("Adding market regime features...")
        df = self.add_market_regime_features(df)
        
        if include_lags:
            print("Adding lag features...")
            df = self.add_lag_features(df)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        print(f"Feature engineering complete. Shape: {df.shape}")
        print(f"Features created: {len(df.columns) - len(data.columns)} new features")
        
        return df
    
    def get_feature_importance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for feature importance analysis"""
        # Select key features for ML
        feature_columns = [
            'Close', 'Volume', 'Returns',
            'SMA_10', 'SMA_20', 'SMA_50',
            'RSI', 'MACD', 'BB_Width',
            'Volatility', 'Price_SMA20_Ratio',
            'Stoch_K', 'Williams_R',
            'Price_Change_5d', 'Volume_Ratio'
        ]
        
        available_features = [col for col in feature_columns if col in data.columns]
        return data[available_features].dropna()

def process_stock_features(data: pd.DataFrame, advanced: bool = True) -> pd.DataFrame:
    """Convenience function to process stock features"""
    engineer = FeatureEngineer()
    
    if advanced:
        return engineer.create_all_features(data)
    else:
        return engineer.add_basic_features(data)