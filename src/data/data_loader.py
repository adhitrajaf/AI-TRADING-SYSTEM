import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class StockDataLoader:
    """Handle stock data loading and basic preprocessing"""
    
    def __init__(self, data_path: str = "../../data"):
        self.data_path = data_path
        self.processed_path = os.path.join(data_path, "processed")
        self.raw_path = os.path.join(data_path, "raw")
        
        # Create directories if they don't exist
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.raw_path, exist_ok=True)
    
    def fetch_yfinance_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            if symbol == 'JKSE':
                yf_symbol = '^JKSE'
            else:
                yf_symbol = f"{symbol}.JK"
            
            stock = yf.Ticker(yf_symbol)
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def load_processed_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load processed data from file"""
        filename = f"{symbol}_processed.csv"
        filepath = os.path.join(self.processed_path, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                return data
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
        return None
    
    def load_raw_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load raw data from file"""
        filename = f"{symbol}_raw.csv"
        filepath = os.path.join(self.raw_path, filename)
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                return data
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
        return None
    
    def save_data(self, data: pd.DataFrame, symbol: str, data_type: str = "processed"):
        """Save data to file"""
        if data_type == "processed":
            filepath = os.path.join(self.processed_path, f"{symbol}_processed.csv")
        else:
            filepath = os.path.join(self.raw_path, f"{symbol}_raw.csv")
        
        data.to_csv(filepath)
        print(f"Data saved: {filepath}")
    
    def get_latest_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Get latest data with technical indicators"""
        try:
            data = self.fetch_yfinance_data(symbol, period=f"{days}d")
            
            # Add basic technical indicators
            data['Returns'] = data['Close'].pct_change()
            data['SMA_10'] = data['Close'].rolling(10).mean()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Volatility
            data['Volatility'] = data['Returns'].rolling(20).std()
            
            return data.dropna()
            
        except Exception as e:
            raise Exception(f"Error getting latest data for {symbol}: {str(e)}")
    
    def load_all_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load all stock data"""
        stock_data = {}
        
        for symbol in symbols:
            # Try processed first, then raw
            data = self.load_processed_data(symbol)
            if data is None:
                data = self.load_raw_data(symbol)
            
            if data is not None:
                stock_data[symbol] = data
                print(f"Loaded {symbol}: {len(data)} records")
            else:
                print(f"No data found for {symbol}")
        
        return stock_data

def get_stock_data(symbols: List[str], data_path: str = "../../data") -> Dict[str, pd.DataFrame]:
    """Convenience function to get stock data"""
    loader = StockDataLoader(data_path)
    return loader.load_all_stocks(symbols)