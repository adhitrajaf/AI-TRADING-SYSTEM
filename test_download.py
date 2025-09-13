# fixed_test_download.py - Quick fix for format error
import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def test_single_stock(symbol):
    """Test download single stock - FIXED VERSION"""
    print(f"Testing download for {symbol}...")
    try:
        # Download data - fix the auto_adjust warning
        data = yf.download(symbol, start="2020-01-01", end="2024-12-31", 
                          progress=False, auto_adjust=True)
        
        if data.empty: # type: ignore
            print(f"❌ No data found for {symbol}")
            return False
        
        # Basic info - FIX the format error
        print(f"✅ Success! Downloaded {len(data)} records") # type: ignore
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}") # type: ignore
        print(f"Columns: {list(data.columns)}") # type: ignore
        
        # Fix the latest price format issue
        try:
            latest_price = data['Close'].iloc[-1] # type: ignore
            if pd.isna(latest_price):
                print(f"Latest close price: N/A")
            else:
                print(f"Latest close price: {float(latest_price):.2f}")
        except:
            print(f"Latest close price: Data available")
            
        print("-" * 50)
        
        return data
    
    except Exception as e:
        print(f"❌ Error downloading {symbol}: {e}")
        return False

def save_and_continue():
    """Quick save and continue to EDA"""
    
    test_symbols = {
        '^JKSE': 'IDX Composite',
        'BBCA.JK': 'Bank Central Asia',
        'BBRI.JK': 'Bank Rakyat Indonesia',
        'TLKM.JK': 'Telkom Indonesia',
        'ASII.JK': 'Astra International',
        'UNVR.JK': 'Unilever Indonesia'
    }
    
    print("🚀 Fixed version - downloading and saving all data...\n")
    
    results = {}
    for symbol, name in test_symbols.items():
        print(f"📊 {name} ({symbol})")
        data = test_single_stock(symbol)
        if data is not False:
            results[symbol] = data
            
            # Save immediately
            clean_symbol = symbol.replace('^', 'JKSE').replace('.JK', '')
            filename = f'data/raw/{clean_symbol}_raw.csv'
            data.to_csv(filename) # type: ignore
            print(f"💾 Saved to {filename}")
    
    print(f"\n🎉 TOTAL SUCCESS: {len(results)} stocks downloaded and saved!")
    print("\n✅ Files created in data/raw/:")
    for symbol in results.keys():
        clean_symbol = symbol.replace('^', 'JKSE').replace('.JK', '')
        print(f"   - {clean_symbol}_raw.csv")
    
    print("\n🚀 READY FOR EDA! Run the Jupyter notebook now!")
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("    FIXED FINANCIAL FORECASTING - DATA TEST")
    print("=" * 60)
    
    # Create data/raw directory if not exists
    os.makedirs('data/raw', exist_ok=True)
    
    results = save_and_continue()