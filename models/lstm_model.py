import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
import os

class StockLSTMModel:
    def __init__(self, sequence_length=60, features=['Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI']):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.history = None
        
    def prepare_data(self, data, target_col='Close'):
        """Prepare data for LSTM training"""
        # Select features
        feature_data = data[self.features].fillna(method='ffill').dropna()
        target_data = data[target_col].fillna(method='ffill').dropna()
        
        # Align data
        min_len = min(len(feature_data), len(target_data))
        feature_data = feature_data.iloc[:min_len]
        target_data = target_data.iloc[:min_len]
        
        # Scale features and target
        scaled_features = self.scaler.fit_transform(feature_data)
        scaled_target = self.target_scaler.fit_transform(target_data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_target[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(25),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001), # type: ignore
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the LSTM model"""
        if self.model is None:
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6, monitor='val_loss')
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.history = self.model.fit( # type: ignore
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1 # type: ignore
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        scaled_predictions = self.model.predict(X) # type: ignore
        predictions = self.target_scaler.inverse_transform(scaled_predictions)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test_actual):
        """Evaluate model performance with multiple metrics"""
        predictions = self.predict(X_test)
        
        # Convert scaled y_test back to actual values
        y_actual = self.target_scaler.inverse_transform(y_test_actual.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, predictions)
        mape = mean_absolute_percentage_error(y_actual, predictions) * 100
        
        # Additional financial metrics
        directional_accuracy = np.mean(np.sign(np.diff(y_actual)) == np.sign(np.diff(predictions))) * 100
        
        # Pin Bar Loss (custom metric for trend prediction)
        def calculate_pin_bar_loss(actual, pred):
            actual_change = np.diff(actual)
            pred_change = np.diff(pred)
            return np.mean(np.abs(actual_change - pred_change))
        
        pin_bar_loss = calculate_pin_bar_loss(y_actual, predictions)
        
        # PPIC (Price Prediction Inconsistency Coefficient)
        def calculate_ppic(actual, pred):
            actual_volatility = np.std(np.diff(actual))
            pred_volatility = np.std(np.diff(pred))
            return abs(actual_volatility - pred_volatility) / actual_volatility
        
        ppic = calculate_ppic(y_actual, predictions)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Pin_Bar_Loss': pin_bar_loss,
            'PPIC': ppic
        }
        
        return metrics, predictions, y_actual
    
    def save_model(self, filepath):
        """Save model and scalers"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        if self.model is not None:
            self.model.save(f"{filepath}_model.h5") # type: ignore
        else:
            raise ValueError("Model is not trained or loaded. Cannot save None model.")
        
        # Save scalers
        joblib.dump(self.scaler, f"{filepath}_feature_scaler.pkl")
        joblib.dump(self.target_scaler, f"{filepath}_target_scaler.pkl")
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model and scalers"""
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        self.scaler = joblib.load(f"{filepath}_feature_scaler.pkl")
        self.target_scaler = joblib.load(f"{filepath}_target_scaler.pkl")
        
        print(f"Model loaded from {filepath}")

# Stationarity test functions
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(timeseries, title):
    """Test for stationarity using ADF and KPSS tests"""
    print(f'\n=== Stationarity Test for {title} ===')
    
    # ADF Test
    adf_result = adfuller(timeseries.dropna())
    print(f'ADF Test Results:')
    print(f'  ADF Statistic: {adf_result[0]:.6f}')
    print(f'  p-value: {adf_result[1]:.6f}')
    print(f'  Critical Values:')
    for key, value in adf_result[4].items(): # type: ignore
        print(f'    {key}: {value:.6f}')
    
    adf_stationary = adf_result[1] <= 0.05
    print(f'  ADF Test Result: {"Stationary" if adf_stationary else "Non-Stationary"}')
    
    # KPSS Test
    try:
        kpss_result = kpss(timeseries.dropna())
        print(f'\nKPSS Test Results:')
        print(f'  KPSS Statistic: {kpss_result[0]:.6f}')
        print(f'  p-value: {kpss_result[1]:.6f}')
        print(f'  Critical Values:')
        for key, value in kpss_result[3].items():
            print(f'    {key}: {value:.6f}')
        
        kpss_stationary = kpss_result[1] >= 0.05
        print(f'  KPSS Test Result: {"Stationary" if kpss_stationary else "Non-Stationary"}')
        
        # Combined result
        if adf_stationary and kpss_stationary:
            conclusion = "STATIONARY"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "NON-STATIONARY"
        else:
            conclusion = "TREND-STATIONARY or UNCERTAIN"
            
        print(f'\nCombined Conclusion: {conclusion}')
        return conclusion
        
    except Exception as e:
        print(f'KPSS Test failed: {e}')
        return "Stationary" if adf_stationary else "Non-Stationary"

# Trading Signal Generator
class TradingAdvisor:
    def __init__(self, model, lookback_period=20):
        self.model = model
        self.lookback_period = lookback_period
        
    def generate_signals(self, current_data, predictions):
        """Generate buy/sell signals based on predictions and technical indicators"""
        signals = []
        
        for i, pred in enumerate(predictions):
            if i < len(current_data) - 1:
                current_price = current_data.iloc[i]['Close']
                predicted_price = pred
                
                # Calculate percentage change
                price_change_pct = (predicted_price - current_price) / current_price * 100
                
                # RSI condition
                rsi = current_data.iloc[i]['RSI']
                
                # Moving average conditions
                sma_20 = current_data.iloc[i]['SMA_20']
                sma_50 = current_data.iloc[i]['SMA_50']
                
                # Volume condition
                volume_ma = current_data.iloc[i]['Volume_MA']
                current_volume = current_data.iloc[i]['Volume']
                
                # Signal generation logic
                signal = "HOLD"
                confidence = 0.5
                reasoning = []
                
                # Price prediction signal
                if price_change_pct > 2:  # Predicted to rise more than 2%
                    signal = "BUY"
                    confidence += 0.2
                    reasoning.append(f"Predicted price increase: {price_change_pct:.2f}%")
                elif price_change_pct < -2:  # Predicted to fall more than 2%
                    signal = "SELL"
                    confidence += 0.2
                    reasoning.append(f"Predicted price decrease: {price_change_pct:.2f}%")
                
                # RSI conditions
                if rsi < 30 and signal != "SELL":
                    signal = "BUY"
                    confidence += 0.15
                    reasoning.append(f"Oversold condition (RSI: {rsi:.1f})")
                elif rsi > 70 and signal != "BUY":
                    signal = "SELL"
                    confidence += 0.15
                    reasoning.append(f"Overbought condition (RSI: {rsi:.1f})")
                
                # Moving average trend
                if current_price > sma_20 > sma_50 and signal != "SELL":
                    confidence += 0.1
                    reasoning.append("Positive MA trend")
                elif current_price < sma_20 < sma_50 and signal != "BUY":
                    confidence += 0.1
                    reasoning.append("Negative MA trend")
                
                # Volume confirmation
                if current_volume > volume_ma * 1.2:
                    confidence += 0.05
                    reasoning.append("High volume confirmation")
                
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
                
                signals.append({
                    'Date': current_data.index[i],
                    'Current_Price': current_price,
                    'Predicted_Price': predicted_price,
                    'Price_Change_Pct': price_change_pct,
                    'Signal': signal,
                    'Confidence': confidence,
                    'RSI': rsi,
                    'Reasoning': '; '.join(reasoning)
                })
        
        return pd.DataFrame(signals)

# Visualization functions
def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
    if 'val_mae' in history.history:
        ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()

def plot_predictions(actual, predicted, dates=None, title="Actual vs Predicted", save_path=None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(15, 6))
    
    x_axis = dates if dates is not None else range(len(actual))
    
    plt.plot(x_axis, actual, label='Actual', color='blue', alpha=0.7)
    plt.plot(x_axis, predicted, label='Predicted', color='red', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price (IDR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if dates is not None:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
    
    plt.show()