"""
Model evaluation utilities
File: src/evaluation/model_evaluator.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelEvaluator:
    """Comprehensive model evaluation for financial forecasting"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic regression metrics
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)
        
        # Financial-specific metrics
        directional_accuracy = self._directional_accuracy(actual, predicted)
        hit_rate = self._hit_rate(actual, predicted, threshold=0.02)
        max_error = np.max(np.abs(actual - predicted))
        
        # Custom financial metrics
        profit_loss = self._calculate_profit_loss(actual, predicted)
        sharpe_ratio = self._calculate_sharpe_ratio(actual, predicted)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy,
            'Hit_Rate': hit_rate,
            'Max_Error': max_error,
            'Profit_Loss': profit_loss,
            'Sharpe_Ratio': sharpe_ratio
        }
    
    def _directional_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate directional accuracy (up/down prediction)"""
        if len(actual) < 2 or len(predicted) < 2:
            return 0.0
        
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        
        correct_predictions = np.sum(actual_direction == predicted_direction)
        total_predictions = len(actual_direction)
        
        return (correct_predictions / total_predictions) * 100
    
    def _hit_rate(self, actual: np.ndarray, predicted: np.ndarray, threshold: float = 0.02) -> float:
        """Calculate hit rate within threshold"""
        percentage_error = np.abs((actual - predicted) / actual)
        hits = np.sum(percentage_error <= threshold)
        return (hits / len(actual)) * 100
    
    def _calculate_profit_loss(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate simulated profit/loss based on predictions"""
        if len(actual) < 2:
            return 0.0
        
        # Simple strategy: buy if price predicted to go up
        returns = np.diff(actual) / actual[:-1]
        predictions = np.diff(predicted) / predicted[:-1]
        
        # Trading signal: 1 if predicted positive return, -1 if negative
        signals = np.where(predictions > 0, 1, -1)
        
        # Calculate strategy returns
        strategy_returns = signals * returns
        cumulative_return = np.prod(1 + strategy_returns) - 1
        
        return cumulative_return * 100
    
    def _calculate_sharpe_ratio(self, actual: np.ndarray, predicted: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for trading strategy"""
        if len(actual) < 2:
            return 0.0
        
        returns = np.diff(actual) / actual[:-1]
        predictions = np.diff(predicted) / predicted[:-1]
        signals = np.where(predictions > 0, 1, -1)
        strategy_returns = signals * returns
        
        if np.std(strategy_returns) == 0:
            return 0.0
        
        excess_return = np.mean(strategy_returns) - risk_free_rate / 252  # Daily risk-free rate
        sharpe = excess_return / np.std(strategy_returns) * np.sqrt(252)  # Annualized
        
        return sharpe
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model") -> Dict:
        """Comprehensive model evaluation"""
        
        # Generate predictions
        predictions = model.predict(X_test)
        
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(y_test.shape) > 1:
            y_test = y_test.flatten()
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, predictions)
        
        # Store in history
        self.metrics_history[model_name] = {
            'metrics': metrics,
            'actual': y_test,
            'predicted': predictions,
            'timestamp': pd.Timestamp.now()
        }
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'actual': y_test
        }
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models"""
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                **metrics
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.round(4)
    
    def plot_predictions(self, actual: np.ndarray, predicted: np.ndarray, 
                        model_name: str = "Model", save_path: Optional[str] = None):
        """Plot actual vs predicted values"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted', 'Residuals', 'Error Distribution', 'Time Series'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Actual vs Predicted scatter
        fig.add_trace(
            go.Scatter(x=actual, y=predicted, mode='markers', name='Predictions',
                      marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Residuals
        residuals = actual - predicted
        fig.add_trace(
            go.Scatter(y=residuals, mode='markers', name='Residuals',
                      marker=dict(color='green', opacity=0.6)),
            row=1, col=2
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Error Distribution', nbinsx=30),
            row=2, col=1
        )
        
        # Time series
        time_index = np.arange(len(actual))
        fig.add_trace(
            go.Scatter(x=time_index, y=actual, mode='lines', name='Actual',
                      line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_index, y=predicted, mode='lines', name='Predicted',
                      line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=f"{model_name} - Evaluation Results")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_metrics_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot model comparison metrics"""
        
        key_metrics = ['RMSE', 'MAPE', 'Directional_Accuracy', 'R2']
        available_metrics = [m for m in key_metrics if m in comparison_df.columns]
        
        if not available_metrics:
            print("No key metrics found for plotting")
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=available_metrics,
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, metric in enumerate(available_metrics[:4]):
            row, col = positions[i]
            
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], y=comparison_df[metric], name=metric),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Model Performance Comparison", showlegend=False)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_evaluation_report(self, model_name: str, save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        
        if model_name not in self.metrics_history:
            return f"No evaluation data found for {model_name}"
        
        data = self.metrics_history[model_name]
        metrics = data['metrics']
        
        report = f"""
# Model Evaluation Report: {model_name}
Generated: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics

### Regression Metrics
- **RMSE**: {metrics['RMSE']:.4f}
- **MAE**: {metrics['MAE']:.4f}
- **MAPE**: {metrics['MAPE']:.2f}%
- **R²**: {metrics['R2']:.4f}

### Financial Metrics
- **Directional Accuracy**: {metrics['Directional_Accuracy']:.2f}%
- **Hit Rate (2% threshold)**: {metrics['Hit_Rate']:.2f}%
- **Simulated P&L**: {metrics['Profit_Loss']:.2f}%
- **Sharpe Ratio**: {metrics['Sharpe_Ratio']:.4f}

### Error Analysis
- **Maximum Error**: {metrics['Max_Error']:.4f}

## Model Performance Assessment

### Accuracy Assessment
"""
        
        # Add performance assessment
        if metrics['MAPE'] < 3:
            report += "- **Excellent** prediction accuracy (MAPE < 3%)\n"
        elif metrics['MAPE'] < 5:
            report += "- **Good** prediction accuracy (MAPE < 5%)\n"
        elif metrics['MAPE'] < 10:
            report += "- **Acceptable** prediction accuracy (MAPE < 10%)\n"
        else:
            report += "- **Poor** directional prediction capability\n"
        
        # Trading performance assessment
        if metrics['Profit_Loss'] > 10:
            report += "- **Excellent** simulated trading performance\n"
        elif metrics['Profit_Loss'] > 5:
            report += "- **Good** simulated trading performance\n"
        elif metrics['Profit_Loss'] > 0:
            report += "- **Positive** simulated trading performance\n"
        else:
            report += "- **Negative** simulated trading performance\n"
        
        # Risk assessment
        if metrics['Sharpe_Ratio'] > 1.0:
            report += "- **Good** risk-adjusted returns\n"
        elif metrics['Sharpe_Ratio'] > 0.5:
            report += "- **Acceptable** risk-adjusted returns\n"
        else:
            report += "- **Poor** risk-adjusted returns\n"
        
        report += """
## Recommendations

### Model Deployment
"""
        
        # Deployment recommendations
        if metrics['MAPE'] < 5 and metrics['Directional_Accuracy'] > 55:
            report += "- **APPROVED** for production deployment\n"
            report += "- Model shows good prediction accuracy and directional capability\n"
        elif metrics['MAPE'] < 10 and metrics['Directional_Accuracy'] > 50:
            report += "- **CONDITIONAL** approval for deployment\n"
            report += "- Consider additional validation and risk management\n"
        else:
            report += "- **NOT RECOMMENDED** for deployment\n"
            report += "- Model requires further improvement\n"
        
        report += """
### Improvement Suggestions
- Monitor model performance regularly
- Consider ensemble methods if accuracy is insufficient
- Implement robust risk management strategies
- Regular retraining with new data

---
*Report generated by AI Trading System Evaluator*
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def backtest_trading_strategy(self, actual_prices: np.ndarray, predicted_prices: np.ndarray,
                                 initial_capital: float = 10000) -> Dict[str, float]:
        """Backtest a simple trading strategy based on predictions"""
        
        if len(actual_prices) < 2 or len(predicted_prices) < 2:
            return {'error': 'Insufficient data for backtesting'}
        
        # Calculate returns
        actual_returns = np.diff(actual_prices) / actual_prices[:-1]
        predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]
        
        # Simple strategy: buy if predicted return > 0, sell if < 0
        positions = np.where(predicted_returns > 0, 1, -1)  # 1 = long, -1 = short
        
        # Calculate portfolio value over time
        portfolio_returns = positions * actual_returns
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        final_value = initial_capital * cumulative_returns[-1]
        
        # Calculate metrics
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Win rate
        winning_trades = np.sum(portfolio_returns > 0)
        total_trades = len(portfolio_returns)
        win_rate = winning_trades / total_trades * 100
        
        # Maximum drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Sharpe ratio
        if np.std(portfolio_returns) > 0:
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'win_rate_pct': win_rate,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'winning_trades': winning_trades
        }

def evaluate_model_performance(model, X_test, y_test, model_name="Model") -> Dict:
    """Convenience function for model evaluation"""
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X_test, y_test, model_name)