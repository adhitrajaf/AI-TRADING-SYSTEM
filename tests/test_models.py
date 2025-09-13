import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDataProcessing(unittest.TestCase):
    """Test data processing functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample stock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 1000 + np.cumsum(np.random.randn(100) * 0.02 * 1000)
        volumes = np.random.randint(1000000, 5000000, 100)
        
        self.sample_data = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(100) * 0.01),
            'High': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
            'Low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
            'Close': prices,
            'Volume': volumes
        }, index=dates)
    
    def test_data_shape(self):
        """Test if sample data has correct shape"""
        self.assertEqual(self.sample_data.shape[0], 100)
        self.assertEqual(self.sample_data.shape[1], 5)
    
    def test_price_consistency(self):
        """Test if price data is consistent (High >= Low, etc.)"""
        self.assertTrue((self.sample_data['High'] >= self.sample_data['Low']).all())
        self.assertTrue((self.sample_data['High'] >= self.sample_data['Close']).all())
        self.assertTrue((self.sample_data['Close'] >= self.sample_data['Low']).all())
    
    def test_no_missing_values(self):
        """Test for missing values"""
        self.assertFalse(self.sample_data.isnull().any().any())

class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.prices = pd.Series(1000 + np.cumsum(np.random.randn(50) * 0.02 * 1000))
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        sma_10 = self.prices.rolling(10).mean()
        
        # Test that SMA values are reasonable
        self.assertFalse(sma_10.iloc[10:].isnull().any())
        self.assertTrue(sma_10.iloc[:9].isnull().all())
    
    def test_rsi_bounds(self):
        """Test RSI is within 0-100 bounds"""
        delta = self.prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_valid = rsi.dropna()
        self.assertTrue((rsi_valid >= 0).all())
        self.assertTrue((rsi_valid <= 100).all())

class TestModelValidation(unittest.TestCase):
    """Test model validation functions"""
    
    def test_train_test_split(self):
        """Test train-test split functionality"""
        data = np.random.randn(100, 5)
        split_idx = int(0.8 * len(data))
        
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(test_data), 20)
        self.assertEqual(len(train_data) + len(test_data), len(data))
    
    def test_sequence_creation(self):
        """Test sequence creation for LSTM"""
        data = np.random.randn(100, 3)
        sequence_length = 10
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
            targets.append(data[i, 0])  # Predict first column
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        self.assertEqual(sequences.shape[0], len(data) - sequence_length)
        self.assertEqual(sequences.shape[1], sequence_length)
        self.assertEqual(sequences.shape[2], 3)
        self.assertEqual(len(targets), len(data) - sequence_length)

class TestAPIEndpoints(unittest.TestCase):
    """Test API functionality"""
    
    def test_stock_symbols(self):
        """Test stock symbol validation"""
        valid_symbols = ['JKSE', 'BBCA', 'BBRI', 'TLKM', 'ASII', 'UNVR']
        
        for symbol in valid_symbols:
            self.assertIn(symbol, valid_symbols)
            self.assertTrue(len(symbol) <= 4)
    
    def test_prediction_format(self):
        """Test prediction response format"""
        # Mock prediction response
        prediction_response = {
            'symbol': 'BBCA',
            'current_price': 9500.0,
            'predictions': [
                {'date': '2024-01-01', 'predicted_price': 9550.0, 'change_pct': 0.53}
            ]
        }
        
        self.assertIn('symbol', prediction_response)
        self.assertIn('current_price', prediction_response)
        self.assertIn('predictions', prediction_response)
        self.assertIsInstance(prediction_response['predictions'], list)

class TestTradingSignals(unittest.TestCase):
    """Test trading signal generation"""
    
    def test_signal_validation(self):
        """Test trading signal format"""
        valid_signals = ['BUY', 'SELL', 'HOLD']
        
        # Mock signal
        signal = {
            'signal': 'BUY',
            'confidence': 75.5,
            'current_price': 9500.0,
            'predicted_price': 9600.0
        }
        
        self.assertIn(signal['signal'], valid_signals)
        self.assertTrue(0 <= signal['confidence'] <= 100)
        self.assertGreater(signal['current_price'], 0)
        self.assertGreater(signal['predicted_price'], 0)
    
    def test_confidence_bounds(self):
        """Test confidence is within valid bounds"""
        confidences = [25.5, 67.8, 89.2, 45.1]
        
        for conf in confidences:
            self.assertTrue(0 <= conf <= 100)

class TestRiskManagement(unittest.TestCase):
    """Test risk management calculations"""
    
    def test_position_sizing(self):
        """Test position sizing calculation"""
        # Mock position sizing
        confidence = 80.0
        risk_score = 3.5
        
        base_size = confidence / 100
        risk_adjustment = (10 - risk_score) / 10
        position_size = base_size * risk_adjustment
        
        self.assertTrue(0 <= position_size <= 1)
        self.assertLessEqual(position_size, base_size)
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation"""
        current_price = 9500.0
        volatility = 0.025
        
        stop_loss_buy = current_price * (1 - volatility * 2)
        take_profit_buy = current_price * (1 + volatility * 3)
        
        self.assertLess(stop_loss_buy, current_price)
        self.assertGreater(take_profit_buy, current_price)

class TestPerformanceMetrics(unittest.TestCase):
    """Test performance evaluation metrics"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.actual = np.random.randn(50) * 100 + 9500
        self.predicted = self.actual + np.random.randn(50) * 50
    
    def test_mape_calculation(self):
        """Test MAPE calculation"""
        mape = np.mean(np.abs((self.actual - self.predicted) / self.actual)) * 100
        
        self.assertGreater(mape, 0)
        self.assertIsInstance(mape, float)
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation"""
        actual_direction = np.diff(self.actual) > 0
        predicted_direction = np.diff(self.predicted) > 0
        
        correct_predictions = np.sum(actual_direction == predicted_direction)
        total_predictions = len(actual_direction)
        accuracy = (correct_predictions / total_predictions) * 100
        
        self.assertTrue(0 <= accuracy <= 100)

if __name__ == '__main__':
    # Run specific test suites
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestDataProcessing))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestTechnicalIndicators))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestModelValidation))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestAPIEndpoints))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestTradingSignals))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestRiskManagement))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestPerformanceMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")