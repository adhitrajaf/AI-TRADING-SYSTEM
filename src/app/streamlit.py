import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import requests

# Page config
st.set_page_config(
    page_title="AI Stock Trading Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
STOCKS = {
    'JKSE': 'IDX Composite',
    'BBCA': 'Bank Central Asia',
    'BBRI': 'Bank Rakyat Indonesia',
    'TLKM': 'Telkom Indonesia',
    'ASII': 'Astra International',
    'UNVR': 'Unilever Indonesia'
}

API_BASE_URL = "http://127.0.0.1:8000"

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.signal-buy {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    color: #155724;
}

.signal-sell {
    background: #f8d7da;
    border-left: 4px solid #dc3545;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    color: #721c24;
}

.signal-hold {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, days=100):
    """Fetch real-time stock data"""
    try:
        if symbol == 'JKSE':
            yf_symbol = '^JKSE'
        else:
            yf_symbol = f"{symbol}.JK"
        
        stock = yf.Ticker(yf_symbol)
        data = stock.history(period=f"{days}d")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def call_api(endpoint, method="GET", data=None):
    """Call API with error handling"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🤖 AI Stock Trading Advisor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # API Status Check
    with st.sidebar:
        st.subheader("System Status")
        health = call_api("")
        if health:
            st.success("✅ API Online")
            st.info(f"Models: {health.get('models_loaded', 0)}")
        else:
            st.error("❌ API Offline")
    
    # Page Selection - Market Overview as default
    page = st.sidebar.selectbox("Choose Page:", [
        "Market Overview",  
        "Stock Analysis",
        "AI Predictions",
        "Trading Signals"
    ])
    
    if page == "Market Overview":
        market_overview()
    elif page == "Stock Analysis":
        stock_analysis()
    elif page == "AI Predictions":
        ai_predictions()
    elif page == "Trading Signals":
        trading_signals()

def market_overview():
    st.header("Market Overview")
    
    # Market data
    overview = call_api("market-overview")
    
    if overview:
        st.subheader("Real-time Prices & AI Signals")
        
        # Create display table
        market_df = pd.DataFrame(overview['overview'])
        
        if not market_df.empty:
            display_df = market_df[['symbol', 'name', 'current_price', 'change_pct', 'ai_signal', 'ai_confidence']].copy()
            display_df.columns = ['Symbol', 'Stock Name', 'Price (IDR)', 'Change %', 'AI Signal', 'AI Confidence']
            
            st.dataframe(display_df, use_container_width=True)
    
    # Price charts
    st.subheader("Price Charts (30 Days)")
    
    cols = st.columns(2)
    for i, (symbol, name) in enumerate(STOCKS.items()):  # Remove [:4] to show all stocks
        with cols[i % 2]:
            data = fetch_stock_data(symbol, days=30)
            if data is not None and not data.empty:
                fig = px.line(
                    x=data.index,
                    y=data['Close'],
                    title=f"{name} (30 days)"
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

def stock_analysis():
    st.header("Individual Stock Analysis")
    
    # Stock selection
    col1, col2 = st.columns(2)
    with col1:
        selected_symbol = st.selectbox(
            "Select Stock:",
            list(STOCKS.keys()),
            format_func=lambda x: f"{x} - {STOCKS[x]}"
        )
    
    with col2:
        period = st.selectbox("Analysis Period:", [30, 60, 90, 180], index=1)
    
    if selected_symbol:
        # Get analysis from API
        analysis = call_api(f"stock/{selected_symbol}/analysis")
        
        if analysis:
            st.subheader(f"{analysis['name']} Analysis")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"IDR {analysis['current_price']:,.0f}")
            
            with col2:
                perf_1d = analysis['price_performance']['1_day']
                st.metric("1-Day Change", f"{perf_1d:+.2f}%")
            
            with col3:
                rsi = analysis['technical_indicators']['rsi']
                st.metric("RSI", f"{rsi:.1f}")
            
            with col4:
                if 'ai_insights' in analysis:
                    confidence = analysis['ai_insights']['model_confidence']
                    st.metric("AI Confidence", confidence)
            
            # AI Insights
            if 'ai_insights' in analysis:
                st.subheader("AI Insights")
                
                ai = analysis['ai_insights']
                predicted_price = ai['next_day_prediction']
                predicted_change = ai['predicted_change']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("AI Prediction (Next Day)", f"IDR {predicted_price:,.0f}")
                
                with col2:
                    st.metric("Predicted Change", f"{predicted_change:+.2f}%")
        
        # Price chart
        st.subheader("Price Chart")
        data = fetch_stock_data(selected_symbol, days=period)
        
        if data is not None and not data.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price"
            ))
            
            fig.update_layout(
                title=f"{STOCKS[selected_symbol]} - Price Chart",
                yaxis_title="Price (IDR)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def ai_predictions():
    st.header("AI Stock Predictions")
    
    # Prediction interface
    col1, col2 = st.columns(2)
    with col1:
        selected_symbol = st.selectbox(
            "Select Stock:",
            list(STOCKS.keys()),
            format_func=lambda x: f"{x} - {STOCKS[x]}"
        )
    
    with col2:
        pred_days = st.slider("Prediction Days:", 1, 30, 7)
    
    if st.button("Generate AI Prediction", type="primary"):
        with st.spinner("AI is analyzing..."):
            prediction = call_api("predict", method="POST", data={
                "symbol": selected_symbol,
                "days": pred_days
            })
        
        if prediction:
            st.success("AI Prediction Generated!")
            
            # Current price
            current_price = prediction['current_price']
            model_type = prediction.get('model_type', 'AI Model')
            confidence = prediction.get('confidence', 'Medium')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"IDR {current_price:,.0f}")
            with col2:
                st.metric("Model Type", model_type)
            with col3:
                st.metric("Confidence", confidence)
            
            # Predictions
            st.subheader("AI Predictions")
            
            pred_df = pd.DataFrame(prediction['predictions'])
            pred_df['date'] = pd.to_datetime(pred_df['date'])
            
            # Prediction chart
            fig = go.Figure()
            
            # Current price
            fig.add_trace(go.Scatter(
                x=[datetime.now()],
                y=[current_price],
                mode='markers',
                name='Current Price',
                marker=dict(color='blue', size=12)
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=pred_df['date'],
                y=pred_df['predicted_price'],
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{STOCKS[selected_symbol]} - AI Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price (IDR)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions table
            display_df = pred_df.copy()
            display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"IDR {x:,.0f}")
            display_df['change_pct'] = display_df['change_pct'].apply(lambda x: f"{x:+.2f}%")
            display_df.columns = ['Date', 'Predicted Price', 'Expected Change %']
            
            st.dataframe(display_df, use_container_width=True)

def trading_signals():
    st.header("AI Trading Signals")
    
    # Signal configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_symbol = st.selectbox(
            "Select Stock:",
            list(STOCKS.keys()),
            format_func=lambda x: f"{x} - {STOCKS[x]}"
        )
    
    with col2:
        analysis_type = "enhanced"  # Fixed to enhanced only
        st.selectbox(
            "Analysis Type:",
            ["AI Analysis"],
            index=0,
            disabled=True,
            help="Using advanced AI analysis model"
        )
    
    with col3:
        signal_days = st.slider("Signal Period:", 5, 30, 25)
    
    # Time range selection
    st.subheader("Select Analysis Period")
    time_range = st.selectbox(
        "Choose time range to analyze:",
        ["Week 1 (Days 1-7)", "Week 2 (Days 8-14)", "Week 3 (Days 15-21)", "Week 4 (Days 22-30)", "Full Month (All Days)"],
        index=0,
        help="Choose which week you want to analyze for trading opportunities"
    )
    
    # Generate signals
    if st.button("Generate Trading Signals", type="primary"):
        with st.spinner("Analyzing optimal timing..."):
            signals_response = call_api("trading-signals", method="POST", data={
                "symbol": selected_symbol,
                "analysis_type": analysis_type,
                "days": signal_days
            })
        
        if signals_response:
            st.success("Trading Signals Generated!")
            
            # Summary
            summary = signals_response['summary']
            recommendation = signals_response['recommendation']
            
            # Display recommendation
            if recommendation == "BUY":
                st.success(f"**OVERALL AI RECOMMENDATION: {recommendation}**")
            elif recommendation == "SELL":
                st.error(f"**OVERALL AI RECOMMENDATION: {recommendation}**")
            else:
                st.info(f"**OVERALL AI RECOMMENDATION: {recommendation}**")
            
            # Overall summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Buy Signals", summary['buy_signals'])
            with col2:
                st.metric("Total Sell Signals", summary['sell_signals'])
            with col3:
                st.metric("Total Hold Signals", summary['hold_signals'])
            with col4:
                st.metric("Overall Confidence", f"{summary['avg_confidence']:.1f}%")
            
            signals = signals_response['signals']
            
            # Determine which signals to show based on time range
            if time_range == "Week 1 (Days 1-7)":
                start_day, end_day = 0, 7
                range_title = "Week 1 Action Plan"
                range_subtitle = "Next 7 Days Trading Strategy"
            elif time_range == "Week 2 (Days 8-14)":
                start_day, end_day = 7, 14
                range_title = "Week 2 Action Plan"
                range_subtitle = "Days 8-14 Trading Strategy"
            elif time_range == "Week 3 (Days 15-21)":
                start_day, end_day = 14, 21
                range_title = "Week 3 Action Plan"
                range_subtitle = "Days 15-21 Trading Strategy"
            elif time_range == "Week 4 (Days 22-30)":
                start_day, end_day = 21, 30
                range_title = "Week 4 Action Plan"
                range_subtitle = "Days 22-30 Trading Strategy"
            else:  # Full Month
                start_day, end_day = 0, len(signals)
                range_title = "Full Month Action Plan"
                range_subtitle = "Complete Monthly Trading Strategy"
            
            # Display selected range
            st.subheader(range_title)
            st.write(f"**{range_subtitle}**")
            
            # Show signals for selected range
            display_signals = signals[start_day:end_day]
            
            if not display_signals:
                st.warning(f"No signals available for {time_range}. Try generating more days or select a different range.")
            else:
                # Action plan for selected period
                for i, signal in enumerate(display_signals):
                    signal_type = signal['signal']
                    confidence = signal['confidence']
                    current_price = signal['current_price']
                    predicted_price = signal['predicted_price']
                    reasoning = signal['reasoning']
                    date = signal['date']
                    
                    day_number = start_day + i + 1
                    
                    if signal_type == "BUY":
                        card_class = "signal-buy"
                        emoji = "📈"
                        action = "**ACTION: Consider buying on this date**"
                    elif signal_type == "SELL":
                        card_class = "signal-sell" 
                        emoji = "📉"
                        action = "**ACTION: Consider selling on this date**"
                    else:
                        card_class = "signal-hold"
                        emoji = "⏸️"
                        action = "**ACTION: Monitor price, no trade needed**"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4>{emoji} Day {day_number} ({date}) - {signal_type} Signal</h4>
                        <p>{action}</p>
                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                        <p><strong>Current Price:</strong> IDR {current_price:,.0f}</p>
                        <p><strong>Target Price:</strong> IDR {predicted_price:,.0f}</p>
                        <p><strong>AI Analysis:</strong> {reasoning}</p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()