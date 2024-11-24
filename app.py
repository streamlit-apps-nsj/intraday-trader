import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from concurrent.futures import ThreadPoolExecutor

class StockAnalyzer:
    def __init__(self):
        # Example stocks for reference
        self.example_stocks = [
            'RELIANCE.NS',
            'TCS.NS',
            'HDFCBANK.NS',
            'INFY.NS',
            'ICICIBANK.NS'
        ]

    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_nse_stocks():
        """Fetch list of NSE stocks"""
        try:
            url = "https://raw.githubusercontent.com/nitinkumar30/indian-stock-symbols/main/symbolsNSE.csv"
            df = pd.read_csv(url)
            return df['Symbol'].tolist()
        except:
            return [
                'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'ITC', 'HINDUNILVR',
                'SBIN', 'BHARTIARTL', 'WIPRO', 'AXISBANK', 'TATAMOTORS', 'MARUTI'
            ]

    def validate_stock(self, symbol):
        """Validate if the stock exists and has data"""
        try:
            stock = yf.Ticker(f"{symbol}.NS")
            info = stock.info
            return True
        except:
            return False

    def fetch_stock_data(self, symbol, period='1d', interval='5m'):
        """Fetch stock data from Yahoo Finance"""
        @st.cache_data(ttl=300)
        def fetch_data(_symbol, _period, _interval):
            try:
                stock = yf.Ticker(_symbol)
                df = stock.history(period=_period, interval=_interval)
                if len(df) > 0:
                    return df, stock.info
                return None, None
            except Exception as e:
                st.error(f"Error fetching data for {_symbol}: {str(e)}")
                return None, None

        return fetch_data(symbol, period, interval)

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()

            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

            # Volume analysis
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

            # Price momentum
            df['ROC'] = df['Close'].pct_change(periods=10) * 100

            # Stochastic Oscillator
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df['Stoch_K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

            return df
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return None

    def calculate_target_prices(self, current_price, signal_type):
        """Calculate entry, stop loss, and target prices"""
        if signal_type == "BUY":
            entry = current_price
            stop_loss = entry * 0.98  # 2% below entry
            target1 = entry * 1.01  # 1% above entry
            target2 = entry * 1.02  # 2% above entry
            target3 = entry * 1.03  # 3% above entry
        elif signal_type == "SELL":
            entry = current_price
            stop_loss = entry * 1.02  # 2% above entry
            target1 = entry * 0.99  # 1% below entry
            target2 = entry * 0.98  # 2% below entry
            target3 = entry * 0.97  # 3% below entry
        else:
            return None

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'target3': target3
        }

    def generate_signal(self, df):
        """Generate trading signal based on technical indicators"""
        if df is None or len(df) < 20:
            return {"signal": "HOLD", "reasons": ["Insufficient data for analysis"]}

        current_price = df['Close'].iloc[-1]
        previous_price = df['Close'].iloc[-2]
        rsi = df['RSI'].iloc[-1]
        volume_ratio = df['Volume_Ratio'].iloc[-1]
        stoch_k = df['Stoch_K'].iloc[-1]
        stoch_d = df['Stoch_D'].iloc[-1]

        buy_signals = []
        sell_signals = []

        # RSI Analysis
        if rsi < 35:
            buy_signals.append(f"RSI is oversold ({rsi:.2f})")
        elif rsi > 65:
            sell_signals.append(f"RSI is overbought ({rsi:.2f})")

        # Price vs Moving Averages
        if current_price > df['MA5'].iloc[-1] > df['MA10'].iloc[-1]:
            buy_signals.append("Price above short-term MAs (Strong uptrend)")
        elif current_price < df['MA5'].iloc[-1] < df['MA10'].iloc[-1]:
            sell_signals.append("Price below short-term MAs (Strong downtrend)")

        # Moving Average Crossovers
        if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] and df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]:
            buy_signals.append("Short-term MA crossed above medium-term MA")
        elif df['MA5'].iloc[-1] < df['MA20'].iloc[-1] and df['MA5'].iloc[-2] >= df['MA20'].iloc[-2]:
            sell_signals.append("Short-term MA crossed below medium-term MA")

        # Bollinger Bands
        if current_price < df['BB_lower'].iloc[-1]:
            buy_signals.append("Price below lower Bollinger Band (Potential reversal)")
        elif current_price > df['BB_upper'].iloc[-1]:
            sell_signals.append("Price above upper Bollinger Band (Potential reversal)")

        # Volume Analysis
        if volume_ratio > 1.5:
            if current_price > previous_price:
                buy_signals.append(f"High volume up move (Volume {volume_ratio:.2f}x average)")
            elif current_price < previous_price:
                sell_signals.append(f"High volume down move (Volume {volume_ratio:.2f}x average)")

        # MACD Analysis
        if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
            buy_signals.append("MACD crossed above signal line")
        elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] >= df['Signal_Line'].iloc[-2]:
            sell_signals.append("MACD crossed below signal line")

        # Stochastic Oscillator
        if stoch_k < 20 and stoch_k > stoch_d:
            buy_signals.append(f"Stochastic indicates oversold with bullish crossover")
        elif stoch_k > 80 and stoch_k < stoch_d:
            sell_signals.append(f"Stochastic indicates overbought with bearish crossover")

        # Decision making
        signal = "HOLD"
        reasons = []

        if len(buy_signals) >= 3:
            signal = "BUY"
            reasons = buy_signals
            reasons.append(f"Strong Buy Signal: {len(buy_signals)} indicators confirm")
        elif len(sell_signals) >= 3:
            signal = "SELL"
            reasons = sell_signals
            reasons.append(f"Strong Sell Signal: {len(sell_signals)} indicators confirm")
        elif len(buy_signals) >= 2:
            signal = "BUY"
            reasons = buy_signals
            reasons.append("Moderate Buy Signal")
        elif len(sell_signals) >= 2:
            signal = "SELL"
            reasons = sell_signals
            reasons.append("Moderate Sell Signal")
        else:
            reasons = ["No strong trading signals detected"]
            if buy_signals:
                reasons.extend(buy_signals)
            if sell_signals:
                reasons.extend(sell_signals)

        prices = self.calculate_target_prices(current_price, signal) if signal != "HOLD" else None

        return {
            "signal": signal,
            "price": current_price,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "reasons": reasons,
            "prices": prices,
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals)
        }

    def create_candlestick_chart(self, df):
        """Create an interactive candlestick chart with indicators"""
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.5, 0.2, 0.15, 0.15])

        # Candlestick chart with Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)

        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                                 line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                                 line=dict(color='gray', dash='dash')), row=1, col=1)

        # Add Moving Averages
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5',
                                 line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20',
                                 line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50',
                                 line=dict(color='red')), row=1, col=1)

        # Volume bars
        colors = ['red' if row['Open'] > row['Close'] else 'green' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                             marker_color=colors), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                 line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=3, col=1)

        # Stochastic
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='Stoch %K',
                                 line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='Stoch %D',
                                 line=dict(color='orange')), row=4, col=1)
        fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="green", row=4, col=1)

        # Update layout
        fig.update_layout(
            title='Technical Analysis Chart',
            yaxis_title='Price',
            yaxis2_title='Volume',
            yaxis3_title='RSI',
            yaxis4_title='Stoch',
            xaxis_rangeslider_visible=False,
            height=1000
        )

        return fig

def analyze_watchlist_stock(analyzer, symbol, period='1d', interval='5m'):
    """Analyze a single watchlist stock"""
    try:
        df, stock_info = analyzer.fetch_stock_data(symbol, period, interval)
        if df is not None and not df.empty:
            df = analyzer.calculate_indicators(df)
            if df is not None:
                signal = analyzer.generate_signal(df)
                current_price = df['Close'].iloc[-1]
                day_change = ((current_price - df['Open'].iloc[0]) / df['Open'].iloc[0]) * 100

                return {
                    'symbol': symbol.replace('.NS', ''),
                    'price': current_price,
                    'change': day_change,
                    'signal': signal['signal'],
                    'rsi': signal['rsi'],
                    'volume_ratio': signal['volume_ratio'],
                    'buy_signals': signal.get('buy_signals', 0),
                    'sell_signals': signal.get('sell_signals', 0),
                    'reasons': signal['reasons'][:2],  # Show only top 2 reasons
                    'market_cap': stock_info.get('marketCap', 0) if stock_info else 0
                }
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
    return None

def create_watchlist_tab():
    st.title("Market Watchlist")

    # Predefined watchlist
    watchlist = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'LT.NS',
        'KOTAKBANK.NS', 'AXISBANK.NS', 'MARUTI.NS', 'TITAN.NS', 'WIPRO.NS'
    ]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        period = st.selectbox(
            "Select Time Period",
            options=['1d', '5d', '1mo'],
            format_func=lambda x: {'1d': '1 Day', '5d': '5 Days', '1mo': '1 Month'}[x],
            key='watchlist_period'
        )

    with col2:
        interval_options = {
            '1d': ['1m', '5m', '15m'],
            '5d': ['5m', '15m', '30m'],
            '1mo': ['30m', '1h', '1d']
        }
        interval = st.selectbox(
            "Select Interval",
            options=interval_options[period],
            format_func=lambda x: {
                '1m': '1 Minute', '5m': '5 Minutes', '15m': '15 Minutes',
                '30m': '30 Minutes', '1h': '1 Hour', '1d': '1 Day'
            }[x],
            key='watchlist_interval'
        )

    with col3:
        if st.button("Refresh Watchlist"):
            st.cache_data.clear()

    # Initialize analyzer
    analyzer = StockAnalyzer()

    # Progress bar
    progress_text = "Analyzing watchlist stocks..."
    progress_bar = st.progress(0)

    # Analyze all stocks in parallel
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_watchlist_stock, analyzer, symbol, period, interval)
                   for symbol in watchlist]

        for i, future in enumerate(futures):
            result = future.result()
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / len(watchlist))

    # Clear progress bar
    progress_bar.empty()

    # Create summary metrics
    buy_signals = sum(1 for r in results if r['signal'] == 'BUY')
    sell_signals = sum(1 for r in results if r['signal'] == 'SELL')

    # Display market sentiment
    sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
    with sentiment_col1:
        st.metric("Buy Signals", f"{buy_signals}/{len(results)}")
    with sentiment_col2:
        st.metric("Sell Signals", f"{sell_signals}/{len(results)}")
    with sentiment_col3:
        market_sentiment = "Bullish" if buy_signals > sell_signals else "Bearish" if sell_signals > buy_signals else "Neutral"
        st.metric("Market Sentiment", market_sentiment)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Sort by different criteria
    sort_col1, sort_col2 = st.columns([1, 3])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by",
            options=['market_cap', 'change', 'signal', 'rsi', 'volume_ratio'],
            format_func=lambda x: {
                'market_cap': 'Market Cap',
                'change': 'Day Change',
                'signal': 'Signal',
                'rsi': 'RSI',
                'volume_ratio': 'Volume Ratio'
            }[x]
        )

    if sort_by == 'market_cap':
        df_results = df_results.sort_values(by='market_cap', ascending=False)
    elif sort_by == 'change':
        df_results = df_results.sort_values(by='change', ascending=False)
    elif sort_by in ['rsi', 'volume_ratio']:
        df_results = df_results.sort_values(by=sort_by, ascending=False)

    # Display results in an expandable table
    for _, row in df_results.iterrows():
        with st.expander(
                f"{row['symbol']} | ₹{row['price']:.2f} | {row['change']:.2f}% | {row['signal']}"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("RSI", f"{row['rsi']:.2f}")
                st.metric("Volume Ratio", f"{row['volume_ratio']:.2f}x")
                st.metric("Market Cap", f"₹{row['market_cap'] / 10000000:.2f}Cr")
            with col2:
                st.metric("Buy Signals", row['buy_signals'])
                st.metric("Sell Signals", row['sell_signals'])
                if row['reasons']:
                    st.write("Top Reasons:")
                    for reason in row['reasons']:
                        st.write(f"• {reason}")

def main():
    st.set_page_config(page_title="Indian Stock Analysis Tool", layout="wide")

    # Create tabs
    tab1, tab2 = st.tabs(["Single Stock Analysis", "Market Watchlist"])

    with tab1:
        st.title("Indian Stock Analysis Tool")
        st.write("Real-time technical analysis and trading signals")

        # Disclaimer
        st.warning(
            "Disclaimer: This tool is for educational purposes only. Always do your own research before trading.")

        # Initialize analyzer
        analyzer = StockAnalyzer()

        # Stock input section
        st.subheader("Stock Selection")

        # Create two columns for stock input
        col1, col2 = st.columns([2, 1])

        with col1:
            # Text input for stock symbol
            stock_input = st.text_input(
                "Enter Stock Symbol (e.g., RELIANCE, TCS, HDFCBANK)",
                help="Enter the stock symbol without .NS suffix"
            ).upper()

        with col2:
            # Time period selection
            period = st.selectbox(
                "Select Time Period",
                options=['1d', '5d', '1mo'],
                format_func=lambda x: {
                    '1d': '1 Day',
                    '5d': '5 Days',
                    '1mo': '1 Month'
                }[x],
                help="Select the time period for analysis"
            )

            # Interval selection based on period
            interval_options = {
                '1d': ['1m', '5m', '15m'],
                '5d': ['5m', '15m', '30m'],
                '1mo': ['30m', '1h', '1d']
            }

            interval = st.selectbox(
                "Select Interval",
                options=interval_options[period],
                format_func=lambda x: {
                    '1m': '1 Minute',
                    '5m': '5 Minutes',
                    '15m': '15 Minutes',
                    '30m': '30 Minutes',
                    '1h': '1 Hour',
                    '1d': '1 Day'
                }[x],
                help="Select the interval for data points"
            )

        # Add refresh button
        if st.button("Refresh Data", key="single_stock_refresh"):
            st.cache_data.clear()

        # Example stocks section
        with st.expander("Click to see example stocks"):
            st.write("Example stocks you can try:")
            example_stocks = [stock.replace('.NS', '') for stock in analyzer.example_stocks]
            st.write(", ".join(example_stocks))

        # Analysis Section
        if stock_input:
            st.write("---")
            st.write(f"Analyzing {stock_input}...")

            # Validate and add .NS suffix
            symbol = f"{stock_input}.NS"

            if analyzer.validate_stock(stock_input):
                # Fetch and analyze data
                df, stock_info = analyzer.fetch_stock_data(symbol, period, interval)

                if df is not None and not df.empty:
                    # Display basic stock info if available
                    if stock_info:
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        with info_col1:
                            st.metric("Market Cap",
                                      f"₹{stock_info.get('marketCap', 0) / 10000000:.2f}Cr")
                        with info_col2:
                            st.metric("52W High",
                                      f"₹{stock_info.get('fiftyTwoWeekHigh', 0):.2f}")
                        with info_col3:
                            st.metric("52W Low", f"₹{stock_info.get('fiftyTwoWeekLow', 0):.2f}")
                        with info_col4:
                            day_range = f"₹{stock_info.get('dayLow', 0):.2f} - ₹{stock_info.get('dayHigh', 0):.2f}"
                            st.metric("Day's Range", day_range)

                    # Calculate indicators
                    df = analyzer.calculate_indicators(df)

                    if df is not None:
                        # Get trading signal
                        signal = analyzer.generate_signal(df)

                        # Create columns for layout
                        chart_col, signal_col = st.columns([2, 1])

                        with chart_col:
                            # Display chart
                            fig = analyzer.create_candlestick_chart(df)
                            st.plotly_chart(fig, use_container_width=True)

                        with signal_col:
                            st.subheader("Trading Signal")

                            # Display current price and signal
                            price_color = "red" if signal['signal'] == "SELL" else "green" if \
                            signal['signal'] == "BUY" else "gray"
                            st.markdown(f"**Current Price:** ₹{signal['price']:.2f}")
                            st.markdown(
                                f"**Signal:** <span style='color:{price_color}'>{signal['signal']}</span>",
                                unsafe_allow_html=True)

                            # Display technical indicators
                            st.subheader("Technical Indicators")

                            # Create three columns for technical indicators
                            tech_col1, tech_col2 = st.columns(2)

                            with tech_col1:
                                st.metric("RSI", f"{signal['rsi']:.2f}")
                                st.metric("Buy Signals", signal.get('buy_signals', 0))
                            with tech_col2:
                                st.metric("Volume Ratio", f"{signal['volume_ratio']:.2f}x")
                                st.metric("Sell Signals", signal.get('sell_signals', 0))

                            if signal['prices'] and signal['signal'] != "HOLD":
                                st.subheader("Price Targets")

                                # Create a DataFrame for price targets
                                targets_df = pd.DataFrame({
                                    'Level': ['Entry', 'Stop Loss', 'Target 1', 'Target 2',
                                              'Target 3'],
                                    'Price': [
                                        signal['prices']['entry'],
                                        signal['prices']['stop_loss'],
                                        signal['prices']['target1'],
                                        signal['prices']['target2'],
                                        signal['prices']['target3']
                                    ],
                                    'Change %': [
                                        0,
                                        (signal['prices']['stop_loss'] - signal['prices'][
                                            'entry']) / signal['prices'][
                                            'entry'] * 100,
                                        (signal['prices']['target1'] - signal['prices'][
                                            'entry']) / signal['prices'][
                                            'entry'] * 100,
                                        (signal['prices']['target2'] - signal['prices'][
                                            'entry']) / signal['prices'][
                                            'entry'] * 100,
                                        (signal['prices']['target3'] - signal['prices'][
                                            'entry']) / signal['prices'][
                                            'entry'] * 100
                                    ]
                                })

                                st.dataframe(
                                    targets_df.style.format({
                                        'Price': '₹{:.2f}',
                                        'Change %': '{:.1f}%'
                                    })
                                )

                            if signal['reasons']:
                                st.subheader("Signal Reasons")
                                for reason in signal['reasons']:
                                    st.write(f"• {reason}")

                            # Add trading volume information
                            st.subheader("Trading Volume")
                            current_volume = df['Volume'].iloc[-1]
                            avg_volume = df['Volume'].mean()
                            volume_change = ((current_volume - avg_volume) / avg_volume) * 100
                            st.metric("Current Volume", f"{current_volume:,.0f}")
                            st.metric("Avg Volume", f"{avg_volume:,.0f}")
                            st.metric("Volume Change", f"{volume_change:+.2f}%")

                else:
                    st.error("Unable to fetch data. Please try again later.")
            else:
                st.error("Invalid stock symbol. Please check the symbol and try again.")

    with tab2:
        create_watchlist_tab()

if __name__ == "__main__":
    main()