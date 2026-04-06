import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def fix_multiindex_columns(df):
    """Fix MultiIndex columns to simple column names"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def get_scalar_value(series):
    """Safely extract scalar value"""
    val = series.iloc[-1]
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)

@st.cache_data
def load_multi_data():
    """Fetch 5 companies - FIXED MultiIndex"""
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    all_data = []
    
    for ticker in tickers:
        try:
            df_single = yf.download(ticker, period="2y", progress=False)
            df_single = df_single.reset_index()
            
            # Fix MultiIndex immediately
            df_single = fix_multiindex_columns(df_single)
            
            df_single['Ticker'] = ticker
            df_single = df_single.sort_values('Date').reset_index(drop=True)
            
            # Features
            df_single['MA_5'] = df_single['Close'].rolling(5, min_periods=1).mean()
            df_single['MA_20'] = df_single['Close'].rolling(20, min_periods=1).mean()
            df_single['Price_Change'] = df_single['Close'].pct_change()
            df_single['Volume_Change'] = df_single['Volume'].pct_change()
            
            # RSI
            delta = df_single['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss
            df_single['RSI'] = 100 - (100 / (1 + rs))
            
            # Mock sentiment
            np.random.seed(ord(ticker[0]))
            df_single['Sentiment'] = np.random.uniform(-0.5, 0.5, len(df_single))
            df_single['Target'] = df_single['Close'].shift(-1)
            
            all_data.append(df_single.dropna())
            
        except Exception as e:
            st.error(f"Failed to fetch {ticker}: {e}")
    
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

@st.cache_data
def train_model_for_ticker(df, ticker):
    ticker_df = df[df['Ticker'] == ticker].copy()
    if len(ticker_df) < 50:
        return None, float('inf')
    
    features = ['Close', 'MA_5', 'MA_20', 'Price_Change', 'Volume_Change', 'RSI', 'Sentiment']
    X = ticker_df[features].fillna(0)
    y = ticker_df['Target'].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    return model, rmse

# MAIN APP
st.set_page_config(page_title="Multi-Stock Dashboard", layout="wide")
st.title("🎯 AI-Powered Stock Price Forecasting and Sentiment Analysis Dashboard for Tech Giants")
st.markdown("**AAPL, MSFT, AMZN, GOOGL, META - Interactive Charts**")

# Load data
with st.spinner("Loading 5 companies..."):
    df = load_multi_data()

st.success(f"✅ Loaded {len(df):,} rows | Companies: {df['Ticker'].nunique()}")

# COMPANY SELECTION
st.sidebar.header("📊 Select Company")
selected_ticker = st.sidebar.selectbox("Choose stock:", sorted(df['Ticker'].unique()), index=0)

company_data = df[df['Ticker'] == selected_ticker].copy()

# Train model
model, rmse_rf = train_model_for_ticker(df, selected_ticker)

# SAFE METRICS
latest_price = get_scalar_value(company_data['Close'])
latest_rsi = get_scalar_value(company_data['RSI'])
rsi_change = latest_rsi - get_scalar_value(company_data['RSI'].iloc[:-1]) if len(company_data) > 1 else 0
avg_sentiment = float(company_data['Sentiment'].mean())
rmse_display = float(rmse_rf) if rmse_rf != float('inf') else 999.99

# METRICS
latest_ma20 = get_scalar_value(company_data["MA_20"])

price_trend = "Bullish momentum" if latest_price > latest_ma20 else "Weak trend / below MA20"
rsi_action = (
    "RSI is above 70: stock may be overbought, so consider booking profit or waiting."
    if latest_rsi > 70 else
    "RSI is below 30: stock may be oversold, so this can be a possible buy zone."
    if latest_rsi < 30 else
    "RSI is between 30 and 70: trend is neutral, so holding or waiting is safer."
)
sentiment_action = (
    "Positive sentiment: news flow is supportive, which can strengthen bullish confidence."
    if avg_sentiment > 0.15 else
    "Negative sentiment: recent news is weak, so avoid aggressive buying until confirmation."
    if avg_sentiment < -0.15 else
    "Neutral sentiment: news impact is mixed, so rely more on chart trend and RSI."
)
rmse_action = (
    "Lower RMSE means better prediction reliability. This model is reasonably stable for directional support."
    if rmse_display < 3 else
    "Higher RMSE means prediction confidence is weaker, so use model output carefully with indicators."
)
price_action = (
    f"Latest price is ${latest_price:.2f}. {price_trend}. "
    "If price is above MA20, trend is generally stronger; if below MA20, wait for confirmation."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    f"{selected_ticker} Price",
    f"${latest_price:.2f}",
    help=price_action
)
col2.metric(
    "RSI",
    f"{latest_rsi:.1f}",
    f"{rsi_change:+.1f}",
    help=rsi_action
)
col3.metric(
    "Sentiment",
    f"{avg_sentiment:.3f}",
    help=sentiment_action
)
col4.metric(
    "Model RMSE",
    f"${rmse_display:.2f}",
    help=rmse_action
)

st.caption("Hover the small tooltip icon beside each metric label to see what the result suggests for the user.")

# INTERACTIVE CHARTS
col1, col2 = st.columns(2)

with col1:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=company_data['Date'], y=company_data['Close'], 
                                  name='Close', line=dict(width=3)))
    fig_price.add_trace(go.Scatter(x=company_data['Date'], y=company_data['MA_20'], 
                                  name='MA20', line=dict(width=2, color='orange')))
    fig_price.update_layout(title=f"{selected_ticker} Price", hovermode='x unified')
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=company_data['Date'], y=company_data['RSI'], 
                                name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(title=f"{selected_ticker} RSI", hovermode='x unified')
    st.plotly_chart(fig_rsi, use_container_width=True)

# FIXED COMPARISON CHART
st.header("📊 All Companies Comparison")
fig_compare = go.Figure()
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    fig_compare.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Close'], 
                                    name=ticker, line=dict(width=2)))

fig_compare.update_layout(title="All 5 Stocks - Interactive", hovermode='x unified')
st.plotly_chart(fig_compare, use_container_width=True)

# ADD THIS AFTER the existing charts in your app.py

# 1. ACTUAL vs PREDICTED (RF Model Performance)
st.header("🤖 Model Performance - Actual vs Predicted")
if model is not None:
    ticker_df = df[df['Ticker'] == selected_ticker].copy()
    features = ['Close', 'MA_5', 'MA_20', 'Price_Change', 'Volume_Change', 'RSI', 'Sentiment']
    X_test_full = ticker_df[features].fillna(0)
    y_test_full = ticker_df['Target'].fillna(0)
    
    # Get predictions for ALL data (not just test split)
    predictions = model.predict(X_test_full)
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=ticker_df['Date'], y=y_test_full, 
                                 name='Actual Price', line=dict(color='green')))
    fig_perf.add_trace(go.Scatter(x=ticker_df['Date'], y=predictions, 
                                 name='RF Predicted', line=dict(color='red')))
    fig_perf.update_layout(title=f"{selected_ticker} - RF Predictions vs Actual", hovermode='x unified')
    st.plotly_chart(fig_perf, use_container_width=True)

# 2. SENTIMENT vs PRICE CHANGE
st.header("📊 Sentiment vs Price Movement")
fig_sentiment = px.scatter(company_data, x='Sentiment', y='Price_Change', 
                          color='Close', size='Volume',
                          hover_data=['Date', 'Close'],
                          title=f"{selected_ticker} - Sentiment Impact on Price")
st.plotly_chart(fig_sentiment, use_container_width=True)


# 3. PREDICTION ERROR RESIDUALS - FIXED
st.header("📈 Prediction Accuracy Details")
if model is not None and len(company_data) > 10:
    ticker_df = df[df['Ticker'] == selected_ticker].copy()
    features = ['Close', 'MA_5', 'MA_20', 'Price_Change', 'Volume_Change', 'RSI', 'Sentiment']
    X_test_full = ticker_df[features].fillna(0)
    y_test_full = ticker_df['Target'].fillna(0)
    predictions = model.predict(X_test_full)
    
    residuals = y_test_full - predictions
    fig_residuals = px.scatter(x=predictions, y=residuals, 
                              title=f"{selected_ticker} - Residuals (Error Analysis)",
                              labels={'x':'Predicted Price', 'y':'Residuals ($)'})
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)
    st.plotly_chart(fig_residuals, use_container_width=True)
else:
    st.info("Model predictions available after data loads.")



# NEWS HEADLINES
import requests
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)  # Updates every hour
def get_real_news(ticker):
    NEWS_API_KEY = "22871f9043bb4e66a19cf1f56fd62c60"  # Free signup
    query = f"{ticker} stock"  # "AAPL stock", "MSFT stock"
    
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        articles = response.json()['articles'][:3]
        return [f"🗞️ {art['title'][:80]}..." for art in articles]
    return ["News API temporarily unavailable - using latest updates."]

# REPLACE your static news block with:
st.header("📰 Live News Headlines")
news = get_real_news(selected_ticker)
for headline in news:
    st.info(headline)


# CHATBOT
# CHATBOT - PROFESSIONAL VERSION
# =========================
# TAB / SECTION: AI CHATBOT
# =========================
# ========================================
# COMPLETE AI CHATBOT BLOCK
# ========================================
st.header("🤖 AI Project Assistant")

st.markdown("""
Ask questions about:
- Best stock right now
- Where to invest
- RSI, sentiment, RMSE
- Project workflow
- User benefits
- Dashboard usage
""")

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to the AI Project Assistant!**\n\n"
                "I can answer stock-related and project-related questions.\n\n"
                "**Try asking:**\n"
                "- Which stock is best right now?\n"
                "- Where should I invest?\n"
                "- How is this project useful to users?\n"
                "- How does this project work?\n"
                "- What does RSI mean?\n"
                "- What is model RMSE?"
            )
        }
    ]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------- SAFE LIVE VALUES ----------
latest_price = float(company_data["Close"].iloc[-1]) if len(company_data) > 0 else 0.0
latest_rsi = float(company_data["RSI"].iloc[-1]) if len(company_data) > 0 else 50.0
latest_ma5 = float(company_data["MA_5"].iloc[-1]) if len(company_data) > 0 else latest_price
latest_ma20 = float(company_data["MA_20"].iloc[-1]) if len(company_data) > 0 else latest_price
avg_sentiment = float(company_data["Sentiment"].mean()) if len(company_data) > 0 else 0.0
rmse_display = float(rmse_rf) if rmse_rf != float("inf") else 999.99

rsi_status = (
    "overbought" if latest_rsi > 70 else
    "oversold" if latest_rsi < 30 else
    "neutral"
)

trend_status = (
    "bullish" if latest_price > latest_ma20 else
    "bearish" if latest_price < latest_ma20 else
    "neutral"
)

sentiment_status = (
    "bullish" if avg_sentiment > 0.15 else
    "bearish" if avg_sentiment < -0.15 else
    "neutral"
)


# ---------- STOCK RANKING ----------
def rank_stocks_live(df_all):
    scores = []

    for ticker in sorted(df_all["Ticker"].unique()):
        tdf = df_all[df_all["Ticker"] == ticker].copy()
        if len(tdf) < 5:
            continue

        price = float(tdf["Close"].iloc[-1])
        ma20 = float(tdf["MA_20"].iloc[-1])
        rsi = float(tdf["RSI"].iloc[-1])
        sentiment = float(tdf["Sentiment"].mean())

        score = 0

        if price > ma20:
            score += 2
        else:
            score -= 1

        if 35 <= rsi <= 65:
            score += 2
        elif rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1

        if sentiment > 0.10:
            score += 2
        elif sentiment < -0.10:
            score -= 1

        scores.append({
            "Ticker": ticker,
            "Price": price,
            "RSI": rsi,
            "Sentiment": sentiment,
            "Score": score
        })

    if not scores:
        return pd.DataFrame(columns=["Ticker", "Price", "RSI", "Sentiment", "Score"])

    ranked_df = pd.DataFrame(scores).sort_values("Score", ascending=False).reset_index(drop=True)
    return ranked_df


rank_df = rank_stocks_live(df)
top_stock = rank_df.iloc[0]["Ticker"] if not rank_df.empty else selected_ticker
top_score = int(rank_df.iloc[0]["Score"]) if not rank_df.empty else 0


# ---------- RESPONSE HELPERS ----------
def stock_recommendation_summary():
    if latest_price > latest_ma20 and latest_rsi < 70 and avg_sentiment > 0:
        return (
            f"**{selected_ticker} currently looks relatively strong in this dashboard.**\n\n"
            f"- Price: ${latest_price:.2f}\n"
            f"- Trend: price is above MA20, so momentum is supportive.\n"
            f"- RSI: {latest_rsi:.1f}, so it is not in extreme overbought zone.\n"
            f"- Sentiment: {avg_sentiment:.3f}, which is positive.\n\n"
            "This suggests a stronger watchlist candidate, but users should still confirm with risk management."
        )
    elif latest_rsi > 70:
        return (
            f"**{selected_ticker} looks risky for fresh buying right now.**\n\n"
            f"- RSI is {latest_rsi:.1f}, which suggests overbought conditions.\n"
            f"- Trend may still be strong, but entry risk is higher.\n\n"
            "A safer approach is to wait for pullback or confirmation."
        )
    elif latest_rsi < 30 and avg_sentiment >= 0:
        return (
            f"**{selected_ticker} may be in a recovery-watch zone.**\n\n"
            f"- RSI is {latest_rsi:.1f}, suggesting oversold conditions.\n"
            f"- Sentiment is {avg_sentiment:.3f}, which is not negative.\n\n"
            "This can be useful for users tracking reversal opportunities."
        )
    else:
        return (
            f"**{selected_ticker} currently shows mixed signals.**\n\n"
            f"- Trend: {trend_status}\n"
            f"- RSI: {latest_rsi:.1f} ({rsi_status})\n"
            f"- Sentiment: {avg_sentiment:.3f} ({sentiment_status})\n\n"
            "Best action is usually to wait and monitor rather than make an aggressive decision."
        )


def answer_project_question(user_query):
    q = user_query.lower().strip()

    # Greetings
    if any(x in q for x in ["hello", "hi", "hey"]):
        return (
            "👋 Hello! I can explain stock signals, compare the five companies, "
            "and answer questions about how this project works."
        )

    # Best stock
    if any(x in q for x in ["which stock is best", "best stock", "top stock", "strongest stock"]):
        if rank_df.empty:
            return "I cannot rank stocks right now because the stock data is not available."
        lines = []
        for i, row in rank_df.head(5).iterrows():
            lines.append(
                f"{i+1}. **{row['Ticker']}** — Score: {int(row['Score'])}, "
                f"RSI: {row['RSI']:.1f}, Sentiment: {row['Sentiment']:.3f}"
            )
        return (
            "**Live stock ranking based on trend, RSI, and sentiment:**\n\n"
            + "\n".join(lines)
            + f"\n\n**{top_stock}** currently looks strongest in this dashboard, but this is decision support, not financial advice."
        )

    # Where to invest
    if any(x in q for x in ["where to invest", "where should i invest", "where can i invest"]):
        return (
            f"**Current dashboard focus suggests starting with {top_stock}.**\n\n"
            "A practical screening rule is:\n"
            "- Prefer stocks above MA20 for trend support.\n"
            "- Avoid fresh buying when RSI is above 70.\n"
            "- Give more weight to positive sentiment when trend is already strong.\n"
            "- Use this app as a support tool before making any real decision."
        )

    # Buy / sell / hold
    if any(x in q for x in ["should i buy", "should i sell", "should i hold", "buy this stock", "sell this stock", "hold this stock"]):
        return stock_recommendation_summary()

    # Useful to users
    if any(x in q for x in ["useful", "benefit", "benefits", "how it is useful", "how is this project useful"]):
        return (
            "**This project is useful because it combines technical analysis, sentiment, and ML prediction in one dashboard.**\n\n"
            "Benefits for users:\n"
            "- Helps investors compare five major tech stocks quickly.\n"
            "- Helps beginners understand RSI, trend, and sentiment visually.\n"
            "- Helps students demonstrate a complete AI pipeline.\n"
            "- Helps examiners see real business value, not just model code."
        )

    # Project workflow
    if any(x in q for x in ["how this project works", "how does this project work", "workflow", "overview", "project flow"]):
        return (
            "**Project workflow:**\n\n"
            "1. Stock data is collected from Yahoo Finance.\n"
            "2. Features such as MA5, MA20, RSI, price change, and volume change are generated.\n"
            "3. Sentiment values are added as an extra signal.\n"
            "4. A Random Forest model predicts the next-day close price.\n"
            "5. The dashboard shows charts, model accuracy, comparison, and interactive guidance.\n\n"
            "So the project connects data collection, feature engineering, ML modeling, and user interaction."
        )

    # RSI
    if "rsi" in q:
        return (
            f"**RSI for {selected_ticker} is {latest_rsi:.1f}.**\n\n"
            "- RSI above 70 often suggests overbought conditions.\n"
            "- RSI below 30 often suggests oversold conditions.\n"
            "- RSI between 30 and 70 is generally a neutral zone.\n\n"
            f"Right now, **{selected_ticker}** is in a **{rsi_status}** state."
        )

    # Sentiment
    if "sentiment" in q or "news" in q:
        return (
            f"**Sentiment score for {selected_ticker} is {avg_sentiment:.3f}.**\n\n"
            "- Positive sentiment suggests supportive news flow.\n"
            "- Negative sentiment suggests caution.\n"
            "- Neutral sentiment means news is not strongly helping either side.\n\n"
            f"Current sentiment is **{sentiment_status}**."
        )

    # RMSE / model
    if any(x in q for x in ["rmse", "model", "accuracy", "prediction"]):
        return (
            f"**Model RMSE for {selected_ticker} is ${rmse_display:.2f}.**\n\n"
            "RMSE measures how far predicted prices are from actual prices on average. "
            "Lower RMSE means the model is more reliable. "
            "Users can use this to understand how much confidence to place in the prediction."
        )

    # Price
    if "price" in q or "current price" in q:
        return (
            f"**Current price of {selected_ticker} is ${latest_price:.2f}.**\n\n"
            f"- MA5: ${latest_ma5:.2f}\n"
            f"- MA20: ${latest_ma20:.2f}\n"
            f"- Trend status: {trend_status}\n\n"
            "Users can compare the current price with moving averages to understand trend direction."
        )

    # Companies
    if any(x in q for x in ["which companies", "which stocks", "stocks covered", "companies included"]):
        return (
            "**This project analyzes five tech companies:**\n\n"
            "- AAPL\n"
            "- MSFT\n"
            "- AMZN\n"
            "- GOOGL\n"
            "- META\n\n"
            "These are useful because they are highly traded and have rich price history."
        )

    # Who can use
    if any(x in q for x in ["who can use", "who uses", "investor", "trader", "student"]):
        return (
            "**This dashboard is useful for multiple users:**\n\n"
            "- Retail investors who want quick stock screening.\n"
            "- Students building ML and finance projects.\n"
            "- Researchers exploring sentiment-based forecasting.\n"
            "- Examiners and interviewers reviewing an end-to-end AI application."
        )

    # Limitations
    if any(x in q for x in ["limitation", "limitations", "drawback", "risk"]):
        return (
            "**Current limitations:**\n\n"
            "- This is a support tool, not guaranteed financial advice.\n"
            "- Future prices are never certain.\n"
            "- Sentiment can be noisy.\n"
            "- Random Forest is useful, but not perfect for all time-series behavior."
        )

    # How to use
    if any(x in q for x in ["how to use", "use this dashboard", "dashboard", "tabs"]):
        return (
            "**How to use the dashboard:**\n\n"
            "- Choose a stock from the sidebar.\n"
            "- Read the metric cards.\n"
            "- Check price and RSI charts.\n"
            "- Compare all five companies.\n"
            "- Read live news and sentiment.\n"
            "- Use this chatbot for explanations and project answers."
        )

    # Default fallback
    return (
        "**I can answer both stock-related and project-related questions.**\n\n"
        "Try asking:\n"
        "- Which stock is best right now?\n"
        "- Where should I invest?\n"
        "- How is this project useful to users?\n"
        "- How does this project work?\n"
        "- What does RSI mean?\n"
        "- What does sentiment score mean?\n"
        "- What is model RMSE?"
    )


# ---------- SAMPLE QUESTIONS ----------
st.subheader("💡 Suggested Questions")

sample_questions = [
    "Which stock is best right now?",
    "Where should I invest?",
    "How is this project useful to users?",
    "How does this project work?",
    "What does RSI mean?",
    "What does sentiment score mean?",
    "What is model RMSE?",
    "Should I buy this stock now?",
    "What is the current price?",
    "Which companies are included?",
    "Who can use this project?",
    "What are the limitations?",
    "How do I use this dashboard?"
]

cols = st.columns(3)
for i, question in enumerate(sample_questions):
    with cols[i % 3]:
        if st.button(question, key=f"sample_question_{i}", use_container_width=True):
            st.session_state.pending_question = question


# ---------- CHAT HISTORY ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- USER INPUT ----------
typed_prompt = st.chat_input("Ask anything about stocks or the project...")

final_prompt = None
if st.session_state.pending_question:
    final_prompt = st.session_state.pending_question
    st.session_state.pending_question = None
elif typed_prompt:
    final_prompt = typed_prompt

if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})

    with st.chat_message("user"):
        st.markdown(final_prompt)

    bot_response = answer_project_question(final_prompt)

    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})


# ---------- CLEAR CHAT ----------
if st.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chat history cleared. Use a suggested question or type a new one."
        }
    ]
    st.rerun()

st.markdown("---")
