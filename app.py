import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of our web app
st.title("📈 Reliance Stock Analysis")
st.write("Analyzing 21 years of Reliance stock data (2000-2021)")

# Load data
df = pd.read_csv('RELIANCE.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.drop(columns=['Symbol', 'Series', 'Last', 'Trades', 
                       'Deliverable Volume', '%Deliverble'])
df = df.dropna()
df = df.reset_index(drop=True)

# Show raw data
st.subheader("📊 Raw Data")
st.write(df.head(10))

# Show basic stats
st.subheader("📋 Basic Statistics")
st.write(df.describe())

# Chart 1 - Close price
st.subheader("📈 Stock Price Over Time")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['Date'], df['Close'], color='royalblue', linewidth=1)
ax.set_title('Reliance Stock Price (2000-2021)')
ax.set_xlabel('Year')
ax.set_ylabel('Price (₹)')
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Chart 2 - Volume
st.subheader("📊 Trading Volume Over Time")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.bar(df['Date'], df['Volume'], color='orange', alpha=0.6)
ax2.set_title('Reliance Trading Volume (2000-2021)')
ax2.set_xlabel('Year')
ax2.set_ylabel('Shares Traded')
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# ML Model
st.subheader("🤖 Price Prediction Model")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df['Tomorrow'] = df['Close'].shift(-1)
df = df.dropna()

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Tomorrow']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

st.success(f"🎯 Model Accuracy: {score * 100:.2f}%")
st.write("This model predicts tomorrow's closing price based on today's Open, High, Low and Volume.")