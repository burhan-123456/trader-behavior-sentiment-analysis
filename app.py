import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
st.markdown("""
<style>

/* Label text (Sentiment, Trade Size, etc.) */
label {
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* Input box numbers/text */
input {
    font-size: 18px !important;
}

/* Dropdown text */
div[data-baseweb="select"] {
    font-size: 18px !important;
}

/* Slider labels */
div[data-baseweb="slider"] {
    font-size: 16px !important;
}

/* Buttons */
button {
    font-size: 16px !important;
}

/* Section spacing */
.stNumberInput, .stSelectbox, .stSlider {
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.figsize': (3, 2),
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7
})

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("final_clustered_data.csv")
cluster_summary = pd.read_csv("cluster_summary.csv")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Trader Behavior & Sentiment Dashboard")

# -------------------------------
# FILTER (TOP)
# -------------------------------
st.subheader("🔍 Select Sentiment")

sentiment_filter = st.multiselect(
    "Choose Sentiment",
    df['sentiment'].unique(),
    default=df['sentiment'].unique()
)

df_filtered = df[df['sentiment'].isin(sentiment_filter)]

# Handle empty selection
if df_filtered.empty:
    st.warning("⚠️ No data available for selected filters")
    st.stop()

# -------------------------------
# OVERVIEW
# -------------------------------
st.subheader("📌 Dataset Overview")
st.write("Total Records:", df_filtered.shape[0])
st.dataframe(df_filtered.head())

# -------------------------------
# ROW 1 (BAR + PIE)
# -------------------------------
col1, col2 = st.columns(2)

sentiment_counts = df_filtered['sentiment'].value_counts()

with col1:
    st.subheader("📊 Sentiment (Bar)")
    fig, ax = plt.subplots(figsize=(2,2))
    sentiment_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel("Sentiment", fontsize=5)
    ax.set_ylabel("Count", fontsize=5)
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.tick_params(axis='y', labelsize=7)
    st.pyplot(fig)

with col2:
    st.subheader("🥧 Sentiment (Pie)")
    fig, ax = plt.subplots(figsize=(2,2))
    sentiment_counts.plot.pie(autopct='%1.1f%%', ax=ax, textprops={'fontsize': 5})
    ax.set_ylabel("")
    st.pyplot(fig)

# -------------------------------
# ROW 2
# -------------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("💰 Avg PnL")
    fig, ax = plt.subplots(figsize=(2,2))
    df_filtered.groupby('sentiment')['closed_pnl'].mean().plot(kind='bar', ax=ax)
    ax.set_xlabel("Sentiment", fontsize=8)
    ax.set_ylabel("Avg PnL", fontsize=8)
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.tick_params(axis='y', labelsize=7)
    st.pyplot(fig)

with col4:
    st.subheader("🎯 Win Rate")
    fig, ax = plt.subplots(figsize=(2,2))
    df_filtered.groupby('sentiment')['is_win'].mean().plot(kind='bar', ax=ax)
    ax.set_xlabel("Sentiment", fontsize=8)
    ax.set_ylabel("Win Rate", fontsize=8)
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.tick_params(axis='y', labelsize=7)
    st.pyplot(fig)

# -------------------------------
# ROW 3
# -------------------------------
col5, col6 = st.columns(2)

df_filtered['date'] = pd.to_datetime(df_filtered['date'])
df_sorted = df_filtered.sort_values('date')

with col5:
    st.subheader("📈 PnL Trend")
    fig, ax = plt.subplots(figsize=(2,2))
    df_sorted.groupby('date')['closed_pnl'].mean().plot(ax=ax)
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("PnL", fontsize=8)
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.tick_params(axis='y', labelsize=7)
    st.pyplot(fig)

with col6:
    st.subheader("📊 Trade Size")
    fig, ax = plt.subplots(figsize=(2,2))
    df_filtered['size_usd'].plot(kind='hist', bins=20, ax=ax)
    ax.set_xlabel("Trade Size", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    st.pyplot(fig)

# -------------------------------
# ROW 4 (CLUSTER)
# -------------------------------
col7, col8 = st.columns(2)

with col7:
    st.subheader("🧠 Cluster Count")
    fig, ax = plt.subplots(figsize=(3,2))

    df_filtered['cluster'].value_counts().plot(kind='bar', ax=ax)

    ax.set_xlabel("Cluster", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)

    plt.tight_layout()
    st.pyplot(fig)


with col8:
    st.subheader("📊 Cluster PnL")
    fig, ax = plt.subplots(figsize=(3,2))

    df_filtered.groupby('cluster')['closed_pnl'].mean().plot(kind='bar', ax=ax)

    ax.set_xlabel("Cluster", fontsize=8)
    ax.set_ylabel("Avg PnL", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)

    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------
# CLUSTER 2D VISUALIZATION (FIXED)
# -------------------------------
st.subheader("📍 Cluster Visualization")

# Features used for clustering
features = [
    'value', 'sentiment_num', 'size_usd',
    'is_long', 'is_short',
    'pnl_lag1', 'pnl_lag2',
    'pnl_mean_5', 'win_rate_5', 'pnl_std_5'
]

X = df_filtered[features]

# Scale data
X_scaled = scaler.transform(X)

# PCA transformation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
fig, ax = plt.subplots(figsize=(3,2.5))

scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df_filtered['cluster'],
    cmap='viridis',
    alpha=0.9,
    s=15
)

plt.colorbar(scatter, ax=ax)
ax.set_xlabel("PCA 1", fontsize=8)
ax.set_ylabel("PCA 2", fontsize=8)

ax.tick_params(axis='both', labelsize=7)



st.pyplot(fig)

# -------------------------------
# CLUSTER SUMMARY
# -------------------------------
st.markdown("""
<div style="padding:10px; border-radius:10px; background-color:#f0f2f6;">
    <p style="font-size:27px;">🟢 <b>Cluster 1:</b> Consistent Winners → High win rate</p>
    <p style="font-size:27px;">🔵 <b>Cluster 0:</b> Medium Traders → Balanced</p>
    <p style="font-size:27px;">🟡 <b>Cluster 2:</b> High Risk → High return</p>
    <p style="font-size:27px;">🟣 <b>Cluster 3:</b> Losing Traders</p>
    <p style="font-size:27px;">🟤 <b>Cluster 4:</b> Conservative</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# USER INPUT
# -------------------------------
st.markdown("<h2 style='font-size:24px;'>🔮 Predict</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    value = st.slider("Sentiment Value", 0, 100, 50)
    sentiment_num = st.selectbox("Sentiment (0=Fear,1=Greed)", [0,1])
    size_usd = st.number_input("Trade Size", value=1000.0)

    pnl_lag1 = st.number_input("PnL Lag1", 0.0)
    pnl_lag2 = st.number_input("PnL Lag2", 0.0)

with col2:
    is_long = st.selectbox("Is Long", [0,1])
    is_short = st.selectbox("Is Short", [0,1])

    pnl_mean_5 = st.number_input("PnL Mean 5", 0.0)
    win_rate_5 = st.slider("Win Rate 5", 0.0, 1.0, 0.5)
    pnl_std_5 = st.number_input("PnL Std 5", 0.0)

# -------------------------------
# PREP INPUT
# -------------------------------
input_data = np.array([[ 
    value, sentiment_num, size_usd,
    is_long, is_short,
    pnl_lag1, pnl_lag2,
    pnl_mean_5, win_rate_5, pnl_std_5
]])

input_scaled = scaler.transform(input_data)

# -------------------------------
# BUTTONS
# -------------------------------
col9, col10 = st.columns(2)

with col9:
    if st.button("Predict Profit"):
        pred = model.predict(input_scaled)[0]
        if pred == 1:
            st.success("✅ Profitable")
        else:
            st.error("❌ Loss")

with col10:
    if st.button("Predict Cluster"):
        cluster_pred = kmeans.predict(input_scaled)[0]

        cluster_names = {
            0: "Medium Trader",
            1: "Consistent Winner",
            2: "High Risk Trader",
            3: "Volatile / Unstable Trader",
            4: "Conservative Trader"
        }

        st.info(f"Cluster: {cluster_pred} → {cluster_names[cluster_pred]}")