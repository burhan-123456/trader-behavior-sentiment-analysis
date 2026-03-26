# 📊 Trader Behavior & Sentiment Analysis

## 🚀 Overview
This project analyzes trader behavior under different market sentiments (Fear, Greed, Neutral) and builds a machine learning system to:

- Predict next-day profitability of traders
- Cluster traders into behavioral archetypes
- Provide actionable trading insights
- Visualize results using an interactive Streamlit dashboard

---

## 🎯 Objectives
- Understand how sentiment affects trading behavior  
- Identify profitable vs risky trading patterns  
- Segment traders into meaningful groups  
- Build a predictive model for profitability  

---

## 📁 Dataset Used
- **Trader Dataset** → Trade-level data (PnL, size, direction, etc.)  
- **Sentiment Dataset** → Fear & Greed Index (market sentiment)  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Cleaned missing values  
- Converted timestamps to date  
- Merged trader data with sentiment data  
- Created new features:
  - `is_long`, `is_short`
  - `is_win`
  - Rolling features (`pnl_mean_5`, `win_rate_5`, `pnl_std_5`)  

---

### 2. Feature Engineering
- Lag features:
  - `pnl_lag1`
  - `pnl_lag2`  
- Rolling metrics:
  - Mean PnL  
  - Win rate  
  - Volatility (standard deviation)  

---

### 3. Predictive Modeling
- Model: Classification model  
- Target:1 → Profitable
- 0 → Loss
- Achieved Accuracy: 98 %

---

### 4. Clustering (Behavior Analysis)
- Algorithm: **KMeans Clustering**
- Traders grouped into behavioral segments:

| Cluster | Description |
|--------|------------|
| 0 | Medium Traders |
| 1 | Consistent Winners |
| 2 | High Risk Traders |
| 3 | Volatile / Losing Traders |
| 4 | Conservative Traders |

---

### 5. Dashboard (Streamlit)
Interactive dashboard includes:
- Sentiment distribution (bar + pie)
- PnL trends over time
- Win rate comparison
- Trade size distribution
- Cluster visualization (PCA)
- Profit prediction system
- Cluster prediction system

---

## 📊 Key Insights
- Fear markets lead to higher profitability and aggressive trading
- while Greed markets result in lower performance and more cautious behavior.
- Traders perform better during **Greed sentiment**
- Fear leads to **smaller trade sizes and cautious behavior**
- High-risk traders show **high volatility in PnL**
- Consistent winners maintain:
  - High win rate  
  - Low volatility  
- Some traders are profitable short-term but belong to unstable clusters  

---

## 💡 Strategy Recommendations

## 🔹 Strategy 1: Trade More During Fear, Be Careful During Greed

### 📌 Insight
- Higher profits and win rate observed during **Fear**
- Lower profits during **Greed**

### ✅ Action
- Increase trading activity during Fear
- Reduce trading and stay cautious during Greed

### 💡 Explanation
- Fear → High volatility → More opportunities  
- Greed → Stable market → Fewer opportunities  

---

## 🔹 Strategy 2: Use Small Trade Sizes

### 📌 Insight
- Small trade sizes lead to more consistent performance  
- Large trades often resulted in losses (especially during Greed)

### ✅ Action
- Avoid investing large capital in a single trade  
- Prefer smaller, controlled position sizes  

### 💡 Explanation
- Large trades = High risk  
- Small trades = Better risk management  

---

## 🔹 Strategy 3: Avoid Overtrading

### 📌 Insight
- Infrequent traders achieved higher profits  
- Frequent traders showed lower performance  

### ✅ Action
- Focus on fewer, high-quality trades  
- Avoid unnecessary repeated trading  

### 💡 Explanation
- More trades ≠ More profit  
- Quality trades > Quantity of trades  

---

## 🔹 Strategy 4: Adapt to Market Conditions

### 📌 Insight
- Trader behavior varies with sentiment:
  - Fear → More aggressive trading  
  - Greed → More cautious trading  

### ✅ Action
- During Fear → Be more active and seize opportunities  
- During Greed → Reduce risk and be selective  

### 💡 Explanation
- A single strategy does not work in all conditions  
- Adapt based on market sentiment  



---

## 🖥️ How to Run

```bash
streamlit run app.py
