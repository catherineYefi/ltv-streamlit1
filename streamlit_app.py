import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="LTV Model", layout="wide")

st.title("💰 LTV / CAC Streamlit Dashboard")

# Inputs
st.sidebar.header("Input parameters")
avg_check = st.sidebar.number_input("Average check (₽)", value=10000, step=500)
purchases_per_year = st.sidebar.number_input("Purchases per year", value=2, step=1)
margin = st.sidebar.slider("Margin %", 0, 100, 60)
cac = st.sidebar.number_input("CAC (₽)", value=15000, step=500)
churn = st.sidebar.slider("Monthly churn %", 0.0, 100.0, 5.0)
discount_rate = st.sidebar.slider("Discount rate % (annual)", 0.0, 50.0, 12.0)
horizon = st.sidebar.selectbox("Horizon (months)", [12, 24, 36], index=2)

# Monthly revenue and margin
monthly_rev = avg_check * purchases_per_year / 12
monthly_margin = monthly_rev * (margin / 100)

# Survival probabilities
months = np.arange(1, horizon+1)
survival = (1 - churn/100) ** (months-1)

# Discount factor
discount = 1 / ((1 + discount_rate/100) ** (months/12))

# Cash flows
cf = monthly_margin * survival
dcf = cf * discount
ltv = dcf.sum()

# Dashboard
col1, col2, col3, col4 = st.columns(4)
col1.metric("LTV", f"{ltv:,.0f} ₽")
col2.metric("CAC", f"{cac:,.0f} ₽")
col3.metric("LTV / CAC", f"{ltv/cac:.2f}" if cac > 0 else "N/A")
# Break-even point
cumulative_cf = np.cumsum(dcf) - cac
breakeven = np.argmax(cumulative_cf > 0) + 1 if any(cumulative_cf > 0) else None
col4.metric("Breakeven (months)", breakeven if breakeven else "Not reached")

# Chart: cumulative CF
fig, ax = plt.subplots()
ax.plot(months, np.cumsum(dcf) - cac, label="Cumulative CF (discounted)")
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Months")
ax.set_ylabel("₽")
ax.legend()
st.pyplot(fig)
