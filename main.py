import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Charger les données
@st.cache_data
def load_data():
    prices = pd.read_csv('data/jse_stocks.csv', index_col='Date', parse_dates=True)
    results = pd.read_csv('results/cointegration_results.csv')
    return prices, results

def main():
    st.set_page_config(page_title="Kalman Portfolio Management", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Kalman Portfolio Management </h1>", unsafe_allow_html=True)

    prices, results = load_data()

    with st.sidebar:
        st.header(" Sélection d'une paire")
        results = results.sort_values(by='Sharpe', ascending=False).reset_index(drop=True)
        options = [f"{row['Ticker1']} - {row['Ticker2']} (Sharpe: {row['Sharpe']:.2f})" for _, row in results.iterrows()]
        choix = st.selectbox("Choisissez une paire :", options)
        start_analysis = st.button("Analyser cette paire")

    if start_analysis:
        selected_idx = options.index(choix)
        selected_pair = results.iloc[selected_idx]
        ticker1 = selected_pair['Ticker1']
        ticker2 = selected_pair['Ticker2']

        st.success(f"Paire sélectionnée : {ticker1} et {ticker2}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ticker 1", ticker1)
        with col2:
            st.metric("Ticker 2", ticker2)
        with col3:
            st.metric("Sharpe Ratio", f"{selected_pair['Sharpe']:.2f}")

        st.subheader(" Evolution normalisée des prix (Base 100)")

        common = prices[[ticker1, ticker2]].dropna()
        normalized = common / common.iloc[0] * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker1], name=ticker1))
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker2], name=ticker2))
        fig.update_layout(title="Prix normalisés", xaxis_title="Date", yaxis_title="Prix (base 100)",
                          legend_title="Tickers", height=600, template="plotly_white")
        
        st.plotly_chart(fig, use_container_width=True)

        st.balloons()

        st.info("Prochaines étapes : Application du Kalman Filter et trading sur le spread ")

if __name__ == "__main__":
    main()
