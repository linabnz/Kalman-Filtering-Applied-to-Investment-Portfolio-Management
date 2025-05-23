import numpy as np


def run_simple_cointegration(cointegrated_pairs, trader, test_data, model, sectors):
    results = []

    for i, (ticker1, ticker2, p_value) in enumerate(cointegrated_pairs):
        sector1 = sectors.get(ticker1, "Inconnu")
        sector2 = sectors.get(ticker2, "Inconnu")
        same_sector = sector1 == sector2
        trades = trader.run_backtest(
            test_data[ticker1],
            test_data[ticker2],
        )
        print(
            f"\nAnalyse de la paire {i+1}: {ticker1} - {ticker2} (p-value: {p_value:.6f})"
        )

        if trades.empty:
            print("  ⚠️  Aucun trade réalisé.")
            continue

        pnl_total = trades["pnl"].sum()
        pnl_moyen = trades["pnl"].mean()
        nb_trades = len(trades)
        max_drawdown = trades["pnl"].min()

        pnl_values = trades["pnl"].values

        if len(pnl_values) >= 2 and np.std(pnl_values) > 0:
            sharpe = np.mean(pnl_values) / np.std(pnl_values) * np.sqrt(len(pnl_values))
        else:
            sharpe = 0

        print(
            f"  Performance: Rendement total: {pnl_total:.4f}, Sharpe: {sharpe:.4f}, Trades: {nb_trades}"
        )

        results.append(
            {
                "Ticker1": ticker1,
                "Ticker2": ticker2,
                "Sector1": sector1,
                "Sector2": sector2,
                "Same_Sector": same_sector,
                "p_value": p_value,
                "Beta": model.beta,
                "Intercept": model.intercept,
                "Total_Return": pnl_total,
                "Mean_PnL": pnl_moyen,
                "Max_Drawdown": max_drawdown,
                "Sharpe": sharpe,
                "Nb_Trades": nb_trades,
            }
        )

    # print total pnl and mean pnl
    total_pnl = sum([result["Total_Return"] for result in results])
    mean_pnl = np.mean([result["Mean_PnL"] for result in results])
    print(f"\nRendement total de toutes les paires: {total_pnl:.4f}")
    print(f"Rendement moyen de toutes les paires: {mean_pnl:.4f}")
