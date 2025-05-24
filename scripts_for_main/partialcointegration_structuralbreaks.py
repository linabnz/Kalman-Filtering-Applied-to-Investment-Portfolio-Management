import os
import numpy as np
from models.cointegrationModel import PartialCointegrationModel
from models.structuralbreaksdetector import (
    generate_structural_break_dataset,
    StructuralBreakDetector,
)
from strategies.SAPT import PartialCointegrationTraderSAPT
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def run_partial_cointegration_with_structural_breaks(
    cointegrated_pairs, train_data, test_data, input_length, sectors
):
    results = []

    dataset = generate_structural_break_dataset(
        train_data,
        cointegrated_pairs,
        input_length=input_length,
        future_window=90,
        adf_threshold=0.2,
    )

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = StructuralBreakDetector(
        input_length=input_length, wavelet_widths=np.arange(1, 11)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    n_epochs = 3

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for spread, y_price, x_price, label in train_loader:
            optimizer.zero_grad()
            output = model(spread, y_price, x_price)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = (output > 0.5).float()
            acc = (pred == label).float().mean()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.2%}")

    test_dataset = generate_structural_break_dataset(
        test_data,
        cointegrated_pairs,
        input_length=input_length,
        future_window=90,
        adf_threshold=0.2,
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for spread, y_price, x_price, label in test_loader:
            output = model(spread, y_price, x_price)
            all_preds.append(output.cpu())
            all_labels.append(label.cpu())

    # Concaténer les résultats
    preds = torch.cat(all_preds).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten()

    binary_preds = (preds > 0.5).astype(int)

    print(f"Accuracy       : {accuracy_score(labels, binary_preds):.2%}")
    print(f"Precision      : {precision_score(labels, binary_preds):.2%}")
    print(f"Recall         : {recall_score(labels, binary_preds):.2%}")
    print(f"ROC AUC Score  : {roc_auc_score(labels, preds):.4f}")

    torch.save(model.state_dict(), "models/structural_breaks_model.pt")

    # Utilisation du modèle pour prédire les ruptures structurelles
    for i, (ticker1, ticker2, p_value) in enumerate(cointegrated_pairs):
        co_model = PartialCointegrationModel(significance_level=0.05)

        trader = PartialCointegrationTraderSAPT(
            co_model,
            entry_threshold=1.25,  # Valeur optimale selon le document
            stop_loss=0.05,
            profit_target=0.05,
            rolling_window=90,
            kalman_gain=0.7,  # Valeur optimale selon le document
        )

        trades = trader.run_backtest(test_data[ticker1], test_data[ticker2])

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

        sector1 = sectors.get(ticker1, "Inconnu")
        sector2 = sectors.get(ticker2, "Inconnu")
        same_sector = sector1 == sector2

        results.append(
            {
                "Ticker1": ticker1,
                "Ticker2": ticker2,
                "Sector1": sector1,
                "Sector2": sector2,
                "Same_Sector": same_sector,
                "p_value": p_value,
                "Beta": co_model.beta,
                "Intercept": co_model.intercept,
                "Rho": co_model.rho,
                "Kalman_Gain": co_model.kalman_gain,
                "Total_Return": pnl_total,
                "Mean_PnL": pnl_moyen,
                "Max_Drawdown": max_drawdown,
                "Sharpe": sharpe,
                "Nb_Trades": nb_trades,
            }
        )
        total_pnl = sum([result["Total_Return"] for result in results])
        mean_pnl = np.mean([result["Mean_PnL"] for result in results])
        print(f"\nRendement total de toutes les paires: {total_pnl:.4f}")
        print(f"Rendement moyen de toutes les paires: {mean_pnl:.4f}")
