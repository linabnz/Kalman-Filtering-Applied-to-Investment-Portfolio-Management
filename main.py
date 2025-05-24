import numpy as np
import json
from utils.download_jse_data import download_jse_data
import os
import pandas as pd
from utils.data_loader import split_data
from utils.statistical_tests import find_cointegrated_pairs
from scripts_for_main.simplecointegration import run_simple_cointegration
from scripts_for_main.partialcointegration import run_partial_cointegration
from scripts_for_main.partialcointegration_structuralbreaks import (
    run_partial_cointegration_with_structural_breaks,
)
from models.cointegrationModel import CointegrationModel, PartialCointegrationModel
from strategies.cointegrationTrader import (
    SimpleCointegrationTrader,
    PartialCointegrationTrader,
)
from models.structuralbreaksdetector import StructuralBreakDetector
from strategies.SAPT import PartialCointegrationTraderSAPT
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Lancement du téléchargement des données JSE...")
    data_file = "data/jse_stocks.csv"
    sectors_file = "data/jse_sectors.csv"

    if os.path.exists(data_file):
        print("Les données existent déjà. Chargement des données...")
        prices = pd.read_csv(data_file, index_col=0, parse_dates=True)
        sectors = pd.read_csv(sectors_file, index_col=0)
        print("Données chargées avec succès.")
    else:
        prices, sectors = download_jse_data(
            start_date="2000-01-01",
            end_date="2025-05-01",
            min_history_years=3,  # Aujourd'hui
        )
    print("\nTéléchargement terminé!")
    print("\nAperçu des prix téléchargés:")
    print(prices.head())

    print("\nInformations sur les données:")
    print(f"Période: du {prices.index[0]} au {prices.index[-1]}")
    print(f"Nombre de jours de trading: {len(prices)}")
    print(f"Nombre d'actions: {prices.shape[1]}")

    if sectors is not None:
        sector_counts = {}
        for sector in sectors.values.flatten():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        print("\nDistribution par secteur:")
        for sector, count in sector_counts.items():
            print(f"- {sector}: {count} actions")

    print("\n Séparation des données en jeu de formation et de trading")
    train_data, test_data = split_data(prices, train_ratio=0.7)

    print(f"Jeu de formation: {train_data.shape}")
    print(f"Jeu de test: {test_data.shape}")

    print("\nSélection des paires co-intégrées")
    print("\nRecherche de paires co-intégrées (cela peut prendre quelques minutes)...")
    cointegrated_pairs = find_cointegrated_pairs(train_data, significance_level=0.05)
    print(
        f"Trouvé {len(cointegrated_pairs)} paires co-intégrées sur toute la période disponible"
    )

    print("STRATEGIE 1 : CO-INTEGRATION TRADING SIMPLE SUR LES PAIRES CO-INTEGRÉES")
    model = CointegrationModel(significance_level=0.05)
    trader = SimpleCointegrationTrader(
        model,
        entry_threshold=1.25,
        stop_loss=0.05,
        profit_target=0.05,
        rolling_window=60,
    )
    run_simple_cointegration(cointegrated_pairs, trader, test_data, model, sectors)

    print("STRATEGIE 2 : CO-INTEGRATION PARTIELLE SUR LES PAIRES CO-INTEGRÉES")
    model = PartialCointegrationModel(significance_level=0.05)
    trader = PartialCointegrationTrader(
        model,
        entry_threshold=1.25,
        stop_loss=0.05,
        profit_target=0.05,
        rolling_window=60,
        kalman_gain=0.7,
    )

    run_partial_cointegration(cointegrated_pairs, trader, test_data, model, sectors)

    print(
        "STRATEGIE 3 : CO-INTEGRATION PARTIELLE AVEC PREDICTION DE BREAKS STRUCTURELS"
    )
    run_partial_cointegration_with_structural_breaks(
        cointegrated_pairs,
        train_data,
        test_data,
        input_length=90,
        sectors=sectors,
    )

    print("\nSTRATEGIE 4 : Q-LEARNING")
    print("\n==================== RL Reporting Module ====================")
    print(" Module : Deep Q-Learning on Co-integrated Stock Pairs")
    print(" Training: 100 Episodes")
    print(" Purpose : Visual summary of learning progression")
    print("============================================================\n")

    # Load rewards data
    with open(
        r"data\rewards_by_pair.json",
        "r",
    ) as f:
        rewards_by_pair = json.load(f)

    #  Rolling average plot
    plt.figure(figsize=(12, 6))
    window = 10
    for pair, rewards in rewards_by_pair.items():
        rewards_series = pd.Series(rewards)
        smooth = rewards_series.rolling(window=window).mean()
        plt.plot(smooth, label=pair)

    plt.title("Smoothed Reward Progression (Rolling Avg 10 episodes)", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Smoothed Reward", fontsize=12)
    plt.legend(title="Pairs", loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("rolling_rewards.png")
    plt.show()

    #  Final reward comparison
    final_rewards = {pair: rewards[-1] for pair, rewards in rewards_by_pair.items()}
    plt.figure(figsize=(10, 5))
    plt.bar(final_rewards.keys(), final_rewards.values(), color="lightgreen")
    plt.title("Final Total Reward at Episode 100", fontsize=14)
    plt.ylabel("Final Reward", fontsize=12)
    plt.xlabel("Pairs", fontsize=12)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("final_rewards_bar.png")
    plt.show()

    #  Summary table
    summary_data = []
    for pair, rewards in rewards_by_pair.items():
        start = rewards[0]
        end = rewards[-1]
        gain = end - start
        summary_data.append(
            {
                "Pair": pair,
                "Start Reward": round(start, 2),
                "End Reward": round(end, 2),
                "Total Gain": round(gain, 2),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="Total Gain", ascending=False)
    summary_df.to_csv("reward_summary.csv", index=False)

    print(" Generated:")
    print(" - rolling_rewards.png")
    print(" - final_rewards_bar.png")
    print(" - reward_summary.csv")
    print("Use these in your report or presentation.")
