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
        entry_threshold=2.5,
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



print("="*60)
print(" Q-LEARNING STRATEGY: TRAINING AND EVALUATION".center(60))
print("="*60)


def afficher_resultats_qlearning(json_path=r"C:\Users\lbenzemma\Desktop\Projets Master2 MOSEF\Kalman-Filtering-Applied-to-Investment-Portfolio-Management-1\rewards_by_pair.json"):
    print("\n" + "="*60)
    print(" RÉSULTATS Q-LEARNING PAR PAIRE 100 episodes".center(60))
    print("="*60)

    # Charger les rewards depuis le fichier
    with open(json_path, "r") as f:
        rewards_data = json.load(f)

    # Afficher les résultats par paire
    for pair, rewards in rewards_data.items():
        rewards = np.array(rewards)
        total_reward = np.sum(rewards)
        average_reward = np.mean(rewards)
        final_reward = rewards[-1]

        print(f"\n🔹 Résultats pour la paire : {pair}")
        print(f"    -  Reward total : {total_reward:.2f}")
        print(f"    -  Moyenne : {average_reward:.2f}")
        print(f"    -  Dernier reward : {final_reward:.2f}")

        for i in range(0, len(rewards), 20):
            batch = rewards[i:i+20]
            batch_mean = np.mean(batch)
            print(f"       Batch {i//20 + 1} (épisodes {i}-{i+len(batch)-1}) : Moyenne = {batch_mean:.2f}")

afficher_resultats_qlearning()