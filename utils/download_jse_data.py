import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time


def download_jse_data(start_date="2015-01-01", end_date=None, min_history_years=5):
    """
    Télécharge les données des actions JSE avec une structure sectorielle.

    Args:
        start_date: Date de début pour les données historiques
        end_date: Date de fin (par défaut: aujourd'hui)
        min_history_years: Nombre minimum d'années d'historique requis

    Returns:
        DataFrame avec les prix de clôture ajustés
    """

    os.makedirs("data", exist_ok=True)

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    jse_sectors = {
        "Resources": [
            "AGL.JO",  # Anglo American
            "BHP.JO",  # BHP Group
            "AMS.JO",  # Anglo American Platinum
            "ANG.JO",  # AngloGold Ashanti
            "GFI.JO",  # Gold Fields
            "IMP.JO",  # Impala Platinum
            "SOL.JO",  # Sasol
            "GLN.JO",  # Glencore
            "SSW.JO",  # Sibanye Stillwater
        ],
        "Financials": [
            "SBK.JO",  # Standard Bank
            "FSR.JO",  # FirstRand
            "NED.JO",  # Nedbank
            "ABG.JO",  # Absa Group
            "CPI.JO",  # Capitec
            "RMH.JO",  # RMB Holdings
            "SLM.JO",  # Sanlam
            "DSY.JO",  # Discovery
            "INP.JO",  # Investec
        ],
        "Industrials": [
            "BVT.JO",  # Bidvest
            "MNP.JO",  # Mondi
            "BAW.JO",  # Barloworld
            "SNH.JO",  # Steinhoff
            "KIO.JO",  # Kumba Iron Ore
            "SHP.JO",  # Shoprite
            "WHL.JO",  # Woolworths
            "PIK.JO",  # Pick n Pay
            "AVI.JO",  # AVI Ltd
        ],
        "Technology": [
            "NPN.JO",  # Naspers
            "PRX.JO",  # Prosus
            "MTN.JO",  # MTN Group
            "VOD.JO",  # Vodacom
            "TFG.JO",  # The Foschini Group
            "PPH.JO",  # Pepkor
            "MCG.JO",  # MultiChoice Group
            "RNI.JO",  # Reinet Investments
        ],
        "Consumer": [
            "BTI.JO",  # British American Tobacco
            "REM.JO",  # Remgro
            "DGH.JO",  # Distell Group
            "CLH.JO",  # City Lodge Hotels
            "TRU.JO",  # Truworths
            "SPP.JO",  # Spur Corporation
            "MRP.JO",  # Mr Price Group
        ],
        "Healthcare": [
            "NTC.JO",  # Netcare
            "MEI.JO",  # Mediclinic
            "APN.JO",  # Aspen Pharmacare
            "DCP.JO",  # Dis-Chem Pharmacies
            "ADH.JO",  # Adcock Ingram
        ],
        "Property": [
            "GRT.JO",  # Growthpoint Properties
            "RDF.JO",  # Redefine Properties
            "HYP.JO",  # Hyprop Investments
            "VKE.JO",  # Vukile Property Fund
        ],
        "Construction": [
            "MUR.JO",  # Murray & Roberts
            "WBO.JO",  # Wilson Bayly Holmes
            "RLO.JO",  # Reunert
            "AEG.JO",  # Aveng
        ],
    }

    all_tickers = []
    for sector, tickers in jse_sectors.items():
        all_tickers.extend(tickers)

    print(
        f"Téléchargement de données pour {len(all_tickers)} actions JSE de {start_date} à {end_date}..."
    )

    batch_size = 15
    all_data = pd.DataFrame()

    for i in range(0, len(all_tickers), batch_size):
        batch_tickers = all_tickers[i : i + batch_size]
        print(
            f"Téléchargement du lot {i//batch_size + 1}/{(len(all_tickers)-1)//batch_size + 1}: {', '.join(batch_tickers)}"
        )

        try:
            batch_data = yf.download(
                tickers=batch_tickers,
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
            )

            if len(batch_tickers) == 1:
                ticker = batch_tickers[0]
                batch_data = pd.DataFrame(
                    {ticker: batch_data["Close"]}, index=batch_data.index
                )
            else:
                close_data = pd.DataFrame()
                for ticker in batch_tickers:
                    if ticker in batch_data.columns:
                        close_data[ticker] = batch_data[ticker]["Close"]
                    else:
                        print(
                            f"Attention: {ticker} non trouvé dans les données téléchargées"
                        )
                batch_data = close_data

            if all_data.empty:
                all_data = batch_data
            else:
                all_data = pd.concat([all_data, batch_data], axis=1)

            time.sleep(2)

        except Exception as e:
            print(f"Erreur lors du téléchargement du lot {i//batch_size + 1}: {e}")
            time.sleep(5)

    print(
        f"\nDonnées téléchargées: {all_data.shape[0]} jours pour {all_data.shape[1]} actions"
    )

    missing_values = all_data.isnull().sum()
    print((missing_values > 0).sum())
    all_data = all_data.dropna(axis=1, how="any")
    print(f"Valeurs manquantes après traitement: {all_data.isnull().sum().sum()}")

    ticker_to_sector = {}
    for sector, tickers in jse_sectors.items():
        for ticker in tickers:
            if ticker in all_data.columns:
                ticker_to_sector[ticker] = sector

    all_data.to_csv("data/jse_stocks.csv")
    print(f"Données de prix enregistrées dans 'data/jse_stocks.csv'")

    pd.DataFrame(list(ticker_to_sector.items()), columns=["Ticker", "Sector"]).to_csv(
        "data/jse_sectors.csv", index=False
    )
    print(f"Information des secteurs enregistrée dans 'data/jse_sectors.csv'")

    weekly_returns = all_data.resample("W").last().pct_change().dropna()
    weekly_returns.to_csv("data/jse_weekly_returns.csv")

    stats = pd.DataFrame(
        {
            "Début": all_data.iloc[0],
            "Fin": all_data.iloc[-1],
            "Rendement(%)": (all_data.iloc[-1] / all_data.iloc[0] - 1) * 100,
            "Volatilité(%)": all_data.pct_change().std() * 100 * np.sqrt(252),
            "Secteur": [
                ticker_to_sector.get(ticker, "Inconnu") for ticker in all_data.columns
            ],
        }
    )

    stats.to_csv("data/jse_stats.csv")
    print(f"Statistiques enregistrées dans 'data/jse_stats.csv'")

    create_visualizations(all_data, ticker_to_sector)

    return all_data, ticker_to_sector


def create_visualizations(price_data, ticker_to_sector):
    """
    Crée des visualisations pour les données téléchargées.
    """
    os.makedirs("data/plots", exist_ok=True)

    sector_groups = {}
    for ticker, sector in ticker_to_sector.items():
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(ticker)

    for sector, tickers in sector_groups.items():
        if len(tickers) < 2:
            continue

        plt.figure(figsize=(12, 6))

        valid_tickers = [t for t in tickers if t in price_data.columns]
        if len(valid_tickers) < 2:
            continue

        sector_data = price_data[valid_tickers]

        normalized = sector_data.div(sector_data.iloc[0]) * 100

        for ticker in normalized.columns:
            plt.plot(normalized.index, normalized[ticker], label=ticker)

        plt.title(f"Évolution des prix - Secteur {sector} (base 100)")
        plt.xlabel("Date")
        plt.ylabel("Prix normalisé (base 100)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"data/plots/jse_sector_{sector}.png", dpi=200)
        plt.close()

    # 2. Matrice de corrélation des rendements
    returns = price_data.pct_change().dropna()
    corr_matrix = returns.corr()

    plt.figure(figsize=(14, 12))
    plt.matshow(corr_matrix, fignum=1, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar()

    # Ajouter les étiquettes
    tickers = corr_matrix.columns
    plt.xticks(range(len(tickers)), tickers, rotation=90)
    plt.yticks(range(len(tickers)), tickers)

    plt.title("Matrice de corrélation des rendements quotidiens")
    plt.tight_layout()
    plt.savefig("data/plots/jse_correlation_matrix.png", dpi=200)
    plt.close()

    print("Visualisations enregistrées dans le dossier 'data/plots/'")


if __name__ == "__main__":
    # Télécharger les données avec au moins 5 ans d'historique
    prices, sectors = download_jse_data(start_date="2015-01-01", min_history_years=5)

    print("\nTéléchargement et analyse terminés!")
