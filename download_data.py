from utils.download_jse_data import download_jse_data

if __name__ == "__main__":
    print("Lancement du téléchargement des données JSE...")
    

    prices, sectors = download_jse_data(
        start_date='2000-01-01', 
        end_date=None,  # Aujourd'hui
        min_history_years=3
    )
    
    print("\nTéléchargement terminé!")
    

    print("\nAperçu des prix téléchargés:")
    print(prices.head())
    
    print("\nInformations sur les données:")
    print(f"Période: du {prices.index[0]} au {prices.index[-1]}")
    print(f"Nombre de jours de trading: {len(prices)}")
    print(f"Nombre d'actions: {prices.shape[1]}")
    
    if sectors:
        sector_counts = {}
        for sector in sectors.values():
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        print("\nDistribution par secteur:")
        for sector, count in sector_counts.items():
            print(f"- {sector}: {count} actions")

