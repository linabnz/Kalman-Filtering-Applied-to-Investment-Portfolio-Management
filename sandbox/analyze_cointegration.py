import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.data_loader import load_stock_data, split_data
from utils.statistical_tests import find_cointegrated_pairs
from models.cointegration import CointegrationModel

def analyze_jse_cointegration(price_file='jse_stocks.csv', sector_file='jse_sectors.csv'):
    """
    Analyse les données de la JSE pour identifier les paires co-intégrées.
    """

    os.makedirs('results', exist_ok=True)
   
    data_path = os.path.join('data', price_file)
    print(f"Chargement des données à partir de {data_path}...")
    
    prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
   
    try:
        sector_path = os.path.join('data', sector_file)
        sectors_df = pd.read_csv(sector_path)
        sectors = dict(zip(sectors_df['Ticker'], sectors_df['Sector']))
        print(f"Informations sectorielles chargées pour {len(sectors)} actions")
    except:
        sectors = {}
        print("Informations sectorielles non disponibles")
    

    train_data, test_data = split_data(prices, train_ratio=0.7)
    print(f"Données divisées: entraînement {train_data.shape}, test {test_data.shape}")
    
  
    print("\nRecherche de paires co-intégrées (cela peut prendre quelques minutes)...")
    cointegrated_pairs = find_cointegrated_pairs(train_data)
    
    print(f"Trouvé {len(cointegrated_pairs)} paires co-intégrées")
    
    
    cointegrated_pairs.sort(key=lambda x: x[2])
    top_pairs = cointegrated_pairs[:20]  
    

    results = []
    
    for i, (ticker1, ticker2, p_value) in enumerate(top_pairs):
        print(f"\nAnalyse de la paire {i+1}: {ticker1} - {ticker2} (p-value: {p_value:.6f})")
        
       
        sector1 = sectors.get(ticker1, "Inconnu")
        sector2 = sectors.get(ticker2, "Inconnu")
        same_sector = sector1 == sector2
        
        print(f"  Secteurs: {ticker1} ({sector1}), {ticker2} ({sector2})")
        print(f"  Même secteur: {'Oui' if same_sector else 'Non'}")
        
       
        model = CointegrationModel()
        model.fit(train_data[ticker1], train_data[ticker2])
        
        print(f"  Coefficient de régression (beta): {model.beta:.4f}")
        print(f"  Constante: {model.intercept:.4f}")
        
        
        z_scores = model.calculate_z_scores(window=20)
        
        
        positions = np.zeros_like(z_scores)
        positions[z_scores < -2] = 1    # Long when spread is low
        positions[z_scores > 2] = -1    # Short when spread is high
        
      
        spread_returns = np.diff(model.residuals)
        strategy_returns = positions[:-1] * spread_returns
        
        
        total_return = np.sum(strategy_returns)
        annualized_return = total_return / (len(train_data) / 252)
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        print(f"  Performance: Rendement total: {total_return:.4f}, Sharpe: {sharpe:.4f}")
        
        
        results.append({
            'Ticker1': ticker1,
            'Ticker2': ticker2,
            'Sector1': sector1,
            'Sector2': sector2,
            'Same_Sector': same_sector,
            'p_value': p_value,
            'Beta': model.beta,
            'Intercept': model.intercept,
            'Total_Return': total_return,
            'Sharpe': sharpe
        })
        
        
        plt.figure(figsize=(12, 10))
        
       
        plt.subplot(3, 1, 1)
        norm1 = train_data[ticker1] / train_data[ticker1].iloc[0]
        norm2 = train_data[ticker2] / train_data[ticker2].iloc[0]
        plt.plot(train_data.index, norm1, label=ticker1)
        plt.plot(train_data.index, norm2, label=ticker2)
        plt.title(f'Prix normalisés: {ticker1} vs {ticker2}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 1, 2)
        plt.plot(train_data.index, model.residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Spread (résidus)')
        plt.grid(True, alpha=0.3)
        
      
        plt.subplot(3, 1, 3)
        plt.plot(train_data.index, z_scores)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.axhline(y=2, color='g', linestyle='--')
        plt.axhline(y=-2, color='g', linestyle='--')
        plt.title('Z-scores avec seuils de trading (±2)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/pair_{ticker1}_{ticker2}.png')
        plt.close()
    
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/cointegration_results.csv', index=False)
    print(f"\nRésultats enregistrés dans 'results/cointegration_results.csv'")
    
  
    print("\nRésumé des 5 meilleures paires co-intégrées:")
    summary = results_df[['Ticker1', 'Ticker2', 'Sector1', 'Sector2', 'Same_Sector', 'p_value', 'Sharpe']].head(5)
    print(summary)
    
    return results_df

if __name__ == "__main__":
    analyze_jse_cointegration()