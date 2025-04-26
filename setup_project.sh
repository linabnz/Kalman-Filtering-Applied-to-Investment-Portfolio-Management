#!/bin/bash

# Définir le dossier de base
BASE_DIR="."

# Créer la structure de répertoires et fichiers sous le bon repo
mkdir -p "$BASE_DIR"/{data,models,utils,strategies}

# Créer les fichiers principaux
touch "$BASE_DIR"/main.py "$BASE_DIR"/requirements.txt "$BASE_DIR"/README.md

# Créer les fichiers dans models
touch "$BASE_DIR"/models/__init__.py \
      "$BASE_DIR"/models/cointegration.py \
      "$BASE_DIR"/models/partial_cointegration.py \
      "$BASE_DIR"/models/kalman_filter.py \
      "$BASE_DIR"/models/cnn_predictor.py \
      "$BASE_DIR"/models/rl_optimizer.py

# Créer les fichiers dans utils
touch "$BASE_DIR"/utils/__init__.py \
      "$BASE_DIR"/utils/data_loader.py \
      "$BASE_DIR"/utils/statistical_tests.py \
      "$BASE_DIR"/utils/visualization.py

# Créer les fichiers dans strategies
touch "$BASE_DIR"/strategies/__init__.py \
      "$BASE_DIR"/strategies/pairs_trading.py \
      "$BASE_DIR"/strategies/statistical_arbitrage.py \
      "$BASE_DIR"/strategies/portfolio_optimizer.py

echo "Structure de projet créée sous $BASE_DIR avec succès."
