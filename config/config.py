# Configuración del API
api:
  base_url: "https://api.football-data.org/v4"
  competition_id: 2021  # Premier League
  season: 2024
  
# Rutas de datos
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  models: "models"

# Configuración del modelo
model:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  algorithms:
    - random_forest
    - xgboost
    - lightgbm

# Features a generar
features:
  rolling_windows: [3, 5, 10]
  include_head_to_head: true
  include_form: true
  include_home_away_stats: true

# Predicción
prediction:
  min_probability: 0.6
  output_format: "csv"
