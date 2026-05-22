import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from src.utils import setup_logging, load_config, ensure_directories

logger = setup_logging(__name__)

class FootballDataPreprocessor:
    """Clase para preprocesar datos de fútbol"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def load_raw_data(self) -> pd.DataFrame:
        """Carga todos los archivos CSV crudos y los consolida"""
        raw_path = Path(self.config['paths']['raw_data'])
        csv_files = list(raw_path.glob('matches_*.csv'))
        
        if not csv_files:
            raise FileNotFoundError("No se encontraron archivos de datos crudos")
        
        # Cargar el más reciente
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Cargando datos desde {latest_file}")
        
        df = pd.read_csv(latest_file)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia los datos"""
        # Convertir fechas
        df['date'] = pd.to_datetime(df['date'])
        
        # Eliminar partidos no finalizados
        df = df[df['status'] == 'FINISHED'].copy()
        
        # Eliminar duplicados
        df = df.drop_duplicates(subset=['match_id'])
        
        # Ordenar por fecha
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Datos limpios: {len(df)} registros")
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea la variable objetivo (resultado del partido)"""
        def get_result(row):
            if row['home_score'] > row['away_score']:
                return 'H'  # Home win
            elif row['home_score'] < row['away_score']:
                return 'A'  # Away win
            else:
                return 'D'  # Draw
        
        df['result'] = df.apply(get_result, axis=1)
        return df
    
    def calculate_rolling_stats(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calcula estadísticas móviles por equipo"""
        for window in windows:
            # Stats para local
            for team_type in ['home', 'away']:
                team_col = f'{team_type}_team_id'
                score_col = f'{team_type}_score'
                
                # Goles anotados
                df[f'{team_type}_goals_avg_{window}'] = df.groupby(team_col)[score_col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
                
                # Goles recibidos
                opponent_score = 'away_score' if team_type == 'home' else 'home_score'
                df[f'{team_type}_goals_conceded_avg_{window}'] = df.groupby(team_col)[opponent_score].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
        
        logger.info(f"Calculadas estadísticas móviles para ventanas: {windows}")
        return df
    
    def calculate_form(self, df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
        """Calcula la forma reciente de los equipos"""
        def points_from_result(result, is_home):
            if result == 'H':
                return 3 if is_home else 0
            elif result == 'A':
                return 0 if is_home else 3
            else:
                return 1
        
        df['home_points'] = df.apply(lambda x: points_from_result(x['result'], True), axis=1)
        df['away_points'] = df.apply(lambda x: points_from_result(x['result'], False), axis=1)
        
        # Forma local
        df['home_form'] = df.groupby('home_team_id')['home_points'].transform(
            lambda x: x.rolling(n_matches, min_periods=1).sum().shift(1)
        )
        
        # Forma visitante
        df['away_form'] = df.groupby('away_team_id')['away_points'].transform(
            lambda x: x.rolling(n_matches, min_periods=1).sum().shift(1)
        )
        
        logger.info(f"Calculada forma reciente ({n_matches} partidos)")
        return df
    
    def calculate_head_to_head(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula estadísticas históricas entre equipos"""
        # Crear clave única para cada enfrentamiento
        df['h2h_key'] = df.apply(
            lambda x: f"{min(x['home_team_id'], x['away_team_id'])}_{max(x['home_team_id'], x['away_team_id'])}", 
            axis=1
        )
        
        # Contar victorias históricas
        df['h2h_home_wins'] = 0
        df['h2h_away_wins'] = 0
        df['h2h_draws'] = 0
        
        for idx, row in df.iterrows():
            h2h_matches = df[
                (df['h2h_key'] == row['h2h_key']) & 
                (df.index < idx)
            ]
            
            if len(h2h_matches) > 0:
                home_wins = len(h2h_matches[
                    ((h2h_matches['home_team_id'] == row['home_team_id']) & (h2h_matches['result'] == 'H')) |
                    ((h2h_matches['away_team_id'] == row['home_team_id']) & (h2h_matches['result'] == 'A'))
                ])
                away_wins = len(h2h_matches[
                    ((h2h_matches['home_team_id'] == row['away_team_id']) & (h2h_matches['result'] == 'H')) |
                    ((h2h_matches['away_team_id'] == row['away_team_id']) & (h2h_matches['result'] == 'A'))
                ])
                draws = len(h2h_matches[h2h_matches['result'] == 'D'])
                
                df.at[idx, 'h2h_home_wins'] = home_wins
                df.at[idx, 'h2h_away_wins'] = away_wins
                df.at[idx, 'h2h_draws'] = draws
        
        logger.info("Calculadas estadísticas head-to-head")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Guarda los datos procesados"""
        ensure_directories()
        filepath = f"{self.config['paths']['processed_data']}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Datos procesados guardados en {filepath}")

def main():
    """Función principal de preprocesamiento"""
    logger.info("Iniciando preprocesamiento de datos")
    
    # Cargar configuración
    config = load_config()
    
    # Crear preprocesador
    preprocessor = FootballDataPreprocessor(config)
    
    # Cargar y limpiar datos
    df = preprocessor.load_raw_data()
    df = preprocessor.clean_data(df)
    
    # Crear variable objetivo
    df = preprocessor.create_target_variable(df)
    
    # Feature engineering
    windows = config['features']['rolling_windows']
    df = preprocessor.calculate_rolling_stats(df, windows)
    
    if config['features']['include_form']:
        df = preprocessor.calculate_form(df)
    
    if config['features']['include_head_to_head']:
        df = preprocessor.calculate_head_to_head(df)
    
    # Eliminar filas con NaN en features importantes
    df = df.dropna()
    
    # Guardar con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    preprocessor.save_processed_data(df, f'processed_data_{timestamp}.csv')
    
    logger.info("Preprocesamiento completado exitosamente")

if __name__ == "__main__":
    main()
