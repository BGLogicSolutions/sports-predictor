import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import time
from src.utils import setup_logging, load_config, ensure_directories, get_api_key

logger = setup_logging(__name__)

class FootballDataExtractor:
    """Clase para extraer datos de la API de football-data.org"""
    
    def __init__(self, api_key: str, config: Dict):
        self.api_key = api_key
        self.config = config
        self.base_url = config['api']['base_url']
        self.headers = {'X-Auth-Token': api_key}
        
    def get_matches(self, competition_id: int, season: int) -> pd.DataFrame:
        """Obtiene los partidos de una competición y temporada"""
        url = f"{self.base_url}/competitions/{competition_id}/matches"
        params = {'season': season}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            matches = []
            for match in data.get('matches', []):
                match_data = {
                    'match_id': match['id'],
                    'date': match['utcDate'],
                    'status': match['status'],
                    'matchday': match['matchday'],
                    'home_team': match['homeTeam']['name'],
                    'home_team_id': match['homeTeam']['id'],
                    'away_team': match['awayTeam']['name'],
                    'away_team_id': match['awayTeam']['id'],
                    'home_score': match['score']['fullTime']['home'],
                    'away_score': match['score']['fullTime']['away'],
                }
                matches.append(match_data)
            
            df = pd.DataFrame(matches)
            logger.info(f"Extraídos {len(df)} partidos de la competición {competition_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener datos: {e}")
            raise
    
    def get_team_stats(self, team_id: int) -> Dict:
        """Obtiene estadísticas de un equipo"""
        url = f"{self.base_url}/teams/{team_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener estadísticas del equipo {team_id}: {e}")
            return {}
    
    def save_raw_data(self, df: pd.DataFrame, filename: str):
        """Guarda los datos crudos en formato CSV"""
        ensure_directories()
        filepath = f"{self.config['paths']['raw_data']}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Datos guardados en {filepath}")

def main():
    """Función principal de extracción de datos"""
    logger.info("Iniciando extracción de datos")
    
    # Cargar configuración
    config = load_config()
    api_key = get_api_key()
    
    # Crear extractor
    extractor = FootballDataExtractor(api_key, config)
    
    # Extraer datos
    competition_id = config['api']['competition_id']
    season = config['api']['season']
    
    df_matches = extractor.get_matches(competition_id, season)
    
    # Guardar con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    extractor.save_raw_data(df_matches, f'matches_{timestamp}.csv')
    
    logger.info("Extracción completada exitosamente")

if __name__ == "__main__":
    main()
