import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

def setup_logging(name: str = __name__) -> logging.Logger:
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sports_predictor.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Carga la configuración desde el archivo YAML"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_directories():
    """Crea los directorios necesarios si no existen"""
    directories = [
        'data/raw',
        'data/processed',
        'models'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_api_key() -> str:
    """Obtiene la API key desde las variables de entorno"""
    api_key = os.getenv('FOOTBALL_API_KEY')
    if not api_key:
        raise ValueError("FOOTBALL_API_KEY no encontrada en variables de entorno")
    return api_key
