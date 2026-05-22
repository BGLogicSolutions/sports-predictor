import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from src.utils import setup_logging, load_config, ensure_directories

logger = setup_logging(__name__)

class FootballPredictor:
    """Clase para entrenar modelos de predicción de fútbol"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.feature_columns = []
        
    def load_processed_data(self) -> pd.DataFrame:
        """Carga los datos procesados más recientes"""
        processed_path = Path(self.config['paths']['processed_data'])
        csv_files = list(processed_path.glob('processed_data_*.csv'))
        
        if not csv_files:
            raise FileNotFoundError("No se encontraron archivos de datos procesados")
        
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Cargando datos procesados desde {latest_file}")
        
        df = pd.read_csv(latest_file)
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara features y target para el modelo"""
        # Definir columnas de features
        exclude_cols = [
            'match_id', 'date', 'status', 'home_team', 'away_team',
            'home_team_id', 'away_team_id', 'home_score', 'away_score',
            'result', 'h2h_key', 'home_points', 'away_points'
        ]
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns]
        y = df['result']
        
        logger.info(f"Features preparadas: {len(self.feature_columns)} columnas")
        logger.info(f"Distribución de clases: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Entrena un modelo Random Forest"""
        logger.info("Entrenando Random Forest...")
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.config['model']['random_state'],
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            rf_model, X_train, y_train, 
            cv=self.config['model']['cv_folds'], 
            scoring='accuracy'
        )
        logger.info(f"Random Forest CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return rf_model
    
    def train_xgboost(self, X_train, y_train) -> xgb.XGBClassifier:
        """Entrena un modelo XGBoost"""
        logger.info("Entrenando XGBoost...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config['model']['random_state'],
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(
            xgb_model, X_train, y_train,
            cv=self.config['model']['cv_folds'],
            scoring='accuracy'
        )
        logger.info(f"XGBoost CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return xgb_model
    
    def train_lightgbm(self, X_train, y_train) -> lgb.LGBMClassifier:
        """Entrena un modelo LightGBM"""
        logger.info("Entrenando LightGBM...")
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config['model']['random_state'],
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(
            lgb_model, X_train, y_train,
            cv=self.config['model']['cv_folds'],
            scoring='accuracy'
        )
        logger.info(f"LightGBM CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return lgb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Evalúa el modelo en el conjunto de prueba"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluación de {model_name}")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        return accuracy
    
    def save_model(self, model, model_name: str):
        """Guarda el modelo entrenado"""
        ensure_directories()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"{self.config['paths']['models']}/{model_name}_{timestamp}.pkl"
        
        joblib.dump(model, filepath)
        logger.info(f"Modelo {model_name} guardado en {filepath}")
        
        # Guardar también el mejor modelo sin timestamp
        best_filepath = f"{self.config['paths']['models']}/{model_name}_best.pkl"
        joblib.dump(model, best_filepath)
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Entrena todos los modelos configurados"""
        algorithms = self.config['model']['algorithms']
        results = {}
        
        for algo in algorithms:
            if algo == 'random_forest':
                model = self.train_random_forest(X_train, y_train)
                accuracy = self.evaluate_model(model, X_test, y_test, 'Random Forest')
                self.models['random_forest'] = model
                results['random_forest'] = accuracy
                self.save_model(model, 'random_forest')
                
            elif algo == 'xgboost':
                model = self.train_xgboost(X_train, y_train)
                accuracy = self.evaluate_model(model, X_test, y_test, 'XGBoost')
                self.models['xgboost'] = model
                results['xgboost'] = accuracy
                self.save_model(model, 'xgboost')
                
            elif algo == 'lightgbm':
                model = self.train_lightgbm(X_train, y_train)
                accuracy = self.evaluate_model(model, X_test, y_test, 'LightGBM')
                self.models['lightgbm'] = model
                results['lightgbm'] = accuracy
                self.save_model(model, 'lightgbm')
        
        # Seleccionar el mejor modelo
        best_model_name = max(results, key=results.get)
        logger.info(f"\n{'='*50}")
        logger.info(f"Mejor modelo: {best_model_name} con accuracy: {results[best_model_name]:.4f}")
        logger.info(f"{'='*50}\n")
        
        return results

def main():
    """Función principal de entrenamiento"""
    logger.info("Iniciando entrenamiento de modelos")
    
    # Cargar configuración
    config = load_config()
    
    # Crear predictor
    predictor = FootballPredictor(config)
    
    # Cargar datos procesados
    df = predictor.load_processed_data()
    
    # Preparar features
    X, y = predictor.prepare_features(df)
    
    # Split train/test
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Datos divididos: Train={len(X_train)}, Test={len(X_test)}")
    
    # Entrenar modelos
    results = predictor.train_all_models(X_train, y_train, X_test, y_test)
    
    logger.info("Entrenamiento completado exitosamente")

if __name__ == "__main__":
    main()
