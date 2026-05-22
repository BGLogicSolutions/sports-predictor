# Sports Predictor (Football UK)

Este proyecto es un sistema automatizado de predicción deportiva diseñado para el análisis de la liga de fútbol del Reino Unido. Utiliza una arquitectura basada en la nube que integra **Google Drive** para el almacenamiento de datos históricos, **Google Colab** para el procesamiento y modelado de datos, y **GitHub** para el control de versiones y despliegue del código.

## Estructura del Proyecto

La estructura está diseñada para mantener una separación clara entre los datos crudos, los datos procesados y la lógica de predicción:

```text
/
├── data/
│   ├── raw/             # Archivos CSV crudos (descargados diariamente)
│   └── processed/       # Archivos CSV limpios y enriquecidos para el modelo
├── notebooks/
│   ├── 01_data_extraction.ipynb  # Extracción vía API y guardado en Drive
│   ├── 02_preprocessing.ipynb    # Limpieza, ingeniería de características y consolidación
│   └── 03_training_model.ipynb   # Entrenamiento del modelo ML y generación de predicciones
├── models/              # Almacenamiento de modelos entrenados (.pkl o .joblib)
├── requirements.txt     # Dependencias de Python
└── README.md            # Documentación del proyecto
