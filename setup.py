from setuptools import setup, find_packages

setup(
    name="sports-predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.6",
        "lightgbm>=4.0.0",
        "requests>=2.31.0",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
    ],
    author="BGLogic Solutions",
    description="Sistema automatizado de predicción deportiva para fútbol UK",
    python_requires=">=3.9",
)
