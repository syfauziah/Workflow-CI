"""
modelling.py - Wine Quality Model Training dengan MLflow
========================================================
Author: Syifa Fauziah
Course: Membangun Sistem Machine Learning - Dicoding

Dijalankan via: mlflow run . --env-manager=local
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn


def load_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    
    X_train = pd.read_csv('winequality_preprocessing/X_train.csv')
    X_test = pd.read_csv('winequality_preprocessing/X_test.csv')
    y_train = pd.read_csv('winequality_preprocessing/y_train.csv')['quality']
    y_test = pd.read_csv('winequality_preprocessing/y_test.csv')['quality']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, n_estimators, max_depth):
    """Train model dengan MLflow autolog."""
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    print("\n" + "="*50)
    print("Training dengan MLflow autolog()")
    print("="*50)
    
    with mlflow.start_run():
        # Create model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"\nParameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nEvaluation Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Wine Quality Model Training')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    
    args = parser.parse_args()
    
    print("="*50)
    print("Wine Quality Model Training with MLflow")
    print("Author: Syifa Fauziah")
    print("="*50)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model
    model = train_model(
        X_train, X_test, y_train, y_test,
        args.n_estimators,
        args.max_depth
    )
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == '__main__':
    main()
