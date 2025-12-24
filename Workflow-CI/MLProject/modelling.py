"""
modelling.py - Wine Quality Model Training dengan MLflow Autolog
================================================================
Author: Syifa Fauziah
Course: Membangun Sistem Machine Learning - Dicoding

Script ini melatih model machine learning untuk prediksi wine quality
menggunakan MLflow autolog() TANPA manual logging.

Untuk melihat perbedaan dengan manual logging, lihat modelling_tuning.py

Usage:
    python modelling.py --data-dir winequality_preprocessing
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn


def load_data(data_dir):
    """Load preprocessed data dari folder."""
    print(f"Loading data dari {data_dir}...")
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')['quality']
    y_test = pd.read_csv(f'{data_dir}/y_test.csv')['quality']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def train_model_with_autolog(X_train, X_test, y_train, y_test, experiment_name):
    """
    Train model menggunakan MLflow autolog() SAJA.
    
    autolog() akan otomatis mencatat:
    - Parameters model
    - Metrics (training)
    - Model artifact
    - Model signature
    """
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Enable autolog - ini akan mencatat SEMUA secara otomatis
    mlflow.sklearn.autolog()
    
    print("\n" + "="*50)
    print("Training dengan MLflow autolog()")
    print("="*50)
    
    # Mulai run - autolog akan handle semua logging
    with mlflow.start_run():
        
        # Buat dan train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit model - autolog akan mencatat parameters
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Hitung metrics untuk ditampilkan (bukan untuk logging manual)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nHasil Evaluasi:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
    
    # Disable autolog setelah selesai
    mlflow.sklearn.autolog(disable=True)
    
    print("\n[INFO] autolog() telah mencatat secara otomatis:")
    print("  - Model parameters (n_estimators, max_depth, dll)")
    print("  - Training metrics")
    print("  - Model artifact (.pkl)")
    print("  - Model signature")
    print("  - Input example")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Wine Quality Model Training dengan Autolog')
    parser.add_argument('--data-dir', type=str, default='winequality_preprocessing',
                        help='Directory containing preprocessed data')
    parser.add_argument('--experiment-name', type=str, default='wine-quality-autolog',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    
    # Train dengan autolog
    model = train_model_with_autolog(
        X_train, X_test, y_train, y_test,
        args.experiment_name
    )
    
    print("\n" + "="*50)
    print("Training selesai!")
    print("="*50)
    print("\nLihat hasil di MLflow UI:")
    print("  mlflow ui --port 5000")
    print("\nUntuk manual logging, jalankan:")
    print("  python modelling_tuning.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
