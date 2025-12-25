"""
modelling.py - Wine Quality Model Training dengan MLflow Autolog
================================================================
Author: Syifa Fauziah
Course: Membangun Sistem Machine Learning - Dicoding

Script ini dijalankan via: mlflow run MLProject

Usage:
    mlflow run . -P data_dir=winequality_preprocessing -P n_estimators=100
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

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


def train_model(X_train, X_test, y_train, y_test, n_estimators, max_depth, min_samples_split, random_state):
    """
    Train model menggunakan MLflow autolog().
    """
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    print("\n" + "="*50)
    print("Training dengan MLflow autolog()")
    print("="*50)
    
    with mlflow.start_run():
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        print(f"\nParameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  min_samples_split: {min_samples_split}")
        print(f"  random_state: {random_state}")
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nHasil Evaluasi:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
    
    mlflow.sklearn.autolog(disable=True)
    
    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}


def main():
    parser = argparse.ArgumentParser(description='Wine Quality Model Training')
    parser.add_argument('--data-dir', type=str, default='winequality_preprocessing')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*50)
    print("Wine Quality Model Training with MLflow")
    print("Author: Syifa Fauziah")
    print("="*50)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    
    # Train model
    model, metrics = train_model(
        X_train, X_test, y_train, y_test,
        args.n_estimators,
        args.max_depth,
        args.min_samples_split,
        args.random_state
    )
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
