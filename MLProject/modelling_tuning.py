"""
Wine Quality Model Training with Hyperparameter Tuning
=======================================================
Author: Syifa Fauziah
Course: Membangun Sistem Machine Learning - Dicoding
Kriteria 2: Advanced (4 pts) - Manual logging with additional artifacts

This module implements hyperparameter tuning using Optuna with manual MLflow
logging (no autolog) and generates additional artifacts beyond standard metrics.

Features:
- Optuna hyperparameter optimization (50+ trials)
- Manual MLflow logging with 10+ metrics
- Additional artifacts: learning curves, hyperparameter importance,
  optimization history, cross-validation analysis
- DagsHub integration for online storage
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

import joblib
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, median_absolute_error
)

import mlflow
from mlflow.models.signature import infer_signature

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning class with Optuna optimization and MLflow tracking.
    
    Implements Advanced criteria requirements:
    - Manual logging (no autolog)
    - 10+ comprehensive metrics
    - Additional artifacts beyond standard MLflow autolog
    """
    
    SUPPORTED_MODELS = ['random_forest', 'gradient_boosting']
    
    def __init__(
        self,
        data_dir: str,
        experiment_name: str = 'wine-quality-tuning',
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            data_dir: Directory containing preprocessed data
            experiment_name: MLflow experiment name
            n_trials: Number of Optuna optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        self.best_model = None
        self.best_params = None
        self.study = None
        
        self.artifact_dir = 'artifacts'
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        logger.info(f'HyperparameterTuner initialized with {n_trials} trials')
    
    def setup_dagshub(self, repo_owner: str, repo_name: str) -> None:
        """
        Configure DagsHub integration for online MLflow tracking.
        
        Args:
            repo_owner: DagsHub repository owner username
            repo_name: DagsHub repository name
        """
        try:
            import dagshub
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            logger.info(f'DagsHub configured: {repo_owner}/{repo_name}')
        except ImportError:
            logger.warning('DagsHub not installed. Using local MLflow tracking.')
        except Exception as e:
            logger.warning(f'DagsHub setup failed: {e}. Using local tracking.')
    
    def load_data(self) -> None:
        """Load preprocessed training and test data."""
        logger.info(f'Loading data from {self.data_dir}')
        
        self.X_train = pd.read_csv(os.path.join(self.data_dir, 'X_train.csv'))
        self.X_test = pd.read_csv(os.path.join(self.data_dir, 'X_test.csv'))
        self.y_train = pd.read_csv(os.path.join(self.data_dir, 'y_train.csv')).values.ravel()
        self.y_test = pd.read_csv(os.path.join(self.data_dir, 'y_test.csv')).values.ravel()
        
        self.feature_names = self.X_train.columns.tolist()
        
        logger.info(f'Data loaded: {len(self.X_train)} train, {len(self.X_test)} test samples')
        logger.info(f'Features: {len(self.feature_names)}')
    
    def _get_param_space(self, model_type: str, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space for each model type.
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        if model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state
            }
        elif model_type == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
        else:
            raise ValueError(f'Unsupported model type: {model_type}')
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance with given parameters."""
        if model_type == 'random_forest':
            return RandomForestRegressor(**params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**params)
        else:
            raise ValueError(f'Unsupported model type: {model_type}')
    
    def _objective(self, trial: optuna.Trial, model_type: str) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            model_type: Type of model to optimize
            
        Returns:
            Mean cross-validation R2 score (to maximize)
        """
        params = self._get_param_space(model_type, trial)
        model = self._create_model(model_type, params)
        
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.cv_folds, scoring='r2', n_jobs=-1
        )
        
        return cv_scores.mean()
    
    def compute_comprehensive_metrics(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics (10+ metrics for Advanced criteria).
        
        Args:
            model: Trained model
            X_train, X_test: Feature matrices
            y_train, y_test: Target arrays
            
        Returns:
            Dictionary containing all computed metrics
        """
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.cv_folds, scoring='r2', n_jobs=-1
        )
        
        n = len(y_test)
        p = X_test.shape[1]
        r2_test = r2_score(y_test, y_test_pred)
        adjusted_r2 = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
        
        ss_res = np.sum((y_test - y_test_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        
        mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-8))) * 100
        
        metrics = {
            'mse_train': mean_squared_error(y_train, y_train_pred),
            'mse_test': mean_squared_error(y_test, y_test_pred),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae_train': mean_absolute_error(y_train, y_train_pred),
            'mae_test': mean_absolute_error(y_test, y_test_pred),
            'r2_train': r2_score(y_train, y_train_pred),
            'r2_test': r2_test,
            'adjusted_r2': adjusted_r2,
            'explained_variance': explained_variance_score(y_test, y_test_pred),
            'max_error': max_error(y_test, y_test_pred),
            'median_ae': median_absolute_error(y_test, y_test_pred),
            'mape': mape,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max(),
            'rss': ss_res,
            'tss': ss_tot,
            'overfit_ratio': r2_score(y_train, y_train_pred) / (r2_test + 1e-8)
        }
        
        return metrics
    
    def generate_learning_curve(
        self,
        model,
        model_type: str,
        save_path: str
    ) -> str:
        """
        Generate and save learning curve plot.
        
        Args:
            model: Model to evaluate
            model_type: Type of model
            save_path: Path to save the plot
            
        Returns:
            Path to saved figure
        """
        logger.info('Generating learning curve...')
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, self.X_train, self.y_train,
            cv=self.cv_folds,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2',
            n_jobs=-1,
            random_state=self.random_state
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='blue')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='orange')
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.plot(train_sizes, val_mean, 'o-', color='orange', label='Cross-Validation Score')
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title(f'Learning Curve - {model_type.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        gap = train_mean[-1] - val_mean[-1]
        ax.annotate(f'Final Gap: {gap:.4f}',
                    xy=(train_sizes[-1], val_mean[-1]),
                    xytext=(train_sizes[-1] * 0.7, val_mean[-1] - 0.1),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f'Learning curve saved: {save_path}')
        return save_path
    
    def generate_hyperparameter_importance(self, save_path: str) -> str:
        """
        Generate hyperparameter importance plot from Optuna study.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved figure
        """
        logger.info('Generating hyperparameter importance plot...')
        
        if self.study is None:
            raise ValueError('No Optuna study available. Run optimization first.')
        
        importance = optuna.importance.get_param_importances(self.study)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = list(importance.keys())
        values = list(importance.values())
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(params)))
        bars = ax.barh(params, values, color=colors, edgecolor='black')
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Hyperparameter Importance Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f'Hyperparameter importance saved: {save_path}')
        return save_path
    
    def generate_optimization_history(self, save_path: str) -> str:
        """
        Generate optimization history plot.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved figure
        """
        logger.info('Generating optimization history plot...')
        
        if self.study is None:
            raise ValueError('No Optuna study available. Run optimization first.')
        
        trials = self.study.trials
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]
        best_values = [max(values[:i+1]) for i in range(len(values))]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(trial_numbers, values, alpha=0.6, c='steelblue', edgecolor='white', s=50)
        axes[0].plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
        axes[0].set_xlabel('Trial Number', fontsize=12)
        axes[0].set_ylabel('Objective Value (R²)', fontsize=12)
        axes[0].set_title('Optimization History', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(max(values), color='red', linestyle='--', linewidth=2, label=f'Best: {max(values):.4f}')
        axes[1].axvline(np.mean(values), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.4f}')
        axes[1].set_xlabel('Objective Value (R²)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Objective Value Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f'Optimization history saved: {save_path}')
        return save_path
    
    def generate_cv_analysis(
        self,
        model,
        model_type: str,
        save_path: str
    ) -> str:
        """
        Generate cross-validation analysis plot.
        
        Args:
            model: Trained model
            model_type: Type of model
            save_path: Path to save the plot
            
        Returns:
            Path to saved figure
        """
        logger.info('Generating cross-validation analysis...')
        
        from sklearn.model_selection import cross_val_predict, KFold
        
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_predictions = cross_val_predict(model, self.X_train, self.y_train, cv=kfold)
        
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train)):
            y_val = self.y_train[val_idx]
            y_pred = cv_predictions[val_idx]
            fold_scores.append({
                'fold': fold + 1,
                'r2': r2_score(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'mae': mean_absolute_error(y_val, y_pred)
            })
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        folds = [s['fold'] for s in fold_scores]
        r2_scores = [s['r2'] for s in fold_scores]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(folds)))
        bars = ax1.bar(folds, r2_scores, color=colors, edgecolor='black')
        ax1.axhline(np.mean(r2_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r2_scores):.4f}')
        ax1.set_xlabel('Fold', fontsize=11)
        ax1.set_ylabel('R² Score', fontsize=11)
        ax1.set_title('R² Score by Fold', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(self.y_train, cv_predictions, alpha=0.5, c='steelblue', edgecolor='white', s=30)
        min_val = min(self.y_train.min(), cv_predictions.min())
        max_val = max(self.y_train.max(), cv_predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Quality', fontsize=11)
        ax2.set_ylabel('Predicted Quality', fontsize=11)
        ax2.set_title('CV Predictions vs Actual', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, :])
        metrics_df = pd.DataFrame(fold_scores)
        x = np.arange(len(folds))
        width = 0.25
        
        ax3.bar(x - width, metrics_df['r2'], width, label='R²', color='steelblue', edgecolor='black')
        ax3.bar(x, metrics_df['rmse'], width, label='RMSE', color='orange', edgecolor='black')
        ax3.bar(x + width, metrics_df['mae'], width, label='MAE', color='green', edgecolor='black')
        
        ax3.set_xlabel('Fold', fontsize=11)
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('All Metrics by Fold', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(folds)
        ax3.legend()
        ax3.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle(f'Cross-Validation Analysis - {model_type.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f'CV analysis saved: {save_path}')
        return save_path
    
    def run_optimization(self, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Run hyperparameter optimization with Optuna and log to MLflow.
        
        Args:
            model_type: Type of model to optimize
            
        Returns:
            Dictionary containing optimization results
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f'Model type must be one of {self.SUPPORTED_MODELS}')
        
        if self.X_train is None:
            self.load_data()
        
        logger.info(f'Starting hyperparameter optimization for {model_type}')
        logger.info(f'Number of trials: {self.n_trials}')
        
        mlflow.set_experiment(self.experiment_name)
        
        self.study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_tuning',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        self.study.optimize(
            lambda trial: self._objective(trial, model_type),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_params['random_state'] = self.random_state
        
        logger.info(f'Best parameters: {self.best_params}')
        logger.info(f'Best CV R²: {self.study.best_value:.4f}')
        
        with mlflow.start_run(run_name=f'{model_type}_tuned_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
            
            mlflow.log_params(self.best_params)
            
            mlflow.set_tag('model_type', model_type)
            mlflow.set_tag('optimization_method', 'Optuna TPE')
            mlflow.set_tag('n_trials', self.n_trials)
            mlflow.set_tag('cv_folds', self.cv_folds)
            mlflow.set_tag('author', 'Syifa Fauziah')
            mlflow.set_tag('course', 'Membangun Sistem Machine Learning - Dicoding')
            mlflow.set_tag('dataset', 'UCI Wine Quality')
            mlflow.set_tag('criteria', 'Advanced (4 pts)')
            
            self.best_model = self._create_model(model_type, self.best_params)
            self.best_model.fit(self.X_train, self.y_train)
            
            metrics = self.compute_comprehensive_metrics(
                self.best_model,
                self.X_train, self.X_test,
                self.y_train, self.y_test
            )
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info('Generating additional artifacts...')
            
            learning_curve_path = os.path.join(self.artifact_dir, 'learning_curve.png')
            self.generate_learning_curve(self.best_model, model_type, learning_curve_path)
            mlflow.log_artifact(learning_curve_path)
            
            hp_importance_path = os.path.join(self.artifact_dir, 'hyperparameter_importance.png')
            self.generate_hyperparameter_importance(hp_importance_path)
            mlflow.log_artifact(hp_importance_path)
            
            opt_history_path = os.path.join(self.artifact_dir, 'optimization_history.png')
            self.generate_optimization_history(opt_history_path)
            mlflow.log_artifact(opt_history_path)
            
            cv_analysis_path = os.path.join(self.artifact_dir, 'cross_validation_analysis.png')
            self.generate_cv_analysis(self.best_model, model_type, cv_analysis_path)
            mlflow.log_artifact(cv_analysis_path)
            
            signature = infer_signature(self.X_train, self.best_model.predict(self.X_train))
            mlflow.sklearn.log_model(
                self.best_model,
                artifact_path='model',
                signature=signature,
                registered_model_name=f'wine_quality_{model_type}'
            )
            
            model_path = os.path.join(self.artifact_dir, f'{model_type}_best_model.pkl')
            joblib.dump(self.best_model, model_path)
            mlflow.log_artifact(model_path)
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f'MLflow Run ID: {run_id}')
        
        results = {
            'model_type': model_type,
            'best_params': self.best_params,
            'best_cv_score': self.study.best_value,
            'metrics': metrics,
            'n_trials': self.n_trials,
            'run_id': run_id
        }
        
        results_path = os.path.join(self.artifact_dir, 'tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info('Hyperparameter tuning completed successfully')
        
        return results


def main():
    """Main entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description='Wine Quality Model Hyperparameter Tuning with MLflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python modelling_tuning.py --data-dir data --model-type random_forest --n-trials 50
  python modelling_tuning.py --data-dir data --model-type gradient_boosting --dagshub-repo username/repo
        '''
    )
    
    parser.add_argument('--data-dir', type=str, default='winequality_preprocessing',
                        help='Directory containing preprocessed data')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting'],
                        help='Type of model to tune')
    parser.add_argument('--experiment-name', type=str, default='wine-quality-tuning',
                        help='MLflow experiment name')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna optimization trials')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--dagshub-repo', type=str, default=None,
                        help='DagsHub repository (format: owner/repo)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    tuner = HyperparameterTuner(
        data_dir=args.data_dir,
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        random_state=args.random_state
    )
    
    if args.dagshub_repo:
        parts = args.dagshub_repo.split('/')
        if len(parts) == 2:
            tuner.setup_dagshub(parts[0], parts[1])
        else:
            logger.warning('Invalid DagsHub repo format. Expected: owner/repo')
    
    results = tuner.run_optimization(model_type=args.model_type)
    
    print('\n' + '=' * 60)
    print('HYPERPARAMETER TUNING RESULTS')
    print('=' * 60)
    print(f'Model Type: {results["model_type"]}')
    print(f'Best CV R²: {results["best_cv_score"]:.4f}')
    print(f'Test R²: {results["metrics"]["r2_test"]:.4f}')
    print(f'Test RMSE: {results["metrics"]["rmse_test"]:.4f}')
    print(f'MLflow Run ID: {results["run_id"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
