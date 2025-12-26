"""
prepare_data.py - Download dan preprocessing Wine Quality dataset
Author: Syifa Fauziah
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print("="*50)
print("STEP 1: Downloading Wine Quality Dataset")
print("="*50)

RED_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
WHITE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

print("Downloading red wine data...")
df_red = pd.read_csv(RED_URL, sep=';')
print(f"Red wine samples: {len(df_red)}")

print("Downloading white wine data...")
df_white = pd.read_csv(WHITE_URL, sep=';')
print(f"White wine samples: {len(df_white)}")

# Add wine type
df_red['wine_type'] = 0
df_white['wine_type'] = 1

# Combine
df = pd.concat([df_red, df_white], ignore_index=True)
df = df.drop_duplicates()

print(f"Total samples after removing duplicates: {len(df)}")

print("\n" + "="*50)
print("STEP 2: Preprocessing Data")
print("="*50)

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")

print("\n" + "="*50)
print("STEP 3: Saving Preprocessed Data")
print("="*50)

output_dir = 'winequality_preprocessing'
os.makedirs(output_dir, exist_ok=True)

X_train_scaled.to_csv(f'{output_dir}/X_train.csv', index=False)
X_test_scaled.to_csv(f'{output_dir}/X_test.csv', index=False)
pd.DataFrame({'quality': y_train}).to_csv(f'{output_dir}/y_train.csv', index=False)
pd.DataFrame({'quality': y_test}).to_csv(f'{output_dir}/y_test.csv', index=False)

print(f"Data saved to {output_dir}/")
print("  - X_train.csv")
print("  - X_test.csv")
print("  - y_train.csv")
print("  - y_test.csv")

print("\n" + "="*50)
print("Data preparation completed!")
print("="*50)
