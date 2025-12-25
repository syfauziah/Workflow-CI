import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print('Downloading data from UCI Repository...')
RED_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
WHITE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

df_red = pd.read_csv(RED_URL, sep=';')
df_white = pd.read_csv(WHITE_URL, sep=';')

df_red['wine_type'] = 0
df_white['wine_type'] = 1

df = pd.concat([df_red, df_white], ignore_index=True)
df = df.drop_duplicates()

print(f'Total samples: {len(df)}')

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

output_dir = 'winequality_preprocessing'
os.makedirs(output_dir, exist_ok=True)

X_train_scaled.to_csv(f'{output_dir}/X_train.csv', index=False)
X_test_scaled.to_csv(f'{output_dir}/X_test.csv', index=False)
pd.DataFrame({'quality': y_train}).to_csv(f'{output_dir}/y_train.csv', index=False)
pd.DataFrame({'quality': y_test}).to_csv(f'{output_dir}/y_test.csv', index=False)

print('Done! Data saved to winequality_preprocessing/')