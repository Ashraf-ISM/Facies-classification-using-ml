# src/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Load dataset."""
    return pd.read_csv(path)

def prepare_supervised(df, features, target="Facies", test_size=0.2, random_state=42):
    """Prepare train/test split for supervised learning."""
    X = df[features].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler

def prepare_unsupervised(df, features):
    """Prepare scaled features for clustering."""
    X = df[features].dropna().values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
