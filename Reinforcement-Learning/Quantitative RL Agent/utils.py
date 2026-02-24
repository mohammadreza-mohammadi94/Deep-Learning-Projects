# utils.py
"""Utility functions for data loading, directory creation, and scaler initialization."""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from environment import MultiStockEnv

def maybe_make_dir(directory: str):
    """Creates a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_data(file_path: str) -> np.ndarray:
    """Reads stock data from a CSV file."""
    df = pd.read_csv(file_path)
    return df.values

def get_scaler(env: MultiStockEnv) -> StandardScaler:
    """
    Fits a StandardScaler by playing random episodes in the environment.
    """
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, _, done, _ = env.step(action)
        states.append(state)
        if done:
            break
    
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler