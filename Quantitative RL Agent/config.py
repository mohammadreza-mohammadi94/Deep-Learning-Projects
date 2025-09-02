# config.py
"""Centralized configuration file for the RL Trader project."""

# --- DIRECTORIES ---
MODELS_FOLDER = 'models'
REWARDS_FOLDER = 'rewards'
DATA_FILE = 'data/aapl_msi_sbux.csv'
SCALER_FILE = f'{MODELS_FOLDER}/scaler.pkl'
MODEL_FILE = f'{MODELS_FOLDER}/dqn.weights.h5'

# --- ENVIRONMENT & AGENT ---
INITIAL_INVESTMENT = 20000.0

# --- DQN HYPERPARAMETERS ---
GAMMA = 0.95              # Discount rate
EPSILON_START = 1.0       # Starting exploration rate
EPSILON_MIN = 0.01        # Minimum exploration rate
EPSILON_DECAY = 0.995     # Decay rate for exploration
LEARNING_RATE = 0.001     # Learning rate for the Adam optimizer
REPLAY_BUFFER_SIZE = 2000 # Size of the experience replay buffer

# --- TRAINING ---
NUM_EPISODES = 2000
BATCH_SIZE = 32