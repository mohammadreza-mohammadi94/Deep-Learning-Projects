# main.py
"""
Main script to train or test the RL Trader agent.
Usage:
  - For training: python main.py --mode train
  - For testing:  python main.py --mode test
"""
import argparse
import pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm

import config
from environment import MultiStockEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir
from sklearn.preprocessing import StandardScaler

def play_one_episode(agent: DQNAgent, env: MultiStockEnv, scaler: StandardScaler,
                     batch_size: int, is_train: bool) -> float:
    """Plays one episode and returns the final portfolio value."""
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        
        if is_train:
            agent.update_replay_memory(state, action, reward, next_state, done)
            agent.replay(batch_size)
            
        state = next_state
        
    return info['cur_val']

def run(mode: str):
    """Main function to run training or testing."""
    data = get_data(config.DATA_FILE)
    n_timesteps, _ = data.shape
    n_train = n_timesteps // 2
    
    train_data = data[:n_train]
    test_data = data[n_train:]

    if mode == 'train':
        env = MultiStockEnv(train_data, config.INITIAL_INVESTMENT)
        scaler = get_scaler(env)
        with open(config.SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
    else: # test mode
        env = MultiStockEnv(test_data, config.INITIAL_INVESTMENT)
        with open(config.SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)

    agent = DQNAgent(
        state_size=env.state_dim,
        action_size=len(env.action_space),
        gamma=config.GAMMA,
        epsilon=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.REPLAY_BUFFER_SIZE
    )

    if mode == 'test':
        agent.load(config.MODEL_FILE)
        agent.epsilon = 0.01 # Low epsilon for testing
        num_episodes = 100 # Run fewer episodes for testing
    else:
        num_episodes = config.NUM_EPISODES

    portfolio_values = []
    
    print(f"--- Running in {mode.upper()} mode ---")
    
    for e in tqdm(range(num_episodes), desc=f"{mode.capitalize()}ing Episodes"):
        val = play_one_episode(agent, env, scaler, config.BATCH_SIZE, is_train=(mode == 'train'))
        portfolio_values.append(val)

    if mode == 'train':
        agent.save(config.MODEL_FILE)

    np.save(f'{config.REWARDS_FOLDER}/{mode}.npy', portfolio_values)
    
    print(f"--- {mode.upper()} mode finished ---")
    final_avg_value = np.mean(portfolio_values[-100:])
    print(f"Average portfolio value over last 100 episodes: {final_avg_value:.2f}")


if __name__ == '__main__':
    maybe_make_dir(config.MODELS_FOLDER)
    maybe_make_dir(config.REWARDS_FOLDER)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'],
                        help='either "train" or "test"')
    args = parser.parse_args()
    
    run(args.mode)