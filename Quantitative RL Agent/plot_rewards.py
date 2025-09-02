# plot_rewards.py
"""
Plots the results from training or testing.
- For training, it shows the portfolio value per episode and a moving average.
- For testing, it shows a histogram of final portfolio values.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def plot_results(mode: str):
    """Loads and plots the rewards based on the mode."""
    try:
        rewards = np.load(f'rewards/{mode}.npy')
    except FileNotFoundError:
        print(f"Error: rewards/{mode}.npy not found. Please run main.py in '{mode}' mode first.")
        return

    avg = rewards.mean()
    min_r = rewards.min()
    max_r = rewards.max()
    
    print(f"--- Results for '{mode}' mode ---")
    print(f"Average portfolio value: {avg:.2f}")
    print(f"Min portfolio value: {min_r:.2f}")
    print(f"Max portfolio value: {max_r:.2f}")

    plt.figure(figsize=(12, 6))

    if mode == 'train':
        plt.plot(rewards, alpha=0.6, label='Portfolio Value per Episode')
        # Add a moving average to see the trend
        moving_avg = pd.Series(rewards).rolling(window=100).mean()
        plt.plot(moving_avg, color='red', linewidth=2, label='100-Episode Moving Average')
        plt.ylabel('Portfolio Value')
        plt.xlabel('Episode')
        plt.title('Training Progress: Portfolio Value Over Episodes')
        plt.legend()
    else:  # test mode
        plt.hist(rewards, bins=30, edgecolor='black')
        plt.axvline(avg, color='red', linestyle='dashed', linewidth=2, label=f'Average Value: {avg:.2f}')
        plt.xlabel('Final Portfolio Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Final Portfolio Values in Test Mode')
        plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test'],
                        help='either "train" or "test"')
    args = parser.parse_args()
    plot_results(args.mode)