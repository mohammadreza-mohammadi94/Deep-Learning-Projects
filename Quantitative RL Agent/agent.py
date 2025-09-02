# agent.py
"""
Contains the DQNAgent, ReplayBuffer, and model creation logic.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from typing import Tuple

class ReplayBuffer:
    """A simple FIFO experience replay buffer for DDPG agents."""
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 32) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])

def mlp(input_dim: int, n_action: int, n_hidden_layers: int = 1, hidden_dim: int = 32) -> Model:
    """A multi-layer perceptron for Q-value estimation."""
    i = Input(shape=(input_dim,))
    x = i
    for _ in range(n_hidden_layers):
        x = Dense(hidden_dim, activation='relu')(x)
    x = Dense(n_action, activation='linear')(x) # Linear activation for Q-values
    model = Model(i, x)
    return model

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, gamma: float, epsilon: float,
                 epsilon_min: float, epsilon_decay: float, learning_rate: float,
                 buffer_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, 1, size=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.model = mlp(state_size, action_size)
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.Huber() # Huber loss is more robust to outliers

    def update_replay_memory(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.store(state.flatten(), action, reward, next_state.flatten(), done)

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model(state, training=False)
        return np.argmax(act_values[0])

    @tf.function
    def replay(self, batch_size: int):
        if self.memory.size < batch_size:
            return

        minibatch = self.memory.sample_batch(batch_size)
        states = tf.convert_to_tensor(minibatch['s'], dtype=tf.float32)
        next_states = tf.convert_to_tensor(minibatch['s2'], dtype=tf.float32)
        actions = tf.convert_to_tensor(minibatch['a'], dtype=tf.int32)
        rewards = tf.convert_to_tensor(minibatch['r'], dtype=tf.float32)
        done = tf.convert_to_tensor(minibatch['d'], dtype=tf.float32)

        # --- MAJOR REWRITE ---
        # The core of the learning algorithm using GradientTape for efficiency
        
        # 1. Predict Q-values for the next states
        q_next = self.model(next_states, training=False)
        
        # 2. Get the maximum Q-value for each next state (Bellman equation)
        target_q = tf.reduce_max(q_next, axis=1)
        
        # 3. Calculate the Bellman target value
        target = rewards + self.gamma * target_q * (1 - done)

        # 4. Use GradientTape to update weights
        with tf.GradientTape() as tape:
            # Get Q-values for the actions that were actually taken
            q_values = self.model(states, training=True)
            action_indices = tf.stack([tf.range(batch_size), actions], axis=1)
            q_action = tf.gather_nd(q_values, action_indices)

            # Calculate loss between the predicted Q-values and the Bellman target
            loss = self.loss_fn(target, q_action)

        # 5. Compute and apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name: str):
        self.model.load_weights(name)

    def save(self, name: str):
        self.model.save_weights(name)