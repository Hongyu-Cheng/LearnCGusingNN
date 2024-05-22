import numpy as np
import collections
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, buffer_capacity):
        self.buffer = collections.deque(maxlen=buffer_capacity)

    def add(self, current_state, action_taken, reward_value, next_state, is_done):
        self.buffer.append((current_state, action_taken, reward_value, next_state, is_done))

    def sample(self, sample_size):
        sampled_transitions = random.sample(self.buffer, sample_size)
        current_state, action_taken, reward_value, next_state, is_done = zip(*sampled_transitions)
        return current_state, action_taken, reward_value, next_state, is_done

    @property
    def size(self):
        return len(self.buffer)

def moving_average(data_series, window_size=10):
    return [np.mean(data_series[max(0, i - window_size + 1):(i + 1)]) for i in range(len(data_series))]

def train(env, agent, total_samples, replay_memory, min_memory_size, batch_size, training_epochs=1):
    episode_returns = []
    total_steps = 0

    for epoch in range(training_epochs):
        progress_bar = tqdm(range(total_samples), desc=f"Epoch {epoch + 1}")
        cumulative_reward = 0
        
        for sample in progress_bar:
            current_state = env.reset()
            is_done = False
            action_taken = agent.take_action(current_state)
            next_state, reward_value, is_done = env.step(action_taken)
            replay_memory.add(current_state, action_taken, reward_value, next_state, is_done)
            cumulative_reward += reward_value.item()
            current_state = next_state

            total_steps += 1

            if replay_memory.size > min_memory_size:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_memory.sample(batch_size)
                batch_states = torch.stack([state.reshape(-1) for state in batch_states])
                batch_actions = torch.stack([action.reshape(-1) for action in batch_actions])
                batch_rewards = torch.stack(batch_rewards)
                batch_next_states = torch.stack([state.reshape(-1) for state in batch_next_states])
                batch_dones = torch.tensor(batch_dones, dtype=torch.int).view(-1, 1)

                transition_dict = {'states': batch_states, 'actions': batch_actions, 'next_states': batch_next_states, 'rewards': batch_rewards, 'dones': batch_dones}
                agent.update(transition_dict)
            episode_returns.append(cumulative_reward)
            
            if sample % 100 == 0 and sample > 0:
                moving_avg = np.mean(episode_returns[-min(1000, len(episode_returns)):])
                progress_bar.set_postfix({"Return": cumulative_reward, "Moving Avg Return": moving_avg})

                cumulative_reward = 0

    return episode_returns
