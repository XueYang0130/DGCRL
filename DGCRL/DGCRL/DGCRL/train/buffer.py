import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((max_size, state_dim))
        self.action_memory = np.zeros((max_size, action_dim))
        self.reward_memory = np.zeros((max_size, ))
        self.next_state_memory = np.zeros((max_size, state_dim))
        self.terminal_memory = np.zeros((max_size, ), dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size
        # print(f"state:{state}\naction:{action}\nreward:{reward}\nstate_:{state_}\ndone:{done}")
        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt >= self.batch_size
