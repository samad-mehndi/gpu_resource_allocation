import gym
from gym import spaces
import numpy as np

class GpuEnv(gym.Env):
    def __init__(self, workload_df):
        super(GpuEnv, self).__init__()
        self.workload = workload_df[['Model_A_Traffic', 'Model_B_Chatbot', 'Model_C_Speech']].values
        self.hour = 0
        self.num_models = 3

        # Action: allocation to 3 models (each 0–100), but will normalize to sum = 100
        self.action_space = spaces.Box(low=0, high=100, shape=(self.num_models,), dtype=np.float32)

        # Observation: demand from each model
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_models,), dtype=np.float32)

    def reset(self):
        self.hour = 0
        return self.workload[self.hour]

    def step(self, action):
        demand = self.workload[self.hour]
        
        # Scale action from [-1, 1] to [0, 100]
        action = np.clip(action, 0, 1)
        alloc = action * 100
        
        # Normalize to sum = 100
        if alloc.sum() > 0:
            alloc = (alloc / alloc.sum()) * 100
        else:
            alloc = np.ones(self.num_models) * (100 / self.num_models)

        diff = np.abs(demand - alloc)
        reward = -np.sum(diff)

        self.hour += 1
        done = self.hour >= len(self.workload)
        next_state = self.workload[self.hour] if not done else np.zeros(self.num_models)

        return next_state, reward, done, {}


    def render(self, mode='human'):
        print(f"Hour {self.hour}: Demand {self.workload[self.hour]}")
