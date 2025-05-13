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
        self.current_step = 0
        self.state = self.workload[self.current_step] / 100.0  # 🔹 Normalize
        return self.state

    
    def step(self, action):
        action = np.clip(action, 0, 1)
        scaled = action * 100
        alloc = (scaled / scaled.sum()) * 100 if scaled.sum() > 0 else np.ones(3) * (100 / 3)

        demand = self.workload[self.current_step]
        diff = np.abs(demand - alloc)

        # 🔻 Modified reward function
        reward = -np.sum(diff ** 1.5)  # Penalize larger mismatches more harshly
        if np.all(diff < 5):
            reward += 10  # Bonus for close match

        self.current_step += 1
        done = self.current_step >= len(self.workload)

        # 🔹 Normalize state for better training
        if not done:
            self.state = self.workload[self.current_step] / 100.0
        else:
            self.state = np.zeros_like(self.workload[0])  # dummy state on end

        return self.state, reward, done, {}



    def render(self, mode='human'):
        print(f"Hour {self.hour}: Demand {self.workload[self.hour]}")
