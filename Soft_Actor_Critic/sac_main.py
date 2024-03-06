# import pybullet_envs
import numpy as np
import gym
import math
import torch as T
import random
from sac_agent import SAC_Agent
from memory_buffer import ReplyBuffer

class SAC_Trader():
    def __init__(self, sigma, dt, bid, ask, max_inventory = 100, time_left = 100):
        self.dt = dt
        self.bid = bid
        self.ask = ask
        self.sigma = sigma
        self.agent = SAC_Agent()
        self.time_left = time_left

    

    def explore_or_exploit(self, x, s, A, k, dt, N):
        PnL = 0
        inventory = 0

        for i in range(1, N):

            state = np.array(x, inventory)
            # here we need to pick random action:
            if i < self.agent.batch_size:
                action = np.array([random.uniform(0, s[i]), random.uniform(0, s[i])])
            
            else:
                action = self.agent.pick_action(state) # action is [delta_bid, delta_ask]

            Bid = s[i] - action[0]
            Ask = s[i] + action[1]


            # prob of arrivals:
            prob_bid = self.lambdaa(action[0], A, k)*dt
            prob_ask = self.lambdaa(action[1], A, k)*dt
            rand_b = random.random()
            rand_a = random.random()

            # let's see what happens: ----------------------------

            # we execute sell order only
            if (rand_b <= prob_bid and rand_a > prob_ask):
                inventory += 1
                x -= Bid
                    # we execute buy order only
            elif rand_b > prob_bid and rand_a <= prob_ask:
                inventory -= 1
                x += Ask
                
            # we execute both sell and buy orders
            elif rand_b <= prob_bid and rand_a <= prob_ask:
                x = x + Ask - Bid


            next_state =  np.array(x, inventory)
            PnL_next_state = x + inventory*s[i]
            risk_penalty = 0.5 * abs(inventory) * self.sigma_sqrt_dt # return to this function as we might want to define it differently
            change_in_PnL = PnL_next_state - PnL
            reward = change_in_PnL - risk_penalty

            self.agent.remember(state, action, reward, next_state, False)
            self.agent.learn() # this doesnt do anything if we explore!



    # prob of arrival
    def lambdaa(self, delta, A, k):
        return A * np.exp(-k * delta)
            
