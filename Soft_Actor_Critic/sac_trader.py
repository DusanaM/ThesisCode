# import pybullet_envs
import numpy as np
import gym
import math
import torch as T
import random
from sac_agent import SAC_Agent
# from memory_buffer import ReplyBuffer

class SAC_Trader():
    def __init__(self, s, sigma, A, k, dt, N, batch_size = 500, gamma=0.1):
        self.dt = dt
        self.sigma = sigma
        self.A = A
        self.k = k
        self.N = N
        self.gamma = gamma

        self.inventory = np.zeros(N)
        self.cash = np.zeros(N)
        self.PnL = np.zeros(N)
        self.reward = np.zeros(N)

        self.s = s
        self.Bid = np.zeros(N)
        self.Ask = np.zeros(N)

        self.agent = SAC_Agent(input_dims = [2], max_action=5*self.sigma, batch_size=batch_size, gamma=self.gamma)


        self.Bid[0] = None
        self.Ask[0] = None




    def explore_or_exploit(self):

        for i in range(1, self.N):

            # state_before = np.array(self.cash[i-1], self.inventory[i-1])
            state_before = [self.N - i + 1, self.inventory[i-1]]
            

            # here we need to pick random action:
            if i < self.agent.batch_size:
                action = np.array([random.uniform(0, self.s[i]), random.uniform(0, self.s[i])]) # picks random [delta_bid, delta_ask] from 0 up to s[i]
            else:
                action = self.agent.pick_action(state_before) # action is [delta_bid, delta_ask]
                # print("i= ", i, "and picked action after processing: ", action)


            self.Bid[i] = self.s[i] - action[0]
            self.Ask[i] = self.s[i] + action[1]

            # if i < self.agent.batch_size:
                # print("all good so far")


            # prob of arrivals:
            prob_bid = self.lambdaa(action[0], self.A, self.k)*self.dt
            prob_ask = self.lambdaa(action[1], self.A, self.k)*self.dt
            rand_b = random.random()
            rand_a = random.random()

            # let's see what happens after our decision: ----------------------------

            # we execute sell order only
            if (rand_b <= prob_bid and rand_a > prob_ask):
                self.inventory[i] = self.inventory[i-1] + 1
                self.cash[i] = self.cash[i-1] - self.Bid[i]
                    # we execute buy order only
            elif rand_b > prob_bid and rand_a <= prob_ask:
                self.inventory[i] = self.inventory[i-1] - 1
                self.cash[i] = self.cash[i-1] + self.Ask[i]
                
            # we execute both sell and buy orders
            elif rand_b <= prob_bid and rand_a <= prob_ask:
                self.inventory[i] = self.inventory[i-1]
                self.cash[i] = self.cash[i-1] + self.Ask[i] - self.Bid[i]

            else:
                self.inventory[i] = self.inventory[i-1]
                self.cash[i] = self.cash[i-1]


            state_now =  [self.N - i, self.inventory[i]]
            self.PnL[i] = self.cash[i] + self.inventory[i]*self.s[i]
            risk_penalty = 0.5 * abs(self.inventory[i]) * self.sigma * math.sqrt(self.dt) # return to this function as we might want to define it differently
            change_in_PnL = self.PnL[i] - self.PnL[i-1]
            self.reward[i] = change_in_PnL - risk_penalty

            self.agent.remember(state_before, action, self.reward[i], state_now, False)
            # self.agent.learn()


    # prob of arrival
    def lambdaa(self, delta, A, k):
        return A * np.exp(-k * delta)
            
