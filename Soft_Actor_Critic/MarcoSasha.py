import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prettytable import PrettyTable
import math

class MarcoSasha():
    def __init__(self, s_array, sigma, N, T, dt, gamma, k, A):
        self.s_array = s_array
        self.sigma = sigma
        self.N = N
        self.dt = dt
        self.T = T
        self.gamma = gamma
        self.k = k
        self.A = A

        self.t = np.linspace(0.0, self.T, self.N)

        self.q = np.zeros(self.N) # inventory
        self.res_price_array = np.zeros(self.N)
        self.spread_array = np.zeros(self.N)
        self.x = np.zeros(self.N) # wealth

        self.Bid_price = np.zeros(self.N)
        self.Ask_price = np.zeros(self.N)

        self.delta_b = np.zeros(self.N)
        self.prob_b = np.zeros(self.N)
        self.delta_a = np.zeros(self.N)
        self.prob_a = np.zeros(self.N)

        self.PnL_array = []
        self.rewards = []

    def res_price(self, s, q, t):
        return s - q * self.gamma * self.sigma**2 * (self.T - t)
    
    def spread(self, t):
        return self.gamma * self.sigma**2 * (self.T - t) + 2 / self.gamma * np.log(1 + self.gamma / self.k)
    
    def lambdaa(self, delta):
        return self.A * np.exp(-self.k * delta)
    

    def simulate(self):
        self.res_price_array[0] = self.res_price(self.s_array[0], self.q[0], self.t[0])
        self.spread_array[0] = self.spread(self.t[0])

        self.Bid_price[0] = self.res_price_array[0] - self.spread_array[0]/2
        self.Ask_price[0] = self.res_price_array[0] + self.spread_array[0]/2
        
        self.PnL_array.append(0)
        self.rewards.append(0)

        for i in range(1, self.N):
            self.res_price_array[i] = self.res_price(self.s_array[i], self.q[i-1], self.t[i])
            self.spread_array[i] = self.spread(self.t[i])

            self.Bid_price[i] = self.res_price_array[i] - self.spread_array[i]/2
            self.Ask_price[i] = self.res_price_array[i] + self.spread_array[i]/2

            self.delta_b[i] = self.s_array[i] - self.Bid_price[i]
            self.delta_a[i] = self.Ask_price[i] - self.s_array[i]

            self.prob_b[i] = self.lambdaa(self.delta_b[i])*self.dt
            self.prob_a[i] = self.lambdaa(self.delta_a[i])*self.dt

            rand_b = random.random()
            rand_a = random.random()

            # we execute sell order only
            if (rand_b <= self.prob_b[i] and rand_a > self.prob_a[i]):
                self.q[i] = self.q[i-1] + 1
                self.x[i] = self.x[i-1] - self.Bid_price[i]

            # we execute buy order only
            elif rand_b > self.prob_b[i] and rand_a <= self.prob_a[i]:
                self.q[i] = self.q[i-1] - 1
                self.x[i] = self.x[i-1] + self.Ask_price[i]
                
            # we execute both sell and buy orders
            elif rand_b <= self.prob_b[i] and rand_a <= self.prob_a[i]:
                self.q[i] = self.q[i-1]
                self.x[i] = self.x[i-1] + self.Ask_price[i] - self.Bid_price[i]
                
            # we dont execute any
            else:
                self.q[i] = self.q[i-1]
                self.x[i] = self.x[i-1]


            self.PnL_array.append(self.x[i] + self.q[i]*self.s_array[i])

            # reward is change in PnL minus risk penalty
            self.rewards.append(self.PnL_array[i] - self.PnL_array[i-1] - 0.5 * abs(self.q[i]) * self.sigma * math.sqrt(self.dt))





            
