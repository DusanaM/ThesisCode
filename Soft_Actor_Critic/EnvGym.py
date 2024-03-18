import numpy as np
import gym
import math
import torch as T
import random
import gymnasium as gym
from gym import Env
from gym.spaces import Box

class MyEnv(Env):

   def __init__(self, s0, sigma, dt, N_prices, A, k):       

    #    self.action_space = Box(96.0, 104.0, (1,))
       # a1 is reservation price and a2 is the spread
       self.action_space = Box(low=np.array([0.9, 1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
    #    self.action_space = Box(low=np.array([96.0]), high=np.array([104.0]), dtype=np.float32)
    #    print(type(self.action_space))


       self.s0 = s0
       self.sigma = sigma
       self.dt = dt
       self.N_prices = N_prices
       self.A = A
       self.k = k

       self.state_now = None
       self.PnL_now = 0
       self.cash = 0
       self.inventory = 0
       self.reward_total = 0
       self.time_left = 1
       self.done = False



   def reset(self):
       self.s = np.zeros(self.N_prices)
       self.s[0] = self.s0
       W = np.random.normal(0, np.sqrt(self.dt), self.N_prices)
       for i in range(1, self.N_prices):
           ds = self.sigma * W[i-1]
           self.s[i] = self.s[i-1] + ds

       self.state_now = (self.s0/100, 1, 0) # random initial state: (s0, time_left, inventory)
       self.PnL_now = 0
       self.cash = 0
       self.inventory = 0
       self.time_left = 1
       self.reward_total = 0
       self.done = False

   def update_state_based_on_spreads(self, state_now, a1, a2):
        # Update self.s, the internal state, based on a1 and a2.
        # implements p(s_{t+1} | s_t, a_t)
        # a2 = 1.5

        print("I recieve action: ", a1, a2)

        a1 = 102.0*a1
        a2 = 2*a2

        print("I transform action: ", a1, a2)

        Bid = ( (a1) - a2/2)
        Ask = ((a1) + a2/2) 

        print("Bid: ", round(Bid, 2), " Price: ", round(state_now[0]*100, 2), " Ask: ", round(Ask, 2))


        delta_b = state_now[0]*100 - Bid
        delta_a = Ask - state_now[0]*100

        # prob of arrivals:
        prob_bid = self.lambdaa(delta_b, self.A, self.k)*self.dt
        prob_ask = self.lambdaa(delta_a, self.A, self.k)*self.dt
        rand_b = random.random()
        rand_a = random.random()
       
        # let's see what happens after our decision: ----------------------------

        # we execute sell order only
        if (rand_b <= prob_bid and rand_a > prob_ask):
            self.inventory += 1
            self.cash -= Bid
            print("Hit Bid")

        # we execute buy order only
        elif rand_b > prob_bid and rand_a <= prob_ask:
            self.inventory -= 1
            self.cash += Ask
            print("Hit Ask")

            
        # we execute both sell and buy orders
        elif rand_b <= prob_bid and rand_a <= prob_ask:
            self.cash = self.cash + Ask - Bid
            print("Hit Both")

        else:
            print("No hit")


        self.time_left -= self.dt
        self.done = self.time_left <= self.dt

        state_next = (self.s[int((1-self.time_left)*self.N_prices)]/100, self.time_left, abs(self.inventory)/100)

    
        return state_next # s_{t+1}


   def step(self, a):
           
        state_next = self.update_state_based_on_spreads(self.state_now, a[0], a[1])


        
        PnL_next = self.cash + self.inventory * self.s[int((1-state_next[1])*self.N_prices)]
        risk_penalty = 0.5 * abs(self.inventory) * self.sigma * math.sqrt(self.dt) # return to this function as we might want to define it differently
        change_in_PnL = PnL_next - self.PnL_now
        reward = change_in_PnL - risk_penalty
        self.reward_total += reward

        self.state_now = state_next
        self.PnL_now = PnL_next

        done = self.done

        return self.state_now, reward, done, False, {}
   


   def lambdaa(self, delta, A, k):
       return A * np.exp(-k * delta)

