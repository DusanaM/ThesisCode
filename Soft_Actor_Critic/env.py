import numpy as np
import gym
import math
import torch as T
import random
from sac_agent import SAC_Agent


class ENV():
    def __init__(self, s0, sigma, dt, N_prices, A, k, agent, trainMode = False):
        self.dt = dt
        self.sigma = sigma
        self.A = A
        self.k = k
        self.N_prices = N_prices
        self.s0 = s0
        self.trainMode = trainMode

        self.state_now = None
        # self.action_now = None
        self.PnL_before = None

        self.cash = 0
        self.inventory = 0

        self.reward_total = 0
        
        self.agent = agent
        self.done = False

        self.time_left = 1

        self.s = np.zeros(self.N_prices)
        self.s[0] = self.s0
        W = np.random.normal(0, np.sqrt(self.dt), self.N_prices)
        for i in range(1, self.N_prices):
            ds = self.sigma * W[i-1]
            self.s[i] = self.s[i-1] + ds



    def reset(self):

        self.s = np.zeros(self.N_prices)
        self.s[0] = self.s0
        W = np.random.normal(0, np.sqrt(self.dt), self.N_prices)
        for i in range(1, self.N_prices):
            ds = self.sigma * W[i-1]
            self.s[i] = self.s[i-1] + ds

        self.state_now = [self.s[int((1-self.time_left)*self.N_prices)], self.time_left, abs(self.inventory)/100]

        # self.action_now = None
        self.PnL_now = 0

        self.cash = 0
        self.inventory = 0

        self.time_left = 1
        self.reward_total = 0
        self.done = False

        return self.state_now
        



    # def get_bid_ask(self, s):

    #     price = self.s[int((1-self.time_left)*self.N_prices)]

    #     state_next = [self.time_left, abs(self.inventory)/100]

    #     if self.trainMode:

    #         PnL = self.cash + self.inventory * s
    #         if self.action_now is not None:
    #             risk_penalty = 0.5 * abs(self.inventory) * self.sigma * math.sqrt(self.dt) # return to this function as we might want to define it differently
    #             change_in_PnL = PnL - self.PnL_before
    #             reward = change_in_PnL - risk_penalty
    #             self.reward_total += reward

    #             # print("here")

    #             self.agent.remember(self.state_now, self.action_now, reward, state_next, self.done)

    #             if self.done:
    #                 self.agent.learn()

            
    #         print("inventory: ", self.inventory, "reward: ", self.reward_total)

    #         explore = self.agent.memory.memory_counter / self.agent.memory.memory_size < random.random()
    #         print("mem counter: ", self.agent.memory.memory_counter, "mem size: ", self.agent.memory.memory_size, "chance of EXPLOITing: ", self.agent.memory.memory_counter / self.agent.memory.memory_size)
    #         print("explore: ", explore)

    #         action = np.array([random.uniform(0,1)]) if explore else self.agent.pick_action(state_next)

    #         self.state_now = state_next
    #         self.PnL_before = PnL
    #         self.action_now = action

    #     else:
    #         # test mode now, so we just get the action - we exploit!
    #         action = self.agent.pick_action(state_next)
            
            
    #     risk_aversion_par = max(1e-5, ((action[0] + 1)/2))


    #     res_price = self.res_price(s, self.inventory, self.sigma, risk_aversion_par, self.time_left)
    #     # print("s: ", s, "inventory: ", self.inventory, "sigma: ", self.sigma, "action: ", action, "risk: ", risk_aversion_par, "timeleft: ", timeleft, "res_price: ", res_price)


    #     spread = self.spread(self.k, self.sigma, risk_aversion_par, self.time_left)
    #     # print("spread", spread)


    #     Bid = res_price - spread/2
    #     Ask = res_price + spread/2

    #     # print("Bid, Ask: ", (Bid,Ask))

    #     return action, Bid, Ask
    

    def interact_with_market(self, Bid, Ask, s):

        delta_b = s - Bid
        delta_a = Ask - s

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

        
    def step(self, action):

        # ------------------------------ up til now WE HAVE A STATE WE ARE IN, AND AN ACTION WE WOULD LIKE TO TAKE ------------------------------

        risk_aversion_par = max(1e-5, ((action[0] + 1)/2))

        res_price = self.res_price(self.state_now[0], self.inventory, self.sigma, risk_aversion_par, self.time_left)
        spread = self.spread(self.k, self.sigma, risk_aversion_par, self.time_left)

        Bid = res_price - spread/2
        Ask = res_price + spread/2

        # ------------------------------ now having BID and ASK, we interact with market (staying at same t)  ------------------------------

        self.interact_with_market(Bid, Ask, self.state_now[0])

        # -------------------- after we interacted with market, we changed self.inventory and self.cash -----------------------------------------
        # -------------------- now we are ready to transition to t+1: -----------------------------------------
        self.time_left -= self.dt
        self.done = self.time_left<= self.dt

        state_next = [self.s[int((1-self.time_left)*self.N_prices)], self.time_left, abs(self.inventory)/100]
        PnL_next = self.cash + self.inventory * self.s[int((1-self.time_left)*self.N_prices)]

        risk_penalty = 0.5 * abs(self.inventory) * self.sigma * math.sqrt(self.dt) # return to this function as we might want to define it differently
        change_in_PnL = PnL_next - self.PnL_now
        reward = change_in_PnL - risk_penalty
        self.reward_total += reward

        # self.agent.remember(self.state_now, self.action_now, reward, state_next, self.done)

        # if self.done:
        #     self.agent.learn()

        self.state_now = state_next
        self.PnL_now = PnL_next

        return state_next, reward




        
        




        # action, Bid, Ask = self.get_bid_ask(s[int((1-self.time_left)*self.N_prices)])
        # interact_with_market(Bid, Ask, s[int((1-self.time_left)*self.N_prices)])


        










    def res_price(self, s, q, sigma, gamma, delta_t):
        return s - q * gamma * sigma**2 * delta_t
    
    def spread(self, k, sigma, gamma, delta_t):
        return gamma * sigma**2 * delta_t + 2 / gamma * np.log(1 + gamma / k)

    def lambdaa(self, delta, A, k):
        return A * np.exp(-k * delta)

