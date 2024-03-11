import numpy as np
import gym
import math
import torch as T
import random
from sac_agent import SAC_Agent


class SAC_Trader():
    def __init__(self, sigma, A, k, dt, agent):
        self.dt = dt
        self.sigma = sigma
        self.A = A
        self.k = k

        self.state_before = None
        self.action_before = None
        self.PnL_before = None

        self.cash = 0
        self.inventory = 0

        self.reward_total = 0
        
        self.agent = agent


    def reset(self):
        self.state_before = None
        self.action_before = None
        self.PnL_before = None

        self.cash = 0
        self.inventory = 0

        self.reward_total = 0


    def get_bid_ask(self, timeleft, s, trainMode = False):

        state_now =  [timeleft, abs(self.inventory)/100]

        if trainMode:

            PnL = self.cash + self.inventory * s
            if self.action_before is not None:
                risk_penalty = 0.5 * abs(self.inventory) * self.sigma * math.sqrt(self.dt) # return to this function as we might want to define it differently
                change_in_PnL = PnL - self.PnL_before
                reward = change_in_PnL - risk_penalty
                self.reward_total += reward

                # print("here")

                self.agent.remember(self.state_before, self.action_before, reward, state_now, timeleft <= 0)
                self.agent.learn()

            
            print("inventory: ", self.inventory, "reward: ", self.reward_total)

            explore = self.agent.memory.memory_counter / self.agent.memory.memory_size < random.random()
            print("mem counter: ", self.agent.memory.memory_counter, "mem size: ", self.agent.memory.memory_size, "chance of EXPLOITing: ", self.agent.memory.memory_counter / self.agent.memory.memory_size)
            print("explore: ", explore)

            action = np.array([random.uniform(0,1)]) if explore else self.agent.pick_action(state_now)

            self.state_before = state_now
            self.PnL_before = PnL
            self.action_before = action

        else:
            # test mode now, so we just get the action - we exploit!
            action = self.agent.pick_action(state_now)
            
            
        risk_aversion_par = max(1e-5, ((action[0] + 1)/2))


        res_price = self.res_price(s, self.inventory, self.sigma, risk_aversion_par, timeleft)
        # print("s: ", s, "inventory: ", self.inventory, "sigma: ", self.sigma, "action: ", action, "risk: ", risk_aversion_par, "timeleft: ", timeleft, "res_price: ", res_price)


        spread = self.spread(self.k, self.sigma, risk_aversion_par, timeleft)
        # print("spread", spread)


        Bid = res_price - spread/2
        Ask = res_price + spread/2

        # print("Bid, Ask: ", (Bid,Ask))

        return Bid, Ask
    

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




    def res_price(self, s, q, sigma, gamma, delta_t):
        return s - q * gamma * sigma**2 * delta_t
    
    def spread(self, k, sigma, gamma, delta_t):
        return gamma * sigma**2 * delta_t + 2 / gamma * np.log(1 + gamma / k)

    def lambdaa(self, delta, A, k):
        return A * np.exp(-k * delta)

