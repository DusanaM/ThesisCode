import numpy as np
import torch

# agent's memory - replay buffer
class ReplyBuffer():
    def __init__(self, memory_size, input_shape, n_actions): 
        # max_size of the memory cuz we don't want it to be unbounded (number of transitions)
        # input_shape corresponds to the observation dimensionality of the env
        # n_actions: here we deal with the continuous action env (multiple actions can be taken in one step)
        self.memory_size = memory_size
        self.memory_counter = 0 # keeps track of the first available memory
        self.idx = 0

        self.state_memory = np.zeros((self.memory_size, *input_shape)) # input_shape is just how many features describes our state
        self.next_state_memory = np.zeros((self.memory_size, *input_shape)) 
        self.action_memory = np.zeros((self.memory_size, n_actions)) 
        self.reward_memory = np.zeros(self.memory_size)
        
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool) # dont get this

    def store_transition(self, state, action, reward, next_state, done): # done is terminal flag
        self.idx = self.memory_counter % self.memory_size  # first available index to store it so when memory_counter goes over (>=) memory_size, then we go from start e.g. 101 % 100 = 1
        self.state_memory[self.idx] = state
        self.next_state_memory[self.idx] = next_state
        self.action_memory[self.idx] = action
        self.reward_memory[self.idx] = reward
        self.terminal_memory[self.idx] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size): # batch_size is how many experiences we want to use in each iteration of training
        max_memory_available = min(self.memory_counter, self.memory_size) # this is how much memory we have available (so it is what we have stored (the counter) but it cannot be greater than max what we can store: memory_size)
        #batch = min(batch, max_memory_available)
        batch = np.random.choice(max_memory_available, batch_size) # pick a rand numbers up to *max_memory_available*, *batch_size* times 

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch] # checks if any of the states were terminating states
        
        return states, next_states, actions, rewards, dones

