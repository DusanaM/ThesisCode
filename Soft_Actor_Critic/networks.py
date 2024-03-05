import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

##### - In SAC we want to solve a problem of how to get robust and stable learning in continuous action space environments
##### - We want to use maximum entropy framework, scaling the cost function that enocurages exploration in a way that is robust to random seeds in our environment, episode to episode variation and starting conditions
##### - Here we will output mean and std for a normal distribution that we will sample to get the actions (building the agent's policy by which he picks next action)
##### - Actor build probabilities, Critis evaluates the value of (state, action) pair
##### - We will have a value network that will evaluate the how valuable the state is
###### https://github.com/rail-berkeley/softlearning/, https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name = 'critic', chkpt_dir='tmp/sac'):
        # beta is the learning rate
        # n_actions: here we deal with the continuous action env (multiple actions can be taken in one step)
        # fc1_dims: first and second fully connected layers
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name + '_sac')

        # we define our NN:
        # we incorporate the the action right from the very beginning of the input to the NN
        # here self.fc1_dims is the number of neurons or units in the first fully connected layer
        # how nn.Linear works: nn.Linear(in_features, out_features, bias=True), in_features is how many nodes are in the previous layer, and out_features are how many nodes are there now?
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims) # here we assume that input_dims[0] corresponds to the state
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        self.q = nn.Linear(self.fc2_dims, 1) 

        # Overall what just happend is that the input to NN is state and action(s) at that state (input_dims[0] + n_actions)
        # This is essentially (state, action) pair. We say n_actions because we have continous actions so more of them can take place at 1 timestep 
        # Output of NN is a sinlge value of quality of this pair

        self.optimizer = optim.Adam(self.parameters(), lr = beta) # self.parameters() are parameters of our deep NN, and it is coming from nn.module
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device for our computation, if we have gpu, we want to use it!
        self.to(self.device) # we want to send our entire network to our device

    def feed_forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value) # activation function
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)
        return q
    
    def save_chkpt(self): # saves the current state of NN, its parameters/weights
        # Save the current state of the neural network to a checkpoint file
        # The state here includes the parameters/weights of each layer
        T.save(self.state_dict(), self.chkpt_file)

    def load_chkpt(self): # loading the saved checkpoint back into the model
        self.load_state_dict(T.load(self.chkpt_file))

# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------

class ValueNetwork(nn.Module): # estimates the value of a particular state or set of states
    def __init__(self, beta, input_dims, fc1_dims = 256, fc2_dims = 256, name = 'value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.V = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device for our computation, if we have gpu, we want to use it!
        self.to(self.device) # we want to send our entire network to our device


    def feed_forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value) # activation function
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        V = self.V(state_value)
        return V
    
    def save_chkpt(self): # saves the current state of NN, its parameters/weights
        # Save the current state of the neural network to a checkpoint file
        # The state here includes the parameters/weights of each layer
        T.save(self.state_dict(), self.chkpt_file)

    def load_chkpt(self): # loading the saved checkpoint back into the model
        self.load_state_dict(T.load(self.chkpt_file))


# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name='actor',chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.reprar_noise = 1e-6

        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions) # each of the possible actions in 1 time step
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # learn more about
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device for our computation, if we have gpu, we want to use it!
        self.to(self.device) # we want to send our entire network to our device

    def feed_forward(self, state):
        probability = self.fc1(state)
        probability = F.relu(probability) # activation function
        probability = self.fc2(probability)
        probability = F.relu(probability)

        mu = self.mu(probability)
        sigma = self.sigma(probability)

        sigma = T.clamp(sigma, min=self.reprar_noise, max=1) # in paper they use -20 and 2; the clamp function restricts the values of the tensor to be within a specified range

        return mu, sigma
    
    # write about why is it the normal and not some other distribution!
    def normal_sample(self, state, reparametrize=True): # there are 2 sample functions for the N dist: one gives a sample and the other gives the sample plus some noise (thats when we say reparametrize=True)
        mu, sigma = self.feed_forward(state)
        probs = Normal(mu, sigma)

        if reparametrize:
            actions = probs.rsample() # this is with noise
        else:
            actions = probs.sample() # this is without the noise

        # actions are the one that are sampled, and action are those that are "processed" futher

        # we get the action for our agent:
        action = T.tanh(actions)*T.tensor(self.max_action) # tanh squashes the values to be within the range [-1, 1]. It's commonly used in the context of continuous control tasks to ensure that the output actions are within the range
        # and mulituplying it with max_action scales it up afterwards
        # we do this since max_action could easly have a value beyond -1 +1
        action.to(self.device)

        # this is for the calculation of our loss function, it doesnt go in the calculation of what action to take but the loss function for updating the weights of our deep NN
        log_probs = probs.log_prob(actions) - T.log(1-action.pow(2) + self.reprar_noise) # this is the formula from the paper: https://arxiv.org/pdf/1812.05905.pdf
        log_probs = log_probs.sum(1, keepdim = True)

        return action, log_probs # The action is the final decision made by the policy (2dim) and log_probs represents the logarithm of the probabilities associated with the sampled action, and it is used for calculating the loss during training
    
    def save_chkpt(self): # saves the current state of NN, its parameters/weights
        # Save the current state of the neural network to a checkpoint file
        # The state here includes the parameters/weights of each layer
        T.save(self.state_dict(), self.chkpt_file)

    def load_chkpt(self): # loading the saved checkpoint back into the model
        self.load_state_dict(T.load(self.chkpt_file))