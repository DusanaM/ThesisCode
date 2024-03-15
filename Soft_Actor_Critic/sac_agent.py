import os
import numpy as np
import torch as T
import torch.nn.functional as F
from memory_buffer import ReplyBuffer
from networks import ActorNetwork, ValueNetwork, CriticNetwork

class SAC_Agent():
    def __init__(self, input_dims, max_action = 1, alpha=0.003, beta = 0.003, gamma = 0.99, n_actions = 2, memory_size = 100000, tau = 0.005, layer1 = 256 , layer2 = 256, batch_size = 256, reward_scale = 2): 
        # reward scaling is how are we going to account for the entropy in the framework - we scale the rewards in the critic loss function
    
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.memory = ReplyBuffer(memory_size, input_dims, n_actions)
        self.actor = ActorNetwork(alpha, input_dims, max_action)
        self.critic1 = CriticNetwork(beta, input_dims, n_actions, layer1, layer2, name='critic1')
        self.critic2 = CriticNetwork(beta, input_dims, n_actions, layer1, layer2, name='critic2')
        self.val = ValueNetwork(beta, input_dims, layer1, layer2, name='value')
        self.target_val = ValueNetwork(beta, input_dims, layer1, layer2, name='target_val')

        self.scale = reward_scale
        self.update_pars(tau = 1) # come back to this!

    def pick_action(self, observation):
        state = T.Tensor(observation).to(self.actor.device) # convert observation to pythorch sensor and send it to the device
        action, _ = self.actor.normal_sample(state, reparametrize = False) # we dont include the noise
        # here action is array of actions cuz we are dealing with a continuous action space
        return action.cpu().detach().numpy() # we extract and the selected action as np array

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    # The update_pars method is used to perform a soft update of the target value network parameters 
    # This is done by taking a weighted average of the current value network parameters and the target value network parameters, 
    # where the weight is determined by the parameter tau. This method helps to stabilize learning by having a slowly changing 
    # target for the value network to learn from.
    def update_pars(self, tau = None):
        # at the beginning of the simulation we want to set the values for the target network to an 
        # exact copy of the value network so the target value should be an exact copy of the value network,
        # but on every other step we want it to be a soft copy
        if tau == None:
            tau = self.tau
        
        target_val_pars = self.target_val.named_parameters()
        val_pars = self.val.named_parameters()

        target_val_state_dict = dict(target_val_pars)
        val_state_dict = dict(val_pars)

        for name in val_state_dict:
            val_state_dict[name] = tau*val_state_dict[name].clone() + (1-tau)*target_val_state_dict[name].clone()

        self.target_val.load_state_dict(val_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_chkpt()
        self.val.save_chkpt()
        self.target_val.save_chkpt()
        self.critic_1.save_chkpt()
        self.critic_2.save_chkpt()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_chkpt()
        self.val.load_chkpt()
        self.target_val.load_chkpt()
        self.critic_1.load_chkpt()
        self.critic_2.load_chkpt()

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            # we dont learn as we don't have enough memory saved
            return
        
        state, next_state, action, reward, done = self.memory.sample_buffer(self.batch_size)

        action = T.tensor(action, dtype = T.float).to(self.actor.device) 
        reward = T.tensor(reward, dtype = T.float).to(self.actor.device) 
        state = T.tensor(state, dtype = T.float).to(self.actor.device) 
        next_state = T.tensor(next_state, dtype = T.float).to(self.actor.device) 
        done = T.tensor(done).to(self.actor.device) 

        # we compute the values of all the states and all next states
        value = self.val.feed_forward(state).view(-1)
        value_next = self.target_val.feed_forward(next_state).view(-1)

        # for the 'done' states we say that next state value is equal to 0 because this is the terminal state and there is no next state so value of the next state is 0
        value_next[done] = 0.0 

        # --------------------- VALUE NETWORK LOSS: ---------------------------------------------------
        # we pick the action that we should do on our state that we took from the buffer; the following line gives the new policy:
        action_now, log_probs = self.actor.normal_sample(state, reparametrize=False)
        log_probs = log_probs.view(-1)

        # having now new policy, we want to evaluate the quality of (state, action_now) pair based on this new policy: 
        # we do this twice and pick the minimum one because this improves the stability of learning 
        Q1_new_pi = self.critic1.feed_forward(state, action_now)
        Q2_new_pi = self.critic2.feed_forward(state, action_now)
        pair_val_new_pi = T.min(Q1_new_pi, Q2_new_pi)
        pair_val_new_pi = pair_val_new_pi.view(-1)

        value_t = pair_val_new_pi - log_probs
        val_loss = 0.5 * F.mse_loss(value, value_t)
        self.val.optimizer.zero_grad() # zero_grad is used to clear the existing gradients before computing gradients in the backpropagation
        val_loss.backward(retain_graph = True)
        self.val.optimizer.step()

        # --------------------- ACTOR NETWORK LOSS: ---------------------------------------------------
        action_now, log_probs = self.actor.normal_sample(state, reparametrize=True)
        log_probs = log_probs.view(-1)

        Q1_new_pi = self.critic1.feed_forward(state, action_now)
        Q2_new_pi = self.critic2.feed_forward(state, action_now)
        pair_val_new_pi = T.min(Q1_new_pi, Q2_new_pi)
        pair_val_new_pi = pair_val_new_pi.view(-1)

        actor_loss = log_probs - pair_val_new_pi
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor.optimizer.step()

        # --------------------- CRITIC NETWORK LOSS: ---------------------------------------------------

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma*value_next
        Q1_old_pi = self.critic1.feed_forward(state, action).view(-1)
        Q2_old_pi = self.critic2.feed_forward(state, action).view(-1)
        critic1_loss = 0.5 * F.mse_loss(Q1_old_pi, q_hat)
        critic2_loss = 0.5 * F.mse_loss(Q2_old_pi, q_hat)
        critic_loss = critic1_loss + critic2_loss

        critic_loss.backward(retain_graph = True)
        self.actor.optimizer.step()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_pars()