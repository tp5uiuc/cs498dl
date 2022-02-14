import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=learning_rate
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )

        # Initialize a target network and initialize the target network to the policy net
        ### CODE ###
        self.target_net = DQN(action_size).to(device)
        self.update_target_net()

        self.criterion = nn.SmoothL1Loss(reduction="mean")

    def load_policy_net(self, path):
        self.policy_net.load_state_dict(torch.load(path))

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        ### CODE ###
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_action(self, state):
        """Get action using policy net using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            ### CODE #### (copy over from agent.py!)
            a = np.random.choice(self.action_size)
        else:
            ### CODE #### (copy over from agent.py!)
            # Choose the best action
            with torch.no_grad():
                # state passed in is a numpy array
                # we need to unsqeeuze it as per 875 in piazza
                state_on_device = torch.from_numpy(state).unsqueeze_(dim=0).to(device)
                current_Q_value = self.policy_net(state_on_device)
                # action is just a scalar value
                a = torch.argmax(current_Q_value).item()  # .numpy()

        return a

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.0
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.0
        next_states = torch.from_numpy(next_states).cuda()

        dones = mini_batch[3]  # checks if the game is over
        musk = torch.tensor(
            list(map(int, dones == False)), dtype=torch.uint8, device=device
        )

        # Your agent.py code here with double DQN modifications
        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        # The gather function used is
        # taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # it is equivalent to [arange(batch_size), actions] to select
        # the Q value to be updated, which corresponds to the action
        # self.policy_net(states) is [32, 3]
        # actions here is [32]
        # they need to have some number of dimensions for gather to work
        # so we do actions.unsqueeze on the last dimension to make it [32, 1]
        current_Q_values = self.policy_net(states).gather(1, actions.unsqueeze(dim=1))

        with torch.no_grad():
            # Compute maximizing action
            next_maximizing_action = torch.argmax(self.policy_net(next_states), dim=1)
            # print(next_maximizing_action.size())

            # Compute Q function of next state
            ### CODE ####
            next_Q_values = self.target_net(next_states)
            # print("next Q ", next_Q_values.size())

            # Find maximum Q-value of action at next state from policy net
            ### CODE ####
            # [0] returns the maximal values
            # [1] returns the argmax
            max_next_Q_values = next_Q_values.gather(
                1, next_maximizing_action.unsqueeze(dim=1)
            ).squeeze(dim=-1)
            # print("max next Q ", max_next_Q_values.size())
            # print("Rewards ", rewards.size())
            # print("Musk ", musk.size())

        # Temporal difference for loss
        expected_Q_values = (self.discount_factor * musk * max_next_Q_values) + rewards
        # print(current_Q_values.size())
        # print(expected_Q_values.size())

        # Compute the Huber Loss
        ### CODE ####
        # current_Q_values is [32, 1]
        # expected_Q_values is [32]
        # so we squeeze current Q values
        loss = self.criterion(current_Q_values.squeeze(dim=-1), expected_Q_values)

        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
