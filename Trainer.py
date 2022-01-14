import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import carla
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

from Memory import ReplayMemory

import gym
import gym_carla


class CL_Trainer(model,memory_size,n_actions,n_episodes,batch_size,gamma,EPS,target_update):
    def __init__(self,model,memory_size,n_actions,n_episodes,batch_size,gamma,EPS,target_update,lr):
        self.policy_net = model.to(device)
        self.target_net = model.to(device)
        self.target_net.load_state_dict(policy_net.state_dict())
        self.memory = ReplayMemory(memory_size)
        self.n_actions = n_actions
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_Start = EPS[0]
        self.EPS_END = EPS[1]
        self.EPS_DECAY = EPS[2]
        self.writer = SummaryWriter()
        self.optimizer = optim.RMSprop(policy_net.parameters(),lr=lr)
        self.steps_done = 0
        self.n_episodes = n_episodes
        self.target_update = target_update
        self.Transition = namedTuple('Transition','state','action','next_state,','reward')
        self.state = None
        self.next_state = None
        self.device = torch.device("cuda" if torch.cude.is_available() else "cpu")


    def select_action(state):
        sample = random.random()
        eps_thres = self.EPS_END + (EPS_START - EPS_END) * \ math.exp(-1*steps_done/EPS_DECAY)
        steps_done += 1
        if sample > eps_thres:
            with torch.no_grad():
                return policy_net(state.float()).argmax.view(1,1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],device=self.device, dtype=torch.long)


    def optimize():
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = self.Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])








