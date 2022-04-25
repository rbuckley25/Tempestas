import torch
import torch.nn as nn
import torch.optim as optim
from utils import replace


import math
import random
import numpy as np

from collections import namedtuple, deque
from itertools import count
from PIL import Image



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    
    def __len__(self):
        return len(self.memory)

def optimize_model(device,input_size,memory,BATCH_SIZE,policy_net,target_net,GAMMA,optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #reshape state_batch for nn
  
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    
    # selects column of output that was selceted 
    state_action_values = policy_net(torch.reshape(state_batch,(BATCH_SIZE,1,input_size)).float()).gather(1,action_batch)
    
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(torch.reshape(non_final_next_states,
                                        (list(non_final_next_states.shape)[0],1,input_size)).float()).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    #gradient clipping
    for param in policy_net.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()
    
    return loss

def select_action(state,policy_net,n_actions,device,EPS_END,EPS_START,EPS_DECAY):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #arg max select the idex of the largest value and view changes shape from (1,) to (1,1)
            #try test net
            return policy_net(state.float()).argmax().view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)