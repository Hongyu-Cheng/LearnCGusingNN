import numpy as np
import torch
from torch.utils.data import DataLoader
from src.IP import *
from src.LTNN import *
from src.data_loader import *

class IPenv:
    def __init__(self, file_paths, num_constraints, num_variables, num_cuts=1, device=torch.device("cpu"), problem_type="knapsack"):
        self.dataset = KnapsackDataset(file_paths)
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.device = device
        self.num_constraints = num_constraints
        self.num_variables = num_variables
        self.num_cuts = num_cuts
        self.problem_type = problem_type
        self.data_loader_iter = iter(data_loader)
        self.reward_list = []
        self.current_state = None
        self.reset()

    def reset(self):
        try:
            self.current_state, self.current_treesize = next(self.data_loader_iter)
            self.current_treesize = self.current_treesize.item()
        except StopIteration:
            data_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            self.data_loader_iter = iter(data_loader)
            self.current_state, self.current_treesize = next(self.data_loader_iter)
            self.current_treesize = self.current_treesize.item()
        return self.current_state

    def get_state(self):
        return self.current_state

    def step(self, action):
        action_np = action.cpu().detach().numpy().reshape(-1,)
        state_np = self.current_state.cpu().detach().numpy().reshape(-1,)
        tree_size_before_cut = self.current_treesize
        ip_instance = vector_to_ip(state_np, self.num_constraints, self.num_variables, self.problem_type)
        alpha, beta = ip_instance.add_chvatal_cut(action_np)
        c = np.array(ip_instance.c)
        ip_instance.optimize()
        tree_size_after_cut = ip_instance.treesize
        reward_value = (tree_size_before_cut - tree_size_after_cut) / tree_size_before_cut
        reward_value = torch.tensor(reward_value).to(self.device).float()

        try:
            next_state = next(self.data_loader_iter)[0]
        except StopIteration:
            data_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            self.data_loader_iter = iter(data_loader)
            next_state = next(self.data_loader_iter)[0]
            is_done = True
        else:
            is_done = False
        self.current_state = next_state

        return next_state, reward_value, is_done
