import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
import sys
sys.path.append('../')
from src.IP import *

def generate_knapsack_ip(N, K, instances=1, filename=None, x_LP = False):
    """
    Generates an integer programming formulation for a multi-dimensional knapsack problem.
    
    Parameters:
    - N: int, number of items
    - K: int, number of knapsacks
    - instances: int, number of problem instances to generate
    - filename: str, optional, output directory to save the integer programming formulations
    
    Returns:
    None
    
    Mathematically, the formulation is:
    $ \max \sum_{k=1}^{K} \sum_{i=1}^{N} w[i] x_{ik} $
    $ \text{s.t.} $
    $ \sum_{i=1}^{N} w[i] x_{ik} \leq W_k, \forall k \in K$
    $ x_{ik} \in \{0, 1\} $
    """
    for instance in range(instances):
        # Generate weights and capacities
        w = np.floor(np.random.normal(50, 2, N))
        W = np.array([np.floor(np.sum(w) / (2 * K)) + (k - 1) for k in range(1, K + 1)])
        
        # Create constraint matrix A
        zero_N = np.zeros(N)
        A_K = [0] * K
        for i in range(K):
            Ai = [zero_N for _ in range(K)]
            Ai[i] = w
            A_K[i] = np.hstack(Ai)
        A_K = np.vstack(A_K)
        A_N = np.hstack([np.eye(N) for _ in range(K)])
        A = np.vstack([A_K, A_N])

        # Create right-hand side vector b and objective function coefficients c
        b = np.hstack([W, np.ones(N)])
        c = np.hstack([w for _ in range(K)])
        if instances == 1:
            return A, c, b

        if filename:
            instance_filename = os.path.join(filename, f"knapsack_ip_data_{instance}.npy")
            if x_LP == False:
                np.save(instance_filename, {'A': A, 'c': c, 'b': b})
            else:
                ip = IP(A=A,c=c,b=b)
                np.save(instance_filename, {'A': A, 'c': c, 'b': b, 'x_LP': ip.x_LP})