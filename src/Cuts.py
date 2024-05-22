import ast, getopt, sys, copy, os
from fractions import Fraction
import numpy as np
import torch
from fractions import Fraction as frac
from src.IP import *
from math import floor, gcd, lcm, inf

def is_integer(x):
    return np.isclose(x, np.round(x))

def get_simplex_tableau(A, c, b):
    tab = [[-frac(x) for x in c] + [0]*(len(b)+1)] + [[frac(x) for x in r] + [i==j for j in range(len(A))] + [frac(b[i])] for i, r in enumerate(A)]
    while any(x < 0 for x in tab[0][:-1]):
        j_star = min((x, i) for i, x in enumerate(tab[0][:-1]))[1]
        i_star = min([(x[-1] / x[j_star], i) for i, x in enumerate(tab) if x[j_star] > 0]+[(inf,inf)])[1]
        if i_star == inf: return None
        tab[i_star] = [x / tab[i_star][j_star] for x in tab[i_star]]
        tab = [x if i == i_star else [y - x[j_star] * tab[i_star][j] for j, y in enumerate(x)] for i, x in enumerate(tab)]
    return tab

class cut_generator:
    def __init__(self, A, c=None, b=None):
        if isinstance(A,np.ndarray):
            self.A = A
            self.c = c
            self.b = b
        else:
            self.ip = A
            self.A = A.A
            self.c = A.c
            self.b = A.b
        self.m, self.n = self.A.shape
        
        self.fraction_tableau=get_simplex_tableau(self.A, self.c, self.b)[1:]
        self.np_tableau = np.array(self.fraction_tableau).astype(float)[:-1,:]
        self.candidate_rows_indices = np.where(~np.isclose(self.np_tableau[:,-1], np.round(self.np_tableau[:,-1])))[0].tolist()   
        self.candidate_rows = self.np_tableau[self.candidate_rows_indices,:]
        
    def candidate_GMI(self):
        self.GMI_a = []
        self.GMI_b = []
        for cut_row_index in self.candidate_rows_indices:
            cut_row = copy.deepcopy(self.fraction_tableau[cut_row_index])
            GMI_f0 = cut_row[-1] - floor(cut_row[-1])
            GMI_f = [Fraction(0,1)] * (self.m+self.n)
            GMI_cut = [Fraction(0,1)] * (self.m+self.n)
            for j in range(self.m+self.n):
                GMI_f[j] = cut_row[j] - floor(cut_row[j])
                if GMI_f[j] <= GMI_f0:
                    GMI_cut[j] = GMI_f[j]/GMI_f0
                else:
                    GMI_cut[j] = (1-GMI_f[j])/(1-GMI_f0)
            GMI_cut_x = [-GMI_cut[i] for i in range(self.n)]
            GMI_cut_s = [-GMI_cut[i] for i in range(self.n, self.m+self.n)]
            alpha = np.array(GMI_cut_x) - np.array(GMI_cut_s)@self.A.astype(int)
            beta = -1 - np.array(GMI_cut_s)@self.b.astype(int)
            LCM = lcm(*([i.denominator for i in alpha] + [beta.denominator]))
            GCD = gcd(*([i.numerator for i in alpha] + [beta.numerator]))
            alpha = (alpha*LCM/GCD).astype(float)
            beta = float(beta*LCM/GCD)
            self.GMI_a.append(alpha)
            self.GMI_b.append(beta)
        self.GMI_a = np.array(self.GMI_a)
        self.GMI_b = np.array(self.GMI_b).reshape(-1,1)
        self.GMI_cuts = np.hstack([self.GMI_a, self.GMI_b])
        return self.GMI_a, self.GMI_b.reshape(-1,)
    
    def candidate_CG(self):
        CG_a = []
        CG_b = []
        for cut_row_index in self.candidate_rows_indices:
            cut_row = copy.deepcopy(self.fraction_tableau[cut_row_index])
            CG_LHS = [cut_row[i] - floor(cut_row[i]) for i in range(self.m+self.n)]
            CG_RHS = cut_row[-1] - floor(cut_row[-1])
            CG_cut_x = [-CG_LHS[i] for i in range(self.n)]
            CG_cut_s = [-CG_LHS[i] for i in range(self.n, self.m+self.n)]
            alpha = np.array(CG_cut_x) - np.array(CG_cut_s) @ self.A.astype(int)
            beta = -CG_RHS - np.array(CG_cut_s) @ self.b.astype(int)
            LCM = lcm(*([i.denominator for i in alpha] + [beta.denominator]))
            GCD = gcd(*([i.numerator for i in alpha] + [beta.numerator]))
            alpha = (alpha*LCM/GCD).astype(float)
            beta = float(beta*LCM/GCD)
            CG_a.append(alpha)
            CG_b.append(beta)
        CG_a = np.array(CG_a)
        CG_b = np.array(CG_b).reshape(-1,1)
        self.CG_cuts = np.hstack([CG_a, CG_b])

        return CG_a, CG_b.reshape(-1,)

