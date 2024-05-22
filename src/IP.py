import numpy as np
import gurobipy as gp
from gurobipy import GRB


class IP:
    def __init__(self, A=None, c=None, b=None, filename=None, vtype='binary', sense='maximize', treesize_limit=100000, presolve=False, verbose=False):
        self.vtype = vtype
        self.sense = sense
        self.verbose = verbose
        self.presolve = presolve
        self.model = gp.Model()
        if verbose == False:
            self.model.setParam("OutputFlag", 0)
        if presolve == False:
            self.model.setParam("PreCrush", 1)
            self.model.setParam("Heuristics", 0.0)
            self.model.setParam("Cuts", 0)
            self.model.setParam("Presolve", 0)
        self.is_optimized = False
        self.treesize_limit = treesize_limit
        self.model.setParam("NodeLimit", self.treesize_limit)
        self.num_cons = 0
        self.num_vars = 0
        if filename:
            self.read_from_file(filename)
        else:
            self.read_from_arrays(A, c, b)

    def read_from_arrays(self, A, c, b):
        if A is None or c is None or b is None:
            print("Error: Input arrays cannot be None.")
            return
        if type(A) != np.ndarray or type(c) != np.ndarray or type(b) != np.ndarray:
            print("Error: Input arrays must be numpy arrays.")
            return
        self.num_cons, self.num_vars = A.shape
        self.A, self.c, self.b = A, c, b
        
        if self.vtype == 'binary':
            self.A = np.vstack([self.A, -np.eye(self.num_vars), np.eye(self.num_vars)])
            self.b = np.hstack([self.b, np.zeros(self.num_vars), np.ones(self.num_vars)])
            
            self.num_cons += 2*self.num_vars
        if self.vtype == 'binary':
            x = self.model.addVars(self.num_vars, vtype=GRB.BINARY, name="x")
        elif self.vtype == 'integer':
            x = self.model.addVars(self.num_vars, vtype=GRB.INTEGER, name="x")
        elif self.vtype == 'continuous':
            x = self.model.addVars(
                self.num_vars, vtype=GRB.CONTINUOUS, name="x")
        else:
            print("Error: Invalid variable type.")
            return

        if self.sense == 'maximize':
            self.model.setObjective(
                sum(self.c[i] * x[i] for i in range(self.num_vars)), GRB.MAXIMIZE)
        elif self.sense == 'minimize':
            self.model.setObjective(
                sum(self.c[i] * x[i] for i in range(self.num_vars)), GRB.MINIMIZE)
        else:
            print("Error: Invalid optimization sense.")
            return

        self.model.addConstrs((gp.quicksum(self.A[i, j] * x[j] for j in range(
            self.num_vars)) <= self.b[i] for i in range(self.num_cons)), name="cons")
        self.model.update()

    def read_from_file(self, filename):
        try:
            data = np.load(filename, allow_pickle=True).item()
            A = data['A']
            c = data['c']
            b = data['b']
            self.read_from_arrays(A, c, b)
            try:
                self.x_LP = data['x_LP']
            except:
                pass
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return

    def add_cut(self, a, b):
        if not isinstance(a, np.ndarray):
            if isinstance(a, list):
                a = np.array(a)
            else:
                try:
                    a = np.array(a.flatten()).astype(np.float64)
                except:
                    print("Error: The input a cannot be converted to a numpy array.")
                    return
        if not np.isscalar(b):
            try:
                b = np.array(b.flatten()).astype(np.float64)[0]
            except:
                print(
                    f"Error: The input b cannot be converted to a scalar. b has {len(b.flatten())} elements.")
                return
        if len(a) != self.num_vars:
            print("Error: The dimension of a must match the number of variables.")
            return
        self.model.addConstr(
            sum(a[i] * self.model.getVarByName(f"x[{i}]") for i in range(self.num_vars)) <= b)

        self.model.update()
        self.A = np.vstack([self.A, a])
        self.b = np.hstack([self.b, b])
        self.num_cons += 1
        self.is_optimized = False

    def add_chvatal_cut(self, u):
        if not isinstance(u, np.ndarray):
            if isinstance(u, list):
                u = np.array(u)
            else:
                try:
                    u = np.array(u.flatten()).astype(np.float64)
                except:
                    print("Error: The input u cannot be converted to a numpy array.")
                    return
        if len(u) != self.num_cons:
            print("Error: The dimension of u must match the number of constraints.")
            return
        a_new = np.floor(u @ self.A)
        b_new = np.floor(u @ self.b)
        self.add_cut(a_new, b_new)
        return a_new, b_new

    def add_sequence_chvatal_cut(self, u, k):
        u = np.array(u).astype(np.float64).flatten()

        expected_length = self.num_cons * k + k * (k - 1) // 2
        if len(u) != expected_length:
            print(f"Error: The dimension of u must match the required length {expected_length}.")
            return

        start_idx = 0
        for i in range(k):
            u_i_length = self.num_cons
            u_i = u[start_idx: start_idx + u_i_length]
            start_idx += u_i_length
            self.add_chvatal_cut(u_i)
    
    def to_standard_form(self, vtype = 'integer', sense = 'maximize'):
        self.standard_model = gp.Model()
        self.standard_model.setParam("OutputFlag", 0)
        self.standard_model.setParam("PreCrush", 1)
        self.standard_model.setParam("Heuristics", 0.0)
        self.standard_model.setParam("Cuts", 0)
        self.standard_model.setParam("Presolve", 0)
        self.standard_model.setParam("NodeLimit", self.treesize_limit)
        s_cons, s_vars = self.A.shape
        s_vars += s_cons
        if vtype == 'binary':
            x = self.standard_model.addVars(s_vars, vtype=GRB.BINARY, name="x")
        elif vtype == 'integer':
            x = self.standard_model.addVars(s_vars, vtype=GRB.INTEGER, name="x")
        elif vtype == 'continuous':
            x = self.standard_model.addVars(
                s_vars, vtype=GRB.CONTINUOUS, name="x")
        else:
            print("Error: Invalid variable type.")
            return
        if sense == 'maximize':
            self.standard_model.setObjective(
                sum(self.c[i] * x[i] for i in range(s_vars-s_cons)), GRB.MAXIMIZE)
        elif sense == 'minimize':
            self.standard_model.setObjective(
                sum(self.c[i] * x[i] for i in range(s_vars-s_cons)), GRB.MINIMIZE)
        else:
            print("Error: Invalid optimization sense.")
            return
        self.standard_model.addConstrs((gp.quicksum(self.A[i, j] * x[j] for j in range(
            s_vars-s_cons)) + x[s_vars-s_cons+i] == self.b[i] for i in range(s_cons)), name="cons")
        self.standard_model.update()
        return self.standard_model

    def optimize(self):
        self.is_optimized = True
        self.model.optimize()
    
    @classmethod
    def from_dict(cls, ip_dict):
        return cls(
            A=ip_dict["A"],
            c=ip_dict["c"],
            b=ip_dict["b"],
            vtype=ip_dict["vtype"],
            sense=ip_dict["sense"],
            treesize_limit=ip_dict["treesize_limit"],
            presolve=ip_dict["presolve"],
            verbose=ip_dict["verbose"]
        )

    @property
    def treesize(self):
        if self.model.status == GRB.OPTIMAL:
            return self.model.NodeCount
        elif self.model.status == 1:
            print("Warning: The model has not been optimized yet.")
            return -1
        else:
            print("Warning: Tree size limit reached.")
            return self.treesize_limit

    @property
    def is_optimal(self):
        if self.is_optimized == False:
            print("Warning: The model has not been optimized yet.")
        else:
            return self.model.status == GRB.OPTIMAL

    @property
    def info(self):
        return {
            "Num of cons": self.num_cons, 
            "Num of vars": self.num_vars, 
            "Is optimized?": bool(self.is_optimized), 
            "Is optimal?": bool(self.is_optimal), 
            "Tree size limit": self.treesize_limit
        }
    
    @property
    def in_channel(self):
        return self.num_cons*self.num_vars + self.num_cons + self.num_vars
    
    def relax(self):
        return self.model.relax()
    
    @property
    def get_LP_sol(self):
        relaxed_model = self.relax()
        relaxed_model.optimize()
        return np.array(relaxed_model.getAttr('x',relaxed_model.getVars()))
    
    @property
    def x_IP(self):
        if self.model.status == GRB.OPTIMAL:
            return np.array(self.model.getAttr('x',self.model.getVars()))
        else:
            self.optimize()
            return np.array(self.model.getAttr('x',self.model.getVars()))

    def print_solution(self):
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found")
            for v in self.model.getVars():
                print(f"{v.varName} = {v.x}")
        else:
            print("No optimal solution found")

    def print_objective(self):
        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal objective value: {self.model.objVal}")
        else:
            print(f"No optimal solution found, objective value: {self.model.objVal}")

    def print_model(self):
        print(self.model)


def matrix_to_vector(A, c, b, mode = 'simple'):
    num_cons, num_vars = A.shape[0], A.shape[1]
    if mode == "simple":
        A = A[:num_cons-2*num_vars, :]
        b = b[:num_cons-2*num_vars]
        return np.hstack([A.flatten(), c.flatten(), b.flatten()])
    elif mode == "full":
        return np.hstack([A.flatten(), c.flatten(), b.flatten()])
        


def ip_to_vector(x, mode='simple'):
    return matrix_to_vector(x.A, x.c, x.b, mode=mode)


def vector_to_matrix(x, num_cons, num_vars, problem_type='knapsack'):
    A = np.array(x[:num_cons*num_vars].reshape(num_cons,num_vars)).astype(np.float64)
    c = np.array(x[num_cons*num_vars:-num_cons]).astype(np.float64)
    b = np.array(x[-num_cons:]).astype(np.float64)
    if problem_type == 'knapsack':
        A = A[:num_cons-2*num_vars, :]
        b = b[:num_cons-2*num_vars]
    return A, c, b


def vector_to_ip(x, num_cons, num_vars, problem_type='knapsack'):
    A, c, b = vector_to_matrix(x, num_cons, num_vars, problem_type)
    return IP(A, c, b)

def f(u, x, num_cons, num_vars, num_cuts=1):
    ip = vector_to_ip(x, num_cons, num_vars)
    ip.add_sequence_chvatal_cut(u, num_cuts)
    ip.optimize()
    return ip.treesize


