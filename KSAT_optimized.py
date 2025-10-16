import numpy as np
# import matplotlib.pyplot as plt

from copy import deepcopy

## The `probl` object must implement these methods:
##    init_config()               # returns None [changes internal config]
##    cost()                      # returns a real number
##    propose_move()              # returns a (problem-dependent) move - must be symmetric!
##    compute_delta_cost(move)    # returns a real number
##    accept_move(move)           # returns None [changes internal config]
##    copy()                      # returns a new, independent opbject
## NOTE: The default beta0 and beta1 are arbitrary.

class KSAT:
    def __init__(self, N, M, K, seed = None):
        if not (isinstance(K, int) and K >= 2):
            raise Exception("k must be an int greater or equal than 2")
        self.K = K
        self.M = M
        self.N = N

        ## Optionally set up the random number generator state
        if seed is not None:
            np.random.seed(seed)

        ## Random elements in {0,1}
    
        s = np.random.choice([-1,1], size=(M,K))
        index = np.zeros((M,K), dtype = int)
        
        for m in range(M):
            index[m] = np.random.choice(N, size=(K), replace=False)
            
        # Dictionary for keeping track of literals in clauses
        clauses = []   
        for n in range(N):
            clauses.append([i for i, row in enumerate(index) if n in row])
        
        self.s, self.index, self.clauses = s, index, clauses        
        
        ## Inizializza la configurazione
        x = np.ones(N, dtype=int)
        self.x = x
        self.init_config()

    ## Initialize (or reset) the current configuration
    def init_config(self):
        N = self.N 
        
        self.x[:] = 2*np.random.randint(0,2, size=(N))-1
        
        
    # Definition of the cost function
    def cost(self):
        K, M, x, s, index = self.K, self.M, self.x, self.s, self.index
        
        conf = np.zeros((M,K))
        for k in range(K):
            for m in range(M):
                conf[m][k] = x[int(index[m][k])]
        
        cost = 0.0
        for m in range(M):
            vec = (1 - conf[m]*s[m])/2
            cost += np.prod(vec)
        return cost
    
    ## Propose a valid random move. 
    def propose_move(self):
        N = self.N
        move = np.random.choice(N)
        return move
    
    ## Modify the current configuration, accepting the proposed move
    def accept_move(self, move):
        self.x[move] *= -1

    ## Compute the extra cost of the move (new-old, negative means convenient)
    def compute_delta_cost(self, move):
        x, s, index, clauses = self.x, self.s, self.index, self.clauses
        
        old_cost = 0
        for m in clauses[move]:
            old_cost += np.prod((1 - s[m] * x[index[m]] )/ 2) 
        
        x[move] *= -1
        
        new_cost = 0
        for m in clauses[move]:
            new_cost += np.prod((1 - s[m] * x[index[m]] )/ 2)
        
        x[move] *= -1
        
        return new_cost - old_cost
    

    ## Make an entirely independent duplicate of the current object.
    def copy(self):
        return deepcopy(self)
    
    def display(self):
        pass
        

