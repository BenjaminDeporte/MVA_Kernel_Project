import numpy as np
from scipy import optimize
from scipy.optimize import LinearConstraint
from utils import sigmoid

#----------------------------------------
# ALGO SVC BEN
#----------------------------------------

class KernelSVCBen():
    
    def __init__(self, C, kernel, type='non-linear', epsilon = 1e-3):
        self.type = type
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None # support vectors
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        self.X = X
        self.y = y
        # compute gram matrix, we might need it :-)
        self.gram = self.kernel(X,X)
        # vector of ones, size N
        self.ones = np.ones(N)
        # matrix NxN of y_i on diagonal
        self.Dy = np.diag(y)

        # Lagrange dual problem
        def loss(alpha):
            objective_function = 1/2 * alpha @ self.gram @ alpha - alpha @ self.y
            return  objective_function

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            gradient = self.gram @ alpha - self.y
            return gradient

        # equality constraint
        fun_eq = lambda alpha: alpha @ self.ones      
        jac_eq = lambda alpha: self.ones
        # inequality constraint avec la classe LinearConstraint de scipy
        inequality_constraint = LinearConstraint(self.Dy, np.zeros(N), self.C * self.ones)
        
        constraints = ( [{'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},
                        inequality_constraint]
                        )

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints,
                                   )
        self.alpha = optRes.x

        ## Assign the required attributes
        # list of indices of support vectors in dataset, None if not a support vector
        self.indices_support = np.array([ i if (self.epsilon < self.alpha[i]*self.y[i]) and (self.alpha[i]*self.y[i] <= self.C) else None for i in range(N) ])
        self.indices_support = self.indices_support[self.indices_support != None].astype(int)
        # support vectors (data points on margin)
        self.support = self.X[self.indices_support]
        # alphas on support vectors
        self.alpha_support = self.alpha[self.indices_support]
        # compute b by averaging over support vectors
        b = self.y - self.gram @ self.alpha
        b_sv = b[self.indices_support]
        self.b = np.mean(b_sv)
        # '''------------------------RKHS norm of the function f ------------------------------'''
        self.norm_f = 1/2 * self.alpha @ self.gram @ self.alpha
        
        return self


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.support) @ self.alpha_support + self.b
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        y_pred = np.where(d + self.b >0, 1, -1)
        
        return y_pred


#----------------------------------------
# ALGO SVC LILIAN
#----------------------------------------

class KernelSVCLilian():
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None # support vectors
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel(X, X)

        # Lagrange dual problem
        def loss(alpha):
            return 0.5*np.dot(alpha*y, np.dot(K, alpha*y)) - np.sum(alpha) #'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return np.dot(K, alpha*y)*y - np.ones_like(alpha) # '''----------------partial derivative of the dual loss wrt alpha -----------------'''


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha: np.dot(alpha, y)# '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha: y #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha: self.C - alpha # '''---------------function defining the inequality constraint-------------------'''     
        jac_ineq = lambda alpha: -np.eye(N) # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        fun_ineq2 = lambda alpha: alpha # '''---------------function defining the inequality constraint-------------------'''     
        jac_ineq2 = lambda alpha: np.eye(N) # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 'fun': fun_ineq, 'jac': jac_ineq},
                       {'type': 'ineq', 'fun': fun_ineq2, 'jac': jac_ineq2}
                      )

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes
        idx = self.alpha > self.epsilon
        
        self.support = X[idx]#'''------------------- A matrix with each row corresponding to support vectors ------------------'''
        self.support_alpha = self.alpha[idx]
        self.support_y = y[idx]
        
        self.b =  np.mean(self.support_y - self.separating_function(self.support))#''' -----------------offset of the classifier------------------ '''
        self.norm_f = np.sqrt(np.dot(self.alpha*y, np.dot(K, self.alpha*y)))# '''------------------------RKHS norm of the function f ------------------------------'''


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K_val = self.kernel(self.support, x)
        return (self.support_alpha*self.support_y)@K_val
    
    
    def predict(self, X):
        """ Predict y values in {0, 1} """
        d = self.separating_function(X)
        return d+self.b> 0

#----------------------------------------
# ALGO KLR
#----------------------------------------

class WeightedKernelRR:
    def __init__(self, kernel, weights, lmbda):
        self.lmbda = lmbda
        self.kernel = kernel
        self.weights = weights
        self.alpha = None
        self.b = None
        self.support = None
        self.type = "ridge"

    def fit(self, y):
        #self.support = X
        #kernel = self.kernel(X, X)
        W = np.sqrt(np.diag(self.weights))
        n = len(y)
        inv = np.linalg.inv(W@self.kernel@W + self.lmbda * n * np.eye(n))
        self.alpha = np.dot(W, np.dot(inv, np.dot(W, y)))
        #self.b = -np.mean(self.regression_function(X))

    ### Implementation of the separeting function $f$
    def regression_function(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.support) @ self.alpha.T

    def predict(self, X):
        return self.regression_function(X) + self.b

class KernelLR:
    def __init__(self, kernel, lmbda, iters, tol):
        self.lmbda = lmbda
        self.kernel = kernel
        self.iters = iters
        self.tol = tol
        self.alpha = None
        self.b = None
        self.support = None
        self.type = "ridge"

    def fit(self, X, y):
        self.support = X
        kernel = self.kernel(X, X)
        n = len(y)
        self.alpha = np.random.rand(n)
        for _ in range(self.iters):
            alpha0 = self.alpha
            m = kernel@self.alpha.T
            W = sigmoid(y*m)*sigmoid(-y*m)
            z = m + y/np.maximum(sigmoid(y*m), 1.e-6)
            wkrr = WeightedKernelRR(kernel=kernel, weights=W, lmbda=self.lmbda)
            wkrr.fit(z)
            self.alpha = wkrr.alpha

            if np.linalg.norm(self.alpha - alpha0) < self.tol:
                break
            
    ### Implementation of the separeting function $f$
    def regression_function(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.support) @ self.alpha.T

    def predict(self, X):
        return np.array(sigmoid(self.regression_function(X))>=0.5, dtype=int)
