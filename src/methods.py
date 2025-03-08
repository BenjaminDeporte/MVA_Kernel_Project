import numpy as np
from scipy import optimize#, LinearConstraint

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
                                   constraints=constraints)
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
        return d+self.b> 0


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
        return np.sum((self.support_alpha*self.support_y)[:, np.newaxis]*K_val, axis=0)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return d+self.b> 0
