import copy

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize

from kernels import Kernel
from utils import validate_X_Y


class GaussianProcess():
    """
    Class for estimating the Gaussian Process given data
    
    Parameters
    ----------
    X_train : numpy array
        samples x variables
    Y_train : numpy array
        samples x 1
    X : numpy array
        new values x variables
    kernel : function
        instance of the kernel class with working kernel selected
    sigma_n : float
        noise added to K diagonal
    jac : bool
        set to False if the kernel does not return a gradient
    """
    
    def __init__(self, X_train, Y_train, kernel, sigma_n=1.0e-5, jac=True):
                
        if not isinstance(kernel, Kernel):
            raise Warning("The passed kernel is not an instance of Kernel.")
        
        self.X_train, self.Y_train = validate_X_Y(X_train, Y_train)
        
        self.kernel = copy.deepcopy(kernel)
        self.jac = jac
        
        self.sigma_n = sigma_n
        self.ntrain = X_train.shape[0]
        self.id_mat = np.eye(self.ntrain)
    
    def update_train_data(self, X, Y, concat=False):
        
        if concat:
            X = X.reshape(-1, self.X_train.shape[1])
            Y = Y.reshape(-1, 1)
            
            self.X_train = np.concatenate((self.X_train, X), axis=0)
            self.Y_train = np.concatenate((self.Y_train, Y), axis=0)
        else:
            self.X_train, self.Y_train = validate_X_Y(X, Y)
        
        self.ntrain = self.X_train.shape[0]
        self.id_mat = np.eye(self.ntrain)

    @property
    def theta(self):
        return np.log(self.kernel.hyper)
    
    def log_marginal_likelihood(self, theta=None, gradient=True, opt_flag=False):
        """
        Calculate the log marginal likelihood as in 
        Rasmussen und Williams (2006) Algorithm 2.1

        Parameters
        ----------
        theta : array, optional
            list of hyper parameters
        gradient : bool, optional
            True if the kernel function should return the gradient
        opt_flag : bool, optional
            True if this function gets called for optimization

        Returns
        -------
        log marginal likelihood : float
        """
        if theta != None:
            self.kernel.hyper = np.exp(theta) if opt_flag else theta
        # self.kernel.hyper
        if gradient:
            K, grad = self.kernel.estimate(self.X_train, gradient=True)
        else:
            K = self.kernel.estimate(self.X_train)
            
        K += self.id_mat * self.sigma_n

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return (np.inf, np.array([0])) if gradient else np.inf
               
        alpha = cho_solve((L, True), self.Y_train)
        
        logl = float(self.Y_train.T.dot(alpha)) / 2
        logl += np.log(np.diag(L)).sum()
        logl += self.ntrain * np.log(2 * np.pi) / 2

        
        if gradient:
            logl_grad = alpha.dot(alpha.T) # einsum is slower
            logl_grad -= cho_solve((L, True), self.id_mat)
            logl_grad = 0.5 * np.einsum('ij,ji -> ', logl_grad, grad) #dot prod and trace combined
            return logl, -np.array([logl_grad])
        return logl
    
    def opt_kernel_hyper(self):
        
        args = (self.jac, True)
        
        opt_res = minimize(self.log_marginal_likelihood, self.theta, 
                           args = args, method = "L-BFGS-B", jac=self.jac, 
                           bounds = self.kernel.bounds)
        
        return opt_res.x
        
    
    def posterior(self, X, opt_kernel=True, calc_cov=True):
        """
        Calculate posterior average and covariance
        Rasmussen und Williams (2006) Algorithm 2.1

        Parameters
        ----------
        X : numpy array
            columns have to be the same size as in X_train.
        opt_kernel : bool, optional
            True if the hyperparameters should be optimized
        calc_cov : bool, optional
            True if the covariance matrix should be returned

        Returns
        -------
        mu
            estimated and predicted avwerage
        cov : optional
            estimated and predicted covariance
        """
        
        X = X.reshape(-1, self.X_train.shape[1])
        if opt_kernel:
            self.kernel.hyper = np.exp(self.opt_kernel_hyper())
        
        K = self.kernel.estimate(self.X_train) + self.id_mat * self.sigma_n
        L = np.linalg.cholesky(K)
        alpha = cho_solve((L, True), self.Y_train)
        K_s = self.kernel.estimate(X, self.X_train)
        
        mu = K_s.dot(alpha)
        
        if calc_cov:
            cov = self.kernel.estimate(X)
            cov -= K_s.dot(cho_solve((L, True), K_s.T))
            return mu, cov
        
        return mu