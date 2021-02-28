import numpy as np
import numexpr as ne
from scipy.linalg.blas import sgemm


def rbf(X, Y = None, gamma = 1.0, gradient=False):
    """
    Compute row wise kernel matrix of X and Y.
    
    Parameters
    ----------
    X : numpy array
        first array of size (n x m).
    Y : numpy array, optional
        second array of size (o x m). The default is None.
    gamma : float, optional
        Length scale. The default is 0.5.

    Returns
    -------
    kernel matrix as numpy array of size (n x o).
    """
    
    
    XX = np.einsum('ij,ij -> i', X, X)
    
    if Y is None:
        Y = X
        YY = XX
        Y_flag = True
    else:
        YY = np.einsum('ij,ij -> i', Y, Y)
        Y_flag = False
    
    
    dist = ne.evaluate('(A + B - C) / g**2', {
            'A' : XX[:,None],
            'B' : YY[None,:],
            'C' : sgemm(alpha=2, a=X, b=Y, trans_b=True),
            'g' : gamma})
    
    if Y_flag: np.fill_diagonal(dist, 0)
    
    K = np.exp(-0.5 * dist)
    
    if gradient:
        grad = K * dist
        np.fill_diagonal(grad, 0)
        
        return K, grad
    
    return K

class RBF:
    
    def __init__(self, gamma, bounds = [(1.0e-5, 1.0e+5)]):
        # self.sigma = np.abs(sigma)
        
        self.gamma = gamma
        self.bounds = bounds
        
    def __call__(self, X, Y=None, gradient=False):
        return rbf(X, Y, self.gamma, gradient)
    
    
def poly_kernel(X, Y=None, degree=3, gamma=None, coef=1):
    """
    Copy paste from sklearn. not yet working.
    """

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = np.dot(X, Y.T)
    K *= gamma
    K += coef
    K **= degree
    return K
       
class PolynomialKernel:
    
    def __init__(self, degree, gamma, coef, 
                 bounds = [(1.0e-5, 1.0e+5),(1.0e-5, 1.0e+5),(1.0e-5, 1.0e+5)]):
        self.degree = degree
        self.gamma = gamma
        self.coef = coef
        self.bounds = bounds
        
    def __call__(self, X, Y=None):
        return poly_kernel(X, Y, self.degree, self.gamma, self.coef)
    




class Kernel:
    """
    Initilizes common interface for Kernels

    Parameters
    ----------
    name : string
        Kernel name ('rbf' or 'poly')
    hyper : array
        array with hyperparameters for the kernel
    bounds : list of tuples, optional
        lower and upper limit for each hyperparameter e.g. [(lower,upper)]
    """
    def __init__(self, name, hyper, bounds = None):

        self._bounds = bounds
        self._hyper = hyper
        
        if bounds != None and len(hyper) != len(bounds):
            raise ValueError("length of hyper parameters and bounds must match.")
            
            
        if name == "rbf":
            self.kernel = RBF(hyper)
        elif name == "poly":
            self.kernel = PolynomialKernel(hyper[0], hyper[1], hyper[2])
        else:
            raise ValueError("Given kernel name not recognized.")
           
    @property
    def bounds(self):
        if self._bounds is not None:
            self.kernel.bounds = self._bounds
        return np.log(self.kernel.bounds)

    
    @bounds.setter
    def bounds(self, bounds):
        self._bounds = bounds
        
    @property
    def hyper(self):            
        return self._hyper

    @hyper.setter
    def hyper(self, hyper):
        if isinstance(self.kernel, RBF):
            self.kernel.gamma = hyper
            
        elif isinstance(self.kernel, PolynomialKernel):
            self.kernel.degree = hyper[0]
            self.kernel.gamma = hyper[1]
            self.kernel.coef = hyper[2]
        self._hyper = hyper
        return hyper
    
        
    def estimate(self, X, Y=None, gradient=False):
        return self.kernel(X, Y, gradient)