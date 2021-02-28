import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize

from gaussian_process import GaussianProcess


def expected_improvement(X_new, X_sample, Y_sample, gpr, xi=0.001):
    """
    Acquisition function expected improvment (compare Martin Krasser)

    Parameters
    ----------
    X_new : numpy array
        new input values.
    X_sample : numpy array
        past input values.
    Y_sample : numpy array
        past function values.
    gpr : GaussianProcess
        initilized gaussian process
    xi : float, optional
        learning rate. The default is 0.1.

    Returns
    -------
    ei : numpy array
        expected improvement

    """
    mu, cov = gpr.posterior(X_new, opt_kernel=False)
    mu_sample = gpr.posterior(X_sample, opt_kernel=False, calc_cov=False)

    mu = mu.ravel()    
    std = np.sqrt(np.diag(cov))
    
    f_max = np.max(mu_sample)
    
    with np.errstate(divide='warn'):
        Z = (mu - f_max - xi) / std
        ei = std * (Z*norm.cdf(Z) + norm.pdf(Z))
        ei[std == 0.0] = 0.0
    return np.ravel(ei)



class BayesianOptimization:
    
    def __init__(self, opt_fun, bounds, kernel, acquisition, 
                 n_random_samples=None, X_train=None, X_pre_calc=None, Y_pre_calc=None):
        """
        Find optima within bounds using bayesian optimization

        Parameters
        ----------
        opt_fun : function call
            function to be optimized.
        bounds : list of tuples
            lower and upper limit for each variable.
        kernel : kernel call
            initilized kernel class.
        acquisition : function call
            function used for determining the next point.
        n_random_samples : int, optional
            number of randomly generated samples within the bounds
        X_train : numpy array, optional
            sample points to estimate
        X_pre_calc : numpy array, optional
            sample points already estimated
        Y_pre_calc : numpy array, optional
            function values for an already calculated sample
        """
        
        self.opt_fun = opt_fun
        self.bounds = bounds
        self.n_vals = len(bounds)
        
        self.kernel = kernel
        self.acquisition = acquisition
        
        self.construct_sample(n_random_samples, X_train, X_pre_calc, Y_pre_calc)
        self.gpr = GaussianProcess(self.X_sample, self.Y_sample, kernel)


    def opt_fun_iter(self, x):
        return np.array([self.opt_fun(x[i,:]) for i in range(x.shape[0])])
    
    
    def construct_sample(self, n_random_samples, X_train, X_pre_calc, Y_pre_calc):
        """
        Function for combining all sample values and creating random input 
        values and their corresponding outputs.
        """
        if isinstance(X_train, np.ndarray):
            Y_train = self.opt_fun_iter(X_train)
        elif X_train == None:
            X_train = np.empty((0, self.n_vals))
            Y_train = np.empty(0)
        else:
            ValueError("X_train needs to be a numpy array.")
            
        if type(n_random_samples) in [int, float]:
            X_rand = np.random.random((n_random_samples, self.n_vals))
            
            for i, val in enumerate(self.bounds):
                X_rand[:,i] *= val[1] - val[0]
                X_rand[:,i] += val[0]
            # implement checking for duplicates

            Y_rand = self.opt_fun_iter(X_rand)
        else:
            X_rand = np.empty((0, self.n_vals))
            Y_rand = np.empty(0)
            
        if not isinstance(X_pre_calc, np.ndarray):
            X_pre_calc = np.empty((0, self.n_vals))
            Y_pre_calc = np.empty(0)
        else:
            Y_pre_calc = Y_pre_calc.reshape(-1)
            
        self.X_sample = np.concatenate((X_pre_calc, X_train, X_rand), axis=0)
        self.Y_sample = np.concatenate((Y_pre_calc, Y_train, Y_rand), axis=0)

    
    def next_location(self, n_restarts):
        """
        Determine the next location for evaluation by minimizing the negative
        acquisition function

        Parameters
        ----------
        n_restarts : int
            number of times the minimization algorithm should be run with 
            random initial values.

        Returns
        -------
        numpy array
            with next values
        """
        dim = self.X_sample.shape[1]
        min_val = 1
        min_x = None
        
        def min_obj(X):
            return -self.acquisition(X, self.X_sample, self.Y_sample, self.gpr)
        
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(np.asarray(self.bounds)[:, 0], np.asarray(self.bounds)[:, 1], size=(n_restarts, dim)):
            try:
                res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
                val = res.fun[0]
            except ValueError:
                val = np.inf
                
            if val < min_val:
                min_val = val
                min_x = res.x           
                
        return min_x#.reshape(1, -1)
        
    def search(self, n_iter, re_opt_gpr, n_restarts, print_progress=True):
        """
        Search for the best function value

        Parameters
        ----------
        n_iter : int
            how many new function evaluations are made
        re_opt_gpr : int
            after how many trials should the kernel hyper parameters be optimized
        n_restarts : int
            number of times the minimization algorithm should be run with 
            random initial values when determining the next location.
        print_progress : bool, optional
            True if the current iteration values should be printed

        Returns
        -------
        ind : int
            index of the best location.
        X_opt : numpy array
            best location.
        Y_opt : float
            best function vlaue.

        """

        for i in range(n_iter):
            
            if i % re_opt_gpr:
                self.gpr.kernel.hyper = np.exp(self.gpr.opt_kernel_hyper())
                
            X_new = self.next_location(n_restarts=n_restarts)
            
            Y_new = self.opt_fun(X_new).reshape(1,)

            self.X_sample = np.concatenate((self.X_sample, X_new.reshape(1,-1)), axis=0)
            self.Y_sample = np.concatenate((self.Y_sample, Y_new), axis=0)
            
            self.gpr.update_train_data(self.X_sample, self.Y_sample)
            
            if print_progress:
                print(f"Iteration {i}: Current function value {Y_new[0]} and max value of {np.max(self.Y_sample)}")

        ind = self.Y_sample.argmax()
        Y_opt = self.Y_sample[ind]
        X_opt = self.X_sample[ind,:]
        return ind, X_opt, Y_opt
    