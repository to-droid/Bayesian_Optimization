import numpy as np

import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def to_2d_ndarray(x):
    """
    Converts an array to a numpy array and verifies its two dimensionality

    Parameters
    ----------
    x : array
    
    Returns
    -------
    x : 2D numpy array
    """
    
    x = np.array(x)
    
    if x.ndim > 2:
        x = np.ravel(x)
    
    if x.ndim == 1:
        x = x[:,None]
        
    if x.ndim != 2:
        raise ValueError("x has to be a 2d numpy array.")
    
    return x
    
    
def validate_X_Y(X, Y):
    """
    Validate the type and shape of X and Y

    Parameters
    ----------
    X and Y : array

    Returns
    -------
    X and Y : 2D numpy array with matching length
    """
    
    X, Y = tuple(map(to_2d_ndarray, (X, Y)))
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError("'X' and 'Y' need to be of the same length.")
    
        
    return X, Y


def get_nan_inds(series):
    ''' Obtain the first and last index of each consecutive NaN group.
    '''
    series = series.reset_index(drop=True)
    index = series[series.isna()].index.to_numpy()
    if len(index) == 0:
        return []
    indices = np.split(index, np.where(np.diff(index) > 1)[0] + 1)
    return [(ind[0], ind[-1] + 1) for ind in indices]

def lag_matrix(x, lags, include_t0=True):
    ''' Create a lagged version of x.
    x (1D or 2D array): input data with (time x variables)
    lags (list or int): number of consecutive lags or list of lags
    include_t0 (bool): include present observation in lag matrix
    '''
    
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    elif len(x.shape) != 2:
        ValueError("x has to be 1D or 2D.")
    
    if isinstance(lags, (int, float)):
        lags = list(range(1, lags + 1))
    elif isinstance(x, (list, tuple, np.ndarray)):
        lags = list(lags)
    else:
        ValueError("lags has to be an int, float, list, tuple or ndarray.")
        
    if include_t0 and 0 not in lags:
        lags = [0] + lags
        
    T, N = x.shape
    max_lag = max(lags)
    
    x_lagged = np.zeros((T - max_lag, len(lags)*N))
    
    for i, l in enumerate(lags):
        x_lagged[:,i*N:(i + 1)*N] = x[max_lag-l:T-l,:]
        
    return x_lagged

def OLS(X, y):

    if len(X.shape) == 1:
        X = X.reshape(-1,1)

    X_t = X.T
    XX_inv = np.linalg.inv(X_t @ X)
    return XX_inv @ X_t @ y

# =============================================================================
# Plots
# =============================================================================

def plot_progress(x, optimum=None, metric_name=None, ax=None):
    ''' Plot the progress of Bayesian Optimization
    '''
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.maximum.accumulate(x), color='k', label=metric_name)
    if isinstance(optimum, (int, float)):
        ax.hlines(optimum, xmin=0, xmax=len(x)-1, color='grey', ls='--', label='Optimum')
    ax.set_xlim((0, len(x)-1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Iterations')
    ax.legend(loc='best')
    return ax

def plot_freq(x, ax = None):
    ''' Plot the frequencies of hourly data x using Fourier transform
    '''
    days_per_year = 365.2425 # Gregorian

    n_days = len(x) / 24
    n_weeks = n_days / 7
    n_months = n_days / (days_per_year / 12)
    n_years = n_days /  days_per_year

    fft = np.fft.rfft(x)
    freq_ind = np.arange(1, len(fft))

    if ax is None:
        ax = plt.gca()
    ax.step(freq_ind, np.abs(fft[1:]))
    ax.set_xscale('log')
    ax.set_ylim(0, None)
    ax.set_xlim([1, None])
    ax.set_xticks([n_years, n_months, n_weeks, n_days])
    ax.set_xticklabels(['Yearly', 'Monthly', 'Weekly', 'Daily'])
    ax.set_xlabel('Frequency (log scale)')


# =============================================================================
# Model Wrapper
# =============================================================================

def train_val_gen(data, prediction_range, target_inds, n_val):
    ''' Generate train and validation data for standing forecast i.e. X is 
    fixed and y moves one step inside range.    
    '''
    t_max = max(prediction_range)
    T = data.shape[0]
    
    if n_val <= t_max:
        raise ValueError("n_val has to be larger than the maximum prediction range.")
    if T - n_val <= t_max:
        raise ValueError("n_val or maximum prediction range is to large.")
        
    for i in target_inds:
        for t in range(prediction_range[0], 
                       prediction_range[1] + 1):
            
            lgb_train = lgb.Dataset(data[:-(n_val+t_max),:], 
                                    data[t:-(n_val + t_max - t),i])
    
            lgb_eval = lgb.Dataset(data[-n_val:-t_max,:], 
                                    data[-(n_val-t):(T-t_max+t),i], 
                                    reference=lgb_train)

            yield i, t, lgb_train, lgb_eval



class Model:
    
    def __init__(self, data, lags, n_oos, n_val, prediction_range, 
                 target_inds, params={}):
        '''
        Wrapper class for optimizing lgb model for standing forecast in 
        time series regression. Fit number of targets times rediction range 
        models.

        Parameters
        ----------
        data : ndarray
            T by N array with T time observations and N features
        lags : list or int
            number of lags or a list of lags
        n_oos : int
            start of the prediction. start counting at the end i.e. T-n_oos
        n_val : int
            number of observations used for validation
        prediction_range : tuple of ints
            start and end of range
        target_inds : list or  int
            list of column indices in given data for the final fit meassure
        params : dict, optional
            additional parameters passed in lgb.train()
        '''

        n_oos += max(lags)
        
        self.T, self.N = data.shape
        self.X_oos = data[-n_oos:,:]
        
        self.lags = lags
        self.n_val = n_val
        self.prediction_range = prediction_range
        self.params = params
        self.target_inds = target_inds
        
        if 0 not in lags:
            lags = [lag - 1 for lag in lags]
        self.data = lag_matrix(data[:-n_oos,:], lags)
        self.best_score = -np.inf
        
    def fit(self, params):
        ''' Fit lgb model with given parameters.
        '''
        if isinstance(params, dict):
            params = list(params.values())
        elif isinstance(params, (list, tuple, np.ndarray)):
            params = list(params)
        else:
            TypeError('Unsupported type for params. Use dict, list, or tuple.')

        lr = params[0]
        lr_change = params[1]
        
        self.params['num_leaves'] = round(params[2])
        self.model_dict = {}
        
        for i, t, lgb_train, lgb_eval in train_val_gen(self.data, 
                                                       self.prediction_range, 
                                                       self.target_inds, 
                                                       self.n_val):
            # print(f"Fitting model for variable {i} at horizon {t}.")
            gbm = lgb.train(self.params, train_set = lgb_train, 
                        valid_sets = [lgb_eval], verbose_eval =False,
                        learning_rates=lambda iter: lr * (lr_change ** iter))
            
            self.model_dict[(i,t)] = gbm
            
    def standing_forecast(self):
        ''' Forecast y in given range using only in-sample data (not rolling).
        '''
        X_pred = np.zeros((int(np.diff(self.prediction_range) + 1),
                           len(self.target_inds)))
        
        t_min = min(self.prediction_range)
        i_min = min(self.target_inds)
        max_lags = max(self.lags)
        
        lag_inds = [max_lags - lag for lag in self.lags]
        data = self.X_oos[lag_inds,:].reshape(1,-1)
        
        for (i, t), mdl in self.model_dict.items():
            X_pred[t-t_min, i-i_min] = mdl.predict(data)
        
        return X_pred

    def obj_fun(self, params):
        ''' Objective function for Bayesian optimization
        '''
        self.fit(params)
        pred = self.standing_forecast()
        
        max_lag = max(self.lags) - 1
        start = max_lag + self.prediction_range[0]
        end = max_lag + self.prediction_range[1] + 1
        
        target = self.X_oos[start:end, self.target_inds]
        
        score = (target - pred)**2
        score = np.sqrt(np.average(score, axis=0))
        score = -np.average(score)
        if score > self.best_score:
            self.best_score = score
            self.best_model = self.model_dict
        return score