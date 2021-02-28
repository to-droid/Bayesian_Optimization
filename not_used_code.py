from random import randrange
import scipy.optimize
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_acf(x, alpha=0.05, nlags=20, axes=None, corr_fun='acf'):
    
    if corr_fun == 'acf':
        auto_corr, confi = acf(x, nlags=nlags, alpha=alpha)
    elif corr_fun == 'pacf':
        auto_corr, confi = pacf(x, nlags=nlags, alpha=alpha)
    else:
        ValueError("corr_fun has to be either 'acf' or 'pacf'.")
        
    lower_bound = confi[:,0] - auto_corr
    upper_bound = confi[:,1] - auto_corr
    
    inds = list(range(len(auto_corr)))
    ymin = min(min(lower_bound), min(auto_corr)) * 1.2
    
    if axes is None:
        axes = plt.gca()
    
    axes.set_xlim([-0.5, len(auto_corr)])
    axes.set_ylim([ymin, max(auto_corr)])
    axes.axhline(color='black', linewidth=.8)
    
    axes.bar(inds, auto_corr, width=.5, color='black')
    axes.fill_between(inds, lower_bound, upper_bound, color='grey', alpha=.4)

ticks = range(24*2, 24*14, 24*2)
fig = plt.figure(figsize=(30,20))
spec = gridspec.GridSpec(nrows=3, ncols=2)
for i, var in enumerate(df.index.unique(1)):
    ax = fig.add_subplot(spec[i])
    ax.title.set_text(var)
    plot_acf(df.loc[(1420, var),:], nlags=24*14, axes=ax)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f'Day {int(i/24)}' for i in ticks])

hours_per_month = 24 * 30
ticks = range(hours_per_month, hours_per_month*12, hours_per_month*2)
fig = plt.figure(figsize=(30,20))
spec = gridspec.GridSpec(nrows=3, ncols=2)
for i, var in enumerate(df.index.unique(1)):
    ax = fig.add_subplot(spec[i])
    ax.title.set_text(var)
    plot_acf(df.loc[(1420, var),:], nlags=24*370, axes=ax)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f'Month {int(i/hours_per_month)}' for i in ticks])



def sinusoid(t, c, A, w, p):
    """ Generate sinusoid function given parameters and array.
    t (array): array of equally spaced times steps
    c (float): constant vertical shift
    A (float): amplitude (max deviation from zero)
    w (float): angular frequency (how often it oscilates per time)
    p (float): phase
    """
    return A * np.sin(w*t + p) + c

def fit_sinusoid(x):
    """ Fit a simple sinusoid to x and return fitted data and parameters
    x (array): array of periodic data
    """

    T = len(x)
    t = np.linspace(0, 1, T)
    
    f = np.fft.fftfreq(T, 1/T)
    fft_x = abs(np.fft.fft(x))
    
    init_c = np.mean(x)
    init_A = np.std(x)
    init_w = 2 * np.pi * abs( f[np.argmax(fft_x[1:]) + 1] ) 
    
    p0 = [init_c, init_A, init_w, 0]
    opt_para, cov = scipy.optimize.curve_fit(sinusoid, t, x, p0=p0)
    
    c, A, w, p = opt_para
    return sinusoid(t, c, A, w, p), {'const':c, 'amplitude':A, 'freq':w, 'phase':p}

def seasonal_sinusoid(x, N):
    """ Fit N consecutive sinusoids to x and return fitted sinusoid and parameters
    x (array): array of periodic data
    N (int): number of different periods (e.g. seasons and daily)
    """

    T = len(x)
    seas_sin = np.zeros(T)
    para_dict = {}
    
    for n in range(N):
        sin, para = fit_sinusoid(x - seas_sin)
        para_dict['season_' + str(n)] = para
        seas_sin += sin
    
    return seas_sin, para_dict

def read_sinusoid_dict(para_dict, time_range=np.linspace(0,1,1000)):
    ''' Fits a series of sinusoids given the result dict by seasonal_sinusoid
    '''
    x = np.zeros(len(time_range))
    for paras in para_dict.values():
        x += sinusoid(time_range, 
                      paras['const'],
                      paras['amplitude'],
                      paras['freq'],
                      paras['phase'])
    return x

def time_series_split(num, X, y, n_train, n_val):
    ''' Generator for time series cross validation.
    num (int): number of folds
    X (2D array): input data
    y (1D array): output data
    n_train (int): number of observations in train set
    n_val (int): number of observations in validation set
    '''
    n_obs = len(y)
    
    if len(X) != n_obs:
        ValueError("Length of X and y has to match.")
    
    if n_obs - (n_train + n_val) < 0:
        ValueError("n_train plus n_val has to be smaller than the sample size.")
        
    for _ in range(num):
        ind = randrange(n_obs - n_train - n_val)
        
        val_start = ind + n_train
        val_end = val_start + n_val
        
        yield X[ind:val_start,:], y[ind:val_start], X[val_start:val_end,:], y[val_start:val_end]
        
        
class Model:
    
    def __init__(self, data, lags, n_oos, n_val, n_steps, 
                 target_inds, params={}):
        '''
        Wrapper class for optimizing lgb model for rolling forecast in 
        time series regression.

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
        n_steps : int
            length of rolling forecast
        target_inds : list or  int
            list of column indices in given data for the final fit meassure
        params : dict, optional
            additional parameters passed in lgb.train()
        '''
        if n_oos < n_steps:
            Warning('''n_oos is to small for calculating the objective function
                    needed for BO. Please chose a term larger than n_steps.''')
        n_oos += max(lags)
        
        self.T, self.N = data.shape
        data = lag_matrix(data, lags)
        
        self.X_oos = data[-n_oos:,:self.N]
        self.X = data[:-n_oos,self.N:]
        self.y = data[:-n_oos,:self.N]
        
        self.lags = lags
        self.n_val = n_val
        self.n_steps = n_steps
        self.params = params
        self.target_inds = target_inds

    def fit(self, params):
        ''' Fit lgb model with given parameters.
        '''
        params = list(params)
        lr = params[0]
        lr_change = params[1]
        
        self.params['num_leaves'] = round(params[2])
        
        self.model_dict = {}
        
        for i in range(self.N):
            
            lgb_train = lgb.Dataset(self.X[:-self.n_val,:],
                                    self.y[:-self.n_val,i].reshape(-1))
            
            lgb_eval = lgb.Dataset(self.X[-self.n_val:,:],
                                   self.y[-self.n_val:,i].reshape(-1), 
                                   reference=lgb_train)
            
            evals_result = {}
            
            gbm = lgb.train(self.params, train_set = lgb_train, 
                            valid_sets = [lgb_eval],
                            evals_result = evals_result,
                            learning_rates=lambda iter: lr * (lr_change ** iter))
            
            self.model_dict[i] = gbm
            print(f"Iteration {i+1} of {self.N} finished.")
    
    def rolling_forecast(self):
        ''' n step ahead rolling forecast. Starting in T - n_oss
        '''
        max_lag = max(self.lags)
        n_lags = len(self.lags)
        X_pred = np.zeros((self.n_steps, self.N))
        
        inds_is = [max_lag - lag - 1 for lag in self.lags]
        inds_oos = []
        
        for t in range(self.n_steps):
            
            x = np.zeros((1, self.N*n_lags))
            
            inds_is = [ind + 1 for ind in inds_is if ind + 1 < max_lag]
            
            if len(inds_is) + len(inds_oos) < n_lags:
                inds_oos.append(-1)
        
            if len(inds_oos) > 0:
                inds_oos = [ind + 1 for ind in inds_oos]
                x[0,:len(inds_oos)*self.N] = X_pred[inds_oos,:].reshape(-1)
                
            if len(inds_is) > 0:
                x[0,-len(inds_is)*self.N:] = self.X_oos[inds_is,:].reshape(-1)
            
            for i in range(self.N):
                mdl_gbt = self.model_dict[i]
                X_pred[t, i] =  mdl_gbt.predict(x)
                
        return X_pred
    
    
    
    def obj_func(self, params):
        ''' Function to be optimized
        '''
        max_lag = max(self.lags)
        
        self.fit(params)
        pred = self.rolling_forecast()
        
        target = self.X_oos[max_lag:max_lag+self.n_steps, self.target_inds]
        
        score = (target - pred[:,self.target_inds])**2
        score = np.sqrt(np.average(score, axis=0))
        return -np.average(score)
    
    
dt = 1/df.shape[1]
t = np.arange(1 + dt, 1 + 24*dt, dt)

for labels, seas in seas_dict.items():
    ind = np.where(df.index == labels)[0]
    for p in seas.values():
        x_seas = sinusoid(t, p['const'], p['amplitude'], p['freq'], p['phase'])
        X_pred[:,ind] += x_seas.reshape(-1,1)