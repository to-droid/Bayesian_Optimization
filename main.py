import os
import contextlib

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from pathlib import Path
from utils import  get_nan_inds, OLS, Model
from bayesian_optimization import expected_improvement, BayesianOptimization
from kernels import Kernel

from ax import RangeParameter, ParameterType, SearchSpace, SimpleExperiment
from ax.modelbridge.registry import Models

project_dir = Path(__file__).resolve().parents[0]
read_dir = project_dir / 'data' / 'input_data'
save_dir = project_dir / 'data' / 'output_data'


@contextlib.contextmanager
def working_directory(path):
    """
    Changes working directory and returns to previous on exit.
    Use together with context manager e.g. with():
    
    Parameters
    ----------
    path: str or path object
        target directory
    """
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        
def read_data(path, var_name):
    
    colnames = ['id', 'date'] + [var_name]
        
    return pd.read_csv(path, usecols=[1,2,3], index_col=[0, 1], names=colnames,
                       parse_dates=[1], date_parser=date_parser,
                       squeeze=True, header=0)




# =============================================================================
# READ IN DATA
# =============================================================================

date_parser = lambda x: pd.to_datetime(x, format='%Y%m%d%H%M')

file_names = ['data_TT_TU_MN009.csv', 'data_FF_MN008.csv', 'data_R1_MN008.csv', 
              'data_P0_MN008.csv', 'data_D_MN003.csv']
var_names = ['temp', 'wind', 'rain', 'pres', 'coor']

with working_directory(read_dir):
    df = pd.concat([read_data(file_name, var_name) for file_name, var_name in 
                     zip(file_names, var_names)], axis=1)

date_range = pd.date_range(start = '1/1/2010 00:00:00', 
                           end = '19/01/2021 23:00:00', 
                           freq = 'H')

station_ids = df.index.unique(0)

ind = pd.MultiIndex.from_product([station_ids, date_range], 
                                 names = ['id', 'date'])

df = pd.DataFrame(index=ind).merge(df, how='left', 
                                   left_index=True, right_index=True)

rad = df.pop('coor') * np.pi / 180

df['wind_x'] = df['wind'] * np.cos(rad)
df['wind_y'] = df.pop('wind') * np.sin(rad)

del rad


# =============================================================================
# INTERPOLATION AND SEASONALITY
# =============================================================================

def ols_interpolation(df):
    ''' Perform interpolation by using OLS of the same variable from other 
    stations at the same date time range. This function is applied using 
    pandas groupby function.
    '''
    df_nan = df.droplevel(axis=0, level=1).T
    df_lin = df_nan.interpolate(axis=0)
    for station_id in df_nan.columns[df_nan.isna().any()]:
        
        X = df_lin.drop(station_id, axis=1).to_numpy()
        y = df_lin.loc[:,station_id].to_numpy()
        
        beta = OLS(X,y)
        
        nan_inds = get_nan_inds(df_nan.loc[:,station_id])
    
        for min_ind, max_ind in nan_inds:
            dates = date_range[min_ind:max_ind]
            df.loc[(station_id, df.name), dates] = X[min_ind:max_ind,:] @ beta
    return df

df = df.stack().unstack(1)
df.index.names = ['id', 'variable']
df = df.groupby(by='variable', axis=0, level=1).apply(ols_interpolation)


hours_per_day = 24
hours_per_year = 365.2425*24
ind = 2 * np.pi * np.arange(len(date_range))

df_seas = pd.DataFrame({('season', 'day_sin') : np.sin(ind / hours_per_day),
                        ('season', 'day_cos') : np.cos(ind / hours_per_day),
                        ('season', 'year_sin') : np.sin(ind / hours_per_year),
                        ('season', 'year_cos') : np.cos(ind / hours_per_year)}, 
                       index = date_range)
df = pd.concat([df, df_seas.T], axis=0)


df_target = df.loc[:,'19/01/2021 00:00:00':]
df_train = df.loc[:,:'19/01/2021 00:00:00']
data_mat = df_train.T.to_numpy()


# =============================================================================
# Modeling
# =============================================================================


# define parameters for model
n_oos = 24*14 # two weeks
n_val = 24*30*6 # six month
lags = [1,2,3,4,5,6] # [1,2,3,4,5,24]
prediction_range = (1, 6)
target_vars_inds = [0, 1] # temperature and rain for Frankfurt

params = {'num_threads': os.cpu_count() - 2,
          'early_stopping_round': 10,
          'num_boost_round' : 20,
          'metric' : 'rmse'} 

bounds = [(1.0e-5, 1.0e-1), # learning rate
          (0.5, 0.9999), # change of learning rate
          (2, 1000)] # number of leaves

n_random_trials = 3 # initiate Bayesian optimization with 3 random draws
n_searches = 10



# Use my Bayesian Optimization
mdl = Model(data_mat, lags, n_oos, n_val, prediction_range, 
            target_vars_inds, params)

kernel = Kernel("rbf", 1)

bo = BayesianOptimization(mdl.obj_fun, bounds, kernel, 
                          expected_improvement, n_random_trials)
ind, best_para_my, y = bo.search(n_searches, 2, 25)





# Use Ax Bayesian Optimization
n_random_trials = 5 # initiate Bayesian optimization with 3 random draws
n_searches = 20

mdl = Model(data_mat, lags, n_oos, n_val, prediction_range, 
            target_vars_inds, params)

search_space = SearchSpace(parameters=[
        RangeParameter(name="lr", lower=1.0e-5, upper=1.0e-1,     
                               parameter_type=ParameterType.FLOAT),
        RangeParameter(name="lr_change", lower=0.5, upper=1.0,    
                               parameter_type=ParameterType.FLOAT),    
        RangeParameter(name="leafes", lower=2, upper=1000,    
                               parameter_type=ParameterType.INT)]
    )


experiment = SimpleExperiment(
    name = f"weather_lbgm_{dt.datetime.today().strftime('%d-%m-%Y')}",
    search_space = search_space,
    evaluation_function = mdl.obj_fun,
)

sobol = Models.SOBOL(experiment.search_space)
for i in range(n_random_trials):
    experiment.new_trial(generator_run=sobol.gen(1))

best_arm = None
for i in range(n_searches):
    gpei = Models.GPEI(experiment=experiment, data=experiment.eval())
    generator_run = gpei.gen(1)
    best_arm, _ = generator_run.best_arm_predictions
    experiment.new_trial(generator_run=generator_run)

best_para_ax = best_arm.parameters


n_oos = 0
params['num_boost_round'] = 200
mdl = Model(data_mat, lags, n_oos, n_val, prediction_range, target_vars_inds, params)
mdl.fit(best_para_ax)
X_pred = mdl.standing_forecast()