"""
                Module for the Lightgbm analysis run with MPI


This module tests different grid types for the Lightgbm algorithm on
batches of the KDD12 dataset. The grid types tested are: regular grid, random,
gridrandom, quasiperiodic, random sequential addition, latin hypercube 
sampling, and hyperuniform. The module is run with MPI4py build against OpenMPI.

"""
import numpy as np

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from lightgbm import early_stopping

# make it possible to pipe in arguments from command line
import sys
import os

# Add the directory containing the modules to Python's search path
sys.path.append('../Grid_Generation')

import Make_grids as mg


import argparse


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--res', type=int, default=3,
                    help='resolution of grid', )
parser.add_argument('--dim', type=int, default=2,
                    help='dimension of grid', )
parser.add_argument('--start_batch', type=int, default=0,
                    help='start batch', )

resolution = np.array([parser.parse_args().res]*parser.parse_args().dim)

edges = np.array([[0.001,3,5,0.5,0.5,0.0001,0.0001,1], [1,20,50,0.99,1,10,10,10]])

edges = edges[:,:parser.parse_args().dim]

from mpi4py import MPI
import socket

if not MPI.Is_initialized():
    MPI.Init()



n_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#get complete number of processes
size = comm.Get_size()

if size != 9:
    if rank == 0:
        print("This script requires exactly 9 ranks, got {}".format(size))
    comm.Abort(1)

if rank == 8:
    print(MPI.Get_library_version())

if rank != 8:
    start_batch = parser.parse_args().start_batch
    if start_batch == -1:
        start_batch = 0
    data_batch = rank + start_batch * 8   # which CSV this worker reads

comm = MPI.COMM_WORLD
print("Rank: ", comm.Get_rank(),'out of', size, "on Host: ", socket.gethostname())

import time
start = time.time()


if rank == 8:
    print('res', resolution)
    print('edges', edges)
    print('batch', parser.parse_args().start_batch)


edges = edges[:,:parser.parse_args().dim]



#NOTE change this for own path
data_path = '/scratch/project_465003303/to_lumi_hyper_par/Lightgbm/Data/Batches/'
if not os.path.exists(data_path):
    sys.exit('Load path does not exist')


# define a function that takes a grid and returns the median minimum loss for each batch
def median_min_loss(grid):
    import pandas as pd

    results_save = []
    if rank == 8:
        print('master setup for reciving')
        # setup structure for saving results
        results_par_save = []
        # setup structure for receiving results from all workers

        for i in range(0,size):
            if i == 8:
                continue
            results_par_save.append(comm.recv(source=i, tag=11))

        print('master recived all packages')
        results_par_save = np.array(results_par_save, dtype=object)


        return results_par_save

    else:
        #OBS: This is changed to fix a numbering error
        q = data_batch
        results_par_save_worker = []
        print('worker {} STARTING, about to read batch {}'.format(rank, q), flush=True)
        try:
            data = pd.read_csv(data_path + 'kdd12_batch_{}.csv'.format(q))
            print('worker {} loaded data from batch {}'.format(rank, q), flush=True)
            X_train = data.drop('2', axis=1)
            y_train = data['2']
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=1999)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
            for point in grid:
                results_par_save_worker.append(
                    run_iteration(point, X_train, X_val, X_test, y_train, y_val, y_test))
        except Exception as e:
            print('worker {} FAILED on batch {}: {!r}'.format(rank, q, e), flush=True)
            results_par_save_worker = []   # send something so the master doesn't hang
        comm.send(results_par_save_worker, dest=8, tag=11)
        return None, None



def run_iteration(i, X_train, X_val, X_test, y_train, y_val, y_test):
    """ 
    Function that runs a single iteration of the Lightgbm algorithm for a given
    point in a grid. It is hard coded what hyperparameters are used for
    what dimension of the grid. The function returns the point in the grid and
    the mean squared error of the model.

    Parameters
    ----------
    i : array
        The point in the grid to be tested. Must be of len 2, 3 or 4.

    X_train : array
        The training data.

    X_val : array
        The validation data.

    X_test : array
        The test data.

    y_train : array
        The training labels.

    y_val : array
        The validation labels.

    y_test : array
        The test labels.

    Returns
    -------
    batch_out : array
        The point in the grid and the mean squared error of the model.
    """

    Estimators = 750
    grid_size_check = len(i)
    Objective = 'l2_root'
    if grid_size_check == 2:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, objective=Objective, verbose = -1,
        n_jobs=n_threads)
        LGBM.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        batch_nr = [*i, mse]
        return batch_nr
    elif grid_size_check == 3:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, num_leaves=int(i[2]), objective=Objective, verbose = -1,n_jobs=n_threads)
        LGBM.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        batch_nr = [*i, mse]
        return batch_nr
    elif grid_size_check == 4:
        if i[3] >= 1:
            print('bagging fraction is larger than 1, number 4 in grid:', i)
            i[3] = 0.99
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, num_leaves=int(i[2]), bagging_fraction=i[3], objective=Objective, verbose = -1, n_jobs=n_threads)
        LGBM.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        i[1] = int(i[1])
        i[2] = int(i[2])
        batch_nr = [*i, mse]
        return batch_nr
    elif grid_size_check == 5:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, num_leaves=int(i[2]), bagging_fraction=i[3], feature_fraction=i[4], objective=Objective, verbose = -1, n_jobs=n_threads)
        LGBM.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        i[1] = int(i[1])
        i[2] = int(i[2])
        batch_nr = [*i, mse]
        return batch_nr
    elif grid_size_check == 6:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, num_leaves=int(i[2]), bagging_fraction=i[3], feature_fraction=i[4], lambda_l1=i[5], objective=Objective, verbose = -1, n_jobs=n_threads)
        LGBM.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        i[1] = int(i[1])
        i[2] = int(i[2])
        batch_nr = [*i, mse]
        return batch_nr
    elif grid_size_check == 7:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, num_leaves=int(i[2]), bagging_fraction=i[3], feature_fraction=i[4], lambda_l1=i[5], lambda_l2=i[6], objective=Objective, verbose = -1, n_jobs=n_threads)
        LGBM.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        i[1] = int(i[1])
        i[2] = int(i[2])
        batch_nr = [*i, mse]
        return batch_nr
    elif grid_size_check == 8:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, num_leaves=int(i[2]), bagging_fraction=i[3], feature_fraction=i[4], lambda_l1=i[5], lambda_l2=i[6], min_data_in_leaf=int(i[7]), objective=Objective, verbose = -1, n_jobs=n_threads)
        LGBM.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        i[1] = int(i[1])
        i[2] = int(i[2])
        i[7] = int(i[7])
        batch_nr = [*i, mse]
        return batch_nr





def totalt_analysis(resolution, edges, t, k, save_path = 'Data/GeneratedData/LGBM/'):
    """
    Function that runs analysis on all grid types and saves the results to a file.
    
    Parameters
    ----------
    resolution : array
        The resolution of the grid. Here given by the pipeline.

    edges : array
        The domain of the grid. Here given by the pipeline.

    t : int
        The number of batches to be run. Here given by the pipeline.

    k : int
        The number of workers to be used. Here given by the pipeline.

    save_path : str, optional
        The path to save the results. The default is 'data/GeneratedData/LGBM/'.
        if the path does not exist the function will exit.

    Returns
    -------
    True : bool
        If the function runs successfully. Only returned by the master node.

    None : None
        If the function runs successfully. Only returned by the worker nodes.
    """
    #check if save path exists
    if not os.path.exists(save_path):
        sys.exit('Save path does not exist')

    d = len(resolution)

    # put each (dim, resolution) run in its own subfolder
    save_path = save_path + 'd_{}_res_{}/'.format(d, resolution[0])
    if rank == 8:
        os.makedirs(save_path, exist_ok=True)

    # RSA
    if rank == 8:
       print(f'{rank} started RSA')
    values_rsa = median_min_loss(mg.rsa_grid(resolution, edges))
    
    if rank == 8:
       pickle.dump(values_rsa, open(save_path + 'rsa_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))

    # regular grid
    if rank == 8:
        print(f'{rank} started regular grid')
    values_reg = median_min_loss(mg.reg_grid(resolution, edges))
    if rank == 8:
        pickle.dump(values_reg, open(save_path + 'reg_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))


    #random uniform grid
    if rank == 8:
        print(f'{rank} started random uniform grid')
    values_rand = median_min_loss(mg.random_grid(resolution, edges))

    if rank == 8:
        pickle.dump(values_rand, open(save_path + 'ran_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))


    #hypertilling
    if rank == 8:
        print(f'{rank} started hypertilling')
    values_hyper = median_min_loss(mg.hyper_tilling_grid(resolution, edges, t, k))

    if rank == 8:
       pickle.dump(values_hyper, open(save_path + 'hyp_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))


    #latin hypercube sampling
    if rank == 8:
        print(f'{rank} started LHS')
    values_lhs = median_min_loss(mg.lhs_grid(resolution, edges))

    if rank == 8:
       pickle.dump(values_lhs, open(save_path + 'lhs_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))


    # Grid random
    if rank == 8:
        print(f'{rank} started grid random')
    values_grid_rand = median_min_loss(mg.make_gridrandom(resolution, edges))
    if rank == 8:
        pickle.dump(values_grid_rand, open(save_path + 'grid_rand_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))

    # Sobol
    if rank == 8:
        print(f'{rank} started Sobol')
    values_sobol = median_min_loss(mg.sobol_grid(resolution, edges))
    if rank == 8:
        pickle.dump(values_sobol, open(save_path + 'sobol_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))

    # Halton
    if rank == 8:
        print(f'{rank} started Halton')
    values_halton = median_min_loss(mg.halton_grid(resolution, edges))
    if rank == 8:
        pickle.dump(values_halton, open(save_path + 'halton_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))


    # true hyper tilling
    if rank == 8:
        print(f'{rank} started true hyper tilling')
    values_true_hyper = median_min_loss(mg.make_true_hyper(resolution, edges))
    if rank == 8:
        pickle.dump(values_true_hyper, open(save_path + 'true_hyper_d_{}_res_{}_batch{}.p'.format(d, resolution[0], parser.parse_args().start_batch), 'wb'))



    if rank == 8:

        return True

    else:
        return None


save_path_lumi = '/scratch/project_465003303/to_lumi_hyper_par/Lightgbm/Data/GeneratedData/LGBM/'

if rank == 8:
    tot_out = totalt_analysis(resolution, edges, 6, 1, save_path = save_path_lumi)
    end = time.time()

    print('-'*80)
    print('end', end)
    print(f"Elapsed time: {end - start} seconds")

else:
    totalt_analysis(resolution, edges, 6, 1, save_path = save_path_lumi)
