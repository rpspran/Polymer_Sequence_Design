#FILE DESCRIPTION

#1 - unconstrained 1 and 2
#1 - Minimization
#1 - GPR Probability of Improvement
#0 - n_sm only


import bo_functions

import numpy as np
import numpy
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import os
import random
import subprocess
import shutil
import time

import sklearn
import modAL
from modAL.models import ActiveLearner
from modAL.models import BayesianOptimizer
from modAL.acquisition import optimizer_PI, max_PI


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.gaussian_process.kernels import Matern

import itertools
from itertools import combinations

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

import scipy
from scipy.stats import norm

import skopt
from skopt import BayesSearchCV

# Global settings
import warnings
warnings.filterwarnings("ignore") # To ignore warnings
n_jobs = -1 # This parameter conrols the parallel processing. -1 means using all processors.
random_state = 42 # This parameter controls the randomness of the data. Using some int value to get same results everytime this code is run.


def GP_regression_PI(optimizer, X, n_instances):
    
    pi = optimizer_PI(optimizer, X)
    pi = pi.ravel()
    
    neg_n_instances = np.negative(n_instances)
    query_indices = pi.argsort()[neg_n_instances :][::-1]
    
    best_query_indices = query_indices.ravel()
    best_pi_values = np.array(pi[best_query_indices.tolist()])

    return (best_query_indices, best_pi_values)




def GP_regression_exploit_explore(optimizer, X, n_instances):
    y_pred_mean, y_pred_sdev = optimizer.predict(X, return_std=True)
    neg_n_instances = np.negative(n_instances)
    
    y_pred_mean = y_pred_mean.ravel()
    y_pred_sdev = y_pred_sdev.ravel()
        
    indices_pure_exploit = y_pred_mean.argsort()[neg_n_instances :][::-1]
    indices_pure_exploit = indices_pure_exploit.ravel()
    
    indices_pure_explore = y_pred_sdev.argsort()[neg_n_instances :][::-1]
    indices_pure_explore = indices_pure_explore.ravel()

    return (indices_pure_exploit, indices_pure_explore)



#Step 1: INITIALIZING HYPER-PARAMETERS

file_var = "N100_sample_1110"
num_weights = 100
n_initial = 112

#Sampling
n_sm = 32
n_dm = 0
n_sixm = 0

#Active learning loop
n_queries_max = 75
n_iterations = 75
iter_count_max = 0


#STEP 2 Defining Initial training set     
#EXTRACTION OF SEQUENCES, FITNESS AND ONES_TRAINING

seq_training = []
fitness_training= []
ones_training = []

for i in range(0,n_initial):
    
    rad_file = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/GPR_training/sequence"+str(i)+".txt", "r")
    list_of_content = rad_file.readlines()
    array_of_content = np.array(list_of_content)
    
    seq_array = np.array([np.int(elem[0]) for elem in array_of_content[:-1]])
    seq_training.append(seq_array)
    
    ones_each_sequence = (seq_array==1).sum()
    ones_training.append(ones_each_sequence)
    
    try:
        fitness_training.append(float(list_of_content[-1][:-1]))

    except:
        fitness_training.append(float(list_of_content[-2][:-1]))
    rad_file.close()


#Storing the no of ones in the training set into a separate file
file_ones_training = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/no_ones_training.txt","w")
file_ones_training.write(str(ones_training))
file_ones_training.close()


X_initial = np.array(seq_training)
AvgRg_training = np.array(fitness_training)
y_initial = AvgRg_training.reshape(-1,1)
y_initial_pos = y_initial
y_initial_neg = np.negative(y_initial)



#STEP 3 Initialize the Optimizer for minimization & getting ready before the for loop

ts_before = time.time()

X_initial_sofar_max = X_initial
y_initial_sofar_max = y_initial_neg

# initializing the optimizer
optimizer = BayesianOptimizer(
    estimator=GaussianProcessRegressor(kernel=Matern(length_scale=1.0)),
    X_training=X_initial_sofar_max, y_training=y_initial_sofar_max,
    query_strategy=max_PI)


ts_after = time.time()

#Storing the time taken for initializing the Bayesian Optimizer, into a separate file

file_oit = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/optimizer_init_time.txt","w")
file_oit.write(str(ts_after - ts_before))
file_oit.close()


#Step 4- running for loop for Minimization
for idx in range(n_queries_max):

    file_GPR_time = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/loop_time.txt","a")
    file_ypredmean_time = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/y_pred_mean.txt","a")
    file_ypredsdev_time = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/y_pred_sdev.txt","a")
    file_candsize_time = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/y_cand_size.txt","a")
    file_AvgRg = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/AvgRg.txt","a")
    file_no_ones = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/no_ones.txt","a")

    file_maxPI = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/maxPI_each_iteration.txt","a")
    file_exploit_maxPI_overlap = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/exploit_maxPI_overlap.txt","a")
    file_explore_maxPI_overlap = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/explore_maxPI_overlap.txt","a")


    
    
    loop_start = time.time()

    #5a) retrieving the expanded search space
    cand_sm_append = bo_functions.calc_cand_sm(X_initial_sofar_max , num_weights)
    
    t1 = time.time()
    
    #5b) retrieving the query    
    query_idx_sm, best_pi_values = GP_regression_PI(optimizer, cand_sm_append, n_instances = n_sm)        
    y_pred_mean, y_pred_sdev = optimizer.predict(cand_sm_append[query_idx_sm], return_std=True)

    indices_pure_exploit, indices_pure_explore  = GP_regression_exploit_explore(optimizer, cand_sm_append, n_instances = n_sm)
    exploit_maxPI_overlap = len(np.intersect1d(query_idx_sm, indices_pure_exploit, assume_unique = True))
    explore_maxPI_overlap = len(np.intersect1d(query_idx_sm, indices_pure_explore, assume_unique = True))


    t2 = time.time()

    #5c) Printing the iteration number
    print('\n')
    print('iter_count = ', iter_count_max)

    #5d) Printing the size of cand_sm_append
    
    print('No. of candidates considered is ', cand_sm_append.shape[0])
    #print('Queried indices: ', query_idx_sm)

    #5e) defining X_sm and computing y_sm
    X_sm = cand_sm_append[query_idx_sm]
    AvgRg_sm = bo_functions.y_query_max_sm(X_sm, iter_count_max, num_weights)
    y_sm = np.negative(AvgRg_sm)

    t3 = time.time()

    #5f) re-training the optimizer with an additional (_, _) pair
    optimizer.teach(X_sm, y_sm.reshape(-1,1))
    
    
    #5g) Stacking the newly obtained X and y into the training set.
    X_initial_sofar_max = np.vstack([X_initial_sofar_max, X_sm])
    y_initial_sofar_max = np.vstack([y_initial_sofar_max, y_sm.reshape(-1,1)])

    t4 = time.time()

    #5h) writing in a file
    list_to_write = [iter_count_max, (t1-loop_start), (t2-t1), (t3-t2), (t4-t3), (t4 - loop_start)]    
    file_GPR_time.write(str(list_to_write))
    file_GPR_time.write("\n")
    
    cand_size_add = cand_sm_append.shape[0]
    file_candsize_time.write(str(cand_size_add))
    file_candsize_time.write("\n")
    
    y_pred_mean_add = list(np.squeeze(y_pred_mean)) 
    file_ypredmean_time.write(str(y_pred_mean_add))
    file_ypredmean_time.write("\n")
    
    y_pred_sdev_add = list(np.squeeze(y_pred_sdev))
    file_ypredsdev_time.write(str(y_pred_sdev_add))
    file_ypredsdev_time.write("\n")
    
    AvgRg_add = list(np.squeeze(AvgRg_sm))
    file_AvgRg.write(str(AvgRg_add))
    file_AvgRg.write("\n")

    no_ones_each = np.sum(X_sm==1, axis = 1)
    no_ones_add = list(no_ones_each)
    file_no_ones.write(str(no_ones_add))
    file_no_ones.write("\n")

    
    
    maxPI_add = list(np.squeeze(best_pi_values))
    file_maxPI.write(str(maxPI_add))
    file_maxPI.write("\n")

    exploit_maxPI_add = exploit_maxPI_overlap
    file_exploit_maxPI_overlap.write(str(exploit_maxPI_add))
    file_exploit_maxPI_overlap.write("\n")
    
    explore_maxPI_add = explore_maxPI_overlap
    file_explore_maxPI_overlap.write(str(explore_maxPI_add))
    file_explore_maxPI_overlap.write("\n")


    
    file_GPR_time.close()    
    file_ypredmean_time.close()
    file_ypredsdev_time.close()
    file_candsize_time.close()
    file_AvgRg.close()
    file_no_ones.close()
    
    
    file_maxPI.close()
    file_exploit_maxPI_overlap.close()
    file_explore_maxPI_overlap.close()




    
    #5i) incrementing the counter by 1
    iter_count_max = iter_count_max + 1

    
    
