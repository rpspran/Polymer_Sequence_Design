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



file_var = "N100_sample_1110"



#STEP 3 Calculating the y_initial (fitness values)


def training_fitness(X_initial,n_initial,num_weights):
     # Calculating the fitness value of each solution in the current population.
     # The fitness function caulcuates the sum of products between each input and its corresponding weight.
     # and writes separate .txt files each with different solution sequences in a population 
        
     
     #STEP 3.1 opening sequence.txt files for all the datapoints in the training set
        
     for i in range(0,n_initial):
         file2 = open("seqb"+str(i)+".txt", "w")
            
         file1 = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/GPR_training/sequence"+str(i)+".txt","w")
         
        
         for j in range(0,num_weights):
             file2.write(str(X_initial[i,j]))
             file1.write(str(X_initial[i,j]))
             file1.write("\n")
             file2.write("\n")
         file2.close()
         file1.close()
     
     #STEP 3.2 Duplicating datatoData1.cpp files and modification
     #generate n_initial number of polymer 1 to 2 data conversion .cpp files
     for i in range(0,n_initial):
         shutil.copy('datatoData1.cpp', 'datatoData1_b'+str(i)+'.cpp')                                                                                                        

     #changing the content of .cpp files according to sequence files to be read
     for i in range(0,n_initial):
         cpp_file = open("datatoData1_b"+str(i)+".cpp", "r")
         list_of_lines = cpp_file.readlines()
         list_of_lines[21] = '        infile1.open("seqb'+str(i)+'.txt");\n'
         list_of_lines[37] = '        outfile.open("polymer2_b_'+str(i)+'.data");\n'
 
         cpp_file = open("datatoData1_b"+str(i)+".cpp", "w")
         cpp_file.writelines(list_of_lines)
         cpp_file.close()
 
     #STEP 3.3
     #polymer1.data to new_polymer data conversion, compiling datatoData1_b files
     for i in range(0,n_initial):
         os.system('g++ datatoData1_b'+str(i)+'.cpp')
         subprocess.call("./a.out")
 
     #STEP 3.4 
     #Duplicating lammps file and modification
     #duplicating lammps file
     for i in range(0,n_initial):
         shutil.copy('in.lammps', 'in_b_'+str(i)+'.lammps')
 
     #change lammps file content
     for i in range(0,n_initial):
         lammps_file = open("in_b_"+str(i)+".lammps", "r")
         list_of_content = lammps_file.readlines()
         list_of_content[13] = ' read_data polymer2_b_'+str(i)+'.data\n'
         list_of_content[58] = 'fix          RgAve  all ave/time 1000 1000 1000000 c_1 file AvgRg_b_'+str(i)+'.data\n'
 
         lammps_file = open("in_b_"+str(i)+".lammps", "w")
         lammps_file.writelines(list_of_content)
         lammps_file.close()
 
     #STEP 3.5 Evaluate generate script - needs to be modified
     subprocess.Popen(["chmod", "+x", "./bo_initial_try.sh"])
     subprocess.call("./bo_initial_try.sh")
    
     #STEP 3.6 GENERATING n_initial AVG Rg files and dump files
    
     while True:
         count_file = 0
         for i in range(0,n_initial):
             if os.path.exists("AvgRg_b_"+str(i)+".data"):
                 count_file = count_file + 1
         if count_file == n_initial:
             break
         else:
             time.sleep(400)
 
     time.sleep(100)
 
     #STEP 3.7 Read the (n_inital) no. of files to get the fitness values
     #get fitness value
     fitness = []
     for i in range(0,n_initial):
         rad_file = open("AvgRg_b_"+str(i)+".data", "r")
         list_of_content = rad_file.readlines()
         fitness.append(float(list_of_content[2].split(" ")[1]))
         rad_file.close()
         
      
         file2=open("/home/tpatra/Praneeth/"+file_var+"/mean_block/GPR_training/sequence"+str(i)+".txt"    ,"r")
         rad_add = file2.readlines()
         rad_add.append(" ")
         rad_add[100]=str(list_of_content[2].split(" ")[1])
         file2.close()
         file2=open("/home/tpatra/Praneeth/"+file_var+"/mean_block/GPR_training/sequence"+str(i)+".txt"    ,"w")
         file2.writelines(rad_add)
         file2.close()
         
 
     fitness = numpy.array(fitness)
     return fitness
 

def apply_single_mutation(oned_array_initial, num_weights):
    loc_overall = numpy.arange(0, num_weights)
    oned_array_output = []
 
    for loc in loc_overall:
        oned_array  = oned_array_initial.copy()
        oned_array[loc] = 2 if oned_array[loc]==1 else 1
        oned_array_output.append(oned_array)
        
    return(numpy.array(oned_array_output))



def y_query_max_sm(X_input,iter_count, num_weights):
    
     #STEP .1 opening sequence.txt files for all the datapoints in the sm
     for i in range(0,X_input.shape[0]):
        file2 = open("seq_sm"+str(i)+".txt", "w")    
        
        file1 = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/sm/sequence_"+str(iter_count)+"_"+str(i)+".txt","w")
        for j in range(0,num_weights):
            file2.write(str(X_input[i,j]))
            file1.write(str(X_input[i,j]))
            file1.write("\n")
            file2.write("\n")
        file2.close()
        file1.close()
      #STEP .2 Duplicating datatoData1.cpp files and modification
     #generate n_initial number of polymer 1 to 2 data conversion .cpp files
     for i in range(0,X_input.shape[0]):
         shutil.copy('datatoData1.cpp', 'datatoData1_sm'+str(i)+'.cpp')                                                                                                        

     #changing the content of .cpp files according to sequence files to be read
     for i in range(0,X_input.shape[0]):
         cpp_file = open("datatoData1_sm"+str(i)+".cpp", "r")
         list_of_lines = cpp_file.readlines()
         list_of_lines[21] = '        infile1.open("seq_sm'+str(i)+'.txt");\n'
         list_of_lines[37] = '        outfile.open("polymer2_sm_'+str(i)+'.data");\n'
 
         cpp_file = open("datatoData1_sm"+str(i)+".cpp", "w")
         cpp_file.writelines(list_of_lines)
         cpp_file.close()
 
     #STEP .3
     #polymer1.data to new_polymer data conversion, compiling datatoData1_b files
     for i in range(0,X_input.shape[0]):
         os.system('g++ datatoData1_sm'+str(i)+'.cpp')
         subprocess.call("./a.out")
 
     #STEP .4 
     #Duplicating lammps file and modification
     #duplicating lammps file
     for i in range(0,X_input.shape[0]):
         shutil.copy('in.lammps', 'in_sm_'+str(i)+'.lammps')
 
     #change lammps file content
     for i in range(0,X_input.shape[0]):
         lammps_file = open("in_sm_"+str(i)+".lammps", "r")
         list_of_content = lammps_file.readlines()
         list_of_content[13] = ' read_data polymer2_sm_'+str(i)+'.data\n'
         list_of_content[58] = 'fix          RgAve  all ave/time 1000 1000 1000000 c_1 file AvgRg_sm_'+str(i)+'.data\n'
 
         lammps_file = open("in_sm_"+str(i)+".lammps", "w")
         lammps_file.writelines(list_of_content)
         lammps_file.close()
 
     #STEP .5 Evaluate generate script - needs to be modified
     subprocess.Popen(["chmod", "+x", "./bo_sm_try.sh"])
     subprocess.call("./bo_sm_try.sh")
    
     #STEP .6 GENERATING n_initial AVG Rg files and dump files
    
     while True:
         count_file = 0
         for i in range(X_input.shape[0]):
             if os.path.exists("AvgRg_sm_"+str(i)+".data"):
                 count_file = count_file + 1
         if count_file == X_input.shape[0]:
             break
         else:
             time.sleep(400)
 
     time.sleep(100)    
     while_count = 0
     
     while True:
        while_count = while_count + 1
        fitness = []

        for i in range(X_input.shape[0]):
            rad_file = open("AvgRg_sm_"+str(i)+".data", "r")

            try:
                list_of_content = rad_file.readlines()

                file2 = open("/home/tpatra/Praneeth/"+file_var+"/mean_block/sm/sequence_"+str(iter_count)+"_"+str(i)+".txt", "r")
                rad_add = file2.readlines()
                rad_add.append(" ")
                rad_add[100]=str(list_of_content[-1].split(" ")[1])
                file2.close()
                file2=open("/home/tpatra/Praneeth/"+file_var+"/mean_block/sm/sequence_"+str(iter_count)+"_"+str(i)+".txt", "w")
                file2.writelines(rad_add)
                file2.close()

                fitness.append(float(list_of_content[2].split(" ")[1]))
                rad_file.close()    


            except:
                pass

        fitness = numpy.array(fitness)

        if len(fitness) == X_input.shape[0]:
            break

        elif while_count == 400:
            break

        else:
            time.sleep(1)
            
     if len(fitness)!=32:
         sys.exit("Exitting on account of list_of_content error")
    
     return(fitness)

        


    

def calc_cand_sm(X_initial, num_weights):
    X_initial_exp = X_initial.copy()
    
    cand_sm = []
    cand_sm = apply_single_mutation(X_initial[0], num_weights)
    
    for r_index in range(1, X_initial.shape[0]):
        exh_sm = apply_single_mutation(X_initial[r_index], num_weights)
        cand_sm = np.vstack([cand_sm, exh_sm])
    
    cand_sm_unique = np.unique(cand_sm, axis =0)
    cand_sm_append = np.array([cand_sm_unique_elem for cand_sm_unique_elem in cand_sm_unique if any(np.equal(X_initial_exp,cand_sm_unique_elem).all(1)) == False ])    
    
    
    return(cand_sm_append)

