# Improved-Hyperparameter-Optimization-
Code repository for the article "Stealthy hyperuniform sampling for hyperparameter optimisation". The data needed for all scripts to run can be found
here https://sid.erda.dk/sharelink/gDtCoRDXC6, from now on called the erda directory.

### Closed_form_functions 

This directory contains files necessary for generating closed-form function results. Specifically, **Generic_generate_results.py** (with the aid of **Hyperuniform.py**) contains the benchmark code, whereas **Figure_generation.py** analyzes the results and generates figures and tables used in the manuscript. Files in the **results** directory contain data presented in the paper.

### Figure_1.py

This program generates Figure 1 from the paper, which demonstrates the advantage of hyperuniform patterns over grid and random search.

### Figure_LGBM and Lightgbm

These folders contain programs and data used for generating figures and tables corresponding to the LightGBM benchmark test.

The LGBM analysis is made by first using kdd12_batch_maker.py to generate
batches of cleaned combined data. These have different sizes. From here the 
Lightgbm_analysis_mpi.py file is run, preferably on a large cluster, using the
pipings to control what dimensions and resolutions to run and how many workers
to use. The results from this analysis are then finally compressed down to the 
data needed for the plots with Results_Compression.py. Here the results coming from some grids have nonuniform
size and taking the mean and standard deviation is therefore not possible with 
array operations. 

#### Data for LGBM analysis
The data is available here https://www.kaggle.com/competitions/kddcup2012-track1/data.
The files needed are rec_log_train.txt and user_profile.txt 

The batches made from the KDD12 dataset with varying sizes and cleaned data for the LGBM analysis are found on the erda directory in the 'Batches' sub directory
The raw results are found in the 'LGBM/All_data' subdirectory. With the compressed results in the Figure_LGBM directory here.