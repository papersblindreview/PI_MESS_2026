
import os 
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import gc
import xarray as xr
import sys
from functions import *

np.random.seed(1)

lake_obs = load_lake_obs()

#########################################################################################################
# drop unnecessary columns
lake_obs.drop(columns=['state','lake_name','driver_gcm_cell_no'], inplace=True)

unique_sites = lake_obs.site_id.value_counts().index

lags = np.arange(1,11) # lags for each meteorological driver
rollings = [14,30,60,90] # windows over which to take rolling averages 
train_stop_year = 2015  # training only up to 2015

############################################################################################
# CREATING TRAINING AND TESTING DATASETS

# The function 'match_weather' grabs, for each site, the historical weather data and creates the covariates for that site
# The input "full=False" means that we just want the days and depths for which there are observations
# If we want to predict all depths and all days, we toggle it to True

print('\nConstructing training-validation sets...')
sys.stdout.flush()
list_dfs = Parallel(n_jobs=-1, verbose=5)(delayed(match_weather)(site, lake_obs, lags, rollings, train_stop_year, full=False) for site in unique_sites)
final_df = pd.concat(list_dfs, ignore_index=True)
final_df.dropna(inplace=True)
sys.stdout.flush()
print('Done.\n')

space_feature_names = ['lon','lat','elevation','area','volume','depth_area_ratio']  
time_feature_names = list(final_df.drop(columns=['site_id','date','temp','depth','max_depth'] + space_feature_names).columns)


np.savez(f'../data/train_val.npz',
        time=final_df[time_feature_names].to_numpy().astype(np.float32),
        space=final_df[space_feature_names].to_numpy().astype(np.float32),
        depth=final_df[['depth']].to_numpy().astype(np.float32),
        max_depth=final_df[['max_depth']].to_numpy().astype(np.float32),
        temperature=final_df[['temp']].to_numpy().astype(np.float32),
        year=np.array(final_df.date.dt.year).astype(np.float32))

with open('../data/feature_names.pkl', 'wb') as f:
  pickle.dump({'time':time_feature_names, 'space':space_feature_names}, f)  

