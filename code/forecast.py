import os
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
from functions import *
import gc
global suffix

suffix = 'MOE' #'PINN', 'NPI'

#################################################################
# Load climate climate data   

def load_climate_data(gcm):
  data_full = np.load(f'./data/{gcm}')
     
  features_time = data_full['time_gcm'] 
  features_space = data_full['space_gcm']
  max_depth = data_full['max_depth_gcm']
  dates = data_full['dates']
  
  return features_time, features_space, max_depth, dates
  

with open(f'./data/climate_site_ids.pkl', 'rb') as f:
  sites = pickle.load(f) 

site_ids = np.array(sites)

################################################################
# Load model

model = tf.keras.saving.load_model(f'./model_results/model_{suffix}.keras', safe_mode=False)

###################################################################
# Predict 

n_depths = 25
gcm_list = [f for f in os.listdir('./data/') if 'climate_data' in f]

for gcm in gcm_list:
  features_time, features_space, max_depth, dates = load_climate_data(gcm)
  
  preds_new = {}
  
  for s in np.unique(sites):
    
    max_d = max_depth[np.where(site_ids == s)[0],0][0]
    depths = np.linspace(0, max_d, n_depths)
    
    time_temp = features_time[np.where(site_ids == s)[0],:]
    space_temp = features_space[np.where(site_ids == s)[0],:]
    n_temp = time_temp.shape[0]
    
    time_temp = np.tile(time_temp, (n_depths,1))
    space_temp = np.tile(space_temp, (n_depths,1))
    depth_temp = np.repeat(depths, n_temp).reshape(-1,1)
    max_depth_temp = max_d * np.ones((time_temp.shape[0],1))
    
    preds_temp = model.predict([time_temp, space_temp, depth_temp, max_depth_temp], verbose=0)
    dates_temp = dates[np.where(site_ids == s)[0],0]
    dates_temp = np.tile(dates_temp, n_depths)
    dates_numeric = dates_temp.astype('datetime64[s]').astype(float)
    
    preds_new[s] = np.stack((dates_numeric, depth_temp[:,0], preds_temp[:,0]), axis=-1)
    
  print(np.unique(sites))
  with open(f"./model_results/forecast_{gcm.split('_')[-1][:-4]}_{suffix}.pkl", 'wb') as f:
    pickle.dump({'sites':np.unique(sites), 'preds':preds_new}, f)














