
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

lake_obs.drop(columns=['state','lake_name'], inplace=True)

# load the site codes to project
with open('../data/manuscript_sites.pkl', 'rb') as f:
  sites_to_proj = pickle.load(f) 

for i, nc_f in enumerate(os.listdir('./data/GCM/')):
  nc_file = "../data/GCM/" + nc_f
  ds = xr.open_dataset(nc_file)
  
  # gcm_cell_id, time, lat, lon, Rain, Snow, AirTemp, RelHum, Shortwave, Longwave, WindSpeed
  # in the variables, each ROW is a location
  
  data_dict = {var: ds[var].values for var in ds.variables}
  data_dict['ShortWave'] = data_dict.pop('Shortwave')
  data_dict['LongWave'] = data_dict.pop('Longwave')
  
  df = pd.DataFrame(np.tile(data_dict['time'], data_dict['lon'].shape[0]), columns=['date'])
  df['date'] = pd.to_datetime(df['date'])
  
  df['lon'] = np.repeat(data_dict['lon'], data_dict['time'].shape[0])
  df['lat'] = np.repeat(data_dict['lat'], data_dict['time'].shape[0])
  df['driver_gcm_cell_no'] = np.repeat(data_dict['gcm_cell_id'], data_dict['time'].shape[0])
  
  
  weather_cols = ['ShortWave', 'LongWave','AirTemp', 'RelHum','WindSpeed','Rain', 'Snow']
  for v in weather_cols:
    df[v] = data_dict[v].flatten()
   
  df = df[(df.date.dt.year >= 2040) & (df.date.dt.year <= 2081)] 
  lake_meta.dropna(subset='driver_gcm_cell_no', inplace=True)
  merge_cols = ['site_id','driver_gcm_cell_no','max_depth','elevation','area']
  df = pd.merge(df, lake_meta[merge_cols], on=['driver_gcm_cell_no'], how='left')
  df.dropna(inplace=True)
    
  print(f'\nConstructing full future data for {nc_f}...')
  sys.stdout.flush()
  list_dfs = Parallel(n_jobs=-1, verbose=5)(delayed(augment_for_proj)(s, df) for s in sites_to_proj)
  final_df = pd.concat(list_dfs, ignore_index=True)
  sys.stdout.flush()
  print('Done.\n')
  
  seasonal_cols = []
  seasonal_cols.append(np.sin(2 * np.pi * final_df.date.dt.day_of_year/365).rename('day_sin'))
  seasonal_cols.append(np.cos(2 * np.pi * final_df.date.dt.day_of_year/365).rename('day_cos'))
  
  seasonal_cols.append(np.sin(2 * np.pi * final_df.date.dt.month/12).rename('month_sin'))
  seasonal_cols.append(np.cos(2 * np.pi * final_df.date.dt.month/12).rename('month_cos'))
  
  seasonal_cols.append(final_df.date.dt.year.rename('year'))
  
  final_df = pd.concat([final_df] + seasonal_cols, axis=1)
  
  ###################################################################################
  # Standardize and save
  time_gcm = final_df[time_features].to_numpy().astype(np.float32)
  space_gcm = final_df[space_features].to_numpy().astype(np.float32)
  max_depth_gcm = final_df[['max_depth']].to_numpy().astype(np.float32)
  
  time_gcm_scaled = (time_gcm-scales['mins_time']) / (scales['maxs_time']-scales['mins_time'])
  space_gcm_scaled = (space_gcm-scales['mins_space']) / (scales['maxs_space']-scales['mins_space'])
  
  
  np.savez(f'../data/climate_data_{nc_f.split('_')[1][:-3]}.npz',
    time_gcm=time_gcm_scaled, space_gcm=space_gcm_scaled, max_depth_gcm=max_depth_gcm,
    dates=final_df[['date']].to_numpy())
  
  with open(f'./data/climate_site_ids.pkl', 'wb') as f:
    pickle.dump(final_df['site_id'].to_list(), f) 


