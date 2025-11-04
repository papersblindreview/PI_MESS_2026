
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

# LOADING LAKE OBSERVATIONS
lake_obs = pd.read_csv('../data/lake_temperature_observations.csv')
lake_obs = lake_obs.loc[:,['date','site_id','depth','temp','source_id']]

# unreliable source, so we drop it
lake_obs = lake_obs[lake_obs.source_id != 'MN_sentinel_lakes_application'].drop(columns='source_id')
lake_obs['date'] = pd.to_datetime(lake_obs['date'])


# LOADING LAKE METADATA (e.g., area, lon, lat)
lake_meta = pd.read_csv('../data/lake_metadata.csv') 
meta_cols = ['site_id','lon','lat','max_depth','elevation','area','driver_nldas_filepath','state','lake_name','driver_gcm_cell_no']
lake_meta.rename(columns={'centroid_lon':'lon', 'centroid_lat':'lat'}, inplace=True)


# MERGING OBSERVATIONS WITH METADATA FOR EACH LAKE
lake_obs = pd.merge(lake_obs, lake_meta[meta_cols], on='site_id', how='left')
lake_obs = lake_obs[lake_obs.driver_nldas_filepath.isin(os.listdir('../data/meteo_csv_files'))].dropna()

# drop observations deeper than max depth
lake_obs = lake_obs[lake_obs.depth < lake_obs.max_depth]

#########################################################################################################
states = lake_obs['state']
names = lake_obs['lake_name']
gcm_cell = lake_obs['driver_gcm_cell_no']
lake_obs.drop(columns=['state','lake_name','driver_gcm_cell_no'], inplace=True)

unique_sites = lake_obs.site_id.value_counts().index

#MERGING LAKE OBS WITH WEATHER CONDITIONS, FOR EACH SITE
lags = np.arange(1,11)
rollings = [14,30,60,90]
train_stop_year = 2015  


############################################################################################
'''
partition = np.array_split(unique_sites, 20)

print('\nConstructing training-validation sets...')
sys.stdout.flush()
list_dfs = Parallel(n_jobs=-1, verbose=5)(delayed(match_weather)(site) for site in partition[part])
final_df = pd.concat(list_dfs, ignore_index=True)
final_df.dropna(inplace=True)
sys.stdout.flush()
print('Done.\n')


space_feature_names = ['lon','lat','elevation','area','volume','depth_area_ratio']  
time_feature_names = list(final_df.drop(columns=['site_id','date','temp','depth','max_depth'] + space_feature_names).columns)


np.savez(f'./data/train_val/part_{part}.npz',
        time=final_df[time_feature_names].to_numpy().astype(np.float32),
        space=final_df[space_feature_names].to_numpy().astype(np.float32),
        depth=final_df[['depth']].to_numpy().astype(np.float32),
        max_depth=final_df[['max_depth']].to_numpy().astype(np.float32),
        temperature=final_df[['temp']].to_numpy().astype(np.float32),
        year=np.array(final_df.date.dt.year).astype(np.float32))

with open('./data/train_val/feature_names.pkl', 'wb') as f:
  pickle.dump({'time':list(time_feature_names), 'space':list(space_feature_names)}, f)  
'''

############################################################################################
## CREATING FULL DATASET TO INTERPOLATE / EXTRAPOLATE
with open('./data/train_val/feature_names.pkl', 'rb') as f:
  feature_names = pickle.load(f)
  
time_features = feature_names['time']
space_features = feature_names['space']

time_vars, space_vars, depth, max_depth, temperature, year = load_data()
datasets, scales = split_data(time_vars, space_vars, depth, max_depth, temperature, year)

  
lake_obs['state'] = states
lake_obs['lake_name'] = names

'''
lake_obs = lake_obs[(lake_obs.date.dt.year > 2015)]

lake_subset = lake_obs.groupby("site_id").filter(lambda g: g["date"].nunique() > 10)
lake_subset.sort_values(by='date', inplace=True)
lake_subset.reset_index(inplace=True)

lake_subset.dropna(inplace=True)

unique_sites_full = lake_subset.site_id.unique()
partition = np.array_split(unique_sites_full, 20)

print('Constructing full validation set...')
sys.stdout.flush()
list_full_dfs = Parallel(n_jobs=-1, verbose=5)(delayed(match_weather)(site, full=True) for site in partition[part])
full_df = pd.concat(list_full_dfs, ignore_index=True).reset_index(drop=True).dropna()
sys.stdout.flush()
print('Done.\n')
  
del list_full_dfs
gc.collect()

time_full = (full_df[time_features] - scales['mins_time']) / (scales['maxs_time']-scales['mins_time'])
space_full = (full_df[space_features] - scales['mins_space']) / (scales['maxs_space']-scales['mins_space'])
 

np.savez(f'./data/test/full_data_{part}.npz', 
         features_time=time_full.to_numpy(),
         features_space=space_full.to_numpy(),
         max_depth=full_df[['max_depth']].to_numpy(),
         dates=full_df[['date']].to_numpy(),
         lonlat=full_df[['lon','lat']].to_numpy()) 

with open(f'./data/test/full_data_info_{part}.pkl', 'wb') as f:
  pickle.dump(list(full_df.site_id), f)  

'''
########################################################################################

lake_obs['driver_gcm_cell_no'] = gcm_cell

#lakes_to_proj = lake_obs[(lake_obs.lake_name.str.contains('Mendota|Mille Lacs'))].drop_duplicates(subset='site_id')
#sites_to_proj = lakes_to_proj.site_id
with open('./data/manuscript_sites.pkl', 'rb') as f:
  sites_to_proj = pickle.load(f)


# Add features as in the training data
def augment(s, df):
  df_temp = df[df.site_id == s].copy()
  
  df_temp['volume'] = df_temp['area'] * df_temp['max_depth']
  df_temp['depth_area_ratio'] = df_temp['max_depth'] / df_temp['area']
    
  lagged_cols = []
  for c in list(weather_cols):
    for l in lags:
      lagged_cols.append(df_temp.loc[:,c].shift(l).rename(f'{c}_{l}_lag'))
    for r in rollings:
      lagged_cols.append(df_temp.loc[:,c].shift(1).rolling(window=r).mean().rename(f'{c}_{r}_mean'))
  
  df_temp = pd.concat([df_temp] + lagged_cols, axis=1)
  return df_temp.dropna()
 

for i, nc_f in enumerate(os.listdir('./data/GCM/')):
  nc_file = "./data/GCM/" + nc_f
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
  
  for col in df.select_dtypes(include='float64'):
    df[col] = df[col].astype('float32', copy=False)
   
  df = df[(df.date.dt.year >= 2040) & (df.date.dt.year <= 2081)] 
  lake_meta.dropna(subset='driver_gcm_cell_no', inplace=True)
  merge_cols = ['site_id','driver_gcm_cell_no','max_depth','elevation','area']
  df = pd.merge(df, lake_meta[merge_cols], on=['driver_gcm_cell_no'], how='left')
  df.dropna(inplace=True)
    
  print(f'\nConstructing full future data for {nc_f}...')
  sys.stdout.flush()
  list_dfs = Parallel(n_jobs=-1, verbose=5)(delayed(augment)(s, df) for s in sites_to_proj)
  final_df = pd.concat(list_dfs, ignore_index=True)
  sys.stdout.flush()
  print('Done.\n')
  
  print(final_df[['date']].describe())
  
  seasonal_cols = []
  seasonal_cols.append(np.sin(2 * np.pi * final_df.date.dt.day_of_year/365).rename('day_sin'))
  seasonal_cols.append(np.cos(2 * np.pi * final_df.date.dt.day_of_year/365).rename('day_cos'))
  
  seasonal_cols.append(np.sin(2 * np.pi * final_df.date.dt.month/12).rename('month_sin'))
  seasonal_cols.append(np.cos(2 * np.pi * final_df.date.dt.month/12).rename('month_cos'))
  
  seasonal_cols.append(final_df.date.dt.year.rename('year'))
  
  final_df = pd.concat([final_df] + seasonal_cols, axis=1)
  
  ###################################################################################
  # Standardize and save
  if i == 0: print(final_df.site_id.value_counts())
  time_gcm = final_df[time_features].to_numpy().astype(np.float32)
  space_gcm = final_df[space_features].to_numpy().astype(np.float32)
  max_depth_gcm = final_df[['max_depth']].to_numpy().astype(np.float32)
  
  time_gcm_scaled = (time_gcm-scales['mins_time']) / (scales['maxs_time']-scales['mins_time'])
  space_gcm_scaled = (space_gcm-scales['mins_space']) / (scales['maxs_space']-scales['mins_space'])
  
  
  np.savez(f'./data/climate_data_{nc_f.split('_')[1][:-3]}.npz',
    time_gcm=time_gcm_scaled, space_gcm=space_gcm_scaled, max_depth_gcm=max_depth_gcm,
    dates=final_df[['date']].to_numpy())
  
  with open(f'./data/climate_site_ids.pkl', 'wb') as f:
    pickle.dump(final_df['site_id'].to_list(), f) 


