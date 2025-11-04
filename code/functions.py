
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

def load_lake_obs():
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
  return lake_obs[lake_obs.depth < lake_obs.max_depth]


def match_weather(site, lake_obs, lags, rollings, train_stop_year, full=False):
  site_df = lake_obs[lake_obs.site_id == site].copy()
  
  site_df['volume'] = site_df['area'] * site_df['max_depth']
  site_df['depth_area_ratio'] = site_df['max_depth'] / site_df['area']
  
  weather_filename = site_df.driver_nldas_filepath.unique()[0]
  weather_df = pd.read_csv('./data/meteo_csv_files/' + weather_filename)
  weather_df.rename(columns={'time':'date'}, inplace=True)
  weather_df['date'] = pd.to_datetime(weather_df['date'])
  weather_df.sort_values(by='date', inplace=True)
  
  # augmenting weather dataset with lags 
  lagged_cols = []
  for c in list(weather_df.columns[1:]):
    for l in lags:
      lagged_cols.append(weather_df.loc[:,c].shift(l).rename(f'{c}_{l}_lag'))
    for r in rollings:
      lagged_cols.append(weather_df.loc[:,c].shift(1).rolling(window=r).mean().rename(f'{c}_{r}_mean'))
  
  weather_df = pd.concat([weather_df] + lagged_cols, axis=1)
  
  if full:
    cols_to_keep = ['lon', 'lat', 'max_depth', 'elevation', 'area','volume','depth_area_ratio']
    values_to_keep = np.asarray(site_df[cols_to_keep].iloc[0,:]).reshape(1,-1)
    
    weather_cols = list(weather_df.columns)[1:]
    df = weather_df.copy()
    df[cols_to_keep] = np.tile(values_to_keep, (df.shape[0],1))
    df = df[['date'] + cols_to_keep + weather_cols]
    df['site_id'] = site
    df['lake_name'] = site_df.lake_name.unique()[0]
    df['state'] = site_df.state.unique()[0]
    df = df[df.date.dt.year > train_stop_year]
  else:
    # merging weather with lake obs
    df = pd.merge(site_df, weather_df, on='date', how='left')
    df.drop(columns=['driver_nldas_filepath'], inplace=True)
  
  #augmenting final data with seasonal indicators
  df['day_sin'] = np.sin(2 * np.pi * df.date.dt.day_of_year/365)
  df['day_cos'] = np.cos(2 * np.pi * df.date.dt.day_of_year/365)
  
  df['month_sin'] = np.sin(2 * np.pi * df.date.dt.month/12)
  df['month_cos'] = np.cos(2 * np.pi * df.date.dt.month/12)
  df['year'] = df.date.dt.year
  
  return df


def load_data():
  data = np.load('../data/train_val.npz')
  time_vars = data['time']
  space_vars = data['space']
  depth = data['depth']
  max_depth = data['max_depth']
  temperature = data['temperature']
  year = data['year']
  return time_vars, space_vars, depth, max_depth, temperature, year

def split_data(time_vars, space_vars, depth, max_depth, temperature, year, train_stop=2015):
  time_train, space_train = time_vars[year <= train_stop,:], space_vars[year <= train_stop,:]
  depth_train, max_depth_train = depth[year <= train_stop,:], max_depth[year <= train_stop,:]
  temp_train = temperature[year <= train_stop,:]
  
  time_val, space_val = time_vars[year > train_stop,:], space_vars[year > train_stop,:]
  depth_val, max_depth_val = depth[year > train_stop,:], max_depth[year > train_stop,:]
  temp_val = temperature[year > train_stop,:]
  
  mins_time, maxs_time = time_train.min(axis=0), time_train.max(axis=0)
  mins_space, maxs_space = space_train.min(axis=0), space_train.max(axis=0)
  maxs_time[-1] = 2080
  
  time_train_scaled = (time_train-mins_time) / (maxs_time-mins_time)
  time_val_scaled = (time_val-mins_time) / (maxs_time-mins_time)
  
  space_train_scaled = (space_train-mins_space) / (maxs_space-mins_space)
  space_val_scaled = (space_val-mins_space) / (maxs_space-mins_space)
  
  datasets = {}
  datasets['X_train'] = (time_train_scaled, space_train_scaled, depth_train, max_depth_train)
  datasets['y_train'] = temp_train
  datasets['X_val'] = (time_val_scaled, space_val_scaled, depth_val, max_depth_val)
  datasets['y_val'] = temp_val
  
  scales = {}
  scales['mins_time'] = mins_time
  scales['maxs_time'] = maxs_time
  scales['mins_space'] = mins_space
  scales['maxs_space'] = maxs_space
  
  return datasets, scales
  
def load_full():
  
  data = np.load('../data/full_data.npz')
  time_vars = data['features_time']
  space_vars = data['features_space']
  max_depth = data['max_depth']
  dates = data['dates']
  lonlat = data['lonlat']

  with open('../data/full_data_info.pkl' + i, 'rb') as f:
      sites = pickle.load(f) 

  return time_vars, space_vars, max_depth, dates, lonlat, np.array(sites).reshape(-1,1)
  
def augment_for_proj(s, df):
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
  

@tf.keras.utils.register_keras_serializable() 
class LakeTempLayerST(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(LakeTempLayerST, self).__init__(**kwargs)

  def call(self, inputs):
    raw_params, depth, max_depth = inputs
    
    T_bot = 1.0 + 45.*raw_params[:, 0:1] + 1e-3
    T_top = 1e-3 + 45.*raw_params[:, 1:2]
    a = 1e-3 + raw_params[:, 2:3] #tf.math.sigmoid() + 1e-3
    k = 1e-2 + raw_params[:, 3:4] #tf.math.softplus() + 1e-3
    
    x = tf.clip_by_value((depth - a*max_depth) / k, -50.0, 50.0)
    return T_bot + (T_top - T_bot) / (1 + tf.exp(x))
    
  @classmethod
  def from_config(cls, config):
    return cls(**config)
  
def build_st_model(time_shape, space_shape, time_sizes, space_sizes, comb_sizes,  drop_time, drop_space, drop_comb, suffix):
    
  input_time = tf.keras.layers.Input(shape=time_shape, name='Weather')
  input_space = tf.keras.layers.Input(shape=space_shape, name='LakeChars') 
  input_depth = tf.keras.layers.Input(shape=(1,), name='Depth')
  input_max_depth = tf.keras.layers.Input(shape=(1,), name='MaxDepth')
  x_time = input_time
  x_space = input_space
    
  for i, s_t in enumerate(time_sizes):
      x_time = tf.keras.layers.Dense(s_t, name=f'Dense_Weather{i+1}')(x_time)
      x_time = tf.keras.layers.BatchNormalization(name=f'Norm_Weather{i+1}')(x_time)
      x_time = tf.keras.layers.LeakyReLU(name=f'LeakyReLU_Weather{i+1}')(x_time)
      x_time = tf.keras.layers.Dropout(drop_time[i], name=f'Drop_Weather{i+1}')(x_time)
  
  for i, s_s in enumerate(space_sizes):
      x_space = tf.keras.layers.Dense(s_s, name=f'Dense_LakeChars{i+1}')(x_space)
      x_space = tf.keras.layers.BatchNormalization(name=f'Norm_LakeChars{i+1}')(x_space)
      x_space = tf.keras.layers.LeakyReLU(name=f'LeakyReLU_LakeChars{i+1}')(x_space)
      x_space = tf.keras.layers.Dropout(drop_space[i], name=f'Drop_LakeChars{i+1}')(x_space)
  
  ###############################################################################################
  if suffix == 'NPI' or suffix == 'MOE':
    # Non-PINN branch
    x_nonpinn = tf.keras.layers.Concatenate(name='Together_NonPINN')([x_time, x_space, input_depth, input_max_depth])
    for i, s in enumerate(comb_sizes):
        x_nonpinn = tf.keras.layers.Dense(s, name=f'Dense_Comb_NonPINN{i+1}')(x_nonpinn)
        x_nonpinn = tf.keras.layers.BatchNormalization(name=f'Norm_Comb_NonPINN{i+1}')(x_nonpinn)
        x_nonpinn = tf.keras.layers.LeakyReLU(name=f'LeakyReLU_Comb_NonPINN{i+1}')(x_nonpinn)
        x_nonpinn = tf.keras.layers.Dropout(drop_comb[i], name=f'Drop_Comb_NonPINN{i+1}')(x_nonpinn)
    x_nonpinn_last = tf.keras.layers.Dense(comb_sizes[-1], name='Dense_Last_NonPINN')(x_nonpinn)
    temp_nonpi = tf.keras.layers.Dense(1, activation='sigmoid', name='ML_Temperature_sigmoid')(x_nonpinn_last)
    temp_nonpi = tf.keras.layers.Lambda(lambda x: 1e-3 + 45.*x, name="ML_Temperature")(temp_nonpi)
    ###############################################################################################
    if suffix == 'NPI': 
      return tf.keras.Model([input_time, input_space, input_depth, input_max_depth], temp_nonpi, name=f'ST_Model_{suffix}')
     
  # PINN branch
  if suffix == 'MOE' or suffix == 'PINN':
    x_pinn = tf.keras.layers.Concatenate(name='Together_PINN')([x_time, x_space])
    for i, s in enumerate(comb_sizes):
        x_pinn = tf.keras.layers.Dense(s, name=f'Dense_Comb_PINN{i+1}')(x_pinn)
        x_pinn = tf.keras.layers.BatchNormalization(name=f'Norm_Comb_PINN{i+1}')(x_pinn)
        x_pinn = tf.keras.layers.LeakyReLU(name=f'LeakyReLU_Comb_PINN{i+1}')(x_pinn)
        x_pinn = tf.keras.layers.Dropout(drop_comb[i], name=f'Drop_Comb_PINN{i+1}')(x_pinn)
    x_pinn_last = tf.keras.layers.Dense(comb_sizes[-1], name='Dense_Last_PINN')(x_pinn)
    
    T_bot = tf.keras.layers.Dense(1, activation='sigmoid', name='T_bot')(x_pinn_last) #sigmoid
    T_top = tf.keras.layers.Dense(1, activation='sigmoid', name='T_top')(x_pinn_last) #sigmoid
    a = tf.keras.layers.Dense(1, activation='sigmoid', name='a')(x_pinn_last)
    k = tf.keras.layers.Dense(1, activation='relu', name='k')(x_pinn_last)
    
    params = tf.keras.layers.Concatenate(name='Parameters')([T_bot, T_top, a, k])
    temp_pi = LakeTempLayerST(name='PINN_Temperature')([params, input_depth, input_max_depth])
    ###############################################################################################
    if suffix == 'PINN':
      return tf.keras.Model([input_time, input_space, input_depth, input_max_depth], temp_pi, name=f'ST_Model_{moe}')
  
  weight_input = tf.keras.layers.Concatenate(name='Weight_Input')([x_time, x_space])
  weight = tf.keras.layers.Dense(1, activation='sigmoid', name="Weight")(weight_input)
  
  # Combine outputs
  temperature = tf.keras.layers.Lambda(
      lambda inputs: inputs[2] * inputs[0] + (1.0 - inputs[2]) * inputs[1], name="Weighted_Temperature")([temp_pi, temp_nonpi, weight])
  
  return tf.keras.Model([input_time, input_space, input_depth, input_max_depth], temperature, name=f'ST_Model_{moe}')
    

class LossLogger(tf.keras.callbacks.Callback):
  def __init__(self, filename):
    super().__init__()
    self.filename = filename
    with open(self.filename, 'w') as f:
      f.write("Model history\n")

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    with open(self.filename, 'a') as f:
      f.write(f"{epoch+1}, Loss: {logs.get('loss'):.4f}, Val Loss: {logs.get('val_loss'):.4f}\n")  
 
  
