import os
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
from functions import *
import gc

suffix = 'MOE' #'PINN', 'NPI'
  
model_config = get_model_config('cases_st.txt', row_number)

time_vars, space_vars, depth, max_depth, temperature, year = load_data()
datasets, _ = split_data(time_vars, space_vars, depth, max_depth, temperature, year)

with tf.device('/GPU:0'):

  model = tf.keras.saving.load_model(f'./models/model_{suffix}.keras', safe_mode=False)
  if suffix == 'MOE':
    weight_model = tf.keras.Model(model.inputs, model.get_layer("Weight").output)
    pars_model = tf.keras.Model(model.inputs, model.get_layer("Parameters").output)
    ml_model = tf.keras.Model(model.inputs, model.get_layer("ML_Temperature").output)
    pi_model = tf.keras.Model(model.inputs, model.get_layer("PINN_Temperature").output)

  signature_val = (tf.TensorSpec(shape=[None,time_vars.shape[1]], dtype=tf.float32),
                   tf.TensorSpec(shape=[None,space_vars.shape[1]], dtype=tf.float32),
                   tf.TensorSpec(shape=[None,1], dtype=tf.float32),
                   tf.TensorSpec(shape=[None,1], dtype=tf.float32),
                   tf.TensorSpec(shape=[None,1], dtype=tf.float32))

  @tf.function(input_signature=signature_val)
  def predict_val_location(time_val_temp, space_val_temp, depth_val_temp, max_depth_val_temp, temp_temp):
    
    preds_temp = model([time_val_temp, space_val_temp, depth_val_temp, max_depth_val_temp], training=False)
    mse_by_lake_temp = tf.reduce_mean((preds_temp-temp_temp)**2)
    mse_temp = tf.reduce_mean((preds_temp-temp_temp)**2) * tf.cast(tf.shape(temp_temp)[0], tf.float32)
    
    return mse_by_lake_temp, mse_temp, preds_temp
    
  signature_full = (tf.TensorSpec(shape=[None,time_vars.shape[1]], dtype=tf.float32),
                   tf.TensorSpec(shape=[None,space_vars.shape[1]], dtype=tf.float32),
                   tf.TensorSpec(shape=[None,1], dtype=tf.float32),
                   tf.TensorSpec(shape=[None,1], dtype=tf.float32))

  @tf.function(input_signature=signature_full)
  def predict_full_location(features_time_temp, features_space_temp, depth_temp, max_depth_temp):
    return model([features_time_temp, features_space_temp, depth_temp, max_depth_temp], training=False)
                   
  @tf.function(input_signature=signature_full)
  def predict_full_location_pars(features_time_temp, features_space_temp, depth_temp, max_depth_temp):
    preds_ml = ml_model([features_time_temp, features_space_temp, depth_temp, max_depth_temp], training=False)
    preds_pi = pi_model([features_time_temp, features_space_temp, depth_temp, max_depth_temp], training=False)
    weight_new = weight_model([features_time_temp, features_space_temp, depth_temp, max_depth_temp], training=False)
    pars_new = pars_model([features_time_temp, features_space_temp, depth_temp, max_depth_temp], training=False)
    return preds_ml, preds_pi, weight_new, pars_new    
     

time_val, space_val = datasets['X_val'][0], datasets['X_val'][1] 
depth_val, max_depth_val, temp_val = datasets['X_val'][2], datasets['X_val'][3], datasets['y_val'] 

preds_val = model.predict([time_val, space_val, depth_val, max_depth_val], verbose=0)
res = preds_val - temp_val
np.savez(f'./model_results/residuals_{suffix}_{row_number:02d}.npz', res = res)


unique_locs_val = np.unique(space_val[:,:2], axis=0)

time_val_tf = tf.constant(time_val, dtype=tf.float32)
space_val_tf = tf.constant(space_val, dtype=tf.float32)
depth_val_tf = tf.constant(depth_val, dtype=tf.float32)
max_depth_val_tf = tf.constant(max_depth_val, dtype=tf.float32)
temp_val_tf = tf.constant(temp_val, dtype=tf.float32)
  
mse = []
mse_by_lake = []
res = []
for loc in unique_locs_val:
  mask = tf.reduce_all(tf.equal(space_val_tf[:, :2], tf.constant(loc, dtype=tf.float32)), axis=1)
    
  time_val_temp = tf.boolean_mask(time_val_tf, mask)
  space_val_temp = tf.boolean_mask(space_val_tf, mask)
  depth_val_temp = tf.boolean_mask(depth_val_tf, mask)
  max_depth_val_temp = tf.boolean_mask(max_depth_val_tf, mask)
  temp_temp = tf.boolean_mask(temp_val_tf, mask)
  
  mse_by_lake_temp, mse_temp, _ = predict_val_location(time_val_temp, space_val_temp, depth_val_temp, max_depth_val_temp, temp_temp)
  mse.append(mse_temp.numpy())
  mse_by_lake.append(mse_by_lake_temp.numpy())

  
mse = np.asarray(mse)
mse_by_lake = np.asarray(mse_by_lake)

mse_tot = mse.sum()/time_val.shape[0]
mse_med = np.median(mse_by_lake)
mse_q1 = np.quantile(mse_by_lake, 0.25)
mse_q2 = np.quantile(mse_by_lake, 0.75)
  

print('\nValidation Results.')
print(f'Overall MSE: {mse_tot:.2f}')
print(f'Lake-wise. Median: {mse_med:.2f}. IQR: ({mse_q1:.2f}, {mse_q2:.2f})')

features_time, features_space, max_depth, dates, lonlat, sites = load_full()


features_time_tf = tf.constant(features_time, dtype=tf.float32)
features_space_tf = tf.constant(features_space, dtype=tf.float32)
max_depth_tf = tf.constant(max_depth, dtype=tf.float32)
dates_tf = tf.constant(dates, dtype=tf.float32)
lonlat_tf = tf.constant(features_space[:,:2], dtype=tf.float32)
sites_tf = tf.constant(sites)
unique_locations = tf.constant(np.unique(features_space[:,:2], axis=0), dtype=tf.float32)

preds_new = []
preds_ml = []
preds_pi = []
weight_new = []
pars_new = []
sites_new = []
n_depths = 25

for i, s in enumerate(unique_locations):
  
  mask = tf.reduce_all(tf.equal(features_space_tf[:,:2], s), axis=1)
  
  features_time_temp = tf.boolean_mask(features_time_tf, mask)
  features_space_temp = tf.boolean_mask(features_space_tf, mask)
  max_depth_temp = tf.boolean_mask(max_depth_tf, mask)
  sites_temp = tf.boolean_mask(sites_tf, mask)[0,0]
  
  features_time_temp = tf.tile(features_time_temp, (n_depths,1))
  features_space_temp = tf.tile(features_space_temp, (n_depths,1))
  max_depth_temp = tf.tile(max_depth_temp, (n_depths,1))
  depth_temp = tf.constant(np.linspace(0, tf.reduce_max(max_depth_temp), n_depths), dtype=tf.float32)
  depth_temp = tf.reshape(tf.tile(depth_temp, [features_time_temp.shape[0] // n_depths]), [-1,1])
  
  if 'moe' in suffix:
    preds_ml_, preds_pi_, weight_new_, pars_new_ = predict_full_location_pars(features_time_temp, features_space_temp, depth_temp, max_depth_temp)
    preds_ml.append(preds_ml_.numpy())
    preds_pi.append(preds_pi_.numpy())
    weight_new.append(weight_new_.numpy())
    pars_new.append(pars_new_.numpy())
  
  preds_new_ = predict_full_location(features_time_temp, features_space_temp, depth_temp, max_depth_temp)
  
  sites_new.append(sites_temp)
  preds_new.append(preds_new_.numpy())
  


np.savez(f'./model_results/preds_val_{suffix}_{row_number:02d}.npz',
        locations = np.asarray(unique_locations),
        preds = np.asarray(preds_new),
        preds_ml = np.asarray(preds_ml),
        preds_pi = np.asarray(preds_pi),
        pinn_w = np.asarray(weight_new),
        pars = np.asarray(pars_new),
        n_depths_per_loc = n_depths)

with open(f'./model_results/preds_val_sites_{suffix}_{row_number:02d}.pkl', 'wb') as f:
  pickle.dump(np.asarray(sites_new), f) 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
