import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Flatten, Input, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split as train_test_split
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

def generate_data():
  # Generate synthetic data
  features, labels = make_blobs(
      n_samples=10000,
      centers=1,
      cluster_std=2.75,
      random_state=42
  )
  anomalies, labels = make_blobs(
      n_samples=5,
      centers=1,
      cluster_std=6.75,
      random_state=42,
      center_box=(30,40) 
  )
  dataset = np.concatenate((features, anomalies),axis=0)
  np.random.shuffle(dataset)
  truth = []
  for anomaly in anomalies:
    anomaly_idx = np.where(dataset == anomaly)
    truth.append(anomaly_idx)
  return truth, dataset
  
def build_model(data):
  shape = data.shape()
  #Autoencoder
  input_layer = Input(shape=(shape))
  layer = BatchNormalization()(input_layer)
  previous_layer = input_layer
  layer = Dense(64, activation='relu', name='Encoded')(previous_layer)
  layer = Dense(16, activation='relu', name='Encoded')(previous_layer)
  layer = Dense(4, activation='relu', name='Encoded')(previous_layer)
  layer = Dense(16, activation='relu', name='Encoded')(previous_layer)
  layer = Dense(32, activation='relu', name='Encoded')(previous_layer)
  layer = Dense(64, activation='relu', name='Encoded')(previous_layer)
  output_layer = Dense(shape, activation='linear')(previous_layer)
  #end Autoencoder
  return Model(inputs=input_layer, outputs=output_layer)

  
def train_model(data, validation_data, epochs=1000, batch_size=32, loss_func='mean_squared_error', learning_rate = 0.001,
                    shuffle=True, verbose=1, lr_factor=.5, lr_patience=10, monitor='val_loss'):
   model = build_model(data)
   optimizer = Adam(learning_rate=learning_rate)
   model.compile(loss=loss_func,optimizer=optimizer)
   reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=lr_factor, patience=lr_patience, verbose=1, min_lr=learning_rate**10, cooldown=3)
   early_stop = EarlyStopping(monitor=monitor, patience=lr_patience*2, verbose=1, restore_best_weights=True)
   history =  model.fit(data,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        verbose=verbose
   
                        
def model_inference(model, data, threshold):
  normalized_data = scalar.transform(data)
  predicted_data = model.predict(normalized_data)
  error = np.square(normalized_data-predicted_data).mean(axis=1)
  for idx, error_score in enumerate(error):
    if error_score > threshold or math.isnan(error_score):
        anomalies.append(original[idx])
        indices_containing_anomalies.append(idx)
  return indices_containing_anomalies
  
