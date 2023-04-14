# Tuning Neural Networks with Normalization - Lab 

## Introduction

In this lab you'll build a neural network to perform a regression task.

It is worth noting that getting regression to work with neural networks can be comparatively difficult because the output is unbounded ($\hat y$ can technically range from $-\infty$ to $+\infty$), and the models are especially prone to exploding gradients. This issue makes a regression exercise the perfect learning case for tinkering with normalization and optimization strategies to ensure proper convergence!

## Objectives

In this lab you will: 

- Fit a neural network to normalized data 
- Implement and observe the impact of various initialization techniques 
- Implement and observe the impact of various optimization techniques 

## Load the data 

First, run the following cell to import all the neccessary libraries and classes you will need in this lab. 


```python
# Necessary libraries and classes
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import initializers
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
```

In this lab, you'll be working with the housing prices data you saw in an earlier section. However, we did a lot of preprocessing for you so you can focus on normalizing numeric features and building neural network models! The following preprocessing steps were taken (all the code can be found in the `data_preprocessing.ipynb` notebook in this repository): 

- The data was split into the training, validate, and test sets 
- All the missing values in numeric columns were replaced by the median of those columns 
- All the missing values in catetgorical columns were replaced with the word 'missing' 
- All the categorical columns were one-hot encoded 

Run the following cells to import the train, validate, and test sets:  


```python
# Load all numeric features
X_train_numeric = pd.read_csv('data/X_train_numeric.csv')
X_test_numeric = pd.read_csv('data/X_test_numeric.csv')
X_val_numeric = pd.read_csv('data/X_val_numeric.csv')

# Load all categorical features
X_train_cat = pd.read_csv('data/X_train_cat.csv')
X_test_cat = pd.read_csv('data/X_test_cat.csv')
X_val_cat = pd.read_csv('data/X_val_cat.csv')

# Load all targets
y_train = pd.read_csv('data/y_train.csv')
y_test = pd.read_csv('data/y_test.csv')
y_val = pd.read_csv('data/y_val.csv')
```


```python
# Combine all features
X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
X_val = pd.concat([X_val_numeric, X_val_cat], axis=1)
X_test = pd.concat([X_test_numeric, X_test_cat], axis=1)
```

As a refresher, preview the training data: 


```python
# Preview the data
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>176</th>
      <th>177</th>
      <th>178</th>
      <th>179</th>
      <th>180</th>
      <th>181</th>
      <th>182</th>
      <th>183</th>
      <th>184</th>
      <th>185</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80.0</td>
      <td>69.0</td>
      <td>21453.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>1969.0</td>
      <td>1969.0</td>
      <td>0.0</td>
      <td>938.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60.0</td>
      <td>79.0</td>
      <td>12420.0</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>2001.0</td>
      <td>2001.0</td>
      <td>0.0</td>
      <td>666.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>75.0</td>
      <td>9742.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>2002.0</td>
      <td>2002.0</td>
      <td>281.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120.0</td>
      <td>39.0</td>
      <td>5389.0</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>1995.0</td>
      <td>1996.0</td>
      <td>0.0</td>
      <td>1180.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60.0</td>
      <td>85.0</td>
      <td>11003.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>2008.0</td>
      <td>2008.0</td>
      <td>160.0</td>
      <td>765.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 222 columns</p>
</div>



## Build a Baseline Model

Building a naive baseline model to compare performance against is a helpful reference point. From there, you can then observe the impact of various tunning procedures which will iteratively improve your model. So, let's do just that! 

In the cell below: 

- Add an input layer with `n_features` units 
- Add two hidden layers, one with 100 and the other with 50 units (make sure you use the `'relu'` activation function) 
- Add an output layer with 1 unit and `'linear'` activation 
- Compile and fit the model 


```python
n_features = (X_train.shape[1],)

print(n_features)
```

    (222,)



```python
np.random.seed(123)
baseline_model = Sequential()

# Hidden layer with 100 units
baseline_model.add(layers.Dense(100, activation='relu', input_shape=n_features))

# Hidden layer with 50 units
baseline_model.add(layers.Dense(50, activation='relu'))

# Output layer
baseline_model.add(layers.Dense(1, activation='linear'))

# Compile the model
baseline_model.compile(optimizer='SGD', 
                       loss='mse', 
                       metrics=['mse'])

# Train the model
baseline_model.fit(X_train, 
                   y_train, 
                   batch_size=32, 
                   epochs=150,
                   validation_data=(X_val, y_val))
```

    Metal device set to: Apple M1 Pro
    Epoch 1/150


    2023-04-14 18:33:37.652863: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2023-04-14 18:33:37.652988: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
    2023-04-14 18:33:37.784944: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz


    11/33 [=========>....................] - ETA: 0s - loss: nan - mse: nan                          

    2023-04-14 18:33:37.882642: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 1s 11ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 2/150
    21/33 [==================>...........] - ETA: 0s - loss: nan - mse: nan

    2023-04-14 18:33:38.276389: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 3/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 4/150
    33/33 [==============================] - 0s 7ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 5/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 6/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 7/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 8/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 9/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 10/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 11/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 12/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 13/150
    33/33 [==============================] - 0s 7ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 14/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 15/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 16/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 17/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 18/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 19/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 20/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 21/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 22/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 23/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 24/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 25/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 26/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 27/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 28/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 29/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 30/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 31/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 32/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 33/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 34/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 35/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 36/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 37/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 38/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 39/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 40/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 41/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 42/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 43/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 44/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 45/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 46/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 47/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 48/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 49/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 50/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 51/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 52/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 53/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 54/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 55/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 56/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 57/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 58/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 59/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 64/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 65/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 66/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 67/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 68/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 69/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 70/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 71/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 72/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 73/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 74/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 75/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 76/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 77/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 78/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 79/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 80/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 81/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 82/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 83/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 84/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 85/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 86/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 87/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 88/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 89/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 90/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 91/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 92/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 95/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 96/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 97/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 98/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 99/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 100/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 101/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 102/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 103/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 104/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 105/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 106/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 107/150
    33/33 [==============================] - 0s 7ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 108/150
    33/33 [==============================] - 0s 7ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 109/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 110/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 111/150
    33/33 [==============================] - 0s 7ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 112/150
    33/33 [==============================] - 0s 7ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 113/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 114/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 115/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 116/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 117/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 118/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 119/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 120/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 121/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 122/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 123/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 124/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 125/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 126/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 127/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 128/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 129/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 130/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 131/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 132/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 133/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 134/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 138/150
    33/33 [==============================] - 0s 7ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 139/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 140/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 141/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 142/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 143/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 144/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 145/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 146/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 147/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 148/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 149/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan





    <keras.callbacks.History at 0x13beab460>



> _**Notice this extremely problematic behavior: all the values for training and validation loss are "nan". This indicates that the algorithm did not converge. The first solution to this is to normalize the input. From there, if convergence is not achieved, normalizing the output may also be required.**_ 

## Normalize the Input Data 

It's now time to normalize the input data. In the cell below: 

- Assign the column names of all numeric columns to `numeric_columns` 
- Instantiate a `StandardScaler` 
- Fit and transform `X_train_numeric`. Make sure you convert the result into a DataFrame (use `numeric_columns` as the column names) 
- Transform validate and test sets (`X_val_numeric` and `X_test_numeric`), and convert these results into DataFrames as well 
- Use the provided to combine the scaled numerical and categorical features 


```python
# Numeric column names
numeric_columns = X_train_numeric.columns 

# Instantiate StandardScaler
ss_X = StandardScaler()

# Fit and transform train data
X_train_scaled = pd.DataFrame(ss_X.fit_transform(X_train_numeric), columns=numeric_columns)

# Transform validate and test data
X_val_scaled = pd.DataFrame(ss_X.transform(X_val_numeric), columns=numeric_columns)
X_test_scaled = pd.DataFrame(ss_X.transform(X_test_numeric), columns=numeric_columns)

# Combine the scaled numerical features and categorical features
X_train = pd.concat([X_train_scaled, X_train_cat], axis=1)
X_val = pd.concat([X_val_scaled, X_val_cat], axis=1)
X_test = pd.concat([X_test_scaled, X_test_cat], axis=1)
```

Now run the following cell to compile a neural network model (with the same architecture as before): 


```python
# Model with all normalized inputs
np.random.seed(123)
normalized_input_model = Sequential()
normalized_input_model.add(layers.Dense(100, activation='relu', input_shape=n_features))
normalized_input_model.add(layers.Dense(50, activation='relu'))
normalized_input_model.add(layers.Dense(1, activation='linear'))

# Compile the model
normalized_input_model.compile(optimizer='SGD', 
                               loss='mse', 
                               metrics=['mse'])
```

In the cell below: 
- Train the `normalized_input_model` on normalized input (`X_train`) and output (`y_train`) 
- Set a batch size of 32 and train for 150 epochs 
- Specify the `validation_data` argument as `(X_val, y_val)` 


```python
# Train the model
normalized_input_model.fit(X_train,  
                           y_train, 
                           batch_size=32, 
                           epochs=150,
                           validation_data=(X_val, y_val))
```

    Epoch 1/150
    24/33 [====================>.........] - ETA: 0s - loss: nan - mse: nan                         

    2023-04-14 18:34:06.738799: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 9ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 2/150
    25/33 [=====================>........] - ETA: 0s - loss: nan - mse: nan

    2023-04-14 18:34:07.021691: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 4/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 5/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 6/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 7/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 8/150
    33/33 [==============================] - 0s 7ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 9/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 10/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 11/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 12/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 13/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 14/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 15/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 16/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 17/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 18/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 19/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 20/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 21/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 22/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 23/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 24/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 25/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 26/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 27/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 28/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 29/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 30/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 31/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 32/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 33/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 34/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 36/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 38/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 39/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 40/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 41/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 42/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 43/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 44/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 45/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 46/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 47/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 48/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 49/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 50/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 51/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 52/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 53/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 54/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 55/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 56/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 58/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 59/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 61/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 62/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 63/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 64/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 65/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 66/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 67/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 68/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 69/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 70/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 71/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 72/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 73/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 74/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 75/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 76/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 77/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 78/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 79/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 80/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 81/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 82/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 83/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 84/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 85/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 87/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 88/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 89/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 90/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 91/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 92/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 95/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 96/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 97/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 98/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 99/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 100/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 101/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 102/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 103/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 104/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 105/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 106/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 107/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 108/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 109/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 110/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 111/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 112/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 113/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 114/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 115/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 116/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 117/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 118/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 119/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 120/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 121/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 122/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 123/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 124/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 125/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 126/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 127/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 128/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 129/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 130/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 131/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 132/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 133/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 134/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 135/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 137/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 138/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 139/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 140/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 141/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 142/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 143/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 144/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 145/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 146/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 147/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 148/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 149/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan





    <keras.callbacks.History at 0x14b39ceb0>



> _**Note that you still haven't achieved convergence! From here, it's time to normalize the output data.**_

## Normalizing the output

Again, use `StandardScaler()` to: 

- Fit and transform `y_train` 
- Transform `y_val` and `y_test` 


```python
# Instantiate StandardScaler
ss_y = StandardScaler()

# Fit and transform train labels
y_train_scaled = ss_y.fit_transform(y_train)

# Transform validate and test labels
y_val_scaled = ss_y.transform(y_val)
y_test_scaled = ss_y.transform(y_test)
```

In the cell below: 
- Train the `normalized_model` on normalized input (`X_train`) and output (`y_train_scaled`) 
- Set a batch size of 32 and train for 150 epochs 
- Specify the `validation_data` as `(X_val, y_val_scaled)` 


```python
# Model with all normalized inputs and outputs
np.random.seed(123)
normalized_model = Sequential()
normalized_model.add(layers.Dense(100, activation='relu', input_shape=n_features))
normalized_model.add(layers.Dense(50, activation='relu'))
normalized_model.add(layers.Dense(1, activation='linear'))

# Compile the model
normalized_model.compile(optimizer='SGD', 
                         loss='mse', 
                         metrics=['mse']) 

# Train the model
normalized_model.fit(X_train, 
                     y_train_scaled, 
                     batch_size=32, 
                     epochs=150,
                     validation_data=(X_val, y_val))
```

    Epoch 1/150
    24/33 [====================>.........] - ETA: 0s - loss: 0.5549 - mse: 0.5549

    2023-04-14 18:34:34.584020: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 9ms/step - loss: 0.4699 - mse: 0.4699 - val_loss: 36093075456.0000 - val_mse: 36093075456.0000
    Epoch 2/150
    25/33 [=====================>........] - ETA: 0s - loss: 0.2978 - mse: 0.2978

    2023-04-14 18:34:34.868520: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 5ms/step - loss: 0.2649 - mse: 0.2649 - val_loss: 36093042688.0000 - val_mse: 36093042688.0000
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1990 - mse: 0.1990 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 4/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1719 - mse: 0.1719 - val_loss: 36093054976.0000 - val_mse: 36093054976.0000
    Epoch 5/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1535 - mse: 0.1535 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 6/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1312 - mse: 0.1312 - val_loss: 36093046784.0000 - val_mse: 36093046784.0000
    Epoch 7/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1308 - mse: 0.1308 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 8/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1182 - mse: 0.1182 - val_loss: 36093046784.0000 - val_mse: 36093046784.0000
    Epoch 9/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1102 - mse: 0.1102 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 10/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1002 - mse: 0.1002 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 11/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0949 - mse: 0.0949 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 12/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0911 - mse: 0.0911 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 13/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0872 - mse: 0.0872 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 14/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0810 - mse: 0.0810 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 15/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0784 - mse: 0.0784 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 16/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0764 - mse: 0.0764 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 17/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0707 - mse: 0.0707 - val_loss: 36093075456.0000 - val_mse: 36093075456.0000
    Epoch 18/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0700 - mse: 0.0700 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 19/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0665 - mse: 0.0665 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 20/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0637 - mse: 0.0637 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 21/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0610 - mse: 0.0610 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 22/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0587 - mse: 0.0587 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 23/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0575 - mse: 0.0575 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 24/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0557 - mse: 0.0557 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 25/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0547 - mse: 0.0547 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 26/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0522 - mse: 0.0522 - val_loss: 36093075456.0000 - val_mse: 36093075456.0000
    Epoch 27/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0512 - mse: 0.0512 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 28/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0497 - mse: 0.0497 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 29/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0490 - mse: 0.0490 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 30/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0475 - mse: 0.0475 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 31/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0466 - mse: 0.0466 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 32/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0452 - mse: 0.0452 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 33/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0434 - mse: 0.0434 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 34/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0427 - mse: 0.0427 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0413 - mse: 0.0413 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 36/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0407 - mse: 0.0407 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0398 - mse: 0.0398 - val_loss: 36093046784.0000 - val_mse: 36093046784.0000
    Epoch 38/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0389 - mse: 0.0389 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 39/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0380 - mse: 0.0380 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 40/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0371 - mse: 0.0371 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 41/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0357 - mse: 0.0357 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 42/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0362 - mse: 0.0362 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 43/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0351 - mse: 0.0351 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 44/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0343 - mse: 0.0343 - val_loss: 36093042688.0000 - val_mse: 36093042688.0000
    Epoch 45/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0338 - mse: 0.0338 - val_loss: 36093054976.0000 - val_mse: 36093054976.0000
    Epoch 46/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0324 - mse: 0.0324 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 47/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0324 - mse: 0.0324 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 48/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0311 - mse: 0.0311 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 49/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0305 - mse: 0.0305 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 50/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0306 - mse: 0.0306 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 51/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0299 - mse: 0.0299 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 52/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0292 - mse: 0.0292 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 53/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0284 - mse: 0.0284 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 54/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0276 - mse: 0.0276 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 55/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0273 - mse: 0.0273 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 56/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0268 - mse: 0.0268 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0262 - mse: 0.0262 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 58/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0257 - mse: 0.0257 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 59/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0251 - mse: 0.0251 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0245 - mse: 0.0245 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0241 - mse: 0.0241 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0239 - mse: 0.0239 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0237 - mse: 0.0237 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 64/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0229 - mse: 0.0229 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 65/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0225 - mse: 0.0225 - val_loss: 36093054976.0000 - val_mse: 36093054976.0000
    Epoch 66/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0226 - mse: 0.0226 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 67/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0220 - mse: 0.0220 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 68/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0217 - mse: 0.0217 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 69/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0211 - mse: 0.0211 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 70/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0208 - mse: 0.0208 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 71/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0205 - mse: 0.0205 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 72/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0202 - mse: 0.0202 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 73/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0197 - mse: 0.0197 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 74/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0201 - mse: 0.0201 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 75/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0195 - mse: 0.0195 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 76/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0189 - mse: 0.0189 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 77/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0189 - mse: 0.0189 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 78/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0186 - mse: 0.0186 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 79/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0183 - mse: 0.0183 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 80/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0179 - mse: 0.0179 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 81/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0178 - mse: 0.0178 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 82/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0173 - mse: 0.0173 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 83/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0173 - mse: 0.0173 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 84/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0171 - mse: 0.0171 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 85/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0168 - mse: 0.0168 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0166 - mse: 0.0166 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 87/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0163 - mse: 0.0163 - val_loss: 36093050880.0000 - val_mse: 36093050880.0000
    Epoch 88/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0164 - mse: 0.0164 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 89/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0158 - mse: 0.0158 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 90/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0157 - mse: 0.0157 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 91/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0156 - mse: 0.0156 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 92/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0154 - mse: 0.0154 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0151 - mse: 0.0151 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0150 - mse: 0.0150 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 95/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0146 - mse: 0.0146 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 96/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0146 - mse: 0.0146 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 97/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0144 - mse: 0.0144 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 98/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0142 - mse: 0.0142 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 99/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0140 - mse: 0.0140 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 100/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0140 - mse: 0.0140 - val_loss: 36093054976.0000 - val_mse: 36093054976.0000
    Epoch 101/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0139 - mse: 0.0139 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 102/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0134 - mse: 0.0134 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 103/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0134 - mse: 0.0134 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 104/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0133 - mse: 0.0133 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 105/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0131 - mse: 0.0131 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 106/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0130 - mse: 0.0130 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 107/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0126 - mse: 0.0126 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 108/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0126 - mse: 0.0126 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 109/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0124 - mse: 0.0124 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 110/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0122 - mse: 0.0122 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 111/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0121 - mse: 0.0121 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 112/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0119 - mse: 0.0119 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 113/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 114/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0117 - mse: 0.0117 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 115/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0115 - mse: 0.0115 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 116/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0117 - mse: 0.0117 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 117/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 118/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 119/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 120/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0109 - mse: 0.0109 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 121/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0109 - mse: 0.0109 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 122/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 123/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 124/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0105 - mse: 0.0105 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 125/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0104 - mse: 0.0104 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 126/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0103 - mse: 0.0103 - val_loss: 36093075456.0000 - val_mse: 36093075456.0000
    Epoch 127/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0102 - mse: 0.0102 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 128/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0100 - mse: 0.0100 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 129/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 130/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0101 - mse: 0.0101 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 131/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 132/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0098 - mse: 0.0098 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 133/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0096 - mse: 0.0096 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 134/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0095 - mse: 0.0095 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0092 - mse: 0.0092 - val_loss: 36093079552.0000 - val_mse: 36093079552.0000
    Epoch 138/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0091 - mse: 0.0091 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 139/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0091 - mse: 0.0091 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 140/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 141/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 142/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0088 - mse: 0.0088 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 143/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0088 - mse: 0.0088 - val_loss: 36093067264.0000 - val_mse: 36093067264.0000
    Epoch 144/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0086 - mse: 0.0086 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 145/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0083 - mse: 0.0083 - val_loss: 36093059072.0000 - val_mse: 36093059072.0000
    Epoch 146/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0084 - mse: 0.0084 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 147/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0083 - mse: 0.0083 - val_loss: 36093063168.0000 - val_mse: 36093063168.0000
    Epoch 148/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000
    Epoch 149/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 36093075456.0000 - val_mse: 36093075456.0000
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0081 - mse: 0.0081 - val_loss: 36093071360.0000 - val_mse: 36093071360.0000





    <keras.callbacks.History at 0x14d7d5700>



Nicely done! After normalizing both the input and output, the model finally converged. 

- Evaluate the model (`normalized_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
normalized_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 4ms/step - loss: 0.0077 - mse: 0.0077


    2023-04-14 18:35:01.752247: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.





    [0.007683976087719202, 0.007683976087719202]



- Evaluate the model (`normalized_model`) on validate data (`X_val` and `y_val_scaled`) 

Since the output is normalized, the metric above is not interpretable. To remedy this: 

- Generate predictions on validate data (`X_val`) 
- Transform these predictions back to original scale using `ss_y` 
- Now you can calculate the RMSE in the original units with `y_val` and `y_val_pred` 


```python
# Generate predictions on validate data
y_val_pred_scaled = normalized_model.predict(X_val)

# Transform the predictions back to original scale
y_val_pred = ss_y.inverse_transform(y_val_pred_scaled)

# RMSE of validate data
np.sqrt(mean_squared_error(y_val, y_val_pred))
```

    2023-04-14 18:35:01.949149: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    9/9 [==============================] - 0s 3ms/step





    28123.691471632756



Great! Now that you have a converged model, you can also experiment with alternative optimizers and initialization strategies to see if you can find a better global minimum. (After all, the current models may have converged to a local minimum.) 

## Using Weight Initializers

In this section you will to use alternative initialization and optimization strategies. At the end, you'll then be asked to select the model which you believe performs the best.  

##  He Initialization

In the cell below, sepcify the following in the first hidden layer:  
  - 100 units 
  - `'relu'` activation 
  - `input_shape` 
  - `kernel_initializer='he_normal'`  


```python
np.random.seed(123)
he_model = Sequential()

# Add the first hidden layer
he_model.add(layers.Dense(100, kernel_initializer='he_normal', activation='relu', input_shape=n_features))

# Add another hidden layer
he_model.add(layers.Dense(50, activation='relu'))

# Add an output layer
he_model.add(layers.Dense(1, activation='linear'))

# Compile the model
he_model.compile(optimizer='SGD', 
                 loss='mse', 
                 metrics=['mse'])

# Train the model
he_model.fit(X_train, 
             y_train_scaled, 
             batch_size=32, 
             epochs=150, 
             validation_data=(X_val, y_val_scaled))
```

    Epoch 1/150


    2023-04-14 18:35:02.098586: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 9ms/step - loss: 0.5188 - mse: 0.5188 - val_loss: 0.3186 - val_mse: 0.3186
    Epoch 2/150
    22/33 [===================>..........] - ETA: 0s - loss: 0.2668 - mse: 0.2668

    2023-04-14 18:35:02.379081: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 6ms/step - loss: 0.2756 - mse: 0.2756 - val_loss: 0.1981 - val_mse: 0.1981
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.2248 - mse: 0.2248 - val_loss: 0.1703 - val_mse: 0.1703
    Epoch 4/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1884 - mse: 0.1884 - val_loss: 0.1575 - val_mse: 0.1575
    Epoch 5/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1721 - mse: 0.1721 - val_loss: 0.1477 - val_mse: 0.1477
    Epoch 6/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1590 - mse: 0.1590 - val_loss: 0.1417 - val_mse: 0.1417
    Epoch 7/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1484 - mse: 0.1484 - val_loss: 0.1515 - val_mse: 0.1515
    Epoch 8/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1400 - mse: 0.1400 - val_loss: 0.1394 - val_mse: 0.1394
    Epoch 9/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1274 - mse: 0.1274 - val_loss: 0.1390 - val_mse: 0.1390
    Epoch 10/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1237 - mse: 0.1237 - val_loss: 0.1318 - val_mse: 0.1318
    Epoch 11/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1139 - mse: 0.1139 - val_loss: 0.1282 - val_mse: 0.1282
    Epoch 12/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1066 - mse: 0.1066 - val_loss: 0.1351 - val_mse: 0.1351
    Epoch 13/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1011 - mse: 0.1011 - val_loss: 0.1294 - val_mse: 0.1294
    Epoch 14/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1004 - mse: 0.1004 - val_loss: 0.1260 - val_mse: 0.1260
    Epoch 15/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0932 - mse: 0.0932 - val_loss: 0.1274 - val_mse: 0.1274
    Epoch 16/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0911 - mse: 0.0911 - val_loss: 0.1359 - val_mse: 0.1359
    Epoch 17/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0870 - mse: 0.0870 - val_loss: 0.1251 - val_mse: 0.1251
    Epoch 18/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0823 - mse: 0.0823 - val_loss: 0.1249 - val_mse: 0.1249
    Epoch 19/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0773 - mse: 0.0773 - val_loss: 0.1268 - val_mse: 0.1268
    Epoch 20/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0764 - mse: 0.0764 - val_loss: 0.1277 - val_mse: 0.1277
    Epoch 21/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0721 - mse: 0.0721 - val_loss: 0.1263 - val_mse: 0.1263
    Epoch 22/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0698 - mse: 0.0698 - val_loss: 0.1237 - val_mse: 0.1237
    Epoch 23/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0665 - mse: 0.0665 - val_loss: 0.1270 - val_mse: 0.1270
    Epoch 24/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0647 - mse: 0.0647 - val_loss: 0.1246 - val_mse: 0.1246
    Epoch 25/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0623 - mse: 0.0623 - val_loss: 0.1249 - val_mse: 0.1249
    Epoch 26/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0597 - mse: 0.0597 - val_loss: 0.1269 - val_mse: 0.1269
    Epoch 27/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0589 - mse: 0.0589 - val_loss: 0.1260 - val_mse: 0.1260
    Epoch 28/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0565 - mse: 0.0565 - val_loss: 0.1269 - val_mse: 0.1269
    Epoch 29/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0538 - mse: 0.0538 - val_loss: 0.1288 - val_mse: 0.1288
    Epoch 30/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0536 - mse: 0.0536 - val_loss: 0.1276 - val_mse: 0.1276
    Epoch 31/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0521 - mse: 0.0521 - val_loss: 0.1261 - val_mse: 0.1261
    Epoch 32/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0497 - mse: 0.0497 - val_loss: 0.1279 - val_mse: 0.1279
    Epoch 33/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0494 - mse: 0.0494 - val_loss: 0.1276 - val_mse: 0.1276
    Epoch 34/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0477 - mse: 0.0477 - val_loss: 0.1261 - val_mse: 0.1261
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0463 - mse: 0.0463 - val_loss: 0.1278 - val_mse: 0.1278
    Epoch 36/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0446 - mse: 0.0446 - val_loss: 0.1244 - val_mse: 0.1244
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0431 - mse: 0.0431 - val_loss: 0.1276 - val_mse: 0.1276
    Epoch 38/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0423 - mse: 0.0423 - val_loss: 0.1276 - val_mse: 0.1276
    Epoch 39/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0422 - mse: 0.0422 - val_loss: 0.1254 - val_mse: 0.1254
    Epoch 40/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0414 - mse: 0.0414 - val_loss: 0.1264 - val_mse: 0.1264
    Epoch 41/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0399 - mse: 0.0399 - val_loss: 0.1261 - val_mse: 0.1261
    Epoch 42/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0395 - mse: 0.0395 - val_loss: 0.1259 - val_mse: 0.1259
    Epoch 43/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0384 - mse: 0.0384 - val_loss: 0.1241 - val_mse: 0.1241
    Epoch 44/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0375 - mse: 0.0375 - val_loss: 0.1255 - val_mse: 0.1255
    Epoch 45/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0362 - mse: 0.0362 - val_loss: 0.1239 - val_mse: 0.1239
    Epoch 46/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0353 - mse: 0.0353 - val_loss: 0.1232 - val_mse: 0.1232
    Epoch 47/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0344 - mse: 0.0344 - val_loss: 0.1219 - val_mse: 0.1219
    Epoch 48/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0343 - mse: 0.0343 - val_loss: 0.1220 - val_mse: 0.1220
    Epoch 49/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0339 - mse: 0.0339 - val_loss: 0.1239 - val_mse: 0.1239
    Epoch 50/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0324 - mse: 0.0324 - val_loss: 0.1235 - val_mse: 0.1235
    Epoch 51/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0315 - mse: 0.0315 - val_loss: 0.1242 - val_mse: 0.1242
    Epoch 52/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0313 - mse: 0.0313 - val_loss: 0.1229 - val_mse: 0.1229
    Epoch 53/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0310 - mse: 0.0310 - val_loss: 0.1235 - val_mse: 0.1235
    Epoch 54/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0303 - mse: 0.0303 - val_loss: 0.1253 - val_mse: 0.1253
    Epoch 55/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0297 - mse: 0.0297 - val_loss: 0.1204 - val_mse: 0.1204
    Epoch 56/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0288 - mse: 0.0288 - val_loss: 0.1199 - val_mse: 0.1199
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0289 - mse: 0.0289 - val_loss: 0.1225 - val_mse: 0.1225
    Epoch 58/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0281 - mse: 0.0281 - val_loss: 0.1239 - val_mse: 0.1239
    Epoch 59/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0278 - mse: 0.0278 - val_loss: 0.1243 - val_mse: 0.1243
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0272 - mse: 0.0272 - val_loss: 0.1219 - val_mse: 0.1219
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0266 - mse: 0.0266 - val_loss: 0.1249 - val_mse: 0.1249
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0260 - mse: 0.0260 - val_loss: 0.1204 - val_mse: 0.1204
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0262 - mse: 0.0262 - val_loss: 0.1200 - val_mse: 0.1200
    Epoch 64/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0250 - mse: 0.0250 - val_loss: 0.1201 - val_mse: 0.1201
    Epoch 65/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0253 - mse: 0.0253 - val_loss: 0.1205 - val_mse: 0.1205
    Epoch 66/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0251 - mse: 0.0251 - val_loss: 0.1211 - val_mse: 0.1211
    Epoch 67/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0245 - mse: 0.0245 - val_loss: 0.1213 - val_mse: 0.1213
    Epoch 68/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0239 - mse: 0.0239 - val_loss: 0.1207 - val_mse: 0.1207
    Epoch 69/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0239 - mse: 0.0239 - val_loss: 0.1188 - val_mse: 0.1188
    Epoch 70/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0233 - mse: 0.0233 - val_loss: 0.1198 - val_mse: 0.1198
    Epoch 71/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0223 - mse: 0.0223 - val_loss: 0.1206 - val_mse: 0.1206
    Epoch 72/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0225 - mse: 0.0225 - val_loss: 0.1197 - val_mse: 0.1197
    Epoch 73/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0220 - mse: 0.0220 - val_loss: 0.1188 - val_mse: 0.1188
    Epoch 74/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0225 - mse: 0.0225 - val_loss: 0.1191 - val_mse: 0.1191
    Epoch 75/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0217 - mse: 0.0217 - val_loss: 0.1204 - val_mse: 0.1204
    Epoch 76/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0215 - mse: 0.0215 - val_loss: 0.1212 - val_mse: 0.1212
    Epoch 77/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0211 - mse: 0.0211 - val_loss: 0.1203 - val_mse: 0.1203
    Epoch 78/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0208 - mse: 0.0208 - val_loss: 0.1208 - val_mse: 0.1208
    Epoch 79/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0207 - mse: 0.0207 - val_loss: 0.1196 - val_mse: 0.1196
    Epoch 80/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0201 - mse: 0.0201 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 81/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0201 - mse: 0.0201 - val_loss: 0.1187 - val_mse: 0.1187
    Epoch 82/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0195 - mse: 0.0195 - val_loss: 0.1190 - val_mse: 0.1190
    Epoch 83/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0194 - mse: 0.0194 - val_loss: 0.1195 - val_mse: 0.1195
    Epoch 84/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0192 - mse: 0.0192 - val_loss: 0.1185 - val_mse: 0.1185
    Epoch 85/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0189 - mse: 0.0189 - val_loss: 0.1181 - val_mse: 0.1181
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0185 - mse: 0.0185 - val_loss: 0.1195 - val_mse: 0.1195
    Epoch 87/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0184 - mse: 0.0184 - val_loss: 0.1177 - val_mse: 0.1177
    Epoch 88/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0181 - mse: 0.0181 - val_loss: 0.1196 - val_mse: 0.1196
    Epoch 89/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0180 - mse: 0.0180 - val_loss: 0.1182 - val_mse: 0.1182
    Epoch 90/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0174 - mse: 0.0174 - val_loss: 0.1188 - val_mse: 0.1188
    Epoch 91/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0178 - mse: 0.0178 - val_loss: 0.1175 - val_mse: 0.1175
    Epoch 92/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0172 - mse: 0.0172 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0172 - mse: 0.0172 - val_loss: 0.1184 - val_mse: 0.1184
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0169 - mse: 0.0169 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 95/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0166 - mse: 0.0166 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 96/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0164 - mse: 0.0164 - val_loss: 0.1184 - val_mse: 0.1184
    Epoch 97/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0163 - mse: 0.0163 - val_loss: 0.1184 - val_mse: 0.1184
    Epoch 98/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0161 - mse: 0.0161 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 99/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0159 - mse: 0.0159 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 100/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0159 - mse: 0.0159 - val_loss: 0.1190 - val_mse: 0.1190
    Epoch 101/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0157 - mse: 0.0157 - val_loss: 0.1180 - val_mse: 0.1180
    Epoch 102/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0154 - mse: 0.0154 - val_loss: 0.1183 - val_mse: 0.1183
    Epoch 103/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0154 - mse: 0.0154 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 104/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0149 - mse: 0.0149 - val_loss: 0.1169 - val_mse: 0.1169
    Epoch 105/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0148 - mse: 0.0148 - val_loss: 0.1172 - val_mse: 0.1172
    Epoch 106/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0147 - mse: 0.0147 - val_loss: 0.1181 - val_mse: 0.1181
    Epoch 107/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0145 - mse: 0.0145 - val_loss: 0.1174 - val_mse: 0.1174
    Epoch 108/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0145 - mse: 0.0145 - val_loss: 0.1165 - val_mse: 0.1165
    Epoch 109/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0144 - mse: 0.0144 - val_loss: 0.1177 - val_mse: 0.1177
    Epoch 110/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0139 - mse: 0.0139 - val_loss: 0.1169 - val_mse: 0.1169
    Epoch 111/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0136 - mse: 0.0136 - val_loss: 0.1177 - val_mse: 0.1177
    Epoch 112/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0136 - mse: 0.0136 - val_loss: 0.1177 - val_mse: 0.1177
    Epoch 113/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0136 - mse: 0.0136 - val_loss: 0.1165 - val_mse: 0.1165
    Epoch 114/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0137 - mse: 0.0137 - val_loss: 0.1159 - val_mse: 0.1159
    Epoch 115/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0133 - mse: 0.0133 - val_loss: 0.1173 - val_mse: 0.1173
    Epoch 116/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0131 - mse: 0.0131 - val_loss: 0.1160 - val_mse: 0.1160
    Epoch 117/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0132 - mse: 0.0132 - val_loss: 0.1189 - val_mse: 0.1189
    Epoch 118/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0127 - mse: 0.0127 - val_loss: 0.1173 - val_mse: 0.1173
    Epoch 119/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0126 - mse: 0.0126 - val_loss: 0.1164 - val_mse: 0.1164
    Epoch 120/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0124 - mse: 0.0124 - val_loss: 0.1159 - val_mse: 0.1159
    Epoch 121/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0125 - mse: 0.0125 - val_loss: 0.1173 - val_mse: 0.1173
    Epoch 122/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0123 - mse: 0.0123 - val_loss: 0.1165 - val_mse: 0.1165
    Epoch 123/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0121 - mse: 0.0121 - val_loss: 0.1180 - val_mse: 0.1180
    Epoch 124/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 0.1173 - val_mse: 0.1173
    Epoch 125/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0117 - mse: 0.0117 - val_loss: 0.1176 - val_mse: 0.1176
    Epoch 126/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 0.1172 - val_mse: 0.1172
    Epoch 127/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0118 - mse: 0.0118 - val_loss: 0.1164 - val_mse: 0.1164
    Epoch 128/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0115 - mse: 0.0115 - val_loss: 0.1161 - val_mse: 0.1161
    Epoch 129/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 0.1166 - val_mse: 0.1166
    Epoch 130/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.1177 - val_mse: 0.1177
    Epoch 131/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.1171 - val_mse: 0.1171
    Epoch 132/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1165 - val_mse: 0.1165
    Epoch 133/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0109 - mse: 0.0109 - val_loss: 0.1171 - val_mse: 0.1171
    Epoch 134/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0108 - mse: 0.0108 - val_loss: 0.1181 - val_mse: 0.1181
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 0.1167 - val_mse: 0.1167
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0106 - mse: 0.0106 - val_loss: 0.1168 - val_mse: 0.1168
    Epoch 138/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0105 - mse: 0.0105 - val_loss: 0.1176 - val_mse: 0.1176
    Epoch 139/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0106 - mse: 0.0106 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 140/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0104 - mse: 0.0104 - val_loss: 0.1168 - val_mse: 0.1168
    Epoch 141/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0102 - mse: 0.0102 - val_loss: 0.1165 - val_mse: 0.1165
    Epoch 142/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0101 - mse: 0.0101 - val_loss: 0.1164 - val_mse: 0.1164
    Epoch 143/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0101 - mse: 0.0101 - val_loss: 0.1170 - val_mse: 0.1170
    Epoch 144/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1177 - val_mse: 0.1177
    Epoch 145/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0098 - mse: 0.0098 - val_loss: 0.1174 - val_mse: 0.1174
    Epoch 146/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0098 - mse: 0.0098 - val_loss: 0.1164 - val_mse: 0.1164
    Epoch 147/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0097 - mse: 0.0097 - val_loss: 0.1171 - val_mse: 0.1171
    Epoch 148/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0097 - mse: 0.0097 - val_loss: 0.1160 - val_mse: 0.1160
    Epoch 149/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0095 - mse: 0.0095 - val_loss: 0.1173 - val_mse: 0.1173
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 0.1166 - val_mse: 0.1166





    <keras.callbacks.History at 0x15bc43a90>



Evaluate the model (`he_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
he_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 4ms/step - loss: 0.0090 - mse: 0.0090





    [0.008950877003371716, 0.008950877003371716]



Evaluate the model (`he_model`) on validate data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data
he_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 3ms/step - loss: 0.1166 - mse: 0.1166





    [0.11658976972103119, 0.11658976972103119]



## Lecun Initialization 

In the cell below, sepcify the following in the first hidden layer:  
  - 100 units 
  - `'relu'` activation 
  - `input_shape` 
  - `kernel_initializer='lecun_normal'` 


```python
np.random.seed(123)
lecun_model = Sequential()

# Add the first hidden layer
lecun_model.add(layers.Dense(100, kernel_initializer='lecun_normal', activation='relu', input_shape=n_features))

# Add another hidden layer
lecun_model.add(layers.Dense(50, activation='relu'))

# Add an output layer
lecun_model.add(layers.Dense(1, activation='linear'))

# Compile the model
lecun_model.compile(optimizer='SGD', 
                    loss='mse', 
                    metrics=['mse'])

# Train the model
lecun_model.fit(X_train, 
                y_train_scaled, 
                batch_size=32, 
                epochs=150, 
                validation_data=(X_val, y_val_scaled))
```

    Epoch 1/150
    24/33 [====================>.........] - ETA: 0s - loss: 0.5655 - mse: 0.5655

    2023-04-14 18:35:29.370135: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 9ms/step - loss: 0.4588 - mse: 0.4588 - val_loss: 0.1934 - val_mse: 0.1934
    Epoch 2/150
    25/33 [=====================>........] - ETA: 0s - loss: 0.2342 - mse: 0.2342

    2023-04-14 18:35:29.650954: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 6ms/step - loss: 0.2253 - mse: 0.2253 - val_loss: 0.1639 - val_mse: 0.1639
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1780 - mse: 0.1780 - val_loss: 0.1459 - val_mse: 0.1459
    Epoch 4/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1511 - mse: 0.1511 - val_loss: 0.1429 - val_mse: 0.1429
    Epoch 5/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1360 - mse: 0.1360 - val_loss: 0.1315 - val_mse: 0.1315
    Epoch 6/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1260 - mse: 0.1260 - val_loss: 0.1341 - val_mse: 0.1341
    Epoch 7/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1134 - mse: 0.1134 - val_loss: 0.1316 - val_mse: 0.1316
    Epoch 8/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1043 - mse: 0.1043 - val_loss: 0.1248 - val_mse: 0.1248
    Epoch 9/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0978 - mse: 0.0978 - val_loss: 0.1281 - val_mse: 0.1281
    Epoch 10/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0949 - mse: 0.0949 - val_loss: 0.1261 - val_mse: 0.1261
    Epoch 11/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0864 - mse: 0.0864 - val_loss: 0.1270 - val_mse: 0.1270
    Epoch 12/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0799 - mse: 0.0799 - val_loss: 0.1248 - val_mse: 0.1248
    Epoch 13/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0754 - mse: 0.0754 - val_loss: 0.1224 - val_mse: 0.1224
    Epoch 14/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0724 - mse: 0.0724 - val_loss: 0.1270 - val_mse: 0.1270
    Epoch 15/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0697 - mse: 0.0697 - val_loss: 0.1281 - val_mse: 0.1281
    Epoch 16/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0653 - mse: 0.0653 - val_loss: 0.1308 - val_mse: 0.1308
    Epoch 17/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0626 - mse: 0.0626 - val_loss: 0.1281 - val_mse: 0.1281
    Epoch 18/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0578 - mse: 0.0578 - val_loss: 0.1322 - val_mse: 0.1322
    Epoch 19/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0591 - mse: 0.0591 - val_loss: 0.1291 - val_mse: 0.1291
    Epoch 20/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0547 - mse: 0.0547 - val_loss: 0.1288 - val_mse: 0.1288
    Epoch 21/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0524 - mse: 0.0524 - val_loss: 0.1263 - val_mse: 0.1263
    Epoch 22/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0505 - mse: 0.0505 - val_loss: 0.1325 - val_mse: 0.1325
    Epoch 23/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0491 - mse: 0.0491 - val_loss: 0.1255 - val_mse: 0.1255
    Epoch 24/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0479 - mse: 0.0479 - val_loss: 0.1286 - val_mse: 0.1286
    Epoch 25/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0467 - mse: 0.0467 - val_loss: 0.1326 - val_mse: 0.1326
    Epoch 26/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0442 - mse: 0.0442 - val_loss: 0.1310 - val_mse: 0.1310
    Epoch 27/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0429 - mse: 0.0429 - val_loss: 0.1324 - val_mse: 0.1324
    Epoch 28/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0413 - mse: 0.0413 - val_loss: 0.1288 - val_mse: 0.1288
    Epoch 29/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0405 - mse: 0.0405 - val_loss: 0.1331 - val_mse: 0.1331
    Epoch 30/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0398 - mse: 0.0398 - val_loss: 0.1321 - val_mse: 0.1321
    Epoch 31/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0390 - mse: 0.0390 - val_loss: 0.1328 - val_mse: 0.1328
    Epoch 32/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0377 - mse: 0.0377 - val_loss: 0.1358 - val_mse: 0.1358
    Epoch 33/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0369 - mse: 0.0369 - val_loss: 0.1334 - val_mse: 0.1334
    Epoch 34/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0361 - mse: 0.0361 - val_loss: 0.1350 - val_mse: 0.1350
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0359 - mse: 0.0359 - val_loss: 0.1327 - val_mse: 0.1327
    Epoch 36/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0340 - mse: 0.0340 - val_loss: 0.1319 - val_mse: 0.1319
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0334 - mse: 0.0334 - val_loss: 0.1365 - val_mse: 0.1365
    Epoch 38/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0326 - mse: 0.0326 - val_loss: 0.1362 - val_mse: 0.1362
    Epoch 39/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0321 - mse: 0.0321 - val_loss: 0.1367 - val_mse: 0.1367
    Epoch 40/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0310 - mse: 0.0310 - val_loss: 0.1369 - val_mse: 0.1369
    Epoch 41/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0313 - mse: 0.0313 - val_loss: 0.1350 - val_mse: 0.1350
    Epoch 42/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0295 - mse: 0.0295 - val_loss: 0.1348 - val_mse: 0.1348
    Epoch 43/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0291 - mse: 0.0291 - val_loss: 0.1362 - val_mse: 0.1362
    Epoch 44/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0287 - mse: 0.0287 - val_loss: 0.1353 - val_mse: 0.1353
    Epoch 45/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0284 - mse: 0.0284 - val_loss: 0.1356 - val_mse: 0.1356
    Epoch 46/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0279 - mse: 0.0279 - val_loss: 0.1353 - val_mse: 0.1353
    Epoch 47/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0279 - mse: 0.0279 - val_loss: 0.1393 - val_mse: 0.1393
    Epoch 48/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0266 - mse: 0.0266 - val_loss: 0.1390 - val_mse: 0.1390
    Epoch 49/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0269 - mse: 0.0269 - val_loss: 0.1393 - val_mse: 0.1393
    Epoch 50/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0258 - mse: 0.0258 - val_loss: 0.1409 - val_mse: 0.1409
    Epoch 51/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0255 - mse: 0.0255 - val_loss: 0.1417 - val_mse: 0.1417
    Epoch 52/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0246 - mse: 0.0246 - val_loss: 0.1400 - val_mse: 0.1400
    Epoch 53/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0245 - mse: 0.0245 - val_loss: 0.1452 - val_mse: 0.1452
    Epoch 54/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0245 - mse: 0.0245 - val_loss: 0.1387 - val_mse: 0.1387
    Epoch 55/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0237 - mse: 0.0237 - val_loss: 0.1396 - val_mse: 0.1396
    Epoch 56/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0233 - mse: 0.0233 - val_loss: 0.1415 - val_mse: 0.1415
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0226 - mse: 0.0226 - val_loss: 0.1427 - val_mse: 0.1427
    Epoch 58/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0227 - mse: 0.0227 - val_loss: 0.1393 - val_mse: 0.1393
    Epoch 59/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0220 - mse: 0.0220 - val_loss: 0.1401 - val_mse: 0.1401
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0217 - mse: 0.0217 - val_loss: 0.1429 - val_mse: 0.1429
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0215 - mse: 0.0215 - val_loss: 0.1407 - val_mse: 0.1407
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0212 - mse: 0.0212 - val_loss: 0.1437 - val_mse: 0.1437
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0206 - mse: 0.0206 - val_loss: 0.1436 - val_mse: 0.1436
    Epoch 64/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0207 - mse: 0.0207 - val_loss: 0.1411 - val_mse: 0.1411
    Epoch 65/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0204 - mse: 0.0204 - val_loss: 0.1429 - val_mse: 0.1429
    Epoch 66/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0200 - mse: 0.0200 - val_loss: 0.1426 - val_mse: 0.1426
    Epoch 67/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0195 - mse: 0.0195 - val_loss: 0.1440 - val_mse: 0.1440
    Epoch 68/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0198 - mse: 0.0198 - val_loss: 0.1443 - val_mse: 0.1443
    Epoch 69/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0193 - mse: 0.0193 - val_loss: 0.1445 - val_mse: 0.1445
    Epoch 70/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0188 - mse: 0.0188 - val_loss: 0.1439 - val_mse: 0.1439
    Epoch 71/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0184 - mse: 0.0184 - val_loss: 0.1435 - val_mse: 0.1435
    Epoch 72/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0185 - mse: 0.0185 - val_loss: 0.1434 - val_mse: 0.1434
    Epoch 73/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0182 - mse: 0.0182 - val_loss: 0.1461 - val_mse: 0.1461
    Epoch 74/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0180 - mse: 0.0180 - val_loss: 0.1453 - val_mse: 0.1453
    Epoch 75/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0176 - mse: 0.0176 - val_loss: 0.1440 - val_mse: 0.1440
    Epoch 76/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0174 - mse: 0.0174 - val_loss: 0.1434 - val_mse: 0.1434
    Epoch 77/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0170 - mse: 0.0170 - val_loss: 0.1453 - val_mse: 0.1453
    Epoch 78/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0170 - mse: 0.0170 - val_loss: 0.1447 - val_mse: 0.1447
    Epoch 79/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0168 - mse: 0.0168 - val_loss: 0.1446 - val_mse: 0.1446
    Epoch 80/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0165 - mse: 0.0165 - val_loss: 0.1441 - val_mse: 0.1441
    Epoch 81/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0162 - mse: 0.0162 - val_loss: 0.1491 - val_mse: 0.1491
    Epoch 82/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0161 - mse: 0.0161 - val_loss: 0.1457 - val_mse: 0.1457
    Epoch 83/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0161 - mse: 0.0161 - val_loss: 0.1462 - val_mse: 0.1462
    Epoch 84/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0158 - mse: 0.0158 - val_loss: 0.1452 - val_mse: 0.1452
    Epoch 85/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0158 - mse: 0.0158 - val_loss: 0.1473 - val_mse: 0.1473
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0154 - mse: 0.0154 - val_loss: 0.1458 - val_mse: 0.1458
    Epoch 87/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0155 - mse: 0.0155 - val_loss: 0.1461 - val_mse: 0.1461
    Epoch 88/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0149 - mse: 0.0149 - val_loss: 0.1467 - val_mse: 0.1467
    Epoch 89/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0150 - mse: 0.0150 - val_loss: 0.1470 - val_mse: 0.1470
    Epoch 90/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0154 - mse: 0.0154 - val_loss: 0.1469 - val_mse: 0.1469
    Epoch 91/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0145 - mse: 0.0145 - val_loss: 0.1481 - val_mse: 0.1481
    Epoch 92/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0143 - mse: 0.0143 - val_loss: 0.1464 - val_mse: 0.1464
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0143 - mse: 0.0143 - val_loss: 0.1482 - val_mse: 0.1482
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0142 - mse: 0.0142 - val_loss: 0.1471 - val_mse: 0.1471
    Epoch 95/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0139 - mse: 0.0139 - val_loss: 0.1479 - val_mse: 0.1479
    Epoch 96/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0138 - mse: 0.0138 - val_loss: 0.1466 - val_mse: 0.1466
    Epoch 97/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0134 - mse: 0.0134 - val_loss: 0.1476 - val_mse: 0.1476
    Epoch 98/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0133 - mse: 0.0133 - val_loss: 0.1480 - val_mse: 0.1480
    Epoch 99/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0133 - mse: 0.0133 - val_loss: 0.1479 - val_mse: 0.1479
    Epoch 100/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0129 - mse: 0.0129 - val_loss: 0.1499 - val_mse: 0.1499
    Epoch 101/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0129 - mse: 0.0129 - val_loss: 0.1493 - val_mse: 0.1493
    Epoch 102/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0128 - mse: 0.0128 - val_loss: 0.1492 - val_mse: 0.1492
    Epoch 103/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0126 - mse: 0.0126 - val_loss: 0.1492 - val_mse: 0.1492
    Epoch 104/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0126 - mse: 0.0126 - val_loss: 0.1478 - val_mse: 0.1478
    Epoch 105/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0123 - mse: 0.0123 - val_loss: 0.1480 - val_mse: 0.1480
    Epoch 106/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0123 - mse: 0.0123 - val_loss: 0.1486 - val_mse: 0.1486
    Epoch 107/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 0.1482 - val_mse: 0.1482
    Epoch 108/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 0.1483 - val_mse: 0.1483
    Epoch 109/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0118 - mse: 0.0118 - val_loss: 0.1477 - val_mse: 0.1477
    Epoch 110/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0118 - mse: 0.0118 - val_loss: 0.1494 - val_mse: 0.1494
    Epoch 111/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0115 - mse: 0.0115 - val_loss: 0.1497 - val_mse: 0.1497
    Epoch 112/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0119 - mse: 0.0119 - val_loss: 0.1490 - val_mse: 0.1490
    Epoch 113/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0115 - mse: 0.0115 - val_loss: 0.1487 - val_mse: 0.1487
    Epoch 114/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0116 - mse: 0.0116 - val_loss: 0.1488 - val_mse: 0.1488
    Epoch 115/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.1505 - val_mse: 0.1505
    Epoch 116/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0111 - mse: 0.0111 - val_loss: 0.1502 - val_mse: 0.1502
    Epoch 117/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1508 - val_mse: 0.1508
    Epoch 118/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 0.1496 - val_mse: 0.1496
    Epoch 119/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0104 - mse: 0.0104 - val_loss: 0.1500 - val_mse: 0.1500
    Epoch 120/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 0.1530 - val_mse: 0.1530
    Epoch 121/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0104 - mse: 0.0104 - val_loss: 0.1501 - val_mse: 0.1501
    Epoch 122/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0102 - mse: 0.0102 - val_loss: 0.1529 - val_mse: 0.1529
    Epoch 123/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0102 - mse: 0.0102 - val_loss: 0.1507 - val_mse: 0.1507
    Epoch 124/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0101 - mse: 0.0101 - val_loss: 0.1500 - val_mse: 0.1500
    Epoch 125/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1522 - val_mse: 0.1522
    Epoch 126/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1525 - val_mse: 0.1525
    Epoch 127/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1523 - val_mse: 0.1523
    Epoch 128/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1505 - val_mse: 0.1505
    Epoch 129/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0098 - mse: 0.0098 - val_loss: 0.1498 - val_mse: 0.1498
    Epoch 130/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0095 - mse: 0.0095 - val_loss: 0.1518 - val_mse: 0.1518
    Epoch 131/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 0.1513 - val_mse: 0.1513
    Epoch 132/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 0.1513 - val_mse: 0.1513
    Epoch 133/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0092 - mse: 0.0092 - val_loss: 0.1512 - val_mse: 0.1512
    Epoch 134/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0091 - mse: 0.0091 - val_loss: 0.1524 - val_mse: 0.1524
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 0.1512 - val_mse: 0.1512
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 0.1526 - val_mse: 0.1526
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 0.1520 - val_mse: 0.1520
    Epoch 138/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0087 - mse: 0.0087 - val_loss: 0.1516 - val_mse: 0.1516
    Epoch 139/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0087 - mse: 0.0087 - val_loss: 0.1527 - val_mse: 0.1527
    Epoch 140/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0085 - mse: 0.0085 - val_loss: 0.1526 - val_mse: 0.1526
    Epoch 141/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0087 - mse: 0.0087 - val_loss: 0.1529 - val_mse: 0.1529
    Epoch 142/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0083 - mse: 0.0083 - val_loss: 0.1521 - val_mse: 0.1521
    Epoch 143/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0084 - mse: 0.0084 - val_loss: 0.1514 - val_mse: 0.1514
    Epoch 144/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.1532 - val_mse: 0.1532
    Epoch 145/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.1530 - val_mse: 0.1530
    Epoch 146/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0080 - mse: 0.0080 - val_loss: 0.1538 - val_mse: 0.1538
    Epoch 147/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0081 - mse: 0.0081 - val_loss: 0.1536 - val_mse: 0.1536
    Epoch 148/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0079 - mse: 0.0079 - val_loss: 0.1533 - val_mse: 0.1533
    Epoch 149/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0078 - mse: 0.0078 - val_loss: 0.1532 - val_mse: 0.1532
    Epoch 150/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0078 - mse: 0.0078 - val_loss: 0.1555 - val_mse: 0.1555





    <keras.callbacks.History at 0x16c37e3a0>



Evaluate the model (`lecun_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
lecun_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 4ms/step - loss: 0.0080 - mse: 0.0080





    [0.00799749419093132, 0.00799749419093132]



Evaluate the model (`lecun_model`) on validate data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data
lecun_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 3ms/step - loss: 0.1555 - mse: 0.1555





    [0.15551191568374634, 0.15551191568374634]



Not much of a difference, but a useful note to consider when tuning your network. Next, let's investigate the impact of various optimization algorithms.

## RMSprop 

Compile the `rmsprop_model` with: 

- `'rmsprop'` as the optimizer 
- track `'mse'` as the loss and metric  


```python
np.random.seed(123)
rmsprop_model = Sequential()
rmsprop_model.add(layers.Dense(100, activation='relu', input_shape=n_features))
rmsprop_model.add(layers.Dense(50, activation='relu'))
rmsprop_model.add(layers.Dense(1, activation='linear'))

# Compile the model
rmsprop_model.compile(optimizer='rmsprop', 
                      loss='mse', 
                      metrics=['mse'])

# Train the model
rmsprop_model.fit(X_train, 
                  y_train_scaled, 
                  batch_size=32, 
                  epochs=150, 
                  validation_data=(X_val, y_val_scaled))
```

    Epoch 1/150
     1/33 [..............................] - ETA: 9s - loss: 0.9326 - mse: 0.9326

    2023-04-14 18:35:56.957068: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 1s 14ms/step - loss: 0.3377 - mse: 0.3377 - val_loss: 0.1505 - val_mse: 0.1505
    Epoch 2/150
     9/33 [=======>......................] - ETA: 0s - loss: 0.1055 - mse: 0.1055

    2023-04-14 18:35:57.500517: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 7ms/step - loss: 0.1919 - mse: 0.1919 - val_loss: 0.1563 - val_mse: 0.1563
    Epoch 3/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.1279 - mse: 0.1279 - val_loss: 0.1524 - val_mse: 0.1524
    Epoch 4/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.1018 - mse: 0.1018 - val_loss: 0.2272 - val_mse: 0.2272
    Epoch 5/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0747 - mse: 0.0747 - val_loss: 0.1330 - val_mse: 0.1330
    Epoch 6/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0726 - mse: 0.0726 - val_loss: 0.1031 - val_mse: 0.1031
    Epoch 7/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0435 - mse: 0.0435 - val_loss: 0.1245 - val_mse: 0.1245
    Epoch 8/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0490 - mse: 0.0490 - val_loss: 0.1096 - val_mse: 0.1096
    Epoch 9/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0353 - mse: 0.0353 - val_loss: 0.1332 - val_mse: 0.1332
    Epoch 10/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0350 - mse: 0.0350 - val_loss: 0.0991 - val_mse: 0.0991
    Epoch 11/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0298 - mse: 0.0298 - val_loss: 0.1150 - val_mse: 0.1150
    Epoch 12/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0329 - mse: 0.0329 - val_loss: 0.1143 - val_mse: 0.1143
    Epoch 13/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0220 - mse: 0.0220 - val_loss: 0.1172 - val_mse: 0.1172
    Epoch 14/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0243 - mse: 0.0243 - val_loss: 0.1393 - val_mse: 0.1393
    Epoch 15/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0232 - mse: 0.0232 - val_loss: 0.1337 - val_mse: 0.1337
    Epoch 16/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0172 - mse: 0.0172 - val_loss: 0.0948 - val_mse: 0.0948
    Epoch 17/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0230 - mse: 0.0230 - val_loss: 0.1018 - val_mse: 0.1018
    Epoch 18/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0195 - mse: 0.0195 - val_loss: 0.1056 - val_mse: 0.1056
    Epoch 19/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0204 - mse: 0.0204 - val_loss: 0.0964 - val_mse: 0.0964
    Epoch 20/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0176 - mse: 0.0176 - val_loss: 0.1075 - val_mse: 0.1075
    Epoch 21/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0178 - mse: 0.0178 - val_loss: 0.1015 - val_mse: 0.1015
    Epoch 22/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0167 - mse: 0.0167 - val_loss: 0.0951 - val_mse: 0.0951
    Epoch 23/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0142 - mse: 0.0142 - val_loss: 0.1197 - val_mse: 0.1197
    Epoch 24/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0167 - mse: 0.0167 - val_loss: 0.0970 - val_mse: 0.0970
    Epoch 25/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0153 - mse: 0.0153 - val_loss: 0.1017 - val_mse: 0.1017
    Epoch 26/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0139 - mse: 0.0139 - val_loss: 0.0968 - val_mse: 0.0968
    Epoch 27/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0153 - mse: 0.0153 - val_loss: 0.1028 - val_mse: 0.1028
    Epoch 28/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0123 - mse: 0.0123 - val_loss: 0.1148 - val_mse: 0.1148
    Epoch 29/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0171 - mse: 0.0171 - val_loss: 0.0959 - val_mse: 0.0959
    Epoch 30/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0126 - mse: 0.0126 - val_loss: 0.1008 - val_mse: 0.1008
    Epoch 31/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0111 - mse: 0.0111 - val_loss: 0.0959 - val_mse: 0.0959
    Epoch 32/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0149 - mse: 0.0149 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 33/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0127 - mse: 0.0127 - val_loss: 0.0961 - val_mse: 0.0961
    Epoch 34/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0140 - mse: 0.0140 - val_loss: 0.0999 - val_mse: 0.0999
    Epoch 35/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0100 - mse: 0.0100 - val_loss: 0.1059 - val_mse: 0.1059
    Epoch 36/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0136 - mse: 0.0136 - val_loss: 0.1035 - val_mse: 0.1035
    Epoch 37/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0108 - mse: 0.0108 - val_loss: 0.1092 - val_mse: 0.1092
    Epoch 38/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 0.0904 - val_mse: 0.0904
    Epoch 39/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0090 - mse: 0.0090 - val_loss: 0.1068 - val_mse: 0.1068
    Epoch 40/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 0.0945 - val_mse: 0.0945
    Epoch 41/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0103 - mse: 0.0103 - val_loss: 0.0975 - val_mse: 0.0975
    Epoch 42/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0105 - mse: 0.0105 - val_loss: 0.0871 - val_mse: 0.0871
    Epoch 43/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.1120 - val_mse: 0.1120
    Epoch 44/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0094 - mse: 0.0094 - val_loss: 0.0987 - val_mse: 0.0987
    Epoch 45/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 0.0837 - val_mse: 0.0837
    Epoch 46/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0111 - mse: 0.0111 - val_loss: 0.0920 - val_mse: 0.0920
    Epoch 47/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0097 - mse: 0.0097 - val_loss: 0.0850 - val_mse: 0.0850
    Epoch 48/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0094 - mse: 0.0094 - val_loss: 0.0894 - val_mse: 0.0894
    Epoch 49/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0095 - mse: 0.0095 - val_loss: 0.0866 - val_mse: 0.0866
    Epoch 50/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0108 - mse: 0.0108 - val_loss: 0.0864 - val_mse: 0.0864
    Epoch 51/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0085 - mse: 0.0085 - val_loss: 0.0884 - val_mse: 0.0884
    Epoch 52/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0084 - mse: 0.0084 - val_loss: 0.0867 - val_mse: 0.0867
    Epoch 53/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0124 - mse: 0.0124 - val_loss: 0.1076 - val_mse: 0.1076
    Epoch 54/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0101 - mse: 0.0101 - val_loss: 0.0922 - val_mse: 0.0922
    Epoch 55/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0063 - mse: 0.0063 - val_loss: 0.0896 - val_mse: 0.0896
    Epoch 56/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0075 - mse: 0.0075 - val_loss: 0.1255 - val_mse: 0.1255
    Epoch 57/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0086 - mse: 0.0086 - val_loss: 0.0874 - val_mse: 0.0874
    Epoch 58/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0097 - mse: 0.0097 - val_loss: 0.0918 - val_mse: 0.0918
    Epoch 59/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.0866 - val_mse: 0.0866
    Epoch 60/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.0864 - val_mse: 0.0864
    Epoch 61/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0087 - mse: 0.0087 - val_loss: 0.0943 - val_mse: 0.0943
    Epoch 62/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.0905 - val_mse: 0.0905
    Epoch 63/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0070 - mse: 0.0070 - val_loss: 0.0892 - val_mse: 0.0892
    Epoch 64/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0084 - mse: 0.0084 - val_loss: 0.0970 - val_mse: 0.0970
    Epoch 65/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.0838 - val_mse: 0.0838
    Epoch 66/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0092 - mse: 0.0092 - val_loss: 0.0853 - val_mse: 0.0853
    Epoch 67/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0072 - mse: 0.0072 - val_loss: 0.0869 - val_mse: 0.0869
    Epoch 68/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0078 - mse: 0.0078 - val_loss: 0.0921 - val_mse: 0.0921
    Epoch 69/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0087 - mse: 0.0087 - val_loss: 0.0799 - val_mse: 0.0799
    Epoch 70/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0067 - mse: 0.0067 - val_loss: 0.0842 - val_mse: 0.0842
    Epoch 71/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0056 - mse: 0.0056 - val_loss: 0.0892 - val_mse: 0.0892
    Epoch 72/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0060 - mse: 0.0060 - val_loss: 0.1074 - val_mse: 0.1074
    Epoch 73/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0079 - mse: 0.0079 - val_loss: 0.0891 - val_mse: 0.0891
    Epoch 74/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.0809 - val_mse: 0.0809
    Epoch 75/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0071 - mse: 0.0071 - val_loss: 0.0843 - val_mse: 0.0843
    Epoch 76/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.0852 - val_mse: 0.0852
    Epoch 77/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0080 - mse: 0.0080 - val_loss: 0.1006 - val_mse: 0.1006
    Epoch 78/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.0810 - val_mse: 0.0810
    Epoch 79/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.0830 - val_mse: 0.0830
    Epoch 80/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0068 - mse: 0.0068 - val_loss: 0.0848 - val_mse: 0.0848
    Epoch 81/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.0974 - val_mse: 0.0974
    Epoch 82/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.0891 - val_mse: 0.0891
    Epoch 83/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0063 - mse: 0.0063 - val_loss: 0.0898 - val_mse: 0.0898
    Epoch 84/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.0893 - val_mse: 0.0893
    Epoch 85/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.0798 - val_mse: 0.0798
    Epoch 86/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0053 - mse: 0.0053 - val_loss: 0.0863 - val_mse: 0.0863
    Epoch 87/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.0759 - val_mse: 0.0759
    Epoch 88/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0075 - mse: 0.0075 - val_loss: 0.0828 - val_mse: 0.0828
    Epoch 89/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0052 - mse: 0.0052 - val_loss: 0.0806 - val_mse: 0.0806
    Epoch 90/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0068 - mse: 0.0068 - val_loss: 0.0828 - val_mse: 0.0828
    Epoch 91/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.0798 - val_mse: 0.0798
    Epoch 92/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0064 - mse: 0.0064 - val_loss: 0.0810 - val_mse: 0.0810
    Epoch 93/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0056 - mse: 0.0056 - val_loss: 0.0801 - val_mse: 0.0801
    Epoch 94/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0052 - mse: 0.0052 - val_loss: 0.0799 - val_mse: 0.0799
    Epoch 95/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.0814 - val_mse: 0.0814
    Epoch 96/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0072 - mse: 0.0072 - val_loss: 0.0819 - val_mse: 0.0819
    Epoch 97/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0056 - mse: 0.0056 - val_loss: 0.0781 - val_mse: 0.0781
    Epoch 98/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0052 - mse: 0.0052 - val_loss: 0.0842 - val_mse: 0.0842
    Epoch 99/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0057 - mse: 0.0057 - val_loss: 0.0811 - val_mse: 0.0811
    Epoch 100/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.0911 - val_mse: 0.0911
    Epoch 101/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0060 - mse: 0.0060 - val_loss: 0.0787 - val_mse: 0.0787
    Epoch 102/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0964 - val_mse: 0.0964
    Epoch 103/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0067 - mse: 0.0067 - val_loss: 0.0773 - val_mse: 0.0773
    Epoch 104/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0051 - mse: 0.0051 - val_loss: 0.0800 - val_mse: 0.0800
    Epoch 105/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0076 - mse: 0.0076 - val_loss: 0.0784 - val_mse: 0.0784
    Epoch 106/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0036 - mse: 0.0036 - val_loss: 0.0782 - val_mse: 0.0782
    Epoch 107/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0054 - mse: 0.0054 - val_loss: 0.0841 - val_mse: 0.0841
    Epoch 108/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0067 - mse: 0.0067 - val_loss: 0.0815 - val_mse: 0.0815
    Epoch 109/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0051 - mse: 0.0051 - val_loss: 0.0793 - val_mse: 0.0793
    Epoch 110/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0060 - mse: 0.0060 - val_loss: 0.0823 - val_mse: 0.0823
    Epoch 111/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0964 - val_mse: 0.0964
    Epoch 112/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.0788 - val_mse: 0.0788
    Epoch 113/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.0807 - val_mse: 0.0807
    Epoch 114/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.0771 - val_mse: 0.0771
    Epoch 115/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0059 - mse: 0.0059 - val_loss: 0.0784 - val_mse: 0.0784
    Epoch 116/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.0829 - val_mse: 0.0829
    Epoch 117/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0052 - mse: 0.0052 - val_loss: 0.0770 - val_mse: 0.0770
    Epoch 118/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0060 - mse: 0.0060 - val_loss: 0.0789 - val_mse: 0.0789
    Epoch 119/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0992 - val_mse: 0.0992
    Epoch 120/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0768 - val_mse: 0.0768
    Epoch 121/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0785 - val_mse: 0.0785
    Epoch 122/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.0880 - val_mse: 0.0880
    Epoch 123/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0807 - val_mse: 0.0807
    Epoch 124/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0785 - val_mse: 0.0785
    Epoch 125/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.0795 - val_mse: 0.0795
    Epoch 126/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0940 - val_mse: 0.0940
    Epoch 127/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0062 - mse: 0.0062 - val_loss: 0.0813 - val_mse: 0.0813
    Epoch 128/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0772 - val_mse: 0.0772
    Epoch 129/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0772 - val_mse: 0.0772
    Epoch 130/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0781 - val_mse: 0.0781
    Epoch 131/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.0813 - val_mse: 0.0813
    Epoch 132/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0036 - mse: 0.0036 - val_loss: 0.0811 - val_mse: 0.0811
    Epoch 133/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0042 - mse: 0.0042 - val_loss: 0.0748 - val_mse: 0.0748
    Epoch 134/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0778 - val_mse: 0.0778
    Epoch 135/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0861 - val_mse: 0.0861
    Epoch 136/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0052 - mse: 0.0052 - val_loss: 0.0817 - val_mse: 0.0817
    Epoch 137/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0787 - val_mse: 0.0787
    Epoch 138/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.0928 - val_mse: 0.0928
    Epoch 139/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.0795 - val_mse: 0.0795
    Epoch 140/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0800 - val_mse: 0.0800
    Epoch 141/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0036 - mse: 0.0036 - val_loss: 0.0961 - val_mse: 0.0961
    Epoch 142/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0754 - val_mse: 0.0754
    Epoch 143/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.0852 - val_mse: 0.0852
    Epoch 144/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.0782 - val_mse: 0.0782
    Epoch 145/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.0795 - val_mse: 0.0795
    Epoch 146/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0926 - val_mse: 0.0926
    Epoch 147/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.0793 - val_mse: 0.0793
    Epoch 148/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0042 - mse: 0.0042 - val_loss: 0.0842 - val_mse: 0.0842
    Epoch 149/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.0844 - val_mse: 0.0844
    Epoch 150/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.0863 - val_mse: 0.0863





    <keras.callbacks.History at 0x17aa58d30>



Evaluate the model (`rmsprop_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
rmsprop_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 4ms/step - loss: 0.0022 - mse: 0.0022





    [0.002239547437056899, 0.002239547437056899]



Evaluate the model (`rmsprop_model`) on training data (`X_val` and `y_val_scaled`) 


```python
# Evaluate the model on validate data
rmsprop_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 3ms/step - loss: 0.0863 - mse: 0.0863





    [0.0863339826464653, 0.0863339826464653]



## Adam 

Compile the `adam_model` with: 

- `'Adam'` as the optimizer 
- track `'mse'` as the loss and metric  


```python
np.random.seed(123)
adam_model = Sequential()
adam_model.add(layers.Dense(100, activation='relu', input_shape=n_features))
adam_model.add(layers.Dense(50, activation='relu'))
adam_model.add(layers.Dense(1, activation='linear'))

# Compile the model
adam_model.compile(optimizer='Adam', 
                   loss='mse', 
                   metrics=['mse'])

# Train the model
adam_model.fit(X_train, 
               y_train_scaled, 
               batch_size=32, 
               epochs=150, 
               validation_data=(X_val, y_val_scaled))
```

    Epoch 1/150
     9/33 [=======>......................] - ETA: 0s - loss: 0.6130 - mse: 0.6130

    2023-04-14 18:36:34.149293: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 10ms/step - loss: 0.4503 - mse: 0.4503 - val_loss: 0.1665 - val_mse: 0.1665
    Epoch 2/150
    22/33 [===================>..........] - ETA: 0s - loss: 0.1821 - mse: 0.1821

    2023-04-14 18:36:34.495092: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 6ms/step - loss: 0.1841 - mse: 0.1841 - val_loss: 0.1547 - val_mse: 0.1547
    Epoch 3/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1180 - mse: 0.1180 - val_loss: 0.1138 - val_mse: 0.1138
    Epoch 4/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0901 - mse: 0.0901 - val_loss: 0.1400 - val_mse: 0.1400
    Epoch 5/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0703 - mse: 0.0703 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 6/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0529 - mse: 0.0529 - val_loss: 0.1251 - val_mse: 0.1251
    Epoch 7/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0416 - mse: 0.0416 - val_loss: 0.1169 - val_mse: 0.1169
    Epoch 8/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0347 - mse: 0.0347 - val_loss: 0.1276 - val_mse: 0.1276
    Epoch 9/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0291 - mse: 0.0291 - val_loss: 0.1064 - val_mse: 0.1064
    Epoch 10/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0247 - mse: 0.0247 - val_loss: 0.1262 - val_mse: 0.1262
    Epoch 11/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0213 - mse: 0.0213 - val_loss: 0.1044 - val_mse: 0.1044
    Epoch 12/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0185 - mse: 0.0185 - val_loss: 0.1426 - val_mse: 0.1426
    Epoch 13/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0166 - mse: 0.0166 - val_loss: 0.1093 - val_mse: 0.1093
    Epoch 14/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0149 - mse: 0.0149 - val_loss: 0.1224 - val_mse: 0.1224
    Epoch 15/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0130 - mse: 0.0130 - val_loss: 0.1055 - val_mse: 0.1055
    Epoch 16/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0123 - mse: 0.0123 - val_loss: 0.1158 - val_mse: 0.1158
    Epoch 17/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0115 - mse: 0.0115 - val_loss: 0.1126 - val_mse: 0.1126
    Epoch 18/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0118 - mse: 0.0118 - val_loss: 0.1248 - val_mse: 0.1248
    Epoch 19/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 0.1145 - val_mse: 0.1145
    Epoch 20/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0121 - mse: 0.0121 - val_loss: 0.1292 - val_mse: 0.1292
    Epoch 21/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0117 - mse: 0.0117 - val_loss: 0.1320 - val_mse: 0.1320
    Epoch 22/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0140 - mse: 0.0140 - val_loss: 0.1301 - val_mse: 0.1301
    Epoch 23/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 0.1051 - val_mse: 0.1051
    Epoch 24/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0104 - mse: 0.0104 - val_loss: 0.1231 - val_mse: 0.1231
    Epoch 25/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0106 - mse: 0.0106 - val_loss: 0.1133 - val_mse: 0.1133
    Epoch 26/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0119 - mse: 0.0119 - val_loss: 0.1141 - val_mse: 0.1141
    Epoch 27/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0088 - mse: 0.0088 - val_loss: 0.1128 - val_mse: 0.1128
    Epoch 28/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.1119 - val_mse: 0.1119
    Epoch 29/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1151 - val_mse: 0.1151
    Epoch 30/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0042 - mse: 0.0042 - val_loss: 0.1150 - val_mse: 0.1150
    Epoch 31/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.1178 - val_mse: 0.1178
    Epoch 32/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.1189 - val_mse: 0.1189
    Epoch 33/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0036 - mse: 0.0036 - val_loss: 0.1133 - val_mse: 0.1133
    Epoch 34/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1176 - val_mse: 0.1176
    Epoch 35/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0037 - mse: 0.0037 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 36/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.1156 - val_mse: 0.1156
    Epoch 37/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0036 - mse: 0.0036 - val_loss: 0.1143 - val_mse: 0.1143
    Epoch 38/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.1135 - val_mse: 0.1135
    Epoch 39/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.1163 - val_mse: 0.1163
    Epoch 40/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1167 - val_mse: 0.1167
    Epoch 41/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0060 - mse: 0.0060 - val_loss: 0.1217 - val_mse: 0.1217
    Epoch 42/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0064 - mse: 0.0064 - val_loss: 0.1198 - val_mse: 0.1198
    Epoch 43/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1239 - val_mse: 0.1239
    Epoch 44/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.1072 - val_mse: 0.1072
    Epoch 45/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.1143 - val_mse: 0.1143
    Epoch 46/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0065 - mse: 0.0065 - val_loss: 0.1101 - val_mse: 0.1101
    Epoch 47/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0080 - mse: 0.0080 - val_loss: 0.1166 - val_mse: 0.1166
    Epoch 48/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0053 - mse: 0.0053 - val_loss: 0.1179 - val_mse: 0.1179
    Epoch 49/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.1234 - val_mse: 0.1234
    Epoch 50/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0996 - val_mse: 0.0996
    Epoch 51/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0036 - mse: 0.0036 - val_loss: 0.1143 - val_mse: 0.1143
    Epoch 52/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0031 - mse: 0.0031 - val_loss: 0.1042 - val_mse: 0.1042
    Epoch 53/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1173 - val_mse: 0.1173
    Epoch 54/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0023 - mse: 0.0023 - val_loss: 0.1047 - val_mse: 0.1047
    Epoch 55/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0014 - mse: 0.0014 - val_loss: 0.1186 - val_mse: 0.1186
    Epoch 56/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0015 - mse: 0.0015 - val_loss: 0.1043 - val_mse: 0.1043
    Epoch 57/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0012 - mse: 0.0012 - val_loss: 0.1200 - val_mse: 0.1200
    Epoch 58/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0012 - mse: 0.0012 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 59/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0012 - mse: 0.0012 - val_loss: 0.1158 - val_mse: 0.1158
    Epoch 60/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0013 - mse: 0.0013 - val_loss: 0.1108 - val_mse: 0.1108
    Epoch 61/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0015 - mse: 0.0015 - val_loss: 0.1130 - val_mse: 0.1130
    Epoch 62/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0021 - mse: 0.0021 - val_loss: 0.1125 - val_mse: 0.1125
    Epoch 63/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0028 - mse: 0.0028 - val_loss: 0.1184 - val_mse: 0.1184
    Epoch 64/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0030 - mse: 0.0030 - val_loss: 0.1065 - val_mse: 0.1065
    Epoch 65/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.1237 - val_mse: 0.1237
    Epoch 66/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.1054 - val_mse: 0.1054
    Epoch 67/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0057 - mse: 0.0057 - val_loss: 0.1138 - val_mse: 0.1138
    Epoch 68/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.1161 - val_mse: 0.1161
    Epoch 69/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0034 - mse: 0.0034 - val_loss: 0.1067 - val_mse: 0.1067
    Epoch 70/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.1223 - val_mse: 0.1223
    Epoch 71/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0054 - mse: 0.0054 - val_loss: 0.1132 - val_mse: 0.1132
    Epoch 72/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0051 - mse: 0.0051 - val_loss: 0.1169 - val_mse: 0.1169
    Epoch 73/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1109 - val_mse: 0.1109
    Epoch 74/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.1124 - val_mse: 0.1124
    Epoch 75/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.1057 - val_mse: 0.1057
    Epoch 76/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0029 - mse: 0.0029 - val_loss: 0.1069 - val_mse: 0.1069
    Epoch 77/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0030 - mse: 0.0030 - val_loss: 0.1221 - val_mse: 0.1221
    Epoch 78/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.1087 - val_mse: 0.1087
    Epoch 79/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0094 - mse: 0.0094 - val_loss: 0.1249 - val_mse: 0.1249
    Epoch 80/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 0.1104 - val_mse: 0.1104
    Epoch 81/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0091 - mse: 0.0091 - val_loss: 0.1327 - val_mse: 0.1327
    Epoch 82/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0076 - mse: 0.0076 - val_loss: 0.1210 - val_mse: 0.1210
    Epoch 83/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 84/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.1122 - val_mse: 0.1122
    Epoch 85/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0033 - mse: 0.0033 - val_loss: 0.1237 - val_mse: 0.1237
    Epoch 86/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.1113 - val_mse: 0.1113
    Epoch 87/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0035 - mse: 0.0035 - val_loss: 0.1206 - val_mse: 0.1206
    Epoch 88/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.1170 - val_mse: 0.1170
    Epoch 89/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0063 - mse: 0.0063 - val_loss: 0.1153 - val_mse: 0.1153
    Epoch 90/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.1030 - val_mse: 0.1030
    Epoch 91/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0052 - mse: 0.0052 - val_loss: 0.1207 - val_mse: 0.1207
    Epoch 92/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.1079 - val_mse: 0.1079
    Epoch 93/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.1198 - val_mse: 0.1198
    Epoch 94/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1118 - val_mse: 0.1118
    Epoch 95/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0026 - mse: 0.0026 - val_loss: 0.1174 - val_mse: 0.1174
    Epoch 96/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0027 - mse: 0.0027 - val_loss: 0.1055 - val_mse: 0.1055
    Epoch 97/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0020 - mse: 0.0020 - val_loss: 0.1177 - val_mse: 0.1177
    Epoch 98/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0015 - mse: 0.0015 - val_loss: 0.1104 - val_mse: 0.1104
    Epoch 99/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0012 - mse: 0.0012 - val_loss: 0.1124 - val_mse: 0.1124
    Epoch 100/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0011 - mse: 0.0011 - val_loss: 0.1079 - val_mse: 0.1079
    Epoch 101/150
    33/33 [==============================] - 0s 6ms/step - loss: 9.5729e-04 - mse: 9.5729e-04 - val_loss: 0.1120 - val_mse: 0.1120
    Epoch 102/150
    33/33 [==============================] - 0s 6ms/step - loss: 7.0777e-04 - mse: 7.0777e-04 - val_loss: 0.1074 - val_mse: 0.1074
    Epoch 103/150
    33/33 [==============================] - 0s 6ms/step - loss: 6.0558e-04 - mse: 6.0558e-04 - val_loss: 0.1121 - val_mse: 0.1121
    Epoch 104/150
    33/33 [==============================] - 0s 6ms/step - loss: 4.5368e-04 - mse: 4.5368e-04 - val_loss: 0.1077 - val_mse: 0.1077
    Epoch 105/150
    33/33 [==============================] - 0s 6ms/step - loss: 4.0104e-04 - mse: 4.0104e-04 - val_loss: 0.1118 - val_mse: 0.1118
    Epoch 106/150
    33/33 [==============================] - 0s 6ms/step - loss: 3.9629e-04 - mse: 3.9629e-04 - val_loss: 0.1101 - val_mse: 0.1101
    Epoch 107/150
    33/33 [==============================] - 0s 6ms/step - loss: 4.1355e-04 - mse: 4.1355e-04 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 108/150
    33/33 [==============================] - 0s 6ms/step - loss: 5.6756e-04 - mse: 5.6756e-04 - val_loss: 0.1115 - val_mse: 0.1115
    Epoch 109/150
    33/33 [==============================] - 0s 6ms/step - loss: 5.6044e-04 - mse: 5.6044e-04 - val_loss: 0.1142 - val_mse: 0.1142
    Epoch 110/150
    33/33 [==============================] - 0s 6ms/step - loss: 7.3796e-04 - mse: 7.3796e-04 - val_loss: 0.1117 - val_mse: 0.1117
    Epoch 111/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.2734e-04 - mse: 8.2734e-04 - val_loss: 0.1097 - val_mse: 0.1097
    Epoch 112/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0014 - mse: 0.0014 - val_loss: 0.1167 - val_mse: 0.1167
    Epoch 113/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0020 - mse: 0.0020 - val_loss: 0.1134 - val_mse: 0.1134
    Epoch 114/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0019 - mse: 0.0019 - val_loss: 0.1158 - val_mse: 0.1158
    Epoch 115/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0024 - mse: 0.0024 - val_loss: 0.1095 - val_mse: 0.1095
    Epoch 116/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0035 - mse: 0.0035 - val_loss: 0.1164 - val_mse: 0.1164
    Epoch 117/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0037 - mse: 0.0037 - val_loss: 0.1021 - val_mse: 0.1021
    Epoch 118/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0039 - mse: 0.0039 - val_loss: 0.1114 - val_mse: 0.1114
    Epoch 119/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.1029 - val_mse: 0.1029
    Epoch 120/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.1116 - val_mse: 0.1116
    Epoch 121/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1106 - val_mse: 0.1106
    Epoch 122/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.1084 - val_mse: 0.1084
    Epoch 123/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.1033 - val_mse: 0.1033
    Epoch 124/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0030 - mse: 0.0030 - val_loss: 0.1099 - val_mse: 0.1099
    Epoch 125/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0019 - mse: 0.0019 - val_loss: 0.1029 - val_mse: 0.1029
    Epoch 126/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0018 - mse: 0.0018 - val_loss: 0.1111 - val_mse: 0.1111
    Epoch 127/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0015 - mse: 0.0015 - val_loss: 0.1042 - val_mse: 0.1042
    Epoch 128/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0012 - mse: 0.0012 - val_loss: 0.1065 - val_mse: 0.1065
    Epoch 129/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0010 - mse: 0.0010 - val_loss: 0.1075 - val_mse: 0.1075
    Epoch 130/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0010 - mse: 0.0010 - val_loss: 0.1050 - val_mse: 0.1050
    Epoch 131/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0021 - mse: 0.0021 - val_loss: 0.1085 - val_mse: 0.1085
    Epoch 132/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0021 - mse: 0.0021 - val_loss: 0.1032 - val_mse: 0.1032
    Epoch 133/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0026 - mse: 0.0026 - val_loss: 0.1057 - val_mse: 0.1057
    Epoch 134/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.1024 - val_mse: 0.1024
    Epoch 135/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1127 - val_mse: 0.1127
    Epoch 136/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.0984 - val_mse: 0.0984
    Epoch 137/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.1201 - val_mse: 0.1201
    Epoch 138/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.0968 - val_mse: 0.0968
    Epoch 139/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.0990 - val_mse: 0.0990
    Epoch 140/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.0963 - val_mse: 0.0963
    Epoch 141/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0059 - mse: 0.0059 - val_loss: 0.1239 - val_mse: 0.1239
    Epoch 142/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0059 - mse: 0.0059 - val_loss: 0.0976 - val_mse: 0.0976
    Epoch 143/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0063 - mse: 0.0063 - val_loss: 0.1064 - val_mse: 0.1064
    Epoch 144/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.1036 - val_mse: 0.1036
    Epoch 145/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 146/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.1017 - val_mse: 0.1017
    Epoch 147/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.1069 - val_mse: 0.1069
    Epoch 148/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.1081 - val_mse: 0.1081
    Epoch 149/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0019 - mse: 0.0019 - val_loss: 0.1022 - val_mse: 0.1022
    Epoch 150/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0012 - mse: 0.0012 - val_loss: 0.1052 - val_mse: 0.1052





    <keras.callbacks.History at 0x17ac03c10>



Evaluate the model (`adam_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
adam_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 4ms/step - loss: 8.5762e-04 - mse: 8.5762e-04





    [0.0008576181717216969, 0.0008576181717216969]



Evaluate the model (`adam_model`) on training data (`X_val` and `y_val_scaled`) 


```python
# Evaluate the model on validate data
adam_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 3ms/step - loss: 0.1052 - mse: 0.1052





    [0.105233334004879, 0.105233334004879]



## Select a Final Model

Now, select the model with the best performance based on the training and validation sets. Evaluate this top model using the test set!


```python
# Evaluate the best model on test data
rmsprop_model.evaluate(X_test, y_test_scaled)
```

    5/5 [==============================] - 0s 7ms/step - loss: 0.2161 - mse: 0.2161





    [0.21607691049575806, 0.21607691049575806]



As earlier, this metric is hard to interpret because the output is scaled. 

- Generate predictions on test data (`X_test`) 
- Transform these predictions back to original scale using `ss_y` 
- Now you can calculate the RMSE in the original units with `y_test` and `y_test_pred` 


```python
# Generate predictions on test data
y_test_pred_scaled = rmsprop_model.predict(X_test)

# Transform the predictions back to original scale
y_test_pred = ss_y.inverse_transform(y_test_pred_scaled)

# MSE of test data
np.sqrt(mean_squared_error(y_test, y_test_pred))
```

    5/5 [==============================] - 0s 4ms/step


    2023-04-14 18:37:05.993815: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.





    36528.82298654173



## Summary  

In this lab, you worked to ensure your model converged properly by normalizing both the input and output. Additionally, you also investigated the impact of varying initialization and optimization routines.
