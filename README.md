
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

    Using TensorFlow backend.


In this lab, you'll be working with the housing prices data you saw in an earlier section. However, we did a lot of preprocessing for you so you can focus on normalizing numeric features and building neural network models! The following preprocessing steps were taken (all the code can be found in the `data_preprocessing.ipynb` notebook in this repository): 

- The data was split into the training, validate, and test sets 
- All the missing values in numeric columns were replaced by the median of those columns 
- All the missing values in catetgorical columns were replaced with the word 'missing' 
- All the categorical columns were one-hot encoded 

Run the following cells to import the train, validate, and test sets:  


```python
# Load all numeric features
X_train_numeric = pd.read_csv('data/X_train_numeric.csv')
X_val_numeric = pd.read_csv('data/X_val_numeric.csv')
X_test_numeric = pd.read_csv('data/X_test_numeric.csv')

# Load all categorical features
X_train_cat = pd.read_csv('data/X_train_cat.csv')
X_val_cat = pd.read_csv('data/X_val_cat.csv')
X_test_cat = pd.read_csv('data/X_test_cat.csv')

# Load all targets
y_train = pd.read_csv('data/y_train.csv')
y_val = pd.read_csv('data/y_val.csv')
y_test = pd.read_csv('data/y_test.csv')
```


```python
# Combine all features
X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
X_val = pd.concat([X_val_numeric, X_val_cat], axis=1)
X_test = pd.concat([X_test_numeric, X_test_cat], axis=1)

# Number of features
n_features = X_train.shape[1]
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
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
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
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 296 columns</p>
</div>



## Build a Baseline Model

Building a naive baseline model to compare performance against is a helpful reference point. From there, you can then observe the impact of various tunning procedures which will iteratively improve your model. So, let's do just that! 

In the cell below: 

- Add an input layer with `n_features` units 
- Add two hidden layers, one with 100 and the other with 50 units (make sure you use the `'relu'` activation function) 
- Add an output layer with 1 unit and `'linear'` activation 
- Compile and fit the model 


```python
np.random.seed(123)
baseline_model = Sequential()

# Hidden layer with 100 units
baseline_model.add(layers.Dense(100, activation='relu', input_shape=(n_features,)))

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

    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    
    Train on 1051 samples, validate on 263 samples
    Epoch 1/150
    1051/1051 [==============================] - 0s 471us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 2/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 3/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 4/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 5/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 6/150
    1051/1051 [==============================] - 0s 66us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 7/150
    1051/1051 [==============================] - 0s 65us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 8/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 9/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 10/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 11/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 12/150
    1051/1051 [==============================] - 0s 65us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 13/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 14/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 15/150
    1051/1051 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 16/150
    1051/1051 [==============================] - 0s 66us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 17/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 18/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 19/150
    1051/1051 [==============================] - 0s 67us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 20/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 21/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 22/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 23/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 24/150
    1051/1051 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 25/150
    1051/1051 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 26/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 27/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 28/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 29/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 30/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 31/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 32/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 33/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 34/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 35/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 36/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 37/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 38/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 39/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 40/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 41/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 42/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 43/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 44/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 45/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 46/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 47/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 48/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 49/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 50/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 51/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 52/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 53/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 54/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 55/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 56/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 57/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 58/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 59/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 60/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 61/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 62/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 63/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 64/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 65/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 66/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 67/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 68/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 69/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 70/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 71/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 72/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 73/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 74/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 75/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 76/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 77/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 78/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 79/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 80/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 81/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 82/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 83/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 84/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 85/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 86/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 87/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 88/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 89/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 90/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 91/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 92/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 93/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 94/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 95/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 96/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 97/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 98/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 99/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 100/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 101/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 102/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 103/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 104/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 105/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 106/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 107/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 108/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 109/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 110/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 111/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 112/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 113/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 114/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 115/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 116/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 117/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 118/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 119/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 120/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 121/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 122/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 123/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 124/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 125/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 126/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 127/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 128/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 129/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 130/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 131/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 132/150
    1051/1051 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 133/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 134/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 135/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 136/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 137/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 138/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 139/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 140/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 141/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 142/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 143/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 144/150
    1051/1051 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 145/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 146/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 147/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 148/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 149/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 150/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan





    <keras.callbacks.History at 0x110aaf278>



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
normalized_input_model.add(layers.Dense(100, activation='relu', input_shape=(n_features,)))
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

    Train on 1051 samples, validate on 263 samples
    Epoch 1/150
    1051/1051 [==============================] - 0s 461us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 2/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 3/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 4/150
    1051/1051 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 5/150
    1051/1051 [==============================] - 0s 68us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 6/150
    1051/1051 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 7/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 8/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 9/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 10/150
    1051/1051 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 11/150
    1051/1051 [==============================] - 0s 66us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 12/150
    1051/1051 [==============================] - 0s 71us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 13/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 14/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 15/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 16/150
    1051/1051 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 17/150
    1051/1051 [==============================] - 0s 69us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 18/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 19/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 20/150
    1051/1051 [==============================] - 0s 66us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 21/150
    1051/1051 [==============================] - 0s 68us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 22/150
    1051/1051 [==============================] - 0s 69us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 23/150
    1051/1051 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 24/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 25/150
    1051/1051 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 26/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 27/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 28/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 29/150
    1051/1051 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 30/150
    1051/1051 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 31/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 32/150
    1051/1051 [==============================] - 0s 66us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 33/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 34/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 35/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 36/150
    1051/1051 [==============================] - 0s 66us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 37/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 38/150
    1051/1051 [==============================] - 0s 49us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 39/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 40/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 41/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 42/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 43/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 44/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 45/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 46/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 47/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 48/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 49/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 50/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 51/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 52/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 53/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 54/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 55/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 56/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 57/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 58/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 59/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 60/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 61/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 62/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 63/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 64/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 65/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 66/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 67/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 68/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 69/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 70/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 71/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 72/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 73/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 74/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 75/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 76/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 77/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 78/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 79/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 80/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 81/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 82/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 83/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 84/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 85/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 86/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 87/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 88/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 89/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 90/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 91/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 92/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 93/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 94/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 95/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 96/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 97/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 98/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 99/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 100/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 101/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 102/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 103/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 104/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 105/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 106/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 107/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 108/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 109/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 110/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 111/150
    1051/1051 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 112/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 113/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 114/150
    1051/1051 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 115/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 116/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 117/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 118/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 119/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 120/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 121/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 122/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 123/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 124/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 125/150
    1051/1051 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 126/150
    1051/1051 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 127/150
    1051/1051 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 128/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 129/150
    1051/1051 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 130/150
    1051/1051 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 131/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 132/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 133/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 134/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 135/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 136/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 137/150
    1051/1051 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 138/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 139/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 140/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 141/150
    1051/1051 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 142/150
    1051/1051 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 143/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 144/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 145/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 146/150
    1051/1051 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 147/150
    1051/1051 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 148/150
    1051/1051 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 149/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 150/150
    1051/1051 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan





    <keras.callbacks.History at 0x1a6cd29f60>



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
normalized_model.add(layers.Dense(100, activation='relu', input_shape=(n_features,)))
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
                     validation_data=(X_val, y_val_scaled))
```

    Train on 1051 samples, validate on 263 samples
    Epoch 1/150
    1051/1051 [==============================] - 1s 484us/step - loss: 0.4368 - mean_squared_error: 0.4368 - val_loss: 0.1992 - val_mean_squared_error: 0.1992
    Epoch 2/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.2275 - mean_squared_error: 0.2275 - val_loss: 0.1614 - val_mean_squared_error: 0.1614
    Epoch 3/150
    1051/1051 [==============================] - 0s 68us/step - loss: 0.1841 - mean_squared_error: 0.1841 - val_loss: 0.1589 - val_mean_squared_error: 0.1589
    Epoch 4/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.1643 - mean_squared_error: 0.1643 - val_loss: 0.1376 - val_mean_squared_error: 0.1376
    Epoch 5/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.1480 - mean_squared_error: 0.1480 - val_loss: 0.1321 - val_mean_squared_error: 0.1321
    Epoch 6/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.1330 - mean_squared_error: 0.1330 - val_loss: 0.1277 - val_mean_squared_error: 0.1277
    Epoch 7/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.1266 - mean_squared_error: 0.1266 - val_loss: 0.1307 - val_mean_squared_error: 0.1307
    Epoch 8/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.1185 - mean_squared_error: 0.1185 - val_loss: 0.1222 - val_mean_squared_error: 0.1222
    Epoch 9/150
    1051/1051 [==============================] - 0s 69us/step - loss: 0.1131 - mean_squared_error: 0.1131 - val_loss: 0.1209 - val_mean_squared_error: 0.1209
    Epoch 10/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.1108 - mean_squared_error: 0.1108 - val_loss: 0.1146 - val_mean_squared_error: 0.1146
    Epoch 11/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.1010 - mean_squared_error: 0.1010 - val_loss: 0.1101 - val_mean_squared_error: 0.1101
    Epoch 12/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0944 - mean_squared_error: 0.0944 - val_loss: 0.1096 - val_mean_squared_error: 0.1096
    Epoch 13/150
    1051/1051 [==============================] - 0s 71us/step - loss: 0.0925 - mean_squared_error: 0.0925 - val_loss: 0.1096 - val_mean_squared_error: 0.1096
    Epoch 14/150
    1051/1051 [==============================] - 0s 67us/step - loss: 0.0879 - mean_squared_error: 0.0879 - val_loss: 0.1066 - val_mean_squared_error: 0.1066
    Epoch 15/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0855 - mean_squared_error: 0.0855 - val_loss: 0.1059 - val_mean_squared_error: 0.1059
    Epoch 16/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0811 - mean_squared_error: 0.0811 - val_loss: 0.1050 - val_mean_squared_error: 0.1050
    Epoch 17/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0797 - mean_squared_error: 0.0797 - val_loss: 0.1032 - val_mean_squared_error: 0.1032
    Epoch 18/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0763 - mean_squared_error: 0.0763 - val_loss: 0.1019 - val_mean_squared_error: 0.1019
    Epoch 19/150
    1051/1051 [==============================] - 0s 69us/step - loss: 0.0721 - mean_squared_error: 0.0721 - val_loss: 0.0991 - val_mean_squared_error: 0.0991
    Epoch 20/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0698 - mean_squared_error: 0.0698 - val_loss: 0.1021 - val_mean_squared_error: 0.1021
    Epoch 21/150
    1051/1051 [==============================] - 0s 68us/step - loss: 0.0666 - mean_squared_error: 0.0666 - val_loss: 0.1024 - val_mean_squared_error: 0.1024
    Epoch 22/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0642 - mean_squared_error: 0.0642 - val_loss: 0.1081 - val_mean_squared_error: 0.1081
    Epoch 23/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0640 - mean_squared_error: 0.0640 - val_loss: 0.1000 - val_mean_squared_error: 0.1000
    Epoch 24/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0616 - mean_squared_error: 0.0616 - val_loss: 0.1003 - val_mean_squared_error: 0.1003
    Epoch 25/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0603 - mean_squared_error: 0.0603 - val_loss: 0.0977 - val_mean_squared_error: 0.0977
    Epoch 26/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0575 - mean_squared_error: 0.0575 - val_loss: 0.1004 - val_mean_squared_error: 0.1004
    Epoch 27/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0551 - mean_squared_error: 0.0551 - val_loss: 0.1010 - val_mean_squared_error: 0.1010
    Epoch 28/150
    1051/1051 [==============================] - 0s 67us/step - loss: 0.0547 - mean_squared_error: 0.0547 - val_loss: 0.0983 - val_mean_squared_error: 0.0983
    Epoch 29/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0530 - mean_squared_error: 0.0530 - val_loss: 0.0981 - val_mean_squared_error: 0.0981
    Epoch 30/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0510 - mean_squared_error: 0.0510 - val_loss: 0.0992 - val_mean_squared_error: 0.0992
    Epoch 31/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0498 - mean_squared_error: 0.0498 - val_loss: 0.0967 - val_mean_squared_error: 0.0967
    Epoch 32/150
    1051/1051 [==============================] - 0s 68us/step - loss: 0.0493 - mean_squared_error: 0.0493 - val_loss: 0.0980 - val_mean_squared_error: 0.0980
    Epoch 33/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0472 - mean_squared_error: 0.0472 - val_loss: 0.0974 - val_mean_squared_error: 0.0974
    Epoch 34/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0460 - mean_squared_error: 0.0460 - val_loss: 0.0998 - val_mean_squared_error: 0.0998
    Epoch 35/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0442 - mean_squared_error: 0.0442 - val_loss: 0.1023 - val_mean_squared_error: 0.1023
    Epoch 36/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0438 - mean_squared_error: 0.0438 - val_loss: 0.0997 - val_mean_squared_error: 0.0997
    Epoch 37/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0427 - mean_squared_error: 0.0427 - val_loss: 0.0986 - val_mean_squared_error: 0.0986
    Epoch 38/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0423 - mean_squared_error: 0.0423 - val_loss: 0.0996 - val_mean_squared_error: 0.0996
    Epoch 39/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0413 - mean_squared_error: 0.0413 - val_loss: 0.0986 - val_mean_squared_error: 0.0986
    Epoch 40/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0399 - mean_squared_error: 0.0399 - val_loss: 0.0999 - val_mean_squared_error: 0.0999
    Epoch 41/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0387 - mean_squared_error: 0.0387 - val_loss: 0.0998 - val_mean_squared_error: 0.0998
    Epoch 42/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0372 - mean_squared_error: 0.0372 - val_loss: 0.1016 - val_mean_squared_error: 0.1016
    Epoch 43/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0375 - mean_squared_error: 0.0375 - val_loss: 0.1013 - val_mean_squared_error: 0.1013
    Epoch 44/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0362 - mean_squared_error: 0.0362 - val_loss: 0.1013 - val_mean_squared_error: 0.1013
    Epoch 45/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0361 - mean_squared_error: 0.0361 - val_loss: 0.1011 - val_mean_squared_error: 0.1011
    Epoch 46/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0351 - mean_squared_error: 0.0351 - val_loss: 0.1020 - val_mean_squared_error: 0.1020
    Epoch 47/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0344 - mean_squared_error: 0.0344 - val_loss: 0.1029 - val_mean_squared_error: 0.1029
    Epoch 48/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0339 - mean_squared_error: 0.0339 - val_loss: 0.1014 - val_mean_squared_error: 0.1014
    Epoch 49/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0327 - mean_squared_error: 0.0327 - val_loss: 0.1031 - val_mean_squared_error: 0.1031
    Epoch 50/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0318 - mean_squared_error: 0.0318 - val_loss: 0.1015 - val_mean_squared_error: 0.1015
    Epoch 51/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0316 - mean_squared_error: 0.0316 - val_loss: 0.1028 - val_mean_squared_error: 0.1028
    Epoch 52/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0309 - mean_squared_error: 0.0309 - val_loss: 0.1031 - val_mean_squared_error: 0.1031
    Epoch 53/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0301 - mean_squared_error: 0.0301 - val_loss: 0.1031 - val_mean_squared_error: 0.1031
    Epoch 54/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0292 - mean_squared_error: 0.0292 - val_loss: 0.1046 - val_mean_squared_error: 0.1046
    Epoch 55/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0286 - mean_squared_error: 0.0286 - val_loss: 0.1041 - val_mean_squared_error: 0.1041
    Epoch 56/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0281 - mean_squared_error: 0.0281 - val_loss: 0.1035 - val_mean_squared_error: 0.1035
    Epoch 57/150
    1051/1051 [==============================] - 0s 49us/step - loss: 0.0276 - mean_squared_error: 0.0276 - val_loss: 0.1033 - val_mean_squared_error: 0.1033
    Epoch 58/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0271 - mean_squared_error: 0.0271 - val_loss: 0.1035 - val_mean_squared_error: 0.1035
    Epoch 59/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0262 - mean_squared_error: 0.0262 - val_loss: 0.1056 - val_mean_squared_error: 0.1056
    Epoch 60/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0259 - mean_squared_error: 0.0259 - val_loss: 0.1064 - val_mean_squared_error: 0.1064
    Epoch 61/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0255 - mean_squared_error: 0.0255 - val_loss: 0.1044 - val_mean_squared_error: 0.1044
    Epoch 62/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0249 - mean_squared_error: 0.0249 - val_loss: 0.1093 - val_mean_squared_error: 0.1093
    Epoch 63/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0249 - mean_squared_error: 0.0249 - val_loss: 0.1064 - val_mean_squared_error: 0.1064
    Epoch 64/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0240 - mean_squared_error: 0.0240 - val_loss: 0.1072 - val_mean_squared_error: 0.1072
    Epoch 65/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0235 - mean_squared_error: 0.0235 - val_loss: 0.1059 - val_mean_squared_error: 0.1059
    Epoch 66/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0230 - mean_squared_error: 0.0230 - val_loss: 0.1072 - val_mean_squared_error: 0.1072
    Epoch 67/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0228 - mean_squared_error: 0.0228 - val_loss: 0.1072 - val_mean_squared_error: 0.1072
    Epoch 68/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0228 - mean_squared_error: 0.0228 - val_loss: 0.1074 - val_mean_squared_error: 0.1074
    Epoch 69/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0223 - mean_squared_error: 0.0223 - val_loss: 0.1062 - val_mean_squared_error: 0.1062
    Epoch 70/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0216 - mean_squared_error: 0.0216 - val_loss: 0.1060 - val_mean_squared_error: 0.1060
    Epoch 71/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0213 - mean_squared_error: 0.0213 - val_loss: 0.1076 - val_mean_squared_error: 0.1076
    Epoch 72/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0209 - mean_squared_error: 0.0209 - val_loss: 0.1075 - val_mean_squared_error: 0.1075
    Epoch 73/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0205 - mean_squared_error: 0.0205 - val_loss: 0.1082 - val_mean_squared_error: 0.1082
    Epoch 74/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0203 - mean_squared_error: 0.0203 - val_loss: 0.1087 - val_mean_squared_error: 0.1087
    Epoch 75/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0200 - mean_squared_error: 0.0200 - val_loss: 0.1084 - val_mean_squared_error: 0.1084
    Epoch 76/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0197 - mean_squared_error: 0.0197 - val_loss: 0.1124 - val_mean_squared_error: 0.1124
    Epoch 77/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0199 - mean_squared_error: 0.0199 - val_loss: 0.1086 - val_mean_squared_error: 0.1086
    Epoch 78/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0192 - mean_squared_error: 0.0192 - val_loss: 0.1088 - val_mean_squared_error: 0.1088
    Epoch 79/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0186 - mean_squared_error: 0.0186 - val_loss: 0.1087 - val_mean_squared_error: 0.1087
    Epoch 80/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0186 - mean_squared_error: 0.0186 - val_loss: 0.1094 - val_mean_squared_error: 0.1094
    Epoch 81/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0182 - mean_squared_error: 0.0182 - val_loss: 0.1099 - val_mean_squared_error: 0.1099
    Epoch 82/150
    1051/1051 [==============================] - 0s 68us/step - loss: 0.0182 - mean_squared_error: 0.0182 - val_loss: 0.1100 - val_mean_squared_error: 0.1100
    Epoch 83/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.1087 - val_mean_squared_error: 0.1087
    Epoch 84/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0176 - mean_squared_error: 0.0176 - val_loss: 0.1095 - val_mean_squared_error: 0.1095
    Epoch 85/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0172 - mean_squared_error: 0.0172 - val_loss: 0.1106 - val_mean_squared_error: 0.1106
    Epoch 86/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0168 - mean_squared_error: 0.0168 - val_loss: 0.1093 - val_mean_squared_error: 0.1093
    Epoch 87/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0167 - mean_squared_error: 0.0167 - val_loss: 0.1101 - val_mean_squared_error: 0.1101
    Epoch 88/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0168 - mean_squared_error: 0.0168 - val_loss: 0.1094 - val_mean_squared_error: 0.1094
    Epoch 89/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0162 - mean_squared_error: 0.0162 - val_loss: 0.1115 - val_mean_squared_error: 0.1115
    Epoch 90/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0160 - mean_squared_error: 0.0160 - val_loss: 0.1113 - val_mean_squared_error: 0.1113
    Epoch 91/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0160 - mean_squared_error: 0.0160 - val_loss: 0.1099 - val_mean_squared_error: 0.1099
    Epoch 92/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0157 - mean_squared_error: 0.0157 - val_loss: 0.1108 - val_mean_squared_error: 0.1108
    Epoch 93/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0156 - mean_squared_error: 0.0156 - val_loss: 0.1110 - val_mean_squared_error: 0.1110
    Epoch 94/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0154 - mean_squared_error: 0.0154 - val_loss: 0.1118 - val_mean_squared_error: 0.1118
    Epoch 95/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0149 - mean_squared_error: 0.0149 - val_loss: 0.1123 - val_mean_squared_error: 0.1123
    Epoch 96/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0147 - mean_squared_error: 0.0147 - val_loss: 0.1119 - val_mean_squared_error: 0.1119
    Epoch 97/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0147 - mean_squared_error: 0.0147 - val_loss: 0.1107 - val_mean_squared_error: 0.1107
    Epoch 98/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0144 - mean_squared_error: 0.0144 - val_loss: 0.1110 - val_mean_squared_error: 0.1110
    Epoch 99/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0143 - mean_squared_error: 0.0143 - val_loss: 0.1125 - val_mean_squared_error: 0.1125
    Epoch 100/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0141 - mean_squared_error: 0.0141 - val_loss: 0.1117 - val_mean_squared_error: 0.1117
    Epoch 101/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0140 - mean_squared_error: 0.0140 - val_loss: 0.1130 - val_mean_squared_error: 0.1130
    Epoch 102/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0137 - mean_squared_error: 0.0137 - val_loss: 0.1123 - val_mean_squared_error: 0.1123
    Epoch 103/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.1127 - val_mean_squared_error: 0.1127
    Epoch 104/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0133 - mean_squared_error: 0.0133 - val_loss: 0.1134 - val_mean_squared_error: 0.1134
    Epoch 105/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0133 - mean_squared_error: 0.0133 - val_loss: 0.1127 - val_mean_squared_error: 0.1127
    Epoch 106/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0130 - mean_squared_error: 0.0130 - val_loss: 0.1129 - val_mean_squared_error: 0.1129
    Epoch 107/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0129 - mean_squared_error: 0.0129 - val_loss: 0.1135 - val_mean_squared_error: 0.1135
    Epoch 108/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0128 - mean_squared_error: 0.0128 - val_loss: 0.1129 - val_mean_squared_error: 0.1129
    Epoch 109/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0125 - mean_squared_error: 0.0125 - val_loss: 0.1136 - val_mean_squared_error: 0.1136
    Epoch 110/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0125 - mean_squared_error: 0.0125 - val_loss: 0.1136 - val_mean_squared_error: 0.1136
    Epoch 111/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0123 - mean_squared_error: 0.0123 - val_loss: 0.1131 - val_mean_squared_error: 0.1131
    Epoch 112/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0121 - mean_squared_error: 0.0121 - val_loss: 0.1139 - val_mean_squared_error: 0.1139
    Epoch 113/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0120 - mean_squared_error: 0.0120 - val_loss: 0.1149 - val_mean_squared_error: 0.1149
    Epoch 114/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0120 - mean_squared_error: 0.0120 - val_loss: 0.1143 - val_mean_squared_error: 0.1143
    Epoch 115/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0117 - mean_squared_error: 0.0117 - val_loss: 0.1142 - val_mean_squared_error: 0.1142
    Epoch 116/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0115 - mean_squared_error: 0.0115 - val_loss: 0.1146 - val_mean_squared_error: 0.1146
    Epoch 117/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0113 - mean_squared_error: 0.0113 - val_loss: 0.1152 - val_mean_squared_error: 0.1152
    Epoch 118/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0114 - mean_squared_error: 0.0114 - val_loss: 0.1140 - val_mean_squared_error: 0.1140
    Epoch 119/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0113 - mean_squared_error: 0.0113 - val_loss: 0.1151 - val_mean_squared_error: 0.1151
    Epoch 120/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0110 - mean_squared_error: 0.0110 - val_loss: 0.1154 - val_mean_squared_error: 0.1154
    Epoch 121/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0110 - mean_squared_error: 0.0110 - val_loss: 0.1156 - val_mean_squared_error: 0.1156
    Epoch 122/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0108 - mean_squared_error: 0.0108 - val_loss: 0.1156 - val_mean_squared_error: 0.1156
    Epoch 123/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0109 - mean_squared_error: 0.0109 - val_loss: 0.1158 - val_mean_squared_error: 0.1158
    Epoch 124/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0106 - mean_squared_error: 0.0106 - val_loss: 0.1137 - val_mean_squared_error: 0.1137
    Epoch 125/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0106 - mean_squared_error: 0.0106 - val_loss: 0.1162 - val_mean_squared_error: 0.1162
    Epoch 126/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.1159 - val_mean_squared_error: 0.1159
    Epoch 127/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.1161 - val_mean_squared_error: 0.1161
    Epoch 128/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.1165 - val_mean_squared_error: 0.1165
    Epoch 129/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.1160 - val_mean_squared_error: 0.1160
    Epoch 130/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.1154 - val_mean_squared_error: 0.1154
    Epoch 131/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0098 - mean_squared_error: 0.0098 - val_loss: 0.1161 - val_mean_squared_error: 0.1161
    Epoch 132/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.1167 - val_mean_squared_error: 0.1167
    Epoch 133/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0096 - mean_squared_error: 0.0096 - val_loss: 0.1167 - val_mean_squared_error: 0.1167
    Epoch 134/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0095 - mean_squared_error: 0.0095 - val_loss: 0.1160 - val_mean_squared_error: 0.1160
    Epoch 135/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.1166 - val_mean_squared_error: 0.1166
    Epoch 136/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.1172 - val_mean_squared_error: 0.1172
    Epoch 137/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0092 - mean_squared_error: 0.0092 - val_loss: 0.1165 - val_mean_squared_error: 0.1165
    Epoch 138/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.1168 - val_mean_squared_error: 0.1168
    Epoch 139/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.1186 - val_mean_squared_error: 0.1186
    Epoch 140/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0090 - mean_squared_error: 0.0090 - val_loss: 0.1176 - val_mean_squared_error: 0.1176
    Epoch 141/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.1176 - val_mean_squared_error: 0.1176
    Epoch 142/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.1175 - val_mean_squared_error: 0.1175
    Epoch 143/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.1174 - val_mean_squared_error: 0.1174
    Epoch 144/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.1187 - val_mean_squared_error: 0.1187
    Epoch 145/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.1179 - val_mean_squared_error: 0.1179
    Epoch 146/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.1172 - val_mean_squared_error: 0.1172
    Epoch 147/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.1178 - val_mean_squared_error: 0.1178
    Epoch 148/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.1179 - val_mean_squared_error: 0.1179
    Epoch 149/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.1176 - val_mean_squared_error: 0.1176
    Epoch 150/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.1187 - val_mean_squared_error: 0.1187





    <keras.callbacks.History at 0x1a6cdbf2e8>



Nicely done! After normalizing both the input and output, the model finally converged. 

- Evaluate the model (`normalized_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
normalized_model.evaluate(X_train, y_train_scaled)
```

    1051/1051 [==============================] - 0s 23us/step





    [0.0077856800507410265, 0.0077856800507410265]



- Evaluate the model (`normalized_model`) on validate data (`X_val` and `y_val_scaled`) 


```python
# Evaluate the model on validate data
normalized_model.evaluate(X_val, y_val_scaled)
```

    263/263 [==============================] - 0s 31us/step





    [0.1187465712144801, 0.1187465712144801]



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




    27079.587222444236



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
he_model.add(layers.Dense(100, kernel_initializer='he_normal', activation='relu', input_shape=(n_features,)))

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

    Train on 1051 samples, validate on 263 samples
    Epoch 1/150
    1051/1051 [==============================] - 1s 530us/step - loss: 0.5084 - mean_squared_error: 0.5084 - val_loss: 0.1703 - val_mean_squared_error: 0.1703
    Epoch 2/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.2216 - mean_squared_error: 0.2216 - val_loss: 0.1432 - val_mean_squared_error: 0.1432
    Epoch 3/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.1793 - mean_squared_error: 0.1793 - val_loss: 0.1376 - val_mean_squared_error: 0.1376
    Epoch 4/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.1567 - mean_squared_error: 0.1567 - val_loss: 0.1250 - val_mean_squared_error: 0.1250
    Epoch 5/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.1403 - mean_squared_error: 0.1403 - val_loss: 0.1192 - val_mean_squared_error: 0.1192
    Epoch 6/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.1262 - mean_squared_error: 0.1262 - val_loss: 0.1151 - val_mean_squared_error: 0.1151
    Epoch 7/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.1193 - mean_squared_error: 0.1193 - val_loss: 0.1167 - val_mean_squared_error: 0.1167
    Epoch 8/150
    1051/1051 [==============================] - 0s 67us/step - loss: 0.1095 - mean_squared_error: 0.1095 - val_loss: 0.1133 - val_mean_squared_error: 0.1133
    Epoch 9/150
    1051/1051 [==============================] - 0s 70us/step - loss: 0.1063 - mean_squared_error: 0.1063 - val_loss: 0.1132 - val_mean_squared_error: 0.1132
    Epoch 10/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.1034 - mean_squared_error: 0.1034 - val_loss: 0.1133 - val_mean_squared_error: 0.1133
    Epoch 11/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0959 - mean_squared_error: 0.0959 - val_loss: 0.1086 - val_mean_squared_error: 0.1086
    Epoch 12/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0880 - mean_squared_error: 0.0880 - val_loss: 0.1080 - val_mean_squared_error: 0.1080
    Epoch 13/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0870 - mean_squared_error: 0.0870 - val_loss: 0.1107 - val_mean_squared_error: 0.1107
    Epoch 14/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0818 - mean_squared_error: 0.0818 - val_loss: 0.1073 - val_mean_squared_error: 0.1073
    Epoch 15/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0794 - mean_squared_error: 0.0794 - val_loss: 0.1058 - val_mean_squared_error: 0.1058
    Epoch 16/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0756 - mean_squared_error: 0.0756 - val_loss: 0.1066 - val_mean_squared_error: 0.1066
    Epoch 17/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0745 - mean_squared_error: 0.0745 - val_loss: 0.1046 - val_mean_squared_error: 0.1046
    Epoch 18/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0717 - mean_squared_error: 0.0717 - val_loss: 0.1046 - val_mean_squared_error: 0.1046
    Epoch 19/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0675 - mean_squared_error: 0.0675 - val_loss: 0.1037 - val_mean_squared_error: 0.1037
    Epoch 20/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0642 - mean_squared_error: 0.0642 - val_loss: 0.1046 - val_mean_squared_error: 0.1046
    Epoch 21/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0624 - mean_squared_error: 0.0624 - val_loss: 0.1040 - val_mean_squared_error: 0.1040
    Epoch 22/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0600 - mean_squared_error: 0.0600 - val_loss: 0.1081 - val_mean_squared_error: 0.1081
    Epoch 23/150
    1051/1051 [==============================] - 0s 69us/step - loss: 0.0593 - mean_squared_error: 0.0593 - val_loss: 0.1046 - val_mean_squared_error: 0.1046
    Epoch 24/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0571 - mean_squared_error: 0.0571 - val_loss: 0.1067 - val_mean_squared_error: 0.1067
    Epoch 25/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0560 - mean_squared_error: 0.0560 - val_loss: 0.1027 - val_mean_squared_error: 0.1027
    Epoch 26/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0535 - mean_squared_error: 0.0535 - val_loss: 0.1027 - val_mean_squared_error: 0.1027
    Epoch 27/150
    1051/1051 [==============================] - 0s 69us/step - loss: 0.0526 - mean_squared_error: 0.0526 - val_loss: 0.1050 - val_mean_squared_error: 0.1050
    Epoch 28/150
    1051/1051 [==============================] - 0s 69us/step - loss: 0.0509 - mean_squared_error: 0.0509 - val_loss: 0.1035 - val_mean_squared_error: 0.1035
    Epoch 29/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0498 - mean_squared_error: 0.0498 - val_loss: 0.1035 - val_mean_squared_error: 0.1035
    Epoch 30/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0482 - mean_squared_error: 0.0482 - val_loss: 0.1044 - val_mean_squared_error: 0.1044
    Epoch 31/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0466 - mean_squared_error: 0.0466 - val_loss: 0.1028 - val_mean_squared_error: 0.1028
    Epoch 32/150
    1051/1051 [==============================] - 0s 73us/step - loss: 0.0457 - mean_squared_error: 0.0457 - val_loss: 0.1035 - val_mean_squared_error: 0.1035
    Epoch 33/150
    1051/1051 [==============================] - 0s 73us/step - loss: 0.0452 - mean_squared_error: 0.0452 - val_loss: 0.1035 - val_mean_squared_error: 0.1035
    Epoch 34/150
    1051/1051 [==============================] - 0s 71us/step - loss: 0.0432 - mean_squared_error: 0.0432 - val_loss: 0.1058 - val_mean_squared_error: 0.1058
    Epoch 35/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0417 - mean_squared_error: 0.0417 - val_loss: 0.1071 - val_mean_squared_error: 0.1071
    Epoch 36/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0418 - mean_squared_error: 0.0418 - val_loss: 0.1049 - val_mean_squared_error: 0.1049
    Epoch 37/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0401 - mean_squared_error: 0.0401 - val_loss: 0.1055 - val_mean_squared_error: 0.1055
    Epoch 38/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0394 - mean_squared_error: 0.0394 - val_loss: 0.1041 - val_mean_squared_error: 0.1041
    Epoch 39/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0391 - mean_squared_error: 0.0391 - val_loss: 0.1050 - val_mean_squared_error: 0.1050
    Epoch 40/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0374 - mean_squared_error: 0.0374 - val_loss: 0.1047 - val_mean_squared_error: 0.1047
    Epoch 41/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0366 - mean_squared_error: 0.0366 - val_loss: 0.1058 - val_mean_squared_error: 0.1058
    Epoch 42/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0353 - mean_squared_error: 0.0353 - val_loss: 0.1064 - val_mean_squared_error: 0.1064
    Epoch 43/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0350 - mean_squared_error: 0.0350 - val_loss: 0.1066 - val_mean_squared_error: 0.1066
    Epoch 44/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0346 - mean_squared_error: 0.0346 - val_loss: 0.1072 - val_mean_squared_error: 0.1072
    Epoch 45/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0334 - mean_squared_error: 0.0334 - val_loss: 0.1080 - val_mean_squared_error: 0.1080
    Epoch 46/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0332 - mean_squared_error: 0.0332 - val_loss: 0.1070 - val_mean_squared_error: 0.1070
    Epoch 47/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0320 - mean_squared_error: 0.0320 - val_loss: 0.1096 - val_mean_squared_error: 0.1096
    Epoch 48/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0318 - mean_squared_error: 0.0318 - val_loss: 0.1084 - val_mean_squared_error: 0.1084
    Epoch 49/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0315 - mean_squared_error: 0.0315 - val_loss: 0.1092 - val_mean_squared_error: 0.1092
    Epoch 50/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0305 - mean_squared_error: 0.0305 - val_loss: 0.1077 - val_mean_squared_error: 0.1077
    Epoch 51/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0297 - mean_squared_error: 0.0297 - val_loss: 0.1076 - val_mean_squared_error: 0.1076
    Epoch 52/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0294 - mean_squared_error: 0.0294 - val_loss: 0.1086 - val_mean_squared_error: 0.1086
    Epoch 53/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0288 - mean_squared_error: 0.0288 - val_loss: 0.1086 - val_mean_squared_error: 0.1086
    Epoch 54/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0283 - mean_squared_error: 0.0283 - val_loss: 0.1091 - val_mean_squared_error: 0.1091
    Epoch 55/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0280 - mean_squared_error: 0.0280 - val_loss: 0.1083 - val_mean_squared_error: 0.1083
    Epoch 56/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0274 - mean_squared_error: 0.0274 - val_loss: 0.1085 - val_mean_squared_error: 0.1085
    Epoch 57/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0264 - mean_squared_error: 0.0264 - val_loss: 0.1091 - val_mean_squared_error: 0.1091
    Epoch 58/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0260 - mean_squared_error: 0.0260 - val_loss: 0.1101 - val_mean_squared_error: 0.1101
    Epoch 59/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0255 - mean_squared_error: 0.0255 - val_loss: 0.1132 - val_mean_squared_error: 0.1132
    Epoch 60/150
    1051/1051 [==============================] - 0s 49us/step - loss: 0.0250 - mean_squared_error: 0.0250 - val_loss: 0.1095 - val_mean_squared_error: 0.1095
    Epoch 61/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0247 - mean_squared_error: 0.0247 - val_loss: 0.1106 - val_mean_squared_error: 0.1106
    Epoch 62/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0245 - mean_squared_error: 0.0245 - val_loss: 0.1133 - val_mean_squared_error: 0.1133
    Epoch 63/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0244 - mean_squared_error: 0.0244 - val_loss: 0.1106 - val_mean_squared_error: 0.1106
    Epoch 64/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0242 - mean_squared_error: 0.0242 - val_loss: 0.1115 - val_mean_squared_error: 0.1115
    Epoch 65/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0236 - mean_squared_error: 0.0236 - val_loss: 0.1113 - val_mean_squared_error: 0.1113
    Epoch 66/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0225 - mean_squared_error: 0.0225 - val_loss: 0.1131 - val_mean_squared_error: 0.1131
    Epoch 67/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0228 - mean_squared_error: 0.0228 - val_loss: 0.1111 - val_mean_squared_error: 0.1111
    Epoch 68/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0221 - mean_squared_error: 0.0221 - val_loss: 0.1131 - val_mean_squared_error: 0.1131
    Epoch 69/150
    1051/1051 [==============================] - 0s 49us/step - loss: 0.0220 - mean_squared_error: 0.0220 - val_loss: 0.1113 - val_mean_squared_error: 0.1113
    Epoch 70/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0215 - mean_squared_error: 0.0215 - val_loss: 0.1117 - val_mean_squared_error: 0.1117
    Epoch 71/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0212 - mean_squared_error: 0.0212 - val_loss: 0.1111 - val_mean_squared_error: 0.1111
    Epoch 72/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0211 - mean_squared_error: 0.0211 - val_loss: 0.1120 - val_mean_squared_error: 0.1120
    Epoch 73/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0206 - mean_squared_error: 0.0206 - val_loss: 0.1120 - val_mean_squared_error: 0.1120
    Epoch 74/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0200 - mean_squared_error: 0.0200 - val_loss: 0.1164 - val_mean_squared_error: 0.1164
    Epoch 75/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0201 - mean_squared_error: 0.0201 - val_loss: 0.1126 - val_mean_squared_error: 0.1126
    Epoch 76/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0196 - mean_squared_error: 0.0196 - val_loss: 0.1177 - val_mean_squared_error: 0.1177
    Epoch 77/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0199 - mean_squared_error: 0.0199 - val_loss: 0.1125 - val_mean_squared_error: 0.1125
    Epoch 78/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0191 - mean_squared_error: 0.0191 - val_loss: 0.1126 - val_mean_squared_error: 0.1126
    Epoch 79/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0184 - mean_squared_error: 0.0184 - val_loss: 0.1128 - val_mean_squared_error: 0.1128
    Epoch 80/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0184 - mean_squared_error: 0.0184 - val_loss: 0.1132 - val_mean_squared_error: 0.1132
    Epoch 81/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0183 - mean_squared_error: 0.0183 - val_loss: 0.1140 - val_mean_squared_error: 0.1140
    Epoch 82/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0180 - mean_squared_error: 0.0180 - val_loss: 0.1159 - val_mean_squared_error: 0.1159
    Epoch 83/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0177 - mean_squared_error: 0.0177 - val_loss: 0.1138 - val_mean_squared_error: 0.1138
    Epoch 84/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.1134 - val_mean_squared_error: 0.1134
    Epoch 85/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0170 - mean_squared_error: 0.0170 - val_loss: 0.1140 - val_mean_squared_error: 0.1140
    Epoch 86/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0172 - mean_squared_error: 0.0172 - val_loss: 0.1140 - val_mean_squared_error: 0.1140
    Epoch 87/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0168 - mean_squared_error: 0.0168 - val_loss: 0.1153 - val_mean_squared_error: 0.1153
    Epoch 88/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0170 - mean_squared_error: 0.0170 - val_loss: 0.1142 - val_mean_squared_error: 0.1142
    Epoch 89/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0166 - mean_squared_error: 0.0166 - val_loss: 0.1142 - val_mean_squared_error: 0.1142
    Epoch 90/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0163 - mean_squared_error: 0.0163 - val_loss: 0.1147 - val_mean_squared_error: 0.1147
    Epoch 91/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0160 - mean_squared_error: 0.0160 - val_loss: 0.1157 - val_mean_squared_error: 0.1157
    Epoch 92/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0157 - mean_squared_error: 0.0157 - val_loss: 0.1146 - val_mean_squared_error: 0.1146
    Epoch 93/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0156 - mean_squared_error: 0.0156 - val_loss: 0.1158 - val_mean_squared_error: 0.1158
    Epoch 94/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0156 - mean_squared_error: 0.0156 - val_loss: 0.1167 - val_mean_squared_error: 0.1167
    Epoch 95/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0152 - mean_squared_error: 0.0152 - val_loss: 0.1162 - val_mean_squared_error: 0.1162
    Epoch 96/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0151 - mean_squared_error: 0.0151 - val_loss: 0.1152 - val_mean_squared_error: 0.1152
    Epoch 97/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0148 - mean_squared_error: 0.0148 - val_loss: 0.1152 - val_mean_squared_error: 0.1152
    Epoch 98/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0147 - mean_squared_error: 0.0147 - val_loss: 0.1156 - val_mean_squared_error: 0.1156
    Epoch 99/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0145 - mean_squared_error: 0.0145 - val_loss: 0.1170 - val_mean_squared_error: 0.1170
    Epoch 100/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0142 - mean_squared_error: 0.0142 - val_loss: 0.1167 - val_mean_squared_error: 0.1167
    Epoch 101/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0141 - mean_squared_error: 0.0141 - val_loss: 0.1166 - val_mean_squared_error: 0.1166
    Epoch 102/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0140 - mean_squared_error: 0.0140 - val_loss: 0.1160 - val_mean_squared_error: 0.1160
    Epoch 103/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0139 - mean_squared_error: 0.0139 - val_loss: 0.1160 - val_mean_squared_error: 0.1160
    Epoch 104/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.1183 - val_mean_squared_error: 0.1183
    Epoch 105/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0135 - mean_squared_error: 0.0135 - val_loss: 0.1167 - val_mean_squared_error: 0.1167
    Epoch 106/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0132 - mean_squared_error: 0.0132 - val_loss: 0.1177 - val_mean_squared_error: 0.1177
    Epoch 107/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0131 - mean_squared_error: 0.0131 - val_loss: 0.1169 - val_mean_squared_error: 0.1169
    Epoch 108/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0130 - mean_squared_error: 0.0130 - val_loss: 0.1175 - val_mean_squared_error: 0.1175
    Epoch 109/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0127 - mean_squared_error: 0.0127 - val_loss: 0.1175 - val_mean_squared_error: 0.1175
    Epoch 110/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0129 - mean_squared_error: 0.0129 - val_loss: 0.1177 - val_mean_squared_error: 0.1177
    Epoch 111/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0126 - mean_squared_error: 0.0126 - val_loss: 0.1187 - val_mean_squared_error: 0.1187
    Epoch 112/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0124 - mean_squared_error: 0.0124 - val_loss: 0.1182 - val_mean_squared_error: 0.1182
    Epoch 113/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0123 - mean_squared_error: 0.0123 - val_loss: 0.1182 - val_mean_squared_error: 0.1182
    Epoch 114/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0122 - mean_squared_error: 0.0122 - val_loss: 0.1188 - val_mean_squared_error: 0.1188
    Epoch 115/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0119 - mean_squared_error: 0.0119 - val_loss: 0.1176 - val_mean_squared_error: 0.1176
    Epoch 116/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0118 - mean_squared_error: 0.0118 - val_loss: 0.1181 - val_mean_squared_error: 0.1181
    Epoch 117/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0117 - mean_squared_error: 0.0117 - val_loss: 0.1192 - val_mean_squared_error: 0.1192
    Epoch 118/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0117 - mean_squared_error: 0.0117 - val_loss: 0.1189 - val_mean_squared_error: 0.1189
    Epoch 119/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0114 - mean_squared_error: 0.0114 - val_loss: 0.1193 - val_mean_squared_error: 0.1193
    Epoch 120/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0113 - mean_squared_error: 0.0113 - val_loss: 0.1195 - val_mean_squared_error: 0.1195
    Epoch 121/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0112 - mean_squared_error: 0.0112 - val_loss: 0.1195 - val_mean_squared_error: 0.1195
    Epoch 122/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0112 - mean_squared_error: 0.0112 - val_loss: 0.1191 - val_mean_squared_error: 0.1191
    Epoch 123/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0111 - mean_squared_error: 0.0111 - val_loss: 0.1190 - val_mean_squared_error: 0.1190
    Epoch 124/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0110 - mean_squared_error: 0.0110 - val_loss: 0.1185 - val_mean_squared_error: 0.1185
    Epoch 125/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0108 - mean_squared_error: 0.0108 - val_loss: 0.1233 - val_mean_squared_error: 0.1233
    Epoch 126/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0107 - mean_squared_error: 0.0107 - val_loss: 0.1198 - val_mean_squared_error: 0.1198
    Epoch 127/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0106 - mean_squared_error: 0.0106 - val_loss: 0.1206 - val_mean_squared_error: 0.1206
    Epoch 128/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0105 - mean_squared_error: 0.0105 - val_loss: 0.1198 - val_mean_squared_error: 0.1198
    Epoch 129/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0105 - mean_squared_error: 0.0105 - val_loss: 0.1201 - val_mean_squared_error: 0.1201
    Epoch 130/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0102 - mean_squared_error: 0.0102 - val_loss: 0.1200 - val_mean_squared_error: 0.1200
    Epoch 131/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0100 - mean_squared_error: 0.0100 - val_loss: 0.1198 - val_mean_squared_error: 0.1198
    Epoch 132/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.1203 - val_mean_squared_error: 0.1203
    Epoch 133/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.1201 - val_mean_squared_error: 0.1201
    Epoch 134/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0098 - mean_squared_error: 0.0098 - val_loss: 0.1203 - val_mean_squared_error: 0.1203
    Epoch 135/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.1204 - val_mean_squared_error: 0.1204
    Epoch 136/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0095 - mean_squared_error: 0.0095 - val_loss: 0.1211 - val_mean_squared_error: 0.1211
    Epoch 137/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0096 - mean_squared_error: 0.0096 - val_loss: 0.1204 - val_mean_squared_error: 0.1204
    Epoch 138/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.1200 - val_mean_squared_error: 0.1200
    Epoch 139/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0093 - mean_squared_error: 0.0093 - val_loss: 0.1222 - val_mean_squared_error: 0.1222
    Epoch 140/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0093 - mean_squared_error: 0.0093 - val_loss: 0.1205 - val_mean_squared_error: 0.1205
    Epoch 141/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0095 - mean_squared_error: 0.0095 - val_loss: 0.1216 - val_mean_squared_error: 0.1216
    Epoch 142/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0090 - mean_squared_error: 0.0090 - val_loss: 0.1218 - val_mean_squared_error: 0.1218
    Epoch 143/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.1205 - val_mean_squared_error: 0.1205
    Epoch 144/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.1211 - val_mean_squared_error: 0.1211
    Epoch 145/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.1208 - val_mean_squared_error: 0.1208
    Epoch 146/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.1212 - val_mean_squared_error: 0.1212
    Epoch 147/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.1213 - val_mean_squared_error: 0.1213
    Epoch 148/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.1219 - val_mean_squared_error: 0.1219
    Epoch 149/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.1214 - val_mean_squared_error: 0.1214
    Epoch 150/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.1221 - val_mean_squared_error: 0.1221





    <keras.callbacks.History at 0x1a6d0146d8>



Evaluate the model (`he_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
he_model.evaluate(X_train, y_train_scaled)
```

    1051/1051 [==============================] - 0s 26us/step





    [0.008211220295605321, 0.008211220295605321]



Evaluate the model (`he_model`) on validate data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data
he_model.evaluate(X_val, y_val_scaled)
```

    263/263 [==============================] - 0s 32us/step





    [0.12206264504518345, 0.12206264504518345]



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
lecun_model.add(layers.Dense(100, kernel_initializer='lecun_normal', activation='relu', input_shape=(n_features,)))

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

    Train on 1051 samples, validate on 263 samples
    Epoch 1/150
    1051/1051 [==============================] - 1s 541us/step - loss: 0.4770 - mean_squared_error: 0.4770 - val_loss: 0.1462 - val_mean_squared_error: 0.1462
    Epoch 2/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.2096 - mean_squared_error: 0.2096 - val_loss: 0.1231 - val_mean_squared_error: 0.1231
    Epoch 3/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.1698 - mean_squared_error: 0.1698 - val_loss: 0.1168 - val_mean_squared_error: 0.1168
    Epoch 4/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.1479 - mean_squared_error: 0.1479 - val_loss: 0.1087 - val_mean_squared_error: 0.1087
    Epoch 5/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.1331 - mean_squared_error: 0.1331 - val_loss: 0.1060 - val_mean_squared_error: 0.1060
    Epoch 6/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.1197 - mean_squared_error: 0.1197 - val_loss: 0.1050 - val_mean_squared_error: 0.1050
    Epoch 7/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.1125 - mean_squared_error: 0.1125 - val_loss: 0.1091 - val_mean_squared_error: 0.1091
    Epoch 8/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.1042 - mean_squared_error: 0.1042 - val_loss: 0.1058 - val_mean_squared_error: 0.1058
    Epoch 9/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.1009 - mean_squared_error: 0.1009 - val_loss: 0.1070 - val_mean_squared_error: 0.1070
    Epoch 10/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0976 - mean_squared_error: 0.0976 - val_loss: 0.1067 - val_mean_squared_error: 0.1067
    Epoch 11/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0901 - mean_squared_error: 0.0901 - val_loss: 0.1021 - val_mean_squared_error: 0.1021
    Epoch 12/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0822 - mean_squared_error: 0.0822 - val_loss: 0.1050 - val_mean_squared_error: 0.1050
    Epoch 13/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0811 - mean_squared_error: 0.0811 - val_loss: 0.1075 - val_mean_squared_error: 0.1075
    Epoch 14/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0758 - mean_squared_error: 0.0758 - val_loss: 0.1062 - val_mean_squared_error: 0.1062
    Epoch 15/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0738 - mean_squared_error: 0.0738 - val_loss: 0.1021 - val_mean_squared_error: 0.1021
    Epoch 16/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0698 - mean_squared_error: 0.0698 - val_loss: 0.1055 - val_mean_squared_error: 0.1055
    Epoch 17/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0688 - mean_squared_error: 0.0688 - val_loss: 0.1026 - val_mean_squared_error: 0.1026
    Epoch 18/150
    1051/1051 [==============================] - 0s 69us/step - loss: 0.0661 - mean_squared_error: 0.0661 - val_loss: 0.1042 - val_mean_squared_error: 0.1042
    Epoch 19/150
    1051/1051 [==============================] - 0s 69us/step - loss: 0.0619 - mean_squared_error: 0.0619 - val_loss: 0.1025 - val_mean_squared_error: 0.1025
    Epoch 20/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0587 - mean_squared_error: 0.0587 - val_loss: 0.1038 - val_mean_squared_error: 0.1038
    Epoch 21/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0567 - mean_squared_error: 0.0567 - val_loss: 0.1054 - val_mean_squared_error: 0.1054
    Epoch 22/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0547 - mean_squared_error: 0.0547 - val_loss: 0.1090 - val_mean_squared_error: 0.1090
    Epoch 23/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0541 - mean_squared_error: 0.0541 - val_loss: 0.1069 - val_mean_squared_error: 0.1069
    Epoch 24/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0522 - mean_squared_error: 0.0522 - val_loss: 0.1094 - val_mean_squared_error: 0.1094
    Epoch 25/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0508 - mean_squared_error: 0.0508 - val_loss: 0.1052 - val_mean_squared_error: 0.1052
    Epoch 26/150
    1051/1051 [==============================] - 0s 67us/step - loss: 0.0480 - mean_squared_error: 0.0480 - val_loss: 0.1070 - val_mean_squared_error: 0.1070
    Epoch 27/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0472 - mean_squared_error: 0.0472 - val_loss: 0.1111 - val_mean_squared_error: 0.1111
    Epoch 28/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0464 - mean_squared_error: 0.0464 - val_loss: 0.1085 - val_mean_squared_error: 0.1085
    Epoch 29/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0449 - mean_squared_error: 0.0449 - val_loss: 0.1088 - val_mean_squared_error: 0.1088
    Epoch 30/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0433 - mean_squared_error: 0.0433 - val_loss: 0.1107 - val_mean_squared_error: 0.1107
    Epoch 31/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0418 - mean_squared_error: 0.0418 - val_loss: 0.1096 - val_mean_squared_error: 0.1096
    Epoch 32/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0411 - mean_squared_error: 0.0411 - val_loss: 0.1104 - val_mean_squared_error: 0.1104
    Epoch 33/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0407 - mean_squared_error: 0.0407 - val_loss: 0.1111 - val_mean_squared_error: 0.1111
    Epoch 34/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0388 - mean_squared_error: 0.0388 - val_loss: 0.1138 - val_mean_squared_error: 0.1138
    Epoch 35/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0374 - mean_squared_error: 0.0374 - val_loss: 0.1149 - val_mean_squared_error: 0.1149
    Epoch 36/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0375 - mean_squared_error: 0.0375 - val_loss: 0.1133 - val_mean_squared_error: 0.1133
    Epoch 37/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0358 - mean_squared_error: 0.0358 - val_loss: 0.1139 - val_mean_squared_error: 0.1139
    Epoch 38/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0353 - mean_squared_error: 0.0353 - val_loss: 0.1115 - val_mean_squared_error: 0.1115
    Epoch 39/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0352 - mean_squared_error: 0.0352 - val_loss: 0.1135 - val_mean_squared_error: 0.1135
    Epoch 40/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0334 - mean_squared_error: 0.0334 - val_loss: 0.1126 - val_mean_squared_error: 0.1126
    Epoch 41/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0328 - mean_squared_error: 0.0328 - val_loss: 0.1147 - val_mean_squared_error: 0.1147
    Epoch 42/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0313 - mean_squared_error: 0.0313 - val_loss: 0.1160 - val_mean_squared_error: 0.1160
    Epoch 43/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0312 - mean_squared_error: 0.0312 - val_loss: 0.1172 - val_mean_squared_error: 0.1172
    Epoch 44/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0307 - mean_squared_error: 0.0307 - val_loss: 0.1177 - val_mean_squared_error: 0.1177
    Epoch 45/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0298 - mean_squared_error: 0.0298 - val_loss: 0.1164 - val_mean_squared_error: 0.1164
    Epoch 46/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0296 - mean_squared_error: 0.0296 - val_loss: 0.1178 - val_mean_squared_error: 0.1178
    Epoch 47/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0284 - mean_squared_error: 0.0284 - val_loss: 0.1207 - val_mean_squared_error: 0.1207
    Epoch 48/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0283 - mean_squared_error: 0.0283 - val_loss: 0.1190 - val_mean_squared_error: 0.1190
    Epoch 49/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0277 - mean_squared_error: 0.0277 - val_loss: 0.1199 - val_mean_squared_error: 0.1199
    Epoch 50/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0268 - mean_squared_error: 0.0268 - val_loss: 0.1166 - val_mean_squared_error: 0.1166
    Epoch 51/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0262 - mean_squared_error: 0.0262 - val_loss: 0.1181 - val_mean_squared_error: 0.1181
    Epoch 52/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0259 - mean_squared_error: 0.0259 - val_loss: 0.1196 - val_mean_squared_error: 0.1196
    Epoch 53/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0253 - mean_squared_error: 0.0253 - val_loss: 0.1198 - val_mean_squared_error: 0.1198
    Epoch 54/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0248 - mean_squared_error: 0.0248 - val_loss: 0.1196 - val_mean_squared_error: 0.1196
    Epoch 55/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0245 - mean_squared_error: 0.0245 - val_loss: 0.1193 - val_mean_squared_error: 0.1193
    Epoch 56/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0243 - mean_squared_error: 0.0243 - val_loss: 0.1192 - val_mean_squared_error: 0.1192
    Epoch 57/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0232 - mean_squared_error: 0.0232 - val_loss: 0.1198 - val_mean_squared_error: 0.1198
    Epoch 58/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0228 - mean_squared_error: 0.0228 - val_loss: 0.1212 - val_mean_squared_error: 0.1212
    Epoch 59/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0223 - mean_squared_error: 0.0223 - val_loss: 0.1240 - val_mean_squared_error: 0.1240
    Epoch 60/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0219 - mean_squared_error: 0.0219 - val_loss: 0.1205 - val_mean_squared_error: 0.1205
    Epoch 61/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0217 - mean_squared_error: 0.0217 - val_loss: 0.1215 - val_mean_squared_error: 0.1215
    Epoch 62/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0216 - mean_squared_error: 0.0216 - val_loss: 0.1258 - val_mean_squared_error: 0.1258
    Epoch 63/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0215 - mean_squared_error: 0.0215 - val_loss: 0.1214 - val_mean_squared_error: 0.1214
    Epoch 64/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0212 - mean_squared_error: 0.0212 - val_loss: 0.1243 - val_mean_squared_error: 0.1243
    Epoch 65/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0207 - mean_squared_error: 0.0207 - val_loss: 0.1224 - val_mean_squared_error: 0.1224
    Epoch 66/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0198 - mean_squared_error: 0.0198 - val_loss: 0.1263 - val_mean_squared_error: 0.1263
    Epoch 67/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0201 - mean_squared_error: 0.0201 - val_loss: 0.1225 - val_mean_squared_error: 0.1225
    Epoch 68/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0195 - mean_squared_error: 0.0195 - val_loss: 0.1251 - val_mean_squared_error: 0.1251
    Epoch 69/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0195 - mean_squared_error: 0.0195 - val_loss: 0.1228 - val_mean_squared_error: 0.1228
    Epoch 70/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0188 - mean_squared_error: 0.0188 - val_loss: 0.1231 - val_mean_squared_error: 0.1231
    Epoch 71/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0187 - mean_squared_error: 0.0187 - val_loss: 0.1224 - val_mean_squared_error: 0.1224
    Epoch 72/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0187 - mean_squared_error: 0.0187 - val_loss: 0.1230 - val_mean_squared_error: 0.1230
    Epoch 73/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0183 - mean_squared_error: 0.0183 - val_loss: 0.1234 - val_mean_squared_error: 0.1234
    Epoch 74/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0176 - mean_squared_error: 0.0176 - val_loss: 0.1282 - val_mean_squared_error: 0.1282
    Epoch 75/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0177 - mean_squared_error: 0.0177 - val_loss: 0.1246 - val_mean_squared_error: 0.1246
    Epoch 76/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0173 - mean_squared_error: 0.0173 - val_loss: 0.1298 - val_mean_squared_error: 0.1298
    Epoch 77/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0175 - mean_squared_error: 0.0175 - val_loss: 0.1234 - val_mean_squared_error: 0.1234
    Epoch 78/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0170 - mean_squared_error: 0.0170 - val_loss: 0.1245 - val_mean_squared_error: 0.1245
    Epoch 79/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0162 - mean_squared_error: 0.0162 - val_loss: 0.1242 - val_mean_squared_error: 0.1242
    Epoch 80/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0162 - mean_squared_error: 0.0162 - val_loss: 0.1250 - val_mean_squared_error: 0.1250
    Epoch 81/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0161 - mean_squared_error: 0.0161 - val_loss: 0.1252 - val_mean_squared_error: 0.1252
    Epoch 82/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0158 - mean_squared_error: 0.0158 - val_loss: 0.1292 - val_mean_squared_error: 0.1292
    Epoch 83/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0156 - mean_squared_error: 0.0156 - val_loss: 0.1257 - val_mean_squared_error: 0.1257
    Epoch 84/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0156 - mean_squared_error: 0.0156 - val_loss: 0.1252 - val_mean_squared_error: 0.1252
    Epoch 85/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0149 - mean_squared_error: 0.0149 - val_loss: 0.1262 - val_mean_squared_error: 0.1262
    Epoch 86/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0151 - mean_squared_error: 0.0151 - val_loss: 0.1260 - val_mean_squared_error: 0.1260
    Epoch 87/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0147 - mean_squared_error: 0.0147 - val_loss: 0.1267 - val_mean_squared_error: 0.1267
    Epoch 88/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0151 - mean_squared_error: 0.0151 - val_loss: 0.1252 - val_mean_squared_error: 0.1252
    Epoch 89/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0146 - mean_squared_error: 0.0146 - val_loss: 0.1260 - val_mean_squared_error: 0.1260
    Epoch 90/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0144 - mean_squared_error: 0.0144 - val_loss: 0.1258 - val_mean_squared_error: 0.1258
    Epoch 91/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0141 - mean_squared_error: 0.0141 - val_loss: 0.1277 - val_mean_squared_error: 0.1277
    Epoch 92/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0137 - mean_squared_error: 0.0137 - val_loss: 0.1267 - val_mean_squared_error: 0.1267
    Epoch 93/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0139 - mean_squared_error: 0.0139 - val_loss: 0.1273 - val_mean_squared_error: 0.1273
    Epoch 94/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0138 - mean_squared_error: 0.0138 - val_loss: 0.1293 - val_mean_squared_error: 0.1293
    Epoch 95/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.1279 - val_mean_squared_error: 0.1279
    Epoch 96/150
    1051/1051 [==============================] - 0s 50us/step - loss: 0.0132 - mean_squared_error: 0.0132 - val_loss: 0.1272 - val_mean_squared_error: 0.1272
    Epoch 97/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0129 - mean_squared_error: 0.0129 - val_loss: 0.1272 - val_mean_squared_error: 0.1272
    Epoch 98/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0129 - mean_squared_error: 0.0129 - val_loss: 0.1268 - val_mean_squared_error: 0.1268
    Epoch 99/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0127 - mean_squared_error: 0.0127 - val_loss: 0.1294 - val_mean_squared_error: 0.1294
    Epoch 100/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0125 - mean_squared_error: 0.0125 - val_loss: 0.1288 - val_mean_squared_error: 0.1288
    Epoch 101/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0125 - mean_squared_error: 0.0125 - val_loss: 0.1294 - val_mean_squared_error: 0.1294
    Epoch 102/150
    1051/1051 [==============================] - 0s 54us/step - loss: 0.0124 - mean_squared_error: 0.0124 - val_loss: 0.1276 - val_mean_squared_error: 0.1276
    Epoch 103/150
    1051/1051 [==============================] - 0s 52us/step - loss: 0.0121 - mean_squared_error: 0.0121 - val_loss: 0.1283 - val_mean_squared_error: 0.1283
    Epoch 104/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0120 - mean_squared_error: 0.0120 - val_loss: 0.1302 - val_mean_squared_error: 0.1302
    Epoch 105/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0118 - mean_squared_error: 0.0118 - val_loss: 0.1286 - val_mean_squared_error: 0.1286
    Epoch 106/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0116 - mean_squared_error: 0.0116 - val_loss: 0.1302 - val_mean_squared_error: 0.1302
    Epoch 107/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0114 - mean_squared_error: 0.0114 - val_loss: 0.1290 - val_mean_squared_error: 0.1290
    Epoch 108/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0114 - mean_squared_error: 0.0114 - val_loss: 0.1302 - val_mean_squared_error: 0.1302
    Epoch 109/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0111 - mean_squared_error: 0.0111 - val_loss: 0.1289 - val_mean_squared_error: 0.1289
    Epoch 110/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0114 - mean_squared_error: 0.0114 - val_loss: 0.1296 - val_mean_squared_error: 0.1296
    Epoch 111/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0110 - mean_squared_error: 0.0110 - val_loss: 0.1310 - val_mean_squared_error: 0.1310
    Epoch 112/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0109 - mean_squared_error: 0.0109 - val_loss: 0.1298 - val_mean_squared_error: 0.1298
    Epoch 113/150
    1051/1051 [==============================] - 0s 67us/step - loss: 0.0107 - mean_squared_error: 0.0107 - val_loss: 0.1304 - val_mean_squared_error: 0.1304
    Epoch 114/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0109 - mean_squared_error: 0.0109 - val_loss: 0.1305 - val_mean_squared_error: 0.1305
    Epoch 115/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0104 - mean_squared_error: 0.0104 - val_loss: 0.1297 - val_mean_squared_error: 0.1297
    Epoch 116/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.1301 - val_mean_squared_error: 0.1301
    Epoch 117/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.1308 - val_mean_squared_error: 0.1308
    Epoch 118/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0104 - mean_squared_error: 0.0104 - val_loss: 0.1306 - val_mean_squared_error: 0.1306
    Epoch 119/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.1318 - val_mean_squared_error: 0.1318
    Epoch 120/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0098 - mean_squared_error: 0.0098 - val_loss: 0.1321 - val_mean_squared_error: 0.1321
    Epoch 121/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.1313 - val_mean_squared_error: 0.1313
    Epoch 122/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0098 - mean_squared_error: 0.0098 - val_loss: 0.1316 - val_mean_squared_error: 0.1316
    Epoch 123/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0096 - mean_squared_error: 0.0096 - val_loss: 0.1316 - val_mean_squared_error: 0.1316
    Epoch 124/150
    1051/1051 [==============================] - 0s 57us/step - loss: 0.0096 - mean_squared_error: 0.0096 - val_loss: 0.1298 - val_mean_squared_error: 0.1298
    Epoch 125/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0095 - mean_squared_error: 0.0095 - val_loss: 0.1365 - val_mean_squared_error: 0.1365
    Epoch 126/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0093 - mean_squared_error: 0.0093 - val_loss: 0.1321 - val_mean_squared_error: 0.1321
    Epoch 127/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0092 - mean_squared_error: 0.0092 - val_loss: 0.1331 - val_mean_squared_error: 0.1331
    Epoch 128/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.1317 - val_mean_squared_error: 0.1317
    Epoch 129/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.1326 - val_mean_squared_error: 0.1326
    Epoch 130/150
    1051/1051 [==============================] - 0s 62us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.1327 - val_mean_squared_error: 0.1327
    Epoch 131/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.1320 - val_mean_squared_error: 0.1320
    Epoch 132/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.1325 - val_mean_squared_error: 0.1325
    Epoch 133/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.1325 - val_mean_squared_error: 0.1325
    Epoch 134/150
    1051/1051 [==============================] - 0s 53us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.1331 - val_mean_squared_error: 0.1331
    Epoch 135/150
    1051/1051 [==============================] - 0s 51us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.1325 - val_mean_squared_error: 0.1325
    Epoch 136/150
    1051/1051 [==============================] - 0s 56us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.1334 - val_mean_squared_error: 0.1334
    Epoch 137/150
    1051/1051 [==============================] - 0s 59us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.1324 - val_mean_squared_error: 0.1324
    Epoch 138/150
    1051/1051 [==============================] - 0s 58us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.1325 - val_mean_squared_error: 0.1325
    Epoch 139/150
    1051/1051 [==============================] - 0s 55us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.1364 - val_mean_squared_error: 0.1364
    Epoch 140/150
    1051/1051 [==============================] - 0s 68us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.1331 - val_mean_squared_error: 0.1331
    Epoch 141/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.1343 - val_mean_squared_error: 0.1343
    Epoch 142/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.1350 - val_mean_squared_error: 0.1350
    Epoch 143/150
    1051/1051 [==============================] - 0s 66us/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.1324 - val_mean_squared_error: 0.1324
    Epoch 144/150
    1051/1051 [==============================] - 0s 63us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.1342 - val_mean_squared_error: 0.1342
    Epoch 145/150
    1051/1051 [==============================] - 0s 61us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.1334 - val_mean_squared_error: 0.1334
    Epoch 146/150
    1051/1051 [==============================] - 0s 60us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.1338 - val_mean_squared_error: 0.1338
    Epoch 147/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.1343 - val_mean_squared_error: 0.1343
    Epoch 148/150
    1051/1051 [==============================] - 0s 64us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.1355 - val_mean_squared_error: 0.1355
    Epoch 149/150
    1051/1051 [==============================] - 0s 65us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.1336 - val_mean_squared_error: 0.1336
    Epoch 150/150
    1051/1051 [==============================] - 0s 68us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.1348 - val_mean_squared_error: 0.1348





    <keras.callbacks.History at 0x1a6d269da0>



Evaluate the model (`lecun_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
lecun_model.evaluate(X_train, y_train_scaled)
```

    1051/1051 [==============================] - 0s 28us/step





    [0.007119152007303728, 0.007119152007303728]



Evaluate the model (`lecun_model`) on validate data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data
lecun_model.evaluate(X_val, y_val_scaled)
```

    263/263 [==============================] - 0s 30us/step





    [0.13479157605920228, 0.13479157605920228]



Not much of a difference, but a useful note to consider when tuning your network. Next, let's investigate the impact of various optimization algorithms.

## RMSprop 

Compile the `rmsprop_model` with: 

- `'rmsprop'` as the optimizer 
- track `'mse'` as the loss and metric  


```python
np.random.seed(123)
rmsprop_model = Sequential()
rmsprop_model.add(layers.Dense(100, activation='relu', input_shape=(n_features,)))
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

    Train on 1051 samples, validate on 263 samples
    Epoch 1/150
    1051/1051 [==============================] - 1s 651us/step - loss: 0.3205 - mean_squared_error: 0.3205 - val_loss: 0.1370 - val_mean_squared_error: 0.1370
    Epoch 2/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.1666 - mean_squared_error: 0.1666 - val_loss: 0.1105 - val_mean_squared_error: 0.1105
    Epoch 3/150
    1051/1051 [==============================] - 0s 94us/step - loss: 0.1247 - mean_squared_error: 0.1247 - val_loss: 0.1262 - val_mean_squared_error: 0.1262
    Epoch 4/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0988 - mean_squared_error: 0.0988 - val_loss: 0.1098 - val_mean_squared_error: 0.1098
    Epoch 5/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0896 - mean_squared_error: 0.0896 - val_loss: 0.1023 - val_mean_squared_error: 0.1023
    Epoch 6/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0661 - mean_squared_error: 0.0661 - val_loss: 0.1076 - val_mean_squared_error: 0.1076
    Epoch 7/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0570 - mean_squared_error: 0.0570 - val_loss: 0.1209 - val_mean_squared_error: 0.1209
    Epoch 8/150
    1051/1051 [==============================] - 0s 80us/step - loss: 0.0545 - mean_squared_error: 0.0545 - val_loss: 0.1067 - val_mean_squared_error: 0.1067
    Epoch 9/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0440 - mean_squared_error: 0.0440 - val_loss: 0.0983 - val_mean_squared_error: 0.0983
    Epoch 10/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0399 - mean_squared_error: 0.0399 - val_loss: 0.1081 - val_mean_squared_error: 0.1081
    Epoch 11/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0329 - mean_squared_error: 0.0329 - val_loss: 0.0922 - val_mean_squared_error: 0.0922
    Epoch 12/150
    1051/1051 [==============================] - 0s 80us/step - loss: 0.0300 - mean_squared_error: 0.0300 - val_loss: 0.1063 - val_mean_squared_error: 0.1063
    Epoch 13/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0232 - mean_squared_error: 0.0232 - val_loss: 0.1100 - val_mean_squared_error: 0.1100
    Epoch 14/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0276 - mean_squared_error: 0.0276 - val_loss: 0.1050 - val_mean_squared_error: 0.1050
    Epoch 15/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0261 - mean_squared_error: 0.0261 - val_loss: 0.1041 - val_mean_squared_error: 0.1041
    Epoch 16/150
    1051/1051 [==============================] - 0s 96us/step - loss: 0.0216 - mean_squared_error: 0.0216 - val_loss: 0.1339 - val_mean_squared_error: 0.1339
    Epoch 17/150
    1051/1051 [==============================] - 0s 93us/step - loss: 0.0206 - mean_squared_error: 0.0206 - val_loss: 0.1175 - val_mean_squared_error: 0.1175
    Epoch 18/150
    1051/1051 [==============================] - 0s 92us/step - loss: 0.0201 - mean_squared_error: 0.0201 - val_loss: 0.1160 - val_mean_squared_error: 0.1160
    Epoch 19/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0202 - mean_squared_error: 0.0202 - val_loss: 0.0928 - val_mean_squared_error: 0.0928
    Epoch 20/150
    1051/1051 [==============================] - 0s 91us/step - loss: 0.0166 - mean_squared_error: 0.0166 - val_loss: 0.1163 - val_mean_squared_error: 0.1163
    Epoch 21/150
    1051/1051 [==============================] - 0s 91us/step - loss: 0.0165 - mean_squared_error: 0.0165 - val_loss: 0.1045 - val_mean_squared_error: 0.1045
    Epoch 22/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0150 - mean_squared_error: 0.0150 - val_loss: 0.0943 - val_mean_squared_error: 0.0943
    Epoch 23/150
    1051/1051 [==============================] - 0s 80us/step - loss: 0.0128 - mean_squared_error: 0.0128 - val_loss: 0.0959 - val_mean_squared_error: 0.0959
    Epoch 24/150
    1051/1051 [==============================] - 0s 92us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.1039 - val_mean_squared_error: 0.1039
    Epoch 25/150
    1051/1051 [==============================] - 0s 93us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.0951 - val_mean_squared_error: 0.0951
    Epoch 26/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.0913 - val_mean_squared_error: 0.0913
    Epoch 27/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0135 - mean_squared_error: 0.0135 - val_loss: 0.1372 - val_mean_squared_error: 0.1372
    Epoch 28/150
    1051/1051 [==============================] - 0s 96us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.1050 - val_mean_squared_error: 0.1050
    Epoch 29/150
    1051/1051 [==============================] - 0s 97us/step - loss: 0.0131 - mean_squared_error: 0.0131 - val_loss: 0.0957 - val_mean_squared_error: 0.0957
    Epoch 30/150
    1051/1051 [==============================] - 0s 94us/step - loss: 0.0104 - mean_squared_error: 0.0104 - val_loss: 0.0886 - val_mean_squared_error: 0.0886
    Epoch 31/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0119 - mean_squared_error: 0.0119 - val_loss: 0.1009 - val_mean_squared_error: 0.1009
    Epoch 32/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0114 - mean_squared_error: 0.0114 - val_loss: 0.0875 - val_mean_squared_error: 0.0875
    Epoch 33/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0106 - mean_squared_error: 0.0106 - val_loss: 0.0959 - val_mean_squared_error: 0.0959
    Epoch 34/150
    1051/1051 [==============================] - 0s 98us/step - loss: 0.0128 - mean_squared_error: 0.0128 - val_loss: 0.0973 - val_mean_squared_error: 0.0973
    Epoch 35/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0107 - mean_squared_error: 0.0107 - val_loss: 0.0951 - val_mean_squared_error: 0.0951
    Epoch 36/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0931 - val_mean_squared_error: 0.0931
    Epoch 37/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.0917 - val_mean_squared_error: 0.0917
    Epoch 38/150
    1051/1051 [==============================] - 0s 93us/step - loss: 0.0090 - mean_squared_error: 0.0090 - val_loss: 0.1029 - val_mean_squared_error: 0.1029
    Epoch 39/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0111 - mean_squared_error: 0.0111 - val_loss: 0.0976 - val_mean_squared_error: 0.0976
    Epoch 40/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0106 - mean_squared_error: 0.0106 - val_loss: 0.0903 - val_mean_squared_error: 0.0903
    Epoch 41/150
    1051/1051 [==============================] - 0s 82us/step - loss: 0.0090 - mean_squared_error: 0.0090 - val_loss: 0.1037 - val_mean_squared_error: 0.1037
    Epoch 42/150
    1051/1051 [==============================] - 0s 92us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.0861 - val_mean_squared_error: 0.0861
    Epoch 43/150
    1051/1051 [==============================] - 0s 92us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.1069 - val_mean_squared_error: 0.1069
    Epoch 44/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0102 - mean_squared_error: 0.0102 - val_loss: 0.0841 - val_mean_squared_error: 0.0841
    Epoch 45/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0947 - val_mean_squared_error: 0.0947
    Epoch 46/150
    1051/1051 [==============================] - 0s 92us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.0844 - val_mean_squared_error: 0.0844
    Epoch 47/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0957 - val_mean_squared_error: 0.0957
    Epoch 48/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0102 - mean_squared_error: 0.0102 - val_loss: 0.0873 - val_mean_squared_error: 0.0873
    Epoch 49/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0933 - val_mean_squared_error: 0.0933
    Epoch 50/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.0989 - val_mean_squared_error: 0.0989
    Epoch 51/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0917 - val_mean_squared_error: 0.0917
    Epoch 52/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.0851 - val_mean_squared_error: 0.0851
    Epoch 53/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0867 - val_mean_squared_error: 0.0867
    Epoch 54/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0874 - val_mean_squared_error: 0.0874
    Epoch 55/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0953 - val_mean_squared_error: 0.0953
    Epoch 56/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0878 - val_mean_squared_error: 0.0878
    Epoch 57/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0918 - val_mean_squared_error: 0.0918
    Epoch 58/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.0958 - val_mean_squared_error: 0.0958
    Epoch 59/150
    1051/1051 [==============================] - 0s 93us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0868 - val_mean_squared_error: 0.0868
    Epoch 60/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.0859 - val_mean_squared_error: 0.0859
    Epoch 61/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0949 - val_mean_squared_error: 0.0949
    Epoch 62/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0898 - val_mean_squared_error: 0.0898
    Epoch 63/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.1062 - val_mean_squared_error: 0.1062
    Epoch 64/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0898 - val_mean_squared_error: 0.0898
    Epoch 65/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0906 - val_mean_squared_error: 0.0906
    Epoch 66/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0853 - val_mean_squared_error: 0.0853
    Epoch 67/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0831 - val_mean_squared_error: 0.0831
    Epoch 68/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.0870 - val_mean_squared_error: 0.0870
    Epoch 69/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0936 - val_mean_squared_error: 0.0936
    Epoch 70/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0061 - mean_squared_error: 0.0061 - val_loss: 0.1008 - val_mean_squared_error: 0.1008
    Epoch 71/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0838 - val_mean_squared_error: 0.0838
    Epoch 72/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.0905 - val_mean_squared_error: 0.0905
    Epoch 73/150
    1051/1051 [==============================] - 0s 82us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0872 - val_mean_squared_error: 0.0872
    Epoch 74/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0848 - val_mean_squared_error: 0.0848
    Epoch 75/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0053 - mean_squared_error: 0.0053 - val_loss: 0.0915 - val_mean_squared_error: 0.0915
    Epoch 76/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0967 - val_mean_squared_error: 0.0967
    Epoch 77/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0869 - val_mean_squared_error: 0.0869
    Epoch 78/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0829 - val_mean_squared_error: 0.0829
    Epoch 79/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0046 - mean_squared_error: 0.0046 - val_loss: 0.0793 - val_mean_squared_error: 0.0793
    Epoch 80/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0055 - mean_squared_error: 0.0055 - val_loss: 0.0912 - val_mean_squared_error: 0.0912
    Epoch 81/150
    1051/1051 [==============================] - 0s 91us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0850 - val_mean_squared_error: 0.0850
    Epoch 82/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0053 - mean_squared_error: 0.0053 - val_loss: 0.0890 - val_mean_squared_error: 0.0890
    Epoch 83/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0057 - mean_squared_error: 0.0057 - val_loss: 0.0887 - val_mean_squared_error: 0.0887
    Epoch 84/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0884 - val_mean_squared_error: 0.0884
    Epoch 85/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0849 - val_mean_squared_error: 0.0849
    Epoch 86/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0050 - mean_squared_error: 0.0050 - val_loss: 0.0879 - val_mean_squared_error: 0.0879
    Epoch 87/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0874 - val_mean_squared_error: 0.0874
    Epoch 88/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0050 - mean_squared_error: 0.0050 - val_loss: 0.0837 - val_mean_squared_error: 0.0837
    Epoch 89/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0060 - mean_squared_error: 0.0060 - val_loss: 0.0904 - val_mean_squared_error: 0.0904
    Epoch 90/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0055 - mean_squared_error: 0.0055 - val_loss: 0.0872 - val_mean_squared_error: 0.0872
    Epoch 91/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0051 - mean_squared_error: 0.0051 - val_loss: 0.0865 - val_mean_squared_error: 0.0865
    Epoch 92/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0048 - mean_squared_error: 0.0048 - val_loss: 0.0870 - val_mean_squared_error: 0.0870
    Epoch 93/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0052 - mean_squared_error: 0.0052 - val_loss: 0.0942 - val_mean_squared_error: 0.0942
    Epoch 94/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0050 - mean_squared_error: 0.0050 - val_loss: 0.0901 - val_mean_squared_error: 0.0901
    Epoch 95/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0054 - mean_squared_error: 0.0054 - val_loss: 0.0936 - val_mean_squared_error: 0.0936
    Epoch 96/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0050 - mean_squared_error: 0.0050 - val_loss: 0.0911 - val_mean_squared_error: 0.0911
    Epoch 97/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0051 - mean_squared_error: 0.0051 - val_loss: 0.0864 - val_mean_squared_error: 0.0864
    Epoch 98/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0048 - mean_squared_error: 0.0048 - val_loss: 0.0892 - val_mean_squared_error: 0.0892
    Epoch 99/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0046 - mean_squared_error: 0.0046 - val_loss: 0.1082 - val_mean_squared_error: 0.1082
    Epoch 100/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0049 - mean_squared_error: 0.0049 - val_loss: 0.0865 - val_mean_squared_error: 0.0865
    Epoch 101/150
    1051/1051 [==============================] - 0s 102us/step - loss: 0.0052 - mean_squared_error: 0.0052 - val_loss: 0.0940 - val_mean_squared_error: 0.0940
    Epoch 102/150
    1051/1051 [==============================] - 0s 95us/step - loss: 0.0046 - mean_squared_error: 0.0046 - val_loss: 0.0907 - val_mean_squared_error: 0.0907
    Epoch 103/150
    1051/1051 [==============================] - 0s 92us/step - loss: 0.0049 - mean_squared_error: 0.0049 - val_loss: 0.0977 - val_mean_squared_error: 0.0977
    Epoch 104/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0040 - mean_squared_error: 0.0040 - val_loss: 0.0811 - val_mean_squared_error: 0.0811
    Epoch 105/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0057 - mean_squared_error: 0.0057 - val_loss: 0.0865 - val_mean_squared_error: 0.0865
    Epoch 106/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0045 - mean_squared_error: 0.0045 - val_loss: 0.0861 - val_mean_squared_error: 0.0861
    Epoch 107/150
    1051/1051 [==============================] - 0s 91us/step - loss: 0.0059 - mean_squared_error: 0.0059 - val_loss: 0.0940 - val_mean_squared_error: 0.0940
    Epoch 108/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0040 - mean_squared_error: 0.0040 - val_loss: 0.0807 - val_mean_squared_error: 0.0807
    Epoch 109/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0048 - mean_squared_error: 0.0048 - val_loss: 0.0852 - val_mean_squared_error: 0.0852
    Epoch 110/150
    1051/1051 [==============================] - 0s 94us/step - loss: 0.0046 - mean_squared_error: 0.0046 - val_loss: 0.0864 - val_mean_squared_error: 0.0864
    Epoch 111/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0046 - mean_squared_error: 0.0046 - val_loss: 0.0858 - val_mean_squared_error: 0.0858
    Epoch 112/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0056 - mean_squared_error: 0.0056 - val_loss: 0.0828 - val_mean_squared_error: 0.0828
    Epoch 113/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0052 - mean_squared_error: 0.0052 - val_loss: 0.0879 - val_mean_squared_error: 0.0879
    Epoch 114/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0038 - mean_squared_error: 0.0038 - val_loss: 0.0881 - val_mean_squared_error: 0.0881
    Epoch 115/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0047 - mean_squared_error: 0.0047 - val_loss: 0.0861 - val_mean_squared_error: 0.0861
    Epoch 116/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0051 - mean_squared_error: 0.0051 - val_loss: 0.0853 - val_mean_squared_error: 0.0853
    Epoch 117/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0044 - mean_squared_error: 0.0044 - val_loss: 0.0849 - val_mean_squared_error: 0.0849
    Epoch 118/150
    1051/1051 [==============================] - 0s 94us/step - loss: 0.0037 - mean_squared_error: 0.0037 - val_loss: 0.0816 - val_mean_squared_error: 0.0816
    Epoch 119/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0049 - mean_squared_error: 0.0049 - val_loss: 0.0869 - val_mean_squared_error: 0.0869
    Epoch 120/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0037 - mean_squared_error: 0.0037 - val_loss: 0.0827 - val_mean_squared_error: 0.0827
    Epoch 121/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0043 - mean_squared_error: 0.0043 - val_loss: 0.0789 - val_mean_squared_error: 0.0789
    Epoch 122/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0047 - mean_squared_error: 0.0047 - val_loss: 0.0845 - val_mean_squared_error: 0.0845
    Epoch 123/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0046 - mean_squared_error: 0.0046 - val_loss: 0.0852 - val_mean_squared_error: 0.0852
    Epoch 124/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0042 - mean_squared_error: 0.0042 - val_loss: 0.0790 - val_mean_squared_error: 0.0790
    Epoch 125/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0037 - mean_squared_error: 0.0037 - val_loss: 0.0915 - val_mean_squared_error: 0.0915
    Epoch 126/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0045 - mean_squared_error: 0.0045 - val_loss: 0.0825 - val_mean_squared_error: 0.0825
    Epoch 127/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0037 - mean_squared_error: 0.0037 - val_loss: 0.0823 - val_mean_squared_error: 0.0823
    Epoch 128/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0041 - mean_squared_error: 0.0041 - val_loss: 0.0794 - val_mean_squared_error: 0.0794
    Epoch 129/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0040 - mean_squared_error: 0.0040 - val_loss: 0.0836 - val_mean_squared_error: 0.0836
    Epoch 130/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0041 - mean_squared_error: 0.0041 - val_loss: 0.0830 - val_mean_squared_error: 0.0830
    Epoch 131/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0041 - mean_squared_error: 0.0041 - val_loss: 0.0939 - val_mean_squared_error: 0.0939
    Epoch 132/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0042 - mean_squared_error: 0.0042 - val_loss: 0.0800 - val_mean_squared_error: 0.0800
    Epoch 133/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0038 - mean_squared_error: 0.0038 - val_loss: 0.0934 - val_mean_squared_error: 0.0934
    Epoch 134/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0874 - val_mean_squared_error: 0.0874
    Epoch 135/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0045 - mean_squared_error: 0.0045 - val_loss: 0.0863 - val_mean_squared_error: 0.0863
    Epoch 136/150
    1051/1051 [==============================] - 0s 81us/step - loss: 0.0039 - mean_squared_error: 0.0039 - val_loss: 0.0875 - val_mean_squared_error: 0.0875
    Epoch 137/150
    1051/1051 [==============================] - 0s 82us/step - loss: 0.0036 - mean_squared_error: 0.0036 - val_loss: 0.0846 - val_mean_squared_error: 0.0846
    Epoch 138/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0038 - mean_squared_error: 0.0038 - val_loss: 0.0859 - val_mean_squared_error: 0.0859
    Epoch 139/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0037 - mean_squared_error: 0.0037 - val_loss: 0.0826 - val_mean_squared_error: 0.0826
    Epoch 140/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0894 - val_mean_squared_error: 0.0894
    Epoch 141/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0045 - mean_squared_error: 0.0045 - val_loss: 0.0898 - val_mean_squared_error: 0.0898
    Epoch 142/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0038 - mean_squared_error: 0.0038 - val_loss: 0.0886 - val_mean_squared_error: 0.0886
    Epoch 143/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0041 - mean_squared_error: 0.0041 - val_loss: 0.0803 - val_mean_squared_error: 0.0803
    Epoch 144/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0032 - mean_squared_error: 0.0032 - val_loss: 0.0937 - val_mean_squared_error: 0.0937
    Epoch 145/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0033 - mean_squared_error: 0.0033 - val_loss: 0.0867 - val_mean_squared_error: 0.0867
    Epoch 146/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0036 - mean_squared_error: 0.0036 - val_loss: 0.0927 - val_mean_squared_error: 0.0927
    Epoch 147/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0048 - mean_squared_error: 0.0048 - val_loss: 0.0880 - val_mean_squared_error: 0.0880
    Epoch 148/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0040 - mean_squared_error: 0.0040 - val_loss: 0.0859 - val_mean_squared_error: 0.0859
    Epoch 149/150
    1051/1051 [==============================] - 0s 91us/step - loss: 0.0038 - mean_squared_error: 0.0038 - val_loss: 0.0915 - val_mean_squared_error: 0.0915
    Epoch 150/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0885 - val_mean_squared_error: 0.0885





    <keras.callbacks.History at 0x1a6d4aee80>



Evaluate the model (`rmsprop_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
rmsprop_model.evaluate(X_train, y_train_scaled)
```

    1051/1051 [==============================] - 0s 23us/step





    [0.0019496748209279651, 0.0019496748209279651]



Evaluate the model (`rmsprop_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data
rmsprop_model.evaluate(X_val, y_val_scaled)
```

    263/263 [==============================] - 0s 31us/step





    [0.08852711957330486, 0.08852711957330486]



## Adam 

Compile the `adam_model` with: 

- `'Adam'` as the optimizer 
- track `'mse'` as the loss and metric  


```python
np.random.seed(123)
adam_model = Sequential()
adam_model.add(layers.Dense(100, activation='relu', input_shape=(n_features,)))
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

    Train on 1051 samples, validate on 263 samples
    Epoch 1/150
    1051/1051 [==============================] - 1s 814us/step - loss: 0.3834 - mean_squared_error: 0.3834 - val_loss: 0.1672 - val_mean_squared_error: 0.1672
    Epoch 2/150
    1051/1051 [==============================] - 0s 95us/step - loss: 0.1617 - mean_squared_error: 0.1617 - val_loss: 0.1316 - val_mean_squared_error: 0.1316
    Epoch 3/150
    1051/1051 [==============================] - 0s 98us/step - loss: 0.1123 - mean_squared_error: 0.1123 - val_loss: 0.1033 - val_mean_squared_error: 0.1033
    Epoch 4/150
    1051/1051 [==============================] - 0s 93us/step - loss: 0.0863 - mean_squared_error: 0.0863 - val_loss: 0.1019 - val_mean_squared_error: 0.1019
    Epoch 5/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0751 - mean_squared_error: 0.0751 - val_loss: 0.1002 - val_mean_squared_error: 0.1002
    Epoch 6/150
    1051/1051 [==============================] - 0s 93us/step - loss: 0.0564 - mean_squared_error: 0.0564 - val_loss: 0.0953 - val_mean_squared_error: 0.0953
    Epoch 7/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0468 - mean_squared_error: 0.0468 - val_loss: 0.1175 - val_mean_squared_error: 0.1175
    Epoch 8/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0411 - mean_squared_error: 0.0411 - val_loss: 0.1080 - val_mean_squared_error: 0.1080
    Epoch 9/150
    1051/1051 [==============================] - 0s 95us/step - loss: 0.0312 - mean_squared_error: 0.0312 - val_loss: 0.1094 - val_mean_squared_error: 0.1094
    Epoch 10/150
    1051/1051 [==============================] - 0s 92us/step - loss: 0.0245 - mean_squared_error: 0.0245 - val_loss: 0.0999 - val_mean_squared_error: 0.0999
    Epoch 11/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0209 - mean_squared_error: 0.0209 - val_loss: 0.1019 - val_mean_squared_error: 0.1019
    Epoch 12/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0168 - mean_squared_error: 0.0168 - val_loss: 0.1129 - val_mean_squared_error: 0.1129
    Epoch 13/150
    1051/1051 [==============================] - 0s 98us/step - loss: 0.0145 - mean_squared_error: 0.0145 - val_loss: 0.1010 - val_mean_squared_error: 0.1010
    Epoch 14/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0116 - mean_squared_error: 0.0116 - val_loss: 0.1038 - val_mean_squared_error: 0.1038
    Epoch 15/150
    1051/1051 [==============================] - 0s 96us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.1034 - val_mean_squared_error: 0.1034
    Epoch 16/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.1088 - val_mean_squared_error: 0.1088
    Epoch 17/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.1019 - val_mean_squared_error: 0.1019
    Epoch 18/150
    1051/1051 [==============================] - 0s 91us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.1123 - val_mean_squared_error: 0.1123
    Epoch 19/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0995 - val_mean_squared_error: 0.0995
    Epoch 20/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.1098 - val_mean_squared_error: 0.1098
    Epoch 21/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.1035 - val_mean_squared_error: 0.1035
    Epoch 22/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.1042 - val_mean_squared_error: 0.1042
    Epoch 23/150
    1051/1051 [==============================] - 0s 91us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.1079 - val_mean_squared_error: 0.1079
    Epoch 24/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0050 - mean_squared_error: 0.0050 - val_loss: 0.1071 - val_mean_squared_error: 0.1071
    Epoch 25/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0045 - mean_squared_error: 0.0045 - val_loss: 0.1086 - val_mean_squared_error: 0.1086
    Epoch 26/150
    1051/1051 [==============================] - 0s 95us/step - loss: 0.0040 - mean_squared_error: 0.0040 - val_loss: 0.1060 - val_mean_squared_error: 0.1060
    Epoch 27/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0039 - mean_squared_error: 0.0039 - val_loss: 0.1111 - val_mean_squared_error: 0.1111
    Epoch 28/150
    1051/1051 [==============================] - 0s 82us/step - loss: 0.0036 - mean_squared_error: 0.0036 - val_loss: 0.1058 - val_mean_squared_error: 0.1058
    Epoch 29/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0042 - mean_squared_error: 0.0042 - val_loss: 0.1181 - val_mean_squared_error: 0.1181
    Epoch 30/150
    1051/1051 [==============================] - 0s 95us/step - loss: 0.0056 - mean_squared_error: 0.0056 - val_loss: 0.1048 - val_mean_squared_error: 0.1048
    Epoch 31/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.1037 - val_mean_squared_error: 0.1037
    Epoch 32/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0056 - mean_squared_error: 0.0056 - val_loss: 0.1035 - val_mean_squared_error: 0.1035
    Epoch 33/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0049 - mean_squared_error: 0.0049 - val_loss: 0.0988 - val_mean_squared_error: 0.0988
    Epoch 34/150
    1051/1051 [==============================] - 0s 83us/step - loss: 0.0054 - mean_squared_error: 0.0054 - val_loss: 0.1095 - val_mean_squared_error: 0.1095
    Epoch 35/150
    1051/1051 [==============================] - 0s 92us/step - loss: 0.0043 - mean_squared_error: 0.0043 - val_loss: 0.0983 - val_mean_squared_error: 0.0983
    Epoch 36/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0050 - mean_squared_error: 0.0050 - val_loss: 0.1098 - val_mean_squared_error: 0.1098
    Epoch 37/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0031 - mean_squared_error: 0.0031 - val_loss: 0.1013 - val_mean_squared_error: 0.1013
    Epoch 38/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0022 - mean_squared_error: 0.0022 - val_loss: 0.1046 - val_mean_squared_error: 0.1046
    Epoch 39/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0017 - mean_squared_error: 0.0017 - val_loss: 0.1065 - val_mean_squared_error: 0.1065
    Epoch 40/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0014 - mean_squared_error: 0.0014 - val_loss: 0.1019 - val_mean_squared_error: 0.1019
    Epoch 41/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0015 - mean_squared_error: 0.0015 - val_loss: 0.1083 - val_mean_squared_error: 0.1083
    Epoch 42/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0020 - mean_squared_error: 0.0020 - val_loss: 0.1019 - val_mean_squared_error: 0.1019
    Epoch 43/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0027 - mean_squared_error: 0.0027 - val_loss: 0.1115 - val_mean_squared_error: 0.1115
    Epoch 44/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0043 - mean_squared_error: 0.0043 - val_loss: 0.0989 - val_mean_squared_error: 0.0989
    Epoch 45/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.1247 - val_mean_squared_error: 0.1247
    Epoch 46/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.0999 - val_mean_squared_error: 0.0999
    Epoch 47/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0047 - mean_squared_error: 0.0047 - val_loss: 0.1004 - val_mean_squared_error: 0.1004
    Epoch 48/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0027 - mean_squared_error: 0.0027 - val_loss: 0.1036 - val_mean_squared_error: 0.1036
    Epoch 49/150
    1051/1051 [==============================] - 0s 97us/step - loss: 0.0019 - mean_squared_error: 0.0019 - val_loss: 0.1043 - val_mean_squared_error: 0.1043
    Epoch 50/150
    1051/1051 [==============================] - 0s 97us/step - loss: 0.0023 - mean_squared_error: 0.0023 - val_loss: 0.1031 - val_mean_squared_error: 0.1031
    Epoch 51/150
    1051/1051 [==============================] - 0s 99us/step - loss: 0.0032 - mean_squared_error: 0.0032 - val_loss: 0.1039 - val_mean_squared_error: 0.1039
    Epoch 52/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.1034 - val_mean_squared_error: 0.1034
    Epoch 53/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0039 - mean_squared_error: 0.0039 - val_loss: 0.1101 - val_mean_squared_error: 0.1101
    Epoch 54/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0033 - mean_squared_error: 0.0033 - val_loss: 0.1021 - val_mean_squared_error: 0.1021
    Epoch 55/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0024 - mean_squared_error: 0.0024 - val_loss: 0.1031 - val_mean_squared_error: 0.1031
    Epoch 56/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0025 - mean_squared_error: 0.0025 - val_loss: 0.1071 - val_mean_squared_error: 0.1071
    Epoch 57/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0025 - mean_squared_error: 0.0025 - val_loss: 0.0991 - val_mean_squared_error: 0.0991
    Epoch 58/150
    1051/1051 [==============================] - 0s 96us/step - loss: 0.0023 - mean_squared_error: 0.0023 - val_loss: 0.1022 - val_mean_squared_error: 0.1022
    Epoch 59/150
    1051/1051 [==============================] - 0s 101us/step - loss: 0.0022 - mean_squared_error: 0.0022 - val_loss: 0.1012 - val_mean_squared_error: 0.1012
    Epoch 60/150
    1051/1051 [==============================] - 0s 91us/step - loss: 0.0024 - mean_squared_error: 0.0024 - val_loss: 0.1073 - val_mean_squared_error: 0.1073
    Epoch 61/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0024 - mean_squared_error: 0.0024 - val_loss: 0.0985 - val_mean_squared_error: 0.0985
    Epoch 62/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0026 - mean_squared_error: 0.0026 - val_loss: 0.1074 - val_mean_squared_error: 0.1074
    Epoch 63/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0032 - mean_squared_error: 0.0032 - val_loss: 0.0990 - val_mean_squared_error: 0.0990
    Epoch 64/150
    1051/1051 [==============================] - 0s 96us/step - loss: 0.0030 - mean_squared_error: 0.0030 - val_loss: 0.1070 - val_mean_squared_error: 0.1070
    Epoch 65/150
    1051/1051 [==============================] - 0s 94us/step - loss: 0.0055 - mean_squared_error: 0.0055 - val_loss: 0.1083 - val_mean_squared_error: 0.1083
    Epoch 66/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0057 - mean_squared_error: 0.0057 - val_loss: 0.0994 - val_mean_squared_error: 0.0994
    Epoch 67/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0032 - mean_squared_error: 0.0032 - val_loss: 0.0959 - val_mean_squared_error: 0.0959
    Epoch 68/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0020 - mean_squared_error: 0.0020 - val_loss: 0.1004 - val_mean_squared_error: 0.1004
    Epoch 69/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0018 - mean_squared_error: 0.0018 - val_loss: 0.0982 - val_mean_squared_error: 0.0982
    Epoch 70/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0013 - mean_squared_error: 0.0013 - val_loss: 0.0967 - val_mean_squared_error: 0.0967
    Epoch 71/150
    1051/1051 [==============================] - 0s 93us/step - loss: 0.0010 - mean_squared_error: 0.0010 - val_loss: 0.1003 - val_mean_squared_error: 0.1003
    Epoch 72/150
    1051/1051 [==============================] - 0s 90us/step - loss: 8.5322e-04 - mean_squared_error: 8.5322e-04 - val_loss: 0.0956 - val_mean_squared_error: 0.0956
    Epoch 73/150
    1051/1051 [==============================] - 0s 88us/step - loss: 8.2070e-04 - mean_squared_error: 8.2070e-04 - val_loss: 0.1005 - val_mean_squared_error: 0.1005
    Epoch 74/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0012 - mean_squared_error: 0.0012 - val_loss: 0.0964 - val_mean_squared_error: 0.0964
    Epoch 75/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0017 - mean_squared_error: 0.0017 - val_loss: 0.0980 - val_mean_squared_error: 0.0980
    Epoch 76/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0030 - mean_squared_error: 0.0030 - val_loss: 0.0983 - val_mean_squared_error: 0.0983
    Epoch 77/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0031 - mean_squared_error: 0.0031 - val_loss: 0.0991 - val_mean_squared_error: 0.0991
    Epoch 78/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0042 - mean_squared_error: 0.0042 - val_loss: 0.0956 - val_mean_squared_error: 0.0956
    Epoch 79/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0035 - mean_squared_error: 0.0035 - val_loss: 0.1039 - val_mean_squared_error: 0.1039
    Epoch 80/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0032 - mean_squared_error: 0.0032 - val_loss: 0.0990 - val_mean_squared_error: 0.0990
    Epoch 81/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0032 - mean_squared_error: 0.0032 - val_loss: 0.0957 - val_mean_squared_error: 0.0957
    Epoch 82/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0030 - mean_squared_error: 0.0030 - val_loss: 0.0962 - val_mean_squared_error: 0.0962
    Epoch 83/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0031 - mean_squared_error: 0.0031 - val_loss: 0.1002 - val_mean_squared_error: 0.1002
    Epoch 84/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0026 - mean_squared_error: 0.0026 - val_loss: 0.0910 - val_mean_squared_error: 0.0910
    Epoch 85/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0024 - mean_squared_error: 0.0024 - val_loss: 0.1013 - val_mean_squared_error: 0.1013
    Epoch 86/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0026 - mean_squared_error: 0.0026 - val_loss: 0.0902 - val_mean_squared_error: 0.0902
    Epoch 87/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0030 - mean_squared_error: 0.0030 - val_loss: 0.0922 - val_mean_squared_error: 0.0922
    Epoch 88/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0031 - mean_squared_error: 0.0031 - val_loss: 0.1003 - val_mean_squared_error: 0.1003
    Epoch 89/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0046 - mean_squared_error: 0.0046 - val_loss: 0.0904 - val_mean_squared_error: 0.0904
    Epoch 90/150
    1051/1051 [==============================] - 0s 90us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.1009 - val_mean_squared_error: 0.1009
    Epoch 91/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0883 - val_mean_squared_error: 0.0883
    Epoch 92/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0113 - mean_squared_error: 0.0113 - val_loss: 0.1176 - val_mean_squared_error: 0.1176
    Epoch 93/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0891 - val_mean_squared_error: 0.0891
    Epoch 94/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0994 - val_mean_squared_error: 0.0994
    Epoch 95/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0055 - mean_squared_error: 0.0055 - val_loss: 0.0957 - val_mean_squared_error: 0.0957
    Epoch 96/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0050 - mean_squared_error: 0.0050 - val_loss: 0.0922 - val_mean_squared_error: 0.0922
    Epoch 97/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0031 - mean_squared_error: 0.0031 - val_loss: 0.0954 - val_mean_squared_error: 0.0954
    Epoch 98/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0019 - mean_squared_error: 0.0019 - val_loss: 0.0940 - val_mean_squared_error: 0.0940
    Epoch 99/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0014 - mean_squared_error: 0.0014 - val_loss: 0.0923 - val_mean_squared_error: 0.0923
    Epoch 100/150
    1051/1051 [==============================] - 0s 87us/step - loss: 9.5184e-04 - mean_squared_error: 9.5184e-04 - val_loss: 0.0908 - val_mean_squared_error: 0.0908
    Epoch 101/150
    1051/1051 [==============================] - 0s 87us/step - loss: 6.4050e-04 - mean_squared_error: 6.4050e-04 - val_loss: 0.0931 - val_mean_squared_error: 0.0931
    Epoch 102/150
    1051/1051 [==============================] - 0s 88us/step - loss: 5.4407e-04 - mean_squared_error: 5.4407e-04 - val_loss: 0.0916 - val_mean_squared_error: 0.0916
    Epoch 103/150
    1051/1051 [==============================] - 0s 86us/step - loss: 4.9791e-04 - mean_squared_error: 4.9791e-04 - val_loss: 0.0925 - val_mean_squared_error: 0.0925
    Epoch 104/150
    1051/1051 [==============================] - 0s 87us/step - loss: 6.5472e-04 - mean_squared_error: 6.5472e-04 - val_loss: 0.0896 - val_mean_squared_error: 0.0896
    Epoch 105/150
    1051/1051 [==============================] - 0s 88us/step - loss: 8.0129e-04 - mean_squared_error: 8.0129e-04 - val_loss: 0.0931 - val_mean_squared_error: 0.0931
    Epoch 106/150
    1051/1051 [==============================] - 0s 87us/step - loss: 8.2676e-04 - mean_squared_error: 8.2676e-04 - val_loss: 0.0901 - val_mean_squared_error: 0.0901
    Epoch 107/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0014 - mean_squared_error: 0.0014 - val_loss: 0.0955 - val_mean_squared_error: 0.0955
    Epoch 108/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0018 - mean_squared_error: 0.0018 - val_loss: 0.0913 - val_mean_squared_error: 0.0913
    Epoch 109/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0031 - mean_squared_error: 0.0031 - val_loss: 0.0969 - val_mean_squared_error: 0.0969
    Epoch 110/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0040 - mean_squared_error: 0.0040 - val_loss: 0.0870 - val_mean_squared_error: 0.0870
    Epoch 111/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0056 - mean_squared_error: 0.0056 - val_loss: 0.0965 - val_mean_squared_error: 0.0965
    Epoch 112/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0882 - val_mean_squared_error: 0.0882
    Epoch 113/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0922 - val_mean_squared_error: 0.0922
    Epoch 114/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0054 - mean_squared_error: 0.0054 - val_loss: 0.0872 - val_mean_squared_error: 0.0872
    Epoch 115/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0949 - val_mean_squared_error: 0.0949
    Epoch 116/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0053 - mean_squared_error: 0.0053 - val_loss: 0.0938 - val_mean_squared_error: 0.0938
    Epoch 117/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0909 - val_mean_squared_error: 0.0909
    Epoch 118/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0026 - mean_squared_error: 0.0026 - val_loss: 0.0871 - val_mean_squared_error: 0.0871
    Epoch 119/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0020 - mean_squared_error: 0.0020 - val_loss: 0.0941 - val_mean_squared_error: 0.0941
    Epoch 120/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0020 - mean_squared_error: 0.0020 - val_loss: 0.0907 - val_mean_squared_error: 0.0907
    Epoch 121/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0016 - mean_squared_error: 0.0016 - val_loss: 0.0904 - val_mean_squared_error: 0.0904
    Epoch 122/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0011 - mean_squared_error: 0.0011 - val_loss: 0.0878 - val_mean_squared_error: 0.0878
    Epoch 123/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0013 - mean_squared_error: 0.0013 - val_loss: 0.0903 - val_mean_squared_error: 0.0903
    Epoch 124/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0013 - mean_squared_error: 0.0013 - val_loss: 0.0902 - val_mean_squared_error: 0.0902
    Epoch 125/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0013 - mean_squared_error: 0.0013 - val_loss: 0.0892 - val_mean_squared_error: 0.0892
    Epoch 126/150
    1051/1051 [==============================] - 0s 85us/step - loss: 9.9258e-04 - mean_squared_error: 9.9258e-04 - val_loss: 0.0886 - val_mean_squared_error: 0.0886
    Epoch 127/150
    1051/1051 [==============================] - 0s 85us/step - loss: 7.7288e-04 - mean_squared_error: 7.7288e-04 - val_loss: 0.0913 - val_mean_squared_error: 0.0913
    Epoch 128/150
    1051/1051 [==============================] - 0s 85us/step - loss: 6.1068e-04 - mean_squared_error: 6.1068e-04 - val_loss: 0.0882 - val_mean_squared_error: 0.0882
    Epoch 129/150
    1051/1051 [==============================] - 0s 87us/step - loss: 6.0950e-04 - mean_squared_error: 6.0950e-04 - val_loss: 0.0904 - val_mean_squared_error: 0.0904
    Epoch 130/150
    1051/1051 [==============================] - 0s 86us/step - loss: 6.4823e-04 - mean_squared_error: 6.4823e-04 - val_loss: 0.0887 - val_mean_squared_error: 0.0887
    Epoch 131/150
    1051/1051 [==============================] - 0s 87us/step - loss: 6.6270e-04 - mean_squared_error: 6.6270e-04 - val_loss: 0.0888 - val_mean_squared_error: 0.0888
    Epoch 132/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0014 - mean_squared_error: 0.0014 - val_loss: 0.0889 - val_mean_squared_error: 0.0889
    Epoch 133/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0018 - mean_squared_error: 0.0018 - val_loss: 0.0936 - val_mean_squared_error: 0.0936
    Epoch 134/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0021 - mean_squared_error: 0.0021 - val_loss: 0.0848 - val_mean_squared_error: 0.0848
    Epoch 135/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0028 - mean_squared_error: 0.0028 - val_loss: 0.0939 - val_mean_squared_error: 0.0939
    Epoch 136/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0025 - mean_squared_error: 0.0025 - val_loss: 0.0832 - val_mean_squared_error: 0.0832
    Epoch 137/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0952 - val_mean_squared_error: 0.0952
    Epoch 138/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0039 - mean_squared_error: 0.0039 - val_loss: 0.0837 - val_mean_squared_error: 0.0837
    Epoch 139/150
    1051/1051 [==============================] - 0s 85us/step - loss: 0.0047 - mean_squared_error: 0.0047 - val_loss: 0.0947 - val_mean_squared_error: 0.0947
    Epoch 140/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0047 - mean_squared_error: 0.0047 - val_loss: 0.0845 - val_mean_squared_error: 0.0845
    Epoch 141/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0925 - val_mean_squared_error: 0.0925
    Epoch 142/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0051 - mean_squared_error: 0.0051 - val_loss: 0.0816 - val_mean_squared_error: 0.0816
    Epoch 143/150
    1051/1051 [==============================] - 0s 84us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.1041 - val_mean_squared_error: 0.1041
    Epoch 144/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.0807 - val_mean_squared_error: 0.0807
    Epoch 145/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0925 - val_mean_squared_error: 0.0925
    Epoch 146/150
    1051/1051 [==============================] - 0s 86us/step - loss: 0.0138 - mean_squared_error: 0.0138 - val_loss: 0.0878 - val_mean_squared_error: 0.0878
    Epoch 147/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0095 - mean_squared_error: 0.0095 - val_loss: 0.0948 - val_mean_squared_error: 0.0948
    Epoch 148/150
    1051/1051 [==============================] - 0s 89us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0897 - val_mean_squared_error: 0.0897
    Epoch 149/150
    1051/1051 [==============================] - 0s 87us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.0902 - val_mean_squared_error: 0.0902
    Epoch 150/150
    1051/1051 [==============================] - 0s 88us/step - loss: 0.0051 - mean_squared_error: 0.0051 - val_loss: 0.0902 - val_mean_squared_error: 0.0902





    <keras.callbacks.History at 0x1a6d7e3be0>



Evaluate the model (`adam_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data
adam_model.evaluate(X_train, y_train_scaled)
```

    1051/1051 [==============================] - 0s 25us/step





    [0.004596830631467159, 0.004596830631467159]



Evaluate the model (`adam_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data
adam_model.evaluate(X_val, y_val_scaled)
```

    263/263 [==============================] - 0s 31us/step





    [0.09024790601500314, 0.09024790601500314]



## Select a Final Model

Now, select the model with the best performance based on the training and validation sets. Evaluate this top model using the test set!


```python
# Evaluate the best model on test data
rmsprop_model.evaluate(X_test, y_test_scaled)
```

    146/146 [==============================] - 0s 38us/step





    [0.16469571688403822, 0.16469571688403822]



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




    31891.332169282312



## Summary  

In this lab, you worked to ensure your model converged properly by normalizing both the input and output. Additionally, you also investigated the impact of varying initialization and optimization routines.
