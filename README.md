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


```python
# __SOLUTION__
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
# __SOLUTION__
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
X_train = None
X_val = None
X_test = None
```


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
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

Here, we're calling .shape on our training data so that we can use the result as n_features, so we know how big to make our input layer.


```python
# How big input layer?
n_features = (X_train.shape[1],)
print(n_features)
```


```python
# __SOLUTION__ 
# How big input layer?
n_features = (X_train.shape[1],)
print(n_features)
```

    (296,)


Create your baseline model. Yo will notice is exihibits strange behavior.

*Note:* When you run this model or other models later on, you may get a comment from tf letting you about optimizing your GPU


```python
# Baseline model
np.random.seed(123)
baseline_model = Sequential()

# Hidden layer with 100 units


# Hidden layer with 50 units


# Output layer


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


```python
# __SOLUTION__

#Baseline model
np.random.seed(123)
baseline_model = Sequential()

# Hidden layer with 100 units
baseline_model.add(layers.Dense(100, activation='relu', input_shape=(n_features)))

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


    2023-06-21 13:07:50.124979: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2023-06-21 13:07:50.125259: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
    2023-06-21 13:07:50.261231: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz


    13/33 [==========>...................] - ETA: 0s - loss: nan - mse: nan                          

    2023-06-21 13:07:50.355492: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 9ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 2/150
    26/33 [======================>.......] - ETA: 0s - loss: nan - mse: nan

    2023-06-21 13:07:50.687109: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 4/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 5/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 6/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 7/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 8/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 9/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 10/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 11/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 12/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 13/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 14/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 15/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 16/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 17/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 18/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 19/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 20/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 21/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 22/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 23/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 24/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 25/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 26/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 27/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 28/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 29/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 30/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 31/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 32/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 33/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 34/150
    33/33 [==============================] - 0s 6ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 36/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 38/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 39/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 56/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 58/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 59/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 64/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 65/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 87/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 134/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 138/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 139/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan





    <keras.callbacks.History at 0x16989e7f0>



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
numeric_columns = None 

# Instantiate StandardScaler
ss_X = None

# Fit and transform train data
X_train_scaled = None

# Transform validate and test data
X_val_scaled = None
X_test_scaled = None

# Combine the scaled numerical features and categorical features
X_train = pd.concat([X_train_scaled, X_train_cat], axis=1)
X_val = pd.concat([X_val_scaled, X_val_cat], axis=1)
X_test = pd.concat([X_test_scaled, X_test_cat], axis=1)
```


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
# Model with all normalized inputs
np.random.seed(123)
normalized_input_model = Sequential()
normalized_input_model.add(layers.Dense(100, activation='relu', input_shape=(n_features)))
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
Again, you may get some strange behavior.


```python
# Train the model

```


```python
# __SOLUTION__ 
# Train the model
normalized_input_model.fit(X_train,  
                           y_train, 
                           batch_size=32, 
                           epochs=150, 
                           validation_data=(X_val, y_val))# Train the model
```

    Epoch 1/150
    25/33 [=====================>........] - ETA: 0s - loss: nan - mse: nan                         

    2023-06-21 13:09:40.148838: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 8ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 2/150
    26/33 [======================>.......] - ETA: 0s - loss: nan - mse: nan

    2023-06-21 13:09:40.419687: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 4/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 5/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 6/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 7/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 8/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 9/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 10/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 11/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 12/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 13/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 14/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 15/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 16/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 17/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 18/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 19/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 20/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 21/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 22/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 23/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 24/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 25/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 26/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 27/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 28/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 29/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 30/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 31/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 32/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 33/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 34/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 36/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 38/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 39/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 56/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 58/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 59/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 64/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 65/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 87/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 134/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 138/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 139/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
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
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: nan - mse: nan - val_loss: nan - val_mse: nan





    <keras.callbacks.History at 0x17afa7a60>



> _**Note that you still haven't achieved convergence! From here, it's time to normalize the output data.**_

## Normalizing the output

Again, use `StandardScaler()` to: 

- Fit and transform `y_train` 
- Transform `y_val` and `y_test` 


```python
# Instantiate StandardScaler
ss_y = None

# Fit and transform train labels
y_train_scaled = None

# Transform validate and test labels
y_val_scaled = None
y_test_scaled = None
```


```python
# __SOLUTION__ 
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
normalized_model.add(layers.Dense(100, activation='relu', input_shape=(n_features)))
normalized_model.add(layers.Dense(50, activation='relu'))
normalized_model.add(layers.Dense(1, activation='linear'))

# Compile the model
normalized_model.compile(optimizer='SGD', 
                         loss='mse', 
                         metrics=['mse']) 

# Train the model

```


```python
# __SOLUTION__ 
# Model with all normalized inputs and outputs
np.random.seed(123)
normalized_model = Sequential()
normalized_model.add(layers.Dense(100, activation='relu', input_shape=(n_features)))
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

    Epoch 1/150
    26/33 [======================>.......] - ETA: 0s - loss: 0.5130 - mse: 0.5130

    2023-06-21 13:10:43.490866: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 8ms/step - loss: 0.4988 - mse: 0.4988 - val_loss: 0.1796 - val_mse: 0.1796
    Epoch 2/150
    27/33 [=======================>......] - ETA: 0s - loss: 0.2884 - mse: 0.2884

    2023-06-21 13:10:43.755700: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 5ms/step - loss: 0.2591 - mse: 0.2591 - val_loss: 0.1515 - val_mse: 0.1515
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1930 - mse: 0.1930 - val_loss: 0.1373 - val_mse: 0.1373
    Epoch 4/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1704 - mse: 0.1704 - val_loss: 0.1302 - val_mse: 0.1302
    Epoch 5/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1449 - mse: 0.1449 - val_loss: 0.1313 - val_mse: 0.1313
    Epoch 6/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1269 - mse: 0.1269 - val_loss: 0.1257 - val_mse: 0.1257
    Epoch 7/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1170 - mse: 0.1170 - val_loss: 0.1230 - val_mse: 0.1230
    Epoch 8/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1069 - mse: 0.1069 - val_loss: 0.1249 - val_mse: 0.1249
    Epoch 9/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1002 - mse: 0.1002 - val_loss: 0.1206 - val_mse: 0.1206
    Epoch 10/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0969 - mse: 0.0969 - val_loss: 0.1242 - val_mse: 0.1242
    Epoch 11/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0884 - mse: 0.0884 - val_loss: 0.1168 - val_mse: 0.1168
    Epoch 12/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0828 - mse: 0.0828 - val_loss: 0.1169 - val_mse: 0.1169
    Epoch 13/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0772 - mse: 0.0772 - val_loss: 0.1149 - val_mse: 0.1149
    Epoch 14/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0750 - mse: 0.0750 - val_loss: 0.1137 - val_mse: 0.1137
    Epoch 15/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0696 - mse: 0.0696 - val_loss: 0.1180 - val_mse: 0.1180
    Epoch 16/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0653 - mse: 0.0653 - val_loss: 0.1216 - val_mse: 0.1216
    Epoch 17/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0635 - mse: 0.0635 - val_loss: 0.1138 - val_mse: 0.1138
    Epoch 18/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0614 - mse: 0.0614 - val_loss: 0.1153 - val_mse: 0.1153
    Epoch 19/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0583 - mse: 0.0583 - val_loss: 0.1164 - val_mse: 0.1164
    Epoch 20/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0557 - mse: 0.0557 - val_loss: 0.1155 - val_mse: 0.1155
    Epoch 21/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0550 - mse: 0.0550 - val_loss: 0.1166 - val_mse: 0.1166
    Epoch 22/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0517 - mse: 0.0517 - val_loss: 0.1124 - val_mse: 0.1124
    Epoch 23/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0510 - mse: 0.0510 - val_loss: 0.1096 - val_mse: 0.1096
    Epoch 24/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0507 - mse: 0.0507 - val_loss: 0.1111 - val_mse: 0.1111
    Epoch 25/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0469 - mse: 0.0469 - val_loss: 0.1114 - val_mse: 0.1114
    Epoch 26/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0455 - mse: 0.0455 - val_loss: 0.1119 - val_mse: 0.1119
    Epoch 27/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0437 - mse: 0.0437 - val_loss: 0.1106 - val_mse: 0.1106
    Epoch 28/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0424 - mse: 0.0424 - val_loss: 0.1111 - val_mse: 0.1111
    Epoch 29/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0420 - mse: 0.0420 - val_loss: 0.1128 - val_mse: 0.1128
    Epoch 30/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0409 - mse: 0.0409 - val_loss: 0.1123 - val_mse: 0.1123
    Epoch 31/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0405 - mse: 0.0405 - val_loss: 0.1115 - val_mse: 0.1115
    Epoch 32/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0384 - mse: 0.0384 - val_loss: 0.1113 - val_mse: 0.1113
    Epoch 33/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0374 - mse: 0.0374 - val_loss: 0.1103 - val_mse: 0.1103
    Epoch 34/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0361 - mse: 0.0361 - val_loss: 0.1132 - val_mse: 0.1132
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0350 - mse: 0.0350 - val_loss: 0.1121 - val_mse: 0.1121
    Epoch 36/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0341 - mse: 0.0341 - val_loss: 0.1136 - val_mse: 0.1136
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0344 - mse: 0.0344 - val_loss: 0.1096 - val_mse: 0.1096
    Epoch 38/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0329 - mse: 0.0329 - val_loss: 0.1116 - val_mse: 0.1116
    Epoch 39/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0315 - mse: 0.0315 - val_loss: 0.1143 - val_mse: 0.1143
    Epoch 40/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0310 - mse: 0.0310 - val_loss: 0.1109 - val_mse: 0.1109
    Epoch 41/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0308 - mse: 0.0308 - val_loss: 0.1101 - val_mse: 0.1101
    Epoch 42/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0304 - mse: 0.0304 - val_loss: 0.1105 - val_mse: 0.1105
    Epoch 43/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0291 - mse: 0.0291 - val_loss: 0.1150 - val_mse: 0.1150
    Epoch 44/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0285 - mse: 0.0285 - val_loss: 0.1116 - val_mse: 0.1116
    Epoch 45/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0281 - mse: 0.0281 - val_loss: 0.1112 - val_mse: 0.1112
    Epoch 46/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0277 - mse: 0.0277 - val_loss: 0.1109 - val_mse: 0.1109
    Epoch 47/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0267 - mse: 0.0267 - val_loss: 0.1105 - val_mse: 0.1105
    Epoch 48/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0269 - mse: 0.0269 - val_loss: 0.1099 - val_mse: 0.1099
    Epoch 49/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0257 - mse: 0.0257 - val_loss: 0.1095 - val_mse: 0.1095
    Epoch 50/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0254 - mse: 0.0254 - val_loss: 0.1104 - val_mse: 0.1104
    Epoch 51/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0244 - mse: 0.0244 - val_loss: 0.1100 - val_mse: 0.1100
    Epoch 52/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0240 - mse: 0.0240 - val_loss: 0.1110 - val_mse: 0.1110
    Epoch 53/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0239 - mse: 0.0239 - val_loss: 0.1104 - val_mse: 0.1104
    Epoch 54/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0241 - mse: 0.0241 - val_loss: 0.1106 - val_mse: 0.1106
    Epoch 55/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0227 - mse: 0.0227 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 56/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0226 - mse: 0.0226 - val_loss: 0.1097 - val_mse: 0.1097
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0219 - mse: 0.0219 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 58/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0220 - mse: 0.0220 - val_loss: 0.1100 - val_mse: 0.1100
    Epoch 59/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0213 - mse: 0.0213 - val_loss: 0.1095 - val_mse: 0.1095
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0212 - mse: 0.0212 - val_loss: 0.1138 - val_mse: 0.1138
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0206 - mse: 0.0206 - val_loss: 0.1105 - val_mse: 0.1105
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0204 - mse: 0.0204 - val_loss: 0.1107 - val_mse: 0.1107
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0200 - mse: 0.0200 - val_loss: 0.1093 - val_mse: 0.1093
    Epoch 64/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0195 - mse: 0.0195 - val_loss: 0.1106 - val_mse: 0.1106
    Epoch 65/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0190 - mse: 0.0190 - val_loss: 0.1120 - val_mse: 0.1120
    Epoch 66/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0192 - mse: 0.0192 - val_loss: 0.1092 - val_mse: 0.1092
    Epoch 67/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0183 - mse: 0.0183 - val_loss: 0.1104 - val_mse: 0.1104
    Epoch 68/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0183 - mse: 0.0183 - val_loss: 0.1093 - val_mse: 0.1093
    Epoch 69/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0180 - mse: 0.0180 - val_loss: 0.1095 - val_mse: 0.1095
    Epoch 70/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0178 - mse: 0.0178 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 71/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0172 - mse: 0.0172 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 72/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0172 - mse: 0.0172 - val_loss: 0.1097 - val_mse: 0.1097
    Epoch 73/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0169 - mse: 0.0169 - val_loss: 0.1103 - val_mse: 0.1103
    Epoch 74/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0169 - mse: 0.0169 - val_loss: 0.1102 - val_mse: 0.1102
    Epoch 75/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0165 - mse: 0.0165 - val_loss: 0.1107 - val_mse: 0.1107
    Epoch 76/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0164 - mse: 0.0164 - val_loss: 0.1102 - val_mse: 0.1102
    Epoch 77/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0159 - mse: 0.0159 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 78/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0159 - mse: 0.0159 - val_loss: 0.1099 - val_mse: 0.1099
    Epoch 79/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0156 - mse: 0.0156 - val_loss: 0.1099 - val_mse: 0.1099
    Epoch 80/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0152 - mse: 0.0152 - val_loss: 0.1086 - val_mse: 0.1086
    Epoch 81/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0150 - mse: 0.0150 - val_loss: 0.1103 - val_mse: 0.1103
    Epoch 82/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0147 - mse: 0.0147 - val_loss: 0.1102 - val_mse: 0.1102
    Epoch 83/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0147 - mse: 0.0147 - val_loss: 0.1102 - val_mse: 0.1102
    Epoch 84/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0142 - mse: 0.0142 - val_loss: 0.1097 - val_mse: 0.1097
    Epoch 85/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0141 - mse: 0.0141 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0139 - mse: 0.0139 - val_loss: 0.1101 - val_mse: 0.1101
    Epoch 87/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0138 - mse: 0.0138 - val_loss: 0.1090 - val_mse: 0.1090
    Epoch 88/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0138 - mse: 0.0138 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 89/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0132 - mse: 0.0132 - val_loss: 0.1085 - val_mse: 0.1085
    Epoch 90/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0133 - mse: 0.0133 - val_loss: 0.1086 - val_mse: 0.1086
    Epoch 91/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0130 - mse: 0.0130 - val_loss: 0.1098 - val_mse: 0.1098
    Epoch 92/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0130 - mse: 0.0130 - val_loss: 0.1093 - val_mse: 0.1093
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0128 - mse: 0.0128 - val_loss: 0.1112 - val_mse: 0.1112
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0128 - mse: 0.0128 - val_loss: 0.1085 - val_mse: 0.1085
    Epoch 95/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0123 - mse: 0.0123 - val_loss: 0.1091 - val_mse: 0.1091
    Epoch 96/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0122 - mse: 0.0122 - val_loss: 0.1086 - val_mse: 0.1086
    Epoch 97/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0121 - mse: 0.0121 - val_loss: 0.1091 - val_mse: 0.1091
    Epoch 98/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0122 - mse: 0.0122 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 99/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0119 - mse: 0.0119 - val_loss: 0.1106 - val_mse: 0.1106
    Epoch 100/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0116 - mse: 0.0116 - val_loss: 0.1098 - val_mse: 0.1098
    Epoch 101/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 102/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 103/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 104/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 105/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1100 - val_mse: 0.1100
    Epoch 106/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0111 - mse: 0.0111 - val_loss: 0.1087 - val_mse: 0.1087
    Epoch 107/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 0.1085 - val_mse: 0.1085
    Epoch 108/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 109/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0104 - mse: 0.0104 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 110/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0104 - mse: 0.0104 - val_loss: 0.1091 - val_mse: 0.1091
    Epoch 111/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0102 - mse: 0.0102 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 112/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1085 - val_mse: 0.1085
    Epoch 113/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1087 - val_mse: 0.1087
    Epoch 114/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0100 - mse: 0.0100 - val_loss: 0.1082 - val_mse: 0.1082
    Epoch 115/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1085 - val_mse: 0.1085
    Epoch 116/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0096 - mse: 0.0096 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 117/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0095 - mse: 0.0095 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 118/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0096 - mse: 0.0096 - val_loss: 0.1086 - val_mse: 0.1086
    Epoch 119/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0094 - mse: 0.0094 - val_loss: 0.1077 - val_mse: 0.1077
    Epoch 120/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 0.1076 - val_mse: 0.1076
    Epoch 121/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0091 - mse: 0.0091 - val_loss: 0.1086 - val_mse: 0.1086
    Epoch 122/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0090 - mse: 0.0090 - val_loss: 0.1077 - val_mse: 0.1077
    Epoch 123/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 124/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 0.1084 - val_mse: 0.1084
    Epoch 125/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0086 - mse: 0.0086 - val_loss: 0.1078 - val_mse: 0.1078
    Epoch 126/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0086 - mse: 0.0086 - val_loss: 0.1079 - val_mse: 0.1079
    Epoch 127/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0085 - mse: 0.0085 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 128/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0085 - mse: 0.0085 - val_loss: 0.1087 - val_mse: 0.1087
    Epoch 129/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0084 - mse: 0.0084 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 130/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0081 - mse: 0.0081 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 131/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0081 - mse: 0.0081 - val_loss: 0.1090 - val_mse: 0.1090
    Epoch 132/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0081 - mse: 0.0081 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 133/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0079 - mse: 0.0079 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 134/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0078 - mse: 0.0078 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0079 - mse: 0.0079 - val_loss: 0.1092 - val_mse: 0.1092
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0078 - mse: 0.0078 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0075 - mse: 0.0075 - val_loss: 0.1086 - val_mse: 0.1086
    Epoch 138/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 139/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 140/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0075 - mse: 0.0075 - val_loss: 0.1100 - val_mse: 0.1100
    Epoch 141/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.1085 - val_mse: 0.1085
    Epoch 142/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.1078 - val_mse: 0.1078
    Epoch 143/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0071 - mse: 0.0071 - val_loss: 0.1089 - val_mse: 0.1089
    Epoch 144/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0073 - mse: 0.0073 - val_loss: 0.1082 - val_mse: 0.1082
    Epoch 145/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1108 - val_mse: 0.1108
    Epoch 146/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0070 - mse: 0.0070 - val_loss: 0.1095 - val_mse: 0.1095
    Epoch 147/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1084 - val_mse: 0.1084
    Epoch 148/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0068 - mse: 0.0068 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 149/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0068 - mse: 0.0068 - val_loss: 0.1079 - val_mse: 0.1079
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.1093 - val_mse: 0.1093





    <keras.callbacks.History at 0x17d6f4ee0>



Nicely done! After normalizing both the input and output, the model finally converged. 

- Evaluate the model (`normalized_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```


```python
# __SOLUTION__ 
# Evaluate the model on training data
normalized_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 3ms/step - loss: 0.0065 - mse: 0.0065





    [0.0065474119037389755, 0.0065474119037389755]



- Evaluate the model (`normalized_model`) on validate data (`X_val` and `y_val_scaled`) 


```python
# Evaluate the model on validate data

```


```python
# __SOLUTION__ 
# Evaluate the model on validate data
normalized_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 4ms/step - loss: 0.1093 - mse: 0.1093





    [0.10926736146211624, 0.10926736146211624]



Since the output is normalized, the metric above is not interpretable. To remedy this: 

- Generate predictions on validate data (`X_val`) 
- Transform these predictions back to original scale using `ss_y` 
- Now you can calculate the RMSE in the original units with `y_val` and `y_val_pred` 


```python
# Generate predictions on validate data
y_val_pred_scaled = None

# Transform the predictions back to original scale
y_val_pred = None

# RMSE of validate data
```


```python
# __SOLUTION__ 
# Generate predictions on validate data
y_val_pred_scaled = normalized_model.predict(X_val)

# Transform the predictions back to original scale
y_val_pred = ss_y.inverse_transform(y_val_pred_scaled)

# RMSE of validate data
np.sqrt(mean_squared_error(y_val, y_val_pred))
```

    9/9 [==============================] - 0s 3ms/step


    2023-06-21 13:11:18.687683: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.





    25976.266357408622



Great. Now that you have a converged model, you can also experiment with alternative optimizers and initialization strategies to see if you can find a better global minimum. (After all, the current models may have converged to a local minimum.) 

## Using Weight Initializers

In this section you will to use alternative initialization and optimization strategies. At the end, you'll then be asked to select the model which you believe performs the best.  

##  He Initialization

In the cell below, sepcify the following in the first hidden layer:  
  - 100 units 
  - `'relu'` activation 
  - `input_shape` 
  - `kernel_initializer='he_normal'` 
  
[Documentation on the He Normal Initializer](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal)


```python
np.random.seed(123)
he_model = Sequential()

# Add the first hidden layer


# Add another hidden layer
he_model.add(layers.Dense(50, activation='relu'))

# Add an output layer
he_model.add(layers.Dense(1, activation='linear'))

# Compile the model

# Train the model

```


```python
# __SOLUTION__ 
np.random.seed(123)
he_model = Sequential()

# Add the first hidden layer
he_model.add(layers.Dense(100, kernel_initializer='he_normal', activation='relu', input_shape=(n_features)))

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
    26/33 [======================>.......] - ETA: 0s - loss: 0.4861 - mse: 0.4861

    2023-06-21 13:11:30.913414: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 8ms/step - loss: 0.4854 - mse: 0.4854 - val_loss: 0.2083 - val_mse: 0.2083
    Epoch 2/150
    26/33 [======================>.......] - ETA: 0s - loss: 0.2497 - mse: 0.2497

    2023-06-21 13:11:31.180608: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 5ms/step - loss: 0.2473 - mse: 0.2473 - val_loss: 0.1742 - val_mse: 0.1742
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1956 - mse: 0.1956 - val_loss: 0.1588 - val_mse: 0.1588
    Epoch 4/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1749 - mse: 0.1749 - val_loss: 0.1610 - val_mse: 0.1610
    Epoch 5/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1522 - mse: 0.1522 - val_loss: 0.1469 - val_mse: 0.1469
    Epoch 6/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1361 - mse: 0.1361 - val_loss: 0.1466 - val_mse: 0.1466
    Epoch 7/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1297 - mse: 0.1297 - val_loss: 0.1411 - val_mse: 0.1411
    Epoch 8/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1166 - mse: 0.1166 - val_loss: 0.1433 - val_mse: 0.1433
    Epoch 9/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1045 - mse: 0.1045 - val_loss: 0.1475 - val_mse: 0.1475
    Epoch 10/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1014 - mse: 0.1014 - val_loss: 0.1446 - val_mse: 0.1446
    Epoch 11/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0960 - mse: 0.0960 - val_loss: 0.1449 - val_mse: 0.1449
    Epoch 12/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0938 - mse: 0.0938 - val_loss: 0.1508 - val_mse: 0.1508
    Epoch 13/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0840 - mse: 0.0840 - val_loss: 0.1466 - val_mse: 0.1466
    Epoch 14/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0803 - mse: 0.0803 - val_loss: 0.1416 - val_mse: 0.1416
    Epoch 15/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0758 - mse: 0.0758 - val_loss: 0.1460 - val_mse: 0.1460
    Epoch 16/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0731 - mse: 0.0731 - val_loss: 0.1503 - val_mse: 0.1503
    Epoch 17/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0703 - mse: 0.0703 - val_loss: 0.1468 - val_mse: 0.1468
    Epoch 18/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0684 - mse: 0.0684 - val_loss: 0.1435 - val_mse: 0.1435
    Epoch 19/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0666 - mse: 0.0666 - val_loss: 0.1453 - val_mse: 0.1453
    Epoch 20/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0610 - mse: 0.0610 - val_loss: 0.1442 - val_mse: 0.1442
    Epoch 21/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0599 - mse: 0.0599 - val_loss: 0.1428 - val_mse: 0.1428
    Epoch 22/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0587 - mse: 0.0587 - val_loss: 0.1477 - val_mse: 0.1477
    Epoch 23/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0563 - mse: 0.0563 - val_loss: 0.1428 - val_mse: 0.1428
    Epoch 24/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0555 - mse: 0.0555 - val_loss: 0.1464 - val_mse: 0.1464
    Epoch 25/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0534 - mse: 0.0534 - val_loss: 0.1429 - val_mse: 0.1429
    Epoch 26/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0515 - mse: 0.0515 - val_loss: 0.1469 - val_mse: 0.1469
    Epoch 27/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0500 - mse: 0.0500 - val_loss: 0.1436 - val_mse: 0.1436
    Epoch 28/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0494 - mse: 0.0494 - val_loss: 0.1449 - val_mse: 0.1449
    Epoch 29/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0468 - mse: 0.0468 - val_loss: 0.1430 - val_mse: 0.1430
    Epoch 30/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0466 - mse: 0.0466 - val_loss: 0.1533 - val_mse: 0.1533
    Epoch 31/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0461 - mse: 0.0461 - val_loss: 0.1444 - val_mse: 0.1444
    Epoch 32/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0444 - mse: 0.0444 - val_loss: 0.1473 - val_mse: 0.1473
    Epoch 33/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0433 - mse: 0.0433 - val_loss: 0.1501 - val_mse: 0.1501
    Epoch 34/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0417 - mse: 0.0417 - val_loss: 0.1477 - val_mse: 0.1477
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0421 - mse: 0.0421 - val_loss: 0.1458 - val_mse: 0.1458
    Epoch 36/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0408 - mse: 0.0408 - val_loss: 0.1454 - val_mse: 0.1454
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0399 - mse: 0.0399 - val_loss: 0.1449 - val_mse: 0.1449
    Epoch 38/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0387 - mse: 0.0387 - val_loss: 0.1448 - val_mse: 0.1448
    Epoch 39/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0388 - mse: 0.0388 - val_loss: 0.1429 - val_mse: 0.1429
    Epoch 40/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0373 - mse: 0.0373 - val_loss: 0.1454 - val_mse: 0.1454
    Epoch 41/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0358 - mse: 0.0358 - val_loss: 0.1470 - val_mse: 0.1470
    Epoch 42/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0358 - mse: 0.0358 - val_loss: 0.1498 - val_mse: 0.1498
    Epoch 43/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0352 - mse: 0.0352 - val_loss: 0.1468 - val_mse: 0.1468
    Epoch 44/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0347 - mse: 0.0347 - val_loss: 0.1463 - val_mse: 0.1463
    Epoch 45/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0334 - mse: 0.0334 - val_loss: 0.1498 - val_mse: 0.1498
    Epoch 46/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0330 - mse: 0.0330 - val_loss: 0.1478 - val_mse: 0.1478
    Epoch 47/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0321 - mse: 0.0321 - val_loss: 0.1478 - val_mse: 0.1478
    Epoch 48/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0314 - mse: 0.0314 - val_loss: 0.1469 - val_mse: 0.1469
    Epoch 49/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0315 - mse: 0.0315 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 50/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0313 - mse: 0.0313 - val_loss: 0.1464 - val_mse: 0.1464
    Epoch 51/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0298 - mse: 0.0298 - val_loss: 0.1472 - val_mse: 0.1472
    Epoch 52/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0295 - mse: 0.0295 - val_loss: 0.1457 - val_mse: 0.1457
    Epoch 53/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0283 - mse: 0.0283 - val_loss: 0.1505 - val_mse: 0.1505
    Epoch 54/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0288 - mse: 0.0288 - val_loss: 0.1464 - val_mse: 0.1464
    Epoch 55/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0278 - mse: 0.0278 - val_loss: 0.1460 - val_mse: 0.1460
    Epoch 56/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0283 - mse: 0.0283 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0280 - mse: 0.0280 - val_loss: 0.1477 - val_mse: 0.1477
    Epoch 58/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0270 - mse: 0.0270 - val_loss: 0.1489 - val_mse: 0.1489
    Epoch 59/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0265 - mse: 0.0265 - val_loss: 0.1457 - val_mse: 0.1457
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0258 - mse: 0.0258 - val_loss: 0.1483 - val_mse: 0.1483
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0258 - mse: 0.0258 - val_loss: 0.1463 - val_mse: 0.1463
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0255 - mse: 0.0255 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0243 - mse: 0.0243 - val_loss: 0.1473 - val_mse: 0.1473
    Epoch 64/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0243 - mse: 0.0243 - val_loss: 0.1470 - val_mse: 0.1470
    Epoch 65/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0245 - mse: 0.0245 - val_loss: 0.1447 - val_mse: 0.1447
    Epoch 66/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0236 - mse: 0.0236 - val_loss: 0.1477 - val_mse: 0.1477
    Epoch 67/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0237 - mse: 0.0237 - val_loss: 0.1452 - val_mse: 0.1452
    Epoch 68/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0231 - mse: 0.0231 - val_loss: 0.1460 - val_mse: 0.1460
    Epoch 69/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0227 - mse: 0.0227 - val_loss: 0.1459 - val_mse: 0.1459
    Epoch 70/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0222 - mse: 0.0222 - val_loss: 0.1454 - val_mse: 0.1454
    Epoch 71/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0219 - mse: 0.0219 - val_loss: 0.1457 - val_mse: 0.1457
    Epoch 72/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0221 - mse: 0.0221 - val_loss: 0.1460 - val_mse: 0.1460
    Epoch 73/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0215 - mse: 0.0215 - val_loss: 0.1456 - val_mse: 0.1456
    Epoch 74/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0207 - mse: 0.0207 - val_loss: 0.1455 - val_mse: 0.1455
    Epoch 75/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0208 - mse: 0.0208 - val_loss: 0.1449 - val_mse: 0.1449
    Epoch 76/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0208 - mse: 0.0208 - val_loss: 0.1504 - val_mse: 0.1504
    Epoch 77/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0203 - mse: 0.0203 - val_loss: 0.1458 - val_mse: 0.1458
    Epoch 78/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0197 - mse: 0.0197 - val_loss: 0.1449 - val_mse: 0.1449
    Epoch 79/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0205 - mse: 0.0205 - val_loss: 0.1443 - val_mse: 0.1443
    Epoch 80/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0190 - mse: 0.0190 - val_loss: 0.1488 - val_mse: 0.1488
    Epoch 81/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0186 - mse: 0.0186 - val_loss: 0.1465 - val_mse: 0.1465
    Epoch 82/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0191 - mse: 0.0191 - val_loss: 0.1451 - val_mse: 0.1451
    Epoch 83/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0189 - mse: 0.0189 - val_loss: 0.1477 - val_mse: 0.1477
    Epoch 84/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0182 - mse: 0.0182 - val_loss: 0.1471 - val_mse: 0.1471
    Epoch 85/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0184 - mse: 0.0184 - val_loss: 0.1487 - val_mse: 0.1487
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0180 - mse: 0.0180 - val_loss: 0.1471 - val_mse: 0.1471
    Epoch 87/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0177 - mse: 0.0177 - val_loss: 0.1465 - val_mse: 0.1465
    Epoch 88/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0173 - mse: 0.0173 - val_loss: 0.1471 - val_mse: 0.1471
    Epoch 89/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0172 - mse: 0.0172 - val_loss: 0.1470 - val_mse: 0.1470
    Epoch 90/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0171 - mse: 0.0171 - val_loss: 0.1464 - val_mse: 0.1464
    Epoch 91/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0167 - mse: 0.0167 - val_loss: 0.1464 - val_mse: 0.1464
    Epoch 92/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0167 - mse: 0.0167 - val_loss: 0.1455 - val_mse: 0.1455
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0162 - mse: 0.0162 - val_loss: 0.1467 - val_mse: 0.1467
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0165 - mse: 0.0165 - val_loss: 0.1472 - val_mse: 0.1472
    Epoch 95/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0164 - mse: 0.0164 - val_loss: 0.1477 - val_mse: 0.1477
    Epoch 96/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0159 - mse: 0.0159 - val_loss: 0.1473 - val_mse: 0.1473
    Epoch 97/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0154 - mse: 0.0154 - val_loss: 0.1455 - val_mse: 0.1455
    Epoch 98/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0156 - mse: 0.0156 - val_loss: 0.1467 - val_mse: 0.1467
    Epoch 99/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0153 - mse: 0.0153 - val_loss: 0.1472 - val_mse: 0.1472
    Epoch 100/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0154 - mse: 0.0154 - val_loss: 0.1470 - val_mse: 0.1470
    Epoch 101/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0150 - mse: 0.0150 - val_loss: 0.1454 - val_mse: 0.1454
    Epoch 102/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0146 - mse: 0.0146 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 103/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0147 - mse: 0.0147 - val_loss: 0.1475 - val_mse: 0.1475
    Epoch 104/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0144 - mse: 0.0144 - val_loss: 0.1456 - val_mse: 0.1456
    Epoch 105/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0140 - mse: 0.0140 - val_loss: 0.1461 - val_mse: 0.1461
    Epoch 106/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0142 - mse: 0.0142 - val_loss: 0.1475 - val_mse: 0.1475
    Epoch 107/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0142 - mse: 0.0142 - val_loss: 0.1491 - val_mse: 0.1491
    Epoch 108/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0141 - mse: 0.0141 - val_loss: 0.1465 - val_mse: 0.1465
    Epoch 109/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0140 - mse: 0.0140 - val_loss: 0.1473 - val_mse: 0.1473
    Epoch 110/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0138 - mse: 0.0138 - val_loss: 0.1473 - val_mse: 0.1473
    Epoch 111/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0134 - mse: 0.0134 - val_loss: 0.1484 - val_mse: 0.1484
    Epoch 112/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0133 - mse: 0.0133 - val_loss: 0.1491 - val_mse: 0.1491
    Epoch 113/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0130 - mse: 0.0130 - val_loss: 0.1494 - val_mse: 0.1494
    Epoch 114/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0131 - mse: 0.0131 - val_loss: 0.1484 - val_mse: 0.1484
    Epoch 115/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0127 - mse: 0.0127 - val_loss: 0.1476 - val_mse: 0.1476
    Epoch 116/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0130 - mse: 0.0130 - val_loss: 0.1480 - val_mse: 0.1480
    Epoch 117/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0126 - mse: 0.0126 - val_loss: 0.1473 - val_mse: 0.1473
    Epoch 118/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0124 - mse: 0.0124 - val_loss: 0.1470 - val_mse: 0.1470
    Epoch 119/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0124 - mse: 0.0124 - val_loss: 0.1471 - val_mse: 0.1471
    Epoch 120/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0122 - mse: 0.0122 - val_loss: 0.1490 - val_mse: 0.1490
    Epoch 121/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 0.1482 - val_mse: 0.1482
    Epoch 122/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0119 - mse: 0.0119 - val_loss: 0.1489 - val_mse: 0.1489
    Epoch 123/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0116 - mse: 0.0116 - val_loss: 0.1489 - val_mse: 0.1489
    Epoch 124/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0116 - mse: 0.0116 - val_loss: 0.1481 - val_mse: 0.1481
    Epoch 125/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0115 - mse: 0.0115 - val_loss: 0.1496 - val_mse: 0.1496
    Epoch 126/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 127/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.1490 - val_mse: 0.1490
    Epoch 128/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1493 - val_mse: 0.1493
    Epoch 129/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1495 - val_mse: 0.1495
    Epoch 130/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0109 - mse: 0.0109 - val_loss: 0.1489 - val_mse: 0.1489
    Epoch 131/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0109 - mse: 0.0109 - val_loss: 0.1495 - val_mse: 0.1495
    Epoch 132/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0105 - mse: 0.0105 - val_loss: 0.1496 - val_mse: 0.1496
    Epoch 133/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 0.1483 - val_mse: 0.1483
    Epoch 134/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0105 - mse: 0.0105 - val_loss: 0.1499 - val_mse: 0.1499
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0103 - mse: 0.0103 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0102 - mse: 0.0102 - val_loss: 0.1502 - val_mse: 0.1502
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0101 - mse: 0.0101 - val_loss: 0.1501 - val_mse: 0.1501
    Epoch 138/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0100 - mse: 0.0100 - val_loss: 0.1498 - val_mse: 0.1498
    Epoch 139/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0102 - mse: 0.0102 - val_loss: 0.1500 - val_mse: 0.1500
    Epoch 140/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1490 - val_mse: 0.1490
    Epoch 141/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0097 - mse: 0.0097 - val_loss: 0.1495 - val_mse: 0.1495
    Epoch 142/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0097 - mse: 0.0097 - val_loss: 0.1493 - val_mse: 0.1493
    Epoch 143/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0094 - mse: 0.0094 - val_loss: 0.1505 - val_mse: 0.1505
    Epoch 144/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0096 - mse: 0.0096 - val_loss: 0.1495 - val_mse: 0.1495
    Epoch 145/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0094 - mse: 0.0094 - val_loss: 0.1501 - val_mse: 0.1501
    Epoch 146/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0091 - mse: 0.0091 - val_loss: 0.1499 - val_mse: 0.1499
    Epoch 147/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0092 - mse: 0.0092 - val_loss: 0.1502 - val_mse: 0.1502
    Epoch 148/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0091 - mse: 0.0091 - val_loss: 0.1498 - val_mse: 0.1498
    Epoch 149/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0090 - mse: 0.0090 - val_loss: 0.1511 - val_mse: 0.1511
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0090 - mse: 0.0090 - val_loss: 0.1501 - val_mse: 0.1501





    <keras.callbacks.History at 0x17daa31f0>



Evaluate the model (`he_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```


```python
# __SOLUTION__ 
# Evaluate the model on training data
he_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 3ms/step - loss: 0.0085 - mse: 0.0085





    [0.008450852707028389, 0.008450852707028389]



Evaluate the model (`he_model`) on validate data (`X_val` and `y_val_scaled`)


```python
# Evaluate the model on validate data

```


```python
# __SOLUTION__ 
# Evaluate the model on validate data
he_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 4ms/step - loss: 0.1501 - mse: 0.1501





    [0.15014097094535828, 0.15014097094535828]



## Lecun Initialization 

In the cell below, sepcify the following in the first hidden layer:  
  - 100 units 
  - `'relu'` activation 
  - `input_shape` 
  - `kernel_initializer='lecun_normal'` 
  
[Documentation on the Lecun Normal Initializer](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal)


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


# Train the model
lecun_model.fit(X_train, 
                y_train_scaled, 
                batch_size=32, 
                epochs=150, 
                validation_data=(X_val, y_val_scaled))
```


```python
# __SOLUTION__ 
np.random.seed(123)
lecun_model = Sequential()

# Add the first hidden layer
lecun_model.add(layers.Dense(100, kernel_initializer='lecun_normal', activation='relu', input_shape=(n_features)))

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
    26/33 [======================>.......] - ETA: 0s - loss: 0.4726 - mse: 0.4726

    2023-06-21 13:12:06.590839: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 9ms/step - loss: 0.4258 - mse: 0.4258 - val_loss: 0.1528 - val_mse: 0.1528
    Epoch 2/150
    27/33 [=======================>......] - ETA: 0s - loss: 0.1524 - mse: 0.1524

    2023-06-21 13:12:06.873200: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 5ms/step - loss: 0.2354 - mse: 0.2354 - val_loss: 0.2437 - val_mse: 0.2437
    Epoch 3/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1988 - mse: 0.1988 - val_loss: 0.1164 - val_mse: 0.1164
    Epoch 4/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.1791 - mse: 0.1791 - val_loss: 0.1194 - val_mse: 0.1194
    Epoch 5/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1592 - mse: 0.1592 - val_loss: 0.1192 - val_mse: 0.1192
    Epoch 6/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.1458 - mse: 0.1458 - val_loss: 0.1099 - val_mse: 0.1099
    Epoch 7/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.1304 - mse: 0.1304 - val_loss: 0.1108 - val_mse: 0.1108
    Epoch 8/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1240 - mse: 0.1240 - val_loss: 0.1112 - val_mse: 0.1112
    Epoch 9/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.1127 - mse: 0.1127 - val_loss: 0.1116 - val_mse: 0.1116
    Epoch 10/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.1033 - mse: 0.1033 - val_loss: 0.1080 - val_mse: 0.1080
    Epoch 11/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0989 - mse: 0.0989 - val_loss: 0.1107 - val_mse: 0.1107
    Epoch 12/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0951 - mse: 0.0951 - val_loss: 0.1131 - val_mse: 0.1131
    Epoch 13/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0871 - mse: 0.0871 - val_loss: 0.1170 - val_mse: 0.1170
    Epoch 14/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0817 - mse: 0.0817 - val_loss: 0.1171 - val_mse: 0.1171
    Epoch 15/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0784 - mse: 0.0784 - val_loss: 0.1200 - val_mse: 0.1200
    Epoch 16/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0752 - mse: 0.0752 - val_loss: 0.1149 - val_mse: 0.1149
    Epoch 17/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0705 - mse: 0.0705 - val_loss: 0.1180 - val_mse: 0.1180
    Epoch 18/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0694 - mse: 0.0694 - val_loss: 0.1214 - val_mse: 0.1214
    Epoch 19/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0649 - mse: 0.0649 - val_loss: 0.1269 - val_mse: 0.1269
    Epoch 20/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0629 - mse: 0.0629 - val_loss: 0.1254 - val_mse: 0.1254
    Epoch 21/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0600 - mse: 0.0600 - val_loss: 0.1262 - val_mse: 0.1262
    Epoch 22/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0585 - mse: 0.0585 - val_loss: 0.1254 - val_mse: 0.1254
    Epoch 23/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0538 - mse: 0.0538 - val_loss: 0.1287 - val_mse: 0.1287
    Epoch 24/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0548 - mse: 0.0548 - val_loss: 0.1267 - val_mse: 0.1267
    Epoch 25/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0498 - mse: 0.0498 - val_loss: 0.1255 - val_mse: 0.1255
    Epoch 26/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0489 - mse: 0.0489 - val_loss: 0.1242 - val_mse: 0.1242
    Epoch 27/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0476 - mse: 0.0476 - val_loss: 0.1322 - val_mse: 0.1322
    Epoch 28/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0458 - mse: 0.0458 - val_loss: 0.1289 - val_mse: 0.1289
    Epoch 29/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0454 - mse: 0.0454 - val_loss: 0.1301 - val_mse: 0.1301
    Epoch 30/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0421 - mse: 0.0421 - val_loss: 0.1327 - val_mse: 0.1327
    Epoch 31/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0413 - mse: 0.0413 - val_loss: 0.1330 - val_mse: 0.1330
    Epoch 32/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0407 - mse: 0.0407 - val_loss: 0.1301 - val_mse: 0.1301
    Epoch 33/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0393 - mse: 0.0393 - val_loss: 0.1321 - val_mse: 0.1321
    Epoch 34/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0372 - mse: 0.0372 - val_loss: 0.1313 - val_mse: 0.1313
    Epoch 35/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0366 - mse: 0.0366 - val_loss: 0.1325 - val_mse: 0.1325
    Epoch 36/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0350 - mse: 0.0350 - val_loss: 0.1374 - val_mse: 0.1374
    Epoch 37/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0351 - mse: 0.0351 - val_loss: 0.1371 - val_mse: 0.1371
    Epoch 38/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0335 - mse: 0.0335 - val_loss: 0.1395 - val_mse: 0.1395
    Epoch 39/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0328 - mse: 0.0328 - val_loss: 0.1358 - val_mse: 0.1358
    Epoch 40/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0328 - mse: 0.0328 - val_loss: 0.1359 - val_mse: 0.1359
    Epoch 41/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0316 - mse: 0.0316 - val_loss: 0.1380 - val_mse: 0.1380
    Epoch 42/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0317 - mse: 0.0317 - val_loss: 0.1391 - val_mse: 0.1391
    Epoch 43/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0303 - mse: 0.0303 - val_loss: 0.1366 - val_mse: 0.1366
    Epoch 44/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0293 - mse: 0.0293 - val_loss: 0.1384 - val_mse: 0.1384
    Epoch 45/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0285 - mse: 0.0285 - val_loss: 0.1379 - val_mse: 0.1379
    Epoch 46/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0289 - mse: 0.0289 - val_loss: 0.1369 - val_mse: 0.1369
    Epoch 47/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0274 - mse: 0.0274 - val_loss: 0.1419 - val_mse: 0.1419
    Epoch 48/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0280 - mse: 0.0280 - val_loss: 0.1427 - val_mse: 0.1427
    Epoch 49/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0265 - mse: 0.0265 - val_loss: 0.1398 - val_mse: 0.1398
    Epoch 50/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0253 - mse: 0.0253 - val_loss: 0.1410 - val_mse: 0.1410
    Epoch 51/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0254 - mse: 0.0254 - val_loss: 0.1408 - val_mse: 0.1408
    Epoch 52/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0246 - mse: 0.0246 - val_loss: 0.1412 - val_mse: 0.1412
    Epoch 53/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0242 - mse: 0.0242 - val_loss: 0.1428 - val_mse: 0.1428
    Epoch 54/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0243 - mse: 0.0243 - val_loss: 0.1431 - val_mse: 0.1431
    Epoch 55/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0242 - mse: 0.0242 - val_loss: 0.1399 - val_mse: 0.1399
    Epoch 56/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0227 - mse: 0.0227 - val_loss: 0.1430 - val_mse: 0.1430
    Epoch 57/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0223 - mse: 0.0223 - val_loss: 0.1423 - val_mse: 0.1423
    Epoch 58/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0220 - mse: 0.0220 - val_loss: 0.1422 - val_mse: 0.1422
    Epoch 59/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0215 - mse: 0.0215 - val_loss: 0.1440 - val_mse: 0.1440
    Epoch 60/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0214 - mse: 0.0214 - val_loss: 0.1429 - val_mse: 0.1429
    Epoch 61/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0212 - mse: 0.0212 - val_loss: 0.1447 - val_mse: 0.1447
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0204 - mse: 0.0204 - val_loss: 0.1455 - val_mse: 0.1455
    Epoch 63/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0202 - mse: 0.0202 - val_loss: 0.1449 - val_mse: 0.1449
    Epoch 64/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0196 - mse: 0.0196 - val_loss: 0.1428 - val_mse: 0.1428
    Epoch 65/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0194 - mse: 0.0194 - val_loss: 0.1465 - val_mse: 0.1465
    Epoch 66/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0189 - mse: 0.0189 - val_loss: 0.1448 - val_mse: 0.1448
    Epoch 67/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0188 - mse: 0.0188 - val_loss: 0.1467 - val_mse: 0.1467
    Epoch 68/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0182 - mse: 0.0182 - val_loss: 0.1459 - val_mse: 0.1459
    Epoch 69/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0179 - mse: 0.0179 - val_loss: 0.1424 - val_mse: 0.1424
    Epoch 70/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0176 - mse: 0.0176 - val_loss: 0.1451 - val_mse: 0.1451
    Epoch 71/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0174 - mse: 0.0174 - val_loss: 0.1464 - val_mse: 0.1464
    Epoch 72/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0168 - mse: 0.0168 - val_loss: 0.1460 - val_mse: 0.1460
    Epoch 73/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0164 - mse: 0.0164 - val_loss: 0.1467 - val_mse: 0.1467
    Epoch 74/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0164 - mse: 0.0164 - val_loss: 0.1461 - val_mse: 0.1461
    Epoch 75/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0161 - mse: 0.0161 - val_loss: 0.1484 - val_mse: 0.1484
    Epoch 76/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0161 - mse: 0.0161 - val_loss: 0.1489 - val_mse: 0.1489
    Epoch 77/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0158 - mse: 0.0158 - val_loss: 0.1474 - val_mse: 0.1474
    Epoch 78/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0158 - mse: 0.0158 - val_loss: 0.1500 - val_mse: 0.1500
    Epoch 79/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0151 - mse: 0.0151 - val_loss: 0.1486 - val_mse: 0.1486
    Epoch 80/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0148 - mse: 0.0148 - val_loss: 0.1464 - val_mse: 0.1464
    Epoch 81/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0145 - mse: 0.0145 - val_loss: 0.1486 - val_mse: 0.1486
    Epoch 82/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0143 - mse: 0.0143 - val_loss: 0.1487 - val_mse: 0.1487
    Epoch 83/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0144 - mse: 0.0144 - val_loss: 0.1489 - val_mse: 0.1489
    Epoch 84/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0142 - mse: 0.0142 - val_loss: 0.1479 - val_mse: 0.1479
    Epoch 85/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0137 - mse: 0.0137 - val_loss: 0.1506 - val_mse: 0.1506
    Epoch 86/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0135 - mse: 0.0135 - val_loss: 0.1498 - val_mse: 0.1498
    Epoch 87/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0135 - mse: 0.0135 - val_loss: 0.1497 - val_mse: 0.1497
    Epoch 88/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0131 - mse: 0.0131 - val_loss: 0.1539 - val_mse: 0.1539
    Epoch 89/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0133 - mse: 0.0133 - val_loss: 0.1515 - val_mse: 0.1515
    Epoch 90/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0129 - mse: 0.0129 - val_loss: 0.1509 - val_mse: 0.1509
    Epoch 91/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0129 - mse: 0.0129 - val_loss: 0.1520 - val_mse: 0.1520
    Epoch 92/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0124 - mse: 0.0124 - val_loss: 0.1523 - val_mse: 0.1523
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0123 - mse: 0.0123 - val_loss: 0.1514 - val_mse: 0.1514
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0122 - mse: 0.0122 - val_loss: 0.1505 - val_mse: 0.1505
    Epoch 95/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0122 - mse: 0.0122 - val_loss: 0.1527 - val_mse: 0.1527
    Epoch 96/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0120 - mse: 0.0120 - val_loss: 0.1507 - val_mse: 0.1507
    Epoch 97/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0117 - mse: 0.0117 - val_loss: 0.1498 - val_mse: 0.1498
    Epoch 98/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 0.1512 - val_mse: 0.1512
    Epoch 99/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.1503 - val_mse: 0.1503
    Epoch 100/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0112 - mse: 0.0112 - val_loss: 0.1512 - val_mse: 0.1512
    Epoch 101/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1535 - val_mse: 0.1535
    Epoch 102/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0108 - mse: 0.0108 - val_loss: 0.1520 - val_mse: 0.1520
    Epoch 103/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0108 - mse: 0.0108 - val_loss: 0.1519 - val_mse: 0.1519
    Epoch 104/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0105 - mse: 0.0105 - val_loss: 0.1531 - val_mse: 0.1531
    Epoch 105/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0106 - mse: 0.0106 - val_loss: 0.1531 - val_mse: 0.1531
    Epoch 106/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0106 - mse: 0.0106 - val_loss: 0.1527 - val_mse: 0.1527
    Epoch 107/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0104 - mse: 0.0104 - val_loss: 0.1531 - val_mse: 0.1531
    Epoch 108/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0100 - mse: 0.0100 - val_loss: 0.1543 - val_mse: 0.1543
    Epoch 109/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1547 - val_mse: 0.1547
    Epoch 110/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1540 - val_mse: 0.1540
    Epoch 111/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1525 - val_mse: 0.1525
    Epoch 112/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0098 - mse: 0.0098 - val_loss: 0.1549 - val_mse: 0.1549
    Epoch 113/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0095 - mse: 0.0095 - val_loss: 0.1538 - val_mse: 0.1538
    Epoch 114/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 0.1549 - val_mse: 0.1549
    Epoch 115/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0093 - mse: 0.0093 - val_loss: 0.1540 - val_mse: 0.1540
    Epoch 116/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0091 - mse: 0.0091 - val_loss: 0.1533 - val_mse: 0.1533
    Epoch 117/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 0.1525 - val_mse: 0.1525
    Epoch 118/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0092 - mse: 0.0092 - val_loss: 0.1523 - val_mse: 0.1523
    Epoch 119/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0087 - mse: 0.0087 - val_loss: 0.1543 - val_mse: 0.1543
    Epoch 120/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0088 - mse: 0.0088 - val_loss: 0.1543 - val_mse: 0.1543
    Epoch 121/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0084 - mse: 0.0084 - val_loss: 0.1547 - val_mse: 0.1547
    Epoch 122/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0084 - mse: 0.0084 - val_loss: 0.1553 - val_mse: 0.1553
    Epoch 123/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0084 - mse: 0.0084 - val_loss: 0.1549 - val_mse: 0.1549
    Epoch 124/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.1555 - val_mse: 0.1555
    Epoch 125/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0081 - mse: 0.0081 - val_loss: 0.1559 - val_mse: 0.1559
    Epoch 126/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0083 - mse: 0.0083 - val_loss: 0.1546 - val_mse: 0.1546
    Epoch 127/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0079 - mse: 0.0079 - val_loss: 0.1564 - val_mse: 0.1564
    Epoch 128/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0080 - mse: 0.0080 - val_loss: 0.1552 - val_mse: 0.1552
    Epoch 129/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0076 - mse: 0.0076 - val_loss: 0.1543 - val_mse: 0.1543
    Epoch 130/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0076 - mse: 0.0076 - val_loss: 0.1554 - val_mse: 0.1554
    Epoch 131/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0076 - mse: 0.0076 - val_loss: 0.1552 - val_mse: 0.1552
    Epoch 132/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0075 - mse: 0.0075 - val_loss: 0.1534 - val_mse: 0.1534
    Epoch 133/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.1559 - val_mse: 0.1559
    Epoch 134/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0074 - mse: 0.0074 - val_loss: 0.1559 - val_mse: 0.1559
    Epoch 135/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0073 - mse: 0.0073 - val_loss: 0.1566 - val_mse: 0.1566
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0072 - mse: 0.0072 - val_loss: 0.1560 - val_mse: 0.1560
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0070 - mse: 0.0070 - val_loss: 0.1560 - val_mse: 0.1560
    Epoch 138/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0071 - mse: 0.0071 - val_loss: 0.1574 - val_mse: 0.1574
    Epoch 139/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0070 - mse: 0.0070 - val_loss: 0.1559 - val_mse: 0.1559
    Epoch 140/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1563 - val_mse: 0.1563
    Epoch 141/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0067 - mse: 0.0067 - val_loss: 0.1558 - val_mse: 0.1558
    Epoch 142/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0067 - mse: 0.0067 - val_loss: 0.1565 - val_mse: 0.1565
    Epoch 143/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0067 - mse: 0.0067 - val_loss: 0.1580 - val_mse: 0.1580
    Epoch 144/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.1580 - val_mse: 0.1580
    Epoch 145/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.1560 - val_mse: 0.1560
    Epoch 146/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0064 - mse: 0.0064 - val_loss: 0.1571 - val_mse: 0.1571
    Epoch 147/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0062 - mse: 0.0062 - val_loss: 0.1567 - val_mse: 0.1567
    Epoch 148/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0063 - mse: 0.0063 - val_loss: 0.1583 - val_mse: 0.1583
    Epoch 149/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0061 - mse: 0.0061 - val_loss: 0.1561 - val_mse: 0.1561
    Epoch 150/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0062 - mse: 0.0062 - val_loss: 0.1567 - val_mse: 0.1567





    <keras.callbacks.History at 0x17ade6af0>



Evaluate the model (`lecun_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```


```python
# __SOLUTION__ 
# Evaluate the model on training data
lecun_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 4ms/step - loss: 0.0056 - mse: 0.0056





    [0.0055966624058783054, 0.0055966624058783054]



Evaluate the model (`lecun_model`) on validate data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data

```


```python
# __SOLUTION__ 
# Evaluate the model on validate data
lecun_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 4ms/step - loss: 0.1567 - mse: 0.1567





    [0.15667259693145752, 0.15667259693145752]



Not much of a difference, but a useful note to consider when tuning your network. Next, let's investigate the impact of various optimization algorithms.

## RMSprop 

Compile the `rmsprop_model` with: 

- `'rmsprop'` as the optimizer 
- track `'mse'` as the loss and metric  

[Documentation on the RMS Prop Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/RMSprop)


```python
np.random.seed(123)
rmsprop_model = Sequential()
rmsprop_model.add(layers.Dense(100, activation='relu', input_shape=n_features))
rmsprop_model.add(layers.Dense(50, activation='relu'))
rmsprop_model.add(layers.Dense(1, activation='linear'))

# Compile the model
# Code here

# Train the model
rmsprop_model.fit(X_train, 
                  y_train_scaled, 
                  batch_size=32, 
                  epochs=150, 
                  validation_data=(X_val, y_val_scaled))
```


```python
# __SOLUTION__ 
np.random.seed(123)
rmsprop_model = Sequential()
rmsprop_model.add(layers.Dense(100, activation='relu', input_shape=(n_features)))
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
     1/33 [..............................] - ETA: 10s - loss: 2.0703 - mse: 2.0703

    2023-06-21 13:12:57.082221: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 1s 11ms/step - loss: 0.4452 - mse: 0.4452 - val_loss: 0.1292 - val_mse: 0.1292
    Epoch 2/150
    19/33 [================>.............] - ETA: 0s - loss: 0.1871 - mse: 0.1871

    2023-06-21 13:12:57.523036: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 7ms/step - loss: 0.1765 - mse: 0.1765 - val_loss: 0.1326 - val_mse: 0.1326
    Epoch 3/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.1240 - mse: 0.1240 - val_loss: 0.1050 - val_mse: 0.1050
    Epoch 4/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0965 - mse: 0.0965 - val_loss: 0.1187 - val_mse: 0.1187
    Epoch 5/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0812 - mse: 0.0812 - val_loss: 0.1016 - val_mse: 0.1016
    Epoch 6/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0610 - mse: 0.0610 - val_loss: 0.1149 - val_mse: 0.1149
    Epoch 7/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0470 - mse: 0.0470 - val_loss: 0.1353 - val_mse: 0.1353
    Epoch 8/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0504 - mse: 0.0504 - val_loss: 0.1362 - val_mse: 0.1362
    Epoch 9/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0376 - mse: 0.0376 - val_loss: 0.0982 - val_mse: 0.0982
    Epoch 10/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0362 - mse: 0.0362 - val_loss: 0.1167 - val_mse: 0.1167
    Epoch 11/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0326 - mse: 0.0326 - val_loss: 0.1337 - val_mse: 0.1337
    Epoch 12/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0278 - mse: 0.0278 - val_loss: 0.1479 - val_mse: 0.1479
    Epoch 13/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0332 - mse: 0.0332 - val_loss: 0.1065 - val_mse: 0.1065
    Epoch 14/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0262 - mse: 0.0262 - val_loss: 0.1176 - val_mse: 0.1176
    Epoch 15/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0259 - mse: 0.0259 - val_loss: 0.1122 - val_mse: 0.1122
    Epoch 16/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0265 - mse: 0.0265 - val_loss: 0.1568 - val_mse: 0.1568
    Epoch 17/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0203 - mse: 0.0203 - val_loss: 0.1050 - val_mse: 0.1050
    Epoch 18/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0206 - mse: 0.0206 - val_loss: 0.0951 - val_mse: 0.0951
    Epoch 19/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0227 - mse: 0.0227 - val_loss: 0.1057 - val_mse: 0.1057
    Epoch 20/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0196 - mse: 0.0196 - val_loss: 0.1248 - val_mse: 0.1248
    Epoch 21/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0187 - mse: 0.0187 - val_loss: 0.1075 - val_mse: 0.1075
    Epoch 22/150
    33/33 [==============================] - 0s 9ms/step - loss: 0.0226 - mse: 0.0226 - val_loss: 0.1104 - val_mse: 0.1104
    Epoch 23/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0152 - mse: 0.0152 - val_loss: 0.1137 - val_mse: 0.1137
    Epoch 24/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0162 - mse: 0.0162 - val_loss: 0.1207 - val_mse: 0.1207
    Epoch 25/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0186 - mse: 0.0186 - val_loss: 0.1267 - val_mse: 0.1267
    Epoch 26/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0156 - mse: 0.0156 - val_loss: 0.1248 - val_mse: 0.1248
    Epoch 27/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0152 - mse: 0.0152 - val_loss: 0.0998 - val_mse: 0.0998
    Epoch 28/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0140 - mse: 0.0140 - val_loss: 0.1160 - val_mse: 0.1160
    Epoch 29/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0183 - mse: 0.0183 - val_loss: 0.1032 - val_mse: 0.1032
    Epoch 30/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0130 - mse: 0.0130 - val_loss: 0.0973 - val_mse: 0.0973
    Epoch 31/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0117 - mse: 0.0117 - val_loss: 0.1078 - val_mse: 0.1078
    Epoch 32/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0131 - mse: 0.0131 - val_loss: 0.0979 - val_mse: 0.0979
    Epoch 33/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0127 - mse: 0.0127 - val_loss: 0.1104 - val_mse: 0.1104
    Epoch 34/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0160 - mse: 0.0160 - val_loss: 0.0987 - val_mse: 0.0987
    Epoch 35/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0103 - mse: 0.0103 - val_loss: 0.1077 - val_mse: 0.1077
    Epoch 36/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 0.1026 - val_mse: 0.1026
    Epoch 37/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0144 - mse: 0.0144 - val_loss: 0.1028 - val_mse: 0.1028
    Epoch 38/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0114 - mse: 0.0114 - val_loss: 0.0963 - val_mse: 0.0963
    Epoch 39/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 0.1148 - val_mse: 0.1148
    Epoch 40/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0131 - mse: 0.0131 - val_loss: 0.0974 - val_mse: 0.0974
    Epoch 41/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0108 - mse: 0.0108 - val_loss: 0.0961 - val_mse: 0.0961
    Epoch 42/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0107 - mse: 0.0107 - val_loss: 0.1053 - val_mse: 0.1053
    Epoch 43/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0123 - mse: 0.0123 - val_loss: 0.1058 - val_mse: 0.1058
    Epoch 44/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0110 - mse: 0.0110 - val_loss: 0.1102 - val_mse: 0.1102
    Epoch 45/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0098 - mse: 0.0098 - val_loss: 0.1019 - val_mse: 0.1019
    Epoch 46/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0113 - mse: 0.0113 - val_loss: 0.0981 - val_mse: 0.0981
    Epoch 47/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0079 - mse: 0.0079 - val_loss: 0.1030 - val_mse: 0.1030
    Epoch 48/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0103 - mse: 0.0103 - val_loss: 0.1167 - val_mse: 0.1167
    Epoch 49/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0099 - mse: 0.0099 - val_loss: 0.1009 - val_mse: 0.1009
    Epoch 50/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0111 - mse: 0.0111 - val_loss: 0.1014 - val_mse: 0.1014
    Epoch 51/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0081 - mse: 0.0081 - val_loss: 0.0969 - val_mse: 0.0969
    Epoch 52/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0100 - mse: 0.0100 - val_loss: 0.1058 - val_mse: 0.1058
    Epoch 53/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0088 - mse: 0.0088 - val_loss: 0.1214 - val_mse: 0.1214
    Epoch 54/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0088 - mse: 0.0088 - val_loss: 0.1114 - val_mse: 0.1114
    Epoch 55/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0085 - mse: 0.0085 - val_loss: 0.1077 - val_mse: 0.1077
    Epoch 56/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0089 - mse: 0.0089 - val_loss: 0.0990 - val_mse: 0.0990
    Epoch 57/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.1028 - val_mse: 0.1028
    Epoch 58/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0078 - mse: 0.0078 - val_loss: 0.1077 - val_mse: 0.1077
    Epoch 59/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0085 - mse: 0.0085 - val_loss: 0.0965 - val_mse: 0.0965
    Epoch 60/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0080 - mse: 0.0080 - val_loss: 0.0947 - val_mse: 0.0947
    Epoch 61/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1074 - val_mse: 0.1074
    Epoch 62/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0079 - mse: 0.0079 - val_loss: 0.0905 - val_mse: 0.0905
    Epoch 63/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0081 - mse: 0.0081 - val_loss: 0.1115 - val_mse: 0.1115
    Epoch 64/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0096 - mse: 0.0096 - val_loss: 0.0998 - val_mse: 0.0998
    Epoch 65/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0065 - mse: 0.0065 - val_loss: 0.1053 - val_mse: 0.1053
    Epoch 66/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.1024 - val_mse: 0.1024
    Epoch 67/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0083 - mse: 0.0083 - val_loss: 0.1046 - val_mse: 0.1046
    Epoch 68/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0061 - mse: 0.0061 - val_loss: 0.0987 - val_mse: 0.0987
    Epoch 69/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1138 - val_mse: 0.1138
    Epoch 70/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0077 - mse: 0.0077 - val_loss: 0.0953 - val_mse: 0.0953
    Epoch 71/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1139 - val_mse: 0.1139
    Epoch 72/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0090 - mse: 0.0090 - val_loss: 0.0988 - val_mse: 0.0988
    Epoch 73/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.0969 - val_mse: 0.0969
    Epoch 74/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.0929 - val_mse: 0.0929
    Epoch 75/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0068 - mse: 0.0068 - val_loss: 0.1132 - val_mse: 0.1132
    Epoch 76/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0072 - mse: 0.0072 - val_loss: 0.0979 - val_mse: 0.0979
    Epoch 77/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.0976 - val_mse: 0.0976
    Epoch 78/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.1065 - val_mse: 0.1065
    Epoch 79/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0073 - mse: 0.0073 - val_loss: 0.1128 - val_mse: 0.1128
    Epoch 80/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0059 - mse: 0.0059 - val_loss: 0.0970 - val_mse: 0.0970
    Epoch 81/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0070 - mse: 0.0070 - val_loss: 0.1043 - val_mse: 0.1043
    Epoch 82/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0065 - mse: 0.0065 - val_loss: 0.0913 - val_mse: 0.0913
    Epoch 83/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.1024 - val_mse: 0.1024
    Epoch 84/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.0968 - val_mse: 0.0968
    Epoch 85/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0075 - mse: 0.0075 - val_loss: 0.0980 - val_mse: 0.0980
    Epoch 86/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0056 - mse: 0.0056 - val_loss: 0.0930 - val_mse: 0.0930
    Epoch 87/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0070 - mse: 0.0070 - val_loss: 0.1028 - val_mse: 0.1028
    Epoch 88/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.0969 - val_mse: 0.0969
    Epoch 89/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0063 - mse: 0.0063 - val_loss: 0.1000 - val_mse: 0.1000
    Epoch 90/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0068 - mse: 0.0068 - val_loss: 0.0990 - val_mse: 0.0990
    Epoch 91/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0053 - mse: 0.0053 - val_loss: 0.0986 - val_mse: 0.0986
    Epoch 92/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0059 - mse: 0.0059 - val_loss: 0.0949 - val_mse: 0.0949
    Epoch 93/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0061 - mse: 0.0061 - val_loss: 0.0978 - val_mse: 0.0978
    Epoch 94/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0061 - mse: 0.0061 - val_loss: 0.0993 - val_mse: 0.0993
    Epoch 95/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0066 - mse: 0.0066 - val_loss: 0.1098 - val_mse: 0.1098
    Epoch 96/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0974 - val_mse: 0.0974
    Epoch 97/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0056 - mse: 0.0056 - val_loss: 0.0990 - val_mse: 0.0990
    Epoch 98/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0054 - mse: 0.0054 - val_loss: 0.1012 - val_mse: 0.1012
    Epoch 99/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0057 - mse: 0.0057 - val_loss: 0.0942 - val_mse: 0.0942
    Epoch 100/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0963 - val_mse: 0.0963
    Epoch 101/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0057 - mse: 0.0057 - val_loss: 0.1074 - val_mse: 0.1074
    Epoch 102/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0057 - mse: 0.0057 - val_loss: 0.0944 - val_mse: 0.0944
    Epoch 103/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0054 - mse: 0.0054 - val_loss: 0.1062 - val_mse: 0.1062
    Epoch 104/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.1016 - val_mse: 0.1016
    Epoch 105/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1006 - val_mse: 0.1006
    Epoch 106/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0056 - mse: 0.0056 - val_loss: 0.1016 - val_mse: 0.1016
    Epoch 107/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.1155 - val_mse: 0.1155
    Epoch 108/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1096 - val_mse: 0.1096
    Epoch 109/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.1049 - val_mse: 0.1049
    Epoch 110/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0896 - val_mse: 0.0896
    Epoch 111/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0051 - mse: 0.0051 - val_loss: 0.1080 - val_mse: 0.1080
    Epoch 112/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0068 - mse: 0.0068 - val_loss: 0.1018 - val_mse: 0.1018
    Epoch 113/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.1024 - val_mse: 0.1024
    Epoch 114/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.1019 - val_mse: 0.1019
    Epoch 115/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0059 - mse: 0.0059 - val_loss: 0.0969 - val_mse: 0.0969
    Epoch 116/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 117/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.0973 - val_mse: 0.0973
    Epoch 118/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.1111 - val_mse: 0.1111
    Epoch 119/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0935 - val_mse: 0.0935
    Epoch 120/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0037 - mse: 0.0037 - val_loss: 0.0993 - val_mse: 0.0993
    Epoch 121/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.1136 - val_mse: 0.1136
    Epoch 122/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.0972 - val_mse: 0.0972
    Epoch 123/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 124/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0977 - val_mse: 0.0977
    Epoch 125/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1026 - val_mse: 0.1026
    Epoch 126/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0049 - mse: 0.0049 - val_loss: 0.1045 - val_mse: 0.1045
    Epoch 127/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.0959 - val_mse: 0.0959
    Epoch 128/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1062 - val_mse: 0.1062
    Epoch 129/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1031 - val_mse: 0.1031
    Epoch 130/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.1118 - val_mse: 0.1118
    Epoch 131/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0051 - mse: 0.0051 - val_loss: 0.1021 - val_mse: 0.1021
    Epoch 132/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0036 - mse: 0.0036 - val_loss: 0.0994 - val_mse: 0.0994
    Epoch 133/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 134/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.0984 - val_mse: 0.0984
    Epoch 135/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0042 - mse: 0.0042 - val_loss: 0.1038 - val_mse: 0.1038
    Epoch 136/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0042 - mse: 0.0042 - val_loss: 0.1013 - val_mse: 0.1013
    Epoch 137/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0044 - mse: 0.0044 - val_loss: 0.1007 - val_mse: 0.1007
    Epoch 138/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0039 - mse: 0.0039 - val_loss: 0.1054 - val_mse: 0.1054
    Epoch 139/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0051 - mse: 0.0051 - val_loss: 0.1001 - val_mse: 0.1001
    Epoch 140/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.1072 - val_mse: 0.1072
    Epoch 141/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.1025 - val_mse: 0.1025
    Epoch 142/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0037 - mse: 0.0037 - val_loss: 0.1024 - val_mse: 0.1024
    Epoch 143/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.0968 - val_mse: 0.0968
    Epoch 144/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1095 - val_mse: 0.1095
    Epoch 145/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.1030 - val_mse: 0.1030
    Epoch 146/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0037 - mse: 0.0037 - val_loss: 0.1051 - val_mse: 0.1051
    Epoch 147/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.0939 - val_mse: 0.0939
    Epoch 148/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.1041 - val_mse: 0.1041
    Epoch 149/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0039 - mse: 0.0039 - val_loss: 0.0979 - val_mse: 0.0979
    Epoch 150/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.1061 - val_mse: 0.1061





    <keras.callbacks.History at 0x17d541e50>



Evaluate the model (`rmsprop_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```


```python
# __SOLUTION__ 
# Evaluate the model on training data
rmsprop_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 4ms/step - loss: 0.0026 - mse: 0.0026





    [0.0025646057911217213, 0.0025646057911217213]



Evaluate the model (`rmsprop_model`) on training data (`X_val` and `y_val_scaled`) 


```python
# Evaluate the model on validate data

```


```python
# __SOLUTION__ 
# Evaluate the model on validate data
rmsprop_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 4ms/step - loss: 0.1061 - mse: 0.1061





    [0.10605713725090027, 0.10605713725090027]



## Adam 

Compile the `adam_model` with: 

- `'Adam'` as the optimizer 
- track `'mse'` as the loss and metric

[Documentation on the Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)


```python
np.random.seed(123)
adam_model = Sequential()
adam_model.add(layers.Dense(100, activation='relu', input_shape=n_features))
adam_model.add(layers.Dense(50, activation='relu'))
adam_model.add(layers.Dense(1, activation='linear'))

# Compile the model
# Code here

# Train the model
adam_model.fit(X_train, 
               y_train_scaled, 
               batch_size=32, 
               epochs=150, 
               validation_data=(X_val, y_val_scaled))
```


```python
# __SOLUTION__ 
np.random.seed(123)
adam_model = Sequential()
adam_model.add(layers.Dense(100, activation='relu', input_shape=(n_features)))
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
    10/33 [========>.....................] - ETA: 0s - loss: 0.7477 - mse: 0.7477

    2023-06-21 13:13:39.327663: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 9ms/step - loss: 0.4966 - mse: 0.4966 - val_loss: 0.1564 - val_mse: 0.1564
    Epoch 2/150
    22/33 [===================>..........] - ETA: 0s - loss: 0.1733 - mse: 0.1733

    2023-06-21 13:13:39.653560: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    33/33 [==============================] - 0s 6ms/step - loss: 0.1752 - mse: 0.1752 - val_loss: 0.1467 - val_mse: 0.1467
    Epoch 3/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.1243 - mse: 0.1243 - val_loss: 0.1213 - val_mse: 0.1213
    Epoch 4/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0927 - mse: 0.0927 - val_loss: 0.1113 - val_mse: 0.1113
    Epoch 5/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0688 - mse: 0.0688 - val_loss: 0.1080 - val_mse: 0.1080
    Epoch 6/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0530 - mse: 0.0530 - val_loss: 0.1093 - val_mse: 0.1093
    Epoch 7/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0430 - mse: 0.0430 - val_loss: 0.1127 - val_mse: 0.1127
    Epoch 8/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0340 - mse: 0.0340 - val_loss: 0.1115 - val_mse: 0.1115
    Epoch 9/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0291 - mse: 0.0291 - val_loss: 0.1136 - val_mse: 0.1136
    Epoch 10/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0232 - mse: 0.0232 - val_loss: 0.1103 - val_mse: 0.1103
    Epoch 11/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0203 - mse: 0.0203 - val_loss: 0.1171 - val_mse: 0.1171
    Epoch 12/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0169 - mse: 0.0169 - val_loss: 0.1097 - val_mse: 0.1097
    Epoch 13/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0149 - mse: 0.0149 - val_loss: 0.1117 - val_mse: 0.1117
    Epoch 14/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0126 - mse: 0.0126 - val_loss: 0.1046 - val_mse: 0.1046
    Epoch 15/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0102 - mse: 0.0102 - val_loss: 0.1311 - val_mse: 0.1311
    Epoch 16/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0116 - mse: 0.0116 - val_loss: 0.1164 - val_mse: 0.1164
    Epoch 17/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0128 - mse: 0.0128 - val_loss: 0.1285 - val_mse: 0.1285
    Epoch 18/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0112 - mse: 0.0112 - val_loss: 0.1066 - val_mse: 0.1066
    Epoch 19/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0083 - mse: 0.0083 - val_loss: 0.1096 - val_mse: 0.1096
    Epoch 20/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0076 - mse: 0.0076 - val_loss: 0.1008 - val_mse: 0.1008
    Epoch 21/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0065 - mse: 0.0065 - val_loss: 0.1072 - val_mse: 0.1072
    Epoch 22/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0048 - mse: 0.0048 - val_loss: 0.1039 - val_mse: 0.1039
    Epoch 23/150
    33/33 [==============================] - 0s 8ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1103 - val_mse: 0.1103
    Epoch 24/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.1118 - val_mse: 0.1118
    Epoch 25/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0051 - mse: 0.0051 - val_loss: 0.1070 - val_mse: 0.1070
    Epoch 26/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 27/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0057 - mse: 0.0057 - val_loss: 0.1071 - val_mse: 0.1071
    Epoch 28/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1205 - val_mse: 0.1205
    Epoch 29/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0060 - mse: 0.0060 - val_loss: 0.1045 - val_mse: 0.1045
    Epoch 30/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1095 - val_mse: 0.1095
    Epoch 31/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0033 - mse: 0.0033 - val_loss: 0.1121 - val_mse: 0.1121
    Epoch 32/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0022 - mse: 0.0022 - val_loss: 0.1078 - val_mse: 0.1078
    Epoch 33/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0019 - mse: 0.0019 - val_loss: 0.1100 - val_mse: 0.1100
    Epoch 34/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0019 - mse: 0.0019 - val_loss: 0.1115 - val_mse: 0.1115
    Epoch 35/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0022 - mse: 0.0022 - val_loss: 0.1104 - val_mse: 0.1104
    Epoch 36/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0029 - mse: 0.0029 - val_loss: 0.1172 - val_mse: 0.1172
    Epoch 37/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.1070 - val_mse: 0.1070
    Epoch 38/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0075 - mse: 0.0075 - val_loss: 0.1106 - val_mse: 0.1106
    Epoch 39/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0062 - mse: 0.0062 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 40/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0062 - mse: 0.0062 - val_loss: 0.1080 - val_mse: 0.1080
    Epoch 41/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.1196 - val_mse: 0.1196
    Epoch 42/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.1211 - val_mse: 0.1211
    Epoch 43/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0069 - mse: 0.0069 - val_loss: 0.1182 - val_mse: 0.1182
    Epoch 44/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0094 - mse: 0.0094 - val_loss: 0.1288 - val_mse: 0.1288
    Epoch 45/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0086 - mse: 0.0086 - val_loss: 0.1004 - val_mse: 0.1004
    Epoch 46/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0065 - mse: 0.0065 - val_loss: 0.1210 - val_mse: 0.1210
    Epoch 47/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.1029 - val_mse: 0.1029
    Epoch 48/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0036 - mse: 0.0036 - val_loss: 0.1063 - val_mse: 0.1063
    Epoch 49/150
    33/33 [==============================] - 0s 7ms/step - loss: 0.0024 - mse: 0.0024 - val_loss: 0.1033 - val_mse: 0.1033
    Epoch 50/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0015 - mse: 0.0015 - val_loss: 0.1074 - val_mse: 0.1074
    Epoch 51/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0011 - mse: 0.0011 - val_loss: 0.1054 - val_mse: 0.1054
    Epoch 52/150
    33/33 [==============================] - 0s 6ms/step - loss: 9.5893e-04 - mse: 9.5893e-04 - val_loss: 0.1071 - val_mse: 0.1071
    Epoch 53/150
    33/33 [==============================] - 0s 6ms/step - loss: 9.0768e-04 - mse: 9.0768e-04 - val_loss: 0.1047 - val_mse: 0.1047
    Epoch 54/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.7799e-04 - mse: 8.7799e-04 - val_loss: 0.1071 - val_mse: 0.1071
    Epoch 55/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.7332e-04 - mse: 8.7332e-04 - val_loss: 0.1040 - val_mse: 0.1040
    Epoch 56/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.8252e-04 - mse: 8.8252e-04 - val_loss: 0.1080 - val_mse: 0.1080
    Epoch 57/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0012 - mse: 0.0012 - val_loss: 0.1019 - val_mse: 0.1019
    Epoch 58/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0011 - mse: 0.0011 - val_loss: 0.1088 - val_mse: 0.1088
    Epoch 59/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0011 - mse: 0.0011 - val_loss: 0.1054 - val_mse: 0.1054
    Epoch 60/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0011 - mse: 0.0011 - val_loss: 0.1083 - val_mse: 0.1083
    Epoch 61/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0015 - mse: 0.0015 - val_loss: 0.1057 - val_mse: 0.1057
    Epoch 62/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0020 - mse: 0.0020 - val_loss: 0.1093 - val_mse: 0.1093
    Epoch 63/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0027 - mse: 0.0027 - val_loss: 0.1046 - val_mse: 0.1046
    Epoch 64/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1147 - val_mse: 0.1147
    Epoch 65/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0043 - mse: 0.0043 - val_loss: 0.1024 - val_mse: 0.1024
    Epoch 66/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0068 - mse: 0.0068 - val_loss: 0.1115 - val_mse: 0.1115
    Epoch 67/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0078 - mse: 0.0078 - val_loss: 0.1077 - val_mse: 0.1077
    Epoch 68/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0058 - mse: 0.0058 - val_loss: 0.1127 - val_mse: 0.1127
    Epoch 69/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0045 - mse: 0.0045 - val_loss: 0.1092 - val_mse: 0.1092
    Epoch 70/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1069 - val_mse: 0.1069
    Epoch 71/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0030 - mse: 0.0030 - val_loss: 0.1036 - val_mse: 0.1036
    Epoch 72/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0030 - mse: 0.0030 - val_loss: 0.1082 - val_mse: 0.1082
    Epoch 73/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.1055 - val_mse: 0.1055
    Epoch 74/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.1092 - val_mse: 0.1092
    Epoch 75/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0033 - mse: 0.0033 - val_loss: 0.1048 - val_mse: 0.1048
    Epoch 76/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.1063 - val_mse: 0.1063
    Epoch 77/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0035 - mse: 0.0035 - val_loss: 0.1054 - val_mse: 0.1054
    Epoch 78/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0037 - mse: 0.0037 - val_loss: 0.1182 - val_mse: 0.1182
    Epoch 79/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0040 - mse: 0.0040 - val_loss: 0.1006 - val_mse: 0.1006
    Epoch 80/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0030 - mse: 0.0030 - val_loss: 0.1079 - val_mse: 0.1079
    Epoch 81/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0026 - mse: 0.0026 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 82/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0027 - mse: 0.0027 - val_loss: 0.1102 - val_mse: 0.1102
    Epoch 83/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0024 - mse: 0.0024 - val_loss: 0.1025 - val_mse: 0.1025
    Epoch 84/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0027 - mse: 0.0027 - val_loss: 0.1094 - val_mse: 0.1094
    Epoch 85/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0026 - mse: 0.0026 - val_loss: 0.1031 - val_mse: 0.1031
    Epoch 86/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0028 - mse: 0.0028 - val_loss: 0.1079 - val_mse: 0.1079
    Epoch 87/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0038 - mse: 0.0038 - val_loss: 0.1078 - val_mse: 0.1078
    Epoch 88/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0037 - mse: 0.0037 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 89/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0039 - mse: 0.0039 - val_loss: 0.1061 - val_mse: 0.1061
    Epoch 90/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1028 - val_mse: 0.1028
    Epoch 91/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.1066 - val_mse: 0.1066
    Epoch 92/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.1015 - val_mse: 0.1015
    Epoch 93/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0223 - mse: 0.0223 - val_loss: 0.1072 - val_mse: 0.1072
    Epoch 94/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0150 - mse: 0.0150 - val_loss: 0.1124 - val_mse: 0.1124
    Epoch 95/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0161 - mse: 0.0161 - val_loss: 0.1040 - val_mse: 0.1040
    Epoch 96/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0160 - mse: 0.0160 - val_loss: 0.1161 - val_mse: 0.1161
    Epoch 97/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0087 - mse: 0.0087 - val_loss: 0.1052 - val_mse: 0.1052
    Epoch 98/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0039 - mse: 0.0039 - val_loss: 0.1079 - val_mse: 0.1079
    Epoch 99/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0030 - mse: 0.0030 - val_loss: 0.1071 - val_mse: 0.1071
    Epoch 100/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0028 - mse: 0.0028 - val_loss: 0.1016 - val_mse: 0.1016
    Epoch 101/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0035 - mse: 0.0035 - val_loss: 0.1093 - val_mse: 0.1093
    Epoch 102/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0050 - mse: 0.0050 - val_loss: 0.1064 - val_mse: 0.1064
    Epoch 103/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0046 - mse: 0.0046 - val_loss: 0.1072 - val_mse: 0.1072
    Epoch 104/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0025 - mse: 0.0025 - val_loss: 0.1032 - val_mse: 0.1032
    Epoch 105/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0014 - mse: 0.0014 - val_loss: 0.1043 - val_mse: 0.1043
    Epoch 106/150
    33/33 [==============================] - 0s 6ms/step - loss: 7.6144e-04 - mse: 7.6144e-04 - val_loss: 0.1045 - val_mse: 0.1045
    Epoch 107/150
    33/33 [==============================] - 0s 6ms/step - loss: 5.6459e-04 - mse: 5.6459e-04 - val_loss: 0.1053 - val_mse: 0.1053
    Epoch 108/150
    33/33 [==============================] - 0s 6ms/step - loss: 6.4414e-04 - mse: 6.4414e-04 - val_loss: 0.1050 - val_mse: 0.1050
    Epoch 109/150
    33/33 [==============================] - 0s 6ms/step - loss: 6.4979e-04 - mse: 6.4979e-04 - val_loss: 0.1038 - val_mse: 0.1038
    Epoch 110/150
    33/33 [==============================] - 0s 7ms/step - loss: 7.1457e-04 - mse: 7.1457e-04 - val_loss: 0.1043 - val_mse: 0.1043
    Epoch 111/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.7667e-04 - mse: 8.7667e-04 - val_loss: 0.1057 - val_mse: 0.1057
    Epoch 112/150
    33/33 [==============================] - 0s 6ms/step - loss: 9.1571e-04 - mse: 9.1571e-04 - val_loss: 0.1043 - val_mse: 0.1043
    Epoch 113/150
    33/33 [==============================] - 0s 6ms/step - loss: 3.7778e-04 - mse: 3.7778e-04 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 114/150
    33/33 [==============================] - 0s 6ms/step - loss: 2.9948e-04 - mse: 2.9948e-04 - val_loss: 0.1042 - val_mse: 0.1042
    Epoch 115/150
    33/33 [==============================] - 0s 6ms/step - loss: 1.8892e-04 - mse: 1.8892e-04 - val_loss: 0.1042 - val_mse: 0.1042
    Epoch 116/150
    33/33 [==============================] - 0s 6ms/step - loss: 1.1807e-04 - mse: 1.1807e-04 - val_loss: 0.1033 - val_mse: 0.1033
    Epoch 117/150
    33/33 [==============================] - 0s 6ms/step - loss: 1.0340e-04 - mse: 1.0340e-04 - val_loss: 0.1042 - val_mse: 0.1042
    Epoch 118/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.6838e-05 - mse: 8.6838e-05 - val_loss: 0.1039 - val_mse: 0.1039
    Epoch 119/150
    33/33 [==============================] - 0s 6ms/step - loss: 7.1290e-05 - mse: 7.1290e-05 - val_loss: 0.1041 - val_mse: 0.1041
    Epoch 120/150
    33/33 [==============================] - 0s 6ms/step - loss: 6.4203e-05 - mse: 6.4203e-05 - val_loss: 0.1039 - val_mse: 0.1039
    Epoch 121/150
    33/33 [==============================] - 0s 6ms/step - loss: 5.4435e-05 - mse: 5.4435e-05 - val_loss: 0.1034 - val_mse: 0.1034
    Epoch 122/150
    33/33 [==============================] - 0s 6ms/step - loss: 5.2463e-05 - mse: 5.2463e-05 - val_loss: 0.1041 - val_mse: 0.1041
    Epoch 123/150
    33/33 [==============================] - 0s 6ms/step - loss: 5.7100e-05 - mse: 5.7100e-05 - val_loss: 0.1039 - val_mse: 0.1039
    Epoch 124/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.8128e-05 - mse: 8.8128e-05 - val_loss: 0.1036 - val_mse: 0.1036
    Epoch 125/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.5462e-05 - mse: 8.5462e-05 - val_loss: 0.1041 - val_mse: 0.1041
    Epoch 126/150
    33/33 [==============================] - 0s 7ms/step - loss: 7.4034e-05 - mse: 7.4034e-05 - val_loss: 0.1037 - val_mse: 0.1037
    Epoch 127/150
    33/33 [==============================] - 0s 6ms/step - loss: 8.9102e-05 - mse: 8.9102e-05 - val_loss: 0.1039 - val_mse: 0.1039
    Epoch 128/150
    33/33 [==============================] - 0s 6ms/step - loss: 1.4844e-04 - mse: 1.4844e-04 - val_loss: 0.1035 - val_mse: 0.1035
    Epoch 129/150
    33/33 [==============================] - 0s 6ms/step - loss: 3.3740e-04 - mse: 3.3740e-04 - val_loss: 0.1047 - val_mse: 0.1047
    Epoch 130/150
    33/33 [==============================] - 0s 6ms/step - loss: 9.0752e-04 - mse: 9.0752e-04 - val_loss: 0.1070 - val_mse: 0.1070
    Epoch 131/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0016 - mse: 0.0016 - val_loss: 0.1034 - val_mse: 0.1034
    Epoch 132/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0024 - mse: 0.0024 - val_loss: 0.1065 - val_mse: 0.1065
    Epoch 133/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0028 - mse: 0.0028 - val_loss: 0.1002 - val_mse: 0.1002
    Epoch 134/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0033 - mse: 0.0033 - val_loss: 0.1093 - val_mse: 0.1093
    Epoch 135/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0056 - mse: 0.0056 - val_loss: 0.1061 - val_mse: 0.1061
    Epoch 136/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0062 - mse: 0.0062 - val_loss: 0.1062 - val_mse: 0.1062
    Epoch 137/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0057 - mse: 0.0057 - val_loss: 0.1001 - val_mse: 0.1001
    Epoch 138/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.1092 - val_mse: 0.1092
    Epoch 139/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0039 - mse: 0.0039 - val_loss: 0.1087 - val_mse: 0.1087
    Epoch 140/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.1015 - val_mse: 0.1015
    Epoch 141/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0056 - mse: 0.0056 - val_loss: 0.1087 - val_mse: 0.1087
    Epoch 142/150
    33/33 [==============================] - 0s 5ms/step - loss: 0.0055 - mse: 0.0055 - val_loss: 0.1005 - val_mse: 0.1005
    Epoch 143/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.1079 - val_mse: 0.1079
    Epoch 144/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0041 - mse: 0.0041 - val_loss: 0.1041 - val_mse: 0.1041
    Epoch 145/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0034 - mse: 0.0034 - val_loss: 0.1030 - val_mse: 0.1030
    Epoch 146/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0026 - mse: 0.0026 - val_loss: 0.0964 - val_mse: 0.0964
    Epoch 147/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0018 - mse: 0.0018 - val_loss: 0.1102 - val_mse: 0.1102
    Epoch 148/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0016 - mse: 0.0016 - val_loss: 0.1014 - val_mse: 0.1014
    Epoch 149/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0017 - mse: 0.0017 - val_loss: 0.1080 - val_mse: 0.1080
    Epoch 150/150
    33/33 [==============================] - 0s 6ms/step - loss: 0.0014 - mse: 0.0014 - val_loss: 0.1016 - val_mse: 0.1016





    <keras.callbacks.History at 0x17d69b550>



Evaluate the model (`adam_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```


```python
# __SOLUTION__ 
# Evaluate the model on training data
adam_model.evaluate(X_train, y_train_scaled)
```

    33/33 [==============================] - 0s 4ms/step - loss: 0.0012 - mse: 0.0012





    [0.0011882828548550606, 0.0011882828548550606]



Evaluate the model (`adam_model`) on training data (`X_val` and `y_val_scaled`) 


```python
# Evaluate the model on validate data

```


```python
# __SOLUTION__ 
# Evaluate the model on validate data
adam_model.evaluate(X_val, y_val_scaled)
```

    9/9 [==============================] - 0s 4ms/step - loss: 0.1016 - mse: 0.1016





    [0.10155995190143585, 0.10155995190143585]



## Select a Final Model

Now, select the model with the best performance based on the training and validation sets. Evaluate this top model using the test set!


```python
# Evaluate the best model on test data

```


```python
# __SOLUTION__ 
# Evaluate the best model on test data
rmsprop_model.evaluate(X_test, y_test_scaled)
```

    5/5 [==============================] - 0s 9ms/step - loss: 0.1717 - mse: 0.1717





    [0.17174449563026428, 0.17174449563026428]



As earlier, this metric is hard to interpret because the output is scaled. 

- Generate predictions on test data (`X_test`) 
- Transform these predictions back to original scale using `ss_y` 
- Now you can calculate the RMSE in the original units with `y_test` and `y_test_pred` 


```python
# Generate predictions on test data
y_test_pred_scaled = None

# Transform the predictions back to original scale
y_test_pred = None

# MSE of test data

```


```python
# __SOLUTION__ 
# Generate predictions on test data
y_test_pred_scaled = rmsprop_model.predict(X_test)

# Transform the predictions back to original scale
y_test_pred = ss_y.inverse_transform(y_test_pred_scaled)

# MSE of test data
np.sqrt(mean_squared_error(y_test, y_test_pred))
```

    5/5 [==============================] - 0s 5ms/step


    2023-06-21 13:14:33.979698: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.





    32566.636953303096



## Summary  

In this lab, you worked to ensure your model converged properly by normalizing both the input and output. Additionally, you also investigated the impact of varying initialization and optimization routines.
