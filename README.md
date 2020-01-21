
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

```

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

```

Nicely done! After normalizing both the input and output, the model finally converged. 

- Evaluate the model (`normalized_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```

- Evaluate the model (`normalized_model`) on validate data (`X_val` and `y_val_scaled`) 


```python
# Evaluate the model on validate data

```

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

Evaluate the model (`he_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```

Evaluate the model (`he_model`) on validate data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data

```

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

Evaluate the model (`lecun_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```

Evaluate the model (`lecun_model`) on validate data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data

```

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


# Train the model
rmsprop_model.fit(X_train, 
                  y_train_scaled, 
                  batch_size=32, 
                  epochs=150, 
                  validation_data=(X_val, y_val_scaled))
```

Evaluate the model (`rmsprop_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```

Evaluate the model (`rmsprop_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data

```

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


# Train the model
adam_model.fit(X_train, 
               y_train_scaled, 
               batch_size=32, 
               epochs=150, 
               validation_data=(X_val, y_val_scaled))
```

Evaluate the model (`adam_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on training data

```

Evaluate the model (`adam_model`) on training data (`X_train` and `y_train_scaled`) 


```python
# Evaluate the model on validate data

```

## Select a Final Model

Now, select the model with the best performance based on the training and validation sets. Evaluate this top model using the test set!


```python
# Evaluate the best model on test data

```

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

## Summary  

In this lab, you worked to ensure your model converged properly by normalizing both the input and output. Additionally, you also investigated the impact of varying initialization and optimization routines.
