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

In this lab, you'll be working with the housing prices data you saw in an earlier section. However, we did a lot of preprocessing for you so you can focus on normalizing numeric features and building neural network models! The following preprocessing steps were taken (all the code can be found in the `data_preprocessing.ipynb` notebook in this repository): 

- The data was split into the training, validate, and test sets 
- All the missing values in numeric columns were replaced by the median of those columns 
- All the missing values in catetgorical columns were replaced with the word 'missing' 
- All the categorical columns were one-hot encoded 

Run the following cells to import the train, validate, and test sets:  

As a refresher, preview the training data: 

## Build a Baseline Model

Building a naive baseline model to compare performance against is a helpful reference point. From there, you can then observe the impact of various tunning procedures which will iteratively improve your model. So, let's do just that! 

In the cell below: 

- Add an input layer with `n_features` units 
- Add two hidden layers, one with 100 and the other with 50 units (make sure you use the `'relu'` activation function) 
- Add an output layer with 1 unit and `'linear'` activation 
- Compile and fit the model 

Here, we're calling .shape on our training data so that we can use the result as n_features, so we know how big to make our input layer.

> _**Notice this extremely problematic behavior: all the values for training and validation loss are "nan". This indicates that the algorithm did not converge. The first solution to this is to normalize the input. From there, if convergence is not achieved, normalizing the output may also be required.**_ 

## Normalize the Input Data 

It's now time to normalize the input data. In the cell below: 

- Assign the column names of all numeric columns to `numeric_columns` 
- Instantiate a `StandardScaler` 
- Fit and transform `X_train_numeric`. Make sure you convert the result into a DataFrame (use `numeric_columns` as the column names) 
- Transform validate and test sets (`X_val_numeric` and `X_test_numeric`), and convert these results into DataFrames as well 
- Use the provided to combine the scaled numerical and categorical features 

Now run the following cell to compile a neural network model (with the same architecture as before): 

In the cell below: 
- Train the `normalized_input_model` on normalized input (`X_train`) and output (`y_train`) 
- Set a batch size of 32 and train for 150 epochs 
- Specify the `validation_data` argument as `(X_val, y_val)` 

> _**Note that you still haven't achieved convergence! From here, it's time to normalize the output data.**_

## Normalizing the output

Again, use `StandardScaler()` to: 

- Fit and transform `y_train` 
- Transform `y_val` and `y_test` 

In the cell below: 
- Train the `normalized_model` on normalized input (`X_train`) and output (`y_train_scaled`) 
- Set a batch size of 32 and train for 150 epochs 
- Specify the `validation_data` as `(X_val, y_val_scaled)` 

Nicely done! After normalizing both the input and output, the model finally converged. 

- Evaluate the model (`normalized_model`) on training data (`X_train` and `y_train_scaled`) 

- Evaluate the model (`normalized_model`) on validate data (`X_val` and `y_val_scaled`) 

Since the output is normalized, the metric above is not interpretable. To remedy this: 

- Generate predictions on validate data (`X_val`) 
- Transform these predictions back to original scale using `ss_y` 
- Now you can calculate the RMSE in the original units with `y_val` and `y_val_pred` 

Great! Now that you have a converged model, you can also experiment with alternative optimizers and initialization strategies to see if you can find a better global minimum. (After all, the current models may have converged to a local minimum.) 

## Using Weight Initializers

In this section you will to use alternative initialization and optimization strategies. At the end, you'll then be asked to select the model which you believe performs the best.  

##  He Initialization

In the cell below, sepcify the following in the first hidden layer:  
  - 100 units 
  - `'relu'` activation 
  - `input_shape` 
  - `kernel_initializer='he_normal'`  

Evaluate the model (`he_model`) on training data (`X_train` and `y_train_scaled`) 

Evaluate the model (`he_model`) on validate data (`X_val` and `y_val_scaled`)

## Lecun Initialization 

In the cell below, sepcify the following in the first hidden layer:  
  - 100 units 
  - `'relu'` activation 
  - `input_shape` 
  - `kernel_initializer='lecun_normal'` 

Evaluate the model (`lecun_model`) on training data (`X_train` and `y_train_scaled`) 

Evaluate the model (`lecun_model`) on validate data (`X_train` and `y_train_scaled`) 

Not much of a difference, but a useful note to consider when tuning your network. Next, let's investigate the impact of various optimization algorithms.

## RMSprop 

Compile the `rmsprop_model` with: 

- `'rmsprop'` as the optimizer 
- track `'mse'` as the loss and metric  

Evaluate the model (`rmsprop_model`) on training data (`X_train` and `y_train_scaled`) 

Evaluate the model (`rmsprop_model`) on training data (`X_val` and `y_val_scaled`) 

## Adam 

Compile the `adam_model` with: 

- `'Adam'` as the optimizer 
- track `'mse'` as the loss and metric  

Evaluate the model (`adam_model`) on training data (`X_train` and `y_train_scaled`) 

Evaluate the model (`adam_model`) on training data (`X_val` and `y_val_scaled`) 

## Select a Final Model

Now, select the model with the best performance based on the training and validation sets. Evaluate this top model using the test set!

As earlier, this metric is hard to interpret because the output is scaled. 

- Generate predictions on test data (`X_test`) 
- Transform these predictions back to original scale using `ss_y` 
- Now you can calculate the RMSE in the original units with `y_test` and `y_test_pred` 

## Summary  

In this lab, you worked to ensure your model converged properly by normalizing both the input and output. Additionally, you also investigated the impact of varying initialization and optimization routines.
