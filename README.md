
# Normalization and Tuning Neural Networks - Lab

## Introduction

For this lab on initialization and optimization, let's look at a slightly different type of neural network. This time, we will not perform a classification task as we've done before (Santa vs not santa, bank complaint types), but we'll look at a linear regression problem.

We can just as well use deep learning networks for linear regression as for a classification problem. Do note that getting regression to work with neural networks is a hard problem because the output is unbounded ($\hat y$ can technically range from $-\infty$ to $+\infty$, and the models are especially prone to exploding gradients. This issue makes a regression exercise the perfect learning case!

## Objectives:
You will be able to:
* Build a nueral network using keras
* Normalize your data to assist algorithm convergence
* Implement and observe the impact of various initialization techniques


```python
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import initializers
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from keras import optimizers
from sklearn.model_selection import train_test_split
```

    /Users/matthew.mitchell/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


## Loading the data

The data we'll be working with is data related to facebook posts published during the year of 2014 on the Facebook's page of a renowned cosmetics brand.  It includes 7 features known prior to post publication, and 12 features for evaluating the post impact. What we want to do is make a predictor for the number of "likes" for a post, taking into account the 7 features prior to posting.

First, let's import the data set and delete any rows with missing data. Afterwards, briefly preview the data.


```python
#Your code here; load the dataset and drop rows with missing values. Then preview the data.
data = pd.read_csv("dataset_Facebook.csv", sep = ";", header=0)
data = data.dropna()
print(np.shape(data))
data.head()
```

    (495, 19)





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
      <th>Page total likes</th>
      <th>Type</th>
      <th>Category</th>
      <th>Post Month</th>
      <th>Post Weekday</th>
      <th>Post Hour</th>
      <th>Paid</th>
      <th>Lifetime Post Total Reach</th>
      <th>Lifetime Post Total Impressions</th>
      <th>Lifetime Engaged Users</th>
      <th>Lifetime Post Consumers</th>
      <th>Lifetime Post Consumptions</th>
      <th>Lifetime Post Impressions by people who have liked your Page</th>
      <th>Lifetime Post reach by people who like your Page</th>
      <th>Lifetime People who have liked your Page and engaged with your post</th>
      <th>comment</th>
      <th>like</th>
      <th>share</th>
      <th>Total Interactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>139441</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>2752</td>
      <td>5091</td>
      <td>178</td>
      <td>109</td>
      <td>159</td>
      <td>3078</td>
      <td>1640</td>
      <td>119</td>
      <td>4</td>
      <td>79.0</td>
      <td>17.0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139441</td>
      <td>Status</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>10</td>
      <td>0.0</td>
      <td>10460</td>
      <td>19057</td>
      <td>1457</td>
      <td>1361</td>
      <td>1674</td>
      <td>11710</td>
      <td>6112</td>
      <td>1108</td>
      <td>5</td>
      <td>130.0</td>
      <td>29.0</td>
      <td>164</td>
    </tr>
    <tr>
      <th>2</th>
      <td>139441</td>
      <td>Photo</td>
      <td>3</td>
      <td>12</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>2413</td>
      <td>4373</td>
      <td>177</td>
      <td>113</td>
      <td>154</td>
      <td>2812</td>
      <td>1503</td>
      <td>132</td>
      <td>0</td>
      <td>66.0</td>
      <td>14.0</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139441</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>10</td>
      <td>1.0</td>
      <td>50128</td>
      <td>87991</td>
      <td>2211</td>
      <td>790</td>
      <td>1119</td>
      <td>61027</td>
      <td>32048</td>
      <td>1386</td>
      <td>58</td>
      <td>1572.0</td>
      <td>147.0</td>
      <td>1777</td>
    </tr>
    <tr>
      <th>4</th>
      <td>139441</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>7244</td>
      <td>13594</td>
      <td>671</td>
      <td>410</td>
      <td>580</td>
      <td>6228</td>
      <td>3200</td>
      <td>396</td>
      <td>19</td>
      <td>325.0</td>
      <td>49.0</td>
      <td>393</td>
    </tr>
  </tbody>
</table>
</div>



## Initialization

## Normalize the Input Data

Let's look at our input data. We'll use the 7 first columns as our predictors. We'll do the following two things:
- Normalize the continuous variables --> you can do this using `np.mean()` and `np.std()`
- Make dummy variables of the categorical variables (you can do this by using `pd.get_dummies`)

We only count "Category" and "Type" as categorical variables. Note that you can argue that "Post month", "Post Weekday" and "Post Hour" can also be considered categories, but we'll just treat them as being continuous for now.

You'll then use these to define X and Y. 

To summarize, X will be:
* Page total likes
* Post Month
* Post Weekday
* Post Hour
* Paid
along with dummy variables for:
* Type
* Category


Be sure to normalize your features by subtracting the mean and dividing by the standard deviation.  

Finally, y will simply be the "like" column.


```python
#Your code here; define X and y.
X0 = data["Page total likes"]
X1 = data["Type"]
X2 = data["Category"]
X3 = data["Post Month"]
X4 = data["Post Weekday"]
X5 = data["Post Hour"]
X6 = data["Paid"]

## standardize/categorize
X0= (X0-np.mean(X0))/(np.std(X0))
dummy_X1= pd.get_dummies(X1)
dummy_X2= pd.get_dummies(X2)
X3= (X3-np.mean(X3))/(np.std(X3))
X4= (X4-np.mean(X4))/(np.std(X4))
X5= (X5-np.mean(X5))/(np.std(X5))

X = pd.concat([X0, dummy_X1, dummy_X2, X3, X4, X5, X6], axis=1)

Y = data["like"]

#Note: you get the same result for standardization if you use StandardScaler from sklearn.preprocessing
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X0 = sc.fit_transform(X0)
```

Our data is fairly small. Let's just split the data up in a training set and a validation set!  The next three code blocks are all provided for you; have a quick review but not need to make edits!


```python
#Code provided; defining training and validation sets
data_clean = pd.concat([X, Y], axis=1)
np.random.seed(123)
train, validation = train_test_split(data_clean, test_size=0.2)

X_val = validation.iloc[:,0:12]
Y_val = validation.iloc[:,12]
X_train = train.iloc[:,0:12]
Y_train = train.iloc[:,12]
```


```python
#Code provided; building an initial model
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=12, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```


```python
#Code provided; previewing the loss through successive epochs
hist.history['loss'][:10]
```




    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]



Did you see what happend? all the values for training and validation loss are "nan". There could be several reasons for that, but as we already mentioned there is likely a vanishing or exploding gradient problem. recall that we normalized out inputs. But how about the outputs? Let's have a look.


```python
Y_train.head()
```




    208     54.0
    290     23.0
    286     15.0
    0       79.0
    401    329.0
    Name: like, dtype: float64



Yes, indeed. We didn't normalize them and we should, as they take pretty high values. Let
s rerun the model but make sure that the output is normalized as well!

## Normalizing the output

Normalize Y as you did X by subtracting the mean and dividing by the standard deviation. Then, resplit the data into training and validation sets as we demonstrated above, and retrain a new model using your normalized X and Y data.


```python
#Your code here: redefine Y after normalizing the data.
Y = (data["like"]-np.mean(data["like"]))/(np.std(data["like"]))
```


```python
#Your code here; create training and validation sets as before. Use random seed 123.
data_clean = pd.concat([X, Y], axis=1)
np.random.seed(123)
train, validation = train_test_split(data_clean, test_size=0.2)

X_val = validation.iloc[:,0:12]
Y_val = validation.iloc[:,12]
X_train = train.iloc[:,0:12]
Y_train = train.iloc[:,12]
```


```python
#Your code here; rebuild a simple model using a relu layer followed by a linear layer. (See our code snippet above!)
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=12, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```

Finally, let's recheck our loss function. Not only should it be populated with numerical data as opposed to null values, but we also should expect to see the loss function decreasing with successive epochs, demonstrating optimization!


```python
hist.history['loss'][:10]
```




    [1.2408857164960918,
     1.175498196874002,
     1.1352486694701995,
     1.1083890091289172,
     1.0902567186740915,
     1.0721662248475383,
     1.0611894782444444,
     1.0505248789835457,
     1.042567062408033,
     1.0360946251888468]



Great! We have a converged model. With that, let's investigate how well the model performed with our good old friend, mean squarred error.


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)  

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)

print("MSE_train:", MSE_train)
print("MSE_val:", MSE_val)
```

    MSE_train: 0.9279475664337523
    MSE_val: 0.9317562611051917


## Using Weight Initializers

##  He Initialization

Let's try and use a weight initializer. In the lecture, we've seen the He normalizer, which initializes the weight vector to have an average 0 and a variance of 2/n, with $n$ the number of features feeding into a layer.


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=12, kernel_initializer= "he_normal",
                activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val),verbose=0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    0.9266351379758461
    0.9474339752163196


The initializer does not really help us to decrease the MSE. We know that initializers can be particularly helpful in deeper networks, and our network isn't very deep. What if we use the `Lecun` initializer with a `tanh` activation?

## Lecun Initialization


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=12, 
                kernel_initializer= "lecun_normal", activation='tanh'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    0.9274710945931817
    0.9463006239264359


Not much of a difference, but a useful note to consider when tuning your network. Next, let's investigate the impace of various optimization algorithms.

## RMSprop


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=12, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "rmsprop" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    0.914450642739685
    0.9437157484784983


## Adam


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=12, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "Adam" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    0.9113685285012638
    0.9444777470972421


## Learning Rate Decay with Momentum



```python
np.random.seed(123)
sgd = optimizers.SGD(lr=0.03, decay=0.0001, momentum=0.9)
model = Sequential()
model.add(layers.Dense(8, input_dim=12, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= sgd ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    0.8188327426055082
    0.9218409795298302


## Additional Resources
* https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Consumer_complaints.ipynb  

* https://catalog.data.gov/dataset/consumer-complaint-database  

* https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/  

* https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/  

* https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/  

* https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network  

* https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/


## Summary  

In this lab, we began to practice some of the concepts regarding normalization and optimization for neural networks. In the final lab for this section, you'll independently practice these concepts on your own in order to tune a model to predict individuals payments to loans.
