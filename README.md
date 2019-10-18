
# Tuning Neural Networks with Normalization - Lab

## Introduction

For this lab on initialization and optimization, you'll build a neural network to perform a regression task.

It is worth noting that getting regression to work with neural networks can be difficult because the output is unbounded ($\hat y$ can technically range from $-\infty$ to $+\infty$, and the models are especially prone to exploding gradients. This issue makes a regression exercise the perfect learning case for tinkering with normalization and optimization strategies to ensure proper convergence!

## Objectives
You will be able to:
* Build a neural network using Keras
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



```python
# __SOLUTION__ 
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

The data we'll be working with is data related to Facebook posts published during the year of 2014 on the Facebook page of a renowned cosmetics brand.  It includes 7 features known prior to post publication, and 12 features for evaluating the post impact. What we want to do is make a predictor for the number of "likes" for a post, taking into account the 7 features prior to posting.

First, let's import the data set, `dataset_Facebook.csv`, and delete any rows with missing data. Afterwards, briefly preview the data.


```python
#Your code here; load the dataset and drop rows with missing values. Then preview the data.
```


```python
# __SOLUTION__ 
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



## Defining the Problem

Define X and Y and perform a train-validation-test split.

X will be:
* Page total likes
* Post Month
* Post Weekday
* Post Hour
* Paid
along with dummy variables for:
* Type
* Category

Y will be the `like` column.


```python
#Your code here; define the problem.
```


```python
# __SOLUTION__ 
X0 = data["Page total likes"]
X1 = data["Type"]
X2 = data["Category"]
X3 = data["Post Month"]
X4 = data["Post Weekday"]
X5 = data["Post Hour"]
X6 = data["Paid"]

## Even for a baseline model some preprocessing may be required (all inputs must be numerical features)
dummy_X1= pd.get_dummies(X1, drop_first=True)
dummy_X2= pd.get_dummies(X2, drop_first=True)

X = pd.concat([X0, dummy_X1, dummy_X2, X3, X4, X5, X6], axis=1)

Y = data["like"]


data_clean = pd.concat([X, Y], axis=1)
np.random.seed(123)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)  
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=123)  

# train, validation = train_test_split(data_clean, test_size=0.2)

# X_val = validation.iloc[:,0:10]
# Y_val = validation.iloc[:,10]
# X_train = train.iloc[:,0:10]
# Y_train = train.iloc[:,10]
```

## Building a Baseline Model

Next, build a naive baseline model to compare performance against is a helpful reference point. From there, you can then observe the impact of various tunning procedures which will iteratively improve your model.


```python
#Simply run this code block, later you'll modify this model to tune the performance
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```


```python
# __SOLUTION__ 
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```

### Evaluating the Baseline

Evaluate the baseline model for the training and validation sets.


```python
#Your code here; evaluate the model with MSE
```


```python
#Your code here; inspect the loss function through the history object
```


```python
# __SOLUTION__ 
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)  

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)

print("MSE_train:", MSE_train)
print("MSE_val:", MSE_val)
```

    MSE_train: nan
    MSE_val: nan



```python
# __SOLUTION__ 
hist.history['loss'][:10]
```




    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]



> Notice this extremely problematic behavior: all the values for training and validation loss are "nan". This indicates that the algorithm did not converge. The first solution to this is to normalize the input. From there, if convergence is not achieved, normalizing the output may also be required.

## Normalize the Input Data

Normalize the input features by subtracting each feature mean and dividing by the standard deviation in order to transform each into a standard normal distribution. Then recreate the train-validate-test sets with the transformed input data.


```python
## standardize/categorize
```


```python
# __SOLUTION__ 
## standardize/categorize
X0= (X0-np.mean(X0))/(np.std(X0))

X3= (X3-np.mean(X3))/(np.std(X3))
X4= (X4-np.mean(X4))/(np.std(X4))
X5= (X5-np.mean(X5))/(np.std(X5))
X6= (X6-np.mean(X6))/(np.std(X6))

X = pd.concat([X0, dummy_X1, dummy_X2, X3, X4, X5, X6], axis=1)

#Note: you get the same result for standardization if you use StandardScaler from sklearn.preprocessing
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X0 = sc.fit_transform(X0)
```


```python
# __SOLUTION__ 
#Code provided; defining training and validation sets
data_clean = pd.concat([X, Y], axis=1)
np.random.seed(123)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)  
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=123)   
# train, validation = train_test_split(data_clean, test_size=0.2)

# X_val = validation.iloc[:,0:12]
# Y_val = validation.iloc[:,12]
# X_train = train.iloc[:,0:12]
# Y_train = train.iloc[:,12]
```

## Refit the Model and Reevaluate

Great! Now refit the model and once again assess it's performance on the training and validation sets.


```python
#Your code here; refit a model as shown above
```


```python
#Rexamine the loss function
```


```python
# __SOLUTION__ 
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```


```python
# __SOLUTION__ 
hist.history['loss'][:10]
```




    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]



> Note that you still haven't achieved convergence! From here, it's time to normalize the output data.

## Normalizing the output

Normalize Y as you did X by subtracting the mean and dividing by the standard deviation. Then, resplit the data into training and validation sets as we demonstrated above, and retrain a new model using your normalized X and Y data.


```python
#Your code here: redefine Y after normalizing the data.
```


```python
#Your code here; create training and validation sets as before. Use random seed 123.
```


```python
#Your code here; rebuild a simple model using a relu layer followed by a linear layer. (See our code snippet above!)
```


```python
# __SOLUTION__ 
#Your code here: redefine Y after normalizing the data.
Y = (data["like"]-np.mean(data["like"]))/(np.std(data["like"]))
```


```python
# __SOLUTION__ 
#Your code here; create training and validation sets as before. Use random seed 123.
data_clean = pd.concat([X, Y], axis=1)
np.random.seed(123)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)  
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=123)  
# train, validation = train_test_split(data_clean, test_size=0.2)

# X_val = validation.iloc[:,0:12]
# Y_val = validation.iloc[:,12]
# X_train = train.iloc[:,0:12]
# Y_train = train.iloc[:,12]
```


```python
# __SOLUTION__ 
#Your code here; rebuild a simple model using a relu layer followed by a linear layer. (See our code snippet above!)
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```

Again, reevaluate the updated model.


```python
#Your code here; MSE
```


```python
#Your code here; loss function
```


```python
# __SOLUTION__ 
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)  

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)

print("MSE_train:", MSE_train)
print("MSE_val:", MSE_val)
```

    MSE_train: 1.0340719824485043
    MSE_val: 0.9638583875606664



```python
# __SOLUTION__ 
hist.history['loss'][:10]
```




    [1.439053112536334,
     1.3336541879545436,
     1.2812423051408168,
     1.246897012330173,
     1.2241152392679386,
     1.2034877556261052,
     1.1872830351584414,
     1.1745527371596753,
     1.1645603426051943,
     1.1568267743908958]



Great! Now that you have a converged model, you can also experiment with alternative optimizers and initialization strategies to see if you can find a better global minimum. (After all, the current models may have converged to a local minimum.)

## Using Weight Initializers

Below, take a look at the code provided to see how to modify the neural network to use alternative initialization and optimization strategies. At the end, you'll then be asked to select the model which you believe is the strongest.

##  He Initialization


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, kernel_initializer= "he_normal",
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

    1.0392949820359312
    0.8658544030836142



```python
# __SOLUTION__ 
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, kernel_initializer= "he_normal",
                activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val),verbose=0)
```


```python
# __SOLUTION__ 
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
# __SOLUTION__ 
print(MSE_train)
print(MSE_val)
```

    1.0392949820359312
    0.8658544030836142


## Lecun Initialization


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, 
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

    1.0307351124941144
    0.9292005788570431



```python
# __SOLUTION__ 
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, 
                kernel_initializer= "lecun_normal", activation='tanh'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```


```python
# __SOLUTION__ 
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
# __SOLUTION__ 
print(MSE_train)
print(MSE_val)
```

    1.0307351124941144
    0.9292005788570431


Not much of a difference, but a useful note to consider when tuning your network. Next, let's investigate the impact of various optimization algorithms.

## RMSprop


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
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

    1.020200641136699
    0.9421919606123765



```python
# __SOLUTION__ 
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "rmsprop" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
# __SOLUTION__ 
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
# __SOLUTION__ 
print(MSE_train)
print(MSE_val)
```

    1.020200641136699
    0.9421919606123765


## Adam


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
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

    1.0219766410555322
    0.9477664629838952



```python
# __SOLUTION__ 
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "Adam" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
# __SOLUTION__ 
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
# __SOLUTION__ 
print(MSE_train)
print(MSE_val)
```

    1.0219766410555322
    0.9477664629838952


## Learning Rate Decay with Momentum



```python
np.random.seed(123)
sgd = optimizers.SGD(lr=0.03, decay=0.0001, momentum=0.9)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
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

    0.8667952792265361
    1.1040802536956849



```python
# __SOLUTION__ 
np.random.seed(123)
sgd = optimizers.SGD(lr=0.03, decay=0.0001, momentum=0.9)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= sgd ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
# __SOLUTION__ 
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
# __SOLUTION__ 
print(MSE_train)
print(MSE_val)
```

    0.8667952792265361
    1.1040802536956849


## Selecting a Final Model

Now, select the model with the best performance based on the training and validation sets. Evaluate this top model using the test set!


```python
#Your code here
```


```python
# __SOLUTION__ 
#Your code here
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, kernel_initializer= "he_normal",
                activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val),verbose=0)

pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)

pred_test = model.predict(X_test).reshape(-1)
MSE_test = np.mean((pred_test-Y_test)**2)

print(MSE_train)
print(MSE_val)
print(MSE_test)
```

    1.0391711439058289
    0.8659758456727069
    0.17878674871641423


## Summary  

In this lab, you worked to ensure your model converged properly. Additionally, you also investigated the impact of varying initialization and optimization routines.
