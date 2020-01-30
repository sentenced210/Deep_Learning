#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal


# In[2]:


iris = load_iris()
X = iris.data
y = iris.target
new_y = []
for e in y:
    v = [0, 0, 0]
    v[e] = 1
    new_y.append(v)
    
y = np.array(new_y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[3]:


print(X_train.shape, y_train.shape)


# In[4]:


def model_init():
    model = Sequential()
    model.add(Dense(6, input_dim=4, activation='relu', use_bias=True, kernel_initializer=RandomNormal(mean=0.0, stddev=1.00), bias_initializer=RandomNormal(mean=0.0, stddev=1.00)))
    model.add(Dense(6, activation='relu', use_bias=True, kernel_initializer=RandomNormal(mean=0.0, stddev=1.00), bias_initializer=RandomNormal(mean=0.0, stddev=1.00)))
    model.add(Dense(3, activation='softmax', use_bias=True, kernel_initializer=RandomNormal(mean=0.0, stddev=1.00),bias_initializer=RandomNormal(mean=0.0, stddev=1.00)))
    model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])    
    return model


def evaluate_model_on_train(model):
    return model.evaluate(X_train, y_train, verbose = 0)

def evaluate_model_on_test(model):
    return model.evaluate(X_test, y_test, verbose = 0)

def get_new_model_weights(model_weights):
    W_b = model_weights.copy()
    for i in range(len(W_b)):
        W_b[i] = np.random.normal(W_b[i], 1) 
    return W_b


# In[5]:


#implementation of Simulated Annealing
t = 0
t_global = 0
N = 1

T = 1
sigma = 0.99
T_min = 0.00001

current_model = model_init()
current_model_result = evaluate_model_on_train(current_model)
current_model_weights =  np.array(current_model.get_weights())

global_best_model_weights = current_model.get_weights()
global_best_result = current_model_result

history_results = []
history_results.append([current_model_result, global_best_result])

proportional_function = lambda x, y: math.exp(-(x/y))

#Minimum temperature reach?
while T > T_min:
    new_model_weights = get_new_model_weights(current_model_weights)
    current_model.set_weights(new_model_weights)
    new_model_result = evaluate_model_on_train(current_model)
    
#     print(new_model_weights - current_model_weights)
    
    #Better score?
    if current_model_result[0] > new_model_result[0]:
        # Accept new position
        current_model_result = new_model_result
        current_model_weights = new_model_weights
        
        #Best position?
        if current_model_result[0] < global_best_result[0]:
            global_best_model_weights = current_model_weights
            global_best_result = current_model_result
    
    else:
        u = random.uniform(0, 1)
        alpha = proportional_function(new_model_result[0],T)/proportional_function(current_model_result[0], T)
        
        #Allow move?
        if u <= alpha:
            # Accept new position
            current_model_result = new_model_result
            current_model_weights = new_model_weights
        
            #Best position?
            if current_model_result[0] < global_best_result[0]:
                global_best_model_weights = current_model_weights
                global_best_result = current_model_result
    
        else:
            current_model.set_weights(current_model_weights)
            
    
    
    
#     history_results.append([current_model_result, global_best_result])
    
    #increase cycle count
    t_global += 1
    t += 1
    
    #Reached max cycles?
    if t == N:
        T = T*sigma
        t = 0
    
    
    
    
    print(T, global_best_result)
    


# In[ ]:


for e in history_results:
    print(e)


# In[ ]:




