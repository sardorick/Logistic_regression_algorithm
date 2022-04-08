import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import e
import math

df = pd.read_csv('dataset/diabetes.csv')
Y = df['Outcome'].values
X = df.drop(['Outcome'], axis=1).values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

def standardize(x_test):
    """Standardizing the data (features) to use them properly"""
    standard_list = []
    for i in range(x_test.shape[1]):
        scaled  = (x_test[:, i] - np.mean(x_test[:, i]))/np.std(x_test[:, i])
        standard_list.append(scaled)
    return standard_list

class Logistic_regression:
    # e = math.exp(e) # Euler's number
    def sigmoid(self, a, b, x):
        """Sigmoid mathematical function"""
        y = a*x + b
        sig = 1/(1+(np.exp(-y))) 
        return sig 

    def fit(self, x, y, alpha):
    # initialise random value for a and b
    # a, b = np.random.rand(2)
        a, b = 1/20, -2
        for i in range(x.shape[0]):

            pred = self.sigmoid(a, b, x[i])
        if y[i] == 0:
            error = 0 - pred
        else:
            error = 1 - pred

        if np.abs(error) < 0.0001:
            return a, b

        else:   
            a -= alpha * error * x[i] 
            b -= alpha * error

        return a, b
    def predict(self, x, a, b):
        """ Create a predict function to check the outcome by comparing it to the sigmoid
            """
        predictions = []
        for i in range(x.shape[0]):
            pred = self.sigmoid(a, b, x[i]) # was getting an error, second index needed to return a single value
        if pred > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

        return np.array(predictions)
    
    def score(self, y, y_hat):
        pred = 0
        true_values = 0
        for i in range(len(y)):
            if pred[i] == 1 and true_values[i] == 1:
                true_values +1
            elif pred[i] == 1 and true_values[i] == 0:
                pred += 1
            else:
                pred += 0
        accuracy = pred(y)/len(true_values)
        return accuracy

