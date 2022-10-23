from random import random
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr=0.01, num_iteration = 25, random_state = 1):
        self.lr = lr
        self.num_iteration = num_iteration
        self.random_state = random_state


    def fit (self, X, y):
        random_gen = np.random.RandomState(self.random_state)
        self.weight = random_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = np.array([0.])
        self.losses = []

        for i in range(self.num_iteration):
            output = self.net(X)
            errors = (y - output)
            self.weight += self.lr * 2.0 * X.T.dot(errors) / X.shape[0]
            self.bias  += self.lr * 2.0 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses.append(loss) 
        
        return self 
    
    def net(self, X):
        return np.dot(X, self.weight) + self.bias 

    def predict(self, X):
        return self.net(X)

    def plot_error(self):
        plt.plot(range(1, self.num_iteration + 1), self.losses)
        plt.ylabel('MSE')
        plt.xlabel('Epcoch')
        plt.show()

    def see_datafit(self, X, y):
        plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
        plt.plot(X, self.predict(X), color='black', lw=2)