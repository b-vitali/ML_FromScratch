import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LinearRegression:
    # Create it with 
    def __init__(self, lr = 0.001, n_iter = 100):
        # Learning rate is how fast we move
        self.lr         = lr
        self.n_iter     = n_iter
        self.weights    = None
        self.bias       = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights    = np.zeros(n_features)
        self.bias       = 0
        
        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = 1/n_samples * np.dot(X.T, (y_pred-y))
            db = 1/n_samples * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

if __name__ == "__main__":
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    LR = LinearRegression(lr = 0.1)
    LR.fit(X_train, y_train)
    predictions = LR.predict(X_test)

    def mse(y_test, predictions):
        return np.mean((y_test-predictions)**2)

    mse = mse(y_test, predictions)
    print(mse)

    colors = plt.cm.tab10.colors

    y_pred_line = LR.predict(X)

    # Plot the training and test data points
    fig = plt.figure(figsize=(8,6))
    m1 = plt.scatter(X_train, y_train, color=colors[0], s=50, label='Train')
    m2 = plt.scatter(X_test, y_test, color=colors[1], s=50, label='Test')

    # Plot the regression line
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')

    # Add title and legend
    plt.title('Example of Linear Regression')
    plt.legend()
    plt.show()