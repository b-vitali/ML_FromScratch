import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Define the euclidean distance between points
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

# Define a function to evaluate the Accuracy = correct / total
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Our K Nearest Neighbours class
class KNN:

    # Create it with a default k value
    def __init__(self, k=3):
        self.k = k
    
    # Define the data to train
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Predict for every x in X and return
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    # Actual prediction
    def _predict(self, x):

        # Evaluate distances to all train data
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Find closest k and keep their lables
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # X are the data and y the labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=954
    )

    #print(X_train.shape)  # 120 4D data    
    #print(y_train)        # 120 labels : 0, 1 or 2 

    #plt.figure()
    ## This plots the first two dimentions, c assigns color from the value y, k for black circle, s is size
    #plt.scatter(X[:,0], X[:,1], c=y, edgecolor = 'k', s = 30)   
    #plt.show()

    k = 5
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print(predictions)
    print("KNN classification accuracy", accuracy(y_test, predictions))