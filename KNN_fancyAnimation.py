import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

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
        
        # Find closest k and keep their labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

if __name__ == "__main__":

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # X are the data and y the labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=999
    )

    k = 5
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Print the results
    print("KNN classification accuracy", accuracy(y_test, predictions))

    # Plotting the training set
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use tab10 color palette
    colors = plt.cm.tab10.colors
    
    # Plot each class in different colors
    for i in range(len(np.unique(y_train))):
        ax.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1], color = colors[i],label=f'Class {i}', s=30)

    # Plot the test set in gray
    test_points = ax.scatter(X_test[:, 0], X_test[:, 1], color='gray', label='Test Set', s=60, alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    # Animation function
    def update(frame):
        pred = predictions[frame]
        ax.scatter(X_test[frame, 0], X_test[frame, 1], color=colors[pred], marker='x', s=200)
        ax.set_title(f'KNN (k={k}): {frame+1} predicted as {pred}')
    
    ani = animation.FuncAnimation(fig, update, frames=len(predictions), interval=200, repeat=False)
    plt.show()

    # Re-make the plot to save the GIF
    ax.clear()
    # Plot each class in different colors
    for i in range(len(np.unique(y_train))):
        ax.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1], color = colors[i],label=f'Class {i}', s=30)    
    # Plot the test set in gray
    test_points = ax.scatter(X_test[:, 0], X_test[:, 1], color='gray', label='Test Set', s=60, alpha=0.5)

    # Set up writer
    Writer = animation.writers['pillow']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    # Save the animation
    ani.save('knn_classification_animation.gif', writer=writer)