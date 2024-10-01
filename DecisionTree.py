import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # Just a check to see if it is a *leaf*
    def is_leaf_node(self):
        return self.value is not None

# Let's define the DecisionTree
class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):

        # Stopping criteria and basic info
        self.min_sample_split=min_sample_split;
        self.max_depth=max_depth;
        self.n_features=n_features
        self.root=None

    # Define the data to train
    def fit(self, X,y):
        # Check if number of features is too high
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)

        # First we create the tree and keep track of the *root*    
        self.root = self._grow_tree(X,y)

    # A recursive function
    def _grow_tree(self, X,y ,depth = 0):

        # Unpack the info
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if (depth > self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):

            # If we want to stop, what is the node filled of?
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        # Find the best split 
        # NB:There is some randomness here! 
        feat_idx = np.random.choice(n_feats, self.n_features, replace = False)
        best_feature, best_threshold = self._best_split(X,y,feat_idx)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:,best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    # Split on which feature and at which threshold?
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        # For each feat ...
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)

            # ... Consider every possible threshold
            for thr in thresholds:
                # Calculate gain fro this split
                gain = self._information_gain(y, X_column, thr)

                # If it is the best until now, keep it
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        # Once we are done with the option return the best found
        return split_idx, split_threshold
    
    # Function to evaluate the gain fro a specific split
    # IG = E(parent) - average * E(children)
    def _information_gain(self, y, X_column, thr):
        
        # Get parent entropy
        parent_entropy = self._entropy(y)

        # Find children
        left_idxs, right_idxs = self._split(X_column, thr)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Get weighted children entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        entropy_l, entropy_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        children_entropy = (n_l/n) * entropy_l + (n_r/n) * entropy_r

        # Evaluate and return the IG
        return parent_entropy - children_entropy

    # Helper function to find the entropy for _information_gain()
    # See definition of entropy on the README
    def _entropy(self, y):
       
        # EXAMPLE : np.bincount(np.array([0, 1, 1, 3, 2, 1, 7])) -> array([1, 3, 1, 1, 0, 0, 0, 1])
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])


    # What is the most common label
    def _most_common_label(self, y):
        #  EXAMPLE : Counter('abracadabra').most_common(3) -> [('a', 5), ('b', 2), ('r', 2)]
        most_common = Counter(y).most_common(1)
        return most_common[0][0]

    # Helper function to split given a threshold
    def _split(self, X_column, thr):
        left_idxs = np.argwhere(X_column<=thr).flatten()
        right_idxs = np.argwhere(X_column>thr).flatten()

        return left_idxs, right_idxs
    
    # Predict for every x in X and return
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])
        
    def _traverse(self, x, node):
        if node.is_leaf_node():
            return node.value
    
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

# Define a function to evaluate the Accuracy = correct / total
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def print_tree(node, depth=0):
    if node.is_leaf_node():
        print(f"{'|  ' * depth}Leaf: {node.value}")
    else:
        print(f"{'|  ' * depth}Feature {node.feature} <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

def plot_tree(node, depth=0, x=0.5, y=1.0, dx=0.1, ax=None):
    if node.is_leaf_node():
        # Draw a leaf node
        ax.text(x, y, f"Leaf: {node.value}", ha='center', va='center', fontsize=10, bbox=dict(facecolor='lightgreen', edgecolor='black'))
    else:
        # Draw a decision node
        ax.text(x, y, f"X[{node.feature}] <= {node.threshold:.2f}", ha='center', va='center', fontsize=10, bbox=dict(facecolor='lightblue', edgecolor='black'))
        
        # Left child
        if node.left:
            # Draw line connecting parent to left child, but connect from center of parent to center of child node
            ax.plot([x, x - dx], [y, y - 0.2], 'k-')  # Connect nodes
            plot_tree(node.left, depth + 1, x - dx, y - 0.2, dx * 0.5, ax)
        
        # Right child
        if node.right:
            # Draw line connecting parent to right child, but connect from center of parent to center of child node
            ax.plot([x, x + dx], [y, y - 0.2], 'k-')  # Connect nodes
            plot_tree(node.right, depth + 1, x + dx, y - 0.2, dx * 0.5, ax)


if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X,y = data.data, data.target

    # X are the data and y the labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    myDT = DecisionTree(max_depth=3)
    myDT.fit(X_train, y_train)
    predictions = myDT.predict(X_test)

    # Print the results
    print("Decision Tree accuracy", accuracy(y_test, predictions))

    # Print the tree structure
    print("\nTree structure:")
    print_tree(myDT.root)

    # Plot the tree structure
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(myDT.root, ax=ax)
    ax.axis('off')  # Hide the axes
    plt.show()