import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from utils import *

dataset_sizes = [32, 128, 512, 2048, 8192]

# load data
X, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\Dbig.txt")

# suffle the data
np.random.seed(42)
shuffled_indices = np.random.permutation(len(y))
shuffled_X = X[shuffled_indices]
shuffled_y = y[shuffled_indices]

# split the train and test
Train_X = shuffled_X[:8192]
Train_y = shuffled_y[:8192]
test_X = shuffled_X[8192:]
test_y = shuffled_y[8192:]

# train tree for different dataset sizes
num_of_node_list = []
error_list = []
for size in dataset_sizes:
    # select the subset for the current size
    now_X = Train_X[:size]
    now_y = Train_y[:size]

    # Create the DecisionTreeClassifier
    mytree = DecisionTreeClassifier()

    # Train the tree and get the node of the tree
    mytree.fit(now_X, now_y)
    num_nodes = mytree.tree_.node_count
    num_of_node_list.append(num_nodes)
    
    # do the prediction
    y_pred = mytree.predict(test_X)
    print('predictions.shape', y_pred.shape, 'test_y.shape', test_y.shape)
    
    # calculate the acc
    acc = np.sum(y_pred == test_y) / len(y_pred)
    error_list.append(1-acc)
    print('num_nodes: ', num_nodes, 'error: ', 1-acc)
    
print("error_list:", error_list, "num_of_node_list:", num_of_node_list)

# plot the error vs training point
fig = plt.figure(figsize=(8.5, 8.5))
plt.scatter(dataset_sizes, error_list, marker='o', c='r')
plt.plot(dataset_sizes, error_list)
plt.xlabel('num of training point')
plt.ylabel('error')
plt.legend()
plt.grid(True)
plt.title('the number training point vs error of the decision tree trained with sklearn')
plt.show()