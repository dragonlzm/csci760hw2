import numpy as np
import matplotlib.pyplot as plt
from mydecisiontree import MyDecisionTree
from utils import load_dataset

# plot the decision area by using points
def plot_decision_boundary(X, y, model):
    # Define the range of area
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # sample points
    sample_x, sample_y = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # predict each point
    sample_predicted_label = np.array(model.predict(np.c_[sample_x.ravel(), sample_y.ravel()]))
    sample_predicted_label = sample_predicted_label.reshape(sample_x.shape)

    # Create a contour plot
    plt.contourf(sample_x, sample_y, sample_predicted_label, cmap=plt.cm.RdBu, alpha=0.6)

    # Plot the data points
    # plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=f'Class {int(1)}', cmap=plt.cm.RdBu, s=12, edgecolor='k')
    # plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=f'Class {int(0)}', cmap=plt.cm.RdBu, s=12, edgecolor='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    #plt.title('Decision Boundary of decision tree build on D1.txt')
    #plt.title('Decision Boundary of decision tree build on D2.txt')
    plt.title('Decision Boundary of decision tree build on ' + str(len(y)) + ' data points')
    plt.show()


if __name__ == "__main__":
    tree = MyDecisionTree()
    #X, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\D1.txt")
    #X, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\D2.txt")
    #tree.fit(X, y)
    #plot_decision_boundary(X, y, tree)
    
    # shuffle the point for Dbig
    X, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\Dbig.txt")
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(y))
    shuffled_X = X[shuffled_indices]
    shuffled_y = y[shuffled_indices]
    # split the train test
    Train_X = shuffled_X[:8192]
    Train_y = shuffled_y[:8192]
    test_X = shuffled_X[8192:]
    test_y = shuffled_y[8192:]
    
    # select a subset of the training set
    subset_num = [32, 128, 512, 2048, 8192]
    error_list = []
    tree_node_num_list = []
    
    for num in subset_num:
        now_X = Train_X[:num]
        now_y = Train_y[:num]
        tree = MyDecisionTree()
        tree.fit(now_X, now_y)
        plot_decision_boundary(now_X, now_y, tree)
        #print(tree.tree)
        #test_data = np.array([[3.0, 4.0], [1.0, 2.0]])