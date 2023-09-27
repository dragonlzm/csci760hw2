import numpy as np
from collections import Counter
from utils import load_dataset, calc_entropy, calc_info_gain, calc_gain_ratio
from matplotlib import pyplot as plt

class MyDecisionTree:
    def __init__(self):
        self.tree = None
        self.node_num = 0

    def find_best_split(self, X, y):
        num_feat = X.shape[-1]
        best_gain = -1
        best_feat_idx = None
        best_thres = None

        # for each feature
        for feat_idx in range(num_feat):
            # find the unique thres
            unique_val = np.unique(X[:, feat_idx])
            # for each threshold
            for thres in unique_val:
                # if the value is the min value then we should stop the iter, since it spilt all the point into one side
                if thres == min(unique_val):
                    continue
                
                # calculate the gain ratio
                gain = calc_gain_ratio(X, y, feat_idx, thres)
                # mark the best (break the tie randomly)
                if gain > best_gain or (gain == best_gain and np.random.randint(2) > 0):
                    best_gain = gain
                    best_feat_idx = feat_idx
                    best_thres = thres

        return best_feat_idx, best_thres, best_gain

    def build_tree(self, X, y):
        # calculate the node
        self.node_num += 1
        
        # if the node is empty
        if len(y) == 0:
            # return 'leaf', categories, number of instance in 
            return ['leaf', 1, 0]
        
        # If the node is pure (entropy = 0) the recursion
        if len(np.unique(y)) == 1:
            return ['leaf', int(y[0]), len(y)]

        # find the best split
        best_feat_idx, best_thres, best_gain = self.find_best_split(X, y)
        # if the best info gain is zero then break (it should be the case that we have 4v4 after divide we have two 2v2)
        if best_gain == 0:
            #print('we are in test case: y:', y)
            return ['leaf', 1, len(y)]
     
        # set the left and right tree idxs
        left_idx = X[:, best_feat_idx] >= best_thres
        right_idx = ~left_idx

        # run the function recursively
        left_subtree = self.build_tree(X[left_idx], y[left_idx])
        right_subtree = self.build_tree(X[right_idx], y[right_idx])

        return [best_feat_idx, best_thres, left_subtree, right_subtree, best_gain]

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        if self.tree is None:
            raise ValueError("You have not build the tree, call 'fit' first.")
        predictions = []
        for test_sample in X:
            now_node = self.tree
            # to see whether is a leaf node
            while len(now_node) != 3:
                feat_idx, threshold, left, right, best_gain = now_node
                if test_sample[feat_idx] >= threshold:
                    now_node = left
                else:
                    now_node = right
            # get the label
            predictions.append(int(now_node[1]))
        return predictions

# Example usage:
if __name__ == "__main__":
    
    # for non-dbig dataset
    #X, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\D1.txt")
    #X, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\D2.txt")
    #X, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\Druns.txt")
    X, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\D3leaves.txt")
    # X = np.array([[0,1],[1,0],[0,0],[1,1]])
    # y = np.array([0,0,1,1])
    
    tree = MyDecisionTree()
    tree.fit(X, y)
    print(tree.tree)
    print("node number:", tree.node_num)
    
    
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
        # build the tree and do the prediction
        tree = MyDecisionTree()
        tree.fit(now_X, now_y)
        predictions = tree.predict(test_X)
        predictions = np.array(predictions)
        #print('predictions.shape', predictions.shape, 'test_y.shape', test_y.shape)
        acc = np.sum(predictions == test_y) / len(predictions)
        error_list.append(1-acc)
        tree_node_num_list.append(tree.node_num)
        #print("error:", 1-acc,"tree node number:", tree.node_num)
        
    print("error_list:", error_list)
    print("tree_node_num_list:", tree_node_num_list)

    # plot the curve
    fig = plt.figure(figsize=(8.5, 8.5))
    plt.scatter(subset_num, error_list, marker='o', c='r', label='cate0')
    plt.xlabel('feat1')
    plt.ylabel('feat2')
    plt.legend()
    plt.grid(True)
    plt.title('data distribution')
    plt.show()