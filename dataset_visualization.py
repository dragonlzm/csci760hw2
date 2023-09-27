from utils import load_dataset
from matplotlib import pyplot as plt
import numpy as np
# visualize the data


# load the data
#x, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\D1.txt")
#x, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\D2.txt")
#x, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\D3leaves.txt")
#x, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\Druns.txt")
x, y = load_dataset("C:\\Users\\Zhuoming Liu\\Desktop\\course_resources\\UWM courses\\(23fall)CS760\\homework\\hw2\\Homework 2 data\\Dbig.txt")

# for question 2.2
# x = np.array([[0,1],[1,0],[0,0],[1,1]])
# y = np.array([0,0,1,1])

np.random.seed(42)
shuffled_indices = np.random.permutation(len(y))
shuffled_X = x[shuffled_indices]
shuffled_y = y[shuffled_indices]
# split the train test
Train_X = shuffled_X[:8192]
Train_y = shuffled_y[:8192]
test_X = shuffled_X[8192:]
test_y = shuffled_y[8192:]
    

feat1 = Train_X[:, 0]
feat2 = Train_X[:, 1]

# split the data base on the labels
label1_feat1 = []
label1_feat2 = []
label2_feat1 = []
label2_feat2 = []

for f1, f2, cate in zip(feat1, feat2, Train_y):
    if cate == 0:
        label1_feat1.append(f1)
        label1_feat2.append(f2)
    else:
        label2_feat1.append(f1)
        label2_feat2.append(f2)


fig = plt.figure(figsize=(8.5, 8.5))
plt.scatter(label1_feat1, label1_feat2, marker='o', c='r', label='cate0')
plt.scatter(label2_feat1, label2_feat2, marker='o', c='g', label='cate1')
plt.xlabel('feat1')
plt.ylabel('feat2')
plt.legend()
plt.grid(True)
plt.title('data distribution')
plt.show()