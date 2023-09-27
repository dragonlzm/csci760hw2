import numpy as np
from collections import Counter

# load the dataset with the path
def load_dataset(path):
    data = np.loadtxt(path)
    x = data[:, :-1]
    y = data[:, -1]
    return x, y

# calculate the entropy base on the labels
def calc_entropy(labels):
    # if it is empty
    if len(labels) == 0:
        return 0
    # count the number for each label and caluclate the prob
    count_per_label = Counter(labels)
    prob_per_label = [count / len(labels) for count in count_per_label.values()]
    
    # calculate the entropy for this split
    # if 1.0 in prob_per_label:
    #     return 0
    # else:
    entropy = -sum(p * np.log2(p) for p in prob_per_label if p > 0)
    return entropy

# calculate the information gain
def calc_info_gain(y, left_idx):
    ori_entro = calc_entropy(y)
    left_entro = calc_entropy(y[left_idx])
    right_entro = calc_entropy(y[~left_idx])
    new_entro = left_entro * (sum(left_idx) / len(y)) + right_entro * (sum(~left_idx) / len(y))
    info_gain = ori_entro - new_entro
    return info_gain

# calculate the gain ratio
def calc_gain_ratio(X, y, split_feat_idx, split_thres):
    left_idx = X[:, split_feat_idx] >= split_thres
    info_gain = calc_info_gain(y, left_idx)
    left_ind_prob = sum(left_idx) / len(y)
    right_ind_prob = sum(~left_idx) / len(y)
    split_entro = - (left_ind_prob * np.log2(left_ind_prob) + right_ind_prob * np.log2(right_ind_prob))
    #print('split_feat_idx:', split_feat_idx, 'split_thres:', split_thres, 'info_gain', info_gain, 'gain_ratio', info_gain / split_entro)
    if split_entro == 0:
        return 0
    return info_gain / split_entro