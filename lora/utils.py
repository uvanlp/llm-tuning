## utils.py

import numpy as np


def identify_label(text, label_set):
    text = text.lower()
    position_indices = []
    for label in label_set:
        # Check which label was generated first
        position_indices.append(text.find(label))
    if position_indices.count(-1) == len(label_set):
        return None
    else:
        position_indices = [1e5 if x == -1 else x for x in position_indices]
    # print(position_indices)
    label_ind = np.argmin(position_indices)
    # print(label_ind)
    return label_set[label_ind]


