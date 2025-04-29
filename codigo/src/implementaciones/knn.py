import numpy as np


def dcos(A, B):
    return 1 - (np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B)))


def knn(k, train_data, test_data, train_types, test_types):
    asserts = np.zeros(len(test_data), dtype=int)
    
    norm_test = test_data / np.linalg.norm(test_data, axis=1, keepdims=True)
    norm_train = train_data / np.linalg.norm(train_data, axis=1, keepdims=True)

    distances = 1 - np.dot(norm_test, norm_train.T)

    k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
    k_nearest = train_types[k_nearest_indices]

    predictions = []
    for types in k_nearest:
        unique, counts = np.unique(types, return_counts=True)
        mode = unique[np.argmax(counts)]
        predictions.append(mode)

    predictions = np.array(predictions)

    asserts = (predictions == test_types)
    return asserts
