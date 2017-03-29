"""Class to perform over-sampling using SMOTE."""

# Authors: Christophe Rannou (rannou.christophe.22@gmail.com

from __future__ import division, print_function

import numpy as np


def smote_nc_dist(a, b, c_indices, n_indices):
    """ Adapted Euclidean Distance between two nc samples

    :param a: Sample compsoed of continuous and nominal values
    :param b: Sample composed of continuous and nominal values
    :param c_indices: Indices of continuous values within the samples
    :param n_indices: Indices of nominal values within the samples
    :return:
    """
    # Get standard deviation for continuous values
    med = np.median(np.std([a[c_indices], b[c_indices]], axis=0))

    # compute continuous distance
    d = np.sum((a - b) ** 2)

    # compute categorical distance
    for index in n_indices:
        d += (a[index] == b[index]) * med ** 2

    return np.sqrt(d)


def nearest_neighbors_computation(s, X, c_features, n_features, k=1):
    main_sample = X[s]

    candidates_dist = [smote_nc_dist(main_sample, candidate_sample, c_features, n_features) for candidate_sample in X]
    candidates_indices = np.argsort(candidates_dist)

    neighbors = []

    m = 0
    while len(neighbors) < k:
        if candidates_indices[m] != s:
            neighbors.append(candidates_indices[m])
        m += 1

    return neighbors

def synthesize_sample(s, X, c_features, n_features, ratio=200):
