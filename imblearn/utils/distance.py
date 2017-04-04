"""utilies for distance computation"""
# Authors : Christophe Rannou <rannou.christophe.22@gmail.com>
# License: MIT

import numpy as np

def smote_nc_dist(u, v, categorical_features):
    """ Computes the adapted Euclidean Distance between to nc samples

    The smote_nc distance is the euclidean distance for continuous
    features and for each discrepancy in the categorical features
    the median standard deviation of the continuous features is added.

    Parameters:
    -----------
    u : (N,) array-like
        Sample
    v : (N,) array-like
        Sample
    categorical_features : (n,) array-like
        Array of indices of categorical features within the samples (n < N)
    """

    continuous_features = np.setdiff1d(range(0, len(u)), categorical_features)

    # Get standard deviation for continuous values
    stds = np.std([u[continuous_features], v[continuous_features]], axis=0)
    med = np.median(stds)
    med_c = med ** 2

    # compute continuous distance
    d = np.sum((u - v) ** 2)

    # compute categorical distance
    for index in categorical_features:
        d += (u[index] == v[index]) * med_c

    return np.sqrt(d)