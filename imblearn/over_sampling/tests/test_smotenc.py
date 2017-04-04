"""Test the module SMOTE."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_raises_regex)
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTENC
from imblearn.utils import smote_nc_dist

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[1151, -6893, 47, 6], [68204389, 1720726, 47, 3],
              [265255957, 106490, 47, 6], [94199088, 27636, 37, 6],
              [106009821, 2007, 47, 6], [34558124, -7915, 48, 3],
              [2706204, 15090191, 99, 4], [338402, -3590, 63, 6],
              [65378, -4675, 47, 6], [313202261, 4390275, 13, 3],
              [705, 1286805, 45, 6], [303298680, 53318269, 47, 6],
              [203998874, 50288256, 47, 3], [87330237, 4829060, 48, 6],
              [349445203, 46481790, 47, 6], [56624969, -3818, 48, 3],
              [59250991, 1634443, 23, 6], [333848750, 6404769, 47, 3],
              [83552727, 8837737, 128, 6], [69498698, 34008783, 47, 6]])
categorical_features = [2, 3]
y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_smote_wrong_kind():
    kind = 'rnd'
    smotenc = SMOTENC(categorical_features=categorical_features, kind=kind, random_state=RND_SEED)
    assert_raises_regex(ValueError, "Unknown kind for SMOTE", smotenc.fit_sample, X, y)


def test_sample_regular():
    # Create the object
    kind = 'regular'
    smotenc = SMOTENC(categorical_features=categorical_features, random_state=RND_SEED, kind=kind)
    # Fit the data
    smotenc.fit(X, y)

    X_resampled, y_resampled = smotenc.fit_sample(X, y)

    X_gt = np.array([[1151, -6893, 47, 6], [68204389, 1720726, 47, 3],
                     [265255957, 106490, 47, 6], [94199088, 27636, 37, 6],
                     [106009821, 2007, 47, 6], [34558124, -7915, 48, 3],
                     [2706204, 15090191, 99, 4], [338402, -3590, 63, 6],
                     [65378, -4675, 47, 6], [313202261, 4390275, 13, 3],
                     [705, 1286805, 45, 6], [303298680, 53318269, 47, 6],
                     [203998874, 50288256, 47, 3], [87330237, 4829060, 48, 6],
                     [349445203, 46481790, 47, 6], [56624969, -3818, 48, 3],
                     [59250991, 1634443, 23, 6], [333848750, 6404769, 47, 3],
                     [83552727, 8837737, 128, 6], [69498698, 34008783, 47, 6],
                     [29444120, 14404011, 47, 6], [131762079, 32478447, 47, 6],
                     [89267945, 22001621, 47, 6], [244069736, 3775633, 47, 6]])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    # Create the object
    ratio = 0.8
    kind = 'regular'
    smotenc = SMOTENC(categorical_features=categorical_features, ratio=ratio, random_state=RND_SEED, kind=kind)
    # Fit the data
    smotenc.fit(X, y)

    X_resampled, y_resampled = smotenc.fit_sample(X, y)

    X_gt = np.array([[1151, -6893, 47, 6], [68204389, 1720726, 47, 3],
                     [265255957, 106490, 47, 6], [94199088, 27636, 37, 6],
                     [106009821, 2007, 47, 6], [34558124, -7915, 48, 3],
                     [2706204, 15090191, 99, 4], [338402, -3590, 63, 6],
                     [65378, -4675, 47, 6], [313202261, 4390275, 13, 3],
                     [705, 1286805, 45, 6], [303298680, 53318269, 47, 6],
                     [203998874, 50288256, 47, 3], [87330237, 4829060, 48, 6],
                     [349445203, 46481790, 47, 6], [56624969, -3818, 48, 3],
                     [59250991, 1634443, 23, 6], [333848750, 6404769, 47, 3],
                     [83552727, 8837737, 128, 6], [69498698, 34008783, 47, 6],
                     [41891727, 20496510, 47, 6]
                     ])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_borderline1():
    # Create the object
    kind = 'borderline1'
    smotenc = SMOTENC(categorical_features=categorical_features, random_state=RND_SEED, kind=kind)
    # Fit the data
    smotenc.fit(X, y)

    X_resampled, y_resampled = smotenc.fit_sample(X, y)

    X_gt = np.array([[1151, -6893, 47, 6], [68204389, 1720726, 47, 3],
                     [265255957, 106490, 47, 6], [94199088, 27636, 37, 6],
                     [106009821, 2007, 47, 6], [34558124, -7915, 48, 3],
                     [2706204, 15090191, 99, 4], [338402, -3590, 63, 6],
                     [65378, -4675, 47, 6], [313202261, 4390275, 13, 3],
                     [705, 1286805, 45, 6], [303298680, 53318269, 47, 6],
                     [203998874, 50288256, 47, 3], [87330237, 4829060, 48, 6],
                     [349445203, 46481790, 47, 6], [56624969, -3818, 48, 3],
                     [59250991, 1634443, 23, 6], [333848750, 6404769, 47, 3],
                     [83552727, 8837737, 128, 6], [69498698, 34008783, 47, 6],
                     [37420021, 6823, 47, 6], [99574351, 15971, 47, 6],
                     [304789084, 3736472, 47, 6], [44889207, 21963631, 47, 6]])

    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_borderline2():
    # Create the object
    kind = 'borderline2'
    smotenc = SMOTENC(categorical_features=categorical_features, random_state=RND_SEED, kind=kind)
    # Fit the data
    smotenc.fit(X, y)

    X_resampled, y_resampled = smotenc.fit_sample(X, y)

    X_gt = np.array([[1151, -6893, 47, 6], [68204389, 1720726, 47, 3],
                     [265255957, 106490, 47, 6], [94199088, 27636, 37, 6],
                     [106009821, 2007, 47, 6], [34558124, -7915, 48, 3],
                     [2706204, 15090191, 99, 4], [338402, -3590, 63, 6],
                     [65378, -4675, 47, 6], [313202261, 4390275, 13, 3],
                     [705, 1286805, 45, 6], [303298680, 53318269, 47, 6],
                     [203998874, 50288256, 47, 3], [87330237, 4829060, 48, 6],
                     [349445203, 46481790, 47, 6], [56624969, -3818, 48, 3],
                     [59250991, 1634443, 23, 6], [333848750, 6404769, 47, 3],
                     [83552727, 8837737, 128, 6], [69498698, 34008783, 47, 6],
                     [26829725, 2941, 47, 6], [98890743, 17455, 47, 6],
                     [86493679, 529505, 48, 3]])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0])

    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_fit_sample_nn_obj():
    # Create the object
    kind = 'borderline1'
    nn_m = NearestNeighbors(n_neighbors=11, metric=smote_nc_dist,
                            metric_params={'categorical_features': categorical_features})
    nn_k = NearestNeighbors(n_neighbors=6, metric=smote_nc_dist,
                            metric_params={'categorical_features': categorical_features})
    smotenc = SMOTENC(categorical_features=categorical_features,
                      random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)

    X_resampled, y_resampled = smotenc.fit_sample(X, y)

    X_gt = np.array([[1151, -6893, 47, 6], [68204389, 1720726, 47, 3],
                     [265255957, 106490, 47, 6], [94199088, 27636, 37, 6],
                     [106009821, 2007, 47, 6], [34558124, -7915, 48, 3],
                     [2706204, 15090191, 99, 4], [338402, -3590, 63, 6],
                     [65378, -4675, 47, 6], [313202261, 4390275, 13, 3],
                     [705, 1286805, 45, 6], [303298680, 53318269, 47, 6],
                     [203998874, 50288256, 47, 3], [87330237, 4829060, 48, 6],
                     [349445203, 46481790, 47, 6], [56624969, -3818, 48, 3],
                     [59250991, 1634443, 23, 6], [333848750, 6404769, 47, 3],
                     [83552727, 8837737, 128, 6], [69498698, 34008783, 47, 6],
                     [37420021, 6823, 47, 6], [99574351, 15971, 47, 6],
                     [304789084, 3736472, 47, 6], [44889207, 21963631, 47, 6]
                     ])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_with_nn():
    # Create the object
    kind = 'regular'
    nn_k = NearestNeighbors(n_neighbors=6, metric=smote_nc_dist,
                            metric_params={'categorical_features': categorical_features})
    smotenc = SMOTENC(categorical_features=categorical_features,
                      random_state=RND_SEED, kind=kind, k_neighbors=nn_k)

    X_resampled, y_resampled = smotenc.fit_sample(X, y)

    X_gt = np.array([[1151, -6893, 47, 6], [68204389, 1720726, 47, 3],
                     [265255957, 106490, 47, 6], [94199088, 27636, 37, 6],
                     [106009821, 2007, 47, 6], [34558124, -7915, 48, 3],
                     [2706204, 15090191, 99, 4], [338402, -3590, 63, 6],
                     [65378, -4675, 47, 6], [313202261, 4390275, 13, 3],
                     [705, 1286805, 45, 6], [303298680, 53318269, 47, 6],
                     [203998874, 50288256, 47, 3], [87330237, 4829060, 48, 6],
                     [349445203, 46481790, 47, 6], [56624969, -3818, 48, 3],
                     [59250991, 1634443, 23, 6], [333848750, 6404769, 47, 3],
                     [83552727, 8837737, 128, 6], [69498698, 34008783, 47, 6],
                     [29444120, 14404011, 47, 6], [131762079, 32478447, 47, 6],
                     [89267945, 22001620, 47, 6], [244069736, 3775633, 47, 6]

                     ])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_wrong_nn():
    # Create the object
    kind = 'borderline1'
    nn_m = 'rnd'
    nn_k = NearestNeighbors(n_neighbors=6, metric=smote_nc_dist,
                            metric_params={'categorical_features': categorical_features})
    smotenc = SMOTENC(categorical_features=categorical_features,
                      random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)

    assert_raises_regex(ValueError, "has to be one of",
                        smotenc.fit_sample, X, y)

    nn_k = 'rnd'
    nn_m = NearestNeighbors(n_neighbors=10)
    smotenc = SMOTENC(categorical_features=categorical_features,
                      random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)

    assert_raises_regex(ValueError, "has to be one of",
                        smotenc.fit_sample, X, y)

    kind = 'regular'
    nn_k = 'rnd'
    smotenc = SMOTENC(categorical_features=categorical_features, kind=kind, k_neighbors=nn_k)

    assert_raises_regex(ValueError, "has to be one of",
                        smotenc.fit_sample, X, y)
