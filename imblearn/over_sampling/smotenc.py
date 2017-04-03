"""Class to perform over-sampling using SMOTE-NC."""

# Authors: Christophe Rannou (rannou.christophe.22@gmail.com

from __future__ import division, print_function

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state

from imblearn.base import BaseBinarySampler
from imblearn.utils import check_neighbors_object

SMOTE_KIND = ('regular', 'borderline1', 'borderline2')


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


class SMOTENC(BaseBinarySampler):
    """
    Class to perform over-sampling using SMOTE-NC.

    This object is an implementation of SMOTE-NC - Synthetic Minority
    Over-sampling TEchnique - Nominal Continuous.

    Nominal/Categorical values are supposed to be passed as integers
    thus if features are plain text labels should be encoded prior
    to the use of SMOTE-NC (using scikit learn LabelEncoder for instance)

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority clas over the number of samples
        in the majority class.

    random_state : int, RandomSrate instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, randpù_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random

    k_neighbors : int or object, optional (default=5)
        If int, number of nearest neighbors to ue to construct
        synthetic samples.
        If object, an estimator that inherits from
        `sklearn.neighbors.base.KNeighborsMixin` that will be used to find
        the k_neighbors.

    kind : str, optional (default='regular')
        The type of SMOTE-NC algorithm to use one of the following options:
        'regular'.

    n_jobs : int, optional (default=1)
        The number of threadds to open if possible.

    Attributes
    ----------
    min_c : str or int
        The identifier of the minority class.

    max_c : str or int
        The identifier of the majority class.

    stats_c : dict of str/int : int
        A dictionary in which the number of occurrences of each class is
        reported.

    categorical_features : array of int
        The indices of the nominal features within a sample

    X_shape : tuple of int
        Shape of the data `X` during fitting.
    """

    def __init__(self,
                 categorical_features,
                 ratio='auto',
                 random_state=None,
                 k_neighbors=5,
                 m_neighbors=10,
                 kind='regular',
                 n_jobs=1):
        super(SMOTENC, self).__init__(ratio=ratio, random_state=random_state)
        self.categorical_features = categorical_features
        self.kind = kind
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.n_jobs = n_jobs

    def _in_danger_noise(self, samples, y, kind='danger'):
        """Estimate if a set of sample are in danger or noise.

        Parameters
        ----------
        samples : ndarray, shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        y : ndarray, shape (n_samples, )
            The true label in order to check the neighbour labels.

        kind : str, optional (default='danger')
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray, shape (n_samples, )
            A boolean array where True refer to samples in danger or noise.

        """

        # Find the NN for each samples
        # Exclude the sample itself
        x = self.nn_m_.kneighbors(samples, return_distance=False)[:, 1:]

        # Compute the number of majority samples in the NN
        nn_label = (y[x] != self.min_c_).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        if kind == 'danger':
            # Samples are in danger for m/2 <= m' < m (if m' = m it is not border but noise)
            return np.bitwise_and(
                n_maj >= float(self.nn_m_.n_neighbors - 1) / 2.,
                n_maj < self.nn_m_.n_neighbors - 1
            )
        elif kind == 'noise':
            # Samples are noise for m = m'
            return n_maj == self.nn_m_.n_neighbors - 1
        else:
            raise NotImplementedError

    def _make_samples(self,
                      X,
                      y_type,
                      nn_data,
                      nn_num,
                      n_samples,
                      step_size=1.):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Points from which the points will be created.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in nn_data.

        n_samples : int
            The number of samples to generate.

        step_size : float, optional (default=1.)
            The step size to create samples.

        Returns
        -------
        X_new : ndarray, shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray, shape (n_samples_new, )
            Target values for synthetic samples.

        """
        # Check the consistency of X
        X = check_array(X)

        # Check the random state
        random_state = check_random_state(self.random_state)

        # A matrix to store the synthetic samples
        X_new = np.zeros((n_samples, X.shape[1]))

        # Get the continuous features indices
        continuous_features = np.setdiff1d(range(0, X.shape[1]), self.categorical_features)

        # Randomly pick sample to construct neighbors from
        samples = random_state.randint(low=0, high=len(nn_num.flatten()), size=n_samples)

        # Loop over the NN matrix and create new samples
        for i, n in enumerate(samples):
            # NN lines relate to original sample, columns to its
            # nearest neighbors
            row, col = divmod(n, nn_num.shape[1])

            # Take a step of random size (0,1) in the direction of the
            # n nearest neighbor
            step = step_size * random_state.uniform()

            # Construct synthetic sample for the continuous part
            X_new[i, continuous_features] = X[row, continuous_features] - step * (
                X[row, continuous_features] - nn_data[nn_num[row, col], continuous_features])

            # Construct synthetic sample for the nominal part
            # The generated nominal features are the result of a majority
            # vote among the k nearest neighbors (the mode along each feature)

            # Mesh to select subarray composed of nearest neighbors with
            # their categorical features
            mesh = np.ix_(nn_num[row], self.categorical_features)

            # Compute modes
            modes = stats.mode(nn_data[mesh])[0][0]

            X_new[i, self.categorical_features] = modes

        # The returned target vector is simple a repetition of the
        # minority label
        y_new = np.array([y_type] * len(X_new))

        self.logger.info('generated %s new samples ...', len(X_new))

        return X_new, y_new

    def _validate_estimator(self):
        # --- NN object
        # Import the NN object from scikit-learn library. Since in the smote
        # variations we must first find samples that are in danger, we
        # intialize the NN object differently depending on the method chosen

        # In case k_neighbors is an int need to initialize smote-nc distance measure
        if isinstance(self.k_neighbors, int):
            # Add an additional neighbor as the sample is considered as
            # the closest neighbor and we wan k_neighbors distinct from sample
            self.k_neighbors = NearestNeighbors(n_neighbors=self.k_neighbors + 1,
                                                metric=smote_nc_dist,
                                                metric_params={'categorical_features': self.categorical_features})

        if self.kind == 'regular':
            # Regular smote does not look for samples in danger, instead it
            # creates synthetic samples directly from the k-th nearest
            # neighbors with no filtering
            self.nn_k_ = check_neighbors_object('k_neighbors',
                                                self.k_neighbors,
                                                additional_neighbor=1)

            # set the number of jobs
            self.nn_k_.set_params(**{'n_jobs': self.n_jobs})

        else:
            # Borderline1, 2 and SVM variations of smote must first look for
            # samples that could be considered noise and samples that live
            # near the boundary between the classes. Therefore, before
            # creating synthetic samples from the k-th nns, it first look
            # for m nearest neighbors to decide whether or not a sample is
            # noise or near the boundary.
            self.nn_k_ = check_neighbors_object('k_neighbors',
                                                self.k_neighbors,
                                                additional_neighbor=1)
            # set the number of jobs
            self.nn_k_.set_params(**{'n_jobs': self.n_jobs})

            if isinstance(self.m_neighbors, int):
                # Add an additional neighbor as the sample is considered as
                # the closest neighbor and we wan k_neighbors distinct from sample
                self.nn_m_ = NearestNeighbors(n_neighbors=self.m_neighbors + 1,
                                              metric=smote_nc_dist,
                                              metric_params={'categorical_features': self.categorical_features})

            self.nn_m_ = check_neighbors_object('m_neighbors',
                                                self.m_neighbors,
                                                additional_neighbor=1)

            # set the number of jobs
            self.nn_m_.set_params(**{'n_jobs': self.n_jobs})

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled

        y : ndarray, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """

        super(SMOTENC, self).fit(X, y)

        self._validate_estimator()

        return self

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n-samples_new)
            The corresponding label of `X_resampled`

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """

        if self.kind not in SMOTE_KIND:
            raise ValueError('Unknown kind for SMOTE algorithm.'
                             ' Choices are {}. Got {} instead.'.format(
                SMOTE_KIND, self.kind))

        random_state = check_random_state(self.random_state)

        # Define the number of sample to create
        # We handle only two classes problem for the moment.
        if self.ratio == 'auto':
            num_samples = (self.stats_c_[self.maj_c_] - self.stats_c_[self.min_c_])
        else:
            num_samples = int((self.ratio * self.stats_c_[self.maj_c_]) - self.stats_c_[self.min_c_])

        # Start by separating minority class features and target values
        X_min = X[y == self.min_c_]

        if self.kind == 'regular':
            self.logger.debug('Finding the %s nearest neighbors ...', self.nn_k_.n_neighbors)

            # Look for the k-th nearest neighbor, excluding, of course, the
            # point itself
            self.nn_k_.fit(X_min)

            # Matrix with k-th nearest neighbors indexes for each minority
            # element.
            nns = self.nn_k_.kneighbors(X_min, return_distance=False)[:, 1:]

            self.logger.debug('Create synthetic samples ...')

            # --- Generating synthetic samples
            # Use static method make_samples to generate minority samples
            X_new, y_new = self._make_samples(X_min, self.min_c_, X_min, nns, num_samples, 1.0)

            # Concatenate the newly generated samples to the original data set
            X_resampled = np.concatenate((X, X_new), axis=0)
            y_resampled = np.concatenate((y, y_new), axis=0)

            return X_resampled, y_resampled

        if self.kind == 'borderline1' or self.kind == 'borderline2':

            self.logger.debug('Finding the %s nearest neighbors ...',
                              self.nn_m_.n_neighbors - 1)

            # Find the NNs
            self.nn_m_.fit(X)

            # Boolean array with True for minority samples in danfer
            danger_index = self._in_danger_noise(X_min, y, kind='danger')

            # If all minority samples are safe, return the original data set
            if not any(danger_index):
                self.logger.debug('There are no samples in danger. No'
                                  ' borderline synthetic samples created.')

                # All are safe, nothing to be done here
                return X, y

            # If we got here is because some samples are in danger, we need to
            # find the NNs among the minority class to create the new synthetic
            # samples.

            self.nn_k_.fit(X_min)

            # Matrix with k-th nearest neighbors indexes for each minority
            # element in danger.
            nns = self.nn_k_.kneighbors(X_min[danger_index], return_distance=False)[:, 1:]

            # B1 and B2 types diverge here!!!
            if self.kind == 'borderline1':
                # Create synthetic samples for borderline points.
                X_new, y_new = self._make_samples(
                    X_min[danger_index], self.min_c_, X_min, nns, num_samples)

                # Concatenate the newly generated samples to the original
                # dataset
                X_resampled = np.concatenate((X, X_new), axis=0)
                y_resampled = np.concatenate((y, y_new), axis=0)

                return X_resampled, y_resampled

            else:
                # Split the number of synthetic samples between only minority
                # (type 1), or minority and majority (with reduced step size)
                # (type 2).
                # The fraction is sampled from a beta distribution centered
                # around 0.5 with variance ~0.01
                fractions = random_state.beta(10, 10)

                # Only minority
                X_new_1, y_new_1 = self._make_samples(
                    X_min[danger_index],
                    self.min_c_,
                    X_min,
                    nns,
                    int(fractions * (num_samples + 1)),
                    step_size=1.)

                # Only majority with smaller step size
                X_new_2, y_new_2 = self._make_samples(
                    X_min[danger_index],
                    self.min_c_,
                    X[y != self.min_c_],
                    nns,
                    int((1 - fractions) * num_samples),
                    step_size=0.5)

                # Concatenate the newly generated samples to the original
                # data set
                X_resampled = np.concatenate((X, X_new_1, X_new_2), axis=0)
                y_resampled = np.concatenate((y, y_new_1, y_new_2), axis=0)

                return X_resampled, y_resampled
