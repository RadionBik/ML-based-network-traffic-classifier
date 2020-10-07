import logging
from collections import Counter

import falconn
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sklearn_classifiers.utils import iterate_batch_indexes
from settings import RANDOM_SEED

logger = logging.getLogger(__name__)


def cos_dist(query, keys):
    # if got vector
    if len(query.shape) == 1:
        query = query.reshape(1, -1)
    return cdist(keys, query, 'cosine').T


def top_k_cosine_similar(query, keys, k=1):
    distances = cos_dist(query, keys)
    top_k = np.argpartition(distances, k)[:, :k]
    return top_k


def voter(obj_votes):
    top_count = Counter(obj_votes).most_common(1)
    # return the top key
    return top_count[0][0]


def batch_voter(class_votes):
    """
    returns vector with the most occurring values within a `class_votes` row, if tie -- selects the first one
    :param class_votes: is a (objects, votes) matrix
    :return:
    """
    top_or_first = np.apply_along_axis(voter, axis=1, arr=class_votes)
    return top_or_first


class KNeighborsCosineClassifier:
    """
    custom K-nn based on cosine similarity

    time: 2h18m
    perf:
    accuracy          0.981014  0.981014  0.981014      0.981014
    macro avg         0.862088  0.861645  0.859317  96705.000000
    weighted avg      0.981303  0.981014  0.981095  96705.000000

    """

    def __init__(self, n_neighbours=3):
        self.n_neighbours = n_neighbours
        self.target_keys: np.ndarray = np.nan
        self.target_classes: np.ndarray = np.nan

    def fit(self, X, y):
        X_train = X.values if isinstance(X, pd.DataFrame) else X
        y_train = y.values if isinstance(y, pd.Series) else y
        assert X_train.shape[0] == y_train.shape[0], 'X and y length must match!'
        # assure the values are of np.ndarray type after all
        self.target_keys = np.array(X_train)
        self.target_classes = np.array(y_train)
        logger.info('fit KNeighborsCosineClassifier')

    def predict(self, X, batch_size=1024):
        X = X.values if isinstance(X, pd.DataFrame) else X
        X = np.array(X)
        predictions = np.empty(X.shape[0])
        for start_idx, end_idx in iterate_batch_indexes(X, batch_size):
            top_indexes = top_k_cosine_similar(query=X[start_idx:end_idx], keys=self.target_keys, k=self.n_neighbours)
            predictions[start_idx:end_idx] = batch_voter(self.target_classes[top_indexes])
        return predictions


class KNeighborsLshClassifier:

    def __init__(self, n_neighbours=3, n_lsh_tables=50, n_bit_hash=18, n_probes=50, random_state=RANDOM_SEED):
        self.target_classes: np.ndarray = np.nan
        self.n_neighbours = n_neighbours
        self.n_lsh_tables = n_lsh_tables
        self.n_bit_hash = n_bit_hash
        self.n_probes = n_probes
        self.random_state = random_state
        self._dataset_centers: np.ndarray
        self._table_params: falconn.LSHConstructionParameters
        self._table: falconn.LSHIndex
        self.query_object = None

    def _construct_table(self, dataset: np.ndarray):
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dataset.shape[1]
        params_cp.lsh_family = falconn.LSHFamily.Hyperplane
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = self.n_lsh_tables
        # we set one rotation, since the data is dense enough,
        # for sparse data set it to 2
        params_cp.num_rotations = 1
        params_cp.seed = self.random_state
        # we want to use all the available threads to set up
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        # we build 18-bit hashes so that each table has
        # 2^18 bins; this is a good choise since 2^18 is of the same
        # order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(self.n_bit_hash, params_cp)
        self._table_params = params_cp
        self._table = falconn.LSHIndex(params_cp)
        self._table.setup(dataset)
        return self._table.construct_query_object()

    def _check_set_features(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X = X.astype(np.float32)
        X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
        return X

    def fit(self, X, y):
        X_train = self._check_set_features(X)
        self._dataset_centers = np.mean(X_train, axis=0)
        X_train -= self._dataset_centers
        self.target_classes = y.values if isinstance(y, pd.Series) else np.array(y)
        self.query_object = self._construct_table(X_train)
        logger.info('fit KNeighborsLshClassifier')

    def predict(self, X):
        X = self._check_set_features(X) - self._dataset_centers

        def query_predictor(query):
            top_indexes = self.query_object.find_k_nearest_neighbors(query, self.n_neighbours)
            return voter(self.target_classes[top_indexes])

        predictions = np.apply_along_axis(query_predictor, axis=1, arr=X)
        return predictions
