import logging
from collections import Counter

import ngtpy
import numpy as np
import pandas as pd
import puffinn
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize

from sklearn_classifiers.utils import iterate_batch_indexes

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


class KNeighborsCosineClassifier(BaseEstimator):
    """
    custom K-nn based on cosine similarity

    time: 2h18m
    perf:
    accuracy          0.981014  0.981014  0.981014      0.981014
    macro avg         0.862088  0.861645  0.859317  96705.000000
    weighted avg      0.981303  0.981014  0.981095  96705.000000

    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
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
        return self

    def predict(self, X, batch_size=1024):
        X = X.values if isinstance(X, pd.DataFrame) else X
        X = np.array(X)
        predictions = np.empty(X.shape[0], dtype=np.int)
        for start_idx, end_idx in iterate_batch_indexes(X, batch_size):
            top_indexes = top_k_cosine_similar(query=X[start_idx:end_idx], keys=self.target_keys, k=self.n_neighbors)
            predictions[start_idx:end_idx] = batch_voter(self.target_classes[top_indexes])
        return predictions


class KNeighborsLshClassifier(BaseEstimator):

    def __init__(self, n_neighbors=1):
        self.target_classes: np.ndarray = np.nan
        self.n_neighbors = n_neighbors
        self.lsh_table = None

    def _construct_table(self, dataset: np.ndarray):
        raise NotImplementedError

    def _check_set_features(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X = X.astype(np.float32)
        normalize(X, copy=False)
        return X

    def fit(self, X, y):
        X_train = self._check_set_features(X)
        self.target_classes = y.values if isinstance(y, pd.Series) else np.array(y)
        self._construct_table(X_train)
        logger.info(f'fit {self.__class__.__name__}')
        return self

    def _predict(self, X):
        raise NotImplementedError

    def predict(self, X):
        X = self._check_set_features(X)
        return self._predict(X)


class KNeighborsPuffinnClassifier(KNeighborsLshClassifier):
    """
    PUFFINN - Parameterless and Universal Fast Finding of Nearest Neighbors
    https://arxiv.org/pdf/1906.12211.pdf

    time: 12m
    perf:
    accuracy          0.981759  0.981759  0.981759      0.981759
    macro avg         0.865334  0.861639  0.860683  96705.000000
    weighted avg      0.981953  0.981759  0.981810  96705.000000

    it is really close to the perf of grid-search K-nn approach but much faster
    """

    def __init__(self, n_neighbors=1, search_recall=0.995, memory_limit=1 * 1024 ** 3):
        super().__init__(n_neighbors)
        self.memory_limit = memory_limit
        self.search_recall = search_recall
        self.lsh_table: puffinn.Index

    def _construct_table(self, dataset: np.ndarray):
        self.lsh_table = puffinn.Index('angular', dataset.shape[1], self.memory_limit)
        for v in dataset:
            self.lsh_table.insert(v.tolist())
        logger.info('building index table...')
        self.lsh_table.rebuild()

    def _predict(self, X):
        def query_predictor(query):
            top_indexes = self.lsh_table.search(query.tolist(), self.n_neighbors, self.search_recall)
            return voter(self.target_classes[top_indexes])

        predictions = np.apply_along_axis(query_predictor, axis=1, arr=X)
        return predictions


class KNeighborsNGTClassifier(KNeighborsLshClassifier):
    """
    ONNG-NGT (https://github.com/yahoojapan/NGT/wiki)

    better keep optimize_* args as defaults, it doesn't work as expected
    """

    def __init__(
            self,
            n_neighbors=1,
            search_epsilon=0.1,
            optimize_n_edges=False,
            optimize_search_params=False,
            index_path='/tmp/knn_ngt_index'
    ):
        super().__init__(n_neighbors)
        self.index_path = index_path
        self.optimize_n_edges = optimize_n_edges
        self.optimize_search_params = optimize_search_params
        self.search_epsilon = search_epsilon

    def _construct_table(self, dataset: np.ndarray):
        # when data is normalized row-wise, the L2 distance metric is similar to the cosine
        ngtpy.create(self.index_path, dataset.shape[1], distance_type='L2')
        index = ngtpy.Index(self.index_path)  # open the index
        index.batch_insert(dataset)
        if self.optimize_n_edges:
            logger.info('optimizing number of edges...')
            index.save()
            optimizer = ngtpy.Optimizer(log_disabled=True)
            try:
                optimizer.optimize_number_of_edges_for_anng(self.index_path)
            except RuntimeError as e:
                logger.error(f'skipping optimization due to: {e}')
        if self.optimize_search_params:
            optimizer = ngtpy.Optimizer(log_disabled=True)
            optimizer.set_processing_modes(
                search_parameter_optimization=True,
                prefetch_parameter_optimization=True,
                accuracy_table_generation=True)
            optimizer.optimize_search_parameters(self.index_path)
        logger.info('building index table...')
        index.build_index()  # build index
        index.save()
        self.lsh_table = index

    def _predict(self, X):
        def query_predictor(query):
            if self.optimize_search_params:
                top_indexes = self.lsh_table.search(query, size=self.n_neighbors, expected_accuracy=0.99)
            else:
                top_indexes = self.lsh_table.search(query, size=self.n_neighbors, epsilon=self.search_epsilon)

            top_indexes = [i[0] for i in top_indexes]
            return voter(self.target_classes[top_indexes])

        predictions = np.apply_along_axis(query_predictor, axis=1, arr=X)
        return predictions
