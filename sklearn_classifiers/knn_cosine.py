import logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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


def voter(class_votes):
    """
    returns vector with the most occurring values within a `class_votes` row, if tie -- selects the first one
    :param class_votes: is a (objects, votes) matrix
    :return:
    """
    def get_result(obj_votes):
        top_count = Counter(obj_votes).most_common(1)
        # return the top key
        return top_count[0][0]

    top_or_first = np.apply_along_axis(get_result, axis=1, arr=class_votes)
    return top_or_first


class KNeighborsCosineClassifier:
    """
    custom K-nn based on cosine similarity
    """

    def __init__(self, n_neighbours=3):
        self.n_neighbours = n_neighbours
        self.X_train: np.ndarray = np.nan
        self.y_train: np.ndarray = np.nan

    def fit(self, X, y):
        X_train = X.values if isinstance(X, pd.DataFrame) else X
        y_train = y.values if isinstance(y, pd.Series) else y
        assert X_train.shape[0] == y_train.shape[0], 'X and y length must match!'
        # assure the values are of np.ndarray type after all
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        logger.info('fit KNeighborsCosineClassifier')

    def predict(self, X, batch_size=1024):
        X = X.values if isinstance(X, pd.DataFrame) else X
        X = np.array(X)
        predictions = np.empty(X.shape[0])
        for start_idx, end_idx in iterate_batch_indexes(X, batch_size):
            top_indexes = top_k_cosine_similar(query=X[start_idx:end_idx], keys=self.X_train, k=self.n_neighbours)
            predictions[start_idx:end_idx] = voter(self.y_train[top_indexes])
        return predictions
