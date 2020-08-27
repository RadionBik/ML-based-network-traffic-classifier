import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def _normalize_by_rows(x: np.array):
    safe_x = x.copy()
    safe_x[safe_x == np.inf] = 10e6
    return normalize(safe_x, axis=1, norm='l1')


def _calc_transition_matrix(seq_matrix, state_numb):
    """ here the states are expected to be integers in [0, state_numb) """
    # init with values close-to-zero for smoothing
    transition_matrix = np.ones((state_numb, state_numb)) * 1e-6
    for row_iter in range(seq_matrix.shape[0]):
        state_seq = seq_matrix[row_iter, :]
        # count number of each possible transition
        for t in range(len(state_seq) - 1):
            j = state_seq[t]
            k = state_seq[t + 1]
            transition_matrix[j, k] += 1

    norm_trans_matrix = _normalize_by_rows(transition_matrix)
    logger.info(f'estimated transition matrix for {norm_trans_matrix.shape[0]} states')
    return norm_trans_matrix


def _calc_prior_probas(seq_matrix, state_numb):
    counts = np.zeros(state_numb)
    for state in range(state_numb):
        counts[state] = np.count_nonzero(seq_matrix[:, 0] == state)
    priors = counts / np.linalg.norm(counts, ord=1)
    logger.info('estimated vector of priors')
    return priors


class BaseGenerator:
    def fit(self, X):
        raise NotImplementedError

    def sample(self, n_sequences):
        raise NotImplementedError


class MarkovGenerator(BaseGenerator):
    def __init__(self):
        self.n_states = None
        self.transition_matrix = None
        self.init_priors = None
        self.index2value = {}
        self.value2index = {}
        self._seq_len = None
        self._states = None
        logger.info('init MarkovGenerator')

    def _map_values_to_indexes(self, X):
        orig_values = X.flatten()
        self.value2index = {value: index for index, value in enumerate(np.unique(orig_values))}
        self.index2value = {index: value for index, value in enumerate(np.unique(orig_values))}
        X_mapped = np.array([self.value2index[val] for val in orig_values]).reshape(-1, self._seq_len)
        return X_mapped

    def _map_indexes_to_values(self, X_mapped):
        mapped_values = X_mapped.flatten()
        X = np.array([self.index2value[val] for val in mapped_values]).reshape(-1, self._seq_len)
        return X

    def fit(self, X):
        self._seq_len = X.shape[1]
        n_states = np.unique(X).size
        self._states = np.arange(n_states)

        X_mapped = self._map_values_to_indexes(X)

        self.transition_matrix = _calc_transition_matrix(X_mapped, n_states)
        self.init_priors = _calc_prior_probas(X_mapped, n_states)
        return self

    def sample(self, n_sequences):
        assert n_sequences > 0
        logger.info(f'started generating {n_sequences} sequences')
        sampled_matrix = np.zeros((n_sequences, self._seq_len), dtype=int)
        for seq_index in range(n_sequences):
            sampled_matrix[seq_index, :] = self._sample_sequence()
        return self._map_indexes_to_values(sampled_matrix)

    def _sample_sequence(self):
        sampled = np.zeros(self._seq_len, dtype=int)
        sampled[0] = np.random.choice(self._states, p=self.init_priors)
        for index in range(1, self._seq_len):
            sampled[index] = np.random.choice(self._states, p=self.transition_matrix[sampled[index-1], :])
        return sampled


class MarkovQuantizedGenerator(BaseGenerator):
    def __init__(self, cluster_limit=200):
        self.cluster_limit = cluster_limit
        self.quantizer = None
        self.generator = MarkovGenerator()

    def _get_cluster_number(self, X):
        unique_points = np.unique(X).size
        cluster_number = self.cluster_limit if unique_points > self.cluster_limit else unique_points
        logger.info(f'selected {cluster_number} clusters for quantization')
        return cluster_number

    def fit(self, X):
        cluster_number = self._get_cluster_number(X)
        self.quantizer = KMeans(n_clusters=cluster_number)
        X_quantized = self.quantizer.fit_predict(X.flatten().reshape(-1, 1)).reshape(X.shape)
        logger.info('quantized input')
        self.generator.fit(X_quantized)

    def sample(self, n_sequences):
        X_gen = self.generator.sample(n_sequences)
        X_restored = self.quantizer.cluster_centers_[X_gen][:, :, 0]
        logger.info('dequantized output')
        return X_restored
