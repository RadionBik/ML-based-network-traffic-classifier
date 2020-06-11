import logging

import numpy as np

from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def _normalize_by_rows(x: np.array):
    safe_x = x.copy()
    safe_x[safe_x == np.inf] = 10e6
    return normalize(safe_x, axis=1, norm='l1')


def _calc_transition_matrix(seq_matrix, state_numb):
    """ here the states are expected to be integers in [0, state_numb) """
    transition_matrix = np.zeros((state_numb, state_numb))
    for row_iter in range(seq_matrix.shape[0]):
        state_seq = seq_matrix[row_iter, :]
        # count number of each possible transition
        for t in range(len(state_seq) - 1):
            j = state_seq[t]
            k = state_seq[t + 1]
            transition_matrix[j, k] += 1

    empty_rows = np.where(transition_matrix.sum(axis=1) == 0)
    empty_row_number = empty_rows[0].size
    if empty_row_number > 0:
        logger.warning(f'found {empty_row_number} empty rows in transition matrix, padding')
        # pad empty transition rows uniformly
        transition_matrix[empty_rows, :] = np.ones((empty_row_number, transition_matrix.shape[0]))
    norm_trans_matrix = _normalize_by_rows(transition_matrix)
    logger.info('estimated transition matrix')
    return norm_trans_matrix


def _calc_prior_probas(seq_matrix, state_numb):
    counts = np.zeros(state_numb)
    for state in range(state_numb):
        counts[state] = np.count_nonzero(seq_matrix[:, 0] == state)
    priors = counts / np.linalg.norm(counts, ord=1)
    logger.info('estimated vector of priors')
    return priors


class MarkovGenerator:
    def __init__(self):
        self.n_states = None
        self.transition_matrix = None
        self.init_priors = None
        self.index2value = {}
        self.value2index = {}
        self._seq_len = None
        self._states = None

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

    def sample(self, n_sequences):
        assert n_sequences > 0
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