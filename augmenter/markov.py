import logging

import numpy as np

from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def _normalize_by_rows(x: np.array):
    safe_x = x.copy()
    safe_x[safe_x == np.inf] = 10e6
    return normalize(safe_x, axis=1, norm='l1')


def _update_transition_matrix(transition_matrix, state_seq):
    states = list(range(transition_matrix.shape[0]))
    for j, state_j in enumerate(states):
        for k, state_k in enumerate(states):
            # count number of each possible transition
            for t in range(len(state_seq) - 1):
                if state_seq[t] == state_j and state_seq[t + 1] == state_k:
                    transition_matrix[j, k] += 1


def _calc_transition_matrix(seq_matrix, state_numb):
    """ here the states are expected to be integers >= 0 with maximal value = state_numb - 1 """
    transition_matrix = np.zeros((state_numb, state_numb))
    for row_iter in range(seq_matrix.shape[0]):
        _update_transition_matrix(transition_matrix, seq_matrix[row_iter, :])

    empty_rows = np.where(transition_matrix.sum(axis=1) == 0)
    # pad empty transition rows uniformly
    transition_matrix[empty_rows, :] = np.ones((empty_rows[0].size, transition_matrix.shape[0]))
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
        self._seq_len = None
        self._states = None

    def fit(self, X):
        self.n_states = np.unique(X).size
        self._states = np.arange(self.n_states)
        self._seq_len = X.shape[1]
        self.transition_matrix = _calc_transition_matrix(X, self.n_states)
        self.init_priors = _calc_prior_probas(X, self.n_states)

    def sample(self, n_sequences):
        assert n_sequences > 0
        sampled_matrix = np.zeros((n_sequences, self._seq_len), dtype=int)
        for seq_index in range(n_sequences):
            sampled_matrix[seq_index, :] = self._sample_sequence()
            if 0 < seq_index % 100 == 0:
                logger.info(f'generated {seq_index} sequences')
        return sampled_matrix

    def _sample_sequence(self):
        sampled = np.zeros(self._seq_len, dtype=int)
        sampled[0] = np.random.choice(self._states, p=self.init_priors)
        for index in range(1, self._seq_len):
            sampled[index] = np.random.choice(self._states, p=self.transition_matrix[sampled[index-1], :])
        return sampled