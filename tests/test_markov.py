import numpy as np

from gpt_model.generator.baseline import markov


def test_norm():
    x = np.array([np.inf, 0, 0, 1]).reshape(2, -1)
    n_x = markov._normalize_by_rows(x)
    assert (n_x == [[1, 0], [0, 1]]).all()

    x = np.array([[10, 0, ], [4, 16]])

    n_x = markov._normalize_by_rows(x)
    exp_x = np.array([[1., 0.], [0.2, 0.8]])
    assert np.isclose(exp_x, n_x, rtol=1e-3).all()


def test_calc_transition_matrix(quantized_packets):
    trans_matrix = markov._calc_transition_matrix(
        seq_matrix=quantized_packets,
        state_numb=np.unique(quantized_packets).size
    )
    # 0 is the reccurent state
    assert np.isclose(trans_matrix[0, 0], 1, atol=1e-6)


def test_priors(quantized_packets):
    priors = markov._calc_prior_probas(quantized_packets,
                                       np.unique(quantized_packets).size)

    assert np.isclose(priors[10], 0.7541, rtol=1e-3)


def test_markov_generator(quantized_packets):
    gener = markov.MarkovGenerator()
    gener.fit(quantized_packets*-1)
    sampled = gener.sample(1000)
    new_gener = markov.MarkovGenerator()
    new_gener.fit(sampled)
    assert np.isclose(gener.init_priors, new_gener.init_priors, atol=0.1).all()
    # accumulated error < 1. for 114x114 matrix is OK
    tr_matrix_frob_norm = np.linalg.norm(gener.transition_matrix - new_gener.transition_matrix, ord='fro')
    assert tr_matrix_frob_norm < 1.


def test_markov_kmeans_augmenter(raw_dataset):
    def _calc_hist_like_pmf(packet_vector):
        pmf = np.histogram(packet_vector, bins=50, range=(0, 1000), density=True)[0]
        return pmf

    raw_packets = raw_dataset.filter(regex='raw_packet').fillna(0)
    gener = markov.MarkovQuantizedGenerator()
    gener.fit(raw_packets.values)
    output = gener.sample(raw_packets.shape[0])
    priors_distrs_norm = np.linalg.norm(
        _calc_hist_like_pmf(output[:, 0]) -
        _calc_hist_like_pmf(raw_packets.iloc[:, 0]),
        ord=1)
    assert priors_distrs_norm < 0.015

    total_distr_norm = np.linalg.norm(
        _calc_hist_like_pmf(output.flatten()) -
        _calc_hist_like_pmf(raw_packets.values.flatten()),
        ord=1)

    assert total_distr_norm < 0.01
