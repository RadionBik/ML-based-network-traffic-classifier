import numpy as np
import pandas as pd

from augmenter import markov, calc_flow_features_raw_packets, oversample_raw_packets


def test_norm():
    x = np.array([np.inf, 0, 0, 1]).reshape(2, -1)
    n_x = markov._normalize_by_rows(x)
    assert (n_x == [[1, 0], [0, 1]]).all()

    x = np.array([[10, 0, ], [4, 16]])

    n_x = markov._normalize_by_rows(x)
    exp_x = np.array([[1., 0.], [0.2425, 0.9701]])
    assert np.isclose(exp_x, n_x, rtol=1e-3).all()


def test_calc_transition_matrix(quantized_packets):
    trans_matrix = markov._calc_transition_matrix(
        seq_matrix=quantized_packets,
        state_numb=np.unique(quantized_packets).size
    )
    # 0 is the reccurent state
    assert trans_matrix[0, 0] == 1


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


def test_feature_calc_from_generated(raw_dataset):
    raw_packets = raw_dataset.filter(regex='raw_packet')
    output = raw_packets.apply(func=calc_flow_features_raw_packets,
                               axis=1,
                               result_type='expand')

    expected_packet_features = raw_dataset.filter(regex='^client_packet|^server_packet')
    pd.testing.assert_frame_equal(expected_packet_features, output,
                                  check_less_precise=2,
                                  check_like=True,
                                  check_dtype=False)


def test_markov_augmenter(raw_dataset, mocker):
    target_class = 'ndpi_category'
    dataset = raw_dataset.filter(regex=f'raw_packet|{target_class}')
    mocker.patch('augmenter.features.TARGET_CLASS_COLUMN', target_class)
    gen_packets = oversample_raw_packets(dataset)
    assert set(dataset.columns) == set(gen_packets.columns)

