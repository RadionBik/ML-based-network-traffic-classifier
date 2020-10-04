import numpy as np
import pytest
from sklearn_classifiers.knn_cosine import cos_dist, top_k_cosine_similar, voter


@pytest.fixture()
def keys():
    return np.array([[1, 0, 0], [0.9, -0.1, 0], [1, 0, 0], [0, 1, 1]])


def test_cos_dist(keys):
    query = np.array([0, 0, 1])
    sim = cos_dist(query, keys)
    assert np.isclose(sim, np.array([1., 1., 1., 0.29289322])).all()


@pytest.mark.parametrize(
    'query,idx,top_k',
    [
        (np.array([0, 0, 1]), [[3]], 1),
        (np.array([1, 0, 0]), [[0, 2]], 2),
        (np.array([1, -0.1, 0]), [[1, 0]], 2),
        (np.array([[1, -0.1, 0], [1, 0, 0]]), [[1, 0], [0, 2]], 2)
    ]
)
def test_cos_top_k(query, idx, top_k, keys):
    top = top_k_cosine_similar(query, keys, top_k)
    assert top.tolist() == idx


def test_target_assignment(keys):
    targets = np.array([2, 2, 0, 1])
    top_2_for_3_queries = np.array([[1, 0], [2, 0], [1, 2]])
    votes = voter(targets[top_2_for_3_queries])
    assert votes.tolist() == [2, 0, 2]
