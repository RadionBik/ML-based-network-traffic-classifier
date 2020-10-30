import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from transformers import set_seed, GPT2Model, GPT2Config

from settings import RANDOM_SEED
from sklearn_classifiers.knn_cosine import (
    cos_dist,
    top_k_cosine_similar,
    batch_voter,
    KNeighborsCosineClassifier,
    KNeighborsPuffinnClassifier,
    KNeighborsNGTClassifier
)


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
    votes = batch_voter(targets[top_2_for_3_queries])
    assert votes.tolist() == [2, 0, 2]


def test_knn_cos(keys):
    targets = np.array([2, 0, 2, 1])
    for classifier_class in [KNeighborsCosineClassifier, KNeighborsPuffinnClassifier, KNeighborsNGTClassifier]:
        clf = classifier_class(2)
        clf.fit(keys, targets)
        X_test = np.array([[0.9, 0, 0]])
        pred = clf.predict(X_test)
        assert pred.tolist() == [2]


@pytest.fixture()
def dummy_gpt2():
    set_seed(RANDOM_SEED)

    config = {
        "vocab_size": 9906,
        "n_positions": 128,
        "n_ctx": 128,
        "n_embd": 512,
        "n_layer": 6,
        "n_head": 8,
     }
    config = GPT2Config(**config)
    model = GPT2Model(config)
    return model


def test_ann_deviation(raw_dataset_with_targets, raw_dataset, tokenizer, dummy_gpt2):

    y = LabelEncoder().fit_transform(raw_dataset_with_targets['ndpi_app'])
    encoded = tokenizer.batch_encode_packets(raw_dataset)
    with torch.no_grad():
        features = dummy_gpt2(**encoded)[0]

    X = features.mean(dim=1).numpy()
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=1)
    ref_preds = KNeighborsCosineClassifier(n_neighbors=1).fit(X_train, y_train).predict(X_test)
    # ref_preds = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(X_train, y_train).predict(X_test)

    accuracy = accuracy_score(y_test, ref_preds)

    pfn_preds = KNeighborsPuffinnClassifier(n_neighbors=1).fit(X_train, y_train).predict(X_test)

    pfn_acc = accuracy_score(y_test, pfn_preds)
    assert accuracy == pfn_acc

    assert accuracy_score(ref_preds, pfn_preds) == 1.0

    ngt_preds = KNeighborsNGTClassifier(n_neighbors=1,
                                        search_epsilon=0.2,
                                        optimize_n_edges=False,
                                        optimize_search_params=False
                                        ).fit(X_train, y_train).predict(X_test)

    assert accuracy_score(ref_preds, ngt_preds) == 1.0
    assert accuracy_score(ngt_preds, pfn_preds) == 1.0
