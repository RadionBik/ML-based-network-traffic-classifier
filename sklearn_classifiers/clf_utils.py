#!/usr/bin/env python

import logging
import typing
from time import time

import yaml
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import settings
from .registered_classes import REGISTERED_CLASSES

logger = logging.getLogger(__file__)


class ClassifierHolder:
    """ simple dataclass """
    def __init__(self, classifier, param_search_space, shortcut_name=None):
        self.classifier = classifier
        self.name = type(classifier).__name__ if not shortcut_name else shortcut_name
        self.param_search_space = param_search_space

    def __repr__(self):
        repr_str = repr(self.classifier)
        if self.param_search_space:
            repr_str += f'\n\tsearch_space: {self.param_search_space}'
        return repr_str


def _read_config_file(config_path) -> dict:
    """ simple wrapper around yaml.load """
    with open(config_path) as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
    return settings


def _process_settings(settings: dict) -> None:
    """ In-place settings transform for ranges"""
    for key, params in settings.items():
        if 'param_search_space' in params:
            ssp = params.get('param_search_space')
            for pname, pvalue in ssp.items():
                if isinstance(pvalue, dict) and 'from' in pvalue:
                    step = pvalue.get('step', 1)
                    ssp[pname] = list(range(pvalue['from'], pvalue['till'], step))


def read_classifier_settings(config_path=None):
    if config_path is None:
        config_path = settings.BASE_DIR / 'sklearn_classifiers/config.yaml'
    config = _read_config_file(config_path)
    _process_settings(config)
    return config


def initialize_classifiers(config: dict,
                           random_seed: int = settings.RANDOM_SEED,
                           classes: typing.Dict[str, type] = REGISTERED_CLASSES) -> typing.Dict[str, ClassifierHolder]:

    result = {}
    for key, params in config.items():
        kwargs = params.get('params', {})

        logger.info(f'Instantiating {params["type"]} with params {kwargs}')
        if 'estimator' in kwargs:  # this works only on one level deeper. No recursion
            sub_kwargs = {'random_state': random_seed}
            kwargs['estimator'] = classes[kwargs['estimator']['type']](**sub_kwargs)
        else:
            kwargs['random_state'] = random_seed

        if params['type'].startswith('KNeighbors'):
            kwargs.pop('random_state')
        classifier = classes[params['type']](**kwargs)
        holder = ClassifierHolder(classifier, params.get('param_search_space', {}), shortcut_name=key)
        result[key] = holder
    return result


def fit_optimal_classifier(classifier: ClassifierHolder, X_train, y_train):
    """ searches through pre-defined parameter space from the .yaml, and fits classifier with found parameters """
    logger.info('Searching parameters for {} through {}'.format(classifier.name, classifier.param_search_space))
    search = GridSearchCV(classifier.classifier,
                          param_grid=classifier.param_search_space,
                          n_jobs=-1,
                          scoring=make_scorer(metrics.f1_score, average='macro'),
                          cv=2,
                          refit=True,
                          verbose=1)

    start = time()
    search.fit(X_train, y_train)
    logger.info('Search took {:.2f} seconds'.format(time() - start))
    logger.info('Best parameters are {} with score {:.4f}'.format(search.best_params_, search.best_score_))
    classifier.classifier = search.best_estimator_
    return classifier
