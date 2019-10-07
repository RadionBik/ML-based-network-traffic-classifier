#!/usr/bin/env python

import logging
import os
import sys
from time import time
import typing

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import yaml

REGISTERED_CLASSES = {
    cls.__name__: cls for cls in [
        MLPClassifier,
        LinearSVC,
        DecisionTreeClassifier,
        RandomForestClassifier,
        GradientBoostingClassifier,
        LogisticRegression,
        OneVsOneClassifier,
    ]
}


class ClassifierHolder:
    """ simple dataclass """
    def __init__(self, classifier, param_search_space):
        self.classifier = classifier
        self.param_search_space = param_search_space


logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
logger = logging.getLogger()

ROOT = os.path.dirname(__file__)


def read_classifier_settings() -> dict:
    """ simple wrapper around yaml.load """
    with open(os.path.join(ROOT, 'classifiers.yaml')) as f:
        settings = yaml.load(f)
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


def _instantiate_holders(settings: dict,
                         random_seed: int,
                         classes: typing.Dict[str, type]) -> typing.Dict[str, ClassifierHolder]:
    result = {}
    for key, params in settings.items():
        kwargs = params.get('params', {})
        if not params.get('norandom', False):
            kwargs['random_state'] = random_seed

        logger.debug(f'Instantiating {params["type"]} with params {kwargs}')
        if 'estimator' in kwargs:  # this works only on one level deeper. No recursion
            new_kwargs = {**kwargs['estimator']['params']}
            new_kwargs['random_state'] = random_seed
            kwargs['estimator'] = classes[kwargs['estimator']['type']](**new_kwargs)
        classifier = classes[params['type']](**kwargs)
        holder = ClassifierHolder(classifier, params.get('param_search_space', {}))
        result[key] = holder

    return result


class ClassifierEnsemble:
    def __init__(self, config,  classifier_settings, file_suffix=None):
        self._config = config
        self.random_seed = int(self._config['offline']['randomSeed'])
        _process_settings(classifier_settings)
        self.holders = _instantiate_holders(classifier_settings,
                                            random_seed=self.random_seed,
                                            classes=REGISTERED_CLASSES)
        self._suffix_for_optimized = '_opt'
        self._suffix = file_suffix or self._config['general']['fileSaverSuffix']

    def _search_classif_parameters(self, classifier_name, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y,
                                                    shuffle=True,
                                                    test_size=.1,
                                                    stratify=y,
                                                    random_state=self.random_seed)
        holder = self.holders[classifier_name]
        logger.info('Searching through %s', holder.param_search_space)
        search = GridSearchCV(holder.classifier,
                              param_grid=holder.param_search_space,
                              n_jobs=-1,
                              scoring=make_scorer(metrics.jaccard_score, average='micro'),
                              cv=3)

        start = time()
        search.fit(X_val, y_val)
        logger.info('Search took {:.2f} seconds'.format(time() - start))
        logger.info('Best parameters are {} with score {:.4f}'.format(search.best_params_, search.best_score_))

        rand_state_key = 'random_state'
        if isinstance(self.holders[classifier_name].classifier, OneVsOneClassifier):
            rand_state_key = 'estimator__random_state'
        return dict(search.best_params_, **{rand_state_key: self.random_seed})

    @property
    def enabled_classifiers(self) -> tuple:
        for name, holder in self.holders.items():
            if self._config['MLtoTest'].getboolean(name):
                yield name, holder.classifier

    def classif_filename(self, classif_name):
        opt_suffix = self._suffix_for_optimized if self.optimized(classif_name) else ''
        filename = os.path.join(self._config['general']['classifiers_folder'],
                                f'{classif_name}{opt_suffix}{self._suffix}.cla')
        return filename

    def optimized(self, classif_name):
        return self._config['MLtoOptimize'].getboolean(classif_name)

    def fit(self, X, y):
        for classif_name, classif in self.enabled_classifiers:
            if self.optimized(classif_name):
                logger.info(f'Searching parameters for {classif_name}...')
                opt_params = self._search_classif_parameters(classif_name, X, y)
                self.holders[classif_name].classifier.set_params(**opt_params)

            logger.info(f'Started fitting {classif_name}...')

            self.holders[classif_name].classifier.fit(X, y)
            joblib.dump(self.holders[classif_name].classifier,
                        self.classif_filename(classif_name))

    def load(self):
        for classif_name, classif in self.enabled_classifiers:
            self.holders[classif_name].classifier = joblib.load(self.classif_filename(classif_name))

    def predict(self, X):
        return {
            classif_name: classif.predict(X)
            for classif_name, classif in self.enabled_classifiers
        }