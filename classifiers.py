#!/usr/bin/env python

import argparse
import configparser
import logging
import os
import sys
from time import time

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from feature_processing import FeatureTransformer, read_csv, prepare_data
from report import ClassifierEvaluator

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
logger = logging.getLogger()


class TrafficClassifiers:
    def __init__(self, config, file_suffix=None):
        self._config = config
        self.random_seed = int(self._config['offline']['randomSeed'])
        self.parameter_search_space = {
            'LogRegr': {"C": [10, 100, 1000]},

            'SVM': {
                'estimator__C': [0.1, 1, 10],
                'estimator__loss': ['squared_hinge'],
            },

            'DecTree': {
                "max_depth": [i for i in range(5, 20) if i % 3 == 0],
                "max_features": [i for i in range(10, 40) if i % 10 == 0],
                "criterion": ["entropy"]
            },
            'RandomForest': {
                "n_estimators": [i for i in range(10, 50) if i % 10 == 0],
                "max_depth": [i for i in range(3, 16) if i % 3 == 0],
                "criterion": ["entropy"]
            },
            'GradBoost': {
                "n_estimators": [50],
                "max_depth": [i for i in range(2, 6)],
                "learning_rate": [0.01, 0.05, 0.1]
            },
            'MLP': {
                "hidden_layer_sizes": [(i, i) for i in range(80, 121) if i % 40 == 0],
                "alpha": [0.0001, 0.001, 0.01]
            }
        }

        self.classifiers = {
            'LogRegr': LogisticRegression(random_state=self.random_seed,
                                          multi_class='auto',
                                          solver='lbfgs',
                                          max_iter=200,
                                          n_jobs=-1),
            'SVM':
                OneVsOneClassifier(LinearSVC(random_state=self.random_seed, tol=1e-5), n_jobs=-1),
            'DecTree': DecisionTreeClassifier(random_state=self.random_seed),
            'RandomForest': RandomForestClassifier(random_state=self.random_seed),
            'GradBoost': GradientBoostingClassifier(random_state=self.random_seed),
            'MLP': MLPClassifier(random_state=self.random_seed, max_iter=300)
        }

        self._suffix_for_optimized = '_opt'
        self._suffix = file_suffix or self._config['general']['fileSaverSuffix']

    def _search_classif_parameters(self, classifier_name, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y,
                                                    shuffle=True,
                                                    test_size=.1,
                                                    stratify=y,
                                                    random_state=self.random_seed)
        search = GridSearchCV(self.classifiers[classifier_name],
                              param_grid=self.parameter_search_space[classifier_name],
                              n_jobs=-1,
                              scoring=make_scorer(metrics.jaccard_score),
                              cv=3)

        start = time()
        search.fit(X_val, y_val)
        logger.info('Search took {:.2f} seconds'.format(time() - start))
        logger.info('Best parameters are {} with score {:.4f}'.format(search.best_params_, search.best_score_))

        rand_state_key = 'random_state'
        if isinstance(self.classifiers[classifier_name], OneVsOneClassifier):
            rand_state_key = 'estimator__random_state'
        return dict(search.best_params_, **{rand_state_key: self.random_seed})

    def fit(self, X, y):
        for classif_name in self.classifiers:
            if self._config['MLtoTest'].getboolean(classif_name):
                opt_suffix = ''
                if self._config['MLtoOptimize'].getboolean(classif_name):
                    logger.info('Searching parameters for {}...'.format(classif_name))
                    opt_suffix = self._suffix_for_optimized
                    opt_params = self._search_classif_parameters(classif_name, X, y)
                    self.classifiers[classif_name].set_params(**opt_params)

                logger.info('Started fitting {}...'.format(classif_name))
                self.classifiers[classif_name].fit(X, y)
                joblib.dump(self.classifiers[classif_name],
                            self._config['general']['classifiers_folder'] + os.sep \
                            + classif_name + opt_suffix + self._suffix + '.cla')

    def load(self):
        for classif_name in self.classifiers:
            if self._config['MLtoTest'].getboolean(classif_name):
                opt_suffix = ''
                if self._config['MLtoOptimize'].getboolean(classif_name):
                    opt_suffix = self._suffix_for_optimized
                filename = os.path.join(self._config['general']['classifiers_folder'],
                                        f'{classif_name}{opt_suffix}{self._suffix}.cla')
                self.classifiers[classif_name] = joblib.load(filename)

    def predict(self, X):
        predictions = {}
        for classif_name in self.classifiers:
            if self._config['MLtoTest'].getboolean(classif_name):
                preds = self.classifiers[classif_name].predict(X)
                predictions.update({classif_name: preds})

        return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="configuration file, defaults to config.ini",
        default='config.ini')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--load-processors', action='store_true',
                       help='Override config to load processors')
    group.add_argument('--fit-processors', action='store_true',
                       help='Override config to fit processors')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--load-classifiers', action='store_true',
                       help='Override config to load classifiers')
    group.add_argument('--fit-classifiers', action='store_true',
                       help='Override config to fit classifiers')
    args = parser.parse_args()
    return args


def _get_overridden_bool_value(maybe_yes, maybe_no, config_default):
    if maybe_yes:
        return True
    if maybe_no:
        return False
    return config_default


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    logger.info('Loading csv file..')
    csv_filename = os.path.join(config['offline']['csv_folder'],
                                config['parser']['csvFileTraining'])

    min_flows_per_app = int(config['parser']['minNumberOfFlowsPerApp'])

    data = read_csv(csv_filename)
    csv_features, csv_targets = prepare_data(data, min_flows_per_app=min_flows_per_app)

    transformer = FeatureTransformer(config=config)
    classif = TrafficClassifiers(config=config)

    if _get_overridden_bool_value(args.load_processors,
                                  args.fit_processors,
                                  config['general'].get('useTrainedFeatureProcessors')):
        logger.info('Loading pretrained feature processors...')
        X_train, y_train, X_test, y_test = transformer.load_transform(csv_features, csv_targets)
    else:
        logger.info('Fitting new feature processors...')
        X_train, y_train, X_test, y_test = transformer.fit_transform(csv_features, csv_targets)

    if _get_overridden_bool_value(args.load_classifiers,
                                  args.fit_classifiers,
                                  config['general'].get('useTrainedClassifiers')):
        logger.info('Loading pretrained classifiers...')
        classif.load()
    else:
        logger.info('Fitting new classifiers...')
        classif.fit(X_train, y_train)

    predictions = classif.predict(X_test)

    logger.info('Plotting evaluation results...')
    ev = ClassifierEvaluator(config, y_test, predictions)
    ev.plot_scores()
    ev.plot_cm(transformer.le.classes_)


if __name__ == '__main__':
    main()
