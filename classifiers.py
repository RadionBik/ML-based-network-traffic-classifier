#!/usr/bin/env python

import argparse
import configparser
import logging
import os
from time import time

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config_loader import Config_Init
from feature_processing import read_csv, FeatureTransformer
from report import ClassifierEvaluator


logger = logging.getLogger(__name__)


class TrafficClassifiers:
    def __init__(self, config):
        self._config = config
        self.random_seed = int(self._config['offline']['randomSeed'])
        self.parameter_search_space = {
            'LogRegr': {"C": [0.1, 1, 10, 100, 1000],
                        "tol": [0.00001, 0.0001, 0.001, 0.01],
                        # "max_features": sp_randint(1, 11),
                        },
            'SVM': [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-2, 1e-3, 1e-4],
                     'C': [0.01, 0.1, 1, 10, 100, 100]},
                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}],
            'DecTree': {"max_depth": [i for i in range(5, 20) if i % 3 == 1],
                        "max_features": [i for i in range(10, 100) if i % 10 == 1],
                        "criterion": ["gini", "entropy"]},
            'RandomForest': {"n_estimators": [i for i in range(10, 100) if i % 10 == 1],
                             "max_depth": [i for i in range(3, 20) if i % 3 == 0],
                             #                  #"max_features": sp_randint(1, 11),
                             "criterion": ["gini"]},
            'GradBoost': {"n_estimators": [100],
                          "max_depth": [i for i in range(1, 6)],  # sp_randint(1,6),
                          #                  #"max_features": sp_randint(1, 11),
                          "learning_rate": [0.1, 0.5, 1]},
            'MLP': {"hidden_layer_sizes": (
            [i for i in range(20, 100) if i % 20 == 1], [i for i in range(20, 100) if i % 20 == 1]),
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        }

        self.classifiers = {
            'LogRegr': LogisticRegression(random_state=self.random_seed,
                                          multi_class='auto',
                                          solver='liblinear'),
            'SVM': SVC(random_state=self.random_seed),
            'DecTree': DecisionTreeClassifier(random_state=self.random_seed),
            'RandomForest': RandomForestClassifier(random_state=self.random_seed),
            'GradBoost': GradientBoostingClassifier(random_state=self.random_seed),
            'MLP': MLPClassifier(random_state=self.random_seed)
        }

        self._suffix_for_optimized = '_optim'

    def _search_classif_parameters(self, classifier_name, X, y):

        X_tr, X_val, y_tr, y_val = train_test_split(X, y,
                                                    shuffle=True,
                                                    test_size=.1,
                                                    stratify=y)
        search = GridSearchCV(self.classifiers[classifier_name],
                              param_grid=self.parameter_search_space[classifier_name],
                              n_jobs=-1,
                              scoring=make_scorer(metrics.jaccard_score),
                              cv=3)

        start = time()
        search.fit(X_val, y_val)
        logger.info('Search took {:.2f} seconds'.format(time() - start))
        logger.info('Best parameters are {} with score {:.4f}'.format(search.best_params_,
                                                                search.best_score_))

        return dict(search.best_params_, **{'random_state': self.random_seed})

    def fit(self, X, y):
        for classif_name in self.classifiers:
            if self._config['MLtoTest'].getboolean(classif_name):
                suffix = ''
                if self._config['MLtoOptimize'].getboolean(classif_name):
                    suffix = self._suffix_for_optimized
                    logger.info('Searching parameters for {}...'.format(classif_name))
                    opt_params = self._search_classif_parameters(classif_name, X, y)
                    self.classifiers[classif_name].set_params(**opt_params)

                logger.info('Started fitting {}...'.format(classif_name))
                self.classifiers[classif_name].fit(X, y)
                joblib.dump(self.classifiers[classif_name],
                            self._config['general']['classifiers_folder'] + os.sep \
                            + classif_name + suffix + '.cla')

    def load(self):
        for classif_name in self.classifiers:
            if self._config['MLtoTest'].getboolean(classif_name):
                suffix = ''
                if self._config['MLtoOptimize'].getboolean(classif_name):
                    suffix = self._suffix_for_optimized
                self.classifiers[classif_name] = joblib.load(self._config['general']['classifiers_folder'] + \
                                                             os.sep + classif_name + suffix + '.cla')

    def predict(self, X):
        predictions = {}
        for classif_name in self.classifiers:
            if self._config['MLtoTest'].getboolean(classif_name):
                preds = self.classifiers[classif_name].predict(X)
                predictions.update({classif_name: preds})

        return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="configuration file, defaults to config.ini",
        default='config.ini')

    args = parser.parse_args()

    conf = Config_Init(args.config).get()

    logger.info('Loading csv file..')
    config = configparser.ConfigParser()
    config.read(args.config)
    csv_features, csv_targets = read_csv(config)
    extract = FeatureTransformer(config=config)
    classif = TrafficClassifiers(config=config)

    if conf['general'].getboolean('useTrainedFeatureProcessors'):
        logger.info('Loading pretrained feature processors...')
        X_train, y_train, X_test, y_test = extract.load_transform(csv_features, csv_targets)
    else:
        logger.info('Fitting new feature processors...')
        X_train, y_train, X_test, y_test = extract.fit_transform(csv_features, csv_targets)

    if conf['general'].getboolean('useTrainedClasiffiers'):
        logger.info('Loading pretrained classifiers...')
        classif.load()
    else:
        logger.info('Fitting new classifiers...')
        classif.fit(X_train, y_train)

    predictions = classif.predict(X_test)

    logger.info('Plotting evaluation results...')
    ev = ClassifierEvaluator(config, y_test, predictions)
    ev.plot_scores()
    ev.plot_cm(extract.le.classes_)


if __name__ == '__main__':
    main()
