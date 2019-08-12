#!/usr/bin/env python

import argparse
import configparser
import logging
import os
from time import time
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from config_loader import ConfigInit
from feature_processing import read_csv, FeatureTransformer
from report import ClassifierEvaluator

import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class TrafficClassifiers:
    def __init__(self, config, file_suffix=None):
        self._config = config
        self.random_seed = int(self._config['offline']['randomSeed'])
        self.parameter_search_space = {
            'LogRegr': {"C": [10, 100, 1000],
                      #"tol": [0.00001,0.0001,0.001],
                      #"max_features": sp_randint(1, 11),
                      },
                      #{'alpha': [0.1, 0.01, 0.001, 0.0001]},

            'SVM': {'estimator__C': [0.1, 1, 10],
                    'estimator__loss': ['squared_hinge'],
                      #'estimator__dual': [True, False]
                    },
                    #{ 'C': [0.1, 1, 10],
                    #  'loss': ['squared_hinge'],
                      #'estimator__dual': [True, False]
                    #},
                    #{'alpha': [0.1, 0.01, 0.001, 0.0001]},
                    #[{'kernel': ['rbf'], 'gamma': [10,1,1e-2,1e-3,1e-4],
                    #    'C': [ 1, 10, 100, 1000]},
                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    #],

            'DecTree': {"max_depth": [i for i in range(5, 20) if i % 3 == 0],
                        "max_features": [i for i in range(10, 40) if i % 10 == 0],
                        "criterion": ["entropy"]},
            'RandomForest': {"n_estimators": [i for i in range(10, 50) if i % 10 == 0],
                             "max_depth": [i for i in range(3, 16) if i % 3 == 0],
                             # "max_features": sp_randint(1, 11),
                             "criterion": ["entropy"]},
            'GradBoost': {"n_estimators" : [50],
                          "max_depth":  [i for i in range(2,6)],
                          # "max_features": sp_randint(1, 11),
                          "learning_rate": [0.01, 0.05, 0.1]},
            'MLP': {"hidden_layer_sizes": [(i, i) for i in range(80, 121) if i % 40 == 0],
                    "alpha": [0.0001, 0.001, 0.01]}
        }

        self.classifiers = {
            'LogRegr':   LogisticRegression(random_state=self.random_seed,
                                            multi_class='auto',
                                            solver='lbfgs',
                                            max_iter=200,
                                            n_jobs=-1),
            #SGDClassifier(loss='log', n_jobs=-1, random_state=self.random_seed,tol=1e-4),

            'SVM':
            #SGDClassifier(loss='hinge', n_jobs=-1, random_state=self.random_seed,tol=1e-4),
            #LinearSVC(random_state=self.random_seed, tol=1e-4, dual=True),
            #SVC(random_state=self.random_seed, cache_size=1000),
            OneVsOneClassifier(LinearSVC(random_state=self.random_seed, tol=1e-5), n_jobs=-1),
            'DecTree': DecisionTreeClassifier(random_state=self.random_seed),
            'RandomForest': RandomForestClassifier(random_state=self.random_seed),
            'GradBoost': GradientBoostingClassifier(random_state=self.random_seed),
            'MLP': MLPClassifier(random_state=self.random_seed, max_iter=300)
        }

        self._suffix_for_optimized = '_opt'
        self._suffix = file_suffix if file_suffix else self._config['general']['fileSaverSuffix']

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
        return dict(search.best_params_, **{rand_state_key : self.random_seed})

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
                            self._config['general']['classifiers_folder']+os.sep\
                            + classif_name+opt_suffix+self._suffix+'.cla')

    def load(self):
        for classif_name in self.classifiers:
            if self._config['MLtoTest'].getboolean(classif_name):
                opt_suffix = ''
                if self._config['MLtoOptimize'].getboolean(classif_name):
                    opt_suffix = self._suffix_for_optimized
                self.classifiers[classif_name]=joblib.load(self._config['general']['classifiers_folder']+\
                                                           os.sep+classif_name+opt_suffix+\
                                                           self._suffix+'.cla')

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

    conf = ConfigInit(args.config).get()

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
