import argparse
import configparser
import logging
import os

from classifiers import ClassifierEnsemble, read_classifier_settings
from feature_processing import FeatureTransformer, read_csv, prepare_data
from report import ClassifierEvaluator


logger = logging.getLogger(__name__)


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
    classifier_settings = read_classifier_settings()
    classif = ClassifierEnsemble(config=config, classifier_settings=classifier_settings)

    if _get_overridden_bool_value(args.load_processors,
                                  args.fit_processors,
                                  config['general'].getboolean('useTrainedFeatureProcessors')):
        logger.info('Loading pretrained feature processors...')
        X_train, y_train, X_test, y_test = transformer.load_transform(csv_features, csv_targets)
    else:
        logger.info('Fitting new feature processors...')
        X_train, y_train, X_test, y_test = transformer.fit_transform(csv_features, csv_targets)

    if _get_overridden_bool_value(args.load_classifiers,
                                  args.fit_classifiers,
                                  config['general'].getboolean('useTrainedClassifiers')):
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
