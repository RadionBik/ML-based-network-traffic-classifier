import argparse
import logging

import neptune
from sklearn.model_selection import train_test_split
from flow_parsing import read_dataset
from evaluation_utils.classification import Reporter
from sklearn_classifiers.featurizer import Featurizer
from sklearn_classifiers.utils import read_classifier_settings, initialize_classifiers
from settings import BASE_DIR, DEFAULT_PACKET_LIMIT_PER_FLOW, NEPTUNE_PROJECT, TARGET_CLASS_COLUMN, RANDOM_SEED

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="configuration file, defaults to config.yaml",
        default=BASE_DIR / 'sklearn_classifiers/config.yaml')

    parser.add_argument(
        '--train_dataset',
        help='path to preprocessed .csv dataset',
        required=True
    )
    parser.add_argument(
        '--test_dataset',
        help='path to preprocessed .csv dataset, if not specified, 1/4 of the training dataset is selected in '
             'stratified manner',
    )
    parser.add_argument(
        '--target_column',
        help='column within the .csv denoting target variable',
        default=TARGET_CLASS_COLUMN
    )

    parser.add_argument(
        '--continuous',
        dest='continuous',
        action='store_true',
        help="when enabled, continuous feature from dataset are accounted for, "
             "e.g. percentiles, sums, etc. of packet size. Defaults to False"
    )
    parser.set_defaults(continuous=False)

    parser.add_argument(
        '--categorical',
        dest='categorical',
        action='store_true',
        help="when enabled, categorical feature from dataset are accounted for, "
             "e.g. IP protocol. Defaults to False"
    )
    parser.set_defaults(categorical=False)

    parser.add_argument(
        "--raw",
        dest='raw',
        type=int,
        help="specify the first N packet raw features to use for classification, "
             "defaults to settings.py:DEFAULT_PACKET_LIMIT_PER_FLOW,"
             "set to 0 to use only the derivatives",
        default=DEFAULT_PACKET_LIMIT_PER_FLOW
    )
    parser.add_argument(
        '--use_iat',
        help='set to use inter-packet time features, as raw features and/or their derivatives',
        action='store_true',
        default=False
    )

    parser.add_argument('--log-neptune', dest='log_neptune', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    """ basic training loop example  """
    args = _parse_args()

    logger.info('Loading csv file..')

    df_train = read_dataset(args.train_dataset, fill_na=True)
    if args.test_dataset:
        df_test = read_dataset(args.test_dataset, fill_na=True)
    else:
        df_train, df_test = train_test_split(df_train,
                                             stratify=df_train[args.target_column],
                                             test_size=1 / 4,
                                             random_state=RANDOM_SEED)

    featurizer = Featurizer(
        cont_features=None if args.continuous else [],
        categorical_features=None if args.categorical else [],
        raw_feature_num=args.raw,
        consider_j3a=False,
        consider_tcp_flags=False,
        consider_iat_features=args.use_iat,
        target_column=args.target_column,
    )
    if args.continuous:
        df_train = featurizer.calc_packets_stats_from_raw(df_train)
        df_test = featurizer.calc_packets_stats_from_raw(df_test)

    X_train, y_train = featurizer.fit_transform_encode(df_train)
    X_test, y_test = featurizer.transform_encode(df_test)

    classifier_settings = read_classifier_settings(args.config)
    clfs = initialize_classifiers(classifier_settings)

    for model_name, model_holder in clfs.items():
        #     fit_optimal_classifier(model_holder, X_train, y_train)
        model_holder.classifier.fit(X_train, y_train)
        y_pred = model_holder.classifier.predict(X_test)
        reporter = Reporter(y_test, y_pred, model_holder.name, featurizer.target_encoder.classes_)

        report_file = f'report_{model_holder.name}.csv'
        report = reporter.clf_report(save_to=report_file)
        print(report)

        if args.log_neptune:
            neptune.init(NEPTUNE_PROJECT)
            parameters = vars(args)
            parameters.update({'classifier': model_name})
            parameters.update(classifier_settings[model_name]['params'])

            neptune.create_experiment(name='sklearn', params=parameters)
            neptune.log_artifact((reporter.save_dir / report_file).as_posix())
            neptune.log_image('confusion_matrix', reporter.plot_conf_matrix())
            for metric_name, metric_value in reporter.scores().items():
                neptune.log_metric(metric_name, metric_value)

            neptune.stop()


if __name__ == '__main__':
    main()
