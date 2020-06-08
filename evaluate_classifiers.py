import argparse
import logging

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classifiers import read_classifier_settings, initialize_classifiers
from datasets import read_dataset
from settings import TARGET_CLASS_COLUMN
from feature_processing import Featurizer

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="configuration file, defaults to classifiers_config.yaml",
        default='classifiers_config.yaml')

    parser.add_argument(
        '--dataset',
        help='path to preprocessed .csv dataset')

    args = parser.parse_args()
    return args


def main():
    """ basic training loop example  """
    args = _parse_args()

    logger.info('Loading csv file..')

    dataset = read_dataset(args.dataset)

    df_train, df_test = train_test_split(dataset,
                                         stratify=dataset[TARGET_CLASS_COLUMN],
                                         random_state=1)
    featurizer = Featurizer()
    X_train, y_train = featurizer.fit_transform_encode(df_train)
    X_test, y_test = featurizer.transform_encode(df_test)

    classifier_settings = read_classifier_settings(args.config)
    clfs = initialize_classifiers(classifier_settings)

    for model_name, model_holder in clfs.items():
        #     fit_optimal_classifier(model_holder, X_train, y_train)
        model_holder.classifier.fit(X_train, y_train)
        y_pred = model_holder.classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=featurizer.target_encoder.classes_, digits=2))


if __name__ == '__main__':
    main()
