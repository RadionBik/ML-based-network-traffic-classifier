from sklearn_classifiers import utils
import settings


def test_config_parsing(classif_config):
    cfg = utils.read_classifier_settings(settings.TEST_STATIC_DIR / 'classifiers_config.yaml')
    assert classif_config == cfg


def test_init_clfs(classif_config):
    clfs = utils.initialize_classifiers(classif_config)
    assert clfs