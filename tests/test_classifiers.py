from sklearn_classifiers import clf_utils
import settings


def test_config_parsing(classif_config):
    cfg = clf_utils.read_classifier_settings(settings.TEST_STATIC_DIR / 'classifiers_config.yaml')
    assert cfg == classif_config


def test_init_clfs(classif_config):
    clfs = clf_utils.initialize_classifiers(classif_config)
    assert clfs
