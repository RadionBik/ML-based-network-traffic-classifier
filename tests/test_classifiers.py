import classifiers
import settings


def test_config_parsing(classif_config):
    cfg = classifiers.read_classifier_settings(settings.TEST_STATIC_DIR / 'classifiers_config.yaml')
    assert classif_config == cfg


def test_init_clfs(classif_config):
    clfs = classifiers.initialize_classifiers(classif_config)
    assert clfs