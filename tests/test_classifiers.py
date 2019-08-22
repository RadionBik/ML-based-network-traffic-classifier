import classifiers


def test_instantiation():
    class SomeClass:
        def __init__(self, a=0, estimator=0, random_state=None):
            pass

    class SomeOtherClass:
        def __init__(self, random_state=None):
            pass

    settings = {
        'simple': {
            'type': 'SomeClass',
            'params': {
                'a': '10',
                'estimator': {
                    'type': 'SomeOtherClass',
                    'params': {}
                }
            }
        }
    }
    holders = classifiers._instantiate_holders(
        settings, 10, {SomeClass.__name__: SomeClass,
                       SomeOtherClass.__name__ : SomeOtherClass})

    assert holders
