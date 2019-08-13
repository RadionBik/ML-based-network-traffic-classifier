import pcapparser


def test_pure_path_simple():
    result = pcapparser._pure_filename('hello/there/how.areyou')
    assert result == 'how'


def test_pure_path_no_ext():
    result = pcapparser._pure_filename('hello/there/how')
    assert result == 'how'


def test_outer_functions():
    result = pcapparser.ip4_from_string('127.0.0.1')
    assert result == bytes([127, 0, 0, 1])
