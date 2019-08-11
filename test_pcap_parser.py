import pcapparser


def test_outer_functions():
    result = pcapparser.ip4_from_string('127.0.0.1')
    assert result == bytes([127, 0, 0, 1])
