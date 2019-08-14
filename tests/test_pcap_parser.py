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


def test_similar_ndpi_parse(static):
    with open(static('ndpi_test_output.txt')) as file:
        raw = file.read()
        result1 = pcapparser._parse_ndpi_output(raw)
        result2 = pcapparser._parse_ndpi_output(raw)
        assert result1 == result2


def test_similar_flows(filename):
    example = filename('pcap_files/example.pcap')
    ndpi = filename('bin/ndpiReader_deb')
    flows1 = pcapparser._get_labeled_flows(ndpi, example, max_packets_per_flow=100)
    flows2 = pcapparser._get_labeled_flows(ndpi, example, max_packets_per_flow=100)
    assert flows1 == flows2


def test_similar_features(filename):
    example = filename('pcap_files/example.pcap')
    ndpi = filename('bin/ndpiReader_deb')
    flows = pcapparser._get_labeled_flows(ndpi, example, max_packets_per_flow=100)
    features = pcapparser._get_flows_features(flows)
    features2 = pcapparser._get_flows_features(flows)
    assert features.equals(features2)
