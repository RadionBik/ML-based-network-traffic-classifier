from evaluation_utils.modeling import convert_ipt_to_iat, evaluate_generated_traffic, flows_to_packets


def test_splitting_by_directions(raw_dataset):
    raw_dataset = raw_dataset.values
    ipt_ds = convert_ipt_to_iat(raw_dataset)
    assert ipt_ds.shape == raw_dataset.shape
    ipt_packets = flows_to_packets(ipt_ds)
    source_packets = flows_to_packets(raw_dataset)
    assert (ipt_packets[:, 0] == source_packets[:, 0]).all()


def test_smoke_evaluate_generated_traffic(raw_dataset):
    results = evaluate_generated_traffic(raw_dataset.values, raw_dataset.values)
    assert all(value == 0 for key, value in results.items() if key.startswith('KL'))
