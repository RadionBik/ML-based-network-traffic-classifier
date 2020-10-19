from tqdm import tqdm


def iterate_batch_indexes(array, batch_size):
    iter_num = len(array) // batch_size
    for iteration in tqdm(range(iter_num + 1)):
        start_idx = iteration * batch_size
        end_idx = (iteration + 1) * batch_size
        yield start_idx, end_idx
