import pathlib

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel

from pretraining.tokenizer import PacketTokenizer
from pretraining.dataset import load_modeling_data_with_classes
from datasets.format_for_classification import save_dataset
from settings import logger


def generate_packets(protocol, n_samples, model: GPT2LMHeadModel, tokenizer, device='cpu', batch_limit=1024):
    logger.info(f'generating {n_samples} flows of "{protocol}"...')

    generated_flows = []
    tokens_to_sample = [batch_limit] * (n_samples // batch_limit)
    if n_samples % batch_limit != 0:
        # add the remainder
        tokens_to_sample += [n_samples % batch_limit]

    counter = 0
    for batch_size in tokens_to_sample:
        input_ids = torch.tensor([tokenizer.tokens_to_ids[protocol]] * batch_size, dtype=torch.long
                                 ).view(batch_size, -1).to(device)

        # no_repeat_ngram_size=1 is a dirty hack to fix duplicating pairs for 2-packet protocols
        out = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=128,
            do_sample=True,
            num_return_sequences=1,
            top_k=len(tokenizer),
            no_repeat_ngram_size=int(protocol in ['DNS', 'NTP']),
            use_cache=True,
        ).cpu()
        torch.cuda.empty_cache()
        packets = tokenizer.batch_decode(out)
        generated_flows.append(packets)
        counter += batch_size
        logger.info(f'generated {counter} flows')

    target_dim_size = max(x.shape[1] for x in generated_flows)
    # pad arrays to equal out their 2nd dim
    generated_flows = list(map(lambda x: np.pad(x, ((0, 0), (0, target_dim_size - x.shape[1])), constant_values=np.nan),
                               generated_flows))
    generated_flows = np.concatenate(generated_flows, axis=0)
    return generated_flows


def plot_packet_per_flow_hist(flows):
    non_packet_mask = ~np.isnan(flows)
    packets_per_flow = non_packet_mask.sum(1) / 2
    pd.Series(packets_per_flow).hist(bins=50)


def main():
    pretrained_path = pathlib.Path('/media/raid_store/pretrained_traffic/gpt2_model_2epochs_classes')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = PacketTokenizer.from_pretrained(pretrained_path)
    model = GPT2LMHeadModel.from_pretrained(pretrained_path).to(device)
    source_flows, classes = load_modeling_data_with_classes('/media/raid_store/pretrained_traffic/train_csv', tokenizer)
    source_class_counts = classes.value_counts()

    generated_flows_path = pretrained_path.parent / ('generated_flows_' + pretrained_path.stem)
    generated_flows_path.mkdir(exist_ok=True)
    for proto in tokenizer.tokens_to_ids.keys():
        if proto.startswith('['):
            continue
        try:
            source_class_count = source_class_counts[proto]
        except KeyError:
            logger.error(f'could not find target class "{proto}" in dataset, skipping')
            continue

        n_flows_to_generate = source_class_count if source_class_count < 20000 else 20000
        gen_flows = generate_packets(proto, n_flows_to_generate, model, tokenizer, device)
        gen_flows = pd.DataFrame(gen_flows, columns=tokenizer.packet_quantizer.raw_columns[:gen_flows.shape[1]])
        save_dataset(gen_flows, save_to=generated_flows_path / f'{proto}.csv')


if __name__ == '__main__':
    main()
