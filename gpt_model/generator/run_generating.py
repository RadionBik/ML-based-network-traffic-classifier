import argparse
import pathlib

import logging
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel

from flow_parsing import save_dataset
from evaluation_utils.modeling import evaluate_generated_traffic, save_metrics
from gpt_model.generator.dataset import load_modeling_data_with_classes
from gpt_model.generator.baseline import MarkovGenerator
from gpt_model.tokenizer import PacketTokenizer
from settings import FilePatterns, REPORT_DIR

logger = logging.getLogger(__name__)


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
        packets = tokenizer.batch_decode_packets(out)
        generated_flows.append(packets)
        counter += batch_size
        logger.info(f'generated {counter} flows')

    target_dim_size = max(x.shape[1] for x in generated_flows)
    # pad arrays to equal out their 2nd dim
    generated_flows = list(map(lambda x: np.pad(x, ((0, 0), (0, target_dim_size - x.shape[1])), constant_values=np.nan),
                               generated_flows))
    generated_flows = np.concatenate(generated_flows, axis=0)
    return generated_flows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dataset',
        help='path to preprocessed .csv dataset',
        default='/media/raid_store/pretrained_traffic/train_csv'
    )
    parser.add_argument(
        '--pretrained_path',
        default='/media/raid_store/pretrained_traffic/gpt2_model_4_6epochs_classes_home_iot'
    )
    parser.add_argument(
        '--flow_limit_per_app',
        default=20000,
        type=int,
    )
    parser.add_argument(
        '--filename_patterns_to_exclude',
        default='mawi',
        help='see settings.py::FilePatterns for the options'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '--markov_model',
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    filename_patterns_to_exclude = getattr(FilePatterns, args.filename_patterns_to_exclude)
    source_dataset_folder = pathlib.Path(args.source_dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_source_flows, classes = load_modeling_data_with_classes(
        source_dataset_folder,
        filename_patterns_to_exclude=filename_patterns_to_exclude
    )
    source_class_counts = classes.value_counts()

    pretrained_path = pathlib.Path(args.pretrained_path)
    tokenizer = PacketTokenizer.from_pretrained(pretrained_path)
    if not args.markov_model:
        model = GPT2LMHeadModel.from_pretrained(pretrained_path).to(device)

    generated_flows_path = pretrained_path.parent / ('generated_flows_' + pretrained_path.stem)
    if args.markov_model:
        generated_flows_path = generated_flows_path.parent / (generated_flows_path.name + '_markov')
    generated_flows_path.mkdir(exist_ok=True)
    metrics = {}
    for proto in tokenizer.tokens_to_ids.keys():
        # skip special tokens
        if proto.startswith('['):
            continue
        try:
            source_class_count = source_class_counts[proto]
        except KeyError:
            logger.error(f'could not find target class "{proto}" in dataset, skipping')
            continue

        n_flows_to_generate = source_class_count \
            if source_class_count < args.flow_limit_per_app \
            else args.flow_limit_per_app

        src_flows = all_source_flows[classes == proto]

        if args.markov_model:
            markov = MarkovGenerator()
            X = tokenizer.batch_encode_packets(src_flows.values.astype(np.float64),
                                               target_class=proto,
                                               add_special_tokens=True,
                                               return_attention_mask=False,
                                               return_tensors='np')['input_ids']

            markov.fit(X)
            gen_tokens = markov.sample(n_flows_to_generate)
            gen_flows = tokenizer.batch_decode_packets(gen_tokens)
        else:
            gen_flows = generate_packets(proto, n_flows_to_generate, model, tokenizer, device)

        gen_flows = pd.DataFrame(gen_flows, columns=tokenizer.packet_quantizer.raw_columns[:gen_flows.shape[1]])
        save_dataset(gen_flows, save_to=generated_flows_path / f'{proto}.csv')

        if args.evaluate:
            results = evaluate_generated_traffic(src_flows.values, gen_flows.values)
            metrics[proto] = results
    if args.evaluate:
        save_metrics(metrics, REPORT_DIR / ('report_' + generated_flows_path.stem + '.csv'))


if __name__ == '__main__':
    main()
