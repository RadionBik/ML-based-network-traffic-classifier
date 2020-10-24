import argparse
import logging
import os
from functools import partial
from pprint import pprint

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader, random_split

from gpt_model.tokenizer import PacketTokenizer
from fs_net.dataset import SimpleClassificationQuantizedDataset, ClassificationPacketSizeDataset
from fs_net.model import FSNETClassifier
from settings import BASE_DIR, DEFAULT_PACKET_LIMIT_PER_FLOW, NEPTUNE_PROJECT, TARGET_CLASS_COLUMN

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dataset',
        help='path to preprocessed .csv dataset',
        required=True
    )
    parser.add_argument(
        '--test_dataset',
        help='path to preprocessed .csv dataset',
    )
    parser.add_argument(
        '--target_column',
        help='column within the .csv denoting target variable',
        default=TARGET_CLASS_COLUMN
    )
    parser.add_argument(
        "--packet_num",
        dest='packet_num',
        type=int,
        help="specify the first N packets to use for classification, "
             "defaults to settings.py:DEFAULT_PACKET_LIMIT_PER_FLOW,",
        default=DEFAULT_PACKET_LIMIT_PER_FLOW
    )
    parser.add_argument(
        "--use_packet_size_only",
        dest='use_packet_size_only',
        action='store_true',
        help="set to use only (truncated) packet size sequences instead of quantized (PS, IPT)",
        default=False
    )
    parser.add_argument(
        "--dynamic_ps_range",
        dest='dynamic_ps_range',
        help="dynamic range for PS parameter which implicitly sets Embedding layer dim, effective only along"
             "with --use_packet_size_only option",
        type=int,
        default=5000
    )
    parser.add_argument(
        '--tokenizer_path',
        help='path to the tokenizer checkpoint, defaults to the one used for tests ooops :)',
        default=BASE_DIR / 'tests/static/quantizer_checkpoint'
    )
    parser.add_argument(
        '--neptune_experiment_name',
        dest='neptune_experiment_name',
        default='FS-NET'
    )
    parser.add_argument(
        '--log_neptune',
        dest='log_neptune',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--learning_rate',
        default=0.0005
    )
    parser.add_argument(
        '--batch_size',
        default=256,
    )
    parser.add_argument(
        '--es_patience',
        default=5,
        type=int,
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    pprint(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cpu_counter = os.cpu_count()

    if args.use_packet_size_only:
        n_tokens = args.dynamic_ps_range
        ds_class = partial(ClassificationPacketSizeDataset, max_size_range=n_tokens)
    else:
        tokenizer = PacketTokenizer.from_pretrained(args.tokenizer_path,
                                                    flow_size=args.packet_num)
        n_tokens = len(tokenizer)
        ds_class = partial(SimpleClassificationQuantizedDataset, tokenizer=tokenizer)

    train_val_dataset = ds_class(dataset_path=args.train_dataset,
                                 target_column=args.target_column)
    train_part_len = int(len(train_val_dataset) * 0.9)
    train_dataset, val_dataset = random_split(train_val_dataset,
                                              [train_part_len, len(train_val_dataset) - train_part_len])

    test_dataset = ds_class(dataset_path=args.test_dataset,
                            label_encoder=train_val_dataset.target_encoder,
                            target_column=args.target_column)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=cpu_counter)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                drop_last=False,
                                shuffle=False,
                                num_workers=cpu_counter)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 drop_last=False,
                                 num_workers=cpu_counter)

    class_labels = train_val_dataset.target_encoder.classes_

    nn_classifier = FSNETClassifier(args, class_labels=class_labels, n_tokens=n_tokens)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=args.es_patience,
        verbose=False,
        mode='min'
    )

    exp_logger = NeptuneLogger(
        offline_mode=not args.log_neptune,
        close_after_fit=False,
        project_name=NEPTUNE_PROJECT,
        experiment_name=args.neptune_experiment_name,
        params=vars(args),
        upload_source_files=[(BASE_DIR / 'fs_net/model.py').as_posix()]
    )

    checkpoint_dir = f'{nn_classifier.__class__.__name__}_checkpoints'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_dir + '/{epoch}-{val_loss:.2f}-{other_metric:.2f}'
    )

    trainer = Trainer(
        early_stop_callback=early_stop_callback,
        callbacks=[LearningRateLogger()],
        checkpoint_callback=model_checkpoint,
        auto_lr_find=False,
        logger=exp_logger,
        gpus=int(device == 'cuda'),
    )

    trainer.fit(nn_classifier, train_dataloader, val_dataloader)
    trainer.test(nn_classifier, test_dataloader)
    exp_logger.experiment.log_artifact(model_checkpoint.best_model_path)
    exp_logger.experiment.stop()


if __name__ == '__main__':
    main()
