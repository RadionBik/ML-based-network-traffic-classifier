import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from gpt_model.classifier.model import GPT2Classifier
from gpt_model.classifier.dataset import ClassificationQuantizedDataset
from gpt_model.tokenizer import PacketTokenizer
from settings import BASE_DIR, DEFAULT_PACKET_LIMIT_PER_FLOW, NEPTUNE_PROJECT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dataset',
        help='path to preprocessed .csv dataset',
    )
    parser.add_argument(
        '--test_dataset',
        help='path to preprocessed .csv dataset',
    )
    parser.add_argument(
        '--pretrained_path',
    )
    parser.add_argument(
        '--freeze_pretrained_model',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--mask_first_token',
        action='store_true',
        default=False,
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
    parser.add_argument(
        '--learning_rate',
        default=None
    )
    parser.add_argument(
        '--fc_dropout',
        default=0.0,
    )
    parser.add_argument(
        '--reinitialize',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--n_layers',
        default=6,
        type=int,
        help='number of transformer layers to use, only in use when --reinitialize is provided'
    )
    parser.add_argument(
        '--log_neptune',
        dest='log_neptune',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--neptune_experiment_name',
        dest='neptune_experiment_name',
        default='gpt2_class_pretrained'
    )

    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 0.0005 if args.freeze_pretrained_model else 0.00002

    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = PacketTokenizer.from_pretrained(args.pretrained_path, flow_size=DEFAULT_PACKET_LIMIT_PER_FLOW)

    train_val_dataset = ClassificationQuantizedDataset(tokenizer,
                                                       dataset_path=args.train_dataset)
    train_part_len = int(len(train_val_dataset) * 0.9)
    train_dataset, val_dataset = random_split(train_val_dataset,
                                              [train_part_len, len(train_val_dataset) - train_part_len])

    test_dataset = ClassificationQuantizedDataset(tokenizer,
                                                  dataset_path=args.test_dataset,
                                                  label_encoder=train_val_dataset.target_encoder)

    collator = ClassificationQuantizedDataset.get_collator(mask_first_token=args.mask_first_token)

    cpu_counter = os.cpu_count()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  collate_fn=collator,
                                  num_workers=cpu_counter)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                drop_last=False,
                                shuffle=False,
                                collate_fn=collator,
                                num_workers=cpu_counter
                                )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 drop_last=False,
                                 collate_fn=collator,
                                 num_workers=cpu_counter)

    class_labels = train_val_dataset.target_encoder.classes_

    nn_classifier = GPT2Classifier(
        args,
        class_labels,
        pretrained_model_path=args.pretrained_path,
        dropout=args.fc_dropout,
        freeze_pretrained_part=args.freeze_pretrained_model,
        reinitialize=args.reinitialize,
        n_layers=args.n_layers
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=args.es_patience,
        verbose=False,
        mode='min'
    )

    logger = NeptuneLogger(
        offline_mode=not args.log_neptune,
        close_after_fit=False,
        project_name=NEPTUNE_PROJECT,
        experiment_name=args.neptune_experiment_name,
        params=vars(args),
        upload_source_files=[(BASE_DIR / 'gpt_model/classifier/model.py').as_posix()]
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
        logger=logger,
        gpus=int(device == 'cuda'),
    )

    trainer.fit(nn_classifier, train_dataloader, val_dataloader)
    trainer.test(nn_classifier, test_dataloader)
    logger.experiment.log_artifact(model_checkpoint.best_model_path)
    logger.experiment.stop()


if __name__ == '__main__':
    main()
