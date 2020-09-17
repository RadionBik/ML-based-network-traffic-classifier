# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    HfArgumentParser,
    TrainingArguments,
    set_seed, AutoModelForCausalLM, Trainer,
)

from gpt_model.generator.dataset import PretrainCollator, PretrainDataset, FinetuningDataset, PretrainDatasetWithClasses
from gpt_model.tokenizer import PacketTokenizer
from settings import FilePatterns

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. "
                    "Leave None if you want to train a model from scratch."
        },
    )
    quantizer_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The quantizer checkpoint for weights initialization. Must be provided when the model"
                    "is trained from scratch. Not used, when the model is initialized from checkpoint"
        },
    )
    model_type: Optional[str] = field(
        default=None,
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    finetune_on_class: str = field(
        default=None,
        metadata={"help": "specifies flow subset within the DF's target column to fine-tune the packet model on"}
    )
    train_with_targets: bool = field(
        default=False,
        metadata={"help": "specifies whether to include flow label as the first special token or to use a generic BOS"}
    )
    file_patterns_to_exclude: str = field(
        default='mawi',
        metadata={"help": "specifies which file patterns from the data folder to exclude, defaults to empty,"
                          " see settings.py::FilePatterns for used combinations"}
    )


def get_dataset(args: DataTrainingArguments, tokenizer: PacketTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    logger.info(f'block_size is {args.block_size} and likely unused')
    file_patterns = getattr(FilePatterns, args.file_patterns_to_exclude)
    if args.finetune_on_class:
        return FinetuningDataset(tokenizer=tokenizer,
                                 dataset_path=file_path,
                                 target_class=args.finetune_on_class)

    if args.train_with_targets:
        return PretrainDatasetWithClasses(tokenizer=tokenizer,
                                          folder_path=file_path,
                                          filename_patterns_to_exclude=file_patterns)

    return PretrainDataset(tokenizer=tokenizer, folder_path=file_path, filename_patterns_to_exclude=file_patterns)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    if data_args.finetune_on_class and data_args.train_with_targets:
        raise ValueError("Pretraining with flow labels and fine-tuning on the class simultaneously not supported.")

    if not model_args.model_name_or_path and not model_args.quantizer_path:
        raise ValueError("Either model or quantizer checkpoint path must be specified")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.model_name_or_path:
        config = GPT2Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = GPT2Config.from_json_file(model_args.config_name)
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.model_name_or_path:
        tokenizer = PacketTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = PacketTokenizer.from_pretrained(model_args.quantizer_path)

    if model_args.model_name_or_path:
        model = GPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    model.resize_token_embeddings(len(tokenizer))
    print(model)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=PretrainCollator(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
