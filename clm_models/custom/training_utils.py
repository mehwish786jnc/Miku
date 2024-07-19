from functools import partial
import logging
import math
import os
import sys

from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
import datasets
import evaluate
import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from clm_models.custom import tokenization
from clm_models.custom.model_arguments import (
    ModelArguments,
    DataTrainingArguments
)

logger = logging.getLogger(__name__)
accuracy_metric = evaluate.load("accuracy")

def get_parsed_arguments():
    """
    Parses command line arguments and returns them as dataclass instances.
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if _console_args_points_to_json():
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    return args

def setup_logging(log_level):
    """
    Configures logging settings.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def get_last_checkpoint_if_exists(training_args):
    """
    Retrieves the last checkpoint if it exists in the output directory.
    """
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint

def log_training_environment_info(training_args):
    """
    Logs information about the training environment.
    """
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

def get_raw_dataset(data_args, model_args):
    """
    Loads the dataset specified by the data arguments.
    """
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True,
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            use_auth_token=True,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            use_auth_token=True,
        )
    return raw_datasets

def get_model_config(model_args):
    """
    Retrieves or creates the model configuration based on the model arguments.
    """
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    return config

def get_tokenizer(model_args):
    """
    Retrieves or creates the tokenizer based on the model arguments.
    """
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True,
        "padding_side": "left",
        "truncation_side": "left",
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if "gpt" in tokenizer.name_or_path.lower():
        tokenizer.pad_token = 50256
    return tokenizer

def get_base_model_for_finetuning(model_args, model_config):
    """
    Loads the pre-trained model for fine-tuning based on the model arguments and configuration.
    """
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=model_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True,
        )
    else:
        model = AutoModelForCausalLM.from_config(model_config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")
    return model

def get_tokenized_dataset(training_args, data_args, raw_datasets, tokenizer):
    """
    Tokenizes the dataset using the specified tokenizer and arguments.
    """
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    block_size = _get_block_size(data_args, tokenizer)

    tokenize_function = partial(
        tokenization.tokenize_function,
        tokenizer=tokenizer,
        block_size=block_size,
        train_to_probs=data_args.train_to_probs
    )

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    return tokenized_datasets

def get_train_dataset(data_args, lm_datasets):
    """
    Retrieves the training dataset.
    """
    if "train" not in lm_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = lm_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    return train_dataset

def get_eval_dataset(data_args, lm_datasets):
    """
    Retrieves the evaluation dataset.
    """
    if "validation" not in lm_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = lm_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    return eval_dataset

def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocesses logits to avoid OOM errors and move data to CPU.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits.cpu()
    entropy = entropy_from_logits(logits)
    return (logits.argmax(dim=-1), entropy)

def entropy_from_logits(logits):
    """
    Computes entropy from logits.
    """
    logits = logits.float()
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy

def compute_metrics(eval_preds):
    """
    Computes metrics from evaluation predictions.
    """
    logits, labels = eval_preds
    preds, entropy = logits

    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)

    entropy = entropy[:, :-1].reshape(-1)
    entropy = entropy[labels != -100]

    metrics = {
        'accuracy': accuracy_metric.compute(
            predictions=preds,
            references=labels
