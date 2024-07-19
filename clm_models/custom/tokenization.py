import torch
import numpy as np
from copy import deepcopy

def tokenize_function(examples, tokenizer, block_size, train_to_probs):
    """
    Tokenizes input and output texts, adjusts token labels, and adds token probabilities if required.

    Args:
        examples (dict): Dictionary containing 'input_text' and 'output_text'.
        tokenizer (Tokenizer): Tokenizer instance.
        block_size (int): Maximum length of tokenized sequences.
        train_to_probs (bool): Whether to include token probabilities.

    Returns:
        dict: Tokenized inputs with adjusted labels and optionally probabilities.
    """
    input_texts = examples['input_text']
    output_texts = examples['output_text']
    data = [input_ + output_ for input_, output_ in zip(input_texts, output_texts)]

    inputs = tokenizer(
        data,
        padding="max_length",
        max_length=block_size,
        truncation=True,
        return_token_type_ids=False
    )

    inputs["labels"] = deepcopy(inputs.input_ids)
    batch_size = len(inputs["labels"])

    inputs = disable_input_text_tokens(tokenizer, output_texts, inputs, batch_size)

    if train_to_probs:
        inputs = add_token_probabilities(inputs, examples, batch_size)

    return inputs

def add_token_probabilities(inputs, examples, batch_size):
    """
    Adds token probabilities to the inputs when training with soft targets.

    Args:
        inputs (dict): Tokenized inputs.
        examples (dict): Dictionary containing 'tokens' and 'logprobs'.
        batch_size (int): Number of samples in the batch.

    Returns:
        dict: Inputs with added token probabilities.
    """
    disabled_grid = torch.tensor(-100.).expand_as(torch.tensor(inputs['labels']))
    inputs["probs"] = disabled_grid.tolist()

    for batch in range(batch_size):
        tokens, logprobs = examples['tokens'][batch], examples['logprobs'][batch]
        assert len(tokens) == len(logprobs)

        for token in range(0, len(tokens)):
            if not tokens[-token - 1] == inputs['labels'][batch][-token - 1]:
                print('Probability tokens do not match output text')

            inputs["probs"][batch][-token - 1] = np.exp(logprobs[-token - 1])

    return inputs

def disable_input_text_tokens(tokenizer, output_texts, inputs, batch_size):
    """
    Disables input text tokens by setting them to -100, as we only train on output text tokens.

    Args:
        tokenizer (Tokenizer): Tokenizer instance.
        output_texts (list): List of output texts.
        inputs (dict): Tokenized inputs.
        batch_size (int): Number of samples in the batch.

    Returns:
        dict: Inputs with input text tokens disabled.
    """
    output_lengths = [len(tokenizer(output_string).input_ids) for output_string in output_texts]

    for batch in range(batch_size):
        num_input_tokens = len(inputs['labels'][batch]) - output_lengths[batch]
        for token in range(0, num_input_tokens):
            inputs["labels"][batch][token] = -100

    return inputs
