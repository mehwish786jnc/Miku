import torch
from transformers import Trainer

class SoftTargetTrainer(Trainer):
    """Trainer class for training against soft target probabilities."""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        probs = inputs.pop("probs")
        outputs = model(**inputs)
        logits = outputs[0]
        loss = soft_binary_entropy_loss(logits, labels, probs)
        return (loss, outputs) if return_outputs else loss

def soft_binary_entropy_loss(logits, target_tokens, target_probs):
    """
    Compute the soft binary cross-entropy loss.

    Args:
        logits (torch.Tensor): Logits from the model of shape (batch_size, num_tokens, vocab_size).
        target_tokens (torch.Tensor): Target token indices of shape (batch_size, num_tokens).
        target_probs (torch.Tensor): Target probabilities of shape (batch_size, num_tokens).

    Returns:
        torch.Tensor: The computed loss.
    """
    batch_size, num_tokens, vocab_size = logits.shape
    assert target_tokens.shape == torch.Size([batch_size, num_tokens])
    assert target_probs.shape == torch.Size([batch_size, num_tokens])

    logits, target_tokens, target_probs = _shift_and_flatten_inputs(logits, target_tokens, target_probs)

    assert logits.shape == torch.Size([batch_size * (num_tokens - 1), vocab_size])
    assert target_tokens.shape == torch.Size([1, batch_size * (num_tokens - 1)])
    assert target_probs.shape == torch.Size([1, batch_size * (num_tokens - 1)])

    train_probs = torch.softmax(logits, dim=-1)
    train_probs, target_tokens, target_probs = _filter_disabled_tokens(train_probs, target_tokens, target_probs)

    assert train_probs.shape[-1] == vocab_size
    num_training_examples, _ = train_probs.shape
    assert target_tokens.shape == torch.Size([num_training_examples])
    assert target_probs.shape == torch.Size([num_training_examples])

    train_token_probs = _select_probability_by_index(train_probs, target_tokens)
    assert train_token_probs.shape == torch.Size([num_training_examples])

    # Compute the loss
    loss = -target_probs * torch.clip(torch.log(train_token_probs), min=-100)
    loss = loss - (1 - target_probs) * torch.clip(torch.log(1 - train_token_probs), min=-100)

    return loss.mean()

def _select_probability_by_index(output_probs, target_tokens):
    """Select the probabilities corresponding to target token indices."""
    target_tokens = target_tokens.unsqueeze(dim=-1)
    num_output_probs = output_probs.shape[0]
    assert target_tokens.shape == torch.Size([num_output_probs, 1])
    probs = torch.gather(output_probs, -1, target_tokens).squeeze(dim=1)
    return probs

def _filter_disabled_tokens(output_probs, target, probs):
    """Filter out tokens marked to be ignored."""
    ignore_ix = target != -100
    length = target.shape[-1]
    target = target.masked_select(ignore_ix)
    probs = probs.masked_select(ignore_ix)
    assert ignore_ix.shape == torch.Size([1, length])
    output_probs = output_probs[ignore_ix[0]]
    return output_probs, target, probs

def _shift_and_flatten_inputs(logits, target, probability):
    """Shift logits and targets to align token predictions and flatten the inputs."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    shift_probs = probability[..., 1:].contiguous()

    flattened_logits = shift_logits.view(-1, shift_logits.size(-1))
    flattened_labels = shift_labels.view(1, -1)
    flattened_probs = shift_probs.view(1, -1)
    return flattened_logits, flattened_labels, flattened_probs
