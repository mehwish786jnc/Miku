import os
import time
import json
import logging
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from contextlib import contextmanager
from transformers import TrainerCallback
from config import RESOURCES_FOLDER

logger = logging.getLogger(__name__)

@contextmanager
def timer(key):
    """Context manager to measure execution time."""
    start = time.time()
    yield
    duration = time.time() - start
    print(f'{key} took {duration:.2f}s')

class RecordExampleAnswersCallback(TrainerCallback):
    """Callback to log model responses to a fixed set of prompts."""

    def __init__(self, name, path, tokenizer, params, num_prompts=10, reward_models=None):
        self.name = name
        self.dataset = os.path.join(RESOURCES_FOLDER, 'finetune_resources', path)
        self.tokenizer = tokenizer
        self.params = params
        self.num_prompts = num_prompts
        self.reward_models = reward_models

    def on_evaluate(self, args, state, control, **kwargs):
        """Evaluate the model and log the results."""
        print('Starting prompt evaluation callback')
        model = kwargs.get('model')

        table = get_metrics(
            self.name,
            self.dataset,
            model,
            self.tokenizer,
            self.params,
            self.num_prompts,
            self.reward_models
        )

        if state.is_world_process_zero:
            step = state.global_step if state.global_step > 0 else 1
            wandb.log(table, step=step)

def get_metrics(name, dataset, model, tokenizer, params, max_prompts, reward_models):
    """Generate and log metrics for model evaluation."""
    logger.info('Generating completion table...')

    with timer('get responses'):
        examples = get_responses(dataset, model, tokenizer, params, max_prompts=max_prompts)

    metrics = {}
    df_results = pd.DataFrame({
        'summary': [e['summary'] for e in examples],
        'response': [e['response'] for e in examples],
        'prompt': [e['prompt'] for e in examples]
    })

    key = f'prompt/{name}'

    if reward_models:
        for model_name, reward_model in reward_models.items():
            scores = compute_reward_scores(reward_model, examples, tokenizer)
            mean_reward = torch.mean(torch.tensor(scores))
            metrics[f'{key}/{model_name}_mean'] = mean_reward
            df_results[model_name] = scores

    metrics[f'{key}/table'] = wandb.Table(dataframe=df_results)
    lengths = torch.tensor([float(len(e['response'])) for e in examples])
    metrics[f'{key}/response_length'] = torch.mean(lengths)

    return metrics

def compute_reward_scores(model, examples, tokenizer):
    """Compute reward scores for model responses."""
    scores = []
    for e in examples:
        text = e['prompt'] + e['response']
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(model.device)

        with torch.inference_mode():
            logits = model(**tokens).logits

        preds = [float(p[1]) for p in torch.softmax(logits, dim=1)]
        scores.append(preds[0])
    return scores

def load_prompts(path, max_prompts=10):
    """Load prompts from a file."""
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    _validate_prompts(lines)
    return lines[:max_prompts]

def _validate_prompts(js):
    """Validate the structure of the prompts."""
    for j in js:
        assert set(j) <= {'prompt', 'summary'}

def get_responses(dataset, model, tokenizer, params, max_prompts=None):
    """Generate responses for a set of prompts."""
    prompts = load_prompts(dataset, max_prompts=max_prompts)
    results = []

    for p in tqdm(prompts):
        response = generate(model, tokenizer, p['prompt'][-2048:], params)[0]
        results.append({
            'summary': p['summary'],
            'response': response,
            'prompt': p['prompt']
        })

    return results

def generate(model, tokenizer, inputs, params):
    """Generate model responses."""
    inputs = [inputs]
    input_ids = tokenizer(inputs, add_special_tokens=False, return_tensors="pt", return_attention_mask=True).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'], **params)

    responses = []
    for ins, outs in zip(inputs, outputs):
        decoded = tokenizer.decode(outs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded = decoded[len(ins):]
        responses.append(decoded.rstrip())

    return responses
