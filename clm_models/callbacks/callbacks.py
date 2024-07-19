import os
from transformers import AutoModelForSequenceClassification
from clm_models.callbacks import prompts, cleanup

def get_callbacks(data_args, training_args, tokenizer):
    callbacks = []

    dep_callback_args = _get_dep_callback_args(tokenizer)
    
    if data_args.eval_prompt_path:
        reward_models = _get_reward_models(data_args, training_args) if data_args.add_reward_scores else None

        callback = prompts.RecordExampleAnswersCallback(
            name='dep',
            path=data_args.eval_prompt_path,
            tokenizer=tokenizer,
            params=dep_callback_args,
            num_prompts=data_args.num_eval_prompts,
            reward_models=reward_models
        )
        callbacks.append(callback)

    if data_args.clean_enabled:
        callback = cleanup.CleanupCallback(
            pattern=os.path.join(training_args.output_dir, 'checkpoint-*', 'global_step*')
        )
        callbacks.append(callback)

    return callbacks

def _get_dep_callback_args(tokenizer):
    if "gpt" in tokenizer.name_or_path.lower():
        return {
            'temperature': 0.72,
            'repetition_penalty': 1.13125,
            'max_new_tokens': 64,
            'top_p': 0.725,
            'top_k': 0,
            'do_sample': True,
            'eos_token_id': 198,
        }
    else:
        return {
            'temperature': 0.7,
            'repetition_penalty': 1 / 0.85,
            'max_new_tokens': 128,
            'top_k': 40,
            'do_sample': True,
            'eos_token_id': 13,
        }

def _get_reward_models(data_args, training_args):
    rank = training_args.local_rank
    reward_model_names = {
        'continue_50m': 'ChaiML/reward_48m_gpt2_target_2',
        'retry_12m': 'ChaiML/gpt2_retry_12m',
        'stars_2m': 'ChaiML/3plus_stars_gpt2_reward',
        'retry_and_continue_12m': 'ChaiML/gpt2_retry_and_continue_12m_v2'
    }

    return {key: _load_reward_model(name, rank) for key, name in reward_model_names.items()}

def _load_reward_model(name, local_rank):
    model = AutoModelForSequenceClassification.from_pretrained(name, use_auth_token=True)
    return model.to(local_rank)
