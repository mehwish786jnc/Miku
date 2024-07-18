
# Miku: Language Model Gym

Welcome to **Miku**, Chai Research's repository dedicated to training and deploying language models for the [Chai](https://apps.apple.com/us/app/chai-chat-with-ai-bots/id1544750895) app. Our framework is based on Reinforcement Learning from Human Feedback, incorporating reward modeling and proximal policy optimization (based on a modified version of trlx). We integrate techniques from [DeepSpeed](https://www.deepspeed.ai) and novel optimizations, aiming to enhance model performance and user interaction. Note that some referenced datasets may not be publicly available.

## Initial Setup

To get started, set up a Python 3 virtual environment and activate it:
```bash
virtualenv -p python3 env
source ./env/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Export the project to the Python path:
```bash
export PYTHONPATH=$PATH_TO_REPO
```

For testing, run `pytest` in the root directory. If needed, install `pytest` separately.

## Reward Modeling

### Repository Structure

Explore the `reward_models` folder, containing:

- `config.py`: Includes Hugging Face authentication token and Weights and Biases token (adjust environment variables or update the file as needed).
- `utils.py`: General utility functions.
- `experiments/`: Contains open-sourced experiments, recommend using 4xA40 GPUs for training.
- `custom/`: Custom callbacks, trainers instantiated from Transformer Trainer class, and training helper functions.
- `evaluation/`: Configure a best-of-N chatbot using `eval_rewards_api.py`.

### Evaluating Trained Reward Models

After training, evaluate your reward model against existing ones:

1. Create an experiment under `reward_models/evaluation/experiments`.
   - Specify `get_base_model()` (we use GPTJ) and a dictionary mapping model names to reward model objects.

2. Run `eval_rewards_api.py`:
   ```bash
   python3 -m IPython -i -- eval_rewards_api.py talk_to_models --experiment_name $NAME_OF_EXPERIMENT_FILE --select_model $ONE_OF_REWARD_MODEL_NAMES
   ```

3. Enter the model prompt and the bot's initial message.
   - View N generated responses and their corresponding reward model scores/ranks for each user input.

---
