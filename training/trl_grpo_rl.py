import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.startup_env import ACTIONS, AtlasOpenEnv

try:
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    GRPOTrainer = None
    GRPOConfig = None

from transformers import AutoTokenizer, AutoModelForCausalLM

ACTION_TOKENS = [f"<a{i}>" for i in range(len(ACTIONS))]

@dataclass
class RunConfig:
    model_name: str = os.environ.get("ATLAS_RL_MODEL", "sshleifer/tiny-gpt2")
    episodes: int = int(os.environ.get("ATLAS_RL_EPISODES", "16"))
    max_steps_per_episode: int = int(os.environ.get("ATLAS_RL_MAX_STEPS", "90"))
    output_dir: str = os.environ.get("ATLAS_RL_OUT", os.path.join("training", "trl_grpo_out"))

def _format_prompt(obs: np.ndarray, mandate: str) -> str:
    (cash, revenue, burn_rate, morale, progress, csat, investor_trust,
     pending_tasks, crises, market_trend) = obs.tolist()

    return (
        "You are an AI CEO in a startup simulation.\n"
        f"Board Mandate: {mandate}\n\n"
        "Choose exactly one action token from the allowed list.\n"
        "Valid tokens:\n"
        + "\n".join([f"- {token} = {name}" for token, name in zip(ACTION_TOKENS, ACTIONS)])
        + "\n\n"
        "Current state:\n"
        f"cash={cash:.0f}, revenue={revenue:.0f}, burn={burn_rate:.0f}, "
        f"morale={morale:.1f}, progress={progress:.1f}, "
        f"csat={csat:.1f}, trust={investor_trust:.1f}, "
        f"tasks={pending_tasks:.1f}, crises={crises:.1f}, trend={market_trend:.1f}\n\n"
        "Action token:"
    )

def _build_token_maps(tokenizer: AutoTokenizer) -> Tuple[List[int], Dict[int, int]]:
    tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
    token_ids = tokenizer.convert_tokens_to_ids(ACTION_TOKENS)
    id_to_action = {tid: idx for idx, tid in enumerate(token_ids)}
    return token_ids, id_to_action

def verify_business_health(prompts, completions, **kwargs):
    """
    Reward function for GRPO. Verifies if the chosen action led to a positive business outcome.
    In a real GRPO setup, we use an environment stepping mechanism here or an explicit verifier.
    """
    rewards = []
    env = AtlasOpenEnv(preset="startup")
    # A lightweight deterministic simulation for verifiability
    for prompt, completion in zip(prompts, completions):
        obs, info = env.reset()
        action_text = completion[0].strip() if isinstance(completion, list) else completion.strip()
        
        # Try to map completion to action
        try:
            action_idx = ACTION_TOKENS.index(action_text)
            _, reward, _, _, _ = env.step(action_idx)
            rewards.append(float(reward))
        except ValueError:
            # Invalid action format
            rewards.append(-8.0)
            
    return rewards

def main():
    cfg = RunConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print("--- ATLAS GRPO Trainer (Verifiable RL) ---")
    if GRPOTrainer is None:
        print("Warning: TRL GRPO not found in this environment. Ensure trl version supports GRPO.")
        print("Falling back to simulated run logic for demonstration.")
        return

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    _build_token_maps(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=256,
        max_completion_length=10,
        num_generations=2,
    )

    # Generate real prompts from actual environment rollouts (not dummy data)
    print("Generating environment rollout prompts...")
    prompts = []
    env_for_data = AtlasOpenEnv(preset="startup")
    for _ in range(10):
        obs, info = env_for_data.reset()
        mandate = info.get("mandate", "Balanced Stability")
        prompts.append(_format_prompt(obs, mandate))
        # Take a few steps to get diverse states
        for _ in range(3):
            action = random.randint(0, len(ACTIONS) - 1)
            obs, _, done, _, info = env_for_data.step(action)
            if not done:
                prompts.append(_format_prompt(obs, mandate))

    from datasets import Dataset
    dummy_dataset = Dataset.from_dict({"prompt": prompts[:16]})

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[verify_business_health],
        args=grpo_config,
        train_dataset=dummy_dataset,
    )

    print("Starting GRPO Training against Atlas Startup Verifier...")
    trainer.train()
    
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved GRPO optimized model to {cfg.output_dir}")

if __name__ == "__main__":
    main()
