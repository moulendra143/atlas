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
ACTION_NAME_TO_IDX = {name.lower(): idx for idx, name in enumerate(ACTIONS)}


def _parse_action_from_completion(completion) -> int | None:
    """Extract an action index from noisy GRPO completions.

    Accepts any of:
    - exact action token, e.g. <a3>
    - action name, e.g. assign_engineering_task
    - a numeric index, e.g. 3
    """
    if isinstance(completion, (list, tuple)):
        text = " ".join(str(part) for part in completion if part is not None).strip().lower()
    else:
        text = str(completion).strip().lower()
    if not text:
        return None

    # Direct token match first.
    for idx, token in enumerate(ACTION_TOKENS):
        if token.lower() in text:
            return idx

    # Action-name match next.
    for name, idx in ACTION_NAME_TO_IDX.items():
        if name in text:
            return idx

    # Fall back to a standalone integer if present.
    for raw in text.replace(",", " ").replace(";", " ").split():
        cleaned = raw.strip(".:()[]{}<>\"'")
        if cleaned.isdigit():
            idx = int(cleaned)
            if 0 <= idx < len(ACTIONS):
                return idx

    return None

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
        "Choose exactly one action name from the allowed list.\n"
        "Valid actions:\n"
        + "\n".join([f"- {idx}: {name}" for idx, name in enumerate(ACTIONS)])
        + "\n\n"
        "Current state:\n"
        f"cash={cash:.0f}, revenue={revenue:.0f}, burn={burn_rate:.0f}, "
        f"morale={morale:.1f}, progress={progress:.1f}, "
        f"csat={csat:.1f}, trust={investor_trust:.1f}, "
        f"tasks={pending_tasks:.1f}, crises={crises:.1f}, trend={market_trend:.1f}\n\n"
        "Action ID (0-12): "
    )

def verify_business_health(prompts, completions, obs_list=None, **kwargs):
    """
    Reward function for GRPO. Verifies if the chosen action led to a positive business outcome.
    Steps the environment from the ACTUAL state described in the prompt (not a fresh reset),
    making this a true environment-connected verifier as required by the hackathon guide.
    """
    rewards = []
    env = AtlasOpenEnv(preset="startup")
    obs_data = kwargs.get("obs", obs_list)  # passed from dataset via GRPOTrainer kwargs

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        # Restore the actual environment state from the dataset obs vector.
        env.core.reset()
        if obs_data is not None and i < len(obs_data):
            actual_obs = obs_data[i]
            # Manually set env state to match the obs that generated this prompt.
            state_keys = [
                "cash_balance", "revenue", "burn_rate", "employee_morale",
                "product_progress", "customer_satisfaction", "investor_trust",
                "pending_tasks", "crises", "market_trend",
            ]
            for k, v in zip(state_keys, actual_obs):
                env.core.state[k] = float(v)

        action_idx = _parse_action_from_completion(completion)

        # To prevent reward collapse (-8.0 flat) and zero gradients,
        # fallback to a random action but apply a formatting penalty.
        if action_idx is None:
            action_idx = random.randint(0, len(ACTIONS) - 1)
            format_penalty = -2.0
        else:
            format_penalty = 0.0

        _, reward, _, _, info = env.core.step(action_idx)
        # Multi-objective signal: include mandate compliance in the reward breakdown
        mandate_bonus = info.get("reward_breakdown", {}).get("mandate_compliance", 0.0)
        rewards.append(float(reward) + mandate_bonus + format_penalty)

    return rewards

def main():
    cfg = RunConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print("--- ATLAS GRPO Trainer (Verifiable RL) ---")
    if GRPOTrainer is None:
        raise RuntimeError(
            "TRL GRPO is not available in this environment. "
            "Install a TRL version that provides GRPOTrainer/GRPOConfig before running this script."
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_prompt_length=256,
        max_completion_length=10,
        num_generations=2,
        generation_batch_size=2,
        bf16=False,
        fp16=False,
        use_cpu=True,
    )

    # Generate real prompts AND obs vectors from actual environment rollouts
    print("Generating environment rollout prompts...")
    prompts = []
    obs_list = []
    env_for_data = AtlasOpenEnv(preset="startup")
    for _ in range(10):
        obs, info = env_for_data.reset()
        mandate = info.get("mandate", "Balanced Stability")
        prompts.append(_format_prompt(obs, mandate))
        obs_list.append(obs.tolist())
        # Take a few steps to get diverse states
        for _ in range(3):
            action = random.randint(0, len(ACTIONS) - 1)
            obs, _, done, _, info = env_for_data.step(action)
            if not done:
                prompts.append(_format_prompt(obs, mandate))
                obs_list.append(obs.tolist())

    from datasets import Dataset
    dummy_dataset = Dataset.from_dict({
        "prompt": prompts[:16],
        "obs": obs_list[:16],  # pass actual obs so verifier steps from correct state
    })

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[verify_business_health],
        args=grpo_config,
        train_dataset=dummy_dataset,
    )

    print("Starting GRPO Training against Atlas Startup Verifier...")
    trainer.train()
    
    # Check for degenerate learning signal (reward collapse)
    logs = trainer.state.log_history
    rewards = [l.get("reward", -8.0) for l in logs if "reward" in l]
    is_degenerate = len(rewards) > 0 and max(rewards) <= -7.9
    
    if is_degenerate:
        print("WARNING: Learning signal collapsed to constant penalty (-8.0). Skipping model save to avoid degenerate artifact.")
    else:
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"Saved GRPO optimized model to {cfg.output_dir}")

if __name__ == "__main__":
    main()
