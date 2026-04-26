---
title: ATLAS
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![OpenEnv 0.2.3](https://img.shields.io/badge/OpenEnv-0.2.3-orange.svg)](https://pypi.org/project/openenv-core/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/nelluru/ATLAS)
[![TRL](https://img.shields.io/badge/TRL-SFT%20%2B%20GRPO-purple.svg)](https://github.com/huggingface/trl)

# ATLAS: AI-Driven Multi-Agent Startup Simulation Environment

> **FOR JUDGES -- Read this in 3 minutes:**
> | | |
> |---|---|
> | **Problem** | LLMs fail at long-horizon, multi-objective resource management under dynamic constraints |
> | **Environment** | 90-day CEO simulation: 10-variable state, 13 actions, 5 NPC department agents, dynamic Board Mandates |
> | **Results** | **+111% reward improvement** (random to trained). Plots, behavioral logs, and reproducible scripts included |
> | **Why it matters** | Proves LLMs can be trained for complex, multi-agent, multi-step real-world planning -- not just chat |

> **OpenEnv Hackathon 2026** -- Themes: Multi-Agent Interactions | Long-Horizon Planning | Self-Improving Agents

| Resource | Link |
|---|---|
| Live Space | https://huggingface.co/spaces/nelluru/ATLAS |
| Live App | https://nelluru-atlas.hf.space |
| Demo Video | https://youtu.be/1aWDCkJ3Uyc |
| Model Weights | https://huggingface.co/nelluru/atlas-ceo-distilgpt2 |
| Google Colab Training | [Run Training Pipeline](https://colab.research.google.com/drive/1zGZNoiwAomnLb2gpLURKu7ELrXdJv8qi) |

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Environment Design](#environment-design)
  - [Observation Space](#observation-space-10-variables)
  - [Action Space](#action-space-13-discrete-actions)
  - [Dynamic Events](#dynamic-events-10-stochastic-events)
  - [Scenario Presets](#scenario-presets)
  - [Board Mandates and Instruction Following](#board-mandates--instruction-following)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [Agent Capabilities](#agent-capabilities)
- [Tasks](#tasks)
- [Reward Signal](#reward-signal)
- [Mandate Compliance Mapping](#mandate-compliance-mapping)
- [Reward Improvement Evidence](#reward-improvement-evidence)
- [Composable Reward Rubrics](#composable-reward-rubrics-anti-hacking-design)
- [Proof of Training and Model Weights](#proof-of-training--model-weights)
- [Self-Improvement Strategy](#self-improvement-strategy)
- [OpenEnv Compliance](#openenv-compliance)
- [TRL Training Pipeline](#trl-training-pipeline-colab--rl)
- [AI CEO Mode](#ai-ceo-mode-llm-integration)
- [Installation and Setup](#installation-and-setup)
- [Run Locally](#run-locally)
- [Docker Deployment](#docker-deployment)
- [API Endpoints](#api-endpoints)
- [Real-Time Control API](#real-time-control-api)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Hackathon Guidelines Alignment](#hackathon-guidelines-alignment)
- [Minimum Requirements Checklist](#minimum-requirements-checklist)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Problem Statement

Current LLMs are good at single-turn reasoning but struggle with:
- **Multi-step strategic planning** under resource constraints
- **Instruction following**: Adapting strategies based on dynamic Board mandates (e.g., Growth vs. Cost-Cutting)
- **Recovering from crises** over long horizons (90 simulated days)

ATLAS trains an LLM-based CEO agent to navigate hiring, product launches, financial crises, and market shocks while strictly following strategic mandates. It is **not only normal fine-tuning**: after SFT initialization, policy updates continue through reinforcement learning from environment rewards.

---

## Environment Design

The ATLAS environment simulates a **90-day startup quarter** with morning, afternoon, and evening phases (270 steps per episode). An AI CEO must make strategic decisions while 5 NPC department agents react, dynamic events fire stochastically, and the Board Mandate constrains the strategy space.

### Observation Space (10 Variables)

The environment exposes a 10-dimensional continuous state vector at every step:

| # | Variable | Range | Description |
|---|---|---|---|
| 0 | `cash_balance` | 0 -- 2,000,000 | Current cash reserves ($) |
| 1 | `revenue` | 0 -- 2,000,000 | Monthly revenue ($) |
| 2 | `burn_rate` | 0 -- 2,000,000 | Monthly operational costs ($) |
| 3 | `employee_morale` | 0 -- 100 | Team happiness and engagement |
| 4 | `product_progress` | 0 -- 100 | Product development completion (%) |
| 5 | `customer_satisfaction` | 0 -- 100 | Customer satisfaction score (CSAT) |
| 6 | `investor_trust` | 0 -- 100 | Investor confidence level |
| 7 | `pending_tasks` | 0 -- 100 | Backlog of engineering tasks |
| 8 | `crises` | 0 -- 20 | Number of active crises |
| 9 | `market_trend` | -100 -- 100 | Market sentiment indicator |

All values are clamped via `_sanitize_state()` to prevent runaway values and reward hacking.

### Action Space (13 Discrete Actions)

The CEO agent selects one action per step from this discrete space:

| Index | Action | State Effects |
|---|---|---|
| 0 | `hire_employee` | burn_rate +2000, product_progress +2, morale +1 |
| 1 | `fire_employee` | burn_rate -1800, morale -5 |
| 2 | `increase_salaries` | burn_rate +3000, morale +4 |
| 3 | `assign_engineering_task` | product_progress +3, pending_tasks -1 |
| 4 | `launch_product` | revenue +7000, product_progress -5, bonus +8.0 |
| 5 | `run_ads` | burn_rate +2500, revenue +3000 |
| 6 | `negotiate_client` | revenue +5000, investor_trust +1 |
| 7 | `reduce_costs` | burn_rate -2500, morale -2 |
| 8 | `raise_funding` | cash_balance +120000, investor_trust -2 |
| 9 | `fix_bug_crisis` | customer_satisfaction +3, crises -1 |
| 10 | `improve_culture` | morale +4, burn_rate +1200 |
| 11 | `give_bonuses` | cash_balance -10000, morale +5 |
| 12 | `change_roadmap` | product_progress +1, pending_tasks +1 |

Any action index outside 0--12 receives an **invalid action penalty of -8.0**.

### Dynamic Events (10 Stochastic Events)

At each step, there is a **25% probability** of a random event firing. Each event modifies state and produces a reward signal:

| Event | State Effects | Reward |
|---|---|---|
| `server_outage` | CSAT -8, crises +1, trend -5 | -5.0 |
| `market_crash` | revenue x0.85, trust -6, trend -20 | -6.0 |
| `viral_growth` | revenue x1.25, CSAT +3, trend +15 | +8.0 |
| `key_employee_resigns` | progress -4, morale -6 | -7.0 |
| `customer_complaints_spike` | CSAT -10, crises +1, trend -8 | -6.0 |
| `investor_metrics_request` | trust -1 | -0.5 |
| `competitor_feature_launch` | CSAT -4, trend -10 | -4.0 |
| `hiring_freeze` | morale -3, tasks +2 | -3.0 |
| `lawsuit_risk` | trust -5, cash -20000 | -5.0 |
| `sales_deal_delayed` | revenue -3000, trust -2 | -3.0 |

### Scenario Presets

Three difficulty presets control initial conditions:

| Preset | Cash ($) | Revenue ($) | Burn Rate ($) | Investor Trust |
|---|---|---|---|---|
| `startup` (default) | 500,000 | 15,000 | 18,000 | 60 |
| `crisis` | 200,000 | 8,000 | 22,000 | 40 |
| `growth` | 1,200,000 | 60,000 | 50,000 | 75 |

All presets share: `employee_morale=70`, `product_progress=20`, `customer_satisfaction=65`, `pending_tasks=5`, `crises=0`, `market_trend=0`.

### Board Mandates and Instruction Following

To explicitly test the hackathon's **Instruction Following** and **Long-Horizon Planning** themes, ATLAS features a dynamic Board Mandate system:

| Mandate | Description |
|---|---|
| **Maximize Growth** | Prioritize product progress and revenue even if burn rate increases |
| **Cost Efficiency** | Minimize burn rate and preserve cash balance at all costs |
| **Balanced Stability** | Maintain a healthy balance between employee morale and revenue |

- Users can set the mandate via the **Settings UI** or the `/reset` API endpoint.
- The active mandate is injected into the LLM's system prompt and tracked in the environment.
- The `mandate_compliance` reward signal assigns bonuses or penalties based on whether the agent's actions align with the active directive.

---

## Multi-Agent Architecture

ATLAS features **5 NPC department agents** that react to CEO decisions. Each agent has its own personality, happiness, performance metrics, memory, and goals:

| Agent Role | Reacts To | Example Behavior |
|---|---|---|
| **Sales Lead** | `run_ads`, `negotiate_client` | "If marketing budget increases, I can close more leads." Performance +2 |
| **Engineering Manager** | `assign_engineering_task`, `launch_product` | "We need 3 more developers to hit roadmap." Performance +1.5, happiness -0.5 |
| **Finance Officer** | `increase_salaries`, `give_bonuses` | "Burn rate is dangerous. Reduce spending." Happiness -1 |
| **HR Recruiter** | Low morale states (`morale < 50`) | "Morale is dropping." Happiness -1 |
| **Customer Success** | Low CSAT states (`CSAT < 60`) | "Support load is rising; churn risk is increasing." Performance -1 |

Each agent maintains a **memory log** of past interactions and has stochastic happiness/performance drift, creating emergent multi-agent dynamics.

---

## Agent Capabilities

The CEO agent is expected to demonstrate the following capabilities in a partially observable, dynamic world:
- **Instruction following** by adapting policy to dynamic Board mandates (Growth, Cost-Cutting, Balanced)
- **Multi-agent coordination** across Engineering, Sales, HR, Finance, and Customer Success actors
- **Long-horizon planning** over a full 90-day quarter with delayed consequences
- **Crisis response and recovery** under shocks (outages, resignations, market events)
- **Resource-constrained optimization** balancing cash, burn, morale, trust, and growth
- **Policy adaptation** across scenario presets (`startup`, `growth`, `crisis`)

## Tasks

Each episode requires the agent to repeatedly perform structured decision tasks:
- Choose one CEO action from the 13-action discrete space at each step
- Align decisions with the specific **Board Mandate** provided at the start of the episode
- Prioritize product, hiring, finance, and customer decisions based on current state and mandate
- Recover from negative events while preventing cascading failures
- Maximize cumulative reward and terminal business health metrics
- Generalize behavior across multiple mandates, presets, and random event sequences

---

## Reward Signal

ATLAS uses **8 independent reward components** per step (multi-objective, anti-hacking):

| Component | Formula | Direction |
|---|---|---|
| `revenue_reward` | 0.00005 x revenue | + |
| `morale_reward` | 0.02 x employee_morale | + |
| `customer_reward` | 0.02 x customer_satisfaction | + |
| `trust_reward` | 0.01 x investor_trust | + |
| `burn_penalty` | -0.00004 x burn_rate | - |
| `crisis_penalty` | -0.02 x crises | - |
| `invalid_action_penalty` | -8.0 (if action out of range) | - |
| **`mandate_compliance`** | **+1.0 or -1.0** | + or - |

The `mandate_compliance` signal gives **+1.0** when the chosen action aligns with the active Board Mandate and **-1.0** when it directly opposes it. This is the **key anti-reward-hacking mechanism** -- an agent cannot spam a single action to game one metric while violating the strategic mandate.

### Mandate Compliance Mapping

The exact action-mandate alignment used by the environment:

| Mandate | Rewarded Actions (+1.0) | Penalized Actions (-1.0) |
|---|---|---|
| **Maximize Growth** | `hire_employee`, `assign_engineering_task`, `launch_product`, `run_ads`, `negotiate_client`, `raise_funding` | `fire_employee`, `reduce_costs` |
| **Cost Efficiency** | `fire_employee`, `reduce_costs`, `negotiate_client`, `fix_bug_crisis` | `hire_employee`, `increase_salaries`, `improve_culture`, `give_bonuses`, `run_ads` |
| **Balanced Stability** | `improve_culture` (+0.5), `give_bonuses` (+0.5), `fix_bug_crisis` (+0.5), `assign_engineering_task` (+0.5) | None (neutral for all others) |

---
## Reward Improvement Evidence

### Environment Baseline Reference (Random vs Heuristic, 20 episodes each)

![Reward Curve: Before vs After](training/reward_curve.png)

*X-axis: Episode number | Y-axis: Total cumulative episode reward. This plot is an environment baseline reference (random policy vs heuristic policy), not a model-training result.*

### TRL SFT: Training Evidence (before-vs-after)

#### Reward Improvement (Before vs After -- Same Axes)
![Combined Training Evidence](training/trl_combined.png)
*Left: Mean reward bar chart with std deviation -- **+111% improvement** (random baseline to trained policy). Right: Episode-by-episode comparison on identical axes. Blue shading = improvement gap.*

#### SFT Training Loss Convergence
![TRL Loss Curve](training/trl_loss_curve.png)
*X-axis: Training step | Y-axis: SFT cross-entropy loss. Loss converges from ~2.5 to ~0.6 over 30 steps, confirming the model learned to imitate environment-optimal actions.*

### TRL GRPO RL: Verifiable Reward-Driven Improvement
The GRPO RL loop connects the trained LLM directly to the **live environment** (not a static dataset). The `verify_business_health` reward function restores the exact env state and steps it with the agent's chosen action, making it a true environment-connected verifier.

Run `python training/trl_grpo_rl.py` (16 episodes, curriculum: growth to startup to crisis):

![GRPO Reward Curve](training/trl_grpo_reward_curve.png)
*X-axis: Episode | Y-axis: Total cumulative reward. Blue = per-episode, Orange = rolling avg, Gray = baseline.*

### Composable Reward Rubrics (Anti-Hacking Design)

Following OpenEnv's recommendation for composable rubrics over monolithic scoring, ATLAS uses **8 independent reward signals** that are impossible to simultaneously game:

| Rubric | Signal | Purpose |
|---|---|---|
| `revenue_reward` | +proportional to revenue | Incentivize growth |
| `morale_reward` | +proportional to morale | Prevent employee churn |
| `customer_reward` | +proportional to CSAT | Incentivize quality |
| `trust_reward` | +proportional to investor trust | Prevent reckless spending |
| `burn_penalty` | -proportional to burn rate | Stop cash waste |
| `crisis_penalty` | -per active crisis | Force crisis resolution |
| `invalid_action_penalty` | -8.0 flat | Enforce action format |
| `mandate_compliance` | +-1.0 | Follow Board instructions |

All 8 signals are visible in `info["reward_breakdown"]` at every step -- fully inspectable by judges and trainers.

## Proof of Training & Model Weights

To provide verifiable proof that the LLM weights were updated via RL, we have provided two key artifacts:

1. **[Training Behavioral Logs](TRAINING_LOGS.md)**: A side-by-side readable log comparing the AI's exact thoughts and actions in Episode 1 (Untrained, Bankrupt on Day 15) vs Episode 16 (Trained, Survived 90 Days). 
   *ðŸ’¡ **Note to Judges:** Our training script automatically appends episode summaries (rewards, steps survived) to this log file in real-time during training.*
2. **Trained Model Weights**: The final fine-tuned LoRA weights (`adapter_model.safetensors`) from our Unsloth/TRL pipeline are hosted on Hugging Face Hub. 
   - ðŸ”— **Weights Link:** [nelluru/atlas-ceo-distilgpt2](https://huggingface.co/nelluru/atlas-ceo-distilgpt2) *(Available for inference and evaluation)*.

### Run It Yourself (Auto-Logging Demo)

Judges can verify the training pipeline and auto-logging functionality in two ways:

#### 1. Cloud Execution (Google Colab)
You can view, inspect, and run our exact **TRL GRPO** training pipeline directly in the cloud.
* ðŸ”— **Google Colab Notebook:** [ATLAS RL Training Pipeline](https://colab.research.google.com/drive/1zGZNoiwAomnLb2gpLURKu7ELrXdJv8qi)

#### 2. Local Execution (Fast 2-Episode Test)
You can also verify the environment locally. As the script runs, it will dynamically append the training data directly to `TRAINING_LOGS.md`.

**Example Command (GRPO -- Primary RL Method):**
```bash
# Windows (PowerShell):
$env:ATLAS_RL_EPISODES="2"; $env:ATLAS_RL_MAX_STEPS="10"; python training/trl_grpo_rl.py

# Linux / Mac / Colab:
ATLAS_RL_EPISODES=2 ATLAS_RL_MAX_STEPS=10 python training/trl_grpo_rl.py
```
*After the script finishes, open `TRAINING_LOGS.md` to see your live test run appended at the bottom!*

---

## Self-Improvement Strategy

ATLAS is designed for recursive capability growth:
1. **Adaptive Curricula**: The environment provides scenario presets (`startup` -> `growth` -> `crisis`). Agents follow an adaptive curriculum, training on stable environments before tackling high-volatility "black swan" events.
2. **Heuristic Distillation (Expert-in-the-loop)**: The environment includes a heuristic "Expert" used to generate initial high-quality trajectories. These are distilled into the agent via TRL SFT to establish a strong baseline.
3. **Verifiable Optimization (TRL GRPO)**: The model is then optimized online inside the environment (`training/trl_grpo_rl.py`) using dense verifiable rewards and penalties from each step, taking advantage of the latest RLVR capabilities.
4. **Trajectory Filtering**: Using the reward signal, the system can filter self-play trajectories, keeping only those that exceed the reward mean of the previous iteration for the next round.

---

## OpenEnv Compliance

- `openenv.yaml` manifest at repo root (spec_version 1)
- `AtlasOpenEnv` in `env/startup_env.py` subclasses `openenv.core.Environment`
- Exposes standard Gym API: `reset()`, `step(action)`, `render()`
- Backend endpoints at both `/api/*` and `/*` (no-prefix) for OpenEnv clients:
  - `POST /reset` -- start a new episode
  - `POST /step` -- take an action, get obs + reward
  - `GET /state` -- current environment state

Environment design (first-class artifact):
1. **Observation**: numeric state vector (cash, revenue, burn, morale, progress, CSAT, trust, tasks, crises, trend)
2. **Actions**: 13 discrete CEO decisions (`ACTIONS` list in `env/startup_env.py`)
3. **Episode end**: quarter horizon reached (`max_days`) or bankruptcy (`cash_balance <= 0`) or invalid numeric state
4. **Reward**: dense business-health formula + event bonuses/penalties
  - Reward breakdown is exposed in `info["reward_breakdown"]` so reviewers and trainers can inspect each component
5. **Abuse prevention**:
  - Invalid action receives penalty and is flagged in `info["invalid_action"]`
  - State sanitization/clamping prevents runaway values and reward hacking
  - Hard episode cap (90-day horizon) prevents infinite loops

Quick verification:
```bash
python training/check_openenv.py
# Expected: OpenEnv adapter check passed.
```

---

## TRL Training Pipeline (Colab + RL)

Open [`training/TRL_Colab_Minimal.ipynb`](training/TRL_Colab_Minimal.ipynb) in Google Colab, or run locally:

```python
!git clone https://github.com/Jaswanth-arjun/atlas.git
%cd atlas
!pip install -r requirements.txt
!python training/trl_colab_minimal.py
!python training/trl_grpo_rl.py
```

Stage 1 (`trl_colab_minimal.py`):
1. Generates `(state -> action)` pairs from the live environment
2. Fine-tunes `distilgpt2` with TRL `SFTTrainer` (optionally loaded via Unsloth)
3. Evaluates reward **before vs after** training
4. Saves `training/trl_reward_curve.png` and `training/trl_loss_curve.png`

Stage 2 (`trl_grpo_rl.py`):
1. Runs model-in-the-loop verification episodes in the environment
2. Uses step rewards/penalties as a training signal
3. Updates policy via TRL `GRPOTrainer`
4. Saves RL-updated policy to `training/trl_grpo_out`
5. Uses `sshleifer/tiny-gpt2` by default for low-resource smoke runs (override with `ATLAS_RL_MODEL`)

Curriculum progression (easy -> medium -> hard):
1. **Easy**: `growth` preset with short horizon
2. **Medium**: `startup` preset with more branching and longer horizon
3. **Hard**: `crisis` preset with the longest horizon

The trainer promotes to a harder stage only after the rolling reward in the current stage is consistently above a threshold, so learning starts with early non-zero success trajectories.

Optional control:
- Set `ATLAS_RL_CURRICULUM=0` to disable curriculum and run fixed-difficulty RL.

### Validate The 3 Project Conditions (Auto)

Run:

```bash
python training/validate_project_conditions.py
```

This script automatically checks all three required conditions:
1. Step-by-step action loop exists and runs for multiple steps.
2. Success is code-verifiable using numeric reward and thresholded scoring.
3. Task is challenging but possible (random has non-zero success, stronger policy is clearly better).

The command exits with non-zero code if any condition fails.

---

## AI CEO Mode (LLM Integration)

ATLAS supports an autonomous AI CEO mode powered by several LLM providers. When an API key is provided, the backend automatically uses the LLM to make strategic decisions.

### Supported Models & Environment Variables

| Provider | Model Used | Environment Variable |
|---|---|---|
| **Google Gemini** | `gemini-1.5-flash` | `GEMINI_API_KEY` |
| **OpenAI** | `gpt-3.5-turbo` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-haiku` | `ANTHROPIC_API_KEY` |
| **Hugging Face** | `Mistral-7B-Instruct` | `HF_TOKEN` |

*Note: If multiple keys are provided, the priority is Gemini > OpenAI > Anthropic > Hugging Face.*

---

## Installation and Setup

### Prerequisites
- Python 3.11+
- Node.js 18+

### Setup Commands
```bash
# Clone repository
git clone https://github.com/Jaswanth-arjun/atlas.git
cd atlas

# Setup Python Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Setup Frontend
cd frontend
npm install
cd ..
```

## Run Locally

You can run the application locally using the provided PowerShell scripts or manually.

```powershell
# Windows (PowerShell)
.\run_backend.ps1
# In a new terminal:
.\run_frontend.ps1
```

```bash
# Linux/Mac (Bash)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# In a new terminal:
cd frontend && npm run dev
```

- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs

## Docker Deployment

To build and run ATLAS using Docker:

```bash
# Build the container
docker build -t atlas-simulation .

# Run the container (starts both backend and served frontend)
docker run -p 7860:7860 atlas-simulation
```

The application will be accessible at http://localhost:7860.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` or `/api/reset` | Start new episode `{"preset": "startup", "mandate": "..."}` |
| POST | `/step` or `/api/step` | Take action `{"action_idx": 0}` |
| GET | `/state` or `/api/state` | Current state |
| GET | `/api/leaderboard` | Episode rankings |
| GET | `/api/replay/{id}` | Replay previous episode |
| GET | `/api/investor-report/{id}` | Download PDF report of an episode |
| WS | `/ws` | WebSocket for real-time simulation streaming |

## Real-Time Control API

The simulation supports pause, resume, and speed adjustment via the following endpoints:

| Method | Path | Description |
|---|---|---|
| POST | `/pause` | Pauses the simulation loop |
| POST | `/resume` | Resumes a paused simulation |
| POST | `/speed?val=<factor>` | Sets simulation speed multiplier (0.1x to 5x) |

```bash
curl -X POST http://localhost:8000/pause
curl -X POST "http://localhost:8000/speed?val=2.0"
```

---

## Architecture Overview

```mermaid
graph TD
    A[Frontend: React Dashboard] <-->|WebSocket & REST API| B(Backend: FastAPI Server)
    B --> C{AtlasStartupEnv}
    C --> D[Environment State]
    C --> E[Reward Engine]
    C --> F[Event Generator]
    B --> G[(SQLite Logs)]
    B --> H[AI CEO / LLM Integration]
    I[OpenEnv Validator] --> B
    J[TRL GRPO Trainer] <--> C
```

## Project Structure

```text
atlas/
â”œâ”€â”€ agents/                   # NPC Agents (Engineering, Sales, etc.)
â”‚   â”œâ”€â”€ employee.py           # Employee agent logic
â”‚   â””â”€â”€ personalities.py      # Personality templates
â”œâ”€â”€ backend/                  # FastAPI app + OpenEnv endpoints
â”‚   â”œâ”€â”€ main.py               # Application entrypoint
â”‚   â”œâ”€â”€ api.py                # REST API Routes
â”‚   â”œâ”€â”€ db.py                 # SQLite database models
â”‚   â”œâ”€â”€ openenv_models.py     # Pydantic schemas for OpenEnv
â”‚   â””â”€â”€ services/             # Core backend services
â”œâ”€â”€ data/                     # Generated reports and exports
â”œâ”€â”€ docker/                   # Additional Docker configurations
â”œâ”€â”€ env/                      # Core Reinforcement Learning Environment
â”‚   â”œâ”€â”€ startup_env.py        # Gym environment + OpenEnv Adapter
â”‚   â”œâ”€â”€ events.py             # Stochastic event logic
â”‚   â””â”€â”€ presets.py            # Scenario configuration
â”œâ”€â”€ frontend/                 # React UI Dashboard
â”œâ”€â”€ training/                 # RL and Fine-Tuning Scripts
â”‚   â”œâ”€â”€ trl_grpo_rl.py        # TRL GRPO Training loop
â”‚   â”œâ”€â”€ trl_colab_minimal.py  # Supervised Fine-Tuning (SFT)
â”‚   â”œâ”€â”€ check_openenv.py      # Manifest validation
â”‚   â””â”€â”€ validate_project_conditions.py # Project compliance tester
â”œâ”€â”€ Dockerfile                # Multi-stage Docker build
â”œâ”€â”€ openenv.yaml              # OpenEnv Manifest
â””â”€â”€ requirements.txt          # Python dependencies
```

## Technology Stack

- **Backend:** Python 3.11, FastAPI, WebSocket, SQLite/SQLAlchemy
- **Frontend:** React, Zustand (Global State + Persistence), Tailwind, Recharts dashboard
- **Environment:** Gymnasium-compatible, OpenEnv adapter
- **AI Features:** Explainable AI (Decision Reasons), TRL `SFTTrainer`/`GRPOTrainer`, Unsloth
- **Training:** Hugging Face TRL (`SFTTrainer`, `GRPOTrainer`), optional Unsloth acceleration, `distilgpt2`
- **Hosting:** Docker, Hugging Face Spaces

---

## Hackathon Guidelines Alignment

This project is designed to align closely with the advanced RL guidelines outlined in the Meta OpenEnv Hackathon Participant Help Guide:

1. **Model Acts Step-by-Step:** The simulation runs across 90 days (270 distinct phases), requiring the CEO agent to make sequential, context-aware decisions at every single step.
2. **Success Checked by Code (Verifiable):** The `AtlasStartupEnv` computes rewards using a strict mathematical formula based on objective metrics (Revenue, Cash, Morale, Trust) rather than human opinion.
3. **Multiple Independent Rewards & Anti-Hacking:** To prevent reward hacking, the RL loop combines environment reward, invalid-action penalties, event penalties, and an explicit reward breakdown in `info["reward_breakdown"]` so reviewers can inspect each component.
4. **Curriculum + RL Loop:** We bootstrap with SFT (Heuristic Distillation) as recommended, then use curriculum-guided RL in `training/trl_grpo_rl.py` so the model starts from easier scenarios and only advances after it begins getting non-zero reward.

---

## Minimum Requirements Checklist

| Requirement | Status | Artifact |
|---|---|---|
| OpenEnv (latest release) `0.2.3` | âœ… | `requirements.txt`, `openenv.yaml` |
| OpenEnv manifest | âœ… | `openenv.yaml` (repo root) |
| TRL SFT training script in Colab | âœ… | `training/TRL_Colab_Minimal.ipynb`, `training/trl_colab_minimal.py` |
| TRL RL trainer (GRPO) | âœ… | `training/trl_grpo_rl.py` |
| Unsloth acceleration integrated | âœ… | `training/trl_colab_minimal.py`, `requirements.txt` |
| Training Evidence (plots) | âœ… | `reward_curve.png`, `trl_reward_curve.png`, `trl_loss_curve.png`, `trl_grpo_reward_curve.png` |
| Mini-video < 2 min | âœ… | https://youtu.be/1aWDCkJ3Uyc |
| Hosted on Hugging Face Spaces | âœ… | https://huggingface.co/spaces/nelluru/ATLAS |
| README with all links + Strategy | âœ… | This file |

---

## Troubleshooting

- **WebSocket connection fails**: Ensure the backend is running on port 8000 and that no firewall blocks the connection.
- **Simulation runs too fast/slow**: Use the speed controls in the UI or adjust the `sim_speed` value via the `/speed` endpoint.
- **Missing environment variables**: Verify that your API keys (e.g., `GEMINI_API_KEY`) are set in the Hugging Face Space under *Settings -> Variables and Secrets*.
- **Training script errors**: Check that `requirements.txt` dependencies are installed and that you are using Python 3.11+. Re-run `pip install -r requirements.txt`.

## Contributing

We welcome contributions! Please fork the repository and submit pull requests. Ensure that any new features include:
- Updated documentation in the README.
- Corresponding unit tests.
- Adjustments to the OpenEnv manifest if you add new observation/action spaces.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

## Citation

If you use ATLAS in your research, please cite:

```bibtex
@software{atlas2026,
  title = {ATLAS: Multi-Agent Startup Management Simulation},
  author = {Jaswanth Arjun},
  year = {2026},
  url = {https://github.com/Jaswanth-arjun/atlas},
}
```

---

(c) 2026 Jaswanth Arjun. All rights reserved. Licensed under Apache 2.0.
