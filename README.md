# Stunting Prevention Follow-up Planner — RL Summative

A reinforcement-learning agent that helps a **community health worker (CHW)** in rural Rwanda decide
*which child to follow up with, when to intervene, and when to reassess*, so that fewer children
become stunted. Four algorithms are trained and compared with **TorchRL** on the same custom
Gymnasium environment: **DQN, REINFORCE, PPO and A2C**.

> The simulation is loosely calibrated to Rwanda DHS 2019-20 figures and is not a clinical tool.
> The agent recommends; the CHW decides.

## Quick start

```bash
uv sync            # creates .venv with Python 3.12 and installs everything from uv.lock
uv run main.py     # runs the best trained agent in the 3D simulation
uv run pytest      # environment tests
```

## Environment: `StuntingCareEnv` (`environment/custom_env.py`)

| Item | Design |
|---|---|
| Agent | One CHW managing 6 children aged 0–24 months for 26 weeks |
| Step | One working day (3 days/week → max 78 steps) |
| Actions | `Discrete(26)`: SCREEN / COUNSEL / SUPPLEMENT / REFER × 6 children, RESTOCK, RECORD_KEEPING |
| Observation | `Box(0, 1, (76,))`: 12 features per child + 4 global features |
| Hidden state | True height-for-age z-score (HAZ), illness, caregiver adherence, intervention effects |
| Start state | Children sampled from household risk-factor prevalences; baseline measurement 0–8 weeks old |
| Terminal | A child stays below HAZ −3 for 2 weeks without referral (failure) |
| Truncation | End of the 26-week period |
| Scenarios | `standard`, `high_risk`, `low_resource` (used to test generalisation) |

### Reward
* `+10 × Σ ΔHAZ` every week (growth of all children)
* `−10` when a child becomes stunted (HAZ < −2), `+5` when a child recovers
* `+3` / `+1` for detecting a stunted child by screening early (≤ 2 weeks) / late
* `−0.5` for wasted actions, `−1` for an unnecessary referral
* `−20` on failure; at the end `+5` per non-stunted child and `−5` per stunted child

## Baselines (100 episodes, `standard` scenario)

| Policy | Mean return | Stunted at end | Failure rate |
|---|---|---|---|
| Idle (record keeping only) | −85.3 | 3.32 | 71% |
| Random | −32.2 | 1.83 | 15% |
| Greedy (highest risk score first) | +2.4 | 1.45 | 1% |

Reproduce with `uv run python -m training.baselines`.

## Project structure

```
environment/   custom_env.py (env), rendering.py (three.js visualisation)
training/      dqn_training.py, pg_training.py, baselines.py
models/        saved best models (dqn/, pg/)
logs/          training CSV logs and sweep results
assets/        figures and screenshots for the report
tests/         environment tests
```
