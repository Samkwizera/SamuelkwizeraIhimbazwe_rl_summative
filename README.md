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
| Observation | `Dict`: `observation` `Box(0, 1, (77,))` (12 features per child + 5 global) and `action_mask` `MultiBinary(26)` |
| Hidden state | True height-for-age z-score (HAZ), illness, caregiver adherence, intervention effects |
| Resources | 6 supplement packs max, restock (+4) once every 4 weeks, 2 referrals per 4 weeks |
| Start state | Children sampled from household risk-factor prevalences; baseline measurement 0–8 weeks old |
| Terminal | A child stays below HAZ −3 for 2 weeks without referral (failure) |
| Truncation | End of the 26-week period |
| Scenarios | `standard`, `high_risk`, `low_resource` (used to test generalisation) |

### Reward
* `+10 × Σ ΔHAZ` every week (growth of all children)
* `−10` when a child becomes stunted (HAZ < −2), `+5` when a child recovers
* `+3` / `+1` for detecting a stunted child by screening early (≤ 2 weeks) / late
* `−0.5` for an action that is not currently possible (agents never take these because of the mask), `−2` for an unnecessary referral
* `−20` on failure; at the end `+5` per non-stunted child and `−5` per stunted child

## Baselines (100 episodes, `standard` scenario)

| Policy | Mean return | Stunted at end | Failure rate |
|---|---|---|---|
| Idle (record keeping only) | −84.1 | 3.24 | 63% |
| Random | −43.9 | 1.69 | 12% |
| Random among valid actions | −24.8 | 1.47 | 8% |
| Greedy (highest risk score first) | −0.7 | 1.29 | 0% |

Reproduce with `uv run python -m training.baselines`.

## Results

Every algorithm was trained for 300k environment steps per run, with 10–12 hyperparameter runs each (43 runs total).
The best run of each was then tested on 100 unseen episodes (seed 60000+) in every scenario.

| Policy | Best run | Standard return | Stunted (of 6) | High risk return | Low resource return |
|---|---|---|---|---|---|
| **PPO** | ppo_11 | **+5.3** | **0.63** | **−13.8** | **−0.1** |
| REINFORCE | reinforce_04 | +1.8 | 0.75 | −20.6 | −0.9 |
| A2C | a2c_04 | −0.1 | 0.88 | −31.5 | −9.5 |
| DQN | dqn_02 | −6.9 | 1.00 | −37.0 | −18.2 |
| Greedy rule | – | +2.7 | 1.05 | −33.8 | −4.1 |
| Random (valid) | – | −25.5 | 1.42 | −48.3 | −29.0 |

All runs and their settings are in `logs/<algo>_sweep.csv`, and the figures are in `assets/`.

## Running things

```bash
uv run main.py                          # best agent (PPO) in the 3D village, opens the browser
uv run main.py --algo dqn               # a specific algorithm: dqn, reinforce, ppo, a2c
uv run main.py --scenario high_risk     # harder scenario (also low_resource)
uv run main.py --policy greedy          # a baseline instead of a trained agent
uv run main.py --no-render              # terminal output only
uv run advisor.py                       # top-3 recommendations for one caseload
uv run advisor.py --serve               # same thing as a JSON API on POST /recommend

uv run python -m training.dqn_training --runs 1,2     # train selected DQN runs
uv run python -m training.pg_training --algo ppo      # train all PPO runs
uv run python -m training.run_sweep                   # every run of every algorithm in parallel
uv run python -m training.evaluate                    # test best models, writes models/best_models.json
uv run python -m training.plots                       # figures in assets/
```

`play.py` does the same as `main.py`.

## Project structure

```
environment/   custom_env.py (env), rendering.py + web/ (three.js visualisation)
training/      dqn_training.py, pg_training.py, common.py, run_sweep.py, evaluate.py, plots.py, baselines.py
models/        trained models (dqn/, pg/) and best_models.json
logs/          per-run CSV logs and sweep results
assets/        figures and screenshots for the report
tests/         environment tests
main.py        runs the trained agent with the 3D visualisation
advisor.py     wraps a trained model as a recommendation service for the CHW
```
