import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchrl.envs import (Compose, DTypeCastTransform, GymWrapper, RewardScaling, RewardSum, StepCounter,
                          TransformedEnv)
from torchrl.modules import MLP

from environment import StuntingCareEnv

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"

# evaluation seeds are never used for training episodes
EVAL_SEED = 50_000
N_ENVS = 8
REWARD_SCALE = 0.1
OBS_DIM = StuntingCareEnv().observation_space["observation"].shape[0]
N_ACTIONS = int(StuntingCareEnv().action_space.n)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed=0, scenario="standard", num_envs=N_ENVS):
    # stepping several copies at once cuts torchrl's per-step overhead about 4x
    vec = gym.vector.SyncVectorEnv([lambda: StuntingCareEnv(scenario=scenario) for _ in range(num_envs)],
                                   autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    env = GymWrapper(vec, categorical_action_encoding=True)
    # returns range roughly -100..+60, scaling keeps critic gradients from swamping the actor's
    # (RewardSum comes first so logged episode returns stay in raw units)
    env = TransformedEnv(env, Compose(
        StepCounter(),
        RewardSum(),
        RewardScaling(loc=0.0, scale=REWARD_SCALE),
        DTypeCastTransform(torch.int8, torch.bool, in_keys=["action_mask"]),
    ))
    env.set_seed(seed)
    return env


def build_mlp(out_dim, hidden, activation=nn.Tanh):
    return MLP(in_features=OBS_DIM, out_features=out_dim, num_cells=list(hidden), activation_class=activation)


def finished_episodes(data):
    done = data["next", "done"].squeeze(-1)
    returns = data["next", "episode_reward"][done].squeeze(-1).tolist()
    lengths = data["next", "step_count"][done].squeeze(-1).tolist()
    return list(zip(returns, lengths))


def masked_argmax(net, obs):
    scores = net(torch.as_tensor(obs["observation"]))
    scores[~torch.as_tensor(obs["action_mask"], dtype=torch.bool)] = -torch.inf
    return int(scores.argmax())


@torch.no_grad()
def evaluate(net, episodes=20, scenario="standard", seed=EVAL_SEED):
    """Greedy rollout of a Q-network or policy-logits network on held-out seeds."""
    env = StuntingCareEnv(scenario=scenario)
    returns, stunted, failures, wasted = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total, n_wasted = False, 0.0, 0
        while not done:
            action = masked_argmax(net, obs)
            obs, r, term, trunc, info = env.step(action)
            total += r
            n_wasted += info["wasted"]
            done = term or trunc
        returns.append(total)
        stunted.append(info["n_stunted"])
        failures.append(term)
        wasted.append(n_wasted)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_stunted": float(np.mean(stunted)),
        "failure_rate": float(np.mean(failures)),
        "mean_wasted": float(np.mean(wasted)),
    }


class RunLogger:
    def __init__(self, algo, cfg, total_frames):
        self.algo = algo
        self.cfg = cfg
        self.name = cfg["name"]
        self.total_frames = total_frames
        self.dir = LOG_DIR / algo / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episodes, self.updates, self.evals = [], [], []
        self.best = None
        self.best_state = None
        self.start = time.time()

    def log_episodes(self, frames, eps):
        for ret, length in eps:
            self.episodes.append({"frames": frames, "episode": len(self.episodes) + 1, "return": ret, "length": length})

    def log_update(self, frames, **values):
        self.updates.append({"frames": frames, **values})

    def recent_return(self, n=50):
        if not self.episodes:
            return float("nan")
        return float(np.mean([e["return"] for e in self.episodes[-n:]]))

    def log_eval(self, frames, net):
        result = evaluate(net)
        self.evals.append({"frames": frames, "episodes": len(self.episodes), **result})
        if self.best is None or result["mean_return"] > self.best["mean_return"]:
            self.best = {"frames": frames, **result}
            self.best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(
            f"[{self.algo} {self.name}] frames {frames:>7} | train return {self.recent_return():7.1f} "
            f"| eval {result['mean_return']:7.1f} stunted {result['mean_stunted']:.2f} "
            f"fail {result['failure_rate']:.2f} | {time.time() - self.start:5.0f}s",
            flush=True,
        )

    def finish(self, net, model_subdir):
        pd.DataFrame(self.episodes).to_csv(self.dir / "episodes.csv", index=False)
        pd.DataFrame(self.updates).to_csv(self.dir / "updates.csv", index=False)
        pd.DataFrame(self.evals).to_csv(self.dir / "evals.csv", index=False)

        net.load_state_dict(self.best_state)
        final = evaluate(net, episodes=50)
        model_dir = MODEL_DIR / model_subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"algo": self.algo, "config": self.cfg, "state_dict": self.best_state,
             "obs_dim": OBS_DIM, "n_actions": N_ACTIONS},
            model_dir / f"{self.name}.pt",
        )

        hparams = {k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in self.cfg.items()}
        row = {
            **hparams,
            "frames": self.total_frames,
            "episodes": len(self.episodes),
            "best_at_frames": self.best["frames"],
            "final_train_return": self.recent_return(100),
            **{f"eval_{k}": v for k, v in final.items()},
            "minutes": round((time.time() - self.start) / 60, 1),
        }
        summary = LOG_DIR / f"{self.algo}_sweep.csv"
        old = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old["name"] != self.name]
        pd.concat([old, pd.DataFrame([row])]).sort_values("name").to_csv(summary, index=False)
        print(f"[{self.algo} {self.name}] done: eval return {final['mean_return']:.1f} "
              f"stunted {final['mean_stunted']:.2f} ({row['minutes']} min)", flush=True)
        return row


def load_policy(path):
    ckpt = torch.load(path, weights_only=False)
    hidden = ckpt["config"].get("hidden", [128, 128])
    activation = nn.ReLU if ckpt["algo"] == "dqn" else nn.Tanh
    net = build_mlp(ckpt["n_actions"], hidden, activation)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
