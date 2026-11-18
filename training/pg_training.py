import argparse

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MaskedCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from environment.custom_env import DAYS_PER_WEEK, N_WEEKS
from training.common import N_ACTIONS, N_ENVS, RunLogger, build_mlp, finished_episodes, make_env, set_seed

MAX_GRAD_NORM = 0.5
EVAL_EVERY = 20_000

REINFORCE_BASE = dict(lr=1e-3, gamma=0.99, baseline="value", entropy_coef=0.01, episodes_per_update=8,
                      value_coef=0.5, hidden=[128, 128])
PPO_BASE = dict(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, epochs=4, rollout=1024, minibatch=256,
                entropy_coef=0.01, value_coef=0.5, hidden=[128, 128])
A2C_BASE = dict(lr=7e-4, gamma=0.99, gae_lambda=1.0, n_steps=16, entropy_coef=0.01, value_coef=0.5,
                hidden=[128, 128])


def variants(prefix, base, changes):
    return [dict(name=f"{prefix}_{i + 1:02d}", **{**base, **c}) for i, c in enumerate(changes)]


RUNS = {
    "reinforce": variants("reinforce", REINFORCE_BASE, [
        {},
        {"lr": 3e-4},
        {"lr": 3e-3},
        {"gamma": 0.95},
        {"baseline": "none"},
        {"baseline": "mean"},
        {"entropy_coef": 0.0},
        {"entropy_coef": 0.05},
        {"episodes_per_update": 2},
        {"episodes_per_update": 16},
    ]),
    "ppo": variants("ppo", PPO_BASE, [
        {},
        {"lr": 1e-3},
        {"clip_eps": 0.1},
        {"clip_eps": 0.3},
        {"epochs": 10},
        {"rollout": 256, "minibatch": 64},
        {"rollout": 4096, "minibatch": 512},
        {"gae_lambda": 0.8},
        {"entropy_coef": 0.0},
        {"entropy_coef": 0.05},
        {"value_coef": 1.0, "hidden": [256, 256]},
    ]),
    "a2c": variants("a2c", A2C_BASE, [
        {},
        {"lr": 1e-4},
        {"lr": 3e-3},
        {"n_steps": 5},
        {"n_steps": 64},
        {"entropy_coef": 0.0},
        {"entropy_coef": 0.05},
        {"value_coef": 1.0},
        {"gae_lambda": 0.95},
        {"gamma": 0.95},
    ]),
}


def build_actor_critic(env, hidden):
    policy_net = build_mlp(N_ACTIONS, hidden)
    actor = ProbabilisticActor(
        TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys={"logits": "logits", "mask": "action_mask"},
        spec=env.action_spec,
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    critic = ValueOperator(build_mlp(1, hidden), in_keys=["observation"])
    return policy_net, actor, critic


def gradient_step(loss, params, optim):
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
    optim.step()


def discounted_returns(rewards, done, gamma):
    out = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0])
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + gamma * running * (~done[:, t])
        out[:, t] = running
    return out


def train_reinforce(cfg, total_frames, seed=0):
    set_seed(seed)
    # one parallel env per episode in the update batch
    env = make_env(seed, num_envs=cfg["episodes_per_update"])
    policy_net, actor, critic = build_actor_critic(env, cfg["hidden"])
    loss_fn = ReinforceLoss(actor, critic)
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    log = RunLogger("reinforce", cfg, total_frames)
    max_steps = N_WEEKS * DAYS_PER_WEEK

    frames, next_eval = 0, EVAL_EVERY
    while frames < total_frames:
        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
            batch = env.rollout(max_steps, actor, break_when_any_done=False)
        rewards = batch["next", "reward"].squeeze(-1)
        done = batch["next", "done"].squeeze(-1)
        # envs that failed early were auto-reset, keep only each env's first full episode
        valid = (done.cumsum(-1) - done.long()) == 0
        G = discounted_returns(rewards, done, cfg["gamma"])
        frames += int(valid.sum())
        first_done = done & valid
        log.log_episodes(frames, list(zip(batch["next", "episode_reward"][first_done].squeeze(-1).tolist(),
                                          valid.sum(-1).tolist())))

        td = batch[valid]
        G = G[valid]
        with torch.no_grad():
            value = critic(td.clone())["state_value"].squeeze(-1)
        if cfg["baseline"] == "value":
            adv = G - value
        elif cfg["baseline"] == "mean":
            adv = G - G.mean()
        else:
            adv = G
        # only rescale (no centring) so the "none" baseline really has no baseline
        adv = adv / (G.std() + 1e-8)
        td["advantage"] = adv.unsqueeze(-1)
        td["value_target"] = G.unsqueeze(-1)

        out = loss_fn(td)
        entropy = actor.get_dist(td).entropy().mean()
        loss = out["loss_actor"] - cfg["entropy_coef"] * entropy
        if cfg["baseline"] == "value":
            loss = loss + cfg["value_coef"] * out["loss_value"]
        gradient_step(loss, loss_fn.parameters(), optim)

        log.log_update(frames, policy_loss=out["loss_actor"].item(), value_loss=out["loss_value"].item(),
                       entropy=entropy.item(), train_return=log.recent_return())
        if frames >= next_eval:
            log.log_eval(frames, policy_net)
            next_eval += EVAL_EVERY

    return log.finish(policy_net, "pg")


def train_ppo(cfg, total_frames, seed=0):
    set_seed(seed)
    env = make_env(seed)
    policy_net, actor, critic = build_actor_critic(env, cfg["hidden"])
    collector = Collector(env, actor, frames_per_batch=cfg["rollout"], total_frames=total_frames,
                          auto_register_policy_transforms=False)
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic)
    loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip_eps"], entropy_coeff=cfg["entropy_coef"],
                          critic_coeff=cfg["value_coef"], normalize_advantage=True)
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["rollout"]), sampler=SamplerWithoutReplacement(),
                          batch_size=cfg["minibatch"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    log = RunLogger("ppo", cfg, total_frames)

    frames, next_eval = 0, EVAL_EVERY
    for data in collector:
        frames += data.numel()
        log.log_episodes(frames, finished_episodes(data))
        stats = []
        for _ in range(cfg["epochs"]):
            # advantages are recomputed each epoch because the critic keeps moving
            with torch.no_grad():
                gae(data)
            buffer.extend(data.reshape(-1))
            for _ in range(cfg["rollout"] // cfg["minibatch"]):
                out = loss_fn(buffer.sample())
                loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                gradient_step(loss, loss_fn.parameters(), optim)
                stats.append([out["loss_objective"].item(), out["loss_critic"].item(),
                              out["entropy"].item(), out["clip_fraction"].item()])
        s = np.mean(stats, axis=0)
        log.log_update(frames, policy_loss=s[0], value_loss=s[1], entropy=s[2], clip_fraction=s[3],
                       train_return=log.recent_return())
        if frames >= next_eval:
            log.log_eval(frames, policy_net)
            next_eval += EVAL_EVERY

    collector.shutdown()
    return log.finish(policy_net, "pg")


def train_a2c(cfg, total_frames, seed=0):
    set_seed(seed)
    env = make_env(seed)
    policy_net, actor, critic = build_actor_critic(env, cfg["hidden"])
    collector = Collector(env, actor, frames_per_batch=cfg["n_steps"] * N_ENVS, total_frames=total_frames,
                          auto_register_policy_transforms=False)
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic)
    loss_fn = A2CLoss(actor, critic, entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["value_coef"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    log = RunLogger("a2c", cfg, total_frames)

    frames, next_eval, next_log = 0, EVAL_EVERY, 2_000
    stats = []
    for data in collector:
        frames += data.numel()
        log.log_episodes(frames, finished_episodes(data))
        with torch.no_grad():
            gae(data)
        out = loss_fn(data)
        loss = sum(v for k, v in out.items() if k.startswith("loss_"))
        gradient_step(loss, loss_fn.parameters(), optim)
        # torchrl drops the entropy term when its coefficient is 0, but we still want it in the logs
        if "entropy" in out.keys():
            entropy = out["entropy"].item()
        else:
            with torch.no_grad():
                entropy = actor.get_dist(data).entropy().mean().item()
        stats.append([out["loss_objective"].item(), out["loss_critic"].item(), entropy])

        if frames >= next_log:
            s = np.mean(stats, axis=0)
            log.log_update(frames, policy_loss=s[0], value_loss=s[1], entropy=s[2], train_return=log.recent_return())
            stats = []
            next_log += 2_000
        if frames >= next_eval:
            log.log_eval(frames, policy_net)
            next_eval += EVAL_EVERY

    collector.shutdown()
    return log.finish(policy_net, "pg")


TRAINERS = {"reinforce": train_reinforce, "ppo": train_ppo, "a2c": train_a2c}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=[*TRAINERS, "all"], default="all")
    parser.add_argument("--runs", default="all", help="'all' or comma separated run numbers, e.g. 1,4,7")
    parser.add_argument("--frames", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)

    algos = list(TRAINERS) if args.algo == "all" else [args.algo]
    for algo in algos:
        runs = RUNS[algo] if args.runs == "all" else [RUNS[algo][int(i) - 1] for i in args.runs.split(",")]
        for cfg in runs:
            TRAINERS[algo](cfg, args.frames, args.seed)


if __name__ == "__main__":
    main()
