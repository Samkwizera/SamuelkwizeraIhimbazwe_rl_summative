import argparse

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

from training.common import N_ACTIONS, RunLogger, build_mlp, finished_episodes, make_env, set_seed

BASE = dict(lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64, eps_end=0.05, eps_fraction=0.3,
            target_update=500, double_dqn=False, hidden=[128, 128])

# one or two settings change per run so each effect can be read off the table
RUNS = [
    dict(name="dqn_01", **BASE),
    dict(name="dqn_02", **{**BASE, "lr": 1e-4}),
    dict(name="dqn_03", **{**BASE, "lr": 5e-3}),
    dict(name="dqn_04", **{**BASE, "gamma": 0.9}),
    dict(name="dqn_05", **{**BASE, "buffer_size": 5_000}),
    dict(name="dqn_06", **{**BASE, "batch_size": 256}),
    dict(name="dqn_07", **{**BASE, "eps_end": 0.1, "eps_fraction": 0.6}),
    dict(name="dqn_08", **{**BASE, "eps_end": 0.01, "eps_fraction": 0.1}),
    dict(name="dqn_09", **{**BASE, "target_update": 50}),
    dict(name="dqn_10", **{**BASE, "target_update": 2_000}),
    dict(name="dqn_11", **{**BASE, "double_dqn": True}),
    dict(name="dqn_12", **{**BASE, "double_dqn": True, "hidden": [256, 256], "lr": 5e-4}),
]

FRAMES_PER_BATCH = 64
UPDATES_PER_BATCH = 8
WARMUP_FRAMES = 2_048


def train(cfg, total_frames, seed=0, eval_every=20_000):
    set_seed(seed)
    env = make_env(seed)
    net = build_mlp(N_ACTIONS, cfg["hidden"], nn.ReLU)
    actor = QValueActor(net, in_keys=["observation"], spec=env.action_spec_unbatched, action_mask_key="action_mask")
    egreedy = EGreedyModule(env.action_spec, eps_init=1.0, eps_end=cfg["eps_end"],
                            annealing_num_steps=int(cfg["eps_fraction"] * total_frames),
                            action_mask_key="action_mask")
    policy = TensorDictSequential(actor, egreedy)

    collector = Collector(env, policy, frames_per_batch=FRAMES_PER_BATCH, total_frames=total_frames,
                          init_random_frames=WARMUP_FRAMES, auto_register_policy_transforms=False)
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])
    loss_fn = DQNLoss(actor, action_space=env.action_spec, delay_value=True, double_dqn=cfg["double_dqn"])
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    target = HardUpdate(loss_fn, value_network_update_interval=cfg["target_update"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    log = RunLogger("dqn", cfg, total_frames)
    frames, next_eval = 0, eval_every
    losses, qvals = [], []
    for data in collector:
        frames += data.numel()
        buffer.extend(data.reshape(-1))
        log.log_episodes(frames, finished_episodes(data))
        egreedy.step(data.numel())

        if frames >= WARMUP_FRAMES:
            for _ in range(UPDATES_PER_BATCH):
                batch = buffer.sample()
                loss = loss_fn(batch)["loss"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                optim.step()
                target.step()
                losses.append(loss.item())
            qvals.append(batch["action_value"].max(-1).values.mean().item())

        if frames % 2_000 < FRAMES_PER_BATCH and losses:
            log.log_update(frames, loss=float(np.mean(losses)), mean_max_q=float(np.mean(qvals)),
                           epsilon=float(egreedy.eps), train_return=log.recent_return())
            losses, qvals = [], []
        if frames >= next_eval:
            log.log_eval(frames, net)
            next_eval += eval_every

    collector.shutdown()
    return log.finish(net, "dqn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="all", help="'all' or comma separated run numbers, e.g. 1,4,7")
    parser.add_argument("--frames", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)

    selected = RUNS if args.runs == "all" else [RUNS[int(i) - 1] for i in args.runs.split(",")]
    for cfg in selected:
        train(cfg, args.frames, args.seed)


if __name__ == "__main__":
    main()
