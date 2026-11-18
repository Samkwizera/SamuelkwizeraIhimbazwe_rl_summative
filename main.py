import argparse
import json

import numpy as np

from environment import StuntingCareEnv
from environment.custom_env import DAYS_PER_WEEK
from training.baselines import BASELINES
from training.common import MODEL_DIR, ROOT, load_policy, masked_argmax

ALGOS = ("best", "dqn", "reinforce", "ppo", "a2c")


def pick_model(algo):
    best_file = MODEL_DIR / "best_models.json"
    if not best_file.exists():
        raise SystemExit("models/best_models.json not found, run `uv run python -m training.evaluate` first")
    best = json.loads(best_file.read_text())
    if algo == "best":
        algo = best["overall"]
    return algo, best[algo]


def print_children(env):
    s = env.snapshot()
    print("      child  age   measured HAZ  true HAZ  risk   status")
    for i in range(env.n_children):
        flags = []
        if s["supplement_weeks"][i]:
            flags.append(f"supplement {s['supplement_weeks'][i]}w")
        if s["treatment_weeks"][i]:
            flags.append("in treatment")
        if s["ill"][i]:
            flags.append("ill")
        if s["true_haz"][i] < -2:
            flags.append("STUNTED")
        print(f"      {i + 1:>5}  {s['age_months'][i]:4.0f}m  {s['measured_haz'][i]:+12.2f}  {s['true_haz'][i]:+8.2f}"
              f"  {s['risk_score'][i]:.2f}   {', '.join(flags)}")


def main():
    parser = argparse.ArgumentParser(description="Run the trained CHW agent in the 3D village simulation")
    parser.add_argument("--algo", choices=ALGOS, default="best")
    parser.add_argument("--policy", choices=["agent", *BASELINES], default="agent",
                        help="'agent' uses the trained model, the others are the baselines")
    parser.add_argument("--scenario", choices=["standard", "high_risk", "low_resource"], default="standard")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between steps in the 3D view")
    parser.add_argument("--no-render", action="store_true", help="terminal output only")
    parser.add_argument("--no-browser", action="store_true", help="start the 3D view without opening a browser")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env = StuntingCareEnv(scenario=args.scenario, render_mode=render_mode,
                          render_options=dict(frame_delay=args.delay, open_browser=not args.no_browser))

    if args.policy == "agent":
        algo, entry = pick_model(args.algo)
        net, ckpt = load_policy(ROOT / entry["path"])
        print(f"Loaded {algo.upper()} model {ckpt['config']['name']} from {entry['path']}")
        print(f"  held-out evaluation: return {entry['mean_return']:.1f}, {entry['mean_stunted']:.2f} children stunted")
        choose = lambda obs: masked_argmax(net, obs)
    else:
        rng = np.random.default_rng(args.seed)
        print(f"Running baseline policy: {args.policy}")
        choose = lambda obs: BASELINES[args.policy](env, rng)

    totals = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        print(f"\n=== Episode {ep + 1}/{args.episodes} | scenario {args.scenario} | seed {args.seed + ep} ===")
        print_children(env)
        total, done = 0.0, False
        while not done:
            obs, reward, term, trunc, info = env.step(choose(obs))
            total += reward
            week, day = (info["step"] - 1) // DAYS_PER_WEEK + 1, (info["step"] - 1) % DAYS_PER_WEEK + 1
            print(f"W{week:>2} D{day} | {info['action_name']:<20} | reward {reward:+7.2f} | return {total:+7.1f} "
                  f"| stock {env.stock} | stunted {info['n_stunted']}/{env.n_children}")
            for event in info["events"]:
                print(f"         -> {event}")
            done = term or trunc
            if day == DAYS_PER_WEEK and week % 4 == 0 and not done:
                print_children(env)
        totals.append(total)
        print(f"--- Episode {ep + 1} finished: return {total:.1f}, "
              f"{env.n_children - info['n_stunted']}/{env.n_children} children not stunted"
              f"{', FAILURE (severe case left untreated)' if term else ''}")
        print_children(env)

    print(f"\nMean return over {args.episodes} episodes: {np.mean(totals):.1f}")
    env.close()


if __name__ == "__main__":
    main()
