import json

import pandas as pd
import torch

from training.baselines import evaluate_baseline
from training.common import LOG_DIR, MODEL_DIR, ROOT, evaluate, load_policy

ALGOS = ("dqn", "reinforce", "ppo", "a2c")
SCENARIOS = ("standard", "high_risk", "low_resource")
# different from the seeds used to pick the best checkpoint, so this is a clean test
TEST_SEED = 60_000
EPISODES = 100


def best_runs():
    best = {}
    for algo in ALGOS:
        sweep = pd.read_csv(LOG_DIR / f"{algo}_sweep.csv")
        row = sweep.loc[sweep["eval_mean_return"].idxmax()]
        folder = "dqn" if algo == "dqn" else "pg"
        best[algo] = {"run": row["name"], "path": f"models/{folder}/{row['name']}.pt"}
    return best


def main():
    torch.set_num_threads(1)
    best = best_runs()
    rows = []
    for algo, entry in best.items():
        net, _ = load_policy(ROOT / entry["path"])
        for scenario in SCENARIOS:
            result = evaluate(net, episodes=EPISODES, scenario=scenario, seed=TEST_SEED)
            rows.append({"policy": algo, "run": entry["run"], "scenario": scenario, **result})
            print(f"{algo:<10}{entry['run']:<14}{scenario:<13}return {result['mean_return']:7.1f}  "
                  f"stunted {result['mean_stunted']:.2f}  fail {result['failure_rate']:.2f}", flush=True)
            if scenario == "standard":
                entry.update(mean_return=result["mean_return"], mean_stunted=result["mean_stunted"])

    for name in ("random_valid", "greedy"):
        for scenario in SCENARIOS:
            r = evaluate_baseline(name, episodes=EPISODES, seed=TEST_SEED, scenario=scenario)
            rows.append({"policy": name, "run": "-", "scenario": scenario, "mean_return": r["mean_return"],
                         "std_return": r["std_return"], "mean_stunted": r["mean_stunted"],
                         "failure_rate": r["failure_rate"], "mean_wasted": r["mean_wasted"]})
            print(f"{name:<24}{scenario:<13}return {r['mean_return']:7.1f}  stunted {r['mean_stunted']:.2f}", flush=True)

    pd.DataFrame(rows).to_csv(LOG_DIR / "final_evaluation.csv", index=False)
    best["overall"] = max(ALGOS, key=lambda a: best[a]["mean_return"])
    (MODEL_DIR / "best_models.json").write_text(json.dumps(best, indent=2))
    print(f"best overall: {best['overall']} ({best[best['overall']]['run']})")


if __name__ == "__main__":
    main()
