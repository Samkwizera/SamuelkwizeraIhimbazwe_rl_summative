import numpy as np

from environment.custom_env import StuntingCareEnv

# baselines only look at what the CHW can observe, never the true HAZ


def random_policy(env, rng):
    return int(rng.integers(env.action_space.n))


def random_valid_policy(env, rng):
    return int(rng.choice(np.flatnonzero(env.action_masks())))


def idle_policy(env, rng):
    return env.record_action


def greedy_policy(env, rng):
    # rule of thumb: always help whoever the risk model ranks highest
    n = env.n_children
    mask = env.action_masks()
    order = np.argsort(-env.risk_score)
    for i in order:
        if mask[3 * n + i] and env.treat_weeks[i] == 0 and (env.ill_reported[i] or env.obs_haz[i] < -2.8):
            return 3 * n + int(i)
    if mask[env.restock_action] and env.stock <= 1:
        return env.restock_action
    for i in order:
        if mask[2 * n + i]:
            return 2 * n + int(i)
    for i in order:
        if mask[i] and env.week - env.last_screen_week[i] >= 4:
            return int(i)
    for i in order:
        if mask[n + i]:
            return n + int(i)
    return env.record_action


BASELINES = {"random": random_policy, "random_valid": random_valid_policy, "idle": idle_policy,
             "greedy": greedy_policy}


def evaluate_baseline(name, episodes=100, seed=50_000, scenario="standard"):
    policy = BASELINES[name]
    env = StuntingCareEnv(scenario=scenario)
    rng = np.random.default_rng(seed)
    returns, stunted, failures, wasted = [], [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done, total, n_wasted = False, 0.0, 0
        while not done:
            _, r, term, trunc, info = env.step(policy(env, rng))
            total += r
            n_wasted += info["wasted"]
            done = term or trunc
        returns.append(total)
        stunted.append(info["n_stunted"])
        failures.append(term)
        wasted.append(n_wasted)
    return {
        "policy": name,
        "scenario": scenario,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_stunted": float(np.mean(stunted)),
        "failure_rate": float(np.mean(failures)),
        "mean_wasted": float(np.mean(wasted)),
    }


if __name__ == "__main__":
    for scenario in ("standard", "high_risk", "low_resource"):
        for name in BASELINES:
            r = evaluate_baseline(name, scenario=scenario)
            print(f"{scenario:<13}{name:<13}return {r['mean_return']:7.1f} +/- {r['std_return']:5.1f}  "
                  f"stunted {r['mean_stunted']:.2f}  fail {r['failure_rate']:.2f}  wasted {r['mean_wasted']:.1f}")
