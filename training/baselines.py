import numpy as np

from environment.custom_env import MIN_SUPPLEMENT_AGE, StuntingCareEnv


# baselines only look at what the CHW can observe, never the true HAZ


def random_policy(env, rng):
    return int(rng.integers(env.action_space.n))


def idle_policy(env, rng):
    return env.record_action


def greedy_policy(env, rng):
    # rule of thumb: always help whoever the risk model ranks highest
    n = env.n_children
    order = np.argsort(-env.risk_score)
    for i in order:
        if env.referrals_left > 0 and env.treat_weeks[i] == 0 and (
            env.ill_reported[i] or env.obs_haz[i] < -2.8
        ):
            return 3 * n + int(i)
    for i in order:
        if env.age[i] >= MIN_SUPPLEMENT_AGE and env.supp_weeks[i] < 2:
            if env.stock > 0:
                return 2 * n + int(i)
            if env.stock < 10:
                return env.restock_action
    for i in order:
        if env.week - env.last_screen_week[i] >= 4:
            return int(i)
    for i in order:
        if env.week - env.last_counsel_week[i] >= 2:
            return n + int(i)
    return env.record_action


BASELINES = {"random": random_policy, "idle": idle_policy, "greedy": greedy_policy}


def evaluate_baseline(name, episodes=100, seed=10_000, scenario="standard"):
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
        "mean_final_stunted": float(np.mean(stunted)),
        "failure_rate": float(np.mean(failures)),
        "mean_wasted_actions": float(np.mean(wasted)),
    }


if __name__ == "__main__":
    for scenario in ("standard", "high_risk", "low_resource"):
        for name in BASELINES:
            print(evaluate_baseline(name, scenario=scenario))
