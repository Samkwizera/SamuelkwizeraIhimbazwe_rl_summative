import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment import ENV_ID, SCENARIOS, StuntingCareEnv
from environment.custom_env import DAYS_PER_WEEK, N_WEEKS


def rollout(env, seed, actions=None, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    obs, _ = env.reset(seed=seed)
    trace = [obs]
    rewards = []
    done = False
    while not done:
        a = int(rng.integers(env.action_space.n)) if actions is None else actions(env)
        obs, r, term, trunc, info = env.step(a)
        trace.append(obs)
        rewards.append(r)
        done = term or trunc
    return np.array(trace), np.array(rewards), term, trunc, info


def test_gymnasium_env_checker():
    check_env(StuntingCareEnv(), skip_render_check=True)


def test_registered_with_gymnasium():
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_spaces():
    env = StuntingCareEnv()
    assert env.action_space.n == 4 * 6 + 2
    assert env.observation_space.shape == (6 * 12 + 4,)
    names = {env.action_name(a) for a in range(env.action_space.n)}
    assert len(names) == env.action_space.n


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_observations_in_bounds(scenario):
    env = StuntingCareEnv(scenario=scenario)
    for seed in range(5):
        trace, rewards, *_ = rollout(env, seed, rng_seed=seed)
        assert trace.dtype == np.float32
        assert (trace >= 0).all() and (trace <= 1).all()
        assert np.isfinite(rewards).all()


def test_deterministic_given_seed():
    a = rollout(StuntingCareEnv(), seed=7, rng_seed=1)
    b = rollout(StuntingCareEnv(), seed=7, rng_seed=1)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_truncates_after_full_period():
    env = StuntingCareEnv()
    trace, rewards, term, trunc, info = rollout(env, seed=3, actions=lambda e: e.record_action)
    # idle CHW may hit the failure condition early; otherwise the period runs to the end
    assert term or (trunc and len(rewards) == N_WEEKS * DAYS_PER_WEEK)


def test_actions_change_hidden_state():
    gains = []
    for act in ("supplement", "idle"):
        env = StuntingCareEnv()
        env.reset(seed=11)
        child = int(np.argmax(env.age))  # oldest child, always >= 6 months
        env.age[child] = max(env.age[child], 12.0)
        start = env.haz[child]
        for _ in range(DAYS_PER_WEEK):
            a = 2 * env.n_children + child if act == "supplement" else env.record_action
            env.step(a)
        gains.append(env.haz[child] - start)
    assert gains[0] > gains[1]


def test_wasted_actions_are_penalised():
    env = StuntingCareEnv()
    env.reset(seed=0)
    env.age[0] = 3.0  # too young for Shisha Kibondo
    _, r, *_, info = env.step(2 * env.n_children + 0)
    assert info["wasted"] and r < 0
    env.stock = 12
    _, _, *_, info = env.step(env.restock_action)
    assert info["wasted"]


def test_screen_reveals_true_haz():
    env = StuntingCareEnv()
    env.reset(seed=5)
    env.last_screen_week[:] = -10
    env.step(0)  # SCREEN child 1
    assert abs(env.obs_haz[0] - env.haz[0]) < 0.3


def test_torchrl_wrapper_specs():
    from torchrl.envs import GymWrapper
    from torchrl.envs.utils import check_env_specs

    env = GymWrapper(StuntingCareEnv(), categorical_action_encoding=True)
    check_env_specs(env)
