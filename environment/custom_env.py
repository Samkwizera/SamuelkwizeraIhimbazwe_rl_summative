from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

N_CHILDREN = 6
DAYS_PER_WEEK = 3  # CHWs are part-time volunteers
N_WEEKS = 26

STUNTING_HAZ = -2.0
SEVERE_HAZ = -3.0

ACTION_TYPES = ("SCREEN", "COUNSEL", "SUPPLEMENT", "REFER")

MAX_STOCK = 12
RESTOCK_AMOUNT = 6
SUPPLEMENT_WEEKS = 4  # one pack of Shisha Kibondo lasts about a month
TREATMENT_WEEKS = 3
REFERRAL_PERIOD_WEEKS = 4
MIN_SUPPLEMENT_AGE = 6.0  # Shisha Kibondo targets 6-23 months, younger babies are breastfed

R_HAZ_GAIN = 10.0
R_NEW_STUNTING = -10.0
R_RECOVERY = 5.0
R_EARLY_DETECT = 3.0
R_LATE_DETECT = 1.0
R_WASTED = -0.5
R_UNNEEDED_REFERRAL = -1.0
R_FAILURE = -20.0
R_FINAL_OK = 5.0
R_FINAL_STUNTED = -5.0

CHILD_FEATURE_NAMES = (
    "measured_haz",
    "weeks_since_screen",
    "age_months",
    "ml_risk_score",
    "food_insecure",
    "poor_sanitation",
    "low_maternal_education",
    "ubudehe_category_1",
    "supplement_weeks_left",
    "illness_reported",
    "treatment_weeks_left",
    "weeks_since_counsel",
)
GLOBAL_FEATURE_NAMES = ("stock", "referrals_left", "day_of_week", "weeks_remaining")

# rough prevalences, standard is close to Rwanda DHS 2019-20 (about 1 in 3 children stunted)
SCENARIOS = {
    "standard": dict(p_food=0.40, p_sanit=0.45, p_edu=0.35, p_ubudehe=0.30,
                     haz_shift=0.0, stock=6, referrals=2),
    "high_risk": dict(p_food=0.60, p_sanit=0.65, p_edu=0.50, p_ubudehe=0.45,
                      haz_shift=-0.3, stock=6, referrals=2),
    "low_resource": dict(p_food=0.40, p_sanit=0.45, p_edu=0.35, p_ubudehe=0.30,
                         haz_shift=0.0, stock=2, referrals=1),
}


class StuntingCareEnv(gym.Env):
    """One CHW following up 6 children for 26 weeks. One step = one working day."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, n_children=N_CHILDREN, n_weeks=N_WEEKS, scenario="standard", render_mode=None,
                 render_options=None):
        super().__init__()
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
        self.n_children = n_children
        self.n_weeks = n_weeks
        self.max_steps = n_weeks * DAYS_PER_WEEK
        self.scenario = scenario
        self.render_mode = render_mode
        self.render_options = render_options or {}

        # actions 0..4n-1 are (type, child) pairs, then restock and record keeping
        self.restock_action = len(ACTION_TYPES) * n_children
        self.record_action = self.restock_action + 1
        self.action_space = spaces.Discrete(self.record_action + 1)

        obs_dim = n_children * len(CHILD_FEATURE_NAMES) + len(GLOBAL_FEATURE_NAMES)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._renderer = None
        self._last_info = {}

    def decode_action(self, action):
        action = int(action)
        if action == self.restock_action:
            return "RESTOCK", None
        if action == self.record_action:
            return "RECORD_KEEPING", None
        kind, child = divmod(action, self.n_children)
        return ACTION_TYPES[kind], child

    def action_name(self, action):
        kind, child = self.decode_action(action)
        return kind if child is None else f"{kind} child {child + 1}"

    def _update_risk_scores(self):
        # stands in for the capstone stunting-risk model, so it only sees what the CHW knows
        z = (
            -1.2
            + 2.2 * self.risk_load
            + 1.4 * np.clip(-self.obs_haz - 1.0, -1.5, 2.5)
            + 0.8 * self.ill_reported
            + self.np_random.normal(0.0, 0.25, self.n_children)
        )
        self.risk_score = 1.0 / (1.0 + np.exp(-z))

    def _measure(self, i):
        self.obs_haz[i] = self.haz[i] + self.np_random.normal(0.0, 0.05)
        self.last_screen_week[i] = self.week
        self.ill_reported[i] = self.ill[i]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            self.scenario = options["scenario"]
        cfg = SCENARIOS[self.scenario]
        n, rng = self.n_children, self.np_random

        self.food = (rng.random(n) < cfg["p_food"]).astype(float)
        self.sanit = (rng.random(n) < cfg["p_sanit"]).astype(float)
        self.edu = (rng.random(n) < cfg["p_edu"]).astype(float)
        self.ubudehe = (rng.random(n) < cfg["p_ubudehe"]).astype(float)
        self.risk_load = 0.35 * self.food + 0.25 * self.sanit + 0.2 * self.edu + 0.2 * self.ubudehe

        self.age = rng.uniform(0.0, 24.0, n)
        mean_haz = -1.3 - 0.5 * (self.risk_load - 0.4) + cfg["haz_shift"]
        self.haz = np.clip(rng.normal(mean_haz, 0.6), -2.8, 0.5)
        self.base_adherence = np.clip(rng.uniform(0.35, 0.75, n) - 0.15 * self.edu, 0.2, 0.9)
        self.adherence = self.base_adherence.copy()
        self.ill = rng.random(n) < 0.05
        self.supp_weeks = np.zeros(n, dtype=int)
        self.treat_weeks = np.zeros(n, dtype=int)
        self.counsel_boost = np.zeros(n)
        self.severe_untreated = np.zeros(n, dtype=int)

        self.week = 0
        self.day = 0
        self.steps = 0
        # baseline measurement was taken 0-8 weeks ago, older ones are less accurate
        baseline_age = rng.integers(0, 9, n)
        self.obs_haz = self.haz + rng.normal(0.0, 0.05 + 0.03 * baseline_age, n)
        self.last_screen_week = -baseline_age
        self.last_counsel_week = np.full(n, -99)
        self.ill_reported = self.ill.copy()
        self.stunted = self.haz < STUNTING_HAZ
        self.stunted_since = np.where(self.stunted, 0, -1)
        self.detected = self.stunted & (self.obs_haz < STUNTING_HAZ)

        self.stock = cfg["stock"]
        self.referral_quota = cfg["referrals"]
        self.referrals_left = self.referral_quota
        self._update_risk_scores()

        self._last_info = self._info(None, False, ["Caseload registered"])
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._last_info

    def step(self, action):
        kind, i = self.decode_action(action)
        reward = 0.0
        wasted = False
        events = []

        if kind == "SCREEN":
            if self.week - self.last_screen_week[i] < 2:
                wasted = True
            else:
                self._measure(i)
                if self.stunted[i] and not self.detected[i]:
                    self.detected[i] = True
                    early = self.week - self.stunted_since[i] <= 2
                    reward += R_EARLY_DETECT if early else R_LATE_DETECT
                    events.append(f"Child {i + 1} found stunted ({'early' if early else 'late'})")
        elif kind == "COUNSEL":
            if self.week - self.last_counsel_week[i] < 2:
                wasted = True
            else:
                self.last_counsel_week[i] = self.week
                self.counsel_boost[i] = 0.02
                self.adherence[i] = min(0.95, self.adherence[i] + 0.1)
        elif kind == "SUPPLEMENT":
            if self.stock <= 0 or self.age[i] < MIN_SUPPLEMENT_AGE or self.supp_weeks[i] >= 2:
                wasted = True
            else:
                self.stock -= 1
                self.supp_weeks[i] = SUPPLEMENT_WEEKS
        elif kind == "REFER":
            if self.referrals_left <= 0:
                wasted = True
            else:
                self.referrals_left -= 1
                if not self.ill[i] and self.haz[i] >= STUNTING_HAZ:
                    reward += R_UNNEEDED_REFERRAL
                    events.append(f"Child {i + 1} referred but was not at risk")
                self.ill[i] = False
                self.treat_weeks[i] = TREATMENT_WEEKS
                self._measure(i)  # the health centre measures the child too
                if self.stunted[i]:
                    self.detected[i] = True
        elif kind == "RESTOCK":
            if self.stock >= MAX_STOCK - 2:
                wasted = True
            else:
                self.stock = min(MAX_STOCK, self.stock + RESTOCK_AMOUNT)

        if wasted:
            reward += R_WASTED

        self.steps += 1
        self.day += 1
        terminated = truncated = False

        if self.day == DAYS_PER_WEEK:
            week_reward, week_events, terminated = self._advance_week()
            reward += week_reward
            events += week_events

        if not terminated and self.steps >= self.max_steps:
            truncated = True
            n_stunted = int(self.stunted.sum())
            reward += R_FINAL_OK * (self.n_children - n_stunted) + R_FINAL_STUNTED * n_stunted
            events.append(f"Period over: {self.n_children - n_stunted}/{self.n_children} children not stunted")

        self._last_info = self._info(action, wasted, events)
        self._last_info["reward"] = reward
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, self._last_info

    def _advance_week(self):
        rng, n = self.np_random, self.n_children
        events = []

        # interventions work best in the first 1000 days
        age_factor = np.clip(1.2 - 0.025 * self.age, 0.5, 1.2)
        drift = -(0.01 + 0.05 * self.risk_load)
        illness = -0.05 * self.ill
        supplement = 0.06 * self.adherence * age_factor * (self.supp_weeks > 0)
        counsel = self.counsel_boost * age_factor
        treatment = 0.07 * (self.treat_weeks > 0)
        dhaz = drift + illness + supplement + counsel + treatment + rng.normal(0.0, 0.015, n)
        self.haz = np.clip(self.haz + dhaz, -5.0, 2.0)
        reward = R_HAZ_GAIN * dhaz.sum()

        severe = self.haz < SEVERE_HAZ
        self.severe_untreated = np.where(severe & (self.treat_weeks == 0), self.severe_untreated + 1, 0)

        now_stunted = self.haz < STUNTING_HAZ
        new_cases = now_stunted & ~self.stunted
        recoveries = ~now_stunted & self.stunted
        reward += R_NEW_STUNTING * new_cases.sum() + R_RECOVERY * recoveries.sum()
        for i in np.flatnonzero(new_cases):
            events.append(f"Child {i + 1} became stunted (HAZ {self.haz[i]:.2f})")
        for i in np.flatnonzero(recoveries):
            events.append(f"Child {i + 1} recovered above HAZ -2")
        self.stunted_since = np.where(new_cases, self.week + 1, np.where(now_stunted, self.stunted_since, -1))
        self.detected = np.where(new_cases | recoveries, False, self.detected)
        self.stunted = now_stunted

        self.week += 1
        self.day = 0
        self.age = self.age + 12 / 52
        self.supp_weeks = np.maximum(self.supp_weeks - 1, 0)
        self.treat_weeks = np.maximum(self.treat_weeks - 1, 0)
        self.counsel_boost *= 0.85
        self.adherence = np.maximum(self.base_adherence, self.adherence - 0.01)
        if self.week % REFERRAL_PERIOD_WEEKS == 0:
            self.referrals_left = self.referral_quota

        # WASH counselling cuts diarrhoea risk for about a month
        recent_counsel = (self.week - self.last_counsel_week) <= 4
        p_ill = (0.03 + 0.07 * self.sanit) * np.where(recent_counsel, 0.6, 1.0)
        recovered = self.ill & (rng.random(n) < 0.35)
        new_ill = ~self.ill & (rng.random(n) < p_ill)
        self.ill = (self.ill & ~recovered) | new_ill
        # caregivers only report about half of illnesses to the CHW
        self.ill_reported = self.ill & (rng.random(n) < 0.5)
        self._update_risk_scores()

        terminated = bool((self.severe_untreated >= 2).any())
        if terminated:
            reward += R_FAILURE
            worst = int(np.argmax(self.severe_untreated))
            events.append(f"FAILURE: child {worst + 1} severely stunted and not referred for 2 weeks")
        return float(reward), events, terminated

    def _obs(self):
        child = np.stack(
            [
                (self.obs_haz + 4.0) / 5.0,
                (self.week - self.last_screen_week) / self.n_weeks,
                self.age / 36.0,
                self.risk_score,
                self.food,
                self.sanit,
                self.edu,
                self.ubudehe,
                self.supp_weeks / SUPPLEMENT_WEEKS,
                self.ill_reported,
                self.treat_weeks / TREATMENT_WEEKS,
                np.minimum(self.week - self.last_counsel_week, 4) / 4.0,
            ],
            axis=1,
        )
        glob = [
            self.stock / MAX_STOCK,
            self.referrals_left / max(self.referral_quota, 1),
            self.day / (DAYS_PER_WEEK - 1),
            (self.n_weeks - self.week) / self.n_weeks,
        ]
        obs = np.concatenate([child.ravel(), glob])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _info(self, action, wasted, events):
        return {
            "step": self.steps,
            "week": self.week,
            "day": self.day,
            "action": None if action is None else int(action),
            "action_name": None if action is None else self.action_name(action),
            "wasted": wasted,
            "events": events,
            "n_stunted": int(self.stunted.sum()),
        }

    def snapshot(self):
        # full state (hidden parts included) for the visualisation and the demo
        return {
            **self._last_info,
            "n_children": self.n_children,
            "n_weeks": self.n_weeks,
            "true_haz": self.haz.round(3).tolist(),
            "measured_haz": self.obs_haz.round(3).tolist(),
            "risk_score": self.risk_score.round(3).tolist(),
            "age_months": self.age.round(1).tolist(),
            "ill": self.ill.tolist(),
            "supplement_weeks": self.supp_weeks.tolist(),
            "treatment_weeks": self.treat_weeks.tolist(),
            "stock": self.stock,
            "referrals_left": self.referrals_left,
            "scenario": self.scenario,
        }

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode == "human":
            if self._renderer is None:
                from environment.rendering import ThreeJSRenderer

                self._renderer = ThreeJSRenderer(**self.render_options)
            self._renderer.update(self.snapshot())

    def _render_text(self):
        s = self.snapshot()
        lines = [
            f"Week {s['week']:>2} Day {s['day']} | action: {s['action_name']} "
            f"| stock {s['stock']} | referrals {s['referrals_left']} | stunted {s['n_stunted']}"
        ]
        for i in range(self.n_children):
            lines.append(
                f"  child {i + 1}: age {s['age_months'][i]:5.1f}m  true HAZ {s['true_haz'][i]:+.2f}  "
                f"measured {s['measured_haz'][i]:+.2f}  risk {s['risk_score'][i]:.2f}"
            )
        lines += [f"  * {e}" for e in s["events"]]
        return "\n".join(lines)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
