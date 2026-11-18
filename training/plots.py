import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.common import LOG_DIR, MODEL_DIR, ROOT

ASSETS = ROOT / "assets"
ALGOS = ("dqn", "reinforce", "ppo", "a2c")
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C",
          "random_valid": "Random (valid)", "greedy": "Greedy rule"}
# fixed colour per algorithm across every figure
COLORS = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SMOOTH = 200

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load(algo, run, name):
    return pd.read_csv(LOG_DIR / algo / run / f"{name}.csv")


def baseline_lines(ax, final, scenario="standard", side="right"):
    x, ha, dx = (1.0, "right", -4) if side == "right" else (0.0, "left", 4)
    for name, style in (("random_valid", ":"), ("greedy", "--")):
        value = final[(final.policy == name) & (final.scenario == scenario)]["mean_return"].iloc[0]
        ax.axhline(value, color=MUTED, linestyle=style, linewidth=1)
        ax.annotate(f"{LABELS[name]} {value:.0f}", (x, value), xycoords=("axes fraction", "data"),
                    xytext=(dx, 3), textcoords="offset points", ha=ha, va="bottom", fontsize=8, color=INK_2)


def spread_labels(ys, gap):
    # nudge end-of-line labels apart so they don't print on top of each other
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def episodes_to_converge(returns):
    smooth = returns.rolling(SMOOTH, min_periods=SMOOTH // 4).mean()
    start, peak = smooth.dropna().iloc[0], smooth.max()
    hit = smooth[smooth >= start + 0.9 * (peak - start)]
    return int(hit.index[0]) + 1, float(peak)


def fig_rewards(best, final):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        eps = load(algo, best[algo]["run"], "episodes")
        ax.plot(eps["episode"], eps["return"], color=COLORS[algo], alpha=0.15, linewidth=0.6)
        ax.plot(eps["episode"], eps["return"].rolling(SMOOTH, min_periods=20).mean(), color=COLORS[algo])
        baseline_lines(ax, final)
        ax.set_title(f"{LABELS[algo]} ({best[algo]['run']})")
        ax.set_xlabel("training episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("episode return")
    fig.suptitle(f"Training return per episode, best run of each method ({SMOOTH}-episode rolling mean)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_cumulative_rewards.png")
    plt.close(fig)


def fig_dqn(best):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    sweep = pd.read_csv(LOG_DIR / "dqn_sweep.csv")
    for run in sweep["name"]:
        if run == best["dqn"]["run"]:
            continue
        upd = load("dqn", run, "updates")
        axes[0].plot(upd["frames"], upd["loss"], color=GRID, linewidth=0.9)
    upd = load("dqn", best["dqn"]["run"], "updates")
    axes[0].plot(upd["frames"], upd["loss"], color=COLORS["dqn"])
    axes[0].set_title("TD loss (objective)")
    axes[0].set_yscale("log")
    axes[1].plot(upd["frames"], upd["mean_max_q"], color=COLORS["dqn"])
    axes[1].set_title("Mean max Q-value of sampled batch")
    axes[2].plot(upd["frames"], upd["epsilon"], color=COLORS["dqn"])
    axes[2].set_title("Exploration rate ε")
    for ax in axes:
        ax.set_xlabel("environment frames")
    fig.suptitle(f"DQN training diagnostics, best run {best['dqn']['run']} (grey: other sweep runs)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_dqn_objective.png")
    plt.close(fig)


def fig_entropy(best):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, algo in zip(axes, ("reinforce", "ppo", "a2c")):
        sweep = pd.read_csv(LOG_DIR / f"{algo}_sweep.csv")
        for run in sweep["name"]:
            if run != best[algo]["run"]:
                upd = load(algo, run, "updates")
                ax.plot(upd["frames"], upd["entropy"].rolling(5, min_periods=1).mean(), color=GRID, linewidth=0.9)
        upd = load(algo, best[algo]["run"], "updates")
        ax.plot(upd["frames"], upd["entropy"].rolling(5, min_periods=1).mean(), color=COLORS[algo])
        ax.set_title(f"{LABELS[algo]} ({best[algo]['run']})")
        ax.set_xlabel("environment frames")
    axes[0].set_ylabel("policy entropy (nats)")
    fig.suptitle("Policy entropy during training (grey: other sweep runs)", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_pg_entropy.png")
    plt.close(fig)


def fig_convergence(best, final):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    rows, ends = [], []
    for algo in ALGOS:
        ev = load(algo, best[algo]["run"], "evals")
        axes[0].plot(ev["frames"], ev["mean_return"], color=COLORS[algo], marker="o", markersize=3.5)
        ends.append((ev["frames"].iloc[-1], ev["mean_return"].iloc[-1]))
        eps = load(algo, best[algo]["run"], "episodes")
        n, peak = episodes_to_converge(eps["return"])
        rows.append((algo, n, peak))
    span = np.ptp(axes[0].get_ylim())
    for algo, (x, _), y in zip(ALGOS, ends, spread_labels([y for _, y in ends], span * 0.05)):
        axes[0].annotate(LABELS[algo], (x, y), xytext=(8, 0), textcoords="offset points",
                         va="center", fontsize=8.5, color=INK_2)
    baseline_lines(axes[0], final, side="left")
    axes[0].set_title("Held-out evaluation return during training")
    axes[0].set_xlabel("environment frames")
    axes[0].set_ylabel("mean return (20 eval episodes)")
    axes[0].set_xlim(right=axes[0].get_xlim()[1] * 1.12)

    names = [LABELS[a] for a, _, _ in rows]
    values = [n for _, n, _ in rows]
    bars = axes[1].barh(names, values, color=[COLORS[a] for a, _, _ in rows], height=0.55)
    axes[1].bar_label(bars, labels=[f"{v:,}" for v in values], padding=3, fontsize=8.5, color=INK_2)
    axes[1].invert_yaxis()
    axes[1].set_title("Episodes to reach 90% of peak smoothed return")
    axes[1].set_xlabel("training episodes")
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Convergence of the best run per method", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_convergence.png")
    plt.close(fig)
    return pd.DataFrame(rows, columns=["algo", "episodes_to_converge", "peak_smoothed_return"])


def fig_generalization(final):
    policies = [*ALGOS, "greedy", "random_valid"]
    scenarios = ("standard", "high_risk", "low_resource")
    # baselines are greys, the hatch keeps the two apart in print
    colors = [COLORS.get(p) for p in ALGOS] + ["#8f8e87", "#d6d5cf"]
    hatches = [None] * len(ALGOS) + [None, "////"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.13
    x = np.arange(len(scenarios))
    for k, (metric, title) in enumerate((("mean_return", "Mean return"), ("mean_stunted", "Children stunted at end (of 6)"))):
        ax = axes[k]
        for j, p in enumerate(policies):
            vals = [final[(final.policy == p) & (final.scenario == s)][metric].iloc[0] for s in scenarios]
            ax.bar(x + (j - 2.5) * width, vals, width * 0.92, color=colors[j], hatch=hatches[j],
                   edgecolor="white", linewidth=0, label=LABELS[p])
        ax.set_xticks(x, [s.replace("_", " ") for s in scenarios])
        ax.set_title(title)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=6, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Generalisation: best models on 100 unseen episodes per scenario", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ASSETS / "fig_generalization.png")
    plt.close(fig)


def fig_sweeps(best):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        sweep = pd.read_csv(LOG_DIR / f"{algo}_sweep.csv").sort_values("name")
        is_best = sweep["name"] == best[algo]["run"]
        colors = [COLORS[algo] if b else "#d6d5cf" for b in is_best]
        labels = [n.split("_")[-1] for n in sweep["name"]]
        bars = ax.bar(labels, sweep["eval_mean_return"], color=colors, width=0.7)
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in sweep["eval_mean_return"]], fontsize=7.5, color=INK_2, padding=2)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.margins(y=0.12)
        ax.set_title(f"{LABELS[algo]} runs (best highlighted)")
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("eval mean return")
    for ax in axes[1]:
        ax.set_xlabel("run number")
    fig.suptitle("Hyperparameter sweep: final evaluation return of every run", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_hyperparameter_sweeps.png")
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    best = json.loads((MODEL_DIR / "best_models.json").read_text())
    final = pd.read_csv(LOG_DIR / "final_evaluation.csv")
    fig_rewards(best, final)
    fig_dqn(best)
    fig_entropy(best)
    conv = fig_convergence(best, final)
    conv.to_csv(LOG_DIR / "convergence.csv", index=False)
    print(conv.to_string(index=False))
    fig_generalization(final)
    fig_sweeps(best)
    print(f"figures written to {ASSETS}")


if __name__ == "__main__":
    main()
