import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from training import dqn_training, pg_training
from training.common import LOG_DIR


def jobs(algos):
    out = []
    # slowest algorithms first so the pool doesn't end on one long run
    for algo in ("dqn", "ppo", "a2c", "reinforce"):
        if algo not in algos:
            continue
        runs = dqn_training.RUNS if algo == "dqn" else pg_training.RUNS[algo]
        for k, cfg in enumerate(runs, start=1):
            if algo == "dqn":
                cmd = [sys.executable, "-u", "-m", "training.dqn_training", "--runs", str(k)]
            else:
                cmd = [sys.executable, "-u", "-m", "training.pg_training", "--algo", algo, "--runs", str(k)]
            out.append((algo, cfg["name"], cmd))
    return out


def run(algo, name, cmd, frames):
    log_file = LOG_DIR / algo / name / "stdout.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_file, "w") as f:
        code = subprocess.call(cmd + ["--frames", str(frames)], stdout=f, stderr=subprocess.STDOUT)
    return name, code, (time.time() - start) / 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", default="dqn,ppo,a2c,reinforce")
    parser.add_argument("--frames", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    todo = jobs(args.algos.split(","))
    print(f"{len(todo)} runs on {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, algo, name, cmd, args.frames) for algo, name, cmd in todo]
        for done, fut in enumerate(as_completed(futures), start=1):
            name, code, minutes = fut.result()
            status = "ok" if code == 0 else f"FAILED ({code})"
            print(f"[{done}/{len(todo)}] {name} {status} in {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
