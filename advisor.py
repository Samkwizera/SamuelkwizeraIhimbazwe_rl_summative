import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from environment import StuntingCareEnv
from training.common import MODEL_DIR, ROOT, load_policy


class CHWAdvisor:
    """Turns a saved checkpoint into ranked recommendations the CHW can accept or override."""

    def __init__(self, path=None):
        if path is None:
            best = json.loads((MODEL_DIR / "best_models.json").read_text())
            path = ROOT / best[best["overall"]]["path"]
        self.net, self.ckpt = load_policy(path)
        self.names = StuntingCareEnv()

    @torch.no_grad()
    def recommend(self, observation, action_mask, top_k=3):
        scores = self.net(torch.as_tensor(observation, dtype=torch.float32))
        scores[~torch.as_tensor(action_mask, dtype=torch.bool)] = -torch.inf
        # DQN gives Q-values and the PG methods give logits, softmax makes both comparable
        probs = torch.softmax(scores, -1)
        top = torch.topk(probs, top_k)
        return [
            {"action": int(a), "name": self.names.action_name(int(a)), "confidence": round(float(p), 3)}
            for p, a in zip(top.values, top.indices)
        ]


def serve(advisor, port):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/recommend":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result = advisor.recommend(body["observation"], body["action_mask"], body.get("top_k", 3))
            out = json.dumps({"model": advisor.ckpt["config"]["name"], "recommendations": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    print(f"POST observations to http://127.0.0.1:{port}/recommend")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Recommend the next CHW actions from a trained model")
    parser.add_argument("--model", help="checkpoint path, defaults to the best model")
    parser.add_argument("--serve", action="store_true", help="start the JSON API instead of a demo")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()
    advisor = CHWAdvisor(args.model)
    if args.serve:
        serve(advisor, args.port)
        return

    env = StuntingCareEnv(render_mode="ansi")
    obs, _ = env.reset(seed=1)
    print(env.render())
    print("\nRecommended next actions:")
    for rec in advisor.recommend(obs["observation"], obs["action_mask"]):
        print(f"{rec['name']:<22} confidence {rec['confidence']:.2f}")


if __name__ == "__main__":
    main()
