"""Entrainement sequentiel des agents RL.

Hierarchie (2 rounds par defaut, chaque agent est entraine contre l'autre fige) :
  round 0 : agent edge contre un device heuristique (80 % vers l'edge),
            puis agent device contre l'agent edge fige ;
  round 1 : on recommence avec les agents du round 0 comme partenaires figes.
Baseline RL centralisee (flat) : un seul agent, vue globale, 3 actions.

Usage :
    python train.py --quick      # test rapide (quelques minutes au plus)
    python train.py              # entrainement complet
"""
import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from core import Config
from envs import DeviceEnv, EdgeEnv, FlatEnv
from policies import sb3_fn


def make_model(env, seed):
    return DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50_000,
               learning_starts=2_000, batch_size=64, gamma=0.9, train_freq=4,
               target_update_interval=500, exploration_fraction=0.4,
               exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[64, 64]),
               seed=seed, verbose=0, device="cpu")


def fit(env, tag, steps, seed):
    env = Monitor(env, filename=f"logs/{tag}")
    model = make_model(env, seed)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    model.save(f"models/{tag}")
    print(f"[{tag}] {steps} pas en {time.time() - t0:.0f} s", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--edge-steps", type=int, default=100_000)
    ap.add_argument("--dev-steps", type=int, default=150_000)
    ap.add_argument("--flat-steps", type=int, default=None,
                    help="par defaut : meme budget total que la hierarchie")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", type=str, default="data/alibaba_batch_task.csv",
                    help="trace CSV start_time,duration ('' : charge synthetique)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.rounds, args.edge_steps, args.dev_steps = 1, 6_000, 6_000
    flat_steps = args.flat_steps or args.rounds * (args.edge_steps + args.dev_steps)

    torch.set_num_threads(1)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    cfg = Config(trace_path=args.trace or None)

    rng0 = np.random.default_rng(args.seed)
    dev_fn = lambda obs: int(rng0.random() < 0.8)  # device initial : 80 % vers l'edge
    edge_model = dev_model = None
    for r in range(args.rounds):
        edge_model = fit(EdgeEnv(cfg, dev_fn, seed=args.seed + 10 * r),
                         f"edge_r{r}", args.edge_steps, args.seed + r)
        edge_fn = sb3_fn(edge_model)
        dev_model = fit(DeviceEnv(cfg, edge_fn, seed=args.seed + 10 * r + 1),
                        f"device_r{r}", args.dev_steps, args.seed + r)
        dev_fn = sb3_fn(dev_model)
    edge_model.save("models/edge")
    dev_model.save("models/device")

    fit(FlatEnv(cfg, seed=args.seed + 99), "flat", flat_steps, args.seed)
    print("Entrainement termine.")


if __name__ == "__main__":
    main()
