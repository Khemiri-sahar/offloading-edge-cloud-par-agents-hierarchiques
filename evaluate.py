"""Evaluation comparative : memes flux de taches pour toutes les politiques.

Produit dans results/ :
  results.csv          moyennes, ecarts-types et IC95 par politique et par charge
  summary.md           tableaux prets a coller dans l'article
  fig_latency.png, fig_hit_rate.png, fig_energy.png, fig_overhead.png,
  fig_beacon.png (sensibilite a la periode des beacons), fig_training.png

Usage :
    python evaluate.py                 # 20 episodes par charge
    python evaluate.py --episodes 5    # plus rapide
"""
import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from core import Config, run_episode, with_load
from policies import (AlwaysEdge, EDFLike, FlatPolicy, GreedyNearest, HierPolicy,
                      RandomPolicy, sb3_fn)

LOADS = [0.5, 1.0, 2.0, 3.0, 4.0]
METRICS = ["latency", "hit_rate", "energy", "e_dev", "reject_rate",
           "frac_device", "frac_edge", "frac_cloud", "frac_fwd", "overhead"]
STYLE = {  # couleur, marqueur
    "random": ("#999999", "x"), "always_edge": ("#8c564b", "s"),
    "greedy_nearest": ("#ff7f0e", "^"), "edf_like": ("#2ca02c", "v"),
    "flat_rl": ("#9467bd", "D"), "hier_rl": ("#d62728", "o"),
}
LABEL = {"random": "Aleatoire", "always_edge": "Toujours edge",
         "greedy_nearest": "Greedy niveau le plus proche",
         "edf_like": "EDF-like (vue globale)", "flat_rl": "RL centralise (flat)",
         "hier_rl": "RL hierarchique"}


def load_model(path):
    if not os.path.exists(path):
        print(f"[!] modele absent : {path}")
        return None
    return DQN.load(path, device="cpu")


def build_policies():
    pols = [RandomPolicy(), AlwaysEdge(), GreedyNearest(), EDFLike()]
    dev, edge = load_model("models/device.zip"), load_model("models/edge.zip")
    flat = load_model("models/flat.zip")
    if flat is not None:
        pols.append(FlatPolicy(sb3_fn(flat)))
    if dev is not None and edge is not None:
        pols.append(HierPolicy(sb3_fn(dev), sb3_fn(edge)))
    return pols


def aggregate(results):
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in results])
        out[m] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0,
                  1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--trace", type=str, default="data/alibaba_batch_task.csv",
                    help="trace CSV start_time,duration ('' : charge synthetique)")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    args.trace = args.trace or None
    cfg = Config(trace_path=args.trace)
    pols = build_policies()
    seeds = [50_000 + k for k in range(args.episodes)]  # jamais vus a l'entrainement

    table = {}  # (policy, load) -> aggregate
    for load in LOADS:
        c = with_load(cfg, load)
        for p in pols:
            table[(p.name, load)] = aggregate([run_episode(c, s, p) for s in seeds])
        print(f"charge {load} evaluee", flush=True)

    with open("results/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "load"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std", "ci95")])
        for (name, load), agg in table.items():
            w.writerow([name, load] + [f"{x:.4f}" for m in METRICS for x in agg[m]])

    # ---------- tableaux ----------
    with open("results/summary.md", "w") as f:
        f.write(f"Resultats sur {args.episodes} episodes par charge (moyenne, IC95 entre parentheses), "
                f"charge : {args.trace or 'synthetique'}.\n"
                "Device/Edge/Cloud = lieu d'execution ; Transfert = part des taches envoyees a l'edge voisin.\n\n")
        for load in LOADS:
            f.write(f"### Charge x{load}\n\n| Politique | Latence (s) | Deadlines tenues (%) | Energie (J/tache) | "
                    "Device/Edge/Cloud (%) | Transfert (%) | Controle (valeurs/tache) |\n|---|---|---|---|---|---|---|\n")
            for p in pols:
                a = table[(p.name, load)]
                f.write(f"| {LABEL[p.name]} | {a['latency'][0]:.2f} ({a['latency'][2]:.2f}) | "
                        f"{100 * a['hit_rate'][0]:.1f} ({100 * a['hit_rate'][2]:.1f}) | "
                        f"{a['energy'][0]:.2f} ({a['energy'][2]:.2f}) | "
                        f"{100 * a['frac_device'][0]:.0f}/{100 * a['frac_edge'][0]:.0f}/{100 * a['frac_cloud'][0]:.0f} | "
                        f"{100 * a['frac_fwd'][0]:.0f} | "
                        f"{a['overhead'][0]:.1f} |\n")
            f.write("\n")

    # ---------- figures ----------
    def line_plot(metric, ylabel, fname, scale=1.0, log=False):
        plt.figure(figsize=(6.2, 4.0))
        for p in pols:
            col, mk = STYLE[p.name]
            y = [scale * table[(p.name, l)][metric][0] for l in LOADS]
            e = [scale * table[(p.name, l)][metric][2] for l in LOADS]
            plt.errorbar(LOADS, y, yerr=e, color=col, marker=mk, capsize=2, label=LABEL[p.name],
                         lw=2.2 if p.name == "hier_rl" else 1.4)
        plt.xlabel("Charge (multiplicateur du taux d'arrivee)")
        plt.ylabel(ylabel)
        if log:
            plt.yscale("log")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"results/{fname}", dpi=160)
        plt.close()

    line_plot("latency", "Latence moyenne de bout en bout (s, echelle log)", "fig_latency.png", log=True)
    line_plot("hit_rate", "Deadlines respectees (%)", "fig_hit_rate.png", scale=100.0)
    line_plot("energy", "Energie moyenne par tache (J)", "fig_energy.png")

    # surcout de controle vs latence a forte charge
    plt.figure(figsize=(6.2, 4.0))
    ref = 3.0
    for p in pols:
        col, mk = STYLE[p.name]
        plt.scatter(table[(p.name, ref)]["overhead"][0], table[(p.name, ref)]["latency"][0],
                    color=col, marker=mk, s=70, label=LABEL[p.name])
    plt.xlabel("Surcout de controle (valeurs scalaires par tache)")
    plt.ylabel(f"Latence moyenne (s), charge x{ref}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("results/fig_overhead.png", dpi=160)
    plt.close()

    # ---------- sensibilite a la periode des beacons (hierarchie) ----------
    hier = [p for p in pols if p.name == "hier_rl"]
    if hier:
        periods = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        lat, ovh, hit = [], [], []
        for per in periods:
            c = with_load(Config(trace_path=args.trace, beacon_period=per), 3.0)
            rs = [run_episode(c, s, hier[0]) for s in seeds]
            lat.append(np.mean([r["latency"] for r in rs]))
            hit.append(np.mean([r["hit_rate"] for r in rs]) * 100)
            ovh.append(np.mean([r["overhead"] for r in rs]))
        with open("results/beacon_study.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["beacon_period_s", "latency_s", "hit_rate_pct", "control_values_per_task"])
            for row in zip(periods, lat, hit, ovh):
                w.writerow([f"{x:.4f}" for x in row])
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.plot(ovh, lat, "o-", color="#d62728")
        for per, x, y in zip(periods, ovh, lat):
            ax.annotate(f"{per}s", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("Surcout de controle (valeurs par tache)")
        ax.set_ylabel("Latence moyenne (s), charge x3")
        ax.set_title("RL hierarchique : periode des beacons (modele entraine a 1 s)", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/fig_beacon.png", dpi=160)
        plt.close(fig)

    # ---------- courbes d'apprentissage ----------
    files = sorted(glob.glob("logs/*.monitor.csv"))
    if files:
        fig, axes = plt.subplots(1, len(files), figsize=(3.2 * len(files), 3.2), squeeze=False)
        for ax, fpath in zip(axes[0], files):
            data = np.genfromtxt(fpath, delimiter=",", skip_header=2)
            if data.ndim == 1:
                data = data[None, :]
            r = data[:, 0]
            k = max(1, min(20, len(r) // 5))
            ax.plot(np.convolve(r, np.ones(k) / k, mode="valid"), color="#1f77b4")
            ax.set_title(os.path.basename(fpath).replace(".monitor.csv", ""), fontsize=9)
            ax.set_xlabel("Episode")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("Retour d'episode (moy. mobile)")
        fig.tight_layout()
        fig.savefig("results/fig_training.png", dpi=160)
        plt.close(fig)
    print("Resultats ecrits dans results/")


if __name__ == "__main__":
    main()
