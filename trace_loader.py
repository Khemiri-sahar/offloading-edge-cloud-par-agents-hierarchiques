"""Chargeur de trace de charge (CSV generique) et extraction Alibaba.

Format attendu, avec en-tete : start_time,duration
  start_time : instant de debut de la tache (secondes, n'importe quelle origine)
  duration   : duree de la tache (proportionnelle a la charge de calcul)

Le depot fournit data/alibaba_batch_task.csv, extrait de la trace Alibaba
cluster-trace-v2018 (fichier batch_task, taches Terminated) :
    python trace_loader.py batch_task.csv data/alibaba_batch_task.csv
puis
    python train.py --trace data/alibaba_batch_task.csv
Les ecarts entre arrivees (rafales comprises) et les durees sont rejoues par
fenetres contigues puis remis a l'echelle de la simulation (voir core.gen_tasks).
"""
import csv
import sys
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def load_trace(path: str):
    starts, durs = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                s, d = float(row["start_time"]), float(row["duration"])
            except (KeyError, ValueError):
                continue
            if d > 0:
                starts.append(s)
                durs.append(d)
    if len(starts) < 1000:
        raise ValueError("trace trop courte (moins de 1000 lignes valides)")
    order = np.argsort(starts, kind="stable")
    starts = np.asarray(starts)[order]
    durs = np.asarray(durs)[order]
    gaps = np.diff(starts)
    gaps = np.maximum(gaps, 1e-3)
    return gaps, durs[1:]


def convert_alibaba(src: str, dst: str, n: int = 20_000) -> None:
    """batch_task.csv d'Alibaba (sans en-tete) -> CSV start_time,duration.
    Colonnes Alibaba : task_name, instance_num, job_name, task_type, status,
    start_time, end_time, plan_cpu, plan_mem. On garde les n premieres taches
    terminees (par date de debut)."""
    rows = []
    with open(src, newline="") as f:
        for r in csv.reader(f):
            try:
                s, e = int(r[5]), int(r[6])
            except (IndexError, ValueError):
                continue
            if r[4] == "Terminated" and e > s > 0:
                rows.append((s, e - s))
    rows.sort()
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_time", "duration"])
        w.writerows(rows[:n])


if __name__ == "__main__":
    convert_alibaba(sys.argv[1], sys.argv[2])
