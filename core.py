"""Mini-simulateur MEC a trois niveaux : device, edge, cloud/HPC.

Hypotheses de modelisation (a citer dans l'article) :
- une decision par tache, les taches sont traitees dans l'ordre d'arrivee ;
- devices et edges sont des files FIFO ;
- le cloud/HPC est un coordinateur global : ses coeurs sont partages entre les
  taches de tous les edges et servis en Earliest-Deadline-First pondere
  (non preemptif, priorite = arrivee + deadline relative / poids de l'edge) ;
- le cloud applique une contre-pression : si l'attente estimee de la tache
  depasse cloud_wait_max, il la refuse et elle revient s'executer sur l'edge
  apres un aller-retour edge-cloud ;
- l'edge peut transferer une tache a l'edge voisin (equilibrage de charge entre
  les groupes de devices) ;
- la latence de bout en bout = attente + transmission + calcul (le retour du
  resultat est neglige) ;
- l'edge diffuse sa charge a ses devices et a l'edge voisin, et le cloud
  diffuse sa charge aux edges, une fois par beacon_period (information
  periodique donc perimee) ;
- l'energie compte le calcul local, l'emission device->edge, le calcul edge et
  l'emission edge->cloud / edge->edge (l'energie du cloud n'est pas comptee).

Avec EDF, une tache cloud deja acceptee peut etre retardee par une tache plus
urgente arrivee ensuite. La recompense RL utilise la fin estimee au moment de
la decision ; les metriques finales (summarize) utilisent la fin reelle.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, replace

import numpy as np

DEVICE, EDGE, CLOUD, NEIGHBOR = 0, 1, 2, 3  # NEIGHBOR : edge voisin
TIER_NAMES = ["device", "edge", "cloud"]
EDGE_ACTIONS = (EDGE, CLOUD, NEIGHBOR)  # action de l'agent edge -> decision

DEV_OBS_DIM = 8
EDGE_OBS_DIM = 6
FLAT_OBS_DIM = 10
OBS_MAX = 1.5


@dataclass
class Config:
    n_dev: int = 6
    n_edge: int = 2
    n_tasks: int = 300
    base_rate: float = 0.5       # taches/s par device pour load = 1
    load: float = 1.0            # multiplicateur de charge
    f_dev_min: float = 0.7       # Gcycles/s
    f_dev_max: float = 1.5
    f_edge: float = 6.0          # Gcycles/s (un seul serveur FIFO par edge)
    f_cloud: float = 8.0         # Gcycles/s par coeur
    n_cloud_cores: int = 2
    bw_dev_edge: float = 30.0    # Mbit/s (multiplie par le fading de la tache)
    bw_edge_cloud: float = 50.0
    bw_edge_edge: float = 100.0
    lat_dev_edge: float = 0.005  # s
    lat_edge_cloud: float = 0.15
    lat_edge_edge: float = 0.01
    kappa_dev: float = 1.2       # J par Gcycle
    p_tx_dev: float = 0.8        # W
    kappa_edge: float = 0.25
    p_tx_edge: float = 3.0
    battery_cap: float = 150.0   # J
    cloud_wait_max: float = 1.0  # s, seuil de contre-pression
    edf_weights: tuple | None = None  # poids EDF par edge (None : EDF simple)
    beacon_period: float = 1.0   # s
    lam_e: float = 0.3           # poids de l'energie dans la recompense
    mu_miss: float = 2.0         # penalite de deadline manquee
    t_ref: float = 1.0
    r_min: float = -15.0
    trace_path: str | None = None


@dataclass
class Task:
    t: float          # instant d'arrivee (s)
    dev: int
    size: float       # Mbit
    cycles: float     # Gcycles
    deadline: float   # s, relative a t
    fade: float       # facteur de qualite du lien device->edge


# (probabilite, taille Mbit, cycles Gcycles, deadline s)
TASK_TYPES = [
    (0.4, (0.2, 1.0), (0.2, 0.6), (1.0, 2.0)),    # legere
    (0.4, (1.0, 3.0), (1.0, 3.0), (2.0, 4.0)),    # moyenne
    (0.2, (3.0, 8.0), (4.0, 10.0), (4.0, 8.0)),   # lourde
]


def gen_tasks(cfg: Config, rng: np.random.Generator) -> list[Task]:
    n = cfg.n_tasks
    rate = cfg.base_rate * cfg.load * cfg.n_dev
    gaps = rng.exponential(1.0 / rate, n)
    cyc_tr = None
    if cfg.trace_path:
        from trace_loader import load_trace
        gaps_tr, cycles_tr = load_trace(cfg.trace_path)
        # rejeu d'une fenetre contigue de la trace (les rafales sont conservees)
        s = int(rng.integers(0, len(gaps_tr) - n + 1))
        gaps = gaps_tr[s:s + n]
        gaps = gaps / gaps.mean() / rate
        cyc_tr = cycles_tr[s:s + n] / cycles_tr.mean() * 2.4
    times = np.cumsum(gaps)
    devs = rng.integers(0, cfg.n_dev, n)
    fade = rng.uniform(0.3, 1.2, n)
    kinds = rng.choice(len(TASK_TYPES), n, p=[t[0] for t in TASK_TYPES])
    tasks = []
    for i in range(n):
        _, s, c, d = TASK_TYPES[kinds[i]]
        size, cyc, dl = rng.uniform(*s), rng.uniform(*c), rng.uniform(*d)
        if cyc_tr is not None:
            cyc = float(np.clip(cyc_tr[i], 0.1, 20.0))
            size = float(np.clip(0.3 + 0.8 * cyc * rng.uniform(0.5, 1.5), 0.1, 10.0))
            dl = 0.5 + cyc * rng.uniform(0.6, 1.2)
        tasks.append(Task(float(times[i]), int(devs[i]), float(size), float(cyc),
                          float(dl), float(fade[i])))
    return tasks


def _clip(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, OBS_MAX)


class Sim:
    """Etat du systeme pour un episode (une suite de taches)."""

    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.tasks = gen_tasks(cfg, rng)
        self.f_dev = rng.uniform(cfg.f_dev_min, cfg.f_dev_max, cfg.n_dev)
        self.battery = rng.uniform(0.4, 1.0, cfg.n_dev)
        self.dev_free = np.zeros(cfg.n_dev)
        self.edge_free = np.zeros(cfg.n_edge)
        self.cloud_free = np.zeros(cfg.n_cloud_cores)  # coeurs, jobs engages
        self.cloud_pending: list[dict] = []            # jobs pas encore demarres
        self.w_edf = cfg.edf_weights or (1.0,) * cfg.n_edge
        self.beacon_idx = -1
        self.edge_beacon = np.zeros(cfg.n_edge)
        self.cloud_beacon = 0.0
        self.i = 0
        self.log: list[dict] = []

    # ---------- flux de taches ----------
    def current(self) -> Task | None:
        if self.i >= len(self.tasks):
            return None
        task = self.tasks[self.i]
        self._commit(task.t)
        self._tick(task.t)
        return task

    def _tick(self, t: float) -> None:
        idx = int(t / self.cfg.beacon_period)
        if idx > self.beacon_idx:
            self.beacon_idx = idx
            self.edge_beacon = np.maximum(0.0, self.edge_free - t)
            self.cloud_beacon = self.cloud_backlog(t)

    def neighbor(self, e: int) -> int:
        return (e + 1) % self.cfg.n_edge

    # ---------- cloud/HPC : EDF pondere non preemptif ----------
    def _cloud_schedule(self, extra: dict | None = None):
        """Deroule l'ordonnancement EDF des jobs en attente (plus extra).
        Renvoie debut, fin et coeur de chaque job, et la liberation des coeurs."""
        jobs = self.cloud_pending if extra is None else self.cloud_pending + [extra]
        n = len(jobs)
        free = self.cloud_free.copy()
        order = sorted(range(n), key=lambda j: jobs[j]["arr"])
        start, fin, core = [0.0] * n, [0.0] * n, [0] * n
        ready, k = [], 0
        for _ in range(n):
            c = int(np.argmin(free))
            now = free[c] if ready else max(free[c], jobs[order[k]]["arr"])
            while k < n and jobs[order[k]]["arr"] <= now:
                heapq.heappush(ready, (jobs[order[k]]["key"], order[k]))
                k += 1
            _, j = heapq.heappop(ready)
            start[j], fin[j], core[j] = now, now + jobs[j]["dur"], c
            free[c] = fin[j]
        return start, fin, core, free

    def _commit(self, t: float) -> None:
        """Les jobs demarres avant t ne peuvent plus etre doubles : on les fige."""
        if not self.cloud_pending:
            return
        start, fin, core, _ = self._cloud_schedule()
        keep = []
        for j, job in enumerate(self.cloud_pending):
            if start[j] <= t:
                job["fin"] = fin[j]
                self.cloud_free[core[j]] = max(self.cloud_free[core[j]], fin[j])
            else:
                keep.append(job)
        self.cloud_pending = keep

    def cloud_backlog(self, t: float) -> float:
        """Attente avant qu'un coeur du cloud se libere (contre-pression)."""
        free = self._cloud_schedule()[3] if self.cloud_pending else self.cloud_free
        return max(0.0, float(free.min()) - t)

    # ---------- planification (sans effet de bord) ----------
    def plan(self, task: Task, tier: int) -> dict:
        c = self.cfg
        d = task.dev
        e = d % c.n_edge
        if tier == DEVICE:
            start = max(task.t, self.dev_free[d])
            fin = start + task.cycles / self.f_dev[d]
            return dict(tier=DEVICE, finish=fin, e_dev=c.kappa_dev * task.cycles,
                        e_edge=0.0, upd=("dev", d, fin))
        t_tx = task.size / (c.bw_dev_edge * task.fade)
        arr_e = task.t + c.lat_dev_edge + t_tx
        e_dev = c.p_tx_dev * t_tx
        if tier == NEIGHBOR:
            nb = self.neighbor(e)
            t_ee = task.size / c.bw_edge_edge
            start = max(arr_e + c.lat_edge_edge + t_ee, self.edge_free[nb])
            fin = start + task.cycles / c.f_edge
            return dict(tier=EDGE, finish=fin, e_dev=e_dev,
                        e_edge=c.p_tx_edge * t_ee + c.kappa_edge * task.cycles,
                        upd=("edge", nb, fin))
        if tier == CLOUD:
            t_ec = task.size / c.bw_edge_cloud
            arr_c = arr_e + c.lat_edge_cloud + t_ec
            job = dict(arr=arr_c, dur=task.cycles / c.f_cloud,
                       key=task.t + task.deadline / self.w_edf[e], fin=None)
            start, fin, _, _ = self._cloud_schedule(job)
            if start[-1] - arr_c <= c.cloud_wait_max:
                return dict(tier=CLOUD, finish=fin[-1], e_dev=e_dev,
                            e_edge=c.p_tx_edge * t_ec, upd=("cloud", job, None))
            arr_e += 2 * c.lat_edge_cloud  # refus : aller-retour perdu
        start = max(arr_e, self.edge_free[e])
        fin = start + task.cycles / c.f_edge
        return dict(tier=EDGE, finish=fin, e_dev=e_dev,
                    e_edge=c.kappa_edge * task.cycles, upd=("edge", e, fin))

    # ---------- execution ----------
    def execute(self, task: Task, tier: int) -> dict:
        p = self.plan(task, tier)
        kind, idx, fin = p["upd"]
        job = None
        if kind == "dev":
            self.dev_free[idx] = fin
        elif kind == "edge":
            self.edge_free[idx] = fin
        else:
            job = idx
            self.cloud_pending.append(job)
        lat = p["finish"] - task.t
        batt = float(self.battery[task.dev])
        rec = dict(lat=lat, miss=float(lat > task.deadline), e_dev=p["e_dev"],
                   e_edge=p["e_edge"], tier=p["tier"], asked=tier,
                   w_batt=1.0 + 2.0 * (1.0 - batt),
                   t0=task.t, dl=task.deadline, job=job)
        self.battery[task.dev] = max(0.02, batt - p["e_dev"] / self.cfg.battery_cap)
        self.log.append(rec)
        return rec

    def finalize(self) -> None:
        """Fin d'episode : execute les jobs cloud restants et corrige la latence
        des taches cloud avec leur fin reelle sous EDF."""
        self._commit(np.inf)
        for r in self.log:
            if r["job"] is not None:
                r["lat"] = r["job"]["fin"] - r["t0"]
                r["miss"] = float(r["lat"] > r["dl"])

    def reward(self, rec: dict, level: str) -> float:
        c = self.cfg
        base = rec["lat"] / c.t_ref + c.mu_miss * rec["miss"]
        if level == "device":
            cost = base + c.lam_e * rec["w_batt"] * rec["e_dev"]
        elif level == "edge":
            cost = base + c.lam_e * rec["e_edge"]
        else:
            cost = base + c.lam_e * rec["w_batt"] * (rec["e_dev"] + rec["e_edge"])
        return max(-cost, c.r_min)

    # ---------- observations ----------
    def dev_obs(self, task: Task) -> np.ndarray:
        """Vue locale du device : sa tache, sa batterie, sa file, son lien,
        et la derniere charge diffusee par son edge."""
        c = self.cfg
        d = task.dev
        local_wait = max(0.0, self.dev_free[d] - task.t)
        bw = c.bw_dev_edge * task.fade
        return _clip([task.size / 8, task.cycles / 10, task.deadline / 8,
                      self.battery[d], local_wait / 5, bw / 36,
                      self.edge_beacon[d % c.n_edge] / 5, self.f_dev[d] / 1.5])

    def edge_obs(self, task: Task) -> np.ndarray:
        """Vue locale de l'edge : la tache recue, sa file, la derniere charge
        diffusee par l'edge voisin et la derniere contre-pression du cloud."""
        c = self.cfg
        e = task.dev % c.n_edge
        t_tx = c.lat_dev_edge + task.size / (c.bw_dev_edge * task.fade)
        arr_e = task.t + t_tx
        wait = max(0.0, self.edge_free[e] - arr_e)
        rem = max(0.0, task.deadline - t_tx)
        return _clip([task.size / 8, task.cycles / 10, rem / 8, wait / 5,
                      self.edge_beacon[self.neighbor(e)] / 5,
                      self.cloud_beacon / c.cloud_wait_max])

    def flat_obs(self, task: Task) -> np.ndarray:
        """Vue globale exacte (agent centralise)."""
        c = self.cfg
        d = task.dev
        e = d % c.n_edge
        t_tx = c.lat_dev_edge + task.size / (c.bw_dev_edge * task.fade)
        arr_e = task.t + t_tx
        arr_c = arr_e + c.lat_edge_cloud + task.size / c.bw_edge_cloud
        return _clip([task.size / 8, task.cycles / 10, task.deadline / 8,
                      self.battery[d],
                      max(0.0, self.dev_free[d] - task.t) / 5,
                      c.bw_dev_edge * task.fade / 36, self.f_dev[d] / 1.5,
                      max(0.0, self.edge_free[e] - arr_e) / 5,
                      max(0.0, self.edge_free[self.neighbor(e)] - arr_e) / 5,
                      self.cloud_backlog(arr_c) / c.cloud_wait_max])


def control_values(sim: Sim, signaling: str) -> float:
    """Nombre de valeurs scalaires de controle echangees (hors donnees)."""
    n = len(sim.log)
    c = sim.cfg
    if signaling == "none":
        return 0.0
    if signaling == "global":  # etat complet remonte + decision renvoyee
        return float(n * (FLAT_OBS_DIM + 1))
    dur = sim.tasks[-1].t
    # edge -> devices, edge -> edge voisin, cloud -> edges
    beacons = (dur / c.beacon_period) * (c.n_dev + 2 * c.n_edge)
    to_edge = sum(1 for r in sim.log if r["asked"] != DEVICE)
    to_up = sum(1 for r in sim.log if r["asked"] in (CLOUD, NEIGHBOR))
    return beacons + 4 * to_edge + 4 * to_up  # 4 = descripteur de tache


def summarize(sim: Sim, signaling: str) -> dict:
    sim.finalize()
    L = sim.log
    n = len(L)
    tiers = np.array([r["tier"] for r in L])
    asked = np.array([r["asked"] for r in L])
    return dict(
        latency=float(np.mean([r["lat"] for r in L])),
        hit_rate=float(1.0 - np.mean([r["miss"] for r in L])),
        energy=float(np.mean([r["e_dev"] + r["e_edge"] for r in L])),
        e_dev=float(np.mean([r["e_dev"] for r in L])),
        reject_rate=float(np.mean((asked == CLOUD) & (tiers != CLOUD))),
        frac_device=float(np.mean(tiers == DEVICE)),
        frac_edge=float(np.mean(tiers == EDGE)),
        frac_cloud=float(np.mean(tiers == CLOUD)),
        frac_fwd=float(np.mean(asked == NEIGHBOR)),
        overhead=control_values(sim, signaling) / n,
    )


def run_episode(cfg: Config, seed: int, policy) -> dict:
    sim = Sim(cfg, seed)
    policy.reset(seed)
    while True:
        task = sim.current()
        if task is None:
            break
        sim.execute(task, policy.act(sim, task))
        sim.i += 1
    return summarize(sim, policy.signaling)


def with_load(cfg: Config, load: float) -> Config:
    return replace(cfg, load=load)
